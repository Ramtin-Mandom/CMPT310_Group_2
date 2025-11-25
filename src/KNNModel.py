from __future__ import annotations

import os
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ==== CONFIG ====


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "data.csv")

#CSV_PATH = "./data.csv"

# Numeric audio feature columns (must exist in data.csv)
FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

TITLE_COL = "name"
ARTIST_COL = "artists"

# first tests
# N_NEIGHBORS_DEFAULT = 11
# METRIC_DEFAULT = "cosine"

# after evalution
N_NEIGHBORS_DEFAULT = 5
METRIC_DEFAULT = "euclidean"


# ==== CORE MODEL BUILDING ====


def load_and_prepare(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load CSV and clean basic issues (NaN, duplicates)."""
    df = pd.read_csv(csv_path)

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(
            "None of the FEATURE_COLUMNS are present. "
            f"Found columns: {list(df.columns)[:20]} ..."
        )

    # Check title / artist columns exist
    for col in [TITLE_COL, ARTIST_COL]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV.")

    # Drop rows with missing features; drop duplicate (title, artist) pairs
    df = df.dropna(subset=available_features).copy()
    df = df.drop_duplicates(subset=[TITLE_COL, ARTIST_COL]).reset_index(drop=True)

    return df


def fit_scaler_and_knn(
    df: pd.DataFrame,
    n_neighbors: int = N_NEIGHBORS_DEFAULT,
    metric: str = METRIC_DEFAULT,
) -> tuple[StandardScaler, NearestNeighbors, np.ndarray]:
    """Fit StandardScaler and KNN on all rows of df; return scaled matrix too."""
    X = df[[c for c in FEATURE_COLUMNS if c in df.columns]].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_scaled)

    return scaler, knn, X_scaled


def build_model(
    csv_path: str = CSV_PATH,
    n_neighbors: int = N_NEIGHBORS_DEFAULT,
    metric: str = METRIC_DEFAULT,
) -> tuple[pd.DataFrame, StandardScaler, NearestNeighbors, np.ndarray]:
    """Convenience: load data and fit scaler + KNN in one call."""
    df = load_and_prepare(csv_path)
    scaler, knn, X_scaled = fit_scaler_and_knn(df, n_neighbors=n_neighbors, metric=metric)
    return df, scaler, knn, X_scaled


# ==== LOOKUP UTILITIES ====


def find_track_index(
    df: pd.DataFrame,
    title: str,
    artist: str | None = None,
) -> int | None:
    """Return the index of the best match (case-insensitive), or None if not found."""
    # First, fuzzy filter by title (substring match)
    mask = df[TITLE_COL].str.casefold().str.contains(title.casefold(), na=False)
    if artist:
        mask &= df[ARTIST_COL].str.casefold().str.contains(artist.casefold(), na=False)

    candidates = df[mask]

    if len(candidates) == 0:
        # Try fuzzy matching on full column
        titles = df[TITLE_COL].astype(str).tolist()
        close = get_close_matches(title, titles, n=1, cutoff=0.8)
        if not close:
            return None
        match_title = close[0]
        idx = df.index[df[TITLE_COL] == match_title]
        return int(idx[0]) if len(idx) > 0 else None

    if len(candidates) == 1:
        return int(candidates.index[0])

    # More than one candidate -> try exact case-insensitive match on both
    exact = df[
        (df[TITLE_COL].str.casefold() == title.casefold())
        & (df[ARTIST_COL].str.casefold() == (artist or "").casefold())
    ]
    if len(exact) == 1:
        return int(exact.index[0])

    # Fallback: just take the first candidate
    return int(candidates.index[0])


# ==== RECOMMENDATION FUNCTIONS ====


def recommend_similar_songs(
    df: pd.DataFrame,
    scaler: StandardScaler,
    knn: NearestNeighbors,
    X_scaled: np.ndarray,
    query_title: str,
    query_artist: str | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Song-to-song KNN recommendations."""
    idx = find_track_index(df, query_title, query_artist)
    if idx is None:
        raise ValueError(f"Track not found: '{query_title}' (artist: {query_artist or 'any'})")

    distances, indices = knn.kneighbors([X_scaled[idx]])
    neighbor_indices = indices[0][1 : top_k + 1]  # skip the query itself
    neighbor_distances = distances[0][1 : top_k + 1]

    out = df.loc[neighbor_indices, [TITLE_COL, ARTIST_COL]].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out["similarity"] = (1 - neighbor_distances).round(4)

    return out.reset_index(drop=True)


def user_vector_from_likes(
    df: pd.DataFrame,
    scaler: StandardScaler,
    liked_tracks: list[tuple[str, str | None]],
) -> np.ndarray:
    """Average the (scaled) feature vectors of the user's liked tracks."""
    feats = []
    for title, artist in liked_tracks:
        idx = find_track_index(df, title, artist)
        if idx is not None:
            feats.append(df.loc[idx, FEATURE_COLUMNS].values)

    if not feats:
        raise ValueError("None of the liked tracks were found in the dataset.")

    X_like = np.vstack(feats)
    X_like_df = pd.DataFrame(X_like, columns=FEATURE_COLUMNS)
    X_like_scaled = scaler.transform(X_like_df)
    return X_like_scaled.mean(axis=0)


def recommend_for_user(
    df: pd.DataFrame,
    scaler: StandardScaler,
    knn: NearestNeighbors,
    X_scaled: np.ndarray,
    liked_tracks: list[tuple[str, str | None]],
    top_k: int = 10,
    max_per_artist: int = 2,
    min_year: int | None = None,
    max_year: int | None = None,
    min_popularity: int | None = None,
    explicit_only: bool | None = None,
) -> pd.DataFrame:
    """
    Recommend songs close to the user's taste vector.
    - Excludes songs already liked (title+artist).
    - Optionally limits #songs per artist and applies simple filters.
    """
    user_vec = user_vector_from_likes(df, scaler, liked_tracks)
    distances, indices = knn.kneighbors([user_vec])

    liked_set = {(t.casefold(), (a or "").casefold()) for t, a in liked_tracks}
    artist_counts: dict[str, int] = {}
    recs: list[tuple[int, float]] = []

    for dist, local_idx in zip(distances[0], indices[0]):
        idx = int(local_idx)
        title = str(df.loc[idx, TITLE_COL])
        artist = str(df.loc[idx, ARTIST_COL])
        key = (title.casefold(), artist.casefold())

        # Exclude already liked
        if key in liked_set:
            continue

        # Simple filters (if columns exist)
        row = df.loc[idx]
        if min_year is not None and "year" in df.columns and row["year"] < min_year:
            continue
        if max_year is not None and "year" in df.columns and row["year"] > max_year:
            continue
        if min_popularity is not None and "popularity" in df.columns and row["popularity"] < min_popularity:
            continue
        if explicit_only is not None and "explicit" in df.columns:
            # explicit is typically 0/1
            if explicit_only and row["explicit"] == 0:
                continue
            if not explicit_only and row["explicit"] == 1:
                continue

        # Diversity: limit #songs per artist
        artist_counts.setdefault(artist, 0)
        if artist_counts[artist] >= max_per_artist:
            continue

        recs.append((idx, dist))
        artist_counts[artist] += 1

        if len(recs) >= top_k:
            break

    if not recs:
        raise ValueError("No recommendations found that satisfy the given filters.")

    out_idx = [i for i, _ in recs]
    out_dist = [d for _, d in recs]

    out = df.loc[out_idx, [TITLE_COL, ARTIST_COL]].copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out["similarity"] = (1 - np.array(out_dist)).round(4)

    return out.reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual test
    df, scaler, knn, X_scaled = build_model()

    QUERY_TITLE = "Back In Black"
    QUERY_ARTIST = "AC/DC"

    try:
        print(f"\nSimilar songs to: {QUERY_TITLE} — {QUERY_ARTIST}\n")
        recs_song = recommend_similar_songs(
            df, scaler, knn, X_scaled, QUERY_TITLE, QUERY_ARTIST, top_k=10
        )
        print(recs_song.to_string(index=False))
    except ValueError as e:
        print("Song-to-song:", e)

    LIKED = [
        ("Back In Black", "AC/DC"),
        ("Thunderstruck", "AC/DC"),
        ("Highway to Hell", "AC/DC"),
    ]

    try:
        print(f"\nRecommendations for user (liked: {', '.join([t for t, _ in LIKED])})\n")
        recs_user = recommend_for_user(
            df,
            scaler,
            knn,
            X_scaled,
            LIKED,
            top_k=10,
            max_per_artist=2,
            min_popularity=40,
        )
        print(recs_user.to_string(index=False))
    except ValueError as e:
        print("User-based:", e)

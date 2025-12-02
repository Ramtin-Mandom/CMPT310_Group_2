from __future__ import annotations

import os
import numpy as np
import pandas as pd
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ==== CONFIG ====

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "Data", "data.csv")

# Numeric audio feature columns
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

# Previously: N_NEIGHBORS_DEFAULT, METRIC_DEFAULT
N_NEIGHBORS = 5
METRIC = "euclidean"

# Feature weights: make some features count more in distance
# 1.0 = default importance. Increase to emphasize a feature.
FEATURE_WEIGHTS: dict[str, float] = {
    "tempo": 2.5,          # beat / BPM matters more
    "energy": 1.5,         # louder/more intense vs calm
    "danceability": 1.3,   # how easy it is to dance to
    # other features fall back to 1.0
}


def apply_feature_weights(X_scaled: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """
    Multiply selected feature dimensions by their weights *after* scaling.
    This effectively makes these features more important in Euclidean distance.
    """
    Xw = X_scaled.copy()
    for i, col in enumerate(feature_columns):
        w = FEATURE_WEIGHTS.get(col, 1.0)
        if w != 1.0:
            Xw[:, i] *= w
    return Xw


# ==== CORE MODEL BUILDING ====


def load_and_prepare(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected numeric feature columns {FEATURE_COLUMNS}, "
            f"but the following are missing: {missing_cols}. "
            f"Check that your data.csv has these columns and that FEATURE_COLUMNS are present. "
            f"Found columns: {list(df.columns)[:20]} ..."
        )

    for col in [TITLE_COL, ARTIST_COL]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV.")

    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    df = df.drop_duplicates(subset=[TITLE_COL, ARTIST_COL]).reset_index(drop=True)

    return df


def fit_scaler_and_knn(
    df: pd.DataFrame,
    n_neighbors: int = N_NEIGHBORS,
    metric: str = METRIC,
) -> tuple[StandardScaler, NearestNeighbors, np.ndarray]:

    # Use only the feature columns that actually exist in this CSV
    feature_columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_columns].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply feature weights so tempo / energy / danceability matter more
    X_scaled = apply_feature_weights(X_scaled, feature_columns)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_scaled)

    return scaler, knn, X_scaled


def build_model(
    csv_path: str = CSV_PATH,
    n_neighbors: int = N_NEIGHBORS,
    metric: str = METRIC,
) -> tuple[pd.DataFrame, StandardScaler, NearestNeighbors, np.ndarray]:

    df = load_and_prepare(csv_path)
    scaler, knn, X_scaled = fit_scaler_and_knn(df, n_neighbors=n_neighbors, metric=metric)
    return df, scaler, knn, X_scaled


# ==== LOOKUP UTILITIES ====


def find_track_index(
    df: pd.DataFrame,
    title: str,
    artist: str | None = None,
) -> int | None:

    mask = df[TITLE_COL].str.casefold().str.contains(title.casefold(), na=False)
    if artist:
        mask &= df[ARTIST_COL].str.casefold().str.contains(artist.casefold(), na=False)

    candidates = df[mask]

    if len(candidates) == 0:
        titles = df[TITLE_COL].astype(str).tolist()
        close = get_close_matches(title, titles, n=3, cutoff=0.6)
        msg = f"No track found matching title '{title}'"
        if artist:
            msg += f" and artist '{artist}'"
        if close:
            msg += f". Did you mean: {', '.join(close)}?"
        print(msg)
        return None

    if len(candidates) > 1:
        print(f"Multiple matches found for '{title}':")
        print(candidates[[TITLE_COL, ARTIST_COL]].head())
        print("Taking the first match. You may want to specify an artist.")

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

    idx = find_track_index(df, query_title, query_artist)
    if idx is None:
        raise ValueError(f"Track not found: '{query_title}' (artist: {query_artist or 'any'})")

    distances, indices = knn.kneighbors([X_scaled[idx]])
    neighbor_indices = indices[0][1 : top_k + 1]
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

    # Apply the same feature weights used to train the KNN model
    X_like_scaled = apply_feature_weights(X_like_scaled, FEATURE_COLUMNS)

    # Final user vector is the (weighted, scaled) average of liked tracks
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

    user_vec = user_vector_from_likes(df, scaler, liked_tracks)
    distances, indices = knn.kneighbors([user_vec])

    liked_set = {
        (title.casefold(), (artist or "").casefold()) for title, artist in liked_tracks
    }

    recs: list[tuple[int, float]] = []
    artist_counts: dict[str, int] = {}

    for dist, local_idx in zip(distances[0], indices[0]):
        idx = int(local_idx)
        title = str(df.loc[idx, TITLE_COL])
        artist = str(df.loc[idx, ARTIST_COL])
        key = (title.casefold(), artist.casefold())

        if key in liked_set:
            continue

        row = df.loc[idx]
        if min_year is not None and "year" in df.columns and row["year"] < min_year:
            continue
        if max_year is not None and "year" in df.columns and row["year"] > max_year:
            continue
        if min_popularity is not None and "popularity" in df.columns and row["popularity"] < min_popularity:
            continue
        if explicit_only is not None and "explicit" in df.columns:
            if explicit_only and row["explicit"] == 0:
                continue
            if not explicit_only and row["explicit"] == 1:
                continue

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



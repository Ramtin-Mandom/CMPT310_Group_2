from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from KNNModel import (
    CSV_PATH,
    FEATURE_COLUMNS,
    load_and_prepare,
)

# --------------------------------------------------------------------
# LABEL CREATION HELPERS
#   We use three evaluation labels:
#   - tempo_bucket (from 'tempo')
#   - energy_bucket (from 'energy')
#   - loudness_bucket (from 'loudness')
# --------------------------------------------------------------------

def make_tempo_bucket_labels(df: pd.DataFrame) -> pd.Series | None:
    """
    Bucket tempo into slow / medium / fast.
    Adjust thresholds if needed for your dataset.
    """
    if "tempo" not in df.columns:
        return None
    tempo = df["tempo"].astype(float)
    # Rough buckets: <90 slow, 90–130 medium, >130 fast
    bins = [-10, 90, 130, 400]
    labels = ["slow", "medium", "fast"]
    return pd.cut(tempo, bins=bins, labels=labels, include_lowest=True)


def make_energy_bucket_labels(df: pd.DataFrame) -> pd.Series | None:
    if "energy" not in df.columns:
        return None
    energy = df["energy"].astype(float)
    bins = [-10, 0.33, 0.66, 10]
    labels = ["low", "medium", "high"]
    return pd.cut(energy, bins=bins, labels=labels, include_lowest=True)


def make_loudness_bucket_labels(df: pd.DataFrame) -> pd.Series | None:
    """
    Bucket loudness (typically negative dB) into a few coarse levels.
    """
    if "loudness" not in df.columns:
        return None
    loud = df["loudness"].astype(float)
    # Very rough buckets, you can tweak:
    bins = [-80, -20, -10, 5]
    labels = ["very_quiet", "quiet", "normal"]
    return pd.cut(loud, bins=bins, labels=labels, include_lowest=True)


def build_all_label_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Collect all valid label series present in data.
    We use: tempo_bucket, energy_bucket, loudness_bucket.
    """
    labels: dict[str, pd.Series] = {}

    tempo_bucket = make_tempo_bucket_labels(df)
    if tempo_bucket is not None and tempo_bucket.nunique() > 1:
        labels["tempo_bucket"] = tempo_bucket

    energy_bucket = make_energy_bucket_labels(df)
    if energy_bucket is not None and energy_bucket.nunique() > 1:
        labels["energy_bucket"] = energy_bucket

    loudness_bucket = make_loudness_bucket_labels(df)
    if loudness_bucket is not None and loudness_bucket.nunique() > 1:
        labels["loudness_bucket"] = loudness_bucket

    return labels


# --------------------------------------------------------------------
# NEIGHBOUR AGREEMENT COMPUTATION
# --------------------------------------------------------------------

def neighbor_label_agreement(
    df: pd.DataFrame,
    labels: pd.Series,
    scaler: StandardScaler,
    knn: NearestNeighbors,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    n_eval: int | None = 200,
) -> float:
    """
    Compute agreement@k for a given label type.
    Uses iloc since indices from KFold/train_test_split are positional.
    """
    if n_eval is not None and n_eval < len(eval_indices):
        rng = np.random.default_rng(0)
        eval_indices = rng.choice(eval_indices, size=n_eval, replace=False)

    X_train = df.iloc[train_indices][FEATURE_COLUMNS].copy()
    X_eval = df.iloc[eval_indices][FEATURE_COLUMNS].copy()

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    knn.fit(X_train_scaled)
    distances, neighbor_idx = knn.kneighbors(X_eval_scaled)

    train_labels = labels.iloc[train_indices].to_numpy()
    eval_labels = labels.iloc[eval_indices].to_numpy()

    agreements: list[float] = []
    for i, neigh_pos_list in enumerate(neighbor_idx):
        true_label = eval_labels[i]
        neigh_labels = train_labels[neigh_pos_list]
        agreements.append((neigh_labels == true_label).mean())

    return float(np.mean(agreements)) if agreements else 0.0


# --------------------------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------------------------

def cross_validate_knn(
    df: pd.DataFrame,
    k_list=None,
    metrics=None,
    n_splits=5,
    random_state=42,
) -> pd.DataFrame:
    """
    k-fold CV on given df, using multiple label types if available:
      - tempo_bucket
      - energy_bucket
      - loudness_bucket

    For each (k, metric) we compute mean/std agreement for each label type
    AND a combined_mean score = average of available label means.
    """
    if k_list is None:
        k_list = [5, 10, 15, 20]
    if metrics is None:
        metrics = ["cosine", "euclidean"]

    label_sets = build_all_label_sets(df)
    if not label_sets:
        raise ValueError("No suitable label columns (tempo/energy/loudness) found.")
    print("Evaluation label sets:", list(label_sets.keys()))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for metric in metrics:
        for k in k_list:
            scores_per_label = {name: [] for name in label_sets.keys()}

            for train_idx, val_idx in kf.split(df):
                for label_name, labels in label_sets.items():
                    scaler = StandardScaler()
                    knn = NearestNeighbors(n_neighbors=k, metric=metric)

                    score = neighbor_label_agreement(
                        df=df,
                        labels=labels,
                        scaler=scaler,
                        knn=knn,
                        train_indices=train_idx,
                        eval_indices=val_idx,
                        n_eval=200,
                    )
                    scores_per_label[label_name].append(score)

            row: dict[str, float | int | str] = {"k": k, "metric": metric}
            combined_means: list[float] = []

            for label_name, fold_scores in scores_per_label.items():
                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))
                row[f"{label_name}_mean"] = mean_score
                row[f"{label_name}_std"] = std_score
                combined_means.append(mean_score)

            row["combined_mean"] = float(np.mean(combined_means))
            rows.append(row)

    return pd.DataFrame(rows).sort_values("combined_mean", ascending=False)


# --------------------------------------------------------------------
# TRAIN/TEST + FINAL EVALUATION
# --------------------------------------------------------------------

def train_test_and_final_model(
    csv_path: str = CSV_PATH,
    test_size: float = 0.1,
    random_state: int = 42,
    k_list=None,
    metrics=None,
):
    """
    1) Split data.csv into train(90%) / test(10%).
    2) Run CV on train to select best (k, metric) based on combined label score.
    3) Train final model on full train with best hyperparameters.
    4) Evaluate once on 10% test set for all label types and combined score.
    """
    full_df = load_and_prepare(csv_path)

    train_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=random_state, shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Cross-validation on training data only
    cv_results = cross_validate_knn(train_df, k_list=k_list, metrics=metrics)

    # Pick best row by combined_mean
    best_row = cv_results.iloc[0]
    best_k = int(best_row["k"])
    best_metric = str(best_row["metric"])
    print("Best hyperparameters from CV:", best_row.to_dict())

    # Build label sets for train and test
    train_labels_all = build_all_label_sets(train_df)
    test_labels_all = build_all_label_sets(test_df)

    # Train final scaler + knn on full train set
    scaler = StandardScaler()
    knn = NearestNeighbors(n_neighbors=best_k, metric=best_metric)

    X_train = train_df[FEATURE_COLUMNS].copy()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    knn.fit(X_train_scaled)

    # Evaluate on held-out 10% test set
    X_test = test_df[FEATURE_COLUMNS].copy()
    X_test_scaled = scaler.transform(X_test)
    distances, neighbor_idx = knn.kneighbors(X_test_scaled)

    test_scores_per_label: dict[str, float] = {}
    for label_name, train_labels_series in train_labels_all.items():
        if label_name not in test_labels_all:
            continue

        train_labels_arr = train_labels_series.to_numpy()
        test_labels_arr = test_labels_all[label_name].to_numpy()

        agreements: list[float] = []
        for test_pos, neigh_pos_list in enumerate(neighbor_idx):
            true_label = test_labels_arr[test_pos]
            neigh_labels = train_labels_arr[neigh_pos_list]
            agreements.append((neigh_labels == true_label).mean())

        test_scores_per_label[label_name] = float(np.mean(agreements)) if agreements else 0.0

    combined_test_score = float(np.mean(list(test_scores_per_label.values()))) if test_scores_per_label else 0.0

    return {
        "cv_results": cv_results,
        "best_k": best_k,
        "best_metric": best_metric,
        "test_scores_per_label": test_scores_per_label,
        "test_combined_score": combined_test_score,
    }


if __name__ == "__main__":
    results = train_test_and_final_model()

    print("\n=== Cross-validation results (top 10 rows) ===")
    print(results["cv_results"].head(10).to_string(index=False))

    print("\n=== Final test scores per label ===")
    for name, score in results["test_scores_per_label"].items():
        print(f"{name}: {score:.3f}")

    print(f"\nFinal combined test agreement: {results['test_combined_score']:.3f}")

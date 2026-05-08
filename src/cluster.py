from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from config import (
    DBSCAN_EPS_KM,
    DBSCAN_MIN_SAMPLES,
    EARTH_RADIUS_KM,
    FEATURES_PARQUET,
    KMEANS_N_CLUSTERS,
    METRICS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_FRACTION,
)


def _train_test_split_chrono(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * (1 - test_fraction))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def fit_kmeans(train_coords: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    model.fit(train_coords)
    return model


def kmeans_diagnostics(train_coords: np.ndarray, k_range: range) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(train_coords), size=min(20000, len(train_coords)), replace=False)
    sample = train_coords[sample_idx]
    rows = []
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = m.fit_predict(sample)
        sse = m.inertia_
        sil = silhouette_score(sample, labels) if k > 1 else np.nan
        rows.append({"k": k, "sse": sse, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_dbscan_sample(train_coords: np.ndarray, eps_km: float, min_samples: int, sample_size: int = 25000) -> tuple[DBSCAN, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(train_coords), size=min(sample_size, len(train_coords)), replace=False)
    radians_coords = np.radians(train_coords[sample_idx])
    eps_rad = eps_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(radians_coords)
    return model, sample_idx, labels


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PARQUET).sort_values("crash_datetime").reset_index(drop=True)

    train_df, test_df = _train_test_split_chrono(df, TEST_FRACTION)

    train_coords = train_df[["latitude", "longitude"]].to_numpy()
    test_coords = test_df[["latitude", "longitude"]].to_numpy()
    all_coords = df[["latitude", "longitude"]].to_numpy()

    print(f"[cluster] kmeans diagnostics for k in 2..12 on training fold...")
    diag = kmeans_diagnostics(train_coords, range(2, 13))
    diag.to_csv(METRICS_DIR / "kmeans_diagnostics.csv", index=False)
    print(diag.to_string(index=False))

    print(f"[cluster] fitting kmeans (k={KMEANS_N_CLUSTERS}) on training fold...")
    kmeans = fit_kmeans(train_coords, KMEANS_N_CLUSTERS)

    df["kmeans_cluster"] = kmeans.predict(all_coords)

    print(f"[cluster] dbscan on training-fold sample (eps={DBSCAN_EPS_KM} km)...")
    _, dbscan_idx, dbscan_labels = fit_dbscan_sample(
        train_coords, DBSCAN_EPS_KM, DBSCAN_MIN_SAMPLES
    )
    n_dbscan_clusters = len({lbl for lbl in dbscan_labels if lbl != -1})
    n_noise = int((dbscan_labels == -1).sum())
    print(f"[cluster]   dbscan found {n_dbscan_clusters} clusters, {n_noise} noise points")

    pd.DataFrame(
        {
            "train_idx": dbscan_idx,
            "label": dbscan_labels,
        }
    ).to_csv(METRICS_DIR / "dbscan_sample_labels.csv", index=False)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["latitude", "longitude"])
    centroids["cluster"] = centroids.index
    centroids = centroids[["cluster", "latitude", "longitude"]]
    counts = (
        df.groupby("kmeans_cluster")
        .agg(n_crashes=("severity", "size"), n_serious=("severity", "sum"))
        .reset_index()
    )
    counts["serious_rate"] = counts["n_serious"] / counts["n_crashes"]
    counts.to_csv(METRICS_DIR / "kmeans_cluster_counts.csv", index=False)
    centroids.to_csv(METRICS_DIR / "kmeans_centroids.csv", index=False)

    out = PROCESSED_DIR / "features_with_clusters.parquet"
    df.to_parquet(out, index=False)
    print(f"[cluster] wrote {out}")

    import joblib

    joblib.dump(kmeans, PROCESSED_DIR / "kmeans.joblib")
    print(f"[cluster] saved kmeans model")


if __name__ == "__main__":
    main()

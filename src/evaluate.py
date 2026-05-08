from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_FRACTION,
)
from src.train import FEATURE_SETS, chrono_split

sns.set_theme(context="paper", style="whitegrid")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] wrote {out}")
    return out


def fig_kmeans_diagnostics() -> None:
    diag = pd.read_csv(METRICS_DIR / "kmeans_diagnostics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(diag["k"], diag["sse"], marker="o")
    axes[0].set_xlabel("number of clusters k")
    axes[0].set_ylabel("SSE")
    axes[0].set_title("Elbow Method")
    axes[1].plot(diag["k"], diag["silhouette"], marker="o", color="C1")
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("silhouette score")
    axes[1].set_title("Silhouette Score")
    fig.tight_layout()
    _save(fig, "kmeans_diagnostics.png")


def fig_cluster_map(df: pd.DataFrame) -> None:
    centroids = pd.read_csv(METRICS_DIR / "kmeans_centroids.csv")
    sample = df.sample(min(20000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=sample,
        x="longitude",
        y="latitude",
        hue="kmeans_cluster",
        s=4,
        linewidth=0,
        alpha=0.5,
        palette="tab10",
        legend=False,
        ax=ax,
    )
    ax.scatter(
        centroids["longitude"],
        centroids["latitude"],
        marker="X",
        s=120,
        c="black",
        edgecolors="white",
        linewidths=1.0,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("KMeans crash hotspot clusters (NYC)")
    fig.tight_layout()
    _save(fig, "kmeans_clusters_map.png")


def fig_severity_by_cluster() -> None:
    counts = pd.read_csv(METRICS_DIR / "kmeans_cluster_counts.csv")
    counts = counts.sort_values("serious_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(counts["kmeans_cluster"].astype(str), counts["serious_rate"])
    ax.set_xlabel("KMeans cluster id")
    ax.set_ylabel("serious crash rate")
    ax.set_title("Serious-crash rate by hotspot cluster")
    fig.tight_layout()
    _save(fig, "severity_by_cluster.png")


def fig_hourly_accidents(df: pd.DataFrame) -> None:
    by_hour = df.groupby("hour").size()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(by_hour.index, by_hour.values)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("number of crashes")
    ax.set_title("Crashes by hour of day")
    fig.tight_layout()
    _save(fig, "hourly_accidents.png")


def fig_borough_severity(df: pd.DataFrame) -> None:
    g = df.groupby("borough")["severity"].agg(total="count", serious="sum").reset_index()
    g = g[g["borough"].ne("UNKNOWN")]
    g["serious_rate"] = g["serious"] / g["total"]
    g = g.sort_values("serious_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(g["borough"], g["serious_rate"])
    ax.set_xlabel("borough")
    ax.set_ylabel("serious crash rate")
    ax.set_title("Serious-crash rate by borough")
    fig.tight_layout()
    _save(fig, "borough_severity.png")


def fig_pr_roc_curves(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, prob in preds.items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        axes[0].plot(recall, precision, label=name)
        fpr, tpr, _ = roc_curve(y_true, prob)
        axes[1].plot(fpr, tpr, label=name)
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall curves")
    axes[0].legend(fontsize="x-small")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC curves")
    axes[1].legend(fontsize="x-small")
    fig.tight_layout()
    _save(fig, "pr_roc_curves.png")


def fig_confusion_matrices(y_true: np.ndarray, preds: dict[str, np.ndarray], thresholds: dict[str, float]) -> None:
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (name, prob) in zip(axes, preds.items()):
        thr = thresholds.get(name, 0.5)
        pred = (prob >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cbar=False,
            cmap="Blues",
            xticklabels=["pred 0", "pred 1"],
            yticklabels=["true 0", "true 1"],
            ax=ax,
        )
        ax.set_title(f"{name}\nthr={thr:.2f}")
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def fig_calibration(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, prob in preds.items():
        CalibrationDisplay.from_predictions(y_true, prob, n_bins=15, name=name, ax=ax)
    ax.set_title("Calibration")
    fig.tight_layout()
    _save(fig, "calibration.png")


def fig_pca_projection(df: pd.DataFrame, num_cols: list[str]) -> None:
    sample = df.sample(min(15000, len(df)), random_state=0)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(sample[num_cols].fillna(0).to_numpy())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=sample["severity"].astype(str).values,
        s=6,
        linewidth=0,
        alpha=0.5,
        palette={"0": "#9ecae1", "1": "#e6550d"},
        ax=ax,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA projection of engineered features")
    ax.legend(title="severity", fontsize="x-small")
    fig.tight_layout()
    _save(fig, "pca_projection.png")


def fig_shap_summary(model_path: Path, X_sample: pd.DataFrame, max_display: int = 15) -> None:
    import shap

    bundle = joblib.load(model_path)
    pre = bundle.named_steps["pre"]
    clf = bundle.named_steps["clf"]
    X_t = pre.transform(X_sample)
    feature_names: list[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer if hasattr(transformer, "get_feature_names_out") else transformer.named_steps.get("onehot", transformer)
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_t)
    fig = plt.figure(figsize=(7, 4.5))
    shap.summary_plot(
        shap_values,
        features=X_t,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="bar",
    )
    _save(plt.gcf(), "shap_summary.png")


def write_ablation_table() -> None:
    metrics = pd.read_csv(METRICS_DIR / "model_metrics.csv")
    ablation = (
        metrics[metrics["model"] == "XGBoost"][
            ["feature_set", "accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "f1_serious", "recall_serious"]
        ]
        .reset_index(drop=True)
    )
    ablation.to_csv(METRICS_DIR / "ablation_xgboost.csv", index=False)
    print(f"[evaluate] ablation table:\n{ablation.to_string(index=False)}")


def write_summary_tables() -> None:
    metrics = pd.read_csv(METRICS_DIR / "model_metrics.csv")
    primary = metrics[metrics["feature_set"].isin(["full_with_cluster", "n/a"])].copy()
    primary = primary[
        [
            "model",
            "feature_set",
            "accuracy",
            "balanced_accuracy",
            "precision_serious",
            "recall_serious",
            "f1_serious",
            "pr_auc",
            "roc_auc",
            "threshold",
        ]
    ]
    primary.to_csv(METRICS_DIR / "summary_models.csv", index=False)
    print(f"[evaluate] summary table:\n{primary.to_string(index=False)}")


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "features_with_clusters.parquet")
    df = df.sort_values("crash_datetime").reset_index(drop=True)

    fig_kmeans_diagnostics()
    fig_cluster_map(df)
    fig_severity_by_cluster()
    fig_hourly_accidents(df)
    fig_borough_severity(df)

    train_df, test_df = chrono_split(df, TEST_FRACTION)

    npz = np.load(METRICS_DIR / "test_predictions.npz")
    y_true = npz["y_true"]
    preds = {k: npz[k] for k in npz.files if k != "y_true"}
    pretty = {
        "XGBoost": preds.get("XGBoost"),
        "XGBoost+SMOTE": preds.get("XGBoost_SMOTE"),
        "XGBoost+ThresholdTuned": preds.get("XGBoost_ThresholdTuned"),
    }
    pretty = {k: v for k, v in pretty.items() if v is not None}

    fig_pr_roc_curves(y_true, pretty)
    metrics = pd.read_csv(METRICS_DIR / "model_metrics.csv")
    thresholds = {
        row["model"]: row["threshold"]
        for _, row in metrics[metrics["feature_set"] == "full_with_cluster"].iterrows()
    }
    fig_confusion_matrices(y_true, pretty, thresholds)
    fig_calibration(y_true, pretty)

    full = FEATURE_SETS["full_with_cluster"]
    fig_pca_projection(test_df, full["num"])

    xgb_path = MODELS_DIR / "xgb_full_with_cluster.joblib"
    if xgb_path.exists():
        feature_cols = full["num"] + full["cat"]
        sample = test_df[feature_cols].sample(min(5000, len(test_df)), random_state=0)
        fig_shap_summary(xgb_path, sample)

    write_summary_tables()
    write_ablation_table()


if __name__ == "__main__":
    main()

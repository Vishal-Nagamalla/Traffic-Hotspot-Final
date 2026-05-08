from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_FRACTION,
)


NUM_BASE = [
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
NUM_WEATHER = [
    "precipitation",
    "rain_mm",
    "temp_max",
    "temp_min",
    "temp_mean",
    "temp_range",
    "cloud_pct",
    "wind_kmh_max",
    "is_rainy",
    "is_heavy_rain",
    "is_freezing",
    "is_hot",
    "is_cloudy",
    "is_windy",
]
NUM_SPATIAL = ["dist_to_midtown_km"]
NUM_CLUSTER: list[str] = []
CAT_BASE = ["borough"]
CAT_CLUSTER = ["kmeans_cluster"]


FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "time_only": {"num": NUM_BASE, "cat": CAT_BASE},
    "time_weather": {"num": NUM_BASE + NUM_WEATHER, "cat": CAT_BASE},
    "time_weather_spatial": {
        "num": NUM_BASE + NUM_WEATHER + NUM_SPATIAL,
        "cat": CAT_BASE,
    },
    "full_with_cluster": {
        "num": NUM_BASE + NUM_WEATHER + NUM_SPATIAL,
        "cat": CAT_BASE + CAT_CLUSTER,
    },
}


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def chrono_split(df: pd.DataFrame, test_fraction: float):
    n = len(df)
    split = int(n * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)
    return {
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_serious": float(report["1"]["precision"]),
        "recall_serious": float(report["1"]["recall"]),
        "f1_serious": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    pred = np.zeros_like(y_test, dtype=int)
    prob = np.zeros_like(y_test, dtype=float)
    metrics = evaluate(y_test.to_numpy(), prob, pred)
    metrics["model"] = "MajorityBaseline"
    metrics["feature_set"] = "n/a"
    return metrics


def fit_logistic(X_train, y_train, preproc) -> Pipeline:
    return Pipeline(
        [
            ("pre", preproc),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    ).fit(X_train, y_train)


def fit_rf(X_train, y_train, preproc) -> Pipeline:
    return Pipeline(
        [
            ("pre", preproc),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ).fit(X_train, y_train)


def fit_xgb(X_train, y_train, preproc, scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("pre", preproc),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                ),
            ),
        ]
    ).fit(X_train, y_train)


def fit_xgb_smote(X_train, y_train, preproc) -> Pipeline:
    pre = preproc.fit(X_train, y_train)
    X_train_t = pre.transform(X_train)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_t, y_train)
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        tree_method="hist",
    ).fit(X_res, y_res)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_threshold_by_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_val, p_val)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    f1s = f1s[:-1]
    if len(f1s) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1s))])


def predict_with_threshold(model: Pipeline, X, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return prob, pred


def run_one(
    name: str,
    feature_set: str,
    fit_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
    save_model_to: Path | None = None,
) -> dict:
    print(f"[train] fitting {name} ({feature_set})...")
    model = fit_fn()
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    metrics = evaluate(y_test.to_numpy(), prob, pred)
    metrics["model"] = name
    metrics["feature_set"] = feature_set
    metrics["threshold"] = threshold
    print(
        f"[train]   acc={metrics['accuracy']:.3f} "
        f"bal_acc={metrics['balanced_accuracy']:.3f} "
        f"PR-AUC={metrics['pr_auc']:.3f} ROC-AUC={metrics['roc_auc']:.3f} "
        f"recall_serious={metrics['recall_serious']:.3f}"
    )
    if save_model_to is not None:
        joblib.dump(model, save_model_to)
    return metrics


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "features_with_clusters.parquet")
    df = df.sort_values("crash_datetime").reset_index(drop=True)

    train_df, test_df = chrono_split(df, TEST_FRACTION)
    val_split = int(len(train_df) * 0.9)
    fit_df = train_df.iloc[:val_split]
    val_df = train_df.iloc[val_split:]

    print(
        f"[train] sizes: fit={len(fit_df):,} val={len(val_df):,} test={len(test_df):,} "
        f"(severity rate train={train_df['severity'].mean():.4f}, test={test_df['severity'].mean():.4f})"
    )

    pos_weight = (fit_df["severity"] == 0).sum() / max((fit_df["severity"] == 1).sum(), 1)
    print(f"[train] scale_pos_weight (XGB): {pos_weight:.2f}")

    all_metrics: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}

    for fs_name, cols in FEATURE_SETS.items():
        feature_cols = cols["num"] + cols["cat"]
        X_fit = fit_df[feature_cols]
        y_fit = fit_df["severity"].astype(int)
        X_val = val_df[feature_cols]
        y_val = val_df["severity"].astype(int)
        X_train_full = train_df[feature_cols]
        y_train_full = train_df["severity"].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["severity"].astype(int)

        preproc = make_preprocessor(cols["num"], cols["cat"])

        save_models = fs_name == "full_with_cluster"

        all_metrics.append(
            run_one(
                "LogisticRegression",
                fs_name,
                lambda: fit_logistic(X_train_full, y_train_full, make_preprocessor(cols["num"], cols["cat"])),
                X_train_full,
                y_train_full,
                X_test,
                y_test,
                save_model_to=(MODELS_DIR / f"logreg_{fs_name}.joblib") if save_models else None,
            )
        )
        all_metrics.append(
            run_one(
                "RandomForest",
                fs_name,
                lambda: fit_rf(X_train_full, y_train_full, make_preprocessor(cols["num"], cols["cat"])),
                X_train_full,
                y_train_full,
                X_test,
                y_test,
                save_model_to=(MODELS_DIR / f"rf_{fs_name}.joblib") if save_models else None,
            )
        )

        xgb_model = fit_xgb(X_train_full, y_train_full, make_preprocessor(cols["num"], cols["cat"]), pos_weight)
        prob_test = xgb_model.predict_proba(X_test)[:, 1]
        pred_test = (prob_test >= 0.5).astype(int)
        m = evaluate(y_test.to_numpy(), prob_test, pred_test)
        m.update({"model": "XGBoost", "feature_set": fs_name, "threshold": 0.5})
        print(
            f"[train]   XGBoost ({fs_name}) acc={m['accuracy']:.3f} "
            f"PR-AUC={m['pr_auc']:.3f} recall_serious={m['recall_serious']:.3f}"
        )
        all_metrics.append(m)
        if save_models:
            joblib.dump(xgb_model, MODELS_DIR / f"xgb_{fs_name}.joblib")
            test_predictions["XGBoost"] = prob_test

        if fs_name == "full_with_cluster":
            xgb_smote = fit_xgb_smote(X_train_full, y_train_full, make_preprocessor(cols["num"], cols["cat"]))
            prob_smote = xgb_smote.predict_proba(X_test)[:, 1]
            pred_smote = (prob_smote >= 0.5).astype(int)
            m_smote = evaluate(y_test.to_numpy(), prob_smote, pred_smote)
            m_smote.update({"model": "XGBoost+SMOTE", "feature_set": fs_name, "threshold": 0.5})
            print(
                f"[train]   XGBoost+SMOTE acc={m_smote['accuracy']:.3f} "
                f"PR-AUC={m_smote['pr_auc']:.3f} recall_serious={m_smote['recall_serious']:.3f}"
            )
            all_metrics.append(m_smote)
            joblib.dump(xgb_smote, MODELS_DIR / "xgb_smote_full_with_cluster.joblib")
            test_predictions["XGBoost+SMOTE"] = prob_smote

            xgb_for_thr = fit_xgb(X_train_full, y_train_full, make_preprocessor(cols["num"], cols["cat"]), pos_weight)
            prob_val = xgb_for_thr.predict_proba(X_val)[:, 1]
            best_thr = best_threshold_by_f1(y_val.to_numpy(), prob_val)
            print(f"[train]   best F1 threshold on val: {best_thr:.3f}")
            prob_thr_test = xgb_for_thr.predict_proba(X_test)[:, 1]
            pred_thr_test = (prob_thr_test >= best_thr).astype(int)
            m_thr = evaluate(y_test.to_numpy(), prob_thr_test, pred_thr_test)
            m_thr.update(
                {
                    "model": "XGBoost+ThresholdTuned",
                    "feature_set": fs_name,
                    "threshold": float(best_thr),
                }
            )
            print(
                f"[train]   XGBoost+ThresholdTuned acc={m_thr['accuracy']:.3f} "
                f"PR-AUC={m_thr['pr_auc']:.3f} recall_serious={m_thr['recall_serious']:.3f} "
                f"f1_serious={m_thr['f1_serious']:.3f}"
            )
            all_metrics.append(m_thr)
            joblib.dump(xgb_for_thr, MODELS_DIR / "xgb_threshold_full_with_cluster.joblib")
            test_predictions["XGBoost+ThresholdTuned"] = prob_thr_test

    base = majority_baseline(train_df["severity"].astype(int), test_df["severity"].astype(int))
    all_metrics.insert(0, base)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    with open(METRICS_DIR / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[train] wrote {METRICS_DIR / 'model_metrics.csv'}")

    np.savez(
        METRICS_DIR / "test_predictions.npz",
        y_true=test_df["severity"].astype(int).to_numpy(),
        **{k.replace("+", "_").replace(" ", "_"): v for k, v in test_predictions.items()},
    )
    print(f"[train] wrote test predictions npz")


if __name__ == "__main__":
    main()

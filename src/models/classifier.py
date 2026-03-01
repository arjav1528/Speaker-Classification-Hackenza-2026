"""
Speaker Classification — Classifier Module

Two-track baseline:
    Track A: Handcrafted features → MI SelectKBest → StandardScaler → LightGBM
    Track B: wav2vec2 embeddings  → PCA            → StandardScaler → Logistic Regression

Both wrapped in sklearn Pipelines for clean CV and serialisation.
"""

import logging
import json
import os
from typing import Any

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve

from configs.config import (
    RANDOM_SEED,
    N_FOLDS,
    CV_REPEATS,
    FEATURE_SELECT_K,
    PCA_COMPONENTS,
    TRACK_A_PARAMS,
    TRACK_B_PARAMS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_track_a_pipeline",
    "build_track_b_pipeline",
    "get_all_pipelines",
    "optimize_threshold",
    "train_and_evaluate",
    "save_model",
    "load_model",
]


# ── Pipeline builders ─────────────────────────────────────────────────────────

def build_track_a_pipeline(n_features: int | None = None) -> Pipeline:
    """
    Track A: Handcrafted features + LightGBM.

    Pipeline: SelectKBest(MI) → StandardScaler → LGBMClassifier
    """
    from lightgbm import LGBMClassifier

    k = min(FEATURE_SELECT_K, n_features) if n_features else FEATURE_SELECT_K

    return Pipeline([
        ("select",  SelectKBest(mutual_info_classif, k=k)),
        ("scale",   StandardScaler()),
        ("clf",     LGBMClassifier(**TRACK_A_PARAMS)),
    ])


def build_track_b_pipeline(n_components: int | None = None) -> Pipeline:
    """
    Track B: wav2vec2 embeddings + Logistic Regression.

    Pipeline: StandardScaler → PCA → LogisticRegression
    """
    n_comp = n_components or PCA_COMPONENTS

    return Pipeline([
        ("scale",  StandardScaler()),
        ("pca",    PCA(n_components=n_comp, random_state=RANDOM_SEED)),
        ("clf",    LogisticRegression(**TRACK_B_PARAMS)),
    ])


def get_all_pipelines(n_features_a: int | None = None,
                      n_features_b: int | None = None) -> dict[str, Pipeline]:
    """Return both track pipelines keyed by name."""
    return {
        "Track A (LightGBM)":       build_track_a_pipeline(n_features_a),
        "Track B (LogReg+PCA)":     build_track_b_pipeline(n_features_b),
    }


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    minority_label: int = 1,
    metric: str = "balanced_accuracy",
) -> float:
    """
    Find the decision threshold that maximises balanced accuracy on the
    minority-class probability column.

    Parameters
    ----------
    y_true       : true binary labels (0/1 encoded)
    y_proba      : predicted probabilities for the positive (minority) class
    minority_label : which column index is the minority class (default 1)
    metric       : optimisation target (only "balanced_accuracy" supported)

    Returns
    -------
    best_threshold : float in [0, 1]
    """
    from sklearn.metrics import balanced_accuracy_score

    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = t

    logger.info("Optimal threshold: %.3f  (balanced_acc=%.4f)", best_thresh, best_score)
    return best_thresh


# ── Cross-validated training & evaluation ─────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    pipeline_name: str = "",
    n_folds: int = N_FOLDS,
    n_repeats: int = CV_REPEATS,
    labels: list | None = None,
) -> dict[str, Any]:
    """
    Run Repeated Stratified K-Fold CV on a single pipeline.

    Returns a dict with per-fold and aggregated metrics.
    """
    from src.utils.metrics import compute_metrics

    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=RANDOM_SEED
    )

    fold_metrics = []
    fold_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline_clone = _clone_pipeline(pipeline)
        pipeline_clone.fit(X_train, y_train)

        y_pred = pipeline_clone.predict(X_val)
        y_proba = None

        if hasattr(pipeline_clone, "predict_proba"):
            try:
                y_proba = pipeline_clone.predict_proba(X_val)
            except Exception:
                pass

        # Threshold optimisation on validation fold
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            best_thresh = optimize_threshold(y_val, y_proba[:, 1])
            y_pred_opt = (y_proba[:, 1] >= best_thresh).astype(int)
            fold_thresholds.append(best_thresh)
        else:
            y_pred_opt = y_pred
            fold_thresholds.append(0.5)

        m = compute_metrics(y_val, y_pred_opt, y_proba, labels=labels)
        m["fold"] = fold_idx
        m["threshold"] = fold_thresholds[-1]
        fold_metrics.append(m)

        logger.info(
            "%s | Fold %2d | balanced_acc=%.3f  f1_macro=%.3f  thresh=%.3f",
            pipeline_name or "Pipeline",
            fold_idx,
            m["balanced_accuracy"],
            m["f1_macro"],
            fold_thresholds[-1],
        )

    # ── aggregate ─────────────────────────────────────────────────────────────
    scalar_keys = [
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
        "f1_macro", "f1_weighted", "cohen_kappa", "matthews_corrcoef",
    ]
    if fold_metrics[0].get("roc_auc") is not None:
        scalar_keys.append("roc_auc")

    agg = {}
    for key in scalar_keys:
        vals = [m[key] for m in fold_metrics if m.get(key) is not None]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))

    agg["threshold_mean"] = float(np.mean(fold_thresholds))
    agg["threshold_std"]  = float(np.std(fold_thresholds))
    agg["n_folds"]        = n_folds
    agg["n_repeats"]      = n_repeats
    agg["total_folds"]    = len(fold_metrics)
    agg["pipeline_name"]  = pipeline_name

    return {"per_fold": fold_metrics, "aggregate": agg}


def _clone_pipeline(pipeline: Pipeline) -> Pipeline:
    """Deep-clone a pipeline so each fold gets a fresh copy."""
    from sklearn.base import clone
    return clone(pipeline)


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    model_dir: str,
    metrics: dict | None = None,
    threshold: float = 0.5,
    feature_info: dict | None = None,
) -> None:
    """Save trained pipeline, label encoder, metrics, and metadata."""
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(model_dir, "pipeline.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

    meta = {
        "threshold": threshold,
        **(feature_info or {}),
    }
    np.savez(os.path.join(model_dir, "feature_info.npz"), **meta)

    if metrics:
        # Filter out non-serialisable items
        clean = _make_serialisable(metrics)
        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(clean, f, indent=2)

    logger.info("Model saved to %s", model_dir)


def load_model(model_dir: str) -> tuple[Pipeline, LabelEncoder, dict]:
    """Load a saved model and return (pipeline, label_encoder, metadata)."""
    pipeline = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    info_path = os.path.join(model_dir, "feature_info.npz")
    meta = {}
    if os.path.exists(info_path):
        with np.load(info_path, allow_pickle=True) as data:
            meta = {k: data[k].item() if data[k].ndim == 0 else data[k]
                    for k in data.files}

    return pipeline, le, meta


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

"""
Metrics computation for Speaker Classification.
"""

import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> dict:
    """Compute comprehensive metrics for binary/multi-class classification."""
    m = {}

    m["accuracy"]            = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"]   = balanced_accuracy_score(y_true, y_pred)
    m["precision_macro"]     = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall_macro"]        = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_macro"]            = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_weighted"]         = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["cohen_kappa"]         = cohen_kappa_score(y_true, y_pred)
    m["matthews_corrcoef"]   = matthews_corrcoef(y_true, y_pred)
    m["confusion_matrix"]    = confusion_matrix(y_true, y_pred, labels=labels)
    m["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    if labels is not None:
        per_class_r = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        per_class_f = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        m["per_class_recall"] = {cls: float(round(v, 4)) for cls, v in zip(labels, per_class_r)}
        m["per_class_f1"]     = {cls: float(round(v, 4)) for cls, v in zip(labels, per_class_f)}
        m["min_class_recall"] = float(min(per_class_r))

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                m["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                m["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            m["roc_auc"] = None
        try:
            m["log_loss"] = log_loss(y_true, y_proba, labels=labels)
        except Exception:
            m["log_loss"] = None

    # Bias detection
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)
    total = len(y_pred)
    m["prediction_distribution"] = {str(k): int(v) for k, v in pred_counts.items()}

    bias_warnings = []
    for cls in (labels or list(true_counts.keys())):
        pred_ratio = pred_counts.get(cls, 0) / total
        true_ratio = true_counts.get(cls, 0) / total
        if pred_ratio < 0.05 and true_ratio > 0.10:
            bias_warnings.append(
                f"⚠ CLASS BIAS: '{cls}' is {true_ratio:.0%} of data but only "
                f"{pred_ratio:.0%} of predictions"
            )
    m["bias_warnings"] = bias_warnings

    return m


def compute_eer(y_true_bin: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal Error Rate — point where FAR == FRR."""
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)

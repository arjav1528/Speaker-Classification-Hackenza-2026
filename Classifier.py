import logging
import os
import json
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
N_FOLDS       = 5        # 5-fold stratified CV
RANDOM_SEED   = 42
MAX_FEATURES  = 80       # select top-K features via mutual information
                         # 218 → 80 reduces noise dimensions while keeping signal


# ── training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str,
) -> dict:
    """
    Train SVM + Random Forest with feature selection and hyperparameter tuning.

    Anti-overfitting measures:
    1. Feature selection (mutual information) — removes noise features
    2. StandardScaler — normalises feature ranges
    3. Stratified k-fold CV — honest evaluation
    4. GridSearchCV — finds best regularisation without manual tuning
    5. Balanced class weights — handles any class imbalance

    Parameters
    ----------
    X         : feature matrix (n_samples, n_features) — typically (320, 218)
    y         : string labels, e.g. ["native", "non_native", ...]
    model_dir : directory to save the trained model pipeline and metrics

    Returns
    -------
    dict with cross-validated metrics for both models
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info("Classes: %s  |  Samples: %s", list(le.classes_),
                dict(zip(*np.unique(y_enc, return_counts=True))))

    # ── determine feature selection k ─────────────────────────────────────────
    # don't select more features than we have
    k = min(MAX_FEATURES, X.shape[1])
    logger.info("Feature selection: keeping top %d of %d features (mutual information)",
                k, X.shape[1])

    # ── CV splitter ───────────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ── define model pipelines ────────────────────────────────────────────────
    # Each pipeline: feature_selection → scale → classify
    # This ensures feature selection is done INSIDE each CV fold (no leakage)

    svm_pipe = Pipeline([
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("scale",  StandardScaler()),
        ("clf",    SVC(probability=True, random_state=RANDOM_SEED,
                       class_weight="balanced")),
    ])

    rf_pipe = Pipeline([
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("scale",  StandardScaler()),
        ("clf",    RandomForestClassifier(random_state=RANDOM_SEED,
                                          class_weight="balanced",
                                          n_jobs=-1)),
    ])

    gbm_pipe = Pipeline([
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("scale",  StandardScaler()),
        ("clf",    GradientBoostingClassifier(random_state=RANDOM_SEED,
                                              n_iter_no_change=10)),
    ])

    # ── hyperparameter grids ──────────────────────────────────────────────────
    svm_params = {
        "clf__C":     [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf"],
    }

    rf_params = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth":    [None, 10, 20],
    }

    gbm_params = {
        "clf__n_estimators":  [50, 100, 200],
        "clf__max_depth":     [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample":     [0.8, 1.0],
    }

    models_and_params = {
        "svm": (svm_pipe, svm_params),
        "rf":  (rf_pipe,  rf_params),
        "gbm": (gbm_pipe, gbm_params),
    }

    # ── train + evaluate each model ───────────────────────────────────────────
    all_metrics: dict = {}
    all_grids:   dict = {}   # name → fitted GridSearchCV (for ensemble)
    best_acc   = -1.0
    best_name  = ""
    best_grid  = None

    for name, (pipe, params) in models_and_params.items():
        logger.info("─── %s: GridSearchCV (%d-fold) ───", name.upper(), N_FOLDS)

        grid = GridSearchCV(
            pipe, params,
            cv=cv,
            scoring="accuracy",
            refit=True,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X, y_enc)

        best_params = grid.best_params_
        best_score  = grid.best_score_

        # also get cross-validated predictions for detailed report
        y_pred = cross_val_predict(grid.best_estimator_, X, y_enc, cv=cv)
        acc    = accuracy_score(y_enc, y_pred)
        report = classification_report(y_enc, y_pred,
                                       target_names=le.classes_,
                                       output_dict=True)
        cm     = confusion_matrix(y_enc, y_pred)

        logger.info("%s best params: %s", name.upper(), best_params)
        logger.info("%s GridSearch best CV: %.2f%%  |  Full CV accuracy: %.2f%%",
                    name.upper(), best_score * 100, acc * 100)
        logger.info("\n%s", classification_report(y_enc, y_pred,
                                                   target_names=le.classes_))
        logger.info("Confusion matrix:\n%s", cm)

        all_metrics[name] = {
            "accuracy": float(round(acc, 4)),
            "grid_best_score": float(round(best_score, 4)),
            "best_params": {k: str(v) for k, v in best_params.items()},
            "report": report,
            "confusion_matrix": cm.tolist(),
        }

        all_grids[name] = grid

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_grid  = grid

    # ── save best individual model ─────────────────────────────────────────────
    logger.info("✓ Best individual model: %s (%.2f%% CV accuracy). Saving...",
                best_name.upper(), best_acc * 100)

    # grid.best_estimator_ is a fitted Pipeline (select → scale → clf)
    joblib.dump(best_grid.best_estimator_, os.path.join(model_dir, "pipeline.joblib"))
    joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

    # ── build soft-voting ensemble from tuned base models ─────────────────────
    # Soft voting averages predicted probabilities → adds zero learnable
    # parameters, so it cannot overfit beyond what the base models already do.
    # This is deliberately chosen over stacking, which would train a
    # meta-learner and risk overfitting on the small dataset.
    tuned_estimators = list(all_grids.items())   # [(name, fitted_grid), ...]
    ensemble = VotingClassifier(
        estimators=[(name, grid.best_estimator_) for name, grid in tuned_estimators],
        voting="soft",
        n_jobs=-1,
    )

    # Evaluate ensemble with honest CV — each fold re-fits from scratch,
    # so no base-model data leakage contaminates the evaluation.
    y_pred_ens = cross_val_predict(ensemble, X, y_enc, cv=cv)
    ens_acc    = accuracy_score(y_enc, y_pred_ens)
    ens_report = classification_report(y_enc, y_pred_ens,
                                       target_names=le.classes_,
                                       output_dict=True)
    ens_cm     = confusion_matrix(y_enc, y_pred_ens)

    # Fit ensemble on full data for saving (after honest CV evaluation above)
    ensemble.fit(X, y_enc)

    logger.info("─── ENSEMBLE (soft voting) ───")
    logger.info("Ensemble CV accuracy: %.2f%%", ens_acc * 100)
    logger.info("\n%s", classification_report(y_enc, y_pred_ens,
                                               target_names=le.classes_))
    logger.info("Confusion matrix:\n%s", ens_cm)

    all_metrics["ensemble"] = {
        "accuracy": float(round(ens_acc, 4)),
        "report": ens_report,
        "confusion_matrix": ens_cm.tolist(),
    }

    # Save ensemble
    joblib.dump(ensemble, os.path.join(model_dir, "ensemble.joblib"))
    logger.info("✓ Ensemble saved to %s", os.path.join(model_dir, "ensemble.joblib"))

    # ── save feature importance (from best individual model) ──────────────────
    selector = best_grid.best_estimator_.named_steps["select"]
    selected_mask = selector.get_support()
    feature_scores = selector.scores_
    n_selected = int(np.sum(selected_mask))
    logger.info("✓ %d / %d features selected by mutual information", n_selected, len(selected_mask))

    # ── save metrics ──────────────────────────────────────────────────────────
    meta = {
        "best_individual_model": best_name,
        "best_individual_cv_accuracy": float(round(best_acc, 4)),
        "ensemble_cv_accuracy": float(round(ens_acc, 4)),
        "n_samples": int(len(y)),
        "n_input_features": int(X.shape[1]),
        "n_selected_features": n_selected,
        "classes": list(le.classes_),
        "all_metrics": {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in all_metrics.items()
        },
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # save feature importance scores for analysis
    np.savez(os.path.join(model_dir, "feature_info.npz"),
             scores=feature_scores,
             selected_mask=selected_mask)

    logger.info("✓ Model artifacts saved to %s", model_dir)
    return all_metrics


# ── prediction ────────────────────────────────────────────────────────────────

def load_model(model_dir: str) -> tuple:
    """Load previously saved pipeline and label encoder."""
    pipeline = joblib.load(os.path.join(model_dir, "pipeline.joblib"))
    le       = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    return pipeline, le


def predict_file(
    X: np.ndarray,
    model_dir: str,
) -> dict:
    """
    Predict the label for a single file's pooled feature vector.

    Parameters
    ----------
    X         : feature vector (1, n_features) or (n_features,) — single file
    model_dir : directory containing saved model artifacts

    Returns
    -------
    {"label": str, "confidence": float, "probabilities": dict}
    """
    pipeline, le = load_model(model_dir)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    proba  = pipeline.predict_proba(X)[0]
    pred   = pipeline.predict(X)[0]
    label  = le.inverse_transform([pred])[0]
    conf   = float(np.max(proba))

    prob_dict = {cls: float(round(p, 4)) for cls, p in zip(le.classes_, proba)}

    return {
        "label": label,
        "confidence": float(round(conf, 4)),
        "probabilities": prob_dict,
    }

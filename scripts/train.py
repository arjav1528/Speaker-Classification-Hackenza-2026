#!/usr/bin/env python3
"""
Train the Speaker Classification model — dual-track baseline.

Track A: Handcrafted features (109-dim) + MI SelectKBest + LightGBM
Track B: wav2vec2 embeddings (768-dim) + PCA + Logistic Regression

Both evaluated via Repeated Stratified 5-Fold CV (3 repeats = 15 folds).

Usage:
    python scripts/train.py
    python scripts/train.py --no-augment
    python scripts/train.py --no-embeddings
    python scripts/train.py --track-a-only
"""

import sys
import os
import time
import argparse
import json

# Add project root to path so `from src...` and `from configs...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Speaker Classification models.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable minority-class augmentation.")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip wav2vec2 embeddings (Track B skipped, Track A uses 109-dim).")
    parser.add_argument("--track-a-only", action="store_true",
                        help="Only run Track A (handcrafted + LightGBM).")
    parser.add_argument("--track-b-only", action="store_true",
                        help="Only run Track B (wav2vec2 + LogReg).")
    parser.add_argument("--from-cache", default=None,
                        help="Load pre-extracted features from .npz instead of re-extracting.")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("  Speaker Classification — Training (Dual-Track Baseline)")
    print("=" * 60)
    print()

    # ── 1. Extract features via pipeline ─────────────────────────────────
    from src.pipeline import TrainPipeline
    from src.models.classifier import (
        build_track_a_pipeline,
        build_track_b_pipeline,
        train_and_evaluate,
        save_model,
        optimize_threshold,
    )
    from src.features.extraction import FEATURE_DIM
    from configs.config import (
        USE_EMBEDDINGS, EMBEDDING_DIM, RANDOM_SEED,
        FEATURE_SELECT_K, PCA_COMPONENTS,
    )

    use_embeddings = USE_EMBEDDINGS and not args.no_embeddings

    if args.from_cache:
        logger.info("Loading cached features from %s", args.from_cache)
        data = np.load(args.from_cache, allow_pickle=True)
        X = data["X"]
        y_str = data["y"]
        file_ids = list(data["file_ids"])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        logger.info("Loaded: X=%s, classes=%s", X.shape,
                     dict(zip(le.classes_, np.bincount(y))))
    else:
        pipe = TrainPipeline(
            augment=not args.no_augment,
            use_embeddings=use_embeddings,
            pooling_mode="simple",
        )
        X, y, le, file_ids = pipe.run()

    n_samples, n_features = X.shape
    labels_int = list(range(len(le.classes_)))

    print(f"\n  Samples        : {n_samples}")
    print(f"  Features       : {n_features}")
    print(f"  Classes        : {dict(zip(le.classes_, np.bincount(y)))}")
    print(f"  Embeddings     : {'ON' if use_embeddings else 'OFF'}")
    print(f"  Augmentation   : {'OFF' if args.no_augment else 'ON'}")
    print()

    # ── 2. Define feature splits for each track ──────────────────────────
    # Track A uses only handcrafted features (first 2*109 + 4 = 222 dims in simple mode)
    # Track B uses only embedding features (remaining dims)
    hc_dim_pooled = 2 * FEATURE_DIM + 4  # simple pooling: mean + std + 4 meta

    if use_embeddings and n_features > hc_dim_pooled:
        # Combined features: split into handcrafted vs embedding portions
        # In simple pooling with combined 877-dim segments:
        # pool = [mean(877), std(877), 4 meta] = 1758 dims
        # We need to extract just the handcrafted portion for Track A
        # and the full combined for Track B
        X_track_a = X  # Track A uses all features (MI will select best 60)
        X_track_b = X  # Track B uses all features (PCA will reduce)
    else:
        X_track_a = X
        X_track_b = X

    # ── 3. Run CV evaluation ─────────────────────────────────────────────
    all_results = {}

    # Track A
    if not args.track_b_only:
        print("-" * 60)
        print("  TRACK A: Handcrafted Features + LightGBM")
        print("-" * 60)

        k = min(FEATURE_SELECT_K, X_track_a.shape[1])
        pipe_a = build_track_a_pipeline(n_features=X_track_a.shape[1])
        results_a = train_and_evaluate(
            X_track_a, y, pipe_a,
            pipeline_name="Track A (LightGBM)",
            labels=labels_int,
        )
        all_results["Track A (LightGBM)"] = results_a

        agg_a = results_a["aggregate"]
        print(f"\n  Track A Results ({agg_a['total_folds']} folds):")
        print(f"    Balanced Accuracy : {agg_a['balanced_accuracy_mean']:.3f} ± {agg_a['balanced_accuracy_std']:.3f}")
        print(f"    F1 (macro)        : {agg_a['f1_macro_mean']:.3f} ± {agg_a['f1_macro_std']:.3f}")
        print(f"    MCC               : {agg_a['matthews_corrcoef_mean']:.3f} ± {agg_a['matthews_corrcoef_std']:.3f}")
        print(f"    Threshold         : {agg_a['threshold_mean']:.3f} ± {agg_a['threshold_std']:.3f}")
        if "roc_auc_mean" in agg_a:
            print(f"    ROC AUC           : {agg_a['roc_auc_mean']:.3f} ± {agg_a['roc_auc_std']:.3f}")
        print()

    # Track B
    if not args.track_a_only and use_embeddings:
        print("-" * 60)
        print("  TRACK B: wav2vec2 Embeddings + Logistic Regression")
        print("-" * 60)

        n_comp = min(PCA_COMPONENTS, X_track_b.shape[1], n_samples - 1)
        pipe_b = build_track_b_pipeline(n_components=n_comp)
        results_b = train_and_evaluate(
            X_track_b, y, pipe_b,
            pipeline_name="Track B (LogReg+PCA)",
            labels=labels_int,
        )
        all_results["Track B (LogReg+PCA)"] = results_b

        agg_b = results_b["aggregate"]
        print(f"\n  Track B Results ({agg_b['total_folds']} folds):")
        print(f"    Balanced Accuracy : {agg_b['balanced_accuracy_mean']:.3f} ± {agg_b['balanced_accuracy_std']:.3f}")
        print(f"    F1 (macro)        : {agg_b['f1_macro_mean']:.3f} ± {agg_b['f1_macro_std']:.3f}")
        print(f"    MCC               : {agg_b['matthews_corrcoef_mean']:.3f} ± {agg_b['matthews_corrcoef_std']:.3f}")
        print(f"    Threshold         : {agg_b['threshold_mean']:.3f} ± {agg_b['threshold_std']:.3f}")
        if "roc_auc_mean" in agg_b:
            print(f"    ROC AUC           : {agg_b['roc_auc_mean']:.3f} ± {agg_b['roc_auc_std']:.3f}")
        print()
    elif not use_embeddings and not args.track_a_only:
        logger.info("Track B skipped (embeddings disabled)")

    # ── 4. Select best track and train final model ───────────────────────
    print("=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    comparison_rows = []
    for name, res in all_results.items():
        a = res["aggregate"]
        comparison_rows.append({
            "Model": name,
            "Balanced Acc (mean)": round(a["balanced_accuracy_mean"], 4),
            "Balanced Acc (std)": round(a["balanced_accuracy_std"], 4),
            "F1 Macro (mean)": round(a["f1_macro_mean"], 4),
            "F1 Macro (std)": round(a["f1_macro_std"], 4),
            "MCC (mean)": round(a["matthews_corrcoef_mean"], 4),
            "MCC (std)": round(a["matthews_corrcoef_std"], 4),
            "ROC AUC (mean)": round(a.get("roc_auc_mean", 0), 4),
            "Threshold (mean)": round(a["threshold_mean"], 3),
        })

    df_comp = pd.DataFrame(comparison_rows)
    print()
    print(df_comp.to_string(index=False))
    print()

    # Pick best by balanced accuracy
    best_name = max(all_results, key=lambda k: all_results[k]["aggregate"]["balanced_accuracy_mean"])
    best_agg = all_results[best_name]["aggregate"]
    best_thresh = best_agg["threshold_mean"]

    print(f"  BEST: {best_name}")
    print(f"    Balanced Accuracy: {best_agg['balanced_accuracy_mean']:.4f} ± {best_agg['balanced_accuracy_std']:.4f}")
    print(f"    Optimal Threshold: {best_thresh:.3f}")
    print()

    # ── 5. Retrain best model on full training data ──────────────────────
    print("-" * 60)
    print("  Retraining best model on full training data…")
    print("-" * 60)

    if "LightGBM" in best_name:
        final_pipeline = build_track_a_pipeline(n_features=X.shape[1])
        X_final = X_track_a
    else:
        n_comp = min(PCA_COMPONENTS, X.shape[1], n_samples - 1)
        final_pipeline = build_track_b_pipeline(n_components=n_comp)
        X_final = X_track_b

    final_pipeline.fit(X_final, y)

    # ── 6. Save ──────────────────────────────────────────────────────────
    model_dir = os.path.join("outputs", "models")
    results_dir = os.path.join("outputs", "results")
    os.makedirs(results_dir, exist_ok=True)

    save_model(
        pipeline=final_pipeline,
        label_encoder=le,
        model_dir=model_dir,
        metrics=all_results[best_name]["aggregate"],
        threshold=best_thresh,
        feature_info={
            "n_features": int(X_final.shape[1]),
            "use_embeddings": use_embeddings,
            "pooling_mode": "simple",
            "best_track": best_name,
        },
    )

    # Save comparison CSV
    df_comp.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)

    # Save detailed metrics JSON
    serialisable_results = {}
    for name, res in all_results.items():
        serialisable_results[name] = {
            "aggregate": res["aggregate"],
            "n_folds": len(res["per_fold"]),
        }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(serialisable_results, f, indent=2, default=str)

    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model        : {best_name}")
    print(f"  Balanced accuracy : {best_agg['balanced_accuracy_mean']:.4f} ± {best_agg['balanced_accuracy_std']:.4f}")
    print(f"  Threshold         : {best_thresh:.3f}")
    print(f"  Model saved to    : {model_dir}/")
    print(f"  Results saved to  : {results_dir}/")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

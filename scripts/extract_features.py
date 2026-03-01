#!/usr/bin/env python3
"""
Extract features from all training audio files and save to disk.

Pipeline per file:
    1. Load CSV  →  resolve audio path
    2. preprocess_in_memory()  →  clean mono 16 kHz + VAD
    3. segment_into_buckets()  →  3.0s windows, 1.0s overlap (hop=2.0s)
    4. extract_combined_batch()  →  109 handcrafted + 768 wav2vec2 per segment
    5. compute_segment_weights()  →  voiced_ratio × log(1+rms) per segment
    6. pool_segment_features()  →  mean + std + weighted_mean + weighted_std + meta

Saves:
    outputs/features/train_features.npz   (X, y, file_ids, feature_dim)

Usage:
    python scripts/extract_features.py
    python scripts/extract_features.py --no-embeddings
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from training audio.")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip wav2vec2 embeddings (faster, uses 109-dim instead of 877-dim per segment).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "features", "train_features.npz"),
        help="Path to save the feature matrix (default: outputs/features/train_features.npz).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # ------------------------------------------------------------------
    # 0. Optionally disable embeddings before importing feature modules
    # ------------------------------------------------------------------
    if args.no_embeddings:
        import configs.config as cfg
        cfg.USE_EMBEDDINGS = False
        logger.info("wav2vec2 embeddings DISABLED (--no-embeddings)")

    # ------------------------------------------------------------------
    # 1. Imports (after config override)
    # ------------------------------------------------------------------
    from src.utils.io import load_train_csv, resolve_audio_path
    from src.preprocessing.audio import preprocess_in_memory
    from src.preprocessing.segment import segment_into_buckets
    from src.features.extraction import (
        extract_combined_batch,
        pool_segment_features,
        compute_segment_weights,
    )
    from configs.config import TARGET_SR, USE_EMBEDDINGS, OVERLAP_SEC, BUCKET_DURATION

    # ------------------------------------------------------------------
    # 2. Load CSV
    # ------------------------------------------------------------------
    df = load_train_csv()
    logger.info("=" * 60)
    logger.info("  Feature Extraction — %d files", len(df))
    logger.info("  Embeddings     : %s", "ON (wav2vec2-base, 768-dim)" if USE_EMBEDDINGS else "OFF")
    logger.info("  Segmentation   : %.1fs windows, %.1fs overlap (hop=%.1fs)",
                BUCKET_DURATION, OVERLAP_SEC, BUCKET_DURATION - OVERLAP_SEC)
    logger.info("  Pooling        : mean + std + weighted_mean + weighted_std + 4 meta")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 3. Process each file
    # ------------------------------------------------------------------
    X_list = []
    y_list = []
    file_ids = []
    skipped = []

    for idx, row in df.iterrows():
        dp_id = row["dp_id"]
        label = row["nativity_status"]
        audio_rel = row["audio_url"]

        audio_path = resolve_audio_path(audio_rel)
        if audio_path is None:
            logger.warning("[%3d] %-6s | File not found: %s — SKIPPING", idx, dp_id, audio_rel)
            skipped.append((dp_id, audio_rel, "file_not_found"))
            continue

        try:
            # 3a. Preprocess (load → clean → VAD)
            y_audio, sr = preprocess_in_memory(audio_path, sr=TARGET_SR)

            if len(y_audio) < sr * 0.3:  # < 0.3s of speech after VAD
                logger.warning("[%3d] %-6s | Too short after VAD (%.2fs) — SKIPPING",
                               idx, dp_id, len(y_audio) / sr)
                skipped.append((dp_id, audio_rel, "too_short"))
                continue

            # 3b. Segment into 3s buckets with 1s overlap
            segments, timestamps = segment_into_buckets(
                y_audio, sr,
                bucket_duration_sec=BUCKET_DURATION,
                overlap_sec=OVERLAP_SEC,
                label=str(dp_id),
            )

            if not segments:
                logger.warning("[%3d] %-6s | No segments produced — SKIPPING", idx, dp_id)
                skipped.append((dp_id, audio_rel, "no_segments"))
                continue

            # 3c. Extract features per segment (handcrafted + optional embeddings)
            seg_features = extract_combined_batch(segments, sr, label=str(dp_id))

            # 3d. Compute per-segment weights (voiced_ratio × log(1+rms))
            weights, voiced_ratios, rms_energies = compute_segment_weights(segments, sr)

            # 3e. Pool across segments → single file-level vector
            file_vector = pool_segment_features(
                seg_features,
                weights=weights,
                voiced_ratios=voiced_ratios,
                rms_energies=rms_energies,
            )

            X_list.append(file_vector)
            y_list.append(label)
            file_ids.append(dp_id)

            logger.info(
                "[%3d] %-6s | %-11s | %d segs → %d features  ✓",
                idx, dp_id, label, len(segments), len(file_vector),
            )

        except Exception as exc:
            logger.error("[%3d] %-6s | FAILED: %s", idx, dp_id, exc)
            skipped.append((dp_id, audio_rel, str(exc)))
            continue

    # ------------------------------------------------------------------
    # 4. Stack & Save
    # ------------------------------------------------------------------
    if not X_list:
        logger.error("No features extracted! Check your audio files.")
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.array(y_list)
    ids = np.array(file_ids)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.output,
        X=X,
        y=y,
        file_ids=ids,
        feature_dim=X.shape[1],
        use_embeddings=USE_EMBEDDINGS,
    )

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Files processed : {len(X_list)} / {len(df)}")
    print(f"  Skipped         : {len(skipped)}")
    print(f"  Feature matrix  : {X.shape}  (samples × features)")
    print(f"  Embeddings      : {'ON' if USE_EMBEDDINGS else 'OFF'}")
    print(f"  Segmentation    : {BUCKET_DURATION}s windows, {OVERLAP_SEC}s overlap")
    print(f"  Pooling         : mean + std + weighted_mean + weighted_std + 4 meta")
    print(f"  Labels          : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Saved to        : {args.output}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print("=" * 60)

    if skipped:
        print(f"\n  Skipped files ({len(skipped)}):")
        for dp_id, path, reason in skipped:
            print(f"    {dp_id} | {path} | {reason}")


if __name__ == "__main__":
    main()

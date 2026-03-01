#!/usr/bin/env python3
"""
Run predictions on audio files or the test dataset.

Usage:
    python scripts/predict.py                                    # predict test set
    python scripts/predict.py path/to/audio.wav                  # predict single file
    python scripts/predict.py --model-dir outputs/models/ *.wav  # predict multiple
"""

import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Speaker Classification — Prediction")
    parser.add_argument("audio_files", nargs="*", default=[],
                        help="Audio file(s) to predict. If empty, predicts the test set.")
    parser.add_argument("--model-dir", default="outputs/models",
                        help="Path to saved model directory.")
    parser.add_argument("--output", default="outputs/results/test_predictions.csv",
                        help="Path for prediction output CSV.")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable wav2vec2 embeddings (must match training).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Speaker Classification — Prediction")
    print("=" * 60)
    print()

    from src.inference.predictor import Predictor
    from src.utils.io import load_test_csv, resolve_audio_path, SHARE_DATA_DIR

    use_embeddings = not args.no_embeddings

    predictor = Predictor(
        model_dir=args.model_dir,
        use_embeddings=use_embeddings,
        pooling_mode="simple",
    )

    if args.audio_files:
        # ── Predict specific file(s) ─────────────────────────────────────
        audio_paths = []
        for pattern in args.audio_files:
            audio_paths.extend(glob.glob(pattern))

        if not audio_paths:
            print("  No audio files found matching the given patterns.")
            sys.exit(1)

        df_results = predictor.predict_batch(audio_paths)
    else:
        # ── Predict test dataset ─────────────────────────────────────────
        df_test = load_test_csv()
        audio_paths = []
        dp_ids = []

        test_wav_dir = os.path.join(SHARE_DATA_DIR, "test", "wav")

        for _, row in df_test.iterrows():
            dp_id = row["dp_id"]
            audio_rel = row.get("audio_url", "")

            # Try to resolve from test directory
            audio_path = None

            # First: try direct resolution
            if pd.notna(audio_rel) and str(audio_rel).strip():
                audio_path = resolve_audio_path(audio_rel, base_dir=SHARE_DATA_DIR)

            # Second: try test/wav/<dp_id>.wav
            if audio_path is None and os.path.isdir(test_wav_dir):
                candidate = os.path.join(test_wav_dir, f"{dp_id}.wav")
                if os.path.isfile(candidate):
                    audio_path = candidate

            if audio_path is None:
                logger.warning("Test file not found for dp_id=%s", dp_id)
                continue

            audio_paths.append(audio_path)
            dp_ids.append(dp_id)

        if not audio_paths:
            print("  No test audio files found. Check share-data/test/wav/")
            sys.exit(1)

        print(f"  Predicting {len(audio_paths)} test files…\n")

        df_results = predictor.predict_batch(audio_paths)
        df_results.insert(0, "dp_id", dp_ids[:len(df_results)])

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results.to_csv(args.output, index=False)

    # ── Print summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  PREDICTION RESULTS")
    print("=" * 60)

    cols = ["dp_id", "file", "label", "confidence"] if "dp_id" in df_results.columns else ["file", "label", "confidence"]
    available_cols = [c for c in cols if c in df_results.columns]
    print(df_results[available_cols].to_string(index=False))

    print(f"\n  Total predictions : {len(df_results)}")
    print(f"  Distribution      : {df_results['label'].value_counts().to_dict()}")
    print(f"  Saved to          : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

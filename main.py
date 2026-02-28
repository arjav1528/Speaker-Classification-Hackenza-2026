"""
Speaker Classification — Native vs Non-Native

CLI entry point that orchestrates the full pipeline:
  1. preprocess  — standardize raw audio files
  2. train       — VAD → segment → extract features → train classifier
  3. predict     — classify a single audio file
"""
import argparse
import logging
import os
import sys

import librosa
import numpy as np

from Stdaudio import standardize_audio, batch_standardize, highpass_filter, rms_normalize, check_clipping
from VAD import extract_speech_only
from Segment import segment_into_buckets
from Features import extract_features, extract_features_batch, pool_segment_features
from Classifier import train_and_evaluate, predict_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
TARGET_SR        = 16_000


# ── helpers ────────────────────────────────────────────────────────────────────

def _discover_audio_files(data_dir: str) -> list[tuple[str, str]]:
    """
    Walk a directory structured as data_dir/<class_label>/*.audio
    Returns list of (file_path, label) tuples.
    """
    pairs = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                pairs.append((os.path.join(label_dir, fname), label))

    if not pairs:
        raise FileNotFoundError(
            f"No audio files found in {data_dir}. "
            f"Expected structure: {data_dir}/<class_label>/<audio_files>"
        )

    logger.info("Found %d audio files across %d classes in %s",
                len(pairs), len({l for _, l in pairs}), data_dir)
    return pairs


def _process_single_audio(audio_path: str, sr: int = TARGET_SR
                           ) -> tuple[np.ndarray, int]:
    """
    Run the full preprocessing chain on one audio file (in-memory, no save).
    Stdaudio → VAD → returns clean speech signal.
    """
    # Step 1: Load + standardize (in-memory version of Stdaudio pipeline)
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # DC offset removal
    y = y - np.mean(y)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=35)

    if len(y) / sr < 0.1:
        raise ValueError(f"Audio too short after trimming: {audio_path}")

    # High-pass filter + normalize (reuse Stdaudio helpers)
    y = highpass_filter(y, sr)
    y, _ = rms_normalize(y)
    y = check_clipping(y, audio_path)

    # Step 2: VAD — strip non-speech
    y, _ = extract_speech_only(y, sr)

    return y, sr


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_preprocess(args: argparse.Namespace) -> None:
    """Standardize all raw audio files → save to output directory."""
    file_pairs = _discover_audio_files(args.data_dir)
    output_dir = args.output_dir
    batch_items = []

    for src_path, label in file_pairs:
        # mirror directory structure in output
        rel    = os.path.relpath(src_path, args.data_dir)
        stem   = os.path.splitext(rel)[0]
        dst    = os.path.join(output_dir, stem + ".wav")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        batch_items.append((src_path, dst))

    results = batch_standardize(batch_items, target_sr=TARGET_SR)
    failed  = sum(1 for v in results.values() if v != "ok")
    if failed:
        logger.warning("%d / %d files failed preprocessing.", failed, len(results))
    else:
        logger.info("✓ All %d files preprocessed → %s", len(results), output_dir)


def cmd_train(args: argparse.Namespace) -> None:
    """Full training pipeline: load → VAD → segment → features → pool → train."""
    file_pairs = _discover_audio_files(args.data_dir)

    all_features = []   # one pooled vector per file
    all_labels   = []   # one label per file
    skipped      = 0

    for i, (fpath, label) in enumerate(file_pairs):
        fname = os.path.basename(fpath)
        try:
            # load already-standardised audio
            y, sr = librosa.load(fpath, sr=TARGET_SR, mono=True)

            # VAD
            y_speech, _ = extract_speech_only(y, sr)

            # Segment into 3s buckets
            buckets, timestamps = segment_into_buckets(
                y_speech, sr,
                bucket_duration_sec=3.0,
                min_bucket_ratio=0.5,
                pad_last=True,
                label=fname,
            )

            # Extract features per bucket → (n_buckets, 109)
            seg_feats = extract_features_batch(buckets, sr, label=fname)

            # Pool across segments → single (218,) vector for this file
            pooled = pool_segment_features(seg_feats)

            all_features.append(pooled)
            all_labels.append(label)   # ONE label per file — no leakage

            if (i + 1) % 50 == 0 or i == len(file_pairs) - 1:
                logger.info("Progress: %d / %d files processed", i + 1, len(file_pairs))

        except Exception as exc:
            logger.error("%s | Skipped: %s", fname, exc)
            skipped += 1

    if not all_features:
        logger.error("No features extracted — cannot train. Aborting.")
        sys.exit(1)

    X = np.vstack(all_features)
    y = np.array(all_labels)
    logger.info("Feature matrix: %s (1 vector per file)  |  Skipped: %d", X.shape, skipped)

    # Train & evaluate
    metrics = train_and_evaluate(X, y, model_dir=args.model_dir)
    logger.info("✓ Training complete. Model saved to %s", args.model_dir)


def cmd_predict(args: argparse.Namespace) -> None:
    """Classify a single audio file using the trained model."""
    if not os.path.isdir(args.model_dir):
        logger.error("Model directory not found: %s", args.model_dir)
        sys.exit(1)

    fpath = args.audio_path
    fname = os.path.basename(fpath)

    # Preprocess in-memory
    y, sr = _process_single_audio(fpath, sr=TARGET_SR)

    # Segment into 3s buckets
    buckets, _ = segment_into_buckets(
        y, sr,
        bucket_duration_sec=3.0,
        min_bucket_ratio=0.5,
        pad_last=True,
        label=fname,
    )

    # Extract features per segment → pool into one vector
    seg_feats = extract_features_batch(buckets, sr, label=fname)
    pooled    = pool_segment_features(seg_feats)

    # Predict
    result = predict_file(pooled, model_dir=args.model_dir)

    print(f"\n{'═' * 50}")
    print(f"  File:          {fname}")
    print(f"  Segments used: {len(buckets)}")
    print(f"  Prediction:    {result['label']}")
    print(f"  Confidence:    {result['confidence']:.1%}")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"{'═' * 50}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Native vs Non-Native Speaker Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # preprocess
    pp = sub.add_parser("preprocess", help="Standardize raw audio files")
    pp.add_argument("--data-dir",   required=True,
                    help="Root dir with <class_label>/<audio_files>")
    pp.add_argument("--output-dir", required=True,
                    help="Destination for standardized WAVs")

    # train
    tr = sub.add_parser("train", help="Train the classifier")
    tr.add_argument("--data-dir",   required=True,
                    help="Root dir with standardized audio (or raw audio)")
    tr.add_argument("--model-dir",  default="models",
                    help="Directory to save trained model (default: models/)")

    # predict
    pr = sub.add_parser("predict", help="Classify a single audio file")
    pr.add_argument("--audio-path", required=True,
                    help="Path to the audio file to classify")
    pr.add_argument("--model-dir",  default="models",
                    help="Directory containing trained model (default: models/)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "preprocess": cmd_preprocess,
        "train":      cmd_train,
        "predict":    cmd_predict,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()

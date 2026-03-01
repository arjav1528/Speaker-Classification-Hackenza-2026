"""
Data loading utilities for Speaker Classification.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARE_DATA_DIR = os.path.join(BASE_DIR, "share-data")
TRAIN_CSV      = os.path.join(SHARE_DATA_DIR, "wav_converted.csv")
TEST_CSV       = os.path.join(SHARE_DATA_DIR, "test", "test_wav_converted.csv")


def load_train_csv() -> pd.DataFrame:
    """Load training CSV with columns: dp_id, audio_url, nativity_status, language."""
    df = pd.read_csv(TRAIN_CSV)
    logger.info("Train CSV: %d rows  |  Columns: %s", len(df), list(df.columns))
    logger.info("Label distribution:\n%s", df["nativity_status"].value_counts().to_string())
    return df


def load_test_csv() -> pd.DataFrame:
    """Load test CSV (unlabeled)."""
    df = pd.read_csv(TEST_CSV)
    logger.info("Test CSV: %d rows", len(df))
    return df


def resolve_audio_path(audio_rel: str, base_dir: str = SHARE_DATA_DIR) -> str | None:
    """
    Resolve a relative audio path from the CSV to an absolute path.
    Checks the direct path first, then looks in Native/ and Non-Native/ subdirs.
    Returns None if not found.
    """
    if pd.isna(audio_rel) or str(audio_rel).strip() == "":
        return None

    audio_path = os.path.join(base_dir, str(audio_rel))

    if os.path.isfile(audio_path):
        return audio_path

    # Try class subdirectories
    fname  = os.path.basename(audio_path)
    parent = os.path.dirname(audio_path)
    for subdir in ("Native", "Non-Native"):
        candidate = os.path.join(parent, subdir, fname)
        if os.path.isfile(candidate):
            return candidate

    return None

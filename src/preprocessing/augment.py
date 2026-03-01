"""
In-memory audio augmentation for minority-class balancing.

Augmentation pipeline (all safe for accent/nativity classification):
    1. Gain          ±4 dB         (p=1.0)  — volume irrelevant, gets RMS-normalized
    2. Gaussian Noise 0.001–0.005  (p=0.5)  — barely audible, SNR ~23 dB
    3. Pitch Shift   ±1 semitone   (p=0.5)  — conservative, preserves formants
    4. Time Stretch  0.95–1.05×    (p=0.5)  — within natural speaking-rate variation

Usage:
    audios_aug, labels_aug = augment_minority(audios, labels, sr, n_augments=3)
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Augmentation pipeline builder
# ---------------------------------------------------------------------------

def build_augmenter() -> Compose:
    """Return the standard augmentation pipeline."""
    return Compose([
        Gain(min_gain_db=-4.0, max_gain_db=4.0, p=1.0),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
        PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.5),
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
    ])


# ---------------------------------------------------------------------------
# Minority-class augmentation
# ---------------------------------------------------------------------------

def augment_minority(
    audios: list[np.ndarray],
    labels: list[str],
    sr: int,
    n_augments: int = 3,
    *,
    seed: int | None = 42,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Augment only the minority class in-memory.

    Parameters
    ----------
    audios : list[np.ndarray]
        Raw audio arrays (float32, mono, already preprocessed).
    labels : list[str]
        Corresponding class labels (e.g. "Native" / "Non-Native").
    sr : int
        Sample rate of the audio arrays.
    n_augments : int
        Number of augmented copies per minority sample.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    (audios_out, labels_out)
        Original samples + augmented minority samples appended.
    """
    if seed is not None:
        np.random.seed(seed)

    counts = Counter(labels)
    if len(counts) < 2:
        logger.warning("Only one class present — skipping augmentation.")
        return audios, labels

    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)

    n_minority = counts[minority_class]
    n_majority = counts[majority_class]

    logger.info(
        "Class distribution before augmentation: %s=%d, %s=%d",
        majority_class, n_majority, minority_class, n_minority,
    )

    augmenter = build_augmenter()

    # Start with copies of all originals
    audios_out = list(audios)
    labels_out = list(labels)

    n_generated = 0
    for audio, label in zip(audios, labels):
        if label != minority_class:
            continue
        for _ in range(n_augments):
            augmented = augmenter(samples=audio.astype(np.float32), sample_rate=sr)
            audios_out.append(augmented)
            labels_out.append(label)
            n_generated += 1

    new_counts = Counter(labels_out)
    logger.info(
        "After augmentation: %s=%d, %s=%d  (+%d augmented)",
        majority_class, new_counts[majority_class],
        minority_class, new_counts[minority_class],
        n_generated,
    )

    return audios_out, labels_out

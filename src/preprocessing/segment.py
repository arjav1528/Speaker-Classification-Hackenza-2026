import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────────────
DEFAULT_BUCKET_SEC    = 3.0   # 3s balances prosodic context vs. more training samples
DEFAULT_MIN_RATIO     = 0.5   # keep final segment if >= 50% of a full bucket
DEFAULT_OVERLAP_SEC   = 1.0   # 1s overlap → hop = 2.0s


def segment_into_buckets(
    y: np.ndarray,
    sr: int,
    bucket_duration_sec: float = DEFAULT_BUCKET_SEC,
    min_bucket_ratio: float    = DEFAULT_MIN_RATIO,
    overlap_sec: float         = DEFAULT_OVERLAP_SEC,
    pad_last: bool             = True,
    label: str                 = "",
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Divide an audio signal into fixed-size buckets, with optional overlap.

    Parameters
    ----------
    y                   : float32 audio signal
    sr                  : sample rate
    bucket_duration_sec : length of each bucket in seconds.
    min_bucket_ratio    : minimum fraction of a full bucket required to keep the
                          final (partial) segment.
    overlap_sec         : overlap between consecutive buckets in seconds.
    pad_last            : if True, zero-pad the final partial segment up to full
                          bucket length.
    label               : optional identifier string used in log messages.

    Returns
    -------
    buckets    : list of np.ndarray, each of length (bucket_size,) if pad_last=True
    timestamps : list of (start_sec, end_sec) tuples in the *original* signal
    """
    # ── validation ───────────────────────────────────────────────────────────────────────
    if bucket_duration_sec <= 0:
        raise ValueError(f"bucket_duration_sec must be positive, got {bucket_duration_sec}.")
    if not (0.0 < min_bucket_ratio <= 1.0):
        raise ValueError(f"min_bucket_ratio must be in (0, 1], got {min_bucket_ratio}.")
    if overlap_sec < 0:
        raise ValueError(f"overlap_sec must be >= 0, got {overlap_sec}.")
    if overlap_sec >= bucket_duration_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}s) must be less than "
            f"bucket_duration_sec ({bucket_duration_sec}s)."
        )

    bucket_size  = int(bucket_duration_sec * sr)
    hop_size     = int((bucket_duration_sec - overlap_sec) * sr)
    min_samples  = int(bucket_size * min_bucket_ratio)
    total_dur    = len(y) / sr

    if len(y) < min_samples:
        raise ValueError(
            f"{label or 'Signal'} is too short ({total_dur:.2f}s) to produce even one "
            f"bucket at min_bucket_ratio={min_bucket_ratio} "
            f"(need >= {min_samples / sr:.2f}s)."
        )

    # ── segmentation ─────────────────────────────────────────────────────────────────────
    buckets: list[np.ndarray]          = []
    timestamps: list[tuple[float, float]] = []

    for start in range(0, len(y), hop_size):
        end     = start + bucket_size
        segment = y[start:end]

        if len(segment) < min_samples:
            logger.debug(
                "%s | Dropping final segment [%.2fs–%.2fs]: %.0f%% of full bucket.",
                label or "segment",
                start / sr, min(end, len(y)) / sr,
                100 * len(segment) / bucket_size,
            )
            break

        if len(segment) < bucket_size:
            if pad_last:
                segment = np.pad(segment, (0, bucket_size - len(segment)))

        buckets.append(segment)
        timestamps.append((start / sr, min(end, len(y)) / sr))

    n_buckets  = len(buckets)
    kept_dur   = n_buckets * bucket_duration_sec if pad_last else sum(e - s for s, e in timestamps)

    logger.info(
        "%s | %d bucket(s) × %.1fs = %.1fs kept of %.1fs total (overlap=%.1fs, hop=%.1fs).",
        label or "segment_into_buckets",
        n_buckets,
        bucket_duration_sec,
        kept_dur,
        total_dur,
        overlap_sec,
        bucket_duration_sec - overlap_sec,
    )

    return buckets, timestamps

import logging
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
MIN_DURATION_SEC = 0.1      # files shorter than this (after trim) are rejected
DEFAULT_SR       = 16_000
DEFAULT_DBFS     = -23.0    # RMS target; -23 gives headroom before 0 dBFS
DEFAULT_TRIM_DB  = 35       # less aggressive than 20 — preserves soft onsets
DEFAULT_HP_CUTOFF = 30      # Hz
DEFAULT_HP_ORDER  = 5


# ── helpers ────────────────────────────────────────────────────────────────────

def highpass_filter(y: np.ndarray, sr: int, cutoff: float = DEFAULT_HP_CUTOFF,
                    order: int = DEFAULT_HP_ORDER) -> np.ndarray:
    """
    Zero-phase high-pass filter using second-order sections (numerically stable).
    sosfilt avoids the phase distortion of lfilter and is more stable at high orders.
    """
    sos = butter(order, cutoff / (0.5 * sr), btype="high", output="sos")
    return sosfilt(sos, y)


def rms_normalize(y: np.ndarray, target_dbfs: float = DEFAULT_DBFS
                  ) -> tuple[np.ndarray, float]:
    """
    Normalize audio to a target RMS level (dBFS).
    Returns the normalized signal and the gain applied (dB) for logging.
    """
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        logger.warning("Signal is silent — skipping RMS normalization.")
        return y, 0.0
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    y = y * (10 ** (gain_db / 20))
    return y, gain_db


def check_clipping(y: np.ndarray, input_path: str) -> np.ndarray:
    """
    Warn if the signal clips after normalization instead of silently rescaling.
    Only applies a safety hard-limit to prevent file corruption — the RMS
    normalization result is preserved as closely as possible.
    """
    peak = np.max(np.abs(y))
    if peak > 1.0:
        logger.warning(
            "%s | Peak %.4f exceeds 0 dBFS after normalization. "
            "Consider lowering target_dbfs. Applying hard limiter.",
            input_path, peak,
        )
        y = np.clip(y, -1.0, 1.0)   # hard clip, not rescale — level intent kept
    return y


# ── main pipeline ──────────────────────────────────────────────────────────────

def standardize_audio(
    input_path: str,
    output_path: str,
    target_sr: int   = DEFAULT_SR,
    trim_silence: bool = True,
    top_db: float    = DEFAULT_TRIM_DB,
    target_dbfs: float = DEFAULT_DBFS,
    hp_cutoff: float = DEFAULT_HP_CUTOFF,
    hp_order: int    = DEFAULT_HP_ORDER,
) -> tuple[np.ndarray, int]:
    """
    Load audio and standardize it for ML training.

    Pipeline
    --------
    1. Load → mono → resample
    2. Remove DC offset
    3. Trim silence
    4. Duration guard
    5. High-pass filter  (sosfilt, zero-phase)
    6. RMS normalize     (with gain logging)
    7. Clip guard        (warn + hard-limit, not silent rescale)
    8. Save as PCM_16 WAV

    Returns
    -------
    (y, target_sr) — processed signal and sample rate
    """

    # 1. Load ──────────────────────────────────────────────────────────────────
    try:
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load '{input_path}': {exc}") from exc

    if y.size == 0:
        raise ValueError(f"'{input_path}' loaded as an empty array.")

    # 2. Remove DC offset ──────────────────────────────────────────────────────
    y = y - np.mean(y)

    # 3. Trim silence ──────────────────────────────────────────────────────────
    if trim_silence:
        y, trim_indices = librosa.effects.trim(y, top_db=top_db)
        logger.info(
            "%s | Trimmed %.2fs → %.2fs",
            input_path,
            trim_indices[0] / target_sr,
            trim_indices[1] / target_sr,
        )

    # 4. Duration guard ────────────────────────────────────────────────────────
    duration = len(y) / target_sr
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"'{input_path}' is only {duration:.3f}s after trimming "
            f"(minimum {MIN_DURATION_SEC}s). Skipping."
        )

    # 5. High-pass filter ──────────────────────────────────────────────────────
    y = highpass_filter(y, target_sr, cutoff=hp_cutoff, order=hp_order)

    # 6. RMS normalize ─────────────────────────────────────────────────────────
    y, gain_db = rms_normalize(y, target_dbfs=target_dbfs)
    logger.info("%s | Applied gain: %+.1f dB", input_path, gain_db)

    # 7. Clip guard ────────────────────────────────────────────────────────────
    y = check_clipping(y, input_path)

    # 8. Save ──────────────────────────────────────────────────────────────────
    sf.write(output_path, y, target_sr, subtype="PCM_16")
    logger.info("%s | Saved → %s (%.2fs, %d Hz)", input_path, output_path, duration, target_sr)

    return y, target_sr


# ── batch helper ───────────────────────────────────────────────────────────────

def batch_standardize(
    file_pairs: list[tuple[str, str]],
    **kwargs,
) -> dict[str, str]:
    """
    Process multiple files.  Failures are caught and reported without
    stopping the rest of the batch.

    Parameters
    ----------
    file_pairs : list of (input_path, output_path) tuples
    **kwargs   : forwarded to standardize_audio()

    Returns
    -------
    dict mapping input_path → "ok" | error message
    """
    results = {}
    for input_path, output_path in file_pairs:
        try:
            standardize_audio(input_path, output_path, **kwargs)
            results[input_path] = "ok"
        except Exception as exc:
            logger.error("%s | FAILED: %s", input_path, exc)
            results[input_path] = str(exc)

    total  = len(results)
    passed = sum(1 for v in results.values() if v == "ok")
    logger.info("Batch complete: %d/%d succeeded.", passed, total)
    return results
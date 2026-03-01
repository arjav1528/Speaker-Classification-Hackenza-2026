import logging
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

from configs.config import TARGET_SR

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
MIN_DURATION_SEC = 0.1
DEFAULT_SR       = TARGET_SR
DEFAULT_DBFS     = -23.0
DEFAULT_TRIM_DB  = 35
DEFAULT_HP_CUTOFF = 30
DEFAULT_HP_ORDER  = 5


# ── helpers ────────────────────────────────────────────────────────────────────

def highpass_filter(y: np.ndarray, sr: int, cutoff: float = DEFAULT_HP_CUTOFF,
                    order: int = DEFAULT_HP_ORDER) -> np.ndarray:
    """High-pass filter using second-order sections (numerically stable)."""
    sos = butter(order, cutoff / (0.5 * sr), btype="high", output="sos")
    return sosfilt(sos, y)


def rms_normalize(y: np.ndarray, target_dbfs: float = DEFAULT_DBFS
                  ) -> tuple[np.ndarray, float]:
    """Normalize audio to a target RMS level (dBFS)."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        logger.warning("Signal is silent — skipping RMS normalization.")
        return y, 0.0
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    y = y * (10 ** (gain_db / 20))
    return y, gain_db


def check_clipping(y: np.ndarray, input_path: str) -> np.ndarray:
    """Warn if signal clips after normalization, apply hard limiter."""
    peak = np.max(np.abs(y))
    if peak > 1.0:
        logger.warning(
            "%s | Peak %.4f exceeds 0 dBFS after normalization. "
            "Consider lowering target_dbfs. Applying hard limiter.",
            input_path, peak,
        )
        y = np.clip(y, -1.0, 1.0)
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

    Pipeline: Load → mono → resample → DC offset → trim → duration guard
              → high-pass → RMS normalize → clip guard → save WAV
    """
    try:
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load '{input_path}': {exc}") from exc

    if y.size == 0:
        raise ValueError(f"'{input_path}' loaded as an empty array.")

    y = y - np.mean(y)

    if trim_silence:
        y, trim_indices = librosa.effects.trim(y, top_db=top_db)
        logger.info("%s | Trimmed %.2fs → %.2fs", input_path,
                    trim_indices[0] / target_sr, trim_indices[1] / target_sr)

    duration = len(y) / target_sr
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"'{input_path}' is only {duration:.3f}s after trimming "
            f"(minimum {MIN_DURATION_SEC}s). Skipping."
        )

    y = highpass_filter(y, target_sr, cutoff=hp_cutoff, order=hp_order)
    y, gain_db = rms_normalize(y, target_dbfs=target_dbfs)
    logger.info("%s | Applied gain: %+.1f dB", input_path, gain_db)
    y = check_clipping(y, input_path)

    sf.write(output_path, y, target_sr, subtype="PCM_16")
    logger.info("%s | Saved → %s (%.2fs, %d Hz)", input_path, output_path, duration, target_sr)

    return y, target_sr


# ── in-memory preprocessing ────────────────────────────────────────────────────

def preprocess_in_memory(
    audio_path: str,
    sr: int = DEFAULT_SR,
) -> tuple[np.ndarray, int]:
    """
    Run the full preprocessing chain on one audio file (in-memory, no save).
    Load → DC offset → trim → duration guard → highpass → RMS norm → clip guard → VAD.
    """
    from src.preprocessing.vad import extract_speech_only

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = y - np.mean(y)
    y, _ = librosa.effects.trim(y, top_db=DEFAULT_TRIM_DB)

    if len(y) / sr < MIN_DURATION_SEC:
        raise ValueError(f"Audio too short after trimming: {audio_path}")

    y = highpass_filter(y, sr)
    y, _ = rms_normalize(y)
    y = check_clipping(y, audio_path)
    y, _ = extract_speech_only(y, sr)

    return y, sr


# ── batch helper ───────────────────────────────────────────────────────────────

def batch_standardize(
    file_pairs: list[tuple[str, str]],
    **kwargs,
) -> dict[str, str]:
    """Process multiple files. Failures are caught without stopping the batch."""
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

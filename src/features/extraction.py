import logging
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

from configs.config import FEATURE_DIM, USE_EMBEDDINGS, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────────────
N_MFCC        = 13
N_FFT         = 2048
HOP_LENGTH    = 512
PITCH_FLOOR   = 75.0
PITCH_CEILING = 500.0
MAX_FORMANTS  = 5
FORMANT_CEIL  = 5500.0

EPS = 1e-8  # numerical stability


# ── individual feature extractors ─────────────────────────────────────────────────

def _mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCCs + delta + delta-delta, summarised as (mean, std). Returns 78 features."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = []
    for coeff_set in (mfcc, delta, delta2):
        features.append(np.mean(coeff_set, axis=1))
        features.append(np.std(coeff_set, axis=1))
    return np.concatenate(features)


def _spectral_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Spectral centroid, bandwidth, rolloff — mean + std each. Returns 6 features."""
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)

    features = []
    for feat in (centroid, bandwidth, rolloff):
        features.extend([np.mean(feat), np.std(feat)])
    return np.array(features)


def _zcr_features(y: np.ndarray) -> np.ndarray:
    """Zero-crossing rate — mean + std. Returns 2 features."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return np.array([np.mean(zcr), np.std(zcr)])


def _energy_features(y: np.ndarray) -> np.ndarray:
    """RMS energy — mean + std. Returns 2 features."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return np.array([np.mean(rms), np.std(rms)])


def _pitch_all_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Pitch statistics + contour shape. Returns 7 features."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    if len(voiced) < 2:
        stats = np.zeros(3)
    else:
        stats = np.array([np.mean(voiced), np.std(voiced),
                          np.max(voiced) - np.min(voiced)])

    if len(voiced) < 4:
        contour = np.zeros(4)
    else:
        f0_norm = (voiced - np.mean(voiced)) / (np.std(voiced) + EPS)
        t = np.arange(len(f0_norm))
        slope     = np.polyfit(t, f0_norm, 1)[0]
        curvature = np.polyfit(t, f0_norm, 2)[0]
        diffs = np.diff(f0_norm)
        rising_ratio   = np.sum(diffs > 0) / len(diffs)
        pitch_velocity = np.mean(np.abs(diffs))
        contour = np.array([slope, curvature, rising_ratio, pitch_velocity])

    return np.concatenate([stats, contour])


def _jitter_shimmer_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Voice quality micro-variation features via Praat. Returns 4 features."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)",
                             PITCH_FLOOR, PITCH_CEILING)
        local_jitter = call(point_process, "Get jitter (local)",
                            0.0, 0.0, 0.0001, 0.02, 1.3)
        rap_jitter   = call(point_process, "Get jitter (rap)",
                            0.0, 0.0, 0.0001, 0.02, 1.3)
        local_shimmer = call([snd, point_process], "Get shimmer (local)",
                             0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        apq3_shimmer  = call([snd, point_process], "Get shimmer (apq3)",
                             0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        vals = [local_jitter, rap_jitter, local_shimmer, apq3_shimmer]
        vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
        return np.array(vals)
    except Exception:
        return np.zeros(4)


def _formant_all_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Formant statistics (F1-F3): mean, std, range. Returns 9 features."""
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    formant = call(snd, "To Formant (burg)", 0.0, MAX_FORMANTS,
                   FORMANT_CEIL, 0.025, 50.0)

    duration = snd.get_total_duration()
    n_steps  = max(1, int(duration / 0.01))
    times    = np.linspace(0, duration, n_steps, endpoint=False)

    means, stds, ranges = [], [], []
    for fi in range(1, 4):
        vals = []
        for t in times:
            v = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
            if not np.isnan(v) and v > 0:
                vals.append(v)
        means.append(np.mean(vals) if vals else 0.0)
        if len(vals) >= 2:
            stds.append(np.std(vals))
            ranges.append(np.max(vals) - np.min(vals))
        else:
            stds.append(0.0)
            ranges.append(0.0)

    return np.array(means + stds + ranges)


def _speaking_rate_feature(y: np.ndarray, sr: int) -> np.ndarray:
    """Speaking rate from onset detection. Returns 1 feature."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onsets    = librosa.onset.onset_detect(onset_envelope=onset_env,
                                           sr=sr, hop_length=HOP_LENGTH)
    duration  = len(y) / sr
    rate      = len(onsets) / duration if duration > 0 else 0.0
    return np.array([rate])


# ── public API ──────────────────────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract a full feature vector from a single audio segment. Returns ~109 dims."""
    parts = [
        _mfcc_features(y, sr),
        _spectral_features(y, sr),
        _zcr_features(y),
        _energy_features(y),
        _pitch_all_features(y, sr),
        _jitter_shimmer_features(y, sr),
        _formant_all_features(y, sr),
        _speaking_rate_feature(y, sr),
    ]
    vec = np.concatenate(parts)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def extract_features_batch(
    segments: list[np.ndarray],
    sr: int,
    label: str = "",
) -> np.ndarray:
    """Extract features from multiple segments. Returns (n_segments, n_features)."""
    if not segments:
        raise ValueError("No segments provided for feature extraction.")

    features = []
    for i, seg in enumerate(segments):
        try:
            features.append(extract_features(seg, sr))
        except Exception as exc:
            logger.error("%s seg %d | Feature extraction failed: %s",
                         label or "batch", i, exc)
            features.append(np.zeros_like(features[0]) if features
                            else np.zeros(FEATURE_DIM))

    matrix = np.vstack(features)
    logger.info("%s | Extracted features: %s", label or "batch", matrix.shape)
    return matrix


# ── segment weighting ───────────────────────────────────────────────────────────────

def compute_segment_weights(
    segments: list[np.ndarray],
    sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-segment importance weights based on voicing and energy.

    For each segment i:
        w_i = voiced_ratio_i * log(1 + rms_energy_i)
        w_i = clip(w_i, 0.05, 0.95)
        w_i = w_i / sum(w)

    Parameters
    ----------
    segments : list of float32 audio arrays (each one bucket)
    sr       : sample rate

    Returns
    -------
    weights       : (n_segments,) normalised importance weights
    voiced_ratios : (n_segments,) fraction of voiced frames per segment
    rms_energies  : (n_segments,) RMS energy per segment
    """
    import webrtcvad

    n = len(segments)
    voiced_ratios = np.zeros(n, dtype=np.float64)
    rms_energies  = np.zeros(n, dtype=np.float64)

    vad = webrtcvad.Vad(2)  # aggressiveness = 2
    frame_dur_ms = 30
    frame_size = int(sr * frame_dur_ms / 1000)

    for i, seg in enumerate(segments):
        # RMS energy
        rms_energies[i] = np.sqrt(np.mean(seg ** 2) + EPS)

        # Voiced ratio via WebRTC VAD
        seg_int16 = np.clip(seg, -1.0, 1.0)
        seg_int16 = (seg_int16 * 32767).astype(np.int16)

        # Pad to full frames
        remainder = len(seg_int16) % frame_size
        if remainder:
            seg_int16 = np.concatenate([seg_int16, np.zeros(frame_size - remainder, dtype=np.int16)])

        n_frames = 0
        n_voiced = 0
        for start in range(0, len(seg_int16), frame_size):
            frame = seg_int16[start : start + frame_size]
            n_frames += 1
            if vad.is_speech(frame.tobytes(), sr):
                n_voiced += 1

        voiced_ratios[i] = n_voiced / max(n_frames, 1)

    # w_i = voiced_ratio_i * log(1 + rms_energy_i)
    raw_weights = voiced_ratios * np.log1p(rms_energies)

    # Clip to avoid dominance
    raw_weights = np.clip(raw_weights, 0.05, 0.95)

    # Normalise
    weight_sum = np.sum(raw_weights)
    if weight_sum < EPS:
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = raw_weights / weight_sum

    return weights, voiced_ratios, rms_energies


# ── pooling ─────────────────────────────────────────────────────────────────────────

def pool_segment_features(
    segment_matrix: np.ndarray,
    weights: np.ndarray | None = None,
    voiced_ratios: np.ndarray | None = None,
    rms_energies: np.ndarray | None = None,
    mode: str = "auto",
) -> np.ndarray:
    """
    Pool per-segment features into a single file-level feature vector.

    Parameters
    ----------
    segment_matrix : (n_segments, D) feature matrix
    weights        : (n_segments,) normalised importance weights
    voiced_ratios  : (n_segments,) fraction of voiced frames per segment
    rms_energies   : (n_segments,) RMS energy per segment
    mode           : "simple" → mean+std+meta (2×D+4)
                     "full"   → mean+std+weighted_mean+weighted_std+meta (4×D+4)
                     "auto"   → read from configs.config.POOLING_MODE

    Returns
    -------
    pooled : (2*D + 4,) or (4*D + 4,) file-level feature vector
    """
    from configs.config import POOLING_MODE

    if mode == "auto":
        mode = POOLING_MODE

    if segment_matrix.ndim != 2 or segment_matrix.shape[0] < 1:
        raise ValueError(f"Expected 2-D matrix with >= 1 row, got {segment_matrix.shape}")

    n_seg, D = segment_matrix.shape

    # ── standard pooling ──────────────────────────────────────────────────────────────
    feat_mean = np.mean(segment_matrix, axis=0)
    feat_std  = np.std(segment_matrix, axis=0)

    # ── meta features ─────────────────────────────────────────────────────────────────
    meta_num_segments   = float(n_seg)
    meta_voiced_mean    = float(np.mean(voiced_ratios)) if voiced_ratios is not None else 0.0
    meta_voiced_std     = float(np.std(voiced_ratios))  if voiced_ratios is not None and len(voiced_ratios) > 1 else 0.0
    meta_rms_mean       = float(np.mean(rms_energies))  if rms_energies is not None else 0.0

    meta = np.array([meta_num_segments, meta_voiced_mean, meta_voiced_std, meta_rms_mean])

    if mode == "simple":
        pooled = np.concatenate([feat_mean, feat_std, meta])
        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug(
            "Pooled %d segments × %d features → %d (2×%d + 4 meta) [simple]",
            n_seg, D, len(pooled), D,
        )
        return pooled

    # ── weighted pooling (full mode) ──────────────────────────────────────────────────
    if weights is not None and len(weights) == n_seg:
        w = weights[:, np.newaxis]  # (n_seg, 1)
        w_mean = np.sum(w * segment_matrix, axis=0)
        w_std  = np.sqrt(np.sum(w * (segment_matrix - w_mean) ** 2, axis=0) + EPS)
    else:
        logger.warning("No weights provided — using uniform weighting for weighted pool.")
        w_mean = feat_mean.copy()
        w_std  = feat_std.copy()

    # ── concatenate ─────────────────────────────────────────────────────────────────────
    pooled = np.concatenate([feat_mean, feat_std, w_mean, w_std, meta])
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(
        "Pooled %d segments × %d features → %d (4×%d + 4 meta) [full]",
        n_seg, D, len(pooled), D,
    )
    return pooled


def extract_combined_batch(
    segments: list[np.ndarray],
    sr: int,
    label: str = "",
) -> np.ndarray:
    """
    Extract combined per-segment features: handcrafted (109) + wav2vec2 (768).
    Returns (n_segments, 877) when embeddings enabled, else (n_segments, 109).
    """
    hc_matrix = extract_features_batch(segments, sr, label=label)

    if not USE_EMBEDDINGS:
        return hc_matrix

    from src.features.embeddings import extract_embeddings_batch

    emb_matrix = extract_embeddings_batch(segments, sr, label=label)
    combined = np.hstack([hc_matrix, emb_matrix])
    logger.info("%s | Combined features: %s (handcrafted %d + embedding %d)",
                label or "batch", combined.shape,
                hc_matrix.shape[1], emb_matrix.shape[1])
    return combined

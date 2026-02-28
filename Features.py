import logging
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
N_MFCC        = 13
N_FFT         = 2048
HOP_LENGTH    = 512
PITCH_FLOOR   = 75.0    # Hz — Praat default for adult speech
PITCH_CEILING = 500.0   # Hz — covers high-pitched speakers
MAX_FORMANTS  = 5
FORMANT_CEIL  = 5500.0  # Hz — standard for broad-population analysis


# ── individual feature extractors ─────────────────────────────────────────────

def _mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    MFCCs + delta + delta-delta, summarised as (mean, std) over time.
    Returns 78 features: 13 × 3 coefficients × 2 statistics.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = []
    for coeff_set in (mfcc, delta, delta2):
        features.append(np.mean(coeff_set, axis=1))
        features.append(np.std(coeff_set, axis=1))
    return np.concatenate(features)                       # (78,)


def _spectral_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral centroid, bandwidth, rolloff — mean + std each.
    Returns 6 features.
    """
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)

    features = []
    for feat in (centroid, bandwidth, rolloff):
        features.extend([np.mean(feat), np.std(feat)])
    return np.array(features)                             # (6,)


def _zcr_features(y: np.ndarray) -> np.ndarray:
    """Zero-crossing rate — mean + std.  Returns 2 features."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return np.array([np.mean(zcr), np.std(zcr)])          # (2,)


def _energy_features(y: np.ndarray) -> np.ndarray:
    """RMS energy — mean + std.  Returns 2 features."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return np.array([np.mean(rms), np.std(rms)])           # (2,)


def _pitch_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    F0 statistics via Praat's autocorrelation method.
    Returns 3 features: mean, std, range of voiced F0.
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    if len(voiced) < 2:
        return np.zeros(3)

    return np.array([np.mean(voiced), np.std(voiced),
                     np.max(voiced) - np.min(voiced)])     # (3,)


def _pitch_contour_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Pitch contour SHAPE features — captures intonation patterns.
    Native speakers tend to have more predictable, wider-ranging contours;
    non-native speakers often have flatter or more erratic pitch movement.
    Returns 4 features: slope, curvature, rising_ratio, pitch_velocity_mean.
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]

    if len(voiced) < 4:
        return np.zeros(4)

    # normalise to remove speaker-level pitch differences
    f0_norm = (voiced - np.mean(voiced)) / (np.std(voiced) + 1e-8)

    # linear slope — overall intonation direction
    t = np.arange(len(f0_norm))
    slope = np.polyfit(t, f0_norm, 1)[0]

    # curvature — second-order polynomial coefficient
    curvature = np.polyfit(t, f0_norm, 2)[0]

    # fraction of frames where pitch is rising
    diffs = np.diff(f0_norm)
    rising_ratio = np.sum(diffs > 0) / len(diffs)

    # mean absolute pitch velocity (how fast pitch changes)
    pitch_velocity = np.mean(np.abs(diffs))

    return np.array([slope, curvature, rising_ratio, pitch_velocity])  # (4,)


def _jitter_shimmer_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Voice quality micro-variation features via Praat.
    Jitter = cycle-to-cycle F0 perturbation (pitch stability).
    Shimmer = cycle-to-cycle amplitude perturbation (loudness stability).
    Non-native speakers may show different patterns due to articulatory effort.
    Returns 4 features: local_jitter, rap_jitter, local_shimmer, apq3_shimmer.
    """
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

        # Praat returns undefined as nan for segments without periodicity
        vals = [local_jitter, rap_jitter, local_shimmer, apq3_shimmer]
        vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
        return np.array(vals)                              # (4,)

    except Exception:
        return np.zeros(4)


def _formant_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Mean F1, F2, F3 via Praat's Burg method.
    Returns 3 features.  Non-native speakers often differ in vowel space
    (F1–F2 mapping), making formants a strong accent cue.
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    formant = call(snd, "To Formant (burg)", 0.0, MAX_FORMANTS,
                   FORMANT_CEIL, 0.025, 50.0)

    duration = snd.get_total_duration()
    n_steps  = max(1, int(duration / 0.01))   # measure every 10 ms
    times    = np.linspace(0, duration, n_steps, endpoint=False)

    f_means = []
    for fi in range(1, 4):   # F1, F2, F3
        vals = []
        for t in times:
            v = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
            if not np.isnan(v) and v > 0:
                vals.append(v)
        f_means.append(np.mean(vals) if vals else 0.0)

    return np.array(f_means)                               # (3,)


def _formant_transition_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Formant DYNAMICS — std of F1, F2, F3 trajectories over time.
    Native speakers typically have more dynamic vowel transitions
    (larger F1/F2 movement within utterances), while non-native speakers
    may produce more static, centralized vowels.
    Returns 6 features: std(F1), std(F2), std(F3), range(F1), range(F2), range(F3).
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    formant = call(snd, "To Formant (burg)", 0.0, MAX_FORMANTS,
                   FORMANT_CEIL, 0.025, 50.0)

    duration = snd.get_total_duration()
    n_steps  = max(1, int(duration / 0.01))
    times    = np.linspace(0, duration, n_steps, endpoint=False)

    result = []
    for fi in range(1, 4):   # F1, F2, F3
        vals = []
        for t in times:
            v = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
            if not np.isnan(v) and v > 0:
                vals.append(v)
        if len(vals) >= 2:
            result.extend([np.std(vals), np.max(vals) - np.min(vals)])
        else:
            result.extend([0.0, 0.0])

    return np.array(result)                                # (6,)


def _speaking_rate_feature(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Rough speaking rate estimate from spectral-flux onset detection.
    Returns 1 feature: onsets per second (proxy for syllabic rate).
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onsets    = librosa.onset.onset_detect(onset_envelope=onset_env,
                                           sr=sr, hop_length=HOP_LENGTH)
    duration  = len(y) / sr
    rate      = len(onsets) / duration if duration > 0 else 0.0
    return np.array([rate])                                # (1,)


# ── public API ────────────────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a full feature vector from a single audio segment.

    Parameters
    ----------
    y  : float32 audio segment (pre-processed / standardized)
    sr : sample rate

    Returns
    -------
    1-D feature vector (≈109 dimensions)
    """
    parts = [
        _mfcc_features(y, sr),                #  78
        _spectral_features(y, sr),            #   6
        _zcr_features(y),                     #   2
        _energy_features(y),                  #   2
        _pitch_features(y, sr),               #   3
        _pitch_contour_features(y, sr),       #   4  (slope, curvature, rising %, velocity)
        _jitter_shimmer_features(y, sr),      #   4  (voice quality micro-variations)
        _formant_features(y, sr),             #   3
        _formant_transition_features(y, sr),  #   6  (F1/F2/F3 dynamics)
        _speaking_rate_feature(y, sr),        #   1
    ]                                         # ─────
    vec = np.concatenate(parts)               # ≈ 109
    # replace any NaN / Inf with 0 to keep downstream classifiers happy
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def extract_features_batch(
    segments: list[np.ndarray],
    sr: int,
    label: str = "",
) -> np.ndarray:
    """
    Extract features from multiple segments.

    Parameters
    ----------
    segments : list of float32 arrays (e.g. output of segment_into_buckets)
    sr       : sample rate (assumed constant across segments)
    label    : optional identifier for logging

    Returns
    -------
    2-D array of shape (n_segments, n_features)
    """
    if not segments:
        raise ValueError("No segments provided for feature extraction.")

    features = []
    for i, seg in enumerate(segments):
        try:
            features.append(extract_features(seg, sr))
        except Exception as exc:
            logger.error("%s seg %d | Feature extraction failed: %s",
                         label or "batch", i, exc)
            # append zeros so indexing stays consistent
            features.append(np.zeros_like(features[0]) if features
                            else np.zeros(109))

    matrix = np.vstack(features)
    logger.info("%s | Extracted features: %s", label or "batch", matrix.shape)
    return matrix


def pool_segment_features(segment_matrix: np.ndarray) -> np.ndarray:
    """
    Pool per-segment features into a single file-level super-vector.

    Given N segments × 109 features, compute statistics across segments
    to produce ONE vector that represents the whole file.

    Statistics computed per feature:
      - mean  : central tendency
      - std   : variability across the file

    Parameters
    ----------
    segment_matrix : (n_segments, n_features) from extract_features_batch

    Returns
    -------
    1-D array of shape (n_features × 2,) = (218,)
    """
    if segment_matrix.ndim != 2 or segment_matrix.shape[0] < 1:
        raise ValueError(f"Expected 2-D matrix with >= 1 row, got {segment_matrix.shape}")

    # If only one segment, std is 0 — that's fine and correct
    feat_mean = np.mean(segment_matrix, axis=0)
    feat_std  = np.std(segment_matrix, axis=0)

    pooled = np.concatenate([feat_mean, feat_std])
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug("Pooled %d segments → %d features", segment_matrix.shape[0], len(pooled))
    return pooled


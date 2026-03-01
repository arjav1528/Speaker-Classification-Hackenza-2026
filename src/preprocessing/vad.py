import logging
import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
VALID_SAMPLE_RATES   = {8000, 16000, 32000}
VALID_FRAME_DURATIONS = {10, 20, 30}          # ms, WebRTC VAD requirement
SEGMENT_GAP_MS       = 75                     # silence gap inserted between speech segments
                                               # preserves prosodic boundaries for accent work


# ── core VAD ──────────────────────────────────────────────────────────────────

def run_vad(
    y: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """
    Run WebRTC Voice Activity Detection on a float32 audio signal.

    Parameters
    ----------
    y                : float32 numpy array, range [-1, 1]
    sr               : sample rate — must be 8000, 16000, or 32000
    aggressiveness   : 0 (permissive) → 3 (strict). Use 2 for accent classification;
                       drop to 1 if soft fricatives or non-native prosody are being cut.
    frame_duration_ms: 10, 20, or 30ms. 30ms is the most reliable for VAD.

    Returns
    -------
    frames       : list of (start_sample, end_sample) for every frame processed
    speech_flags : parallel bool list — True = WebRTC classified frame as speech
    """
    if sr not in VALID_SAMPLE_RATES:
        raise ValueError(
            f"Sample rate {sr} Hz not supported by WebRTC VAD. "
            f"Must be one of {VALID_SAMPLE_RATES}."
        )
    if frame_duration_ms not in VALID_FRAME_DURATIONS:
        raise ValueError(
            f"frame_duration_ms must be one of {VALID_FRAME_DURATIONS}, got {frame_duration_ms}."
        )
    if not (0 <= aggressiveness <= 3):
        raise ValueError(f"aggressiveness must be 0–3, got {aggressiveness}.")

    vad        = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_duration_ms / 1000)   # samples per frame

    # float32 → int16 in one vectorised call (avoids per-frame Python overhead)
    y_int16 = np.clip(y, -1.0, 1.0)
    y_int16 = (y_int16 * 32767).astype(np.int16)

    # zero-pad so the final partial frame is not silently discarded
    remainder = len(y_int16) % frame_size
    if remainder:
        padding  = frame_size - remainder
        y_int16  = np.concatenate([y_int16, np.zeros(padding, dtype=np.int16)])

    frames, speech_flags = [], []
    for start in range(0, len(y_int16), frame_size):
        end   = start + frame_size
        frame = y_int16[start:end]
        is_speech = vad.is_speech(frame.tobytes(), sr)
        speech_flags.append(is_speech)
        frames.append((start, end))

    return frames, speech_flags


# ── segment extraction ────────────────────────────────────────────────────────

def extract_speech_only(
    y: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    segment_gap_ms: int = SEGMENT_GAP_MS,
    min_speech_ms: int = 100,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Remove non-speech regions and return speech segments joined by short silence gaps.

    Segments are NOT directly concatenated — a small silence gap is preserved between
    them to maintain prosodic continuity, which matters for accent classification
    (rhythm, intonation, and inter-word juncture cues live at segment boundaries).

    Parameters
    ----------
    y                : float32 audio signal
    sr               : sample rate
    aggressiveness   : passed to run_vad()
    frame_duration_ms: passed to run_vad()
    segment_gap_ms   : ms of silence to insert between speech segments (default 75ms)
    min_speech_ms    : discard speech segments shorter than this (likely noise bursts)

    Returns
    -------
    speech_only     : processed audio array
    speech_segments : list of (start_sample, end_sample) in the *original* signal
    """
    frames, speech_flags = run_vad(y, sr, aggressiveness, frame_duration_ms)

    # ── group consecutive speech frames into segments ─────────────────────────
    speech_segments: list[tuple[int, int]] = []
    in_speech  = False
    seg_start  = 0
    min_samples = int(sr * min_speech_ms / 1000)

    for i, (start, end) in enumerate(frames):
        if speech_flags[i] and not in_speech:
            in_speech = True
            seg_start = start
        elif not speech_flags[i] and in_speech:
            in_speech = False
            if (start - seg_start) >= min_samples:
                speech_segments.append((seg_start, start))

    # handle file ending mid-speech
    if in_speech:
        seg_end = frames[-1][1]
        if (seg_end - seg_start) >= min_samples:
            speech_segments.append((seg_start, min(seg_end, len(y))))

    # ── fallback: no speech found ─────────────────────────────────────────────
    if not speech_segments:
        logger.warning(
            "VAD found no speech segments (aggressiveness=%d). "
            "Returning full signal. Consider lowering aggressiveness.",
            aggressiveness,
        )
        return y, [(0, len(y))]

    logger.info(
        "VAD: %d speech segment(s) found, total %.2fs of %.2fs kept.",
        len(speech_segments),
        sum(e - s for s, e in speech_segments) / sr,
        len(y) / sr,
    )

    # ── join with silence gap instead of hard concatenation ──────────────────
    gap        = np.zeros(int(sr * segment_gap_ms / 1000), dtype=y.dtype)
    parts      = []
    for i, (start, end) in enumerate(speech_segments):
        parts.append(y[start : min(end, len(y))])
        if i < len(speech_segments) - 1:
            parts.append(gap)

    speech_only = np.concatenate(parts)
    return speech_only, speech_segments

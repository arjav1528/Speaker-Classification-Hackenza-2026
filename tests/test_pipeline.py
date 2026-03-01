"""
Unit tests for the Speaker Classification pipeline.

Run with:  python -m pytest tests/ -v
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from configs.config import TARGET_SR

# ── fixtures ──────────────────────────────────────────────────────────────────

SR = TARGET_SR  # standard sample rate used throughout the project


@pytest.fixture
def sine_wave():
    """3 seconds of 440 Hz sine wave at 16 kHz — simulates a clean tonal signal."""
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def speech_like_signal():
    """5 seconds of band-limited noise — rough proxy for speech energy distribution."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(int(SR * 5)).astype(np.float32)
    # crude band-pass via rolling mean (simulates low + mid freq content)
    kernel = np.ones(16) / 16
    y = np.convolve(y, kernel, mode="same").astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-8)  # normalize to [-1, 1]
    return y


@pytest.fixture
def short_signal():
    """0.05 seconds — too short to process."""
    return np.zeros(int(SR * 0.05), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Segment tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSegmentIntoBuckets:

    def test_basic_segmentation(self, sine_wave):
        from src.preprocessing.segment import segment_into_buckets

        buckets, timestamps = segment_into_buckets(sine_wave, SR, bucket_duration_sec=1.0, overlap_sec=0.0)
        assert len(buckets) == 3
        assert all(len(b) == SR for b in buckets)
        assert len(timestamps) == 3

    def test_padding_last_bucket(self):
        from src.preprocessing.segment import segment_into_buckets

        # 2.5 seconds → should produce 2 full buckets + 1 padded
        y = np.ones(int(SR * 2.5), dtype=np.float32)
        buckets, _ = segment_into_buckets(y, SR, bucket_duration_sec=1.0, overlap_sec=0.0, pad_last=True)
        assert len(buckets) == 3                         # 1.0 + 1.0 + 0.5(padded)
        assert len(buckets[-1]) == SR                    # padded to full bucket size

    def test_no_padding_last_bucket(self):
        from src.preprocessing.segment import segment_into_buckets

        y = np.ones(int(SR * 2.5), dtype=np.float32)
        buckets, _ = segment_into_buckets(y, SR, bucket_duration_sec=1.0, overlap_sec=0.0, pad_last=False)
        assert len(buckets) == 3
        assert len(buckets[-1]) == int(SR * 0.5)         # natural length

    def test_min_bucket_ratio_drops_short_tail(self):
        from src.preprocessing.segment import segment_into_buckets

        # 3.2 seconds with 3s buckets, min_ratio=0.5 → tail is 0.2s = 6.7% → dropped
        y = np.ones(int(SR * 3.2), dtype=np.float32)
        buckets, _ = segment_into_buckets(y, SR, bucket_duration_sec=3.0, min_bucket_ratio=0.5)
        assert len(buckets) == 1

    def test_too_short_raises(self, short_signal):
        from src.preprocessing.segment import segment_into_buckets

        with pytest.raises(ValueError, match="too short"):
            segment_into_buckets(short_signal, SR, bucket_duration_sec=1.0, overlap_sec=0.0)

    def test_overlap_produces_more_buckets(self, sine_wave):
        from src.preprocessing.segment import segment_into_buckets

        # 3s signal, 1s buckets, 0.5s overlap → should produce more than 3
        buckets_no_overlap, _ = segment_into_buckets(sine_wave, SR, bucket_duration_sec=1.0, overlap_sec=0.0)
        buckets_overlap, _ = segment_into_buckets(
            sine_wave, SR, bucket_duration_sec=1.0, overlap_sec=0.5
        )
        assert len(buckets_overlap) > len(buckets_no_overlap)

    def test_invalid_params(self, sine_wave):
        from src.preprocessing.segment import segment_into_buckets

        with pytest.raises(ValueError):
            segment_into_buckets(sine_wave, SR, bucket_duration_sec=-1.0)
        with pytest.raises(ValueError):
            segment_into_buckets(sine_wave, SR, overlap_sec=5.0, bucket_duration_sec=3.0)

    def test_timestamps_are_monotonic(self, sine_wave):
        from src.preprocessing.segment import segment_into_buckets

        _, timestamps = segment_into_buckets(sine_wave, SR, bucket_duration_sec=1.0, overlap_sec=0.0)
        starts = [t[0] for t in timestamps]
        assert starts == sorted(starts)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:

    def test_feature_vector_shape(self, speech_like_signal):
        from src.features.extraction import extract_features

        vec = extract_features(speech_like_signal, SR)
        assert vec.ndim == 1
        assert len(vec) == 109  # documented feature dimensionality

    def test_no_nans_or_infs(self, speech_like_signal):
        from src.features.extraction import extract_features

        vec = extract_features(speech_like_signal, SR)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_batch_extraction(self, speech_like_signal):
        from src.features.extraction import extract_features_batch
        from src.preprocessing.segment import segment_into_buckets

        buckets, _ = segment_into_buckets(speech_like_signal, SR, bucket_duration_sec=1.0, overlap_sec=0.0)
        matrix = extract_features_batch(buckets, SR, label="test")
        assert matrix.shape == (len(buckets), 109)

    def test_batch_empty_raises(self):
        from src.features.extraction import extract_features_batch

        with pytest.raises(ValueError, match="No segments"):
            extract_features_batch([], SR)


class TestPoolSegmentFeatures:

    def test_pool_simple_shape(self):
        from src.features.extraction import pool_segment_features

        # simulate 5 segments × 109 features
        matrix = np.random.randn(5, 109).astype(np.float32)
        pooled = pool_segment_features(matrix, mode="simple")
        assert pooled.shape == (2 * 109 + 4,)  # mean + std + 4 meta = 222

    def test_pool_full_shape(self):
        from src.features.extraction import pool_segment_features

        matrix = np.random.randn(5, 109).astype(np.float32)
        weights = np.ones(5) / 5
        voiced = np.ones(5) * 0.8
        rms = np.ones(5) * 0.1
        pooled = pool_segment_features(matrix, weights=weights,
                                        voiced_ratios=voiced,
                                        rms_energies=rms, mode="full")
        assert pooled.shape == (4 * 109 + 4,)  # mean+std+wmean+wstd+meta = 440

    def test_pool_single_segment(self):
        from src.features.extraction import pool_segment_features

        matrix = np.random.randn(1, 109).astype(np.float32)
        pooled = pool_segment_features(matrix, mode="simple")
        assert pooled.shape == (2 * 109 + 4,)
        # std should be all zeros for a single segment
        assert np.allclose(pooled[109:218], 0.0)

    def test_pool_no_nans(self):
        from src.features.extraction import pool_segment_features

        matrix = np.random.randn(3, 109).astype(np.float32)
        matrix[0, 5] = np.nan  # inject a NaN
        pooled = pool_segment_features(matrix, mode="simple")
        assert not np.any(np.isnan(pooled))

    def test_pool_invalid_input(self):
        from src.features.extraction import pool_segment_features

        with pytest.raises(ValueError):
            pool_segment_features(np.array([1.0, 2.0, 3.0]))  # 1-D


# ══════════════════════════════════════════════════════════════════════════════
# Audio preprocessing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStdaudio:

    def test_highpass_filter(self, sine_wave):
        from src.preprocessing.audio import highpass_filter

        filtered = highpass_filter(sine_wave, SR)
        assert filtered.shape == sine_wave.shape
        assert not np.allclose(filtered, 0.0)  # 440 Hz passes through

    def test_highpass_removes_dc(self):
        from src.preprocessing.audio import highpass_filter

        # pure DC signal should be almost entirely removed
        dc = np.ones(SR, dtype=np.float32) * 0.5
        filtered = highpass_filter(dc, SR)
        # After transient settles, DC should be heavily attenuated
        assert np.max(np.abs(filtered[SR // 2:])) < 0.1

    def test_rms_normalize(self, sine_wave):
        from src.preprocessing.audio import rms_normalize

        normalized, gain = rms_normalize(sine_wave)
        rms = np.sqrt(np.mean(normalized ** 2))
        target_rms = 10 ** (-23.0 / 20)  # default -23 dBFS
        assert abs(rms - target_rms) < 0.01

    def test_rms_normalize_silent(self):
        from src.preprocessing.audio import rms_normalize

        silent = np.zeros(SR, dtype=np.float32)
        result, gain = rms_normalize(silent)
        assert gain == 0.0
        assert np.allclose(result, 0.0)

    def test_check_clipping(self):
        from src.preprocessing.audio import check_clipping

        loud = np.array([1.5, -1.3, 0.8], dtype=np.float32)
        clipped = check_clipping(loud, "test.wav")
        assert np.max(np.abs(clipped)) <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# VAD tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVAD:

    def test_run_vad_returns_frames(self, speech_like_signal):
        from src.preprocessing.vad import run_vad

        frames, flags = run_vad(speech_like_signal, SR)
        assert len(frames) == len(flags)
        assert len(frames) > 0
        assert all(isinstance(f, bool) for f in flags)

    def test_run_vad_invalid_sr(self, speech_like_signal):
        from src.preprocessing.vad import run_vad

        with pytest.raises(ValueError, match="not supported"):
            run_vad(speech_like_signal, 44100)

    def test_extract_speech_returns_array(self, speech_like_signal):
        from src.preprocessing.vad import extract_speech_only

        speech, segments = extract_speech_only(speech_like_signal, SR)
        assert isinstance(speech, np.ndarray)
        assert speech.dtype == speech_like_signal.dtype
        assert len(speech) > 0
        assert isinstance(segments, list)

    def test_vad_silent_input_fallback(self):
        from src.preprocessing.vad import extract_speech_only

        silent = np.zeros(SR * 2, dtype=np.float32)
        speech, segments = extract_speech_only(silent, SR)
        # should fall back to full signal
        assert len(speech) == len(silent)


# ══════════════════════════════════════════════════════════════════════════════
# Classifier tests
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifier:

    def test_build_track_a_pipeline(self):
        from src.models.classifier import build_track_a_pipeline

        pipe = build_track_a_pipeline(n_features=100)
        assert pipe is not None
        assert len(pipe.steps) == 3  # select, scale, clf

    def test_build_track_b_pipeline(self):
        from src.models.classifier import build_track_b_pipeline

        pipe = build_track_b_pipeline(n_components=20)
        assert pipe is not None
        assert len(pipe.steps) == 3  # scale, pca, clf

    def test_track_a_fit_predict(self):
        from src.models.classifier import build_track_a_pipeline

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 100))
        y = np.array([0] * 30 + [1] * 20)

        pipe = build_track_a_pipeline(n_features=100)
        pipe.fit(X, y)

        preds = pipe.predict(X[:5])
        assert len(preds) == 5

        proba = pipe.predict_proba(X[:5])
        assert proba.shape == (5, 2)

    def test_track_b_fit_predict(self):
        from src.models.classifier import build_track_b_pipeline

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 200))
        y = np.array([0] * 30 + [1] * 20)

        pipe = build_track_b_pipeline(n_components=20)
        pipe.fit(X, y)

        preds = pipe.predict(X[:5])
        assert len(preds) == 5

    def test_optimize_threshold(self):
        from src.models.classifier import optimize_threshold

        rng = np.random.default_rng(42)
        y_true = np.array([0] * 20 + [1] * 10)
        y_proba = rng.random(30)
        y_proba[20:] += 0.3  # make positive class slightly higher

        thresh = optimize_threshold(y_true, y_proba)
        assert 0.0 < thresh < 1.0

    def test_save_and_load_model(self, tmp_path):
        from src.models.classifier import (
            build_track_a_pipeline, save_model, load_model
        )
        from sklearn.preprocessing import LabelEncoder

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 100))
        y = np.array([0] * 30 + [1] * 20)

        pipe = build_track_a_pipeline(n_features=100)
        pipe.fit(X, y)

        le = LabelEncoder()
        le.fit(["Native", "Non-Native"])

        model_dir = str(tmp_path / "model")
        save_model(pipe, le, model_dir, threshold=0.35)

        pipe_loaded, le_loaded, meta = load_model(model_dir)
        assert list(le_loaded.classes_) == ["Native", "Non-Native"]
        assert meta.get("threshold") == 0.35

        preds = pipe_loaded.predict(X[:5])
        assert len(preds) == 5

    def test_predict_file_compat(self, tmp_path):
        """predict_file should accept both 1-D and 2-D input."""
        from src.inference.predictor import predict_file
        from src.models.classifier import build_track_a_pipeline, save_model
        from sklearn.preprocessing import LabelEncoder

        n_features = 222
        rng = np.random.default_rng(42)
        X_mock = rng.standard_normal((50, n_features))
        y_mock = np.array([0] * 30 + [1] * 20)

        pipe = build_track_a_pipeline(n_features=n_features)
        pipe.fit(X_mock, y_mock)

        le = LabelEncoder()
        le.fit(["Native", "Non-Native"])

        model_dir = str(tmp_path / "model")
        save_model(pipe, le, model_dir, threshold=0.4)

        # test 1-D input
        x = rng.standard_normal(n_features)
        result = predict_file(x, model_dir=model_dir)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0

        # test 2-D input
        x2d = x.reshape(1, -1)
        result2 = predict_file(x2d, model_dir=model_dir)
        assert result2["label"] == result["label"]

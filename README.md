# Speaker Classification — Native vs Non-Native

A machine learning pipeline that classifies speakers as **native** or **non-native** English speakers from audio recordings. Built for **Hackenza 2026**.

## Pipeline Overview

```
Raw Audio → Standardize → VAD → Segment → Features → Pool → Classify
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Standardize** | `Stdaudio.py` | Mono conversion, resampling to 16 kHz, DC removal, silence trimming, high-pass filter, RMS normalization |
| **VAD** | `VAD.py` | WebRTC voice activity detection — strips non-speech while preserving prosodic gaps |
| **Segment** | `Segment.py` | Splits audio into fixed-length buckets (default 3s) with optional overlap |
| **Features** | `Features.py` | Extracts 109-dim feature vector per segment (see below) |
| **Pool** | `Features.py` | Aggregates segment features → 218-dim file-level vector (mean + std) |
| **Classify** | `Classifier.py` | SVM + Random Forest + GBM with GridSearchCV, soft-voting ensemble |

## Features Extracted (109 dimensions per segment)

| Feature Group | Count | Source |
|---|---|---|
| MFCCs + Δ + ΔΔ (mean, std) | 78 | librosa |
| Spectral centroid, bandwidth, rolloff | 6 | librosa |
| Zero-crossing rate | 2 | librosa |
| RMS energy | 2 | librosa |
| Pitch (F0) statistics | 3 | Praat (parselmouth) |
| Pitch contour shape (slope, curvature, rising ratio, velocity) | 4 | Praat |
| Jitter & Shimmer (voice quality) | 4 | Praat |
| Formant means (F1, F2, F3) | 3 | Praat |
| Formant dynamics (std, range of F1–F3) | 6 | Praat |
| Speaking rate (onsets/sec) | 1 | librosa |

## Setup

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Project structure

```
├── main.py           # CLI entry point
├── Stdaudio.py       # Audio standardization
├── VAD.py            # Voice activity detection
├── Segment.py        # Audio segmentation
├── Features.py       # Feature extraction
├── Classifier.py     # Model training & prediction
├── tests/            # Unit tests
│   └── test_pipeline.py
└── pyproject.toml    # Project config & dependencies
```

## Usage

### 1. Preprocess raw audio

Standardize all audio files (resample, normalize, filter):

```bash
python main.py preprocess --data-dir data/raw --output-dir data/processed
```

**Expected data layout:**
```
data/raw/
├── native/
│   ├── speaker01.wav
│   ├── speaker02.mp3
│   └── ...
└── non_native/
    ├── speaker10.wav
    ├── speaker11.flac
    └── ...
```

Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`

### 2. Train the classifier

```bash
python main.py train --data-dir data/processed --model-dir models
```

This runs the full pipeline: **VAD → Segment → Extract Features → Pool → Train (SVM + RF + GBM + Ensemble)** with 5-fold stratified cross-validation and hyperparameter tuning via GridSearchCV.

**Output artifacts** (saved to `models/`):
- `pipeline.joblib` — best individual model pipeline
- `ensemble.joblib` — soft-voting ensemble
- `label_encoder.joblib` — label mapping
- `metrics.json` — cross-validated accuracy, confusion matrices, per-class reports
- `feature_info.npz` — mutual information feature scores

### 3. Predict on a new audio file

```bash
python main.py predict --audio-path path/to/audio.wav --model-dir models
```

**Example output:**
```
══════════════════════════════════════════════════
  File:          test_speaker.wav
  Segments used: 4
  Prediction:    native
  Confidence:    87.3%
  Probabilities: {'native': 0.873, 'non_native': 0.127}
══════════════════════════════════════════════════
```

## Anti-Overfitting Measures

- **Feature selection** via mutual information (218 → 80 features) inside each CV fold
- **StandardScaler** inside the pipeline — no leakage
- **Stratified k-fold CV** — honest evaluation, no same-speaker contamination
- **GridSearchCV** — automated hyperparameter tuning
- **Balanced class weights** — handles class imbalance
- **Soft-voting ensemble** — zero extra learnable parameters (unlike stacking)
- **Per-file pooling** — one label per file, segments from the same file never split across train/test

## Running Tests

```bash
python -m pytest tests/ -v
```

## Dependencies

| Package | Purpose |
|---|---|
| librosa | Audio loading, MFCCs, spectral features |
| praat-parselmouth | Pitch, formants, jitter/shimmer via Praat |
| scikit-learn | SVM, Random Forest, GBM, pipelines, CV |
| webrtcvad-wheels | WebRTC voice activity detection |
| scipy | High-pass filtering (Butterworth) |
| soundfile | WAV file I/O |
| numpy | Numerical operations |
| joblib | Model serialization |

## License

MIT
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

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd Speaker-Classification-Hackenza-2026
uv sync                    # or: pip install -e .

# 2. Preprocess raw audio files
python main.py preprocess --data-dir share-audio/data/raw --output-dir share-audio/data/wav

# 3. Train the classifier
python main.py train --data-dir share-audio/data/wav --model-dir models

# 4. Predict on a new audio file
python main.py predict --audio-path path/to/audio.wav --model-dir models

# 5. Run the full accuracy evaluation
python test_accuracy.py
```

## Features Extracted (109 dimensions per segment)

| Feature Group | Count | Source |
|---|---|---|
| MFCCs + Δ + ΔΔ (mean, std) | 78 | librosa |
| Spectral centroid, bandwidth, rolloff | 6 | librosa |
| Zero-crossing rate | 2 | librosa |
| RMS energy | 2 | librosa |
| Pitch (F0) statistics + contour shape | 7 | Praat (parselmouth) |
| Jitter & Shimmer (voice quality) | 4 | Praat |
| Formant statistics (F1–F3 means, stds, ranges) | 9 | Praat |
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
├── main.py              # CLI entry point (preprocess / train / predict)
├── test_accuracy.py     # Full accuracy evaluation pipeline
├── config.py            # Shared constants (TARGET_SR, RANDOM_SEED, etc.)
├── Stdaudio.py          # Audio standardization + in-memory preprocessing
├── VAD.py               # Voice activity detection
├── Segment.py           # Audio segmentation
├── Features.py          # Feature extraction (MFCCs, pitch, formants, etc.)
├── Classifier.py        # Model training & prediction
├── tests/               # Unit tests
│   └── test_pipeline.py
├── results/             # Accuracy reports, plots, predictions (auto-generated)
├── share-audio/         # Training & test data
│   ├── wav_converted.csv
│   ├── data/wav/        # Preprocessed training audio
│   └── test/            # Unlabeled test audio
└── pyproject.toml       # Project config & dependencies
```

## Usage

### 1. Preprocess raw audio

Standardize all audio files (resample to 16 kHz, normalize, filter):

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

### 4. Run the full accuracy evaluation

```bash
python test_accuracy.py
```

This evaluates **8 classifiers** (SVM RBF, SVM Linear, Random Forest, Gradient Boosting, KNN, MLP, Logistic Regression, AdaBoost) plus a soft-voting ensemble using:

- **80/20 stratified holdout split** — primary accuracy estimate
- **5-fold stratified cross-validation** — more robust generalization estimate
- **Comprehensive metrics** — see below

**Options:**
```bash
python test_accuracy.py --skip-plots           # Skip generating charts
python test_accuracy.py --models svm rf mlp    # Only evaluate specific models
```

**Output** (saved to `results/`):
- `results_summary.csv` — model comparison table
- `metrics.json` — all metrics as JSON
- `test_predictions.csv` — predictions for unlabeled test set
- `model_comparison.png` — bar chart comparing all models
- `cm_*.png` — confusion matrices per model
- `best_model_radar.png` — radar chart of best model's metrics

### Metrics & Class Bias Detection

The evaluation reports these metrics to prevent a model from gaming accuracy by always predicting the majority class:

| Metric | What it catches |
|---|---|
| **Accuracy** | Overall correctness |
| **Balanced Accuracy** | Average per-class recall — immune to class imbalance tricks |
| **Per-class Precision / Recall / F1** | Performance on each class separately |
| **Min Class Recall** | Worst class recall — 0 means a class is completely ignored |
| **ROC AUC** | Discriminative ability; 0.5 = random guessing |
| **Cohen's Kappa** | Agreement beyond chance |
| **Matthews Correlation Coefficient** | Best single metric for imbalanced datasets |
| **EER** | Equal error rate (speaker verification standard) |
| **Log Loss** | Penalizes confident wrong predictions |

**Automatic warnings:**
- **⚠ CLASS BIAS** — fires if any class gets <5% of predictions but >10% of the data
- **⚠ Accuracy vs Balanced Accuracy gap** — fires if gap >0.05 (imbalance exploitation)
- **⚠ flag** on per-class rows where recall < 0.30

Best model is selected by **balanced accuracy** (not raw accuracy), so a model must perform well on *both* classes.

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

## Configuration

All shared constants live in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `TARGET_SR` | 16000 | Sample rate (Hz) |
| `BUCKET_DURATION` | 3.0 | Segment length (seconds) |
| `MIN_BUCKET_RATIO` | 0.5 | Keep final segment if ≥ 50% of full bucket |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `N_FOLDS` | 5 | Stratified CV folds |
| `MAX_FEATURES` | 80 | Top-K features (mutual information) |
| `FEATURE_DIM` | 109 | Per-segment feature vector length |

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
| pandas | Data loading & result tables |
| seaborn | Visualization (confusion matrices) |
| matplotlib | Charts & plots |

## License

MIT
# Speaker Classification — Native vs Non-Native

A machine learning pipeline that classifies speakers as **Native** or **Non-Native** English speakers from audio recordings. Built for **Hackenza 2026**.

---

## Overview

The pipeline uses a **dual-track baseline** approach:

- **Track A**: Handcrafted acoustic features (109-dim per segment) → Mutual Information feature selection → LightGBM
- **Track B**: wav2vec2 embeddings (768-dim) → PCA → Logistic Regression
- **Ensemble**: Soft-vote combining Track A + Track B

Both tracks are evaluated via **Repeated Stratified 5-Fold CV** (3 repeats = 15 folds). The best model (by balanced accuracy) is retrained on full data and saved for inference.

---

## Pipeline Flow

```
Raw Audio → Preprocess (VAD, standardize) → Segment (3s windows, 1s overlap) → Features → Pool → Classify
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Preprocess** | `src/preprocessing/audio.py` | Mono, 16 kHz, DC removal, high-pass filter, RMS normalization |
| **VAD** | `src/preprocessing/vad.py` | WebRTC voice activity detection — strips non-speech |
| **Segment** | `src/preprocessing/segment.py` | 3s buckets with 1s overlap (2s hop) |
| **Features** | `src/features/extraction.py` | 109 handcrafted + optional 768 wav2vec2 per segment |
| **Pool** | `src/features/extraction.py` | mean + std + meta → file-level vector |
| **Classify** | `src/models/classifier.py` | Track A (LightGBM), Track B (LogReg), or Ensemble |

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
# Clone the repo
git clone <repo-url> && cd Speaker-Classification-Hackenza-2026

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Train

```bash
python scripts/train.py
```

### Predict

```bash
python scripts/predict.py                                    # Test set
python scripts/predict.py path/to/audio.wav                  # Single file
python scripts/predict.py --model-dir outputs/models/ *.wav  # Multiple files
```

---

## Project Structure

```
Speaker-Classification-Hackenza-2026/
├── src/
│   ├── pipeline.py              # TrainPipeline, FeatureExtractor
│   ├── preprocessing/
│   │   ├── audio.py             # preprocess_in_memory, standardize_audio
│   │   ├── vad.py               # WebRTC VAD
│   │   ├── segment.py           # segment_into_buckets
│   │   └── augment.py           # minority-class augmentation
│   ├── features/
│   │   ├── extraction.py        # 109-dim handcrafted features
│   │   └── embeddings.py        # wav2vec2 embeddings
│   ├── models/
│   │   └── classifier.py        # Track A/B pipelines, ensemble, save/load
│   ├── inference/
│   │   └── predictor.py         # Predictor class for inference
│   └── utils/
│       ├── io.py                # load_train_csv, resolve_audio_path
│       └── metrics.py           # evaluation metrics
├── scripts/
│   ├── train.py                 # Train dual-track, select best, save model
│   ├── predict.py               # Run predictions
│   ├── extract_features.py      # Extract and cache features to .npz
│   └── run.py                   # Full pipeline entrypoint
├── configs/
│   └── config.py                # TARGET_SR, BUCKET_DURATION, etc.
├── share-data/
│   ├── wav_converted.csv        # Training: dp_id, audio_url, nativity_status
│   └── test/
│       ├── test_wav_converted.csv
│       └── wav/                 # Test audio files
├── outputs/
│   ├── models/                  # Saved pipeline, label encoder, metrics
│   ├── results/                 # Predictions, metrics.json, results_summary.csv
│   └── features/                # Cached train_features.npz
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

---

## Data Format

### Training CSV (`share-data/wav_converted.csv`)

| Column | Description |
|--------|-------------|
| `dp_id` | Unique sample ID |
| `audio_url` | Relative path (e.g. `data/wav/716.wav`) |
| `nativity_status` | `Native` or `Non-Native` |
| `language` | e.g. `Arabic_QA`, `Arabic_SA` |

Audio paths are resolved from `share-data/`; files may live under `Native/` or `Non-Native/` subdirs.

### Test CSV (`share-data/test/test_wav_converted.csv`)

Same structure with `nativity_status` as `-` (unlabeled). Predictions are written to `outputs/results/test_predictions.csv`.

---

## Usage

### Train

```bash
python scripts/train.py
python scripts/train.py --no-augment           # Disable minority augmentation
python scripts/train.py --no-embeddings        # Track A only (no wav2vec2)
python scripts/train.py --track-a-only         # Only LightGBM
python scripts/train.py --track-b-only         # Only wav2vec2 + LogReg
python scripts/train.py --from-cache outputs/features/train_features.npz
```

**Output** (saved to `outputs/`):

- `models/`: `pipeline.joblib`, `label_encoder.joblib`, `metrics.json`, etc.
- `results/`: `results_summary.csv`, `metrics.json`

### Predict

```bash
python scripts/predict.py                                    # Predict test set
python scripts/predict.py path/to/audio.wav                  # Single file
python scripts/predict.py --model-dir outputs/models/ *.wav  # Multiple files
python scripts/predict.py --no-embeddings                    # Match model trained without embeddings
python scripts/predict.py --output my_predictions.csv        # Custom output path
```

### Extract Features (Cache)

```bash
python scripts/extract_features.py
python scripts/extract_features.py --no-embeddings --output outputs/features/train_features.npz
```

Cached `.npz` can be passed to `train.py --from-cache` to skip re-extraction.

### Full Pipeline

```bash
python scripts/run.py
python scripts/run.py --preprocess-only
```

---

## Features

### Handcrafted (109-dim per segment)

| Feature Group | Count | Source |
|---------------|-------|--------|
| MFCCs + Δ + ΔΔ (mean, std) | 78 | librosa |
| Spectral centroid, bandwidth, rolloff | 6 | librosa |
| Zero-crossing rate | 2 | librosa |
| RMS energy | 2 | librosa |
| Pitch (F0) statistics + contour shape | 7 | Praat (parselmouth) |
| Jitter & Shimmer | 4 | Praat |
| Formant stats (F1–F3) | 9 | Praat |
| Speaking rate | 1 | librosa |

### Pooling (per file)

- **Simple**: mean + std + 4 meta → `2×109 + 4 = 222` dims (handcrafted only)
- **Full**: mean + std + weighted_mean + weighted_std + 4 meta (with segment weights)

With wav2vec2: per-segment dim = `109 + 768 = 877`, pooled to file-level accordingly.

---

## Configuration

Key constants in `configs/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TARGET_SR` | 16000 | Sample rate (Hz) |
| `BUCKET_DURATION` | 3.0 | Segment length (seconds) |
| `OVERLAP_SEC` | 1.0 | Overlap between segments |
| `N_AUGMENTS` | 3 | Augmented copies per minority sample |
| `FEATURE_SELECT_K` | 100 | Top-K features (MI SelectKBest) |
| `PCA_COMPONENTS` | 50 | PCA dims for Track B |
| `USE_EMBEDDINGS` | True | Include wav2vec2 embeddings |
| `EMBEDDING_MODEL` | `facebook/wav2vec2-base` | Pretrained model |

---

## Evaluation

- **Repeated Stratified 5-Fold CV** (3 repeats)
- **Metrics**: Balanced Accuracy, F1 Macro, MCC, ROC AUC
- **Threshold optimization** per fold for better recall/precision trade-off
- **Group-aware splits** so augmented copies do not leak into validation

Best model is selected by **balanced accuracy**.

---

## Anti-Overfitting

- Mutual information feature selection (Track A)
- PCA for embeddings (Track B)
- Stratified CV with group awareness (augmented samples)
- Balanced class weights
- Soft-voting ensemble (no extra learnable parameters)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| librosa | MFCCs, spectral features |
| praat-parselmouth | Pitch, formants, jitter/shimmer |
| scikit-learn | Pipelines, CV, PCA |
| lightgbm | Track A classifier |
| webrtcvad / webrtcvad-wheels | Voice activity detection |
| torch, transformers | wav2vec2 embeddings |
| audiomentations | Minority-class augmentation |
| soundfile, scipy, numpy | Audio I/O and filtering |

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## License

MIT

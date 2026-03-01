"""
Shared constants for the Speaker Classification pipeline.

Single source of truth — avoids duplicating values across modules.
"""

# ── audio ────────────────────────────────────────────────────────────────────────────
TARGET_SR          = 16_000     # Hz — sample rate used everywhere
AUDIO_EXTENSIONS   = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# ── segmentation ─────────────────────────────────────────────────────────────────────
BUCKET_DURATION    = 3.0        # seconds per segment
OVERLAP_SEC        = 1.0        # 1s overlap → 2s hop between windows
MIN_BUCKET_RATIO   = 0.5        # keep final segment if >= 50 % of a full bucket

# ── classifier ───────────────────────────────────────────────────────────────────────
RANDOM_SEED        = 42
N_FOLDS            = 5          # stratified CV folds
CV_REPEATS         = 3          # repeated stratified CV for stability
FEATURE_SELECT_K   = 60         # top-K dims after MI SelectKBest (Track A)
PCA_COMPONENTS     = 50         # PCA components for embedding track (Track B)
MAX_FEATURES       = 60         # alias for FEATURE_SELECT_K (backward compat)
PCA_VARIANCE       = 0.95       # retain 95% variance (auto-selects n_components)

# ── pooling ──────────────────────────────────────────────────────────────────────────
POOLING_MODE       = "simple"   # "simple" = mean+std (2×D+4), "full" = 4×D+4

# ── augmentation ─────────────────────────────────────────────────────────────────────
N_AUGMENTS         = 3          # augmented copies per minority-class sample

# ── features ─────────────────────────────────────────────────────────────────────────
FEATURE_DIM        = 109        # per-segment handcrafted feature vector length

# ── embeddings ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "facebook/wav2vec2-base"   # pretrained model name
EMBEDDING_DIM      = 768        # wav2vec2-base hidden size
USE_EMBEDDINGS     = True       # concatenate embeddings with handcrafted features

# ── Track A: Handcrafted + LightGBM ─────────────────────────────────────────────────
TRACK_A_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_samples": 5,
    "scale_pos_weight": 2.6,     # 110 Native / 43 Non-Native ≈ 2.56
    "random_state": RANDOM_SEED,
    "verbosity": -1,
    "n_jobs": -1,
}

# ── Track B: wav2vec2 + Logistic Regression ──────────────────────────────────────────
TRACK_B_PARAMS = {
    "C": 0.1,
    "class_weight": "balanced",
    "solver": "saga",
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
}

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
MAX_FEATURES       = 100        # top-K dims after reduction (MI SelectKBest)
PCA_VARIANCE       = 0.95       # retain 95% variance (auto-selects n_components)

# ── features ─────────────────────────────────────────────────────────────────────────
FEATURE_DIM        = 109        # per-segment handcrafted feature vector length

# ── embeddings ───────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "facebook/wav2vec2-base"   # pretrained model name
EMBEDDING_DIM      = 768        # wav2vec2-base hidden size
USE_EMBEDDINGS     = True       # concatenate embeddings with handcrafted features

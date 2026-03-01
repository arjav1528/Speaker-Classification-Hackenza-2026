"""
Speaker Classification — Classifier Module

TODO: Build classification pipeline from scratch.

Suggested architecture:
    - Define get_models() → dict of named sklearn Pipelines
    - train_and_evaluate(X, y) → train models, run CV, return metrics
    - save_model(model, path) / load_model(path)

Available features per file (from src.features):
    - 109-dim handcrafted (MFCCs, spectral, pitch, formants, jitter/shimmer, rate)
    - 768-dim wav2vec2 embeddings (optional)
    - Pooled via mean+std → 218 or 1754 dims per file

Dataset characteristics:
    - 153 training samples (110 Native / 43 Non-Native)
    - Binary classification: Native vs Non-Native
    - Severe class imbalance (~72/28 split)
    - Consider: class_weight='balanced', SMOTE, threshold tuning
"""

import logging

logger = logging.getLogger(__name__)

__all__ = []  # populate as you add functions

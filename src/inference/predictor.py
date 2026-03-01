"""
Speaker Classification — Predictor Module

TODO: Build prediction pipeline from scratch.

Suggested API:
    - load_model(model_dir) → loaded pipeline + label encoder
    - predict_file(audio_path, model) → label, confidence
    - predict_batch(audio_paths, model) → DataFrame of predictions
"""

import logging

logger = logging.getLogger(__name__)

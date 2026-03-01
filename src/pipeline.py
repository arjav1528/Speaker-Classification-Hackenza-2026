"""
Speaker Classification — Pipeline Orchestrator

Ties together preprocessing → feature extraction → model training/prediction.

TODO: Implement the full pipeline once classifier is built.

Usage (planned):
    from src.pipeline import TrainPipeline, PredictPipeline

    # Training
    pipe = TrainPipeline(config)
    pipe.run()

    # Prediction
    pipe = PredictPipeline(model_dir, audio_path)
    result = pipe.run()
"""

import logging

logger = logging.getLogger(__name__)

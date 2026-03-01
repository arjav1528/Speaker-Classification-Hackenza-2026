#!/usr/bin/env python3
"""
Run predictions on audio files.

Usage:
    python scripts/predict.py path/to/audio.wav
    python scripts/predict.py --model-dir outputs/models/ path/to/audio.wav
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    TODO: Implement prediction pipeline.

    Suggested flow:
        1. Load saved model   → src.inference.predictor (YOUR CODE)
        2. Preprocess audio   → src.preprocessing.audio.preprocess_in_memory()
        3. Segment            → src.preprocessing.segment.segment_into_buckets()
        4. Extract features   → src.features.extraction.extract_combined_batch()
        5. Pool               → src.features.extraction.pool_segment_features()
        6. Predict            → model.predict() / predict_proba()
        7. Print results
    """
    print("=" * 60)
    print("  Speaker Classification — Prediction")
    print("=" * 60)
    print()
    print("  TODO: Implement predictor in src/inference/predictor.py")
    print("  Then wire it up here.")
    print()


if __name__ == "__main__":
    main()

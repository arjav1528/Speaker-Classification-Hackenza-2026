#!/usr/bin/env python3
"""
Train the Speaker Classification model.

Usage:
    python scripts/train.py
    python scripts/train.py --skip-plots
"""

import sys
import os

# Add project root to path so `from src...` and `from configs...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    TODO: Implement training pipeline.

    Suggested flow:
        1. Load training CSV        → src.utils.io.load_train_csv()
        2. Resolve audio paths       → src.utils.io.resolve_audio_path()
        3. Preprocess each file      → src.preprocessing.audio.preprocess_in_memory()
        4. Segment into buckets      → src.preprocessing.segment.segment_into_buckets()
        5. Extract features          → src.features.extraction.extract_combined_batch()
        6. Pool to file-level        → src.features.extraction.pool_segment_features()
        7. Train classifier          → src.models.classifier (YOUR CODE)
        8. Evaluate with CV          → src.utils.metrics.compute_metrics()
        9. Save model                → outputs/models/
        10. Save results             → outputs/results/
    """
    print("=" * 60)
    print("  Speaker Classification — Training")
    print("=" * 60)
    print()
    print("  TODO: Implement classifier in src/models/classifier.py")
    print("  Then wire it up here.")
    print()


if __name__ == "__main__":
    main()

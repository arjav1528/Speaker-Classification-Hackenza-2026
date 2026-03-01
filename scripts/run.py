#!/usr/bin/env python3
"""
Run the full pipeline: preprocess → train → evaluate.

Usage:
    python scripts/run.py
    python scripts/run.py --preprocess-only
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Speaker Classification — Full Pipeline")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run preprocessing, skip training")
    args = parser.parse_args()

    print("=" * 60)
    print("  Speaker Classification — Full Pipeline")
    print("=" * 60)
    print()

    if args.preprocess_only:
        print("  TODO: Run batch preprocessing")
    else:
        print("  TODO: Run full pipeline (preprocess + train + evaluate)")
    print()


if __name__ == "__main__":
    main()

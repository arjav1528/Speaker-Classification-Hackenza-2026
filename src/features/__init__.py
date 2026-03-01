"""Feature extraction: handcrafted audio features + wav2vec2 embeddings."""

from .extraction import (
    extract_features,
    extract_features_batch,
    extract_combined_batch,
    pool_segment_features,
    compute_segment_weights,
)
from .embeddings import extract_embedding, extract_embeddings_batch

__all__ = [
    "extract_features",
    "extract_features_batch",
    "extract_combined_batch",
    "pool_segment_features",
    "compute_segment_weights",
    "extract_embedding",
    "extract_embeddings_batch",
]

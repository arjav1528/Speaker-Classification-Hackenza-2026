"""
Wav2Vec2 segment-level embedding extraction.

Uses facebook/wav2vec2-base (frozen, no fine-tuning) to produce a 768-dim
embedding per audio segment.  The model is lazy-loaded on first call.
"""

import logging

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from configs.config import TARGET_SR, EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ── lazy-loaded globals ────────────────────────────────────────────────────────
_processor: Wav2Vec2Processor | None = None
_model: Wav2Vec2Model | None = None
_device: torch.device | None = None


def _ensure_model_loaded() -> None:
    """Download & cache the wav2vec2 model on first call."""
    global _processor, _model, _device

    if _model is not None:
        return

    logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)

    _processor = Wav2Vec2Processor.from_pretrained(EMBEDDING_MODEL)
    _model = Wav2Vec2Model.from_pretrained(EMBEDDING_MODEL)

    _model.eval()
    for param in _model.parameters():
        param.requires_grad = False

    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model = _model.to(_device)
    logger.info("Embedding model loaded on %s", _device)


def extract_embedding(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Extract a wav2vec2 embedding from a single audio segment. Returns (768,)."""
    _ensure_model_loaded()

    if sr != 16_000:
        raise ValueError(f"wav2vec2 requires 16 kHz audio, got {sr} Hz")

    y_f32 = y.astype(np.float32)
    inputs = _processor(y_f32, sampling_rate=sr, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(_device)

    with torch.no_grad():
        outputs = _model(input_values)
        hidden = outputs.last_hidden_state

    embedding = hidden.squeeze(0).mean(dim=0).cpu().numpy()

    assert embedding.shape == (EMBEDDING_DIM,), (
        f"Expected ({EMBEDDING_DIM},), got {embedding.shape}"
    )
    return embedding


def extract_embeddings_batch(
    segments: list[np.ndarray],
    sr: int = TARGET_SR,
    label: str = "",
) -> np.ndarray:
    """Extract wav2vec2 embeddings for a list of segments. Returns (n_segments, 768)."""
    _ensure_model_loaded()

    if not segments:
        raise ValueError("No segments provided for embedding extraction.")

    embeddings = []
    for i, seg in enumerate(segments):
        try:
            emb = extract_embedding(seg, sr)
            embeddings.append(emb)
        except Exception as exc:
            logger.error("%s seg %d | Embedding extraction failed: %s",
                         label or "batch", i, exc)
            embeddings.append(np.zeros(EMBEDDING_DIM))

    matrix = np.vstack(embeddings)
    logger.info("%s | Extracted embeddings: %s", label or "batch", matrix.shape)
    return matrix

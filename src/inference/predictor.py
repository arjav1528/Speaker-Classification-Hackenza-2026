"""
Speaker Classification — Predictor Module

Wraps model loading, feature extraction, and prediction into a clean API.

Usage:
    pred = Predictor("outputs/models")
    result = pred.predict_file("path/to/audio.wav")
    results_df = pred.predict_batch(["file1.wav", "file2.wav"])
"""

import logging
import os

import numpy as np
import pandas as pd

from src.models.classifier import load_model
from src.pipeline import FeatureExtractor

logger = logging.getLogger(__name__)


class Predictor:
    """
    End-to-end predictor: audio file → label + confidence.
    """

    def __init__(
        self,
        model_dir: str = "outputs/models",
        use_embeddings: bool = True,
        pooling_mode: str = "simple",
    ):
        self.model_dir = model_dir
        self.pipeline, self.label_encoder, self.meta = load_model(model_dir)
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.feature_extractor = FeatureExtractor(
            use_embeddings=use_embeddings,
            pooling_mode=pooling_mode,
        )
        logger.info(
            "Predictor loaded from %s (threshold=%.3f, classes=%s)",
            model_dir, self.threshold, list(self.label_encoder.classes_),
        )

    def predict_file(self, audio_path: str) -> dict:
        """
        Predict the nativity label for a single audio file.

        Returns
        -------
        dict with keys: label, confidence, probabilities, threshold
        """
        features = self.feature_extractor.extract(audio_path)
        X = features.reshape(1, -1)

        # Probability-based prediction with threshold
        if hasattr(self.pipeline, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba(X)[0]
                # Use threshold on the positive (minority) class — index 1
                if len(proba) == 2:
                    pred_idx = int(proba[1] >= self.threshold)
                else:
                    pred_idx = int(np.argmax(proba))
                label = self.label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(proba[pred_idx])
                proba_dict = {
                    str(cls): float(p)
                    for cls, p in zip(self.label_encoder.classes_, proba)
                }
                return {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": proba_dict,
                    "threshold": self.threshold,
                }
            except Exception:
                pass

        # Fallback: hard prediction
        pred_idx = self.pipeline.predict(X)[0]
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        return {
            "label": label,
            "confidence": 1.0,
            "probabilities": {},
            "threshold": self.threshold,
        }

    def predict_batch(self, audio_paths: list[str]) -> pd.DataFrame:
        """
        Predict labels for multiple audio files.

        Returns a DataFrame with columns: file, label, confidence, probabilities.
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict_file(path)
                result["file"] = os.path.basename(path)
                result["path"] = path
                results.append(result)
            except Exception as exc:
                logger.error("Prediction failed for %s: %s", path, exc)
                results.append({
                    "file": os.path.basename(path),
                    "path": path,
                    "label": "ERROR",
                    "confidence": 0.0,
                    "probabilities": {},
                    "threshold": self.threshold,
                })

        df = pd.DataFrame(results)
        return df


def predict_file(features: np.ndarray, model_dir: str = "outputs/models") -> dict:
    """
    Predict from pre-extracted features (backward compat with tests).

    Parameters
    ----------
    features  : 1-D or 2-D feature vector
    model_dir : path to saved model directory

    Returns
    -------
    dict with label, confidence, probabilities
    """
    pipeline, le, meta = load_model(model_dir)
    threshold = float(meta.get("threshold", 0.5))

    X = features.reshape(1, -1) if features.ndim == 1 else features

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            if len(proba) == 2:
                pred_idx = int(proba[1] >= threshold)
            else:
                pred_idx = int(np.argmax(proba))
            label = le.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    cls: float(p) for cls, p in zip(le.classes_, proba)
                },
            }
        except Exception:
            pass

    pred_idx = pipeline.predict(X)[0]
    label = le.inverse_transform([pred_idx])[0]
    return {"label": label, "confidence": 1.0, "probabilities": {}}

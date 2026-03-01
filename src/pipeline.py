"""
Speaker Classification — Pipeline Orchestrator

Ties together preprocessing → augmentation → feature extraction → pooling.

Provides two pipelines:
    - TrainPipeline : load CSV, preprocess, augment, extract features → (X, y)
    - FeatureExtractor : single-file feature extraction for inference
"""

import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder

from configs.config import (
    TARGET_SR,
    BUCKET_DURATION,
    OVERLAP_SEC,
    N_AUGMENTS,
    POOLING_MODE,
)

logger = logging.getLogger(__name__)


class TrainPipeline:
    """
    End-to-end pipeline from raw audio CSV to feature matrices.

    Usage:
        pipe = TrainPipeline(augment=True, use_embeddings=True)
        X, y, le, file_ids = pipe.run()
    """

    def __init__(
        self,
        augment: bool = True,
        use_embeddings: bool = True,
        n_augments: int = N_AUGMENTS,
        pooling_mode: str = POOLING_MODE,
    ):
        self.augment = augment
        self.use_embeddings = use_embeddings
        self.n_augments = n_augments
        self.pooling_mode = pooling_mode

    def run(self) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list]:
        """
        Execute the full training pipeline.

        Returns
        -------
        X        : (n_samples, n_features) feature matrix
        y        : (n_samples,) integer-encoded labels
        le       : fitted LabelEncoder
        file_ids : list of dp_ids corresponding to each row in X
        """
        from src.utils.io import load_train_csv, resolve_audio_path
        from src.preprocessing.audio import preprocess_in_memory
        from src.preprocessing.augment import augment_minority
        from src.preprocessing.segment import segment_into_buckets
        from src.features.extraction import (
            extract_features_batch,
            extract_combined_batch,
            pool_segment_features,
            compute_segment_weights,
        )

        # Optionally override embedding config
        if not self.use_embeddings:
            import configs.config as cfg
            cfg.USE_EMBEDDINGS = False

        # ── 1. Load CSV ──────────────────────────────────────────────────────
        df = load_train_csv()
        logger.info("=" * 60)
        logger.info("  Training Pipeline — %d files", len(df))
        logger.info("  Augmentation   : %s (n=%d)", self.augment, self.n_augments)
        logger.info("  Embeddings     : %s", self.use_embeddings)
        logger.info("  Pooling mode   : %s", self.pooling_mode)
        logger.info("=" * 60)

        # ── 2. Load & preprocess all audio ───────────────────────────────────
        audios = []
        labels = []
        file_ids = []
        skipped = []

        for idx, row in df.iterrows():
            dp_id = row["dp_id"]
            label = row["nativity_status"]
            audio_rel = row["audio_url"]

            audio_path = resolve_audio_path(audio_rel)
            if audio_path is None:
                skipped.append((dp_id, "file_not_found"))
                continue

            try:
                y_audio, sr = preprocess_in_memory(audio_path, sr=TARGET_SR)
                if len(y_audio) < sr * 0.3:
                    skipped.append((dp_id, "too_short"))
                    continue
                audios.append(y_audio)
                labels.append(label)
                file_ids.append(dp_id)
            except Exception as exc:
                logger.error("[%s] Preprocessing failed: %s", dp_id, exc)
                skipped.append((dp_id, str(exc)))

        logger.info("Loaded %d / %d files (%d skipped)",
                     len(audios), len(df), len(skipped))

        # ── 3. Augment minority class ────────────────────────────────────────
        n_original = len(file_ids)
        parent_indices = list(range(n_original))  # default: each sample is its own group

        if self.augment and self.n_augments > 0:
            audios, labels, parent_indices = augment_minority(
                audios, labels, TARGET_SR, n_augments=self.n_augments
            )
            # file_ids for augmented samples: mark as "aug_<parent_id>"
            n_augmented = len(audios) - n_original
            for i in range(n_augmented):
                parent_fid = file_ids[parent_indices[n_original + i]]
                file_ids.append(f"aug_{parent_fid}_{i}")

        # ── 4. Feature extraction per file ───────────────────────────────────
        X_list = []
        y_list = []
        valid_ids = []

        for i, (audio, label) in enumerate(zip(audios, labels)):
            fid = file_ids[i] if i < len(file_ids) else f"sample_{i}"
            try:
                segments, _ = segment_into_buckets(
                    audio, TARGET_SR,
                    bucket_duration_sec=BUCKET_DURATION,
                    overlap_sec=OVERLAP_SEC,
                    label=str(fid),
                )
                if not segments:
                    continue

                if self.use_embeddings:
                    seg_features = extract_combined_batch(segments, TARGET_SR, label=str(fid))
                else:
                    seg_features = extract_features_batch(segments, TARGET_SR, label=str(fid))

                weights, voiced_ratios, rms_energies = compute_segment_weights(
                    segments, TARGET_SR
                )

                file_vector = pool_segment_features(
                    seg_features,
                    weights=weights,
                    voiced_ratios=voiced_ratios,
                    rms_energies=rms_energies,
                    mode=self.pooling_mode,
                )

                X_list.append(file_vector)
                y_list.append(label)
                valid_ids.append(fid)

            except Exception as exc:
                logger.error("[%s] Feature extraction failed: %s", fid, exc)

        if not X_list:
            raise RuntimeError("No features extracted from any file!")

        X = np.vstack(X_list)
        y_str = np.array(y_list)

        # ── 5. Build groups array (parent index for each valid sample) ───────
        # Maps each sample to its original (non-augmented) parent index.
        # This prevents augmented copies from leaking into validation folds.
        # valid_parent maps valid_ids → parent group index in the *valid* set.
        fid_to_parent_idx = {}  # parent file_id → index in valid set
        groups = np.zeros(len(valid_ids), dtype=int)
        for i, fid in enumerate(valid_ids):
            # Look up the parent from the full file_ids → parent_indices mapping
            full_idx = file_ids.index(fid) if fid in file_ids else i
            parent_orig_idx = parent_indices[full_idx]
            parent_fid = file_ids[parent_orig_idx]
            if parent_fid not in fid_to_parent_idx:
                fid_to_parent_idx[parent_fid] = len(fid_to_parent_idx)
            groups[i] = fid_to_parent_idx[parent_fid]

        # ── 6. Encode labels ─────────────────────────────────────────────────
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        logger.info("Feature matrix: %s | Labels: %s | Groups: %d unique",
                     X.shape, dict(zip(le.classes_, np.bincount(y))),
                     len(fid_to_parent_idx))

        return X, y, le, valid_ids, groups


class FeatureExtractor:
    """
    Single-file feature extraction for inference.

    Usage:
        fe = FeatureExtractor(use_embeddings=True)
        features = fe.extract(audio_path)
    """

    def __init__(self, use_embeddings: bool = True, pooling_mode: str = POOLING_MODE):
        self.use_embeddings = use_embeddings
        self.pooling_mode = pooling_mode

    def extract(self, audio_path: str) -> np.ndarray:
        """Extract pooled features from a single audio file. Returns (n_features,)."""
        from src.preprocessing.audio import preprocess_in_memory
        from src.preprocessing.segment import segment_into_buckets
        from src.features.extraction import (
            extract_features_batch,
            extract_combined_batch,
            pool_segment_features,
            compute_segment_weights,
        )

        if not self.use_embeddings:
            import configs.config as cfg
            cfg.USE_EMBEDDINGS = False

        y_audio, sr = preprocess_in_memory(audio_path, sr=TARGET_SR)

        segments, _ = segment_into_buckets(
            y_audio, sr,
            bucket_duration_sec=BUCKET_DURATION,
            overlap_sec=OVERLAP_SEC,
            label=audio_path,
        )

        if not segments:
            raise ValueError(f"No segments produced from {audio_path}")

        if self.use_embeddings:
            seg_features = extract_combined_batch(segments, sr, label=audio_path)
        else:
            seg_features = extract_features_batch(segments, sr, label=audio_path)

        weights, voiced_ratios, rms_energies = compute_segment_weights(segments, sr)

        return pool_segment_features(
            seg_features,
            weights=weights,
            voiced_ratios=voiced_ratios,
            rms_energies=rms_energies,
            mode=self.pooling_mode,
        )

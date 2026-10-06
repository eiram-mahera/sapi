"""
DeepCell nuclear segmentation model wrapper.

DeepCell (Van Valen Lab, Caltech) provides a pretrained PanopticNet model
for nuclear instance segmentation. Weights are downloaded automatically
to ~/.deepcell/models/ on first use.

Repository: https://github.com/vanvalenlab/deepcell-tf

Install:
    pip install deepcell

Note:
    DeepCell requires TensorFlow. Place in an environment with tensorflow[and-cuda]
    (e.g. stardist_env). Fine-tuning is not supported; inference only.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)


class DeepCellModel(BaseSegmentationModel):
    """DeepCell NuclearSegmentation wrapper."""

    supports_finetune = False

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.model_name = "deepcell"
        self.model_version = config.get("model_version", "1.1")
        self.image_mpp = config.get("image_mpp", None)  # None = no rescaling

    def _ensure_loaded(self):
        if self._model is not None:
            return

        from deepcell.applications import NuclearSegmentation

        # Configure TF to use the correct GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus and self.device != "cpu":
                gpu_idx = int(self.device.replace("cuda:", "")) if ":" in self.device else 0
                if gpu_idx < len(gpus):
                    # set_visible_devices requires a list, not a single device
                    tf.config.set_visible_devices([gpus[gpu_idx]], "GPU")
                    tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
        except Exception as e:
            logger.debug(f"DeepCell GPU config: {e}")

        logger.info(f"Loading DeepCell NuclearSegmentation v{self.model_version} "
                    f"(weights auto-downloaded to ~/.deepcell/)...")
        self._model = NuclearSegmentation.from_version(self.model_version)
        logger.info("DeepCell NuclearSegmentation loaded.")

    @staticmethod
    def _to_hw1(image: np.ndarray) -> np.ndarray:
        """Convert to (H, W, 1) float32 — DeepCell's expected channel format."""
        if image.ndim == 2:
            return image[:, :, np.newaxis].astype(np.float32)
        if image.ndim == 3 and image.shape[2] > 1:
            return image.mean(axis=2, keepdims=True).astype(np.float32)
        return image.astype(np.float32)

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        img_batch = self._to_hw1(image)[np.newaxis]          # (1, H, W, 1)
        labeled = self._model.predict(img_batch, image_mpp=self.image_mpp)
        return labeled[0, :, :, 0].astype(np.uint16)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Batch inference: stack up to batch_size images into one predict() call."""
        self._ensure_loaded()
        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            batch = np.stack([self._to_hw1(img) for img in chunk], axis=0)  # (B, H, W, 1)
            labeled = self._model.predict(batch, image_mpp=self.image_mpp)  # (B, H, W, 1)
            for b in range(len(chunk)):
                results.append(labeled[b, :, :, 0].astype(np.uint16))
        return results

    def get_param_count(self) -> int:
        """Return parameter count of the underlying Keras model."""
        if self._model is None:
            return -1
        try:
            return self._model.model.count_params()
        except Exception:
            return -1

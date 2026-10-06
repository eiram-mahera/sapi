"""
Mesmer (DeepCell) model wrapper.

Mesmer expects a two-channel input [nuclear_marker, cytoplasm_marker].
For single-channel modalities (DAPI, ssDNA, mIF) the nuclear channel is used
and the cytoplasm channel is set to zero.

Install: pip install deepcell

Note: full supervised training and fine-tuning of Mesmer require access to
DeepCell's training pipeline (deepcell.utils.train_utils), which this wrapper
does not use.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)


class MesmerModel(BaseSegmentationModel):
    """DeepCell Mesmer wrapper for nuclear/whole-cell segmentation."""

    model_name = "mesmer"
    supports_finetune = False
    num_channels = 1

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.min_size = int(config.get("min_size", 1))
        self.compartment = config.get("compartment", "nuclear")  # 'nuclear' or 'whole-cell'
        # None = no rescaling (CellBinDB's true pixel size is not stated in the
        # paper; see configs/models/mesmer.yaml for the removed 0.5 assumption).
        self.image_mpp = config.get("image_mpp", None)  # microns per pixel

    def _ensure_loaded(self):
        if self._model is None:
            from deepcell.applications import Mesmer
            import tensorflow as tf
            # Restrict TensorFlow to ONLY use the first visible GPU (set by CUDA_VISIBLE_DEVICES).
            gpus = tf.config.list_physical_devices("GPU")
            if gpus and self.device != "cpu":
                tf.config.set_visible_devices([gpus[0]], "GPU")
                tf.config.experimental.set_memory_growth(gpus[0], True)
            self._model = Mesmer()
            logger.info("Loaded Mesmer (DeepCell) model.")

    def _to_mesmer_input(self, image: np.ndarray) -> np.ndarray:
        """Convert a single/multi-channel image to Mesmer's (1, H, W, 2) float32
        format (channel 0 = nuclear, channel 1 = cytoplasm zeros).  Intensity is
        left raw — Mesmer.predict() normalises internally."""
        if image.ndim == 2:
            nuclear = image.astype(np.float32)
        elif image.ndim == 3 and image.shape[2] == 3:
            nuclear = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            ).astype(np.float32)
        else:
            nuclear = image[:, :, 0].astype(np.float32)

        cytoplasm = np.zeros_like(nuclear)
        combined = np.stack([nuclear, cytoplasm], axis=-1)  # (H, W, 2)
        return combined[np.newaxis]  # (1, H, W, 2)

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        inp = self._to_mesmer_input(image)
        # Dataset-level minimum instance size (configs/datasets/<name>.yaml).
        # Mesmer's own postprocess default is 15, which drops real nuclei clipped by
        # the crop border.  Both compartments are set since self.compartment is
        # config-driven.  The threshold is assumed to follow the skimage
        # convention (`area < threshold` -> drop), matching the benchmark's definition.
        _size_filter = {"small_objects_threshold": self.min_size}
        pred = self._model.predict(
            inp,
            image_mpp=self.image_mpp,
            compartment=self.compartment,
            postprocess_kwargs_nuclear=dict(_size_filter),
            postprocess_kwargs_whole_cell=dict(_size_filter),
        )
        # pred shape: (1, H, W, 1) — deep_watershed output for the configured compartment
        mask = pred[0, :, :, 0].astype(np.uint16)
        return mask

    def get_param_count(self) -> int:
        self._ensure_loaded()
        try:
            return int(self._model.model.count_params())
        except Exception:
            return -1

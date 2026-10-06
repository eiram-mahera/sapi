"""
CellSAM model wrapper.

CellSAM (Marks et al., Nature Methods 2025) is a foundation model for cell
segmentation that combines SAM (ViT-B backbone) with CellFinder (AnchorDETR
object detector). CellFinder automatically generates bounding-box prompts from
ViT image features, which SAM's mask decoder then uses to produce instance masks.

Architecture:
  - SAM ViT-B image encoder (~89 M params) — shared via model_cp in adv_mode
  - SAM mask decoder + prompt encoder (~4 M params)
  - CellFinder AnchorDETR decoder (~20 M params)
  Two copies of the SAM model are stored in adv_mode (model_cp handles both
  embedding and mask prediction); total parameter count ~206 M.

Weights:
  Auto-download to ~/.deepcell/models/cellsam_v1.2/ via the DeepCell API.
  First download requires DEEPCELL_ACCESS_TOKEN env var (token from
  https://users.deepcell.org).  Subsequent runs use the cached weights.
  Alternatively, set CELLSAM_CHECKPOINT to a local .pt file.

Install:
    pip install -e third_party/cellSAM-master/
    # or set CELLSAM_ROOT to the cellSAM-master directory

Notes:
  - Inference-only: the CellSAM source exposes no public fine-tuning API.
  - needs_raw_he_image=False: CellSAM is a generalist, so it receives a 1-channel
    image for every modality (H&E arrives as the Hematoxylin channel).  CellSAM's
    own format_image_shape() then zero-pads it into the last of three channels —
    the channel layout its released weights expect.
  - normalize=True: CellSAM's internal percentile-threshold + CLAHE pipeline
    is applied to every image (RGB or single-channel), so raw input is fine.

Paper:
    Marks et al. "CellSAM: a foundation model for cell segmentation."
    Nature Methods 22, 2585–2593 (2025). https://doi.org/10.1038/s41592-025-02879-w
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)

def _resolve_cellsam_root(config: dict) -> str | None:
    return config.get("cellsam_root") or os.environ.get("CELLSAM_ROOT")


class CellSAMModel(BaseSegmentationModel):
    """CellSAM foundation model wrapper (Marks et al., Nature Methods 2025).

    Inference-only. No fine-tuning API is provided in the CellSAM source.
    """

    supports_finetune = False
    # Generalist model (all six modalities) -> receives a 1-channel image for every
    # modality, so H&E arrives as the stain-separated Hematoxylin channel.  Channel
    # adaptation is left to CellSAM's own reference code: format_image_shape()
    # zero-pads a 1-channel image into the LAST of three channels (R=G=0), which is
    # the format its released weights were trained with.  Do not pre-replicate here.
    needs_raw_he_image = False

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.model_name    = "cellsam"
        self.model_variant = config.get("model_variant", "cellsam_general")
        self.bbox_threshold = float(config.get("bbox_threshold", 0.4))

        # Optional source path (when not installed as a package)
        cellsam_root = _resolve_cellsam_root(config)
        if cellsam_root and str(cellsam_root) not in sys.path:
            sys.path.insert(0, str(cellsam_root))

        # Local checkpoint overrides the auto-download
        ckpt = config.get("checkpoint_path") or os.environ.get("CELLSAM_CHECKPOINT")
        self._checkpoint_path = str(ckpt) if ckpt else None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return

        from cellSAM.model import get_model, get_local_model

        if self._checkpoint_path:
            logger.info(f"Loading CellSAM from local checkpoint: {self._checkpoint_path}")
            model = get_local_model(self._checkpoint_path)
        else:
            logger.info(
                f"Loading CellSAM ({self.model_variant}) — "
                "auto-downloading weights to ~/.deepcell/models/cellsam_v1.2/ "
                "(DEEPCELL_ACCESS_TOKEN required on first download)"
            )
            model = get_model(model=self.model_variant)

        model = model.eval()
        model = model.to(self.device)
        model.bbox_threshold = self.bbox_threshold
        self._model = model
        actual_device = next(model.parameters()).device
        logger.info(f"CellSAM loaded on device: {actual_device}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Segment a single image.

        Args:
            image: raw float32 array, shape (H, W).  CellSAM is a generalist
                   model (needs_raw_he_image=False): every modality, including
                   H&E (arrives as the Hematoxylin channel), is 1-channel.
                   CellSAM normalises internally (percentile + CLAHE) via
                   normalize=True.

        Returns:
            uint32 instance label map, shape (H, W). Background = 0.
        """
        self._ensure_loaded()

        from cellSAM.model import segment_cellular_image

        try:
            # normalize=True: applies CellSAM's percentile-threshold + CLAHE
            # pipeline.  Works on raw float input (both RGB and single-channel —
            # format_image_shape pads 1-ch to 3-ch first).
            mask, _, _ = segment_cellular_image(
                image,
                model=self._model,
                normalize=True,
                postprocess=False,
                bbox_threshold=self.bbox_threshold,
                fast=False,
                device=self.device,
            )
        except (TypeError, AttributeError) as exc:
            # segment_cellular_image has a latent bug: when predict() returns
            # (None, None, None, None) for empty detections, the function tries
            # to call fill_holes_and_remove_small_masks(None), which does
            # `masks.ndim` on None and raises AttributeError (not TypeError).
            # We catch this and return an empty mask rather than crashing.
            logger.warning(
                f"CellSAM inference failed for image shape {image.shape}: {exc}"
            )
            return np.zeros(image.shape[:2], dtype=np.uint32)

        if mask is None or not isinstance(mask, np.ndarray):
            return np.zeros(image.shape[:2], dtype=np.uint32)

        return mask.astype(np.uint32)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Segment a list of images.

        CellSAM has no true batch inference API (fast=True is alpha); images
        are processed sequentially.
        """
        return [self.predict_single(img) for img in images]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_param_count(self) -> int:
        """Total parameter count across all sub-modules (model + model_cp + cellfinder)."""
        if self._model is None:
            return -1
        return sum(p.numel() for p in self._model.parameters())

"""
PathoSAM model wrapper: SAM fine-tuned for nucleus segmentation in histopathology.

PathoSAM (Griebel, Archit & Pape, MIDL 2025) is a vision foundation model built
on top of Segment Anything (SAM) and μSAM.  It was trained on six diverse H&E
histopathology datasets with nucleus instance annotations, covering nuclei from
different tissue types and staining conditions.

Three generalist checkpoint variants are provided:
  vit_b_histopathology  — ViT-B encoder (~123 M total with UNETR decoder)
  vit_l_histopathology  — ViT-L encoder (~337 M total)
  vit_h_histopathology  — ViT-H encoder (~666 M total)

Inference uses the same Automatic Instance Segmentation (AIS) pipeline as μSAM:
SAM image encoder → UNETR decoder (3-channel: foreground / center dist /
boundary dist) → seeded watershed → instance label map.

Fine-tuning uses MicroSAMModel.finetune() verbatim (no override in this class):
the same per-instance centroid-prompt loop with a Dice loss, and the same
post-fine-tuning re-calibration of the AIS thresholds on the support set.
freeze_encoder=True freezes the image encoder (requires_grad=False) for
decoder-only fine-tuning; freeze_encoder=False trains all parameters. Note this
is a plain parameter freeze, not PEFT/LoRA — no LoRA adapter is inserted for
either MicroSAM or PathoSAM in the current implementation.

H&E-only: PathoSAM was trained exclusively on H&E-stained tissue.  Set
compatible_datasets to [HE, 10xGenomics_HE] in the YAML.  needs_raw_he_image=True
so the full raw RGB image is passed (not the extracted hematoxylin channel) —
matching PathoSAM's training distribution.

Requirements:
  micro-sam >= 1.1  (pip install micro-sam)
  segment_anything  (pulled in by micro-sam)

Full PathoSAM pretraining uses the patho_sam library's own training utilities,
which this benchmark does not call: inference and few-shot fine-tuning only
need micro_sam APIs.

Paper: Griebel, Archit & Pape, "Segment Anything for Histopathology", MIDL 2025.
       https://doi.org/10.48550/arXiv.2502.00408
"""

from __future__ import annotations

import logging

from .microsam_model import MicroSAMModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PathoSAM-specific AIS thresholds
# (micro_sam ships no histopathology entry in _model_settings.py, so we set
#  conservative defaults appropriate for tightly-packed nuclei in H&E images)
# ---------------------------------------------------------------------------

_PATHO_AIS_DEFAULTS: dict = {
    "center_distance_threshold": 0.5,
    "boundary_distance_threshold": 0.4,
    "distance_smoothing": 1.6,
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PathoSAMModel(MicroSAMModel):
    """PathoSAM: SAM fine-tuned on histopathology H&E data (MIDL 2025).

    Thin subclass of MicroSAMModel — identical AIS inference and fine-tuning
    pipeline, but with histopathology-specific checkpoints (vit_b/l/h_histopathology)
    and H&E-only compatible datasets.

    H&E images are passed as raw RGB (needs_raw_he_image=True), matching
    PathoSAM's training distribution.
    """

    supports_finetune = True
    needs_raw_he_image = True  # PathoSAM trained on RGB H&E; full colour matters

    def __init__(self, config: dict, device: str = "cuda"):
        # Default model_type to vit_b_histopathology if not specified
        if "model_type" not in config:
            config = dict(config, model_type="vit_b_histopathology")

        super().__init__(config, device)

    def _ais_thresholds(self) -> dict:
        """Return PathoSAM AIS thresholds: YAML config > histopathology defaults."""
        return {
            "center_distance_threshold": self.config.get(
                "center_distance_threshold",
                _PATHO_AIS_DEFAULTS["center_distance_threshold"],
            ),
            "boundary_distance_threshold": self.config.get(
                "boundary_distance_threshold",
                _PATHO_AIS_DEFAULTS["boundary_distance_threshold"],
            ),
            "foreground_threshold":  self.config.get("foreground_threshold",  0.5),
            "foreground_smoothing":  self.config.get("foreground_smoothing",   1.0),
            "distance_smoothing":    self.config.get(
                "distance_smoothing",
                _PATHO_AIS_DEFAULTS["distance_smoothing"],
            ),
            # Dataset-level minimum instance size, injected from
            # configs/datasets/<name>.yaml (see MicroSAMModel._ais_thresholds).
            # A previous model-level 50 discarded ~12% of CellBinDB H&E GT cells.
            "min_size": self.config.get("min_size", 1),
        }

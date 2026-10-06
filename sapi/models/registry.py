"""
Model registry: maps model names to their wrapper classes and default configs.

Usage:
    from sapi.models import get_model, list_models

    model = get_model("cellpose_cyto3", device="cuda")
    masks = model.predict_single(image)
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry definition
# Each entry: model_name -> (ModelClass, default_config)
# ---------------------------------------------------------------------------

def _build_registry() -> dict:
    from .cellpose_model import CellposeModel
    from .cellpose_sam_model import CellposeSAMModel
    from .stardist_model import StarDistModel
    from .mesmer_model import MesmerModel
    from .cellvit_model import CellViTModel
    from .deepcell_model import DeepCellModel
    from .mediar_model import MEDIARModel
    from .hovernet_model import HoverNetModel
    from .instanseg_model import InstanSegModel
    from .cellsam_model import CellSAMModel
    from .microsam_model import MicroSAMModel
    from .pathosam_model import PathoSAMModel

    return {
        # Cellpose 3.x variants
        "cellpose_cyto2": (
            CellposeModel,
            {"variant": "cellpose_cyto2", "diameter": None, "num_channels": 1},
        ),
        "cellpose_cyto3": (
            CellposeModel,
            {"variant": "cellpose_cyto3", "diameter": None, "num_channels": 1},
        ),
        # Cellpose 4.x / CellposeSAM (ViT-L backbone, requires cellpose >= 4.0)
        "cellpose_sam": (
            CellposeSAMModel,
            {"variant": "cellpose_sam", "diameter": None},
        ),
        # StarDist variants
        "stardist_fluo": (
            StarDistModel,
            {"variant": "stardist_fluo", "num_channels": 1},
        ),
        "stardist_he": (
            StarDistModel,
            {"variant": "stardist_he", "num_channels": 3},
        ),
        # Mesmer (DeepCell multiplex segmentation)
        "mesmer": (
            MesmerModel,
            {"compartment": "nuclear", "image_mpp": None},
        ),
        # CellViT — ViT-based cell instance segmentation (Hörst et al. 2024)
        # Available checkpoints (authors provide only these two backbones):
        #   cellvit_256   — ViT-256 backbone (~28.7 M params)
        #   cellvit_sam_h — SAM ViT-H backbone (~632 M params)
        "cellvit_256": (
            CellViTModel,
            {"variant": "cellvit_256", "num_classes": 6, "patch_size": 256},
        ),
        "cellvit_sam_h": (
            CellViTModel,
            {"variant": "cellvit_sam_h", "num_classes": 6, "patch_size": 256},
        ),
        # DeepCell nuclear segmentation (TensorFlow, inference-only)
        "deepcell": (
            DeepCellModel,
            {"model_version": "1.1", "image_mpp": None},
        ),
        # MEDIAR (NeurIPS 2022 Cell Seg Challenge winner)
        "mediar": (
            MEDIARModel,
            {"variant": "mediar", "weights_path": None},
        ),
        # HoverNet (H&E specialist, requires HE modality)
        "hovernet": (
            HoverNetModel,
            {"variant": "hovernet", "weights_path": None},
        ),
        # InstanSeg (embedding-based, all modalities)
        # single_channel_nuclei checkpoint: 1-channel input, nuclei output only.
        # Architecture (in_channels, cells_and_nuclei, mlp_width) is auto-detected
        # from checkpoint keys; YAML values override these registry defaults.
        "instanseg": (
            InstanSegModel,
            {"variant": "instanseg", "in_channels": 1, "cells_and_nuclei": False},
        ),
        # CellSAM — SAM ViT-B + CellFinder (AnchorDETR) foundation model
        # (Marks et al., Nature Methods 2025; inference-only, all modalities)
        "cellsam": (
            CellSAMModel,
            {"model_variant": "cellsam_general", "bbox_threshold": 0.4},
        ),
        # MicroSAM — AIS (Automatic Instance Segmentation) via UNETR decoder
        # (Archit et al., Nature Methods 2024)
        # All three variants are LM generalist models with a pretrained decoder.
        "microsam_vit_t_lm": (
            MicroSAMModel,
            {"variant": "microsam_vit_t_lm", "model_type": "vit_t_lm"},
        ),
        "microsam_vit_b_lm": (
            MicroSAMModel,
            {"variant": "microsam_vit_b_lm", "model_type": "vit_b_lm"},
        ),
        "microsam_vit_l_lm": (
            MicroSAMModel,
            {"variant": "microsam_vit_l_lm", "model_type": "vit_l_lm"},
        ),
        # PathoSAM — SAM fine-tuned on H&E histopathology (Griebel et al., MIDL 2025)
        # H&E-only. AIS inference via UNETR decoder (same pipeline as MicroSAM).
        # Checkpoints: vit_b/l/h_histopathology auto-downloaded via micro_sam model hub.
        "pathosamvit_b": (
            PathoSAMModel,
            {"variant": "pathosamvit_b", "model_type": "vit_b_histopathology"},
        ),
        "pathosamvit_l": (
            PathoSAMModel,
            {"variant": "pathosamvit_l", "model_type": "vit_l_histopathology"},
        ),
        "pathosamvit_h": (
            PathoSAMModel,
            {"variant": "pathosamvit_h", "model_type": "vit_h_histopathology"},
        ),
    }


MODEL_REGISTRY: dict = {}  # populated lazily to avoid circular imports at load time


def _ensure_registry():
    global MODEL_REGISTRY
    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_build_registry())


def list_models() -> list:
    """Return all registered model names."""
    _ensure_registry()
    return sorted(MODEL_REGISTRY.keys())


def get_model(
    model_name: str,
    device: str = "cuda",
    extra_config: Optional[dict] = None,
) -> BaseSegmentationModel:
    """
    Instantiate a model by name.

    Args:
        model_name:   one of the keys returned by list_models()
        device:       'cuda' or 'cpu'
        extra_config: override specific config keys for this model

    Returns:
        BaseSegmentationModel instance (not yet loaded / weights not yet downloaded)
    """
    _ensure_registry()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list_models()}"
        )
    cls, default_cfg = MODEL_REGISTRY[model_name]
    cfg = dict(default_cfg)
    if extra_config:
        cfg.update(extra_config)
    logger.info(f"Instantiating model '{model_name}' on device '{device}'")
    return cls(config=cfg, device=device)

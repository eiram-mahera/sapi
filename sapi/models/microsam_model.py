"""
MicroSAM model wrapper using Automatic Instance Segmentation (AIS).

AIS uses a UNETR decoder on top of SAM's image encoder to predict three maps:
  - Channel 0: foreground probability
  - Channel 1: center distance map           (low = near cell centre)
  - Channel 2: INVERTED boundary distance    (low = near cell centre, high at the
               boundary) — the watershed elevation map
A seeded watershed on these maps produces instance labels — no point-grid prompts.
Seeds are the pixels where channels 1 AND 2 are both below their thresholds, so
both channels must be low at cell centres; channel 2 is then flooded as the
elevation map (basins at centres, ridges at boundaries).

Reference: Archit et al. "Segment Anything for Microscopy", Nature Methods 2024.
Code: https://github.com/computational-cell-analytics/micro-sam

Model types that ship with a pretrained UNETR decoder (recommended):
  vit_b_lm, vit_l_lm, vit_t_lm              — light-microscopy generalist
  vit_b_histopathology, vit_l_histopathology  — histopathology

Plain SAM models (vit_b, vit_l, vit_h) have no bundled decoder.  A fresh UNETR
is initialised from random weights and must be fine-tuned before inference quality
is useful.

Fine-tuning trains all four components jointly:
  1. Per-instance SAM mask-decoder Dice loss (centroid point prompts), matching
     micro-sam's SamTrainer mask loss.
  2. Per-image UNETR 3-channel Dice loss, matching the official micro-sam
     DiceBasedDistanceLoss(mask_distances_in_bg=True):
       ch 0 (foreground):      DiceLoss(pred, fg_target)
       ch 1 (center distance): DiceLoss(pred * fg, center_dist_target * fg)
       ch 2 (boundary dist):   DiceLoss(pred * fg, boundary_dist_target * fg)

freeze_encoder=True  → SAM image encoder fully frozen (requires_grad=False).
                        SAM prompt encoder, mask decoder, UNETR decoder: trainable.
freeze_encoder=False → All four components fully trainable (full fine-tuning).

Requirements:
  micro-sam  ≥ 1.1  (pip install micro-sam)
  torch_em              (pulled in by micro-sam)
  segment_anything      (pulled in by micro-sam)

Environment setup:
  Install micro-sam in the active conda env:
    pip install micro-sam
"""

from __future__ import annotations

import copy
import logging
from typing import List

import numpy as np

from .base import BaseSegmentationModel, _EarlyStopper


# Module-level kornia augmentation pipeline — created once, reused every call.
# torch_em's 2D default: random H+V flip (p=0.5 each), no rotation, no intensity.
# same_on_batch=True ensures image and mask receive identical flip decisions.
_MICROSAM_AUG_PIPELINE = None

# torch-em's UNETR target transform — built once on first use (see
# _compute_unetr_targets), never at import time.
_UNETR_TARGET_TRANSFORM = None


def _microsam_augment(img: np.ndarray, mask: np.ndarray):
    """MicroSAM's native augmentation: random horizontal + vertical flip (p=0.5 each)."""
    import torch
    import kornia.augmentation as K

    global _MICROSAM_AUG_PIPELINE
    if _MICROSAM_AUG_PIPELINE is None:
        _MICROSAM_AUG_PIPELINE = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
            same_on_batch=True,
        )

    arr = img[None, None] if img.ndim == 2 else img.transpose(2, 0, 1)[None]
    it = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)
    mt = torch.as_tensor(mask[None, None].astype(np.float32))
    it_aug, mt_aug = _MICROSAM_AUG_PIPELINE(it, mt)
    a_img = it_aug[0].numpy()
    a_img = a_img[0] if a_img.shape[0] == 1 else a_img.transpose(1, 2, 0)
    return a_img.astype(np.float32), mt_aug[0, 0].numpy().astype(np.uint32)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model types that include a pretrained UNETR decoder in their checkpoint.
_HAS_DECODER: frozenset = frozenset([
    "vit_b_lm", "vit_l_lm", "vit_t_lm",
    "vit_b_em_organelles", "vit_l_em_organelles", "vit_t_em_organelles",
    "vit_b_histopathology", "vit_l_histopathology", "vit_h_histopathology",
    "vit_b_medical_imaging",
])

# Per-model recommended AIS thresholds from micro_sam/_model_settings.py.
_AIS_DEFAULTS: dict = {
    "vit_b_lm": {
        "center_distance_threshold": 0.4,
        "boundary_distance_threshold": 0.5,
        "distance_smoothing": 2.0,
    },
    "vit_l_lm": {
        "center_distance_threshold": 0.4,
        "boundary_distance_threshold": 0.4,
        "distance_smoothing": 1.6,
    },
    "vit_t_lm": {
        "center_distance_threshold": 0.5,
        "boundary_distance_threshold": 0.5,
        "distance_smoothing": 2.0,
    },
    # fallback for plain SAM / unlisted models
    "_default": {
        "center_distance_threshold": 0.5,
        "boundary_distance_threshold": 0.5,
        "distance_smoothing": 1.6,
    },
}

# Maximum instances sampled per image per training step — mirrors micro-sam's
# n_objects_per_batch=25 default.
_MAX_INSTANCES_PER_STEP: int = 25


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _dice_loss(pred, target) -> "torch.Tensor":
    """Binary Dice loss (smooth=1e-5)."""
    smooth = 1e-5
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def _compute_unetr_targets(inst_mask: np.ndarray) -> np.ndarray:
    """Build 3-channel float32 UNETR targets from an instance label mask.

    Delegates to micro-sam's own target transform,
    ``torch_em.transform.label.PerObjectDistanceTransform`` — the same object
    micro-sam's finetuning scripts pass as ``label_transform`` — rather than
    reimplementing it.  The channel polarity here is a contract with the
    watershed at inference time, so it must not drift from the reference.

    Channel 0 — foreground: 1 inside any cell, 0 in background.
    Channel 1 — center distance: 0 at the instance centre, rising to 1 at the
        farthest pixel.  The centre is the centroid, corrected to the pixel
        farthest from the boundary when the centroid falls outside a non-convex
        instance (``correct_centers=True``).
    Channel 2 — INVERTED boundary distance: 0 at the instance centre, 1 at the
        boundary.  Both distance channels are low at cell centres because AIS
        seeds on ``(center < t1) & (boundary < t2)`` and then floods channel 2
        as its elevation map — basins at centres, ridges at boundaries
        (``torch_em.util.segmentation.watershed_from_center_and_boundary_distances``,
        whose signature documents the argument as the "inverted distance
        prediction to object boundaries").

    Background is 1.0 in both distance channels (``distance_fill_value``),
    though it never reaches the loss: DiceBasedDistanceLoss(
    mask_distances_in_bg=True) masks both distance channels by the foreground
    target.

    ``min_size=1`` matches the benchmark's dataset-level min_size: no instance
    has an area below 1 px, so no ground-truth object is dropped from the
    targets.  micro-sam's own scripts pass 25 here; keeping
    every instance is the benchmark-wide convention and measures identically
    (only 59 of 102,480 CellBinDB instances are 1 px, and AJI+ is unchanged to
    4 dp either way).

    ``instances=False`` gives the (3, H, W) layout the UNETR decoder predicts;
    micro-sam's scripts pass True to prepend the raw label map, which its own
    trainer consumes separately and we do not need.

    Returns: float32 array of shape (3, H, W).
    """
    # Imported lazily, like every other micro-sam/torch-em import in this
    # module: sapi/models/registry.py imports this file in *every*
    # environment, most of which have no micro-sam installed.
    global _UNETR_TARGET_TRANSFORM
    if _UNETR_TARGET_TRANSFORM is None:
        from torch_em.transform.label import PerObjectDistanceTransform

        _UNETR_TARGET_TRANSFORM = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=False,
            min_size=1,
        )
    return _UNETR_TARGET_TRANSFORM(inst_mask).astype(np.float32, copy=False)


def _dice_loss_unetr(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """3-channel Dice loss for the UNETR decoder output.

    Replicates micro-sam's DiceBasedDistanceLoss(mask_distances_in_bg=True):
      ch 0 (foreground):      DiceLoss(pred[:,0], target[:,0])
      ch 1 (center distance): DiceLoss applied only within the foreground mask
      ch 2 (boundary dist):   DiceLoss applied only within the foreground mask

    pred:   (1, 3, H', W') — sigmoid-activated output from DecoderAdapter (already in [0,1])
    target: (1, 3, H', W') — output of _compute_unetr_targets, resized to match
    """
    smooth = 1.0
    # pred is already sigmoid-activated (DecoderAdapter uses final_activation="Sigmoid").
    # Applying sigmoid again would squash [0,1] → [0.5,0.73] and destroy gradient signal.
    p = pred
    fg = target[:, 0:1]         # foreground mask as weight for ch 1 and 2

    def _dice(a, b):
        inter = (a * b).sum()
        return 1.0 - (2.0 * inter + smooth) / (a.sum() + b.sum() + smooth)

    loss_fg = _dice(p[:, 0], target[:, 0])
    loss_cd = _dice(p[:, 1] * fg, target[:, 1] * fg)
    loss_bd = _dice(p[:, 2] * fg, target[:, 2] * fg)
    return loss_fg + loss_cd + loss_bd


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class MicroSAMModel(BaseSegmentationModel):
    """Micro-SAM with Automatic Instance Segmentation (AIS) via UNETR decoder.

    Inference: encoder → UNETR decoder (3-channel) → seeded watershed → instance labels.
    Fine-tuning: per-instance centroid prompts → SAM mask decoder Dice loss;
                 UNETR decoder is trained jointly with the mask decoder in both modes.
                 freeze_encoder=True freezes the image encoder (plain requires_grad=False,
                 not PEFT/LoRA); freeze_encoder=False fully fine-tunes the encoder too.
    """

    supports_finetune = True
    num_channels      = 1     # grayscale input; converted to RGB internally for SAM
    needs_raw_he_image = False  # Fluorescence-only model (the *_lm variants list no H&E
                                # dataset), so it always receives a 1-channel image.
                                # _to_uint8_rgb() replicates that channel 3x, matching
                                # micro_sam's own reference conversion (util.py::_to_image).
                                # PathoSAMModel is H&E-only and overrides this to True.

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.model_name     = config.get("variant", "microsam_vit_b_lm")
        self._model_type    = config.get("model_type", "vit_b_lm")
        self._ckpt_path     = config.get("checkpoint_path", None) or None
        self._decoder_path  = config.get("decoder_path", None) or None

        # Lazily loaded components
        self._predictor     = None  # SamPredictor
        self._decoder       = None  # DecoderAdapter (UNETR decoder)
        self._decoder_state = None  # OrderedDict — kept for rebuilding after finetune
        self._segmenter     = None  # InstanceSegmentationWithDecoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._predictor is None:
            self._load_model()

    def _load_model(self) -> None:
        """Download/load SAM predictor + UNETR decoder and build AIS segmenter."""
        from micro_sam.util import get_sam_model
        from micro_sam.instance_segmentation import (
            get_unetr,
            get_decoder,
            InstanceSegmentationWithDecoder,
        )

        logger.info(f"[{self.model_name}] Loading SAM model '{self._model_type}' …")
        predictor, state = get_sam_model(
            model_type=self._model_type,
            checkpoint_path=self._ckpt_path,
            device=self.device,
            return_state=True,
            decoder_path=self._decoder_path,
        )
        self._predictor = predictor

        if "decoder_state" in state:
            self._decoder_state = state["decoder_state"]
            self._decoder = get_decoder(
                predictor.model.image_encoder,
                self._decoder_state,
                self.device,
            )
            logger.info(f"[{self.model_name}] Loaded pretrained UNETR decoder → AIS ready")
        else:
            logger.warning(
                f"[{self.model_name}] '{self._model_type}' has no pretrained decoder. "
                "Initialising a fresh UNETR (random weights). "
                "AIS inference quality will be poor until the model is fine-tuned."
            )
            self._decoder = get_unetr(predictor.model.image_encoder, device=self.device)
            self._decoder_state = None

        self._segmenter = InstanceSegmentationWithDecoder(predictor, self._decoder)

    def _to_uint8_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert a raw Layer-A image to uint8 RGB (H×W×3) for SAM.

        H&E arrives as raw RGB [0,255] (cast directly); fluorescence arrives as
        single-channel raw counts (per-image min-max to [0,255], replicated to 3).
        SAM applies its own pixel mean/std normalisation internally.
        """
        if image.ndim == 3:
            return np.clip(image, 0, 255).astype(np.uint8)
        img = image.astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        norm = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
        u8 = (norm * 255.0).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    def _ais_thresholds(self) -> dict:
        """Return AIS thresholds: YAML config > per-model defaults > global defaults."""
        defaults = {
            **_AIS_DEFAULTS["_default"],
            **_AIS_DEFAULTS.get(self._model_type, {}),
        }
        return {
            "center_distance_threshold": self.config.get(
                "center_distance_threshold", defaults["center_distance_threshold"]
            ),
            "boundary_distance_threshold": self.config.get(
                "boundary_distance_threshold", defaults["boundary_distance_threshold"]
            ),
            "foreground_threshold": self.config.get("foreground_threshold", 0.5),
            "foreground_smoothing": self.config.get("foreground_smoothing", 1.0),
            "distance_smoothing": self.config.get(
                "distance_smoothing", defaults["distance_smoothing"]
            ),
            # Dataset-level minimum instance size, injected from
            # configs/datasets/<name>.yaml (never from a model YAML).  elf's
            # size_filter drops objects with `counts < min_size`, which is exactly
            # the benchmark's definition, so the value passes through unchanged;
            # min_size=1 is a no-op, matching micro_sam's own AIS default of 0.
            # (A previous benchmark-level 100 discarded ~29-35% of CellBinDB
            # ground-truth cells on HE/ssDNA/DAPI — mostly real nuclei clipped by
            # the crop border.)
            "min_size": self.config.get("min_size", 1),
        }

    # ------------------------------------------------------------------
    # AIS threshold calibration
    # ------------------------------------------------------------------

    def calibrate_ais_thresholds(
        self,
        cal_images: list,
        cal_masks: list,
        center_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        boundary_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    ) -> dict:
        """Grid-search center/boundary distance thresholds on labeled calibration images.

        Encodes each calibration image once, then sweeps all threshold combinations
        via cheap watershed-only generate() calls (no repeated encoder passes).
        Updates self.config so subsequent predict_single/predict_batch calls use the
        found thresholds automatically via _ais_thresholds().

        Args:
            cal_images: Raw float32 images in the same format as predict_single input.
            cal_masks:  Corresponding uint16 instance label maps (ground truth).
            center_thresholds:   Candidate center_distance_threshold values.
            boundary_thresholds: Candidate boundary_distance_threshold values.

        Returns:
            dict with keys: center_distance_threshold, boundary_distance_threshold,
            cal_aji (AJI on calibration images), n_cal_images, n_combos_searched.
        """
        self._ensure_loaded()
        from sapi.metrics import compute_all_metrics

        # Fixed params taken from the current config (not searched over).
        thr = self._ais_thresholds()
        fixed = {
            "foreground_threshold": thr["foreground_threshold"],
            "foreground_smoothing": thr["foreground_smoothing"],
            "distance_smoothing":   thr["distance_smoothing"],
            "min_size":             thr["min_size"],
        }

        from itertools import product as _product
        combos = list(_product(center_thresholds, boundary_thresholds))
        # combo_preds[(cdt, bdt)] accumulates one prediction per calibration image.
        combo_preds: dict = {c: [] for c in combos}

        for img in cal_images:
            img_rgb = self._to_uint8_rgb(img)
            self._segmenter.initialize(img_rgb)   # one encoder pass per image
            for (cdt, bdt) in combos:
                pred = self._segmenter.generate(
                    center_distance_threshold=cdt,
                    boundary_distance_threshold=bdt,
                    **fixed,
                )
                combo_preds[(cdt, bdt)].append(pred.astype(np.uint16))

        best_aji, best_cdt, best_bdt = -1.0, center_thresholds[3], boundary_thresholds[3]
        for (cdt, bdt), preds in combo_preds.items():
            aji = compute_all_metrics(preds, cal_masks).summary_dict()["aji_plus"]
            if aji > best_aji:
                best_aji, best_cdt, best_bdt = aji, cdt, bdt

        self.config["center_distance_threshold"]  = best_cdt
        self.config["boundary_distance_threshold"] = best_bdt
        logger.info(
            f"[{self.model_name}] AIS calibrated on {len(cal_images)} images  "
            f"cdt={best_cdt}  bdt={best_bdt}  cal_AJI={best_aji:.4f}  "
            f"({len(combos)} combos searched)"
        )
        return {
            "center_distance_threshold":  best_cdt,
            "boundary_distance_threshold": best_bdt,
            "cal_aji":           best_aji,
            "n_cal_images":      len(cal_images),
            "n_combos_searched": len(combos),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Run AIS on a single raw image.  Returns uint16 instance labels."""
        self._ensure_loaded()

        img_rgb = self._to_uint8_rgb(image)
        thr = self._ais_thresholds()

        self._segmenter.initialize(img_rgb)
        seg = self._segmenter.generate(
            center_distance_threshold=thr["center_distance_threshold"],
            boundary_distance_threshold=thr["boundary_distance_threshold"],
            foreground_threshold=thr["foreground_threshold"],
            foreground_smoothing=thr["foreground_smoothing"],
            distance_smoothing=thr["distance_smoothing"],
            min_size=thr["min_size"],
        )
        return seg.astype(np.uint16)

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def get_param_count(self) -> int:
        """Total parameters: SAM model + UNETR decoder (encoder shared, counted once)."""
        self._ensure_loaded()
        sam_params = sum(p.numel() for p in self._predictor.model.parameters())
        # DecoderAdapter contains only the decoder-side layers (not the encoder),
        # so we can sum all its parameters without double-counting.
        decoder_params = sum(p.numel() for p in self._decoder.parameters())
        return sam_params + decoder_params

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        """Fine-tune SAM + UNETR decoder jointly.

        Two losses are computed per image and summed:
          1. SAM mask-decoder Dice loss (per-instance centroid point prompts).
          2. UNETR decoder 3-channel Dice loss (foreground + center-dist + boundary-dist),
             matching micro-sam's DiceBasedDistanceLoss(mask_distances_in_bg=True).

        Args:
            support_images: raw images (augmented per step inside the loop).
            support_masks:  uint16 instance label arrays.
            **kwargs:
                n_epochs       (int)   training epochs              [100]
                learning_rate  (float) AdamW lr                     [1e-5]
                weight_decay   (float) AdamW weight decay           [1e-2]
                freeze_encoder (bool)  True  → image encoder frozen,
                                               prompt enc + mask dec + UNETR trainable.
                                       False → all four components fully trainable.
                patience       (int)   early-stopping patience      [20]
                val_images     (list)  validation images
                val_masks      (list)  validation masks
                scheduler_patience (int) ReduceLROnPlateau patience [10]
                scheduler_factor   (float) ReduceLROnPlateau factor [0.9]

            LR scheduler: ReduceLROnPlateau(mode='min', factor=0.9, patience=10).
            Source: micro_sam/finetuning/livecell_finetuning.py line 57.  Stepped
            on the held-out val loss when one is supplied, otherwise on the
            training loss — the benchmark's fewshot experiments hold out no
            validation image, and skipping the step would pin the LR for the
            whole run.

        AIS thresholds:
            Re-calibrated on the SUPPORT set after training and before every
            checkpoint evaluation — see the note at the end of this method.

        Precision:
            fp16 AMP (autocast + GradScaler) on CUDA, fp32 on CPU.
            freeze_encoder=True: encoder activations are not stored for backprop
              (~3 GB total); AMP is still applied for V100 fp16 Tensor Core speed.
            freeze_encoder=False: ViT-H attention weights require ~38 GB fp32 or
              ~19 GB fp16.  AMP makes full-encoder finetuning fit within V100-32 GB.

        Returns:
            dict with training metadata.
        """
        import torch
        # Load the predictor + decoder from checkpoint before training so that
        # self._predictor, self._decoder, and self._decoder_state are populated.
        # Without this, the fewshot framework (which creates a fresh model instance
        # per shot level and calls finetune() directly) would crash when trying to
        # rebuild the segmenter after training because self._predictor is None.
        self._ensure_loaded()
        from micro_sam.training.util import get_trainable_sam_model
        from micro_sam.instance_segmentation import (
            get_unetr,
            get_decoder,
            InstanceSegmentationWithDecoder,
        )

        n_epochs       = kwargs.get("n_epochs", 100)
        lr             = kwargs.get("learning_rate", 1e-5)
        wd             = kwargs.get("weight_decay", 1e-2)
        batch_size     = int(kwargs.get("batch_size", 8))
        freeze_encoder = kwargs.get("freeze_encoder", True)
        patience       = kwargs.get("patience", 20)
        val_images     = kwargs.get("val_images", [])
        val_masks      = kwargs.get("val_masks", [])

        # ── Load a fresh trainable copy of SAM (no PEFT) ────────────────
        # All SAM parameters start trainable; we freeze the image encoder below
        # if requested.  The prompt encoder and mask decoder are always trainable.
        trainable_sam, _ = get_trainable_sam_model(
            model_type=self._model_type,
            device=self.device,
            checkpoint_path=self._ckpt_path,
            return_state=True,
        )
        trainable_sam = trainable_sam.to(self.device)
        sam = trainable_sam.sam

        # Optionally freeze the image encoder entirely (no LoRA).
        if freeze_encoder:
            for p in sam.image_encoder.parameters():
                p.requires_grad_(False)
            n_frozen = sum(p.numel() for p in sam.image_encoder.parameters())
            logger.info(
                f"[{self.model_name}] Image encoder frozen: {n_frozen:,} params. "
                "Prompt encoder + mask decoder + UNETR decoder are trainable."
            )
        else:
            logger.info(
                f"[{self.model_name}] Full fine-tuning: all SAM components trainable."
            )

        trainable_params = [p for p in trainable_sam.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total     = sum(p.numel() for p in trainable_sam.parameters())
        logger.info(
            f"[{self.model_name}] SAM trainable params: "
            f"{n_trainable:,} / {n_total:,} ({100*n_trainable/max(n_total,1):.2f}%)"
        )

        # Build UNETR decoder connected to the *trainable* image encoder so that
        # its hooks capture intermediate encoder features during forward passes.
        # If a pretrained decoder state exists, load those weights; otherwise
        # start from the fresh random initialisation from get_unetr.
        # Not wrapped in try/except: the UNETR decoder is the only trainable
        # component AIS actually uses at inference, so falling back to
        # training_decoder=None would leave the run a silent no-op (in
        # frozen_encoder mode the fine-tuned model would equal the pretrained
        # one) while still reporting success.  A failure here is fatal.
        if self._decoder_state is not None:
            training_decoder = get_decoder(
                sam.image_encoder, self._decoder_state, self.device
            )
            logger.info(
                f"[{self.model_name}] UNETR decoder: pretrained weights loaded for training"
            )
        else:
            training_decoder = get_unetr(sam.image_encoder, device=self.device)
            logger.info(
                f"[{self.model_name}] UNETR decoder: fresh init for training"
            )
        decoder_params = list(training_decoder.parameters())
        n_dec = sum(p.numel() for p in decoder_params)
        trainable_params = trainable_params + decoder_params
        logger.info(
            f"[{self.model_name}] UNETR decoder trainable params: {n_dec:,}"
        )

        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # captured here where trainable_params is final (SAM trainable + UNETR
        # decoder).  n_frozen is the frozen SAM image encoder (0 in full mode);
        # the UNETR decoder is always trainable and adds nothing to the frozen
        # count.
        _n_trainable_params = int(sum(p.numel() for p in trainable_params))
        _n_frozen_params    = int(n_total - n_trainable)
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        # ReduceLROnPlateau — mirrors micro_sam/finetuning/livecell_finetuning.py:57
        # scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}
        sched_patience = kwargs.get("scheduler_patience", 10)
        sched_factor   = kwargs.get("scheduler_factor", 0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_factor, patience=sched_patience
        )

        # fp16 AMP: halves the attention-weight activation storage needed for
        # backprop.  For ViT-H with freeze_encoder=False this brings memory from
        # ~38 GB (fp32) to ~19 GB (fp16), fitting within V100-32 GB.
        # freeze_encoder=True already fits in fp32 (~3 GB) because the frozen
        # encoder's intermediate activations are never stored by autograd; AMP
        # still accelerates those runs via V100 fp16 Tensor Cores.
        # GradScaler prevents fp16 gradient underflow.
        # Disabled on CPU (fp16 matmul is unsupported on CPU).
        use_amp = str(self.device).startswith("cuda")
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except AttributeError:
            from torch.cuda.amp import GradScaler as _GS
            scaler = _GS(enabled=use_amp)

        # Training images stay raw (augmented per step); val converted once.
        train_imgs = list(support_images)
        train_msks = [m.astype(np.uint32) for m in support_masks]
        val_imgs   = [self._to_uint8_rgb(img) for img in val_images]
        val_msks   = [m.astype(np.uint32) for m in val_masks]

        # ── Training loop ────────────────────────────────────────────────
        stopper     = _EarlyStopper(patience=patience)
        train_losses: List[float] = []
        val_losses:   List[float] = []
        sam_losses:   List[float] = []   # per-instance SAM prompt loss (train)
        unetr_losses: List[float] = []   # UNETR decoder Dice loss (train)
        best_epoch  = 1
        best_decoder_sd: dict | None = None   # saved alongside stopper for the decoder

        # PathoSAM ViT-H full_model fine-tuning requires batch_size=1 to fit on V100 (32GB)
        epoch_batch_size = batch_size
        if self.model_name == "pathosamvit_h" and not freeze_encoder:
            epoch_batch_size = 1
            logger.info(f"[{self.model_name}] Full model fine-tuning: using batch_size=1")

        for epoch in range(n_epochs):
            # ─ Train ─────────────────────────────────────────────────────
            trainable_sam.train()
            # trainable_sam.train() puts BatchNorm layers in training mode, which
            # updates running_mean/running_var on every forward pass.  TinyViT
            # (vit_t) has BatchNorm in its ConvBN stem.  With only 3–20 support
            # images, the running stats quickly drift toward the specific images
            # selected by this seed, making encoder outputs seed-dependent.
            # Putting the image encoder in eval() stops the running stat updates
            # while leaving gradient flow intact for full_model mode — eval() only
            # controls BN stat accumulation, not requires_grad.
            sam.image_encoder.eval()
            if not freeze_encoder:
                for p in sam.image_encoder.parameters():
                    p.requires_grad_(True)
            epoch_loss, epoch_valid = 0.0, 0
            # The total is the sum of two independent objectives — the SAM
            # per-instance prompt loss and the UNETR decoder loss.  Only the
            # latter drives AIS at inference, and the two can move in opposite
            # directions and hide each other in the total, so track them apart.
            ep_sam, ep_sam_n, ep_dec, ep_dec_n = 0.0, 0, 0.0, 0
            order = np.random.permutation(len(train_imgs))

            for b_start in range(0, len(order), epoch_batch_size):
                batch_idx = order[b_start:b_start + epoch_batch_size]
                n_batch = len(batch_idx)

                optimizer.zero_grad()
                batch_valid = 0
                for idx in batch_idx:
                    aug_img, aug_msk = _microsam_augment(train_imgs[idx], train_msks[idx])
                    img_u8 = self._to_uint8_rgb(aug_img)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        loss, sam_term, dec_term = self._compute_step(
                            sam, trainable_sam, img_u8, aug_msk,
                            decoder=training_decoder,
                        )
                    if loss is None:
                        continue
                    if np.isfinite(sam_term):
                        ep_sam += sam_term
                        ep_sam_n += 1
                    if np.isfinite(dec_term):
                        ep_dec += dec_term
                        ep_dec_n += 1
                    # Accumulate unscaled; divide by valid count after the loop
                    # so the mean gradient is over images that actually produced
                    # a loss, not the total batch allocation.
                    scaler.scale(loss).backward()
                    epoch_loss += loss.detach().float().item()
                    batch_valid += 1

                if batch_valid > 0:
                    # Rescale accumulated gradients to true mean over valid images
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.div_(batch_valid)
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    # update() must follow a step()/unscale_() in the same
                    # iteration: an enabled GradScaler asserts that an inf-check
                    # was recorded, so calling it after a batch that produced no
                    # valid loss (batch_valid == 0) raises "No inf checks were
                    # recorded prior to update()". Keeping it inside this guard
                    # means a valid all-small-cell batch is simply skipped, not
                    # crashed.
                    scaler.update()
                    epoch_valid += batch_valid

            # Divide by the number of images that actually produced a loss this
            # epoch (not the number of batches) so avg_train is a true per-image
            # mean, comparable to avg_val below.
            avg_train = epoch_loss / max(epoch_valid, 1)
            train_losses.append(avg_train)
            avg_sam = ep_sam / ep_sam_n if ep_sam_n else float("nan")
            avg_dec = ep_dec / ep_dec_n if ep_dec_n else float("nan")
            sam_losses.append(avg_sam)
            unetr_losses.append(avg_dec)

            # ─ Validation ────────────────────────────────────────────────
            trainable_sam.eval()
            val_total, n_val = 0.0, 0
            with torch.no_grad():
                for img, msk in zip(val_imgs, val_msks):
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        loss, _, _ = self._compute_step(
                            sam, trainable_sam, img, msk,
                            decoder=training_decoder,
                        )
                    if loss is not None:
                        val_total += loss.detach().float().item()
                        n_val += 1

            # Use inf sentinel when all val images failed — prevents early stopping from
            # monitoring train loss (which decreases monotonically) and masking overfitting.
            avg_val = val_total / max(n_val, 1) if n_val > 0 else float("inf")
            # Only record a real val signal; `inf` is not a value and is not valid
            # JSON either.
            if n_val > 0:
                val_losses.append(avg_val)

            old_lr = optimizer.param_groups[0]["lr"]
            # Step the scheduler on the held-out val loss when there is one, and on
            # the training loss otherwise.  The benchmark's fewshot experiments hold
            # out no validation image (avg_val is `inf`), and without a step the
            # LR would stay pinned for the whole run with nothing to damp a
            # divergence.  micro-sam's own training always drives this scheduler
            # from a val loader.
            #
            # The training loss is the only signal available here and it is free.
            # Reusing the support images as an unaugmented monitor set (tried, and
            # it does work) would add ~33% to the measured training energy — one
            # extra forward per support image per epoch on top of ~3 for
            # forward+backward — which lands on Et → SAPI_LC for this model family
            # alone.  Never step on the `inf` sentinel.
            scheduler.step(avg_val if n_val > 0 else avg_train)
            new_lr = optimizer.param_groups[0]["lr"]

            lr_tag = f"  lr={new_lr:.2e}" if new_lr != old_lr else ""
            val_tag = f"  val={avg_val:.4f}" if n_val > 0 else ""
            logger.info(
                f"[{self.model_name}] epoch {epoch + 1}/{n_epochs}  "
                f"train={avg_train:.4f} (sam={avg_sam:.4f} unetr={avg_dec:.4f})"
                f"{val_tag}{lr_tag}"
            )

            # Early stopping and best-weight selection require a genuine held-out
            # signal — never the training loss, which would just select the most
            # overfitted epoch.
            fired = stopper.step(avg_val, trainable_sam.state_dict()) if n_val > 0 else False
            # Snapshot the decoder alongside the encoder on the same epoch the
            # stopper itself recognises as an improvement (stopper.counter == 0),
            # so both halves of the model are restored from one consistent epoch.
            # With no val set (or patience==0), there is no best-weight tracking:
            # just record the final epoch and let the decoder keep training.
            if n_val > 0 and patience > 0:
                if stopper.counter == 0:
                    best_epoch = epoch + 1
                    if training_decoder is not None:
                        best_decoder_sd = copy.deepcopy(training_decoder.state_dict())
            else:
                best_epoch = epoch + 1
            if fired:
                logger.info(
                    f"[{self.model_name}] Early stopping at epoch {epoch + 1}; "
                    f"best epoch: {best_epoch}"
                )
                break

        # Restore best checkpoint (encoder + decoder both to the same best epoch)
        stopper.restore_best(trainable_sam)
        if training_decoder is not None and best_decoder_sd is not None:
            training_decoder.load_state_dict(best_decoder_sd)

        # ── Rebuild predictor + decoder with the trained encoder ─────────
        # Replace the predictor's underlying SAM model so inference uses the
        # updated encoder.  Rebuild the decoder with the new encoder reference
        # (the UNETR's encoder attribute is a Python reference, not a copy).
        new_sam = trainable_sam.sam.to(self.device)
        self._predictor.model = new_sam

        if training_decoder is not None:
            # Rebuild decoder using get_decoder (same API that created training_decoder)
            # so the architecture and key naming (.block.) match the trained state_dict.
            # get_unetr() produces an incompatible architecture (.conv. keys + encoder
            # submodule) and must NOT be used here.
            trained_state = training_decoder.state_dict()
            self._decoder = get_decoder(new_sam.image_encoder, trained_state, self.device)
            self._decoder_state = trained_state   # keep for future fine-tune calls
        elif self._decoder_state is not None:
            self._decoder = get_decoder(
                new_sam.image_encoder, self._decoder_state, self.device
            )
        else:
            self._decoder = get_unetr(new_sam.image_encoder, device=self.device)

        self._segmenter = InstanceSegmentationWithDecoder(self._predictor, self._decoder)

        # ── Re-calibrate the AIS thresholds for the fine-tuned decoder ────
        # AIS inference is threshold-based — seeds are the pixels where BOTH
        # distance channels fall below their thresholds — while the training
        # objective (Dice on continuous distance maps) is threshold-free and
        # largely insensitive to a global shift in output calibration.  So
        # fine-tuning can leave the decoder fitting its targets just as well
        # while moving the set of pixels that pass a fixed threshold enormously.
        # The published `_AIS_DEFAULTS` are micro-sam's values for the
        # PRETRAINED checkpoint; applying them unchanged to a fine-tuned decoder
        # measures the wrong thing (observed: vit_b/mIF AJI+ 0.50 -> 0.08 with a
        # dead-flat decoder training loss).
        #
        # Calibrated on the SUPPORT images only — never the test set, which
        # would leak test labels into the evaluation.  The support masks are the
        # only labels a few-shot model is entitled to see, so this is part of
        # adaptation, not of evaluation.
        cal = self.calibrate_ais_thresholds(support_images, support_masks)

        out = {
            "n_train_images":  len(support_images),
            "n_train_patches": len(support_images),   # whole images, one pass/epoch (no tiling)
            "n_trainable_params": _n_trainable_params,
            "n_frozen_params":    _n_frozen_params,
            "n_epochs_max":    n_epochs,
            "n_epochs_run":    len(train_losses),
            "train_losses":    train_losses,
            # The two objectives that make up train_losses, tracked separately:
            # only unetr_losses drives AIS at inference.
            "sam_losses":      sam_losses,
            "unetr_losses":    unetr_losses,
            "best_epoch":      best_epoch,
            "lr_final":        optimizer.param_groups[0]["lr"],
            # Thresholds actually used for the post-fine-tuning evaluation,
            # fitted on the support set (see the note above).
            "ais_calibration": cal,
            # Empty unless a genuine held-out set was supplied — the benchmark's
            # fewshot experiments supply none.
            "val_losses":      val_losses,
        }
        return out

    # ------------------------------------------------------------------
    # Per-image step (used for both train and val)
    # ------------------------------------------------------------------

    def _compute_step(
        self,
        sam,
        trainable_sam,
        img_rgb: np.ndarray,
        inst_mask: np.ndarray,
        decoder=None,
    ):
        """Forward pass + loss for one image.

        Computes two losses and sums them:
          1. Per-instance SAM mask-decoder Dice loss (centroid point prompts).
          2. Per-image UNETR 3-channel Dice loss when ``decoder`` is provided,
             matching micro-sam's DiceBasedDistanceLoss(mask_distances_in_bg=True).

        Called inside a gradient context for training; wrapped in torch.no_grad()
        by the caller for validation.

        Returns ``(total_loss, sam_term, unetr_term)``.  ``total_loss`` is the
        tensor to backpropagate, or None if the image yielded neither loss.  The
        two terms are detached floats for logging (NaN when that term was not
        computed) — they are reported separately because only the UNETR term
        drives AIS at inference, so a rising SAM term can hide a falling UNETR
        term in the total, or vice versa.
        """
        import torch
        import torch.nn.functional as F

        _NA = float("nan")
        instances = np.unique(inst_mask)
        instances = instances[instances > 0]
        if len(instances) == 0:
            return None, _NA, _NA

        # Randomly subsample if there are too many instances per step.
        if len(instances) > _MAX_INSTANCES_PER_STEP:
            instances = np.random.choice(
                instances, _MAX_INSTANCES_PER_STEP, replace=False
            )

        device = self.device

        # ── Preprocess image through SAM's transform pipeline ────────────
        # TrainableSAM.preprocess: resize longest-side → normalise → pad to square.
        img_t = (
            torch.as_tensor(img_rgb.copy(), dtype=torch.float32)
            .permute(2, 0, 1)[None]          # (1, 3, H, W)
            .to(device)
        )
        img_pre, input_size = trainable_sam.preprocess(img_t)

        # ── Image encoder (gradients flow here during training) ───────────
        image_emb = sam.image_encoder(img_pre)  # (1, 256, 64, 64)

        orig_size = img_rgb.shape[:2]  # (H, W)
        total_loss = None
        n_valid = 0

        for inst_id in instances:
            binary = (inst_mask == inst_id)
            # Small cells (<10 px) are valid data — they are excluded ONLY from
            # this auxiliary per-instance SAM prompt loss (a centroid prompt with
            # a few-pixel target is an unstable training signal). They still
            # participate fully in the UNETR decoder loss below (built from the
            # complete instance mask), which is the signal AIS inference relies on.
            if binary.sum() < 10:
                continue

            # ── Centroid prompt in original-image (x, y) coordinates ─────
            yx = np.argwhere(binary).mean(axis=0)           # (y, x)
            pt_orig = np.array([[yx[1], yx[0]]])             # (1, 2) → (x, y)
            pt_sam  = trainable_sam.transform.apply_coords(pt_orig, orig_size)

            pt_t = (
                torch.tensor(pt_sam, dtype=torch.float32)
                .unsqueeze(0)           # (1, 1, 2)
                .to(device)
            )
            lb_t = torch.ones(1, 1, dtype=torch.int).to(device)  # (1, 1)

            # ── Prompt encoder + mask decoder ─────────────────────────────
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=(pt_t, lb_t), boxes=None, masks=None
            )
            low_res, iou_preds = sam.mask_decoder(
                image_embeddings=image_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,       # 3 mask candidates
            )

            # Select the mask with the highest predicted IoU
            best_idx  = iou_preds.argmax(dim=1).item()
            best_low  = low_res[:, best_idx : best_idx + 1]    # (1, 1, 256, 256)
            upscaled  = sam.postprocess_masks(
                best_low, input_size=input_size, original_size=orig_size
            )                                                    # (1, 1, H, W)

            # ── Loss against ground-truth binary mask ──────────────────────
            gt   = torch.tensor(
                binary.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)               # (1, 1, H, W)
            pred = torch.sigmoid(upscaled)

            # Dice only — micro-sam's SamTrainer uses
            # torch_em.loss.DiceLoss(reduce_channel=None) for the mask loss
            # (sam_trainer.py:56) and adds only a bounded IoU-regression term.
            # No BCE term: it appears in no micro-sam code path and, being
            # unbounded, could run away.
            loss = _dice_loss(pred, gt)
            total_loss = loss if total_loss is None else total_loss + loss
            n_valid += 1

        # Mean per-instance SAM loss. This may be None when every instance fell
        # below the auxiliary size guard above (a valid all-small-cell image) —
        # that is fine: the UNETR decoder loss below is computed independently
        # from the full instance mask and is added regardless.
        if total_loss is not None:
            total_loss = total_loss / n_valid
        sam_term = float(total_loss.detach()) if total_loss is not None else _NA

        # ── UNETR decoder 3-channel Dice loss ─────────────────────────────
        # Supervises all three output channels against distance-transform targets
        # derived from the instance mask, matching micro-sam's official training
        # loss DiceBasedDistanceLoss(mask_distances_in_bg=True).
        # image_emb was already computed above — no second encoder forward pass.
        # This block is deliberately NOT wrapped in try/except: a swallowed
        # failure would reduce training to the SAM prompt loss alone, which moves
        # nothing AIS uses at inference, leaving a plausible loss curve and a
        # quietly untrained decoder.  Let it raise.
        if decoder is not None:
            # Build (3, H, W) targets: foreground / center-dist / boundary-dist
            unetr_targets_np = _compute_unetr_targets(inst_mask)   # (3, H, W)
            unetr_targets = torch.from_numpy(unetr_targets_np).unsqueeze(0).to(device)
            # (1, 3, H, W)

            # DecoderAdapter.forward requires (input_, input_shape, original_shape).
            # Passing orig_size here means postprocess_masks resizes output to
            # original image resolution, so unetr_targets resize below is a no-op.
            decoder_pred = decoder(image_emb, input_size, orig_size)  # (1, 3, H, W)

            # Resize targets to decoder output spatial resolution.
            # Bilinear for smooth distance maps; nearest for binary fg channel.
            H_out, W_out = decoder_pred.shape[-2:]
            if unetr_targets.shape[-2:] != (H_out, W_out):
                fg_ch  = F.interpolate(unetr_targets[:, :1],  size=(H_out, W_out), mode="nearest")
                dist_ch = F.interpolate(unetr_targets[:, 1:], size=(H_out, W_out), mode="bilinear",
                                        align_corners=False)
                unetr_targets = torch.cat([fg_ch, dist_ch], dim=1)

            dec_loss = _dice_loss_unetr(decoder_pred, unetr_targets)
            unetr_term = float(dec_loss.detach())
            total_loss = dec_loss if total_loss is None else total_loss + dec_loss
        else:
            unetr_term = _NA

        # None only if the SAM per-instance loop produced nothing AND no decoder
        # was available (decoder is None).
        return total_loss, sam_term, unetr_term

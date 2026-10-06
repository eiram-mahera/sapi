"""
MEDIAR model wrapper (MEDIARFormer with MIT-B5 encoder).

MEDIAR won the NeurIPS 2022 Cell Segmentation Challenge. It uses an MAnet
architecture with a MiT-B5 (Mix Transformer) encoder and two output heads:
  - gradflow_head: 2-channel gradient flow field (dP)
  - cellprob_head: 1-channel cell probability

Post-processing replicates Cellpose's gradient flow tracking (compute_masks).

Paper: https://arxiv.org/abs/2212.03465

Requirements:
  segmentation_models_pytorch >= 0.3
  monai >= 1.0
  fastremap
  timm (for MiT-B5 encoder)

Setup:
  Set MEDIAR_ROOT env var (or 'mediar_root' in configs/models/mediar.yaml) to the
  directory containing the MEDIAR source (the folder with train_tools/ and core/).
  Set MEDIAR_WEIGHTS env var (or 'weights_path') to the from_phase2.pth file.

MEDIAR normalises its own input (percentile [0, 99.5] -> [0, 1], its training
convention) and augments inside finetune().  Single-channel grayscale is
replicated to 3 channels; H&E is preprocessed via HED stain separation
(Hematoxylin channel only) before replication, restoring the cell=bright /
background=dark polarity that matches MEDIAR's fluorescence-dominant training
distribution.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)

_MEDIAR_MISSING = (
    "MEDIAR source directory not configured.\n"
    "Set MEDIAR_ROOT environment variable or 'mediar_root' in configs/models/mediar.yaml "
    "to the MEDIAR source directory (the folder containing train_tools/ and core/)."
)


def _resolve_mediar_root(config: dict) -> "Path | None":
    """Return the MEDIAR source root, or None if not configured."""
    root = os.environ.get("MEDIAR_ROOT") or config.get("mediar_root", None)
    return Path(os.path.expanduser(root)) if root else None


def _ensure_mediar_path(mediar_root: "Path | None"):
    if mediar_root is None:
        raise ImportError(_MEDIAR_MISSING)
    # Disable wandb before any MEDIAR import — core/utils.py imports wandb at
    # the top level and would fail or hang without a configured account.
    os.environ.setdefault("WANDB_MODE", "disabled")
    for p in [
        str(mediar_root),                    # for internal 'from core.BaseTrainer'
        str(mediar_root / "train_tools"),    # for 'from models.MEDIARFormer'
        str(mediar_root / "core"),           # for 'from MEDIAR.utils'
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


@contextlib.contextmanager
def _forced_min_size(min_size: int):
    """Force MEDIAR's post-processing size filter to the dataset's min_size.

    MEDIAR's ``compute_masks`` hardcodes ``morphology.remove_small_objects(cp_mask,
    min_size=16)`` — it is a literal, not a parameter, so the only way to reach it
    without editing the vendored MEDIAR source is to swap the function it calls for
    the duration of that call.  skimage drops components with ``size < min_size``,
    so min_size=1 is an exact no-op.

    16 px discarded real ground-truth nuclei clipped by the crop border.  The patch
    is restored in ``finally`` and only wraps the one call, so nothing else in the
    process sees a modified skimage.
    """
    from skimage import morphology

    original = morphology.remove_small_objects
    forced = min_size          # captured under a different name — see below

    # The shim's second parameter MUST be named `min_size`: MEDIAR calls
    # `remove_small_objects(cp_mask, min_size=16)` with a KEYWORD argument, so the
    # name is what swallows the hardcoded 16.  Renaming it lets the keyword fall
    # through to **kwargs and raises
    # `TypeError: got multiple values for argument 'min_size'`.
    def _forced(ar, min_size=64, *args, **kwargs):
        return original(ar, forced, *args, **kwargs)

    morphology.remove_small_objects = _forced
    try:
        yield
    finally:
        morphology.remove_small_objects = original


def _mediar_normalize(img: np.ndarray) -> np.ndarray:
    """MEDIAR's own normalisation: percentile [0, 99.5] on non-zero pixels rescaled
    to [0, 1] (replicates CustomNormalizeImage + ScaleIntensity)."""
    img = img.astype(np.float32)
    nz = img[img != 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(nz, [0.0, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)


_MEDIAR_AUG: dict = {}  # {patch_size: Compose}; built lazily


def _build_mediar_aug(patch_size: int = 256):
    """MEDIAR's native MONAI augmentation, matching the spatial+intensity transforms
    in train_tools/data_utils/transforms.py (applied AFTER normalisation).

    patch_size controls SpatialPadd / RandSpatialCropd: use 256 for 256×256
    fluorescence images and 512 for 512×512 H&E images, so the crop size matches
    both training context and the sliding_window_inference roi_size used at inference.
    """
    from monai.transforms import (
        Compose, RandZoomd, SpatialPadd, RandSpatialCropd, RandAxisFlipd,
        RandRotate90d, RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd,
        RandHistogramShiftd, RandGaussianSharpend,
    )
    transforms = [
        RandZoomd(keys=["img", "label"], prob=0.5, min_zoom=0.25, max_zoom=1.5,
                  mode=["area", "nearest"], keep_size=False),
        SpatialPadd(keys=["img", "label"], spatial_size=patch_size),
        RandSpatialCropd(keys=["img", "label"], roi_size=patch_size, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
    ]
    try:
        from train_tools.data_utils.custom import IntensityDiversification
        transforms.append(IntensityDiversification(keys=["img"], allow_missing_keys=True))
    except Exception as exc:
        logger.warning(f"MEDIAR IntensityDiversification unavailable, skipping: {exc}")
    transforms += [
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandGaussianSharpend(keys=["img"], prob=0.25),
    ]
    return Compose(transforms)


def _mediar_augment(img: np.ndarray, mask: np.ndarray, patch_size: int = 256):
    """Apply MEDIAR's native MONAI augmentation to a normalised [0,1] image + mask.

    img: (H,W) or (H,W,3) float32 in [0,1]; mask: (H,W) uint16.  Returns the same
    formats.  The MONAI dict pipeline applies geometry to img+label and intensity
    to img only (matching MEDIAR train_transforms).

    patch_size should match the image spatial dimensions (256 for fluorescence,
    512 for H&E), ensuring augmentation crop = inference roi_size = image size."""
    import torch
    global _MEDIAR_AUG
    if patch_size not in _MEDIAR_AUG:
        _MEDIAR_AUG[patch_size] = _build_mediar_aug(patch_size)
    img_c = img[None] if img.ndim == 2 else img.transpose(2, 0, 1)  # (C,H,W)
    d = {
        "img":   torch.as_tensor(np.ascontiguousarray(img_c), dtype=torch.float32),
        "label": torch.as_tensor(mask[None].astype(np.float32)),
    }
    out = _MEDIAR_AUG[patch_size](d)

    def _to_np(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    a_img = _to_np(out["img"])                          # (C,H,W)
    a_msk = _to_np(out["label"])[0].astype(np.uint16)   # (H,W)
    a_img = a_img[0] if a_img.shape[0] == 1 else a_img.transpose(1, 2, 0)
    return a_img.astype(np.float32), a_msk


class MEDIARModel(BaseSegmentationModel):
    """MEDIAR segmentation model wrapper (MEDIARFormer, MIT-B5 encoder)."""

    supports_finetune = True
    needs_raw_he_image = False  # HED stain separation applied by loader; Hematoxylin channel replicated to 3ch

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.min_size = int(config.get("min_size", 1))
        self.model_name = config.get("variant", "mediar")
        self._device = device
        self._mediar_root = _resolve_mediar_root(config)
        env_path = os.environ.get("MEDIAR_WEIGHTS", "")
        raw = env_path or config.get("weights_path", None)
        self.weights_path = os.path.expanduser(raw) if raw else None
        config["weights_path"] = self.weights_path  # keep config snapshot consistent with env-var override

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _ensure_mediar_path(self._mediar_root)
        import torchvision  # must be fully initialized before timm/smp touches it
        import torch
        from models.MEDIARFormer import MEDIARFormer

        net = MEDIARFormer(
            encoder_name="mit_b5",
            encoder_weights=None,
            in_channels=3,
            classes=3,
        )

        if self.weights_path and Path(self.weights_path).exists():
            state = torch.load(self.weights_path, map_location="cpu", weights_only=False)  # noqa: S614
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            result = net.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logger.warning(
                    f"MEDIAR checkpoint mismatch: missing={result.missing_keys}, "
                    f"unexpected={result.unexpected_keys}"
                )
            logger.info(f"Loaded MEDIAR weights from {self.weights_path}")
        else:
            logger.warning(
                "MEDIAR pretrained weights not found. Set 'weights_path' in "
                "configs/models/mediar.yaml or the MEDIAR_WEIGHTS environment variable."
            )

        dev = self._device if self._device != "cpu" else "cpu"
        net = net.to(dev)
        net.eval()
        self._model = net

    # ------------------------------------------------------------------
    # Pre/post helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_3ch_tensor(image: np.ndarray):
        """Convert float32 H×W (or H×W×C) to torch (1, 3, H, W) tensor."""
        import torch
        if image.ndim == 2:
            img3 = np.stack([image, image, image], axis=0)   # (3, H, W)
        elif image.ndim == 3 and image.shape[2] in (1, 3):
            if image.shape[2] == 1:
                img3 = np.concatenate([image] * 3, axis=2).transpose(2, 0, 1)
            else:
                img3 = image.transpose(2, 0, 1).astype(np.float32)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        return torch.from_numpy(img3).float().unsqueeze(0)

    def _infer_tensor(self, tensor):
        """Run sliding-window inference and return (3, H, W) numpy array."""
        import torch
        from monai.inferers import sliding_window_inference

        dev = next(self._model.parameters()).device
        tensor = tensor.to(dev)
        roi_size = min(tensor.shape[-2], tensor.shape[-1])  # matches image size (256 or 512)
        with torch.no_grad():
            pred = sliding_window_inference(
                tensor,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=self._model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.5,
            )
        return pred.cpu().squeeze(0).numpy()   # (3, H, W)

    @staticmethod
    def _postprocess(pred: np.ndarray, use_gpu: bool, device, min_size: int = 1,
                     flow_threshold: float = 0.4,
                     cellprob_threshold: float = 0.5) -> np.ndarray:
        """Convert model output to instance mask via gradient-flow tracking.

        flow_threshold/cellprob_threshold were hardcoded here at MEDIAR's
        published values, which are tuned for the PRETRAINED checkpoint.  They
        are now parameters so calibrate_thresholds() can re-fit them on the
        support set after fine-tuning.
        """
        from MEDIAR.utils import compute_masks

        dP = pred[:2]                           # (2, H, W)
        cellprob = 1.0 / (1.0 + np.exp(-pred[2]))  # sigmoid

        try:
            with _forced_min_size(min_size):
                mask, _ = compute_masks(
                    dP,
                    cellprob,
                    use_gpu=use_gpu,
                    flow_threshold=flow_threshold,
                    device=device,
                    cellprob_threshold=cellprob_threshold,
                )
        except Exception as e:
            # compute_masks in MEDIAR source has a latent bug where 'mask' is
            # returned before assignment when no cells pass the probability
            # threshold.  Return an empty mask so the image is counted as
            # producing zero predictions rather than crashing the run.
            logger.warning(f"MEDIAR compute_masks failed ({e}); returning empty mask")
            h, w = pred.shape[1], pred.shape[2]
            return np.zeros((h, w), dtype=np.uint16)
        return mask.astype(np.uint16)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        _ensure_mediar_path(self._mediar_root)
        tensor = self._to_3ch_tensor(_mediar_normalize(image))
        pred = self._infer_tensor(tensor)
        use_gpu = self._device.startswith("cuda")
        dev = next(self._model.parameters()).device
        return self._postprocess(pred, use_gpu=use_gpu, device=dev,
                                 min_size=self.min_size)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Batch inference: stack up to batch_size images and run one sliding-window call."""
        self._ensure_loaded()
        _ensure_mediar_path(self._mediar_root)
        import torch
        from monai.inferers import sliding_window_inference

        dev = next(self._model.parameters()).device
        use_gpu = self._device.startswith("cuda")
        results = []

        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            # Stack into (B, 3, H, W) — _to_3ch_tensor returns (1, 3, H, W)
            batch = torch.cat(
                [self._to_3ch_tensor(_mediar_normalize(img)) for img in chunk], dim=0
            ).to(dev)                                      # (B, 3, H, W)

            roi_size = min(batch.shape[-2], batch.shape[-1])  # matches image size (256 or 512)
            with torch.no_grad():
                pred = sliding_window_inference(
                    batch,
                    roi_size=roi_size,
                    sw_batch_size=len(chunk),              # all images per window step
                    predictor=self._model,
                    padding_mode="constant",
                    mode="gaussian",
                    overlap=0.5,
                )                                          # (B, 3, H, W)

            pred_np = pred.cpu().numpy()                   # (B, 3, H, W)
            for b in range(len(chunk)):
                results.append(self._postprocess(pred_np[b], use_gpu=use_gpu, device=dev,
                                                min_size=self.min_size))

        return results

    # NOTE — MEDIAR thresholds are deliberately NOT calibrated.
    # flow_threshold=0.4 / cellprob_threshold=0.5 are MEDIAR's published values
    # (literals in its own compute_masks call).  MEDIAR ships no threshold
    # optimisation of any kind and the paper only notes that it "adopts the
    # gradient flow tracking in CellPose", whose authors fixed these once and
    # kept them.  Fitting them here would be a benchmark-level intervention
    # beyond the reference.

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        _ensure_mediar_path(self._mediar_root)
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from MEDIAR.utils import masks_to_flows

        self._ensure_loaded()

        n_epochs      = kwargs.get("n_epochs", 25)
        lr            = kwargs.get("learning_rate", 1e-5)
        wd            = kwargs.get("weight_decay", 1e-4)
        batch_size    = int(kwargs.get("batch_size", 8))
        freeze_encoder = kwargs.get("freeze_encoder", True)
        val_images    = kwargs.get("val_images", [])
        val_masks     = kwargs.get("val_masks", [])
        patience      = int(kwargs.get("patience", 20))
        # Derive patch size from image dimensions so that training context matches
        # inference roi_size (which is min(H,W) in predict_batch).
        # 256×256 fluorescence images → patch_size=256; 512×512 H&E → patch_size=512.
        from .base import split_into_patches
        if support_images:
            _h, _w = support_images[0].shape[:2]
            patch_size = 512 if min(_h, _w) >= 512 else 256
        else:
            patch_size = 256
        logger.info(f"MEDIAR finetune: patch_size={patch_size}")
        patched_images: list = []
        patched_masks:  list = []
        for img, msk in zip(support_images, support_masks):
            for img_p, msk_p in split_into_patches(img, msk, patch_size):
                patched_images.append(img_p)
                patched_masks.append(msk_p)
        support_images = patched_images
        support_masks  = patched_masks
        n_shots        = len(support_images)
        logger.info(f"MEDIAR finetune: {n_shots} training patches")

        dev = next(self._model.parameters()).device
        use_gpu = self._device.startswith("cuda")

        if freeze_encoder and hasattr(self._model, "encoder"):
            for param in self._model.encoder.parameters():
                param.requires_grad = False
            n_enc = sum(p.numel() for p in self._model.encoder.parameters())
            logger.info(f"MEDIAR encoder frozen: {n_enc:,} parameters frozen")

        bce_loss  = nn.BCEWithLogitsLoss()
        mse_loss  = nn.MSELoss()
        optimizer = optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=lr, weight_decay=wd
        )
        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # captured while the freeze flags are active (restored before return).
        _n_trainable_params = int(sum(p.numel() for p in self._model.parameters() if p.requires_grad))
        _n_frozen_params    = int(sum(p.numel() for p in self._model.parameters() if not p.requires_grad))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

        # AMP: mirrors authors' amp=true in finetuning2.json
        use_amp = use_gpu
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except AttributeError:
            from torch.cuda.amp import GradScaler as _GS
            scaler = _GS(enabled=use_amp)

        # ── Helper: convert one image+mask pair to MEDIAR training tensors ──
        def _to_mediar_tensors(img, mask):
            img  = _mediar_normalize(img)
            mask = mask.astype(np.uint16)
            if img.ndim == 2:
                img3 = np.stack([img, img, img], axis=0)
            elif img.ndim == 3 and img.shape[2] == 3:
                img3 = img.transpose(2, 0, 1)
            else:
                img3 = np.stack([img[..., 0]] * 3, axis=0)
            binary = (mask > 0).astype(np.float32)
            if mask.max() > 0:
                flows = masks_to_flows(mask.astype(np.int32), use_gpu=use_gpu, device=dev)
            else:
                h, w = mask.shape
                flows = np.zeros((2, h, w), dtype=np.float32)
            return (torch.from_numpy(img3[None]).float().to(dev),
                    torch.from_numpy(flows[None]).float().to(dev),
                    torch.from_numpy(binary[None]).float().to(dev))

        # Pre-process val tensors once (expensive flow computation done upfront)
        val_tensors = [_to_mediar_tensors(im, mk)
                       for im, mk in zip(val_images, val_masks)] if val_images else []

        from .base import _EarlyStopper
        stopper = _EarlyStopper(patience)

        self._model.train()
        train_losses: list = []
        val_losses:   list = []
        best_epoch    = 0

        for epoch in range(n_epochs):
            indices     = np.random.permutation(n_shots)
            epoch_loss  = 0.0
            n_batches   = 0

            for batch_start in range(0, n_shots, batch_size):
                batch_idx = indices[batch_start: batch_start + batch_size]
                imgs_b, flows_b, binary_b = [], [], []

                for i in batch_idx:
                    img_n = _mediar_normalize(support_images[i])
                    aug_img, aug_mask = _mediar_augment(img_n, support_masks[i], patch_size)
                    if aug_img.ndim == 2:
                        img3 = np.stack([aug_img, aug_img, aug_img], axis=0)
                    elif aug_img.ndim == 3 and aug_img.shape[2] == 3:
                        img3 = aug_img.transpose(2, 0, 1)
                    else:
                        img3 = np.stack([aug_img[..., 0]] * 3, axis=0)
                    binary = (aug_mask > 0).astype(np.float32)
                    if aug_mask.max() > 0:
                        flows = masks_to_flows(aug_mask.astype(np.int32), use_gpu=use_gpu, device=dev)
                    else:
                        h, w = aug_mask.shape
                        flows = np.zeros((2, h, w), dtype=np.float32)
                    imgs_b.append(img3)
                    flows_b.append(flows)
                    binary_b.append(binary)

                imgs_t   = torch.from_numpy(np.stack(imgs_b)).float().to(dev)
                flows_t  = torch.from_numpy(np.stack(flows_b)).float().to(dev)
                binary_t = torch.from_numpy(np.stack(binary_b)).float().to(dev)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred     = self._model(imgs_t)
                    loss_bce = bce_loss(pred[:, 2], binary_t)
                    loss_mse = mse_loss(pred[:, :2], flows_t * 5.0)
                    loss     = loss_bce + 0.5 * loss_mse
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            train_losses.append(avg)
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  MEDIAR finetune epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

            # ── Early stopping ─────────────────────────────────────────────
            if val_tensors and patience > 0:
                self._model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    vl_total = 0.0
                    for it, ft, bt in val_tensors:
                        pred_v = self._model(it)   # single forward pass
                        vl_total += (
                            bce_loss(pred_v[:, 2], bt) +
                            0.5 * mse_loss(pred_v[:, :2], ft * 5.0)
                        ).item()
                    vl = vl_total / len(val_tensors)
                self._model.train()
                val_losses.append(vl)
                if stopper.step(vl, self._model.state_dict()):
                    logger.info(
                        f"  MEDIAR early stop at epoch {epoch+1}  "
                        f"best_val={stopper.best:.4f} (epoch {best_epoch})"
                    )
                    break
                if stopper.counter == 0:
                    best_epoch = epoch + 1
            else:
                best_epoch = epoch + 1

        stopper.restore_best(self._model)

        if freeze_encoder and hasattr(self._model, "encoder"):
            for param in self._model.encoder.parameters():
                param.requires_grad = True
            logger.info("MEDIAR encoder unfrozen")

        self._model.eval()

        return {
            "n_epochs_run":        len(train_losses),
            "n_epochs_max":        n_epochs,
            "best_epoch":          best_epoch,
            "early_stopping_mode": "patience_restore_best_weights",
            "n_train_images":      n_shots,
            "n_train_patches":     n_shots,   # support images tiled to patch_size (one pass/epoch)
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
            "train_losses":        train_losses,
            "val_losses":          val_losses,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_param_count(self) -> int:
        self._ensure_loaded()
        try:
            return sum(p.numel() for p in self._model.parameters())
        except Exception:
            return -1

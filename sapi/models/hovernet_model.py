"""
HoverNet model wrapper (H&E specialist).

HoverNet uses a ResNet50-like encoder with two decoder branches:
  - NP branch: nucleus probability (2-channel softmax)
  - HV branch: horizontal-vertical gradient maps (2-channel regression)

Post-processing: Sobel gradients on hover maps + watershed to separate nuclei.

Paper: https://arxiv.org/abs/1812.06499

Requirements:
  opencv-python, scipy, scikit-image, scikit-learn, torch

Setup:
  Set HOVERNET_ROOT env var (or 'hovernet_root' in configs/models/hovernet.yaml)
  to the hover_net source directory (the folder containing models/ and post_proc.py).
  Set HOVERNET_WEIGHTS env var (or 'weights_path') to the pretrained .tar file.

Compatible datasets: HE, 10xGenomics_HE only.  H&E is passed as raw RGB
(H×W×3, float32 [0,255], needs_raw_he_image=True); HoverNet's forward() divides
by 255 internally.  finetune() augments geometrically per step.

Tiling: original mode uses 270×270 input → 80×80 output patches with step=80.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)

# Input/output patch sizes for 'original' mode
_WIN_SIZE = 270
_MSK_SIZE = 80
_BORDER = (_WIN_SIZE - _MSK_SIZE) // 2    # 95 px padding on each side

_HOVERNET_MISSING = (
    "HoverNet source directory not configured.\n"
    "Set HOVERNET_ROOT environment variable or 'hovernet_root' in "
    "configs/models/hovernet.yaml to the hover_net source directory "
    "(the folder containing models/ and post_proc.py)."
)


def _resolve_hovernet_root(config: dict) -> "Path | None":
    """Return the hover_net source root, or None if not configured."""
    root = os.environ.get("HOVERNET_ROOT") or config.get("hovernet_root", None)
    return Path(os.path.expanduser(root)) if root else None


def _ensure_hovernet_path(hovernet_root: "Path | None"):
    if hovernet_root is None:
        raise ImportError(_HOVERNET_MISSING)
    for p in [str(hovernet_root), str(hovernet_root / "models")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Standalone HV-map generation (no hover_net import required)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _forced_min_size(min_size: int):
    """Force HoverNet's post-processing size filters to the dataset's min_size.

    ``__proc_np_hv`` hardcodes ``remove_small_objects(..., min_size=10)`` twice — once
    on the foreground blobs and once on the watershed markers.  Both are literals, not
    parameters, so the only way to reach them without editing the vendored hover_net
    source is to swap the function ``post_proc`` resolved at import time.  HoverNet's
    own implementation (misc/utils.py, a copy of skimage's) drops components with
    ``size < min_size``, so min_size=1 is an exact no-op.

    Note this is the narrower of the two patches in this codebase: HoverNet imports the
    function into its own module namespace, so only ``post_proc.remove_small_objects``
    is touched, never skimage itself.  Restored in ``finally``.

    Callers have already run _ensure_hovernet_path() and imported process(), so the
    hovernet package is importable by the time this runs.
    """
    from hovernet import post_proc

    original = post_proc.remove_small_objects
    forced = min_size          # captured under a different name — see below

    # The shim's second parameter MUST be named `min_size`: __proc_np_hv calls
    # `remove_small_objects(blb, min_size=10)` with a KEYWORD argument, so the name
    # is what swallows the hardcoded 10.  Renaming it raises
    # `TypeError: unexpected keyword argument 'min_size'`.
    def _forced(pred, min_size=64, connectivity=1):
        return original(pred, forced, connectivity)

    post_proc.remove_small_objects = _forced
    try:
        yield
    finally:
        post_proc.remove_small_objects = original


def _gen_hv_map(mask: np.ndarray) -> np.ndarray:
    """HoverNet HV distance-map target — the shared faithful port of the reference
    `targets.py::gen_instance_hv_map` (int-rounded centroid on a 1-based grid +
    independent half-normalisation).  Delegates to
    `sapi/models/base.py::gen_instance_hv_map`, which HoVer-Net and CellViT
    both use so the convention cannot drift.  Returns (H, W, 2): [horizontal, vertical].
    """
    from .base import gen_instance_hv_map
    return gen_instance_hv_map(mask)



# ---------------------------------------------------------------------------
# Loss functions
# Ported from third_party/hover_net-master/models/hovernet/utils.py and
# opt.py.  All functions expect NHWC float32 tensors (matching the original
# run_desc.py convention after permuting model output from NCHW).
#
# Per opt.py the combined loss is:
#   NP branch : xentropy (w=1) + dice  (w=1)
#   HV branch : mse      (w=1) + msge  (w=1)
# finetune() below deviates from opt.py and applies 2.0 * _msge_loss(...).
# ---------------------------------------------------------------------------

def _xentropy_loss(true: "torch.Tensor", pred: "torch.Tensor") -> "torch.Tensor":
    """Cross-entropy for NP branch.
    true: one-hot  (N,H,W,2) float32;  pred: softmax probs (N,H,W,2) float32."""
    import torch
    epsilon = 1e-8
    pred = pred / (pred.sum(-1, keepdim=True) + epsilon)
    pred = pred.clamp(epsilon, 1.0 - epsilon)
    return -(true * torch.log(pred)).sum(-1).mean()


def _dice_loss(true: "torch.Tensor", pred: "torch.Tensor", smooth: float = 1e-3) -> "torch.Tensor":
    """Dice loss for NP branch.  (N,H,W,C) float32, sum over C."""
    inse = (pred * true).sum((0, 1, 2))
    l    = pred.sum((0, 1, 2))
    r    = true.sum((0, 1, 2))
    return (1.0 - (2.0 * inse + smooth) / (l + r + smooth)).sum()


def _mse_loss(true: "torch.Tensor", pred: "torch.Tensor") -> "torch.Tensor":
    """Mean squared error for HV branch.  (N,H,W,2) float32."""
    diff = pred - true
    return (diff * diff).mean()


def _msge_loss(
    true: "torch.Tensor",
    pred: "torch.Tensor",
    focus: "torch.Tensor",
    device: "torch.device",
) -> "torch.Tensor":
    """Mean squared error of Sobel gradients of HV maps, restricted to nuclei.

    Penalises errors in the spatial gradient of the HV prediction only inside
    nucleus pixels (focus mask).  This is the term that sharpens the HV
    discontinuity at touching cell boundaries.

    Args:
        true:   (N,H,W,2) ground-truth HV maps, NHWC float32.
        pred:   (N,H,W,2) predicted  HV maps, NHWC float32.
        focus:  (N,H,W)   binary nucleus mask (foreground channel of one-hot NP).
        device: torch device used for Sobel kernel tensors.
    """
    import torch
    import torch.nn.functional as F

    def _sobel_kernels(device):
        r = torch.arange(-2, 3, dtype=torch.float32, device=device)
        h, v = torch.meshgrid(r, r, indexing="ij")
        d = h * h + v * v + 1e-15
        return (h / d).view(1, 1, 5, 5), (v / d).view(1, 1, 5, 5)

    def _grad_hv(hv):
        kh, kv = _sobel_kernels(device)
        h_ch = hv[..., 0].unsqueeze(1)   # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)
        dhv = torch.cat([
            F.conv2d(h_ch, kh, padding=2),
            F.conv2d(v_ch, kv, padding=2),
        ], dim=1).permute(0, 2, 3, 1).contiguous()   # back to NHWC
        return dhv

    focus2 = focus.unsqueeze(-1).expand_as(pred)   # (N,H,W,2)
    diff   = _grad_hv(pred) - _grad_hv(true)
    loss   = focus2 * (diff * diff)
    return loss.sum() / (focus2.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _prepare_patching(img_hw3: np.ndarray):
    """Pad image and compute patch coordinates.

    Returns (padded_img, coord_list) where coord_list is a list of (row, col)
    top-left coordinates in the padded image for each patch.
    """
    h, w = img_hw3.shape[:2]
    step = _MSK_SIZE

    def _last_step(length):
        n = math.ceil((length - _MSK_SIZE) / step)
        return int((n + 1) * step)

    last_h = _last_step(h)
    last_w = _last_step(w)

    padt = _BORDER
    padl = _BORDER
    padb = last_h + _WIN_SIZE - h
    padr = last_w + _WIN_SIZE - w

    padded = np.pad(img_hw3, ((padt, padb), (padl, padr), (0, 0)), mode="reflect")

    coords = []
    for r in range(0, last_h, step):
        for c in range(0, last_w, step):
            coords.append((r, c))
    return padded, coords, h, w


_HOVERNET_AUG = None


def _build_hovernet_aug():
    """HoverNet's native augmentation (its imgaug pipeline reproduced in albumentations).
    Source: hover_net dataloader/train_loader.py __get_augmentation + augs.py.
      shape: Affine(scale 0.8-1.2, translate +/-1%, shear +/-5, rotate +/-179) + h/v flip
      input: OneOf(GaussianBlur, MedianBlur, GaussNoise) + ColorJitter(hue/sat/bright/contrast)."""
    import albumentations as A
    import cv2
    return A.Compose([
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.01, 0.01), shear=(-5, 5),
                 rotate=(-179, 179), interpolation=cv2.INTER_NEAREST,
                 mask_interpolation=cv2.INTER_NEAREST, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(std_range=(0.0, 0.05), p=1.0),  # albumentations 2.x API (var_limit → std_range)
        ], p=1.0),
        A.ColorJitter(brightness=26 / 255, contrast=0.25, saturation=0.2, hue=8 / 180, p=1.0),
    ])


def _hovernet_augment(img: np.ndarray, mask: np.ndarray):
    """Apply HoverNet's native augmentation to a [0,255] RGB image + mask."""
    global _HOVERNET_AUG
    if _HOVERNET_AUG is None:
        _HOVERNET_AUG = _build_hovernet_aug()
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    out = _HOVERNET_AUG(image=u8, mask=mask.astype(np.uint16))
    return out["image"].astype(np.float32), out["mask"].astype(np.uint16)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class HoverNetModel(BaseSegmentationModel):
    """HoverNet wrapper (H&E only, original mode, no type prediction)."""

    supports_finetune = True
    num_channels = 3           # HoverNet was trained on full RGB H&E
    needs_raw_he_image = True  # skip stain separation; pass raw RGB to the model

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.min_size = int(config.get("min_size", 1))
        self.model_name = config.get("variant", "hovernet")
        self._device = device
        self._hovernet_root = _resolve_hovernet_root(config)
        # NP-branch probability threshold used by __proc_np_hv's `blb_raw >= t`
        # gate. Defaults to the reference's own hardcoded 0.5; finetune()'s
        # threshold calibration (see below) updates this so predict_single /
        # predict_batch apply the calibrated value, not just the pretrained
        # default, on the fine-tuned model.
        self.np_threshold = 0.5
        env_path = os.environ.get("HOVERNET_WEIGHTS", "")
        raw = env_path or config.get("weights_path", None)
        self.weights_path = os.path.expanduser(raw) if raw else None

        # No pixel-size rescaling: images are processed at their original
        # resolution and size. Neither the Kumar/MoNuSeg paper nor the
        # hover_net reference repo states a training pixel size, and the
        # original CellBinDB paper's own HoverNet integration also applies no
        # rescaling.

    def _ensure_loaded(self):
        if self._model is not None:
            return
        _ensure_hovernet_path(self._hovernet_root)
        import torch
        from hovernet.net_desc import HoVerNet

        net = HoVerNet(input_ch=3, nr_types=None, mode="original")

        if self.weights_path and Path(self.weights_path).exists():
            state = torch.load(self.weights_path, map_location="cpu")
            if "desc" in state:
                state = state["desc"]
            net.load_state_dict(state, strict=True)
            logger.info(f"Loaded HoverNet weights from {self.weights_path}")
        else:
            logger.warning(
                "HoverNet pretrained weights not found. "
                "Set weights_path in configs/models/hovernet.yaml or HOVERNET_WEIGHTS env var."
            )

        net = net.to(self._device)
        net.eval()
        self._model = net

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _to_hw3(self, image: np.ndarray) -> np.ndarray:
        """Return (H, W, 3) float32 in [0,255]; HoverNet's forward() divides by 255.

        H&E is received as raw RGB [0,255] (needs_raw_he_image=True); a single
        channel is replicated to 3.
        """
        img = image.astype(np.float32)
        if img.ndim == 2:
            return np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] == 1:
            return np.concatenate([img, img, img], axis=2)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return np.stack([img[..., 0]] * 3, axis=-1)

    # ------------------------------------------------------------------
    # Tiled inference
    # ------------------------------------------------------------------

    def _apply_np_threshold(self, pred_map: np.ndarray) -> np.ndarray:
        """Shift pred_map's NP channel so __proc_np_hv's hardcoded `>= 0.5`
        gate effectively tests `>= self.np_threshold` (see finetune()'s
        threshold-calibration comment for the derivation). No-op at the
        default threshold of 0.5, so zero-shot inference is unaffected.
        """
        if self.np_threshold == 0.5:
            return pred_map
        pred_map = pred_map.copy()
        pred_map[..., 0] = pred_map[..., 0] + (0.5 - self.np_threshold)
        return pred_map

    def _infer_image(self, img_hw3_f: np.ndarray) -> np.ndarray:
        """Run tile-based inference. Returns instance mask (H, W)."""
        import torch
        _ensure_hovernet_path(self._hovernet_root)
        from hovernet.post_proc import process

        padded, coords, orig_h, orig_w = _prepare_patching(img_hw3_f)
        dev = next(self._model.parameters()).device

        # Output accumulator: store (np_prob, hv_x, hv_y) per pixel
        out_np = np.zeros((orig_h, orig_w), dtype=np.float32)
        out_hv = np.zeros((orig_h, orig_w, 2), dtype=np.float32)

        patches, patch_coords = [], []
        for r, c in coords:
            patch = padded[r: r + _WIN_SIZE, c: c + _WIN_SIZE, :]   # (270, 270, 3)
            patches.append(patch)
            patch_coords.append((r, c))

            if len(patches) >= 8:
                self._process_batch(patches, patch_coords, out_np, out_hv, dev, orig_h, orig_w)
                patches, patch_coords = [], []

        if patches:
            self._process_batch(patches, patch_coords, out_np, out_hv, dev, orig_h, orig_w)

        # Build pred_map: (H, W, 3) = [np_prob, hv_x, hv_y]
        pred_map = np.stack([out_np, out_hv[..., 0], out_hv[..., 1]], axis=-1)
        pred_map = self._apply_np_threshold(pred_map)

        with _forced_min_size(self.min_size):
            inst_mask, _ = process(pred_map, nr_types=None, return_centroids=False)
        return inst_mask.astype(np.uint16)

    # NOTE — of __proc_np_hv's two hardcoded thresholds, only `blb_raw >= 0.5`
    # is calibrated in this benchmark (via finetune()'s np_threshold search
    # and _apply_np_threshold's channel-shift trick, applied above). The
    # `overall >= 0.4` watershed-elevation threshold is deliberately NOT
    # calibrated, because shifting it would corrupt the gradient-based
    # instance separation rather than just moving a probability cutoff — see
    # finetune()'s threshold-calibration comment.

    def _process_batch(self, patches, patch_coords, out_np, out_hv, dev, orig_h, orig_w):
        import torch

        batch = np.stack(patches, axis=0).transpose(0, 3, 1, 2)    # (B, 3, 270, 270)
        batch_t = torch.tensor(batch, dtype=torch.float32).to(dev)

        with torch.no_grad():
            out_dict = self._model(batch_t)

        np_out = out_dict["np"].cpu().numpy()    # (B, 2, 80, 80) — softmax logits
        hv_out = out_dict["hv"].cpu().numpy()    # (B, 2, 80, 80)

        # Softmax over NP channels → channel 1 = nucleus prob
        np_exp = np.exp(np_out - np_out.max(axis=1, keepdims=True))
        np_prob = np_exp[:, 1] / (np_exp.sum(axis=1) + 1e-8)   # (B, 80, 80)

        for b, (r, c) in enumerate(patch_coords):
            # Map patch coordinates back to original image
            r0 = r
            c0 = c
            r1 = min(r0 + _MSK_SIZE, orig_h)
            c1 = min(c0 + _MSK_SIZE, orig_w)
            ph = r1 - r0
            pw = c1 - c0
            if ph <= 0 or pw <= 0:
                continue
            out_np[r0:r1, c0:c1] = np_prob[b, :ph, :pw]
            out_hv[r0:r1, c0:c1, 0] = hv_out[b, 0, :ph, :pw]
            out_hv[r0:r1, c0:c1, 1] = hv_out[b, 1, :ph, :pw]

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        img3 = self._to_hw3(image)
        return self._infer_image(img3)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Batch inference: collect patches from up to batch_size images, process in GPU sub-batches."""
        self._ensure_loaded()
        _ensure_hovernet_path(self._hovernet_root)
        import torch
        from hovernet.post_proc import process

        dev = next(self._model.parameters()).device
        results = []

        for batch_start in range(0, len(images), batch_size):
            chunk = images[batch_start:batch_start + batch_size]
            B = len(chunk)

            img_hw3_list = [self._to_hw3(img) for img in chunk]
            padded_list, coords_list, orig_shapes = [], [], []
            for img_hw3 in img_hw3_list:
                padded, coords, orig_h, orig_w = _prepare_patching(img_hw3)
                padded_list.append(padded)
                coords_list.append(coords)
                orig_shapes.append((orig_h, orig_w))

            out_nps = [np.zeros(s,           dtype=np.float32) for s in orig_shapes]
            out_hvs  = [np.zeros((s[0], s[1], 2), dtype=np.float32) for s in orig_shapes]

            # Collect all (image_idx, r, c, patch) tuples from all images
            all_meta, all_patches = [], []
            for b in range(B):
                for r, c in coords_list[b]:
                    all_meta.append((b, r, c))
                    all_patches.append(padded_list[b][r:r + _WIN_SIZE, c:c + _WIN_SIZE, :])

            # Process all patches in GPU sub-batches of batch_size
            for i in range(0, len(all_patches), batch_size):
                sub_patches = all_patches[i:i + batch_size]
                sub_meta    = all_meta[i:i + batch_size]

                batch_arr = np.stack(sub_patches, axis=0).transpose(0, 3, 1, 2)  # (N, 3, 270, 270)
                batch_t   = torch.tensor(batch_arr, dtype=torch.float32).to(dev)

                with torch.no_grad():
                    out_dict = self._model(batch_t)

                np_out  = out_dict["np"].cpu().numpy()    # (N, 2, 80, 80)
                hv_out  = out_dict["hv"].cpu().numpy()    # (N, 2, 80, 80)
                np_exp  = np.exp(np_out - np_out.max(axis=1, keepdims=True))
                np_prob = np_exp[:, 1] / (np_exp.sum(axis=1) + 1e-8)  # (N, 80, 80)

                for j, (b, r, c) in enumerate(sub_meta):
                    orig_h, orig_w = orig_shapes[b]
                    r1 = min(r + _MSK_SIZE, orig_h)
                    c1 = min(c + _MSK_SIZE, orig_w)
                    ph, pw = r1 - r, c1 - c
                    if ph <= 0 or pw <= 0:
                        continue
                    out_nps[b][r:r1, c:c1]       = np_prob[j, :ph, :pw]
                    out_hvs[b][r:r1, c:c1, 0]    = hv_out[j, 0, :ph, :pw]
                    out_hvs[b][r:r1, c:c1, 1]    = hv_out[j, 1, :ph, :pw]

            for b in range(B):
                pred_map = np.stack(
                    [out_nps[b], out_hvs[b][..., 0], out_hvs[b][..., 1]], axis=-1
                )
                pred_map = self._apply_np_threshold(pred_map)
                with _forced_min_size(self.min_size):
                    inst_mask, _ = process(pred_map, nr_types=None, return_centroids=False)
                results.append(inst_mask.astype(np.uint16))

        return results

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        """Fine-tune HoverNet with a single-stage loop.

        Two modes selected by the `freeze_encoder` kwarg (fewshot_finetuning
        runs both, one after the other, from the single
        `configs/models/hovernet.yaml` file — there is no separate
        "hovernet_frozen" model or config):
            False — entire model fine-tuned for n_epochs
            True  — encoder frozen, branch decoders only  (YAML default)

        LR schedule: StepLR step_size=25, gamma=0.1
            → LR drops from lr to lr×0.1 after epoch 25.

        Loss per paper Section 3.2 (opt.py itself weights msge at 1, not 2):
            NP branch : xentropy (w=1) + dice  (w=1)
            HV branch : mse      (w=1) + msge  (w=2)
        """
        import torch
        import torch.nn.functional as F
        import torch.optim as optim
        from .base import split_into_patches, calibrate_thresholds_on_support

        self._ensure_loaded()
        _ensure_hovernet_path(self._hovernet_root)

        n_epochs       = kwargs.get("n_epochs", 50)
        lr             = kwargs.get("learning_rate", 1e-4)
        wd             = kwargs.get("weight_decay", 1e-4)
        batch_size     = int(kwargs.get("batch_size", 8))
        freeze_encoder = kwargs.get("freeze_encoder", True)
        val_images     = list(kwargs.get("val_images", []))
        val_masks      = list(kwargs.get("val_masks", []))
        patience       = int(kwargs.get("patience", 20))

        # ── Split into 270×270 patches ──────────────────────────────────────
        # No pixel-size rescaling: images are split at their native/original
        # size, same as inference — _prepare_patching() also tiles the
        # native-size image directly.
        #   HE:              512×512 → _positions(512, 270) = [0, 242] →
        #                    2×2 = 4 patches per support image.
        #   10xGenomics_HE:  256×256 is smaller than the 270×270 patch, so
        #                    split_into_patches() centre-pads it to a single
        #                    patch per support image.
        n_support_images = len(support_images)   # original count, before patch expansion
        patched_images: List[np.ndarray] = []
        patched_masks:  List[np.ndarray] = []
        for img, msk in zip(support_images, support_masks):
            for img_p, msk_p in split_into_patches(img, msk, _WIN_SIZE):
                patched_images.append(img_p)
                patched_masks.append(msk_p)
        support_images = patched_images
        support_masks  = patched_masks

        n_shots        = len(support_images)   # patch count, used for batch iteration
        dev            = next(self._model.parameters()).device
        train_losses: List[float] = []

        # ── Freeze/unfreeze based on architecture ─────────────────────────
        # HoverNet architecture (net_desc.py):
        #   Encoder: conv0, d0, d1, d2, d3, conv_bot (ResNet50-style blocks)
        #   Decoder: self.decoder (ModuleDict with 'np' and 'hv' branches)
        # Source: third_party/hover_net-master/models/hovernet/net_desc.py
        _ENCODER_ATTRS = ("conv0", "d0", "d1", "d2", "d3", "conv_bot")

        if freeze_encoder:
            # Freeze all encoder attributes that exist on the model
            n_frozen = 0
            for attr in _ENCODER_ATTRS:
                module = getattr(self._model, attr, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = False
                        n_frozen += p.numel()
            # Ensure decoder branches remain trainable
            if hasattr(self._model, "decoder"):
                for p in self._model.decoder.parameters():
                    p.requires_grad = True
            n_total = sum(p.numel() for p in self._model.parameters())
            trainable = [p for p in self._model.parameters() if p.requires_grad]
            logger.info(
                f"HoverNet finetune (decoder-only): encoder frozen "
                f"({n_frozen:,}/{n_total:,} params).  "
                f"LR={lr:.1e}, n_epochs={n_epochs}"
            )
        else:
            for p in self._model.parameters():
                p.requires_grad = True
            trainable = list(self._model.parameters())
            logger.info(
                f"HoverNet finetune (full model): all parameters trainable.  "
                f"LR={lr:.1e}, n_epochs={n_epochs}"
            )

        optimizer = optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # captured while the freeze flags are active (restored before return).
        _n_trainable_params = int(sum(p.numel() for p in trainable))
        _n_frozen_params    = int(sum(p.numel() for p in self._model.parameters())) - _n_trainable_params

        # ── Pre-process val images ─────────────────────────────────────────
        # Split each val image into _WIN_SIZE×_WIN_SIZE patches (same as
        # training) so the val loss covers the full image.  For 512×512 HE
        # val images this produces 4 patches (100%) instead of one center
        # 270×270 crop (28%).
        def _hovernet_val_tensors(images, masks):
            tensors = []
            for img, msk in zip(images, masks):
                img = img.astype(np.float32)
                msk = msk.astype(np.uint16)
                for ci, cm in split_into_patches(img, msk, _WIN_SIZE):
                    img3 = self._to_hw3(ci)
                    np_bin = (cm > 0).astype(np.float32)
                    hv     = _gen_hv_map(cm.astype(np.int32))
                    np_c   = np_bin[_BORDER:_BORDER+_MSK_SIZE, _BORDER:_BORDER+_MSK_SIZE]
                    hv_c   = hv[_BORDER:_BORDER+_MSK_SIZE, _BORDER:_BORDER+_MSK_SIZE]
                    img_t  = torch.tensor(np.ascontiguousarray(img3.transpose(2, 0, 1)),
                                          dtype=torch.float32).unsqueeze(0).to(dev)
                    np_t   = F.one_hot(
                        torch.tensor(np_c.astype(np.int64), dtype=torch.int64).to(dev),
                        num_classes=2).float().unsqueeze(0)
                    hv_t   = torch.tensor(hv_c, dtype=torch.float32).unsqueeze(0).to(dev)
                    tensors.append((img_t, np_t, hv_t))
            return tensors

        val_tensors = _hovernet_val_tensors(val_images, val_masks) if val_images else []

        from .base import _EarlyStopper
        stopper    = _EarlyStopper(patience)
        val_losses: List[float] = []
        best_epoch = 0

        # ── Training loop ─────────────────────────────────────────────────
        # With freeze_encoder=True, keep the encoder's BatchNorm layers in
        # eval mode so their running stats do not drift on the tiny support
        # set (same fix as MicroSAM TinyViT, an earlier revision Bug B).
        def _set_train_mode():
            self._model.train()
            if freeze_encoder:
                for attr in _ENCODER_ATTRS:
                    module = getattr(self._model, attr, None)
                    if module is not None:
                        module.eval()

        _set_train_mode()
        for epoch in range(n_epochs):
            indices    = np.random.permutation(n_shots)
            epoch_loss = 0.0
            n_steps    = 0

            for b_start in range(0, n_shots, batch_size):
                batch_idx = indices[b_start:b_start + batch_size]

                imgs_t_list: List["torch.Tensor"] = []
                true_np_list: List["torch.Tensor"] = []
                true_hv_list: List["torch.Tensor"] = []

                for idx in batch_idx:
                    # Patches are _WIN_SIZE×_WIN_SIZE; HoverNet-style augmentation per step.
                    aug_img, aug_mask = _hovernet_augment(
                        support_images[idx], support_masks[idx]
                    )
                    img3 = self._to_hw3(aug_img)

                    np_binary = (aug_mask > 0).astype(np.float32)
                    hv_map    = _gen_hv_map(aug_mask.astype(np.int32))
                    np_crop   = np_binary[_BORDER:_BORDER + _MSK_SIZE,
                                          _BORDER:_BORDER + _MSK_SIZE]
                    hv_crop   = hv_map[_BORDER:_BORDER + _MSK_SIZE,
                                       _BORDER:_BORDER + _MSK_SIZE]

                    imgs_t_list.append(
                        torch.tensor(np.ascontiguousarray(img3.transpose(2, 0, 1)),
                                     dtype=torch.float32)
                    )
                    np_int = torch.tensor(np_crop.astype(np.int64), dtype=torch.int64)
                    true_np_list.append(F.one_hot(np_int, num_classes=2).float())
                    true_hv_list.append(
                        torch.tensor(hv_crop, dtype=torch.float32)
                    )

                # Stack into batch tensors and move to device in one call
                imgs_batch   = torch.stack(imgs_t_list).to(dev)          # (B,3,270,270)
                # NHWC convention expected by loss functions
                true_np_batch = torch.stack(true_np_list).to(dev)        # (B,80,80,2)
                true_hv_batch = torch.stack(true_hv_list).to(dev)        # (B,80,80,2)

                optimizer.zero_grad()
                out_dict = self._model(imgs_batch)
                # Permute NCHW → NHWC (matches original run_desc.py convention)
                pred_np = F.softmax(
                    out_dict["np"].permute(0, 2, 3, 1).contiguous(), dim=-1
                )                                                          # (B,H,W,2)
                pred_hv = out_dict["hv"].permute(0, 2, 3, 1).contiguous() # (B,H,W,2)

                # Loss per paper Section 3.2: msge weighted by 2 (all others w=1)
                loss = (
                    _xentropy_loss(true_np_batch, pred_np)
                    + _dice_loss(true_np_batch, pred_np)
                    + _mse_loss(true_hv_batch, pred_hv)
                    + 2.0 * _msge_loss(true_hv_batch, pred_hv,
                                       focus=true_np_batch[..., 1], device=dev)
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

            avg = epoch_loss / max(n_steps, 1)
            train_losses.append(avg)
            scheduler.step()

            ep = epoch + 1
            if ep % 10 == 0 or ep == 1:
                logger.info(
                    f"  HoverNet epoch {ep}/{n_epochs}  "
                    f"loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # ── Early stopping ─────────────────────────────────────────────
            if val_tensors and patience > 0:
                self._model.eval()
                with torch.no_grad():
                    vl = 0.0
                    _VAL_BATCH = 8
                    for b_start in range(0, len(val_tensors), _VAL_BATCH):
                        sub    = val_tensors[b_start:b_start + _VAL_BATCH]
                        imgs_b = torch.cat([t[0] for t in sub], dim=0)  # (B,3,270,270)
                        np_b   = torch.cat([t[1] for t in sub], dim=0)  # (B,80,80,2)
                        hv_b   = torch.cat([t[2] for t in sub], dim=0)  # (B,80,80,2)
                        out    = self._model(imgs_b)
                        p_np   = F.softmax(
                            out["np"].permute(0, 2, 3, 1).contiguous(), dim=-1
                        )
                        p_hv   = out["hv"].permute(0, 2, 3, 1).contiguous()
                        for i in range(len(sub)):
                            vl += (
                                _xentropy_loss(np_b[i:i+1], p_np[i:i+1])
                                + _dice_loss(np_b[i:i+1], p_np[i:i+1])
                                + _mse_loss(hv_b[i:i+1], p_hv[i:i+1])
                                + 2.0 * _msge_loss(
                                    hv_b[i:i+1], p_hv[i:i+1],
                                    focus=np_b[i:i+1, ..., 1], device=dev,
                                )
                            ).item()
                    vl /= len(val_tensors)
                _set_train_mode()
                val_losses.append(vl)
                if stopper.step(vl, self._model.state_dict()):
                    logger.info(
                        f"  HoverNet early stop at epoch {ep}  "
                        f"best_val={stopper.best:.4f} (epoch {best_epoch})"
                    )
                    break
                if stopper.counter == 0:
                    best_epoch = ep
            else:
                best_epoch = ep

        stopper.restore_best(self._model)

        if freeze_encoder:
            for p in self._model.parameters():
                p.requires_grad = True
            logger.info("HoverNet encoder unfrozen")

        self._model.eval()

        # ── Calibrate post-processing thresholds on support set ────────────
        # HoverNet's np_threshold (nucleus probability) can be calibrated via
        # input shifting: __proc_np_hv compares blb_raw >= 0.5. To test
        # threshold t, we shift the np channel: pred_map[..., 0] + (0.5 - t).
        # This gives: (blb_raw + shift) >= 0.5 ⟺ blb_raw >= t.
        # The overall >= 0.4 threshold is deliberately NOT calibrated
        # because it feeds into the watershed elevation map, where shifting
        # would corrupt the gradient-based instance separation.
        threshold_calibration = {}
        if n_support_images > 0:
            def _predict_with_np_threshold(img, np_threshold=0.5):
                """Predict with custom np_threshold via pred_map channel shifting."""
                _ensure_hovernet_path(self._hovernet_root)
                from hovernet.post_proc import process

                img3 = self._to_hw3(img)
                padded, coords, orig_h, orig_w = _prepare_patching(img3)
                dev = next(self._model.parameters()).device

                out_np = np.zeros((orig_h, orig_w), dtype=np.float32)
                out_hv = np.zeros((orig_h, orig_w, 2), dtype=np.float32)

                patches, patch_coords = [], []
                for r, c in coords:
                    patch = padded[r: r + _WIN_SIZE, c: c + _WIN_SIZE, :]
                    patches.append(patch)
                    patch_coords.append((r, c))

                    if len(patches) >= 8:
                        self._process_batch(patches, patch_coords, out_np, out_hv, dev, orig_h, orig_w)
                        patches, patch_coords = [], []

                if patches:
                    self._process_batch(patches, patch_coords, out_np, out_hv, dev, orig_h, orig_w)

                # Build pred_map and apply threshold shift
                pred_map = np.stack([out_np, out_hv[..., 0], out_hv[..., 1]], axis=-1)
                shift = 0.5 - np_threshold
                pred_map[..., 0] = pred_map[..., 0] + shift

                with _forced_min_size(self.min_size):
                    inst_mask, _ = process(pred_map, nr_types=None, return_centroids=False)
                return inst_mask.astype(np.uint16)

            threshold_calibration = calibrate_thresholds_on_support(
                _predict_with_np_threshold,
                support_images,
                support_masks,
                grid={"np_threshold": [0.5, 0.4, 0.6]},
                model_name="hovernet"
            )
            # Persist the calibrated threshold so predict_single / predict_batch
            # apply it to post-finetune predictions.
            self.np_threshold = float(threshold_calibration.get("np_threshold", 0.5))

        return {
            "n_epochs_run":        len(train_losses),
            "n_epochs_max":        n_epochs,
            "best_epoch":          best_epoch,
            "early_stopping_mode": "patience_restore_best_weights",
            "n_train_images":      n_support_images,
            "n_train_patches":     n_shots,   # support images tiled to _WIN_SIZE (one pass/epoch)
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
            "train_losses":        train_losses,
            "val_losses":          val_losses,
            "freeze_encoder":      freeze_encoder,
            "threshold_calibration": threshold_calibration,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_param_count(self) -> int:
        try:
            self._ensure_loaded()
            return sum(p.numel() for p in self._model.parameters())
        except Exception as exc:
            logger.warning(f"Failed to get parameter count for {self.model_name}: {exc}")
            return -1

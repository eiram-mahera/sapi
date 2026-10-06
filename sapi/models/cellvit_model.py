"""
CellViT model wrapper.

CellViT is a vision-transformer-based cell instance segmentation model
published in: Hörst et al., CellViT (2023).

Repository: https://github.com/TIO-IKIM/CellViT
            (clone and install before using this wrapper)

Variants with available checkpoints (authors provide only these two at 40×):
    cellvit_256     - CellViT-256  (ViT-256 / ViT-Small/16 DINO backbone, ~28.7 M params)
    cellvit_sam_h   - CellViT-SAM-H (SAM ViT-H backbone, ~632 M params)

Note: cellvit_sam_b and cellvit_sam_l are NOT registered — no checkpoints provided by authors.

Setup:
    pip install cellvit          # install from PyPI / the official GitHub release
    # Download model weights and set 'weight_path' in configs/models/cellvit_*.yaml,
    # or override with the CELLVIT_WEIGHTS environment variable.

Note: CellViT was designed for H&E histology patches (256×256, 0.25 µm/px).
      For other modalities a colour-normalisation step is applied.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)

# Sobel kernel size for the HoVer post-processing.  CellViT's
# DetectionCellPostProcessor picks k_size from the data magnification: 21 at 40×,
# 11 at 20×.  This benchmark loads the x20 checkpoints (see _WEIGHT_URLS), so 11
# is the reference value.  Switching to x40 checkpoints means using 21.
_CELLVIT_KSIZE = 11


def _proc_np_hv(pred: np.ndarray, min_size: int = 1, ksize: int = _CELLVIT_KSIZE) -> np.ndarray:
    """Separate nuclei instances from the binary + HV maps (CellViT reference).

    Faithful port of CellViT's own post-processing
    (`cell_segmentation/utils/post_proc_cellvit.py::DetectionCellPostProcessor.__proc_np_hv`,
    itself adapted from HoVer-Net).  ``pred`` is (H, W, 3):
        channel 0 = binary nuclei foreground (0/1, argmax of the 2-class head),
        channel 1 = regressed horizontal map,
        channel 2 = regressed vertical map.

    Watershed markers come from the Sobel *ridges* of the H/V maps
    (``marker = blb - overall``), which is what splits touching nuclei — the
    mechanism CellViT's HV branch exists to provide.  ``min_size`` is the
    benchmark's dataset-level size filter: it replaces the reference's
    hardcoded 10 / object_size literals, so the value is the data's, not the
    model's.  At the current dataset value (1) both filters are no-ops.
    """
    import cv2
    from scipy.ndimage import label as _ndlabel, binary_fill_holes
    from skimage.segmentation import watershed
    from skimage.morphology import remove_small_objects

    pred = np.asarray(pred, dtype=np.float32)
    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    blb = (blb_raw >= 0.5).astype(np.int32)
    blb = _ndlabel(blb)[0]
    blb = remove_small_objects(blb, min_size=min_size)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form mountains, so invert to get basins for the watershed.
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = (overall >= 0.4).astype(np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = _ndlabel(marker)[0]
    marker = remove_small_objects(marker, min_size=min_size)

    return watershed(dist, markers=marker, mask=blb).astype(np.int32)


def _gen_hv_targets(mask: np.ndarray) -> np.ndarray:
    """Generate normalised HV distance maps for CellViT fine-tuning targets.

    Returns float32 (H, W, 2): channel 0 = horizontal, 1 = vertical.  Delegates
    to the shared faithful port of the reference
    (`datasets/pannuke.py::gen_instance_hv_map`, whose per-instance loop is
    byte-identical to HoVer-Net's) in `sapi/models/base.py`: int-rounded
    centroid on a 1-based grid + independent half-normalisation.  Both extremes
    reach ±1 even for off-centre cells — a single symmetric ``/abs().max()`` (an
    even earlier revision) lets only the longer side reach ±1, producing a target
    the reference post-processing's Sobel-ridge watershed does not expect.
    """
    from .base import gen_instance_hv_map
    return gen_instance_hv_map(mask)


def _sobel_msge_loss(
    pred_hv: "torch.Tensor",
    target_hv: "torch.Tensor",
    focus: "torch.Tensor",
    device: str,
) -> "torch.Tensor":
    """MSGE loss with 5×5 Sobel kernels, weighted by foreground focus mask.

    Replicates CellViT base_ml/base_loss.py MSGELossMaps exactly.
    pred_hv, target_hv: (B, 2, H, W)
    focus: (B, H, W) binary foreground mask (0/1 float)
    """
    import torch
    import torch.nn.functional as F

    h_range = torch.arange(-2, 3, dtype=torch.float32, device=device, requires_grad=False)
    v_range = torch.arange(-2, 3, dtype=torch.float32, device=device, requires_grad=False)
    h, v = torch.meshgrid(h_range, v_range, indexing="ij")
    kernel_h = (h / (h * h + v * v + 1e-15)).view(1, 1, 5, 5)
    kernel_v = (v / (h * h + v * v + 1e-15)).view(1, 1, 5, 5)

    def _grad(hv):
        return torch.cat([
            F.conv2d(hv[:, 0:1], kernel_h, padding=2),
            F.conv2d(hv[:, 1:2], kernel_v, padding=2),
        ], dim=1)  # (B, 2, H, W)

    diff = _grad(pred_hv) - _grad(target_hv)  # (B, 2, H, W)
    focus_2ch = focus.unsqueeze(1).expand_as(diff)  # (B, 2, H, W)
    return (focus_2ch * diff * diff).sum() / (focus_2ch.sum() + 1e-8)


def _dice_loss(inp: "torch.Tensor", target: "torch.Tensor", smooth: float = 1e-3) -> "torch.Tensor":
    """CellViT ``base_ml/base_loss.py::DiceLoss`` — verbatim.

    Dice summed over **all** classes (background + foreground), pooled over the
    whole batch, smooth=1e-3.  ``inp`` is the softmax probability map and
    ``target`` the one-hot ground truth, both (N, C, H, W).
    """
    import torch
    inp = inp.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)
    inse = torch.sum(inp * target, (0, 1, 2))
    l = torch.sum(inp, (0, 1, 2))
    r = torch.sum(target, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    return torch.sum(loss)


def _focaltversky_loss(
    inp: "torch.Tensor",
    target: "torch.Tensor",
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma_f: float = 4.0 / 3.0,
    smooth: float = 1e-6,
) -> "torch.Tensor":
    """CellViT ``base_ml/base_loss.py::FocalTverskyLoss`` — verbatim.

    ``inp`` is passed the **softmax'd** nuclei_binary_map (exactly what the
    reference ``unpack_predictions`` produces) and is softmax'd **again** here,
    reproducing the reference's double-softmax.  ``target`` is one-hot (N, C, H, W).
    tp/fp/fn are pooled over the whole batch.
    """
    import torch
    inp = inp.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1).reshape(-1)
    inp = torch.softmax(inp, dim=-1).reshape(-1)
    tp = (inp * target).sum()
    fp = ((1 - target) * inp).sum()
    fn = (target * (1 - inp)).sum()
    tversky = (tp + smooth) / (tp + alpha_t * fn + beta_t * fp + smooth)
    return (1 - tversky) ** gamma_f


def _tile_starts(length: int, patch_size: int, stride: int) -> list:
    """Return starting positions for non-overlapping-edge tiles."""
    if length <= patch_size:
        return [0]
    starts = []
    s = 0
    while s + patch_size <= length:
        starts.append(s)
        s += stride
    last = length - patch_size
    if starts[-1] < last:
        starts.append(last)
    return starts

_WEIGHT_URLS = {
    # This benchmark uses the x20 checkpoints, chosen as the more plausible match for
    # CellBinDB's apparent scale.  CellBinDB does not state its pixel size, so this is
    # a judgment call, not a confirmed match — see configs/models/cellvit_*.yaml.
    # The choice also fixes the precision: every published x20 config omits
    # `training.mixed_precision` (-> False -> fp32), while every x40 config sets it
    # true.  Switching to x40 therefore means switching finetune() to fp16 AMP.
    # x20:
    #   CellViT-256-x20.pth   → https://drive.google.com/uc?export=download&id=1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2
    #   CellViT-SAM-H-x20.pth → https://drive.google.com/uc?export=download&id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30
    # x40:
    #   CellViT-256-x40.pth   → https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q
    #   CellViT-SAM-H-x40.pth → https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV
    "cellvit_256":   None,
    "cellvit_sam_h": None,
}


# CellViT's trained transform: ToTensor (/255) then Normalize(0.5, 0.5) -> [-1, +1].
# Source: logs_paper/PanNuke/CellViTHV/SAM-H/Fold-1/config.yaml and ViT256 config.
_CELLVIT_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_CELLVIT_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _cellvit_normalize(img: np.ndarray) -> np.ndarray:
    """Raw RGB [0,255] -> [-1, 1] via ToTensor(/255) + Normalize(0.5, 0.5)."""
    return ((img.astype(np.float32) / 255.0 - _CELLVIT_MEAN) / _CELLVIT_STD).astype(np.float32)


_CELLVIT_AUG = None


def _build_cellvit_aug():
    """CellViT's native albumentations pipeline with the published PanNuke parameters.
    Source: logs_paper/PanNuke/CellViTHV/.../config.yaml + experiment_cellvit_pannuke.
    get_transforms().  (Normalize is applied separately by _cellvit_normalize.)"""
    import albumentations as A
    # albumentations >= 1.4 renamed several kwargs; handle both old and new API
    try:
        _downscale = A.Downscale(p=0.15, scale_range=(0.5, 0.5))
    except TypeError:
        _downscale = A.Downscale(p=0.15, scale_min=0.5, scale_max=0.5)
    try:
        _rand_crop = A.RandomSizedCrop(min_max_height=(128, 256), size=(256, 256), p=0.1)
    except TypeError:
        _rand_crop = A.RandomSizedCrop(min_max_height=(128, 256), height=256, width=256, p=0.1)
    try:
        _gauss_noise = A.GaussNoise(p=0.25, std_range=(0.0, 0.028))  # var_limit=50 equivalent
    except TypeError:
        _gauss_noise = A.GaussNoise(p=0.25, var_limit=50)
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        _downscale,
        A.Blur(p=0.2, blur_limit=(3, 11)),   # config blur_limit=10; albumentations needs odd
        _gauss_noise,
        A.ColorJitter(p=0.2, brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05),
        A.Superpixels(p=0.1, p_replace=0.1, n_segments=200, max_size=128),
        A.ZoomBlur(p=0.1, max_factor=1.05),
        _rand_crop,
        A.ElasticTransform(p=0.2, sigma=25, alpha=0.5),
    ])


def _cellvit_augment(img: np.ndarray, mask: np.ndarray):
    """Apply CellViT's native albumentations augmentation to a [0,255] RGB image + mask."""
    global _CELLVIT_AUG
    if _CELLVIT_AUG is None:
        _CELLVIT_AUG = _build_cellvit_aug()
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    out = _CELLVIT_AUG(image=u8, mask=mask.astype(np.uint16))
    return out["image"].astype(np.float32), out["mask"].astype(np.uint16)


class CellViTModel(BaseSegmentationModel):
    """CellViT wrapper for cell instance segmentation."""

    supports_finetune = True
    num_channels = 3            # CellViT expects RGB H&E input
    needs_raw_he_image = True   # receive full RGB, not hematoxylin channel

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        variant = config.get("variant", "cellvit_256")
        self.model_name = variant
        self.patch_size = config.get("patch_size", 256)
        self.num_classes = config.get("num_classes", 6)
        # PanNuke (training set) has 19 tissue classes; must match checkpoint.
        self.num_tissue_classes = config.get("num_tissue_classes", 19)
        env_path = os.environ.get("CELLVIT_WEIGHTS", "")
        raw = env_path or config.get("weight_path", None)
        self.weight_path = os.path.expanduser(raw) if raw else None
        config["weight_path"] = self.weight_path  # keep config snapshot consistent with env-var override
        # Dataset-level minimum instance area, applied inside the reference
        # HoVer post-processing.  Default 1 (a no-op).
        self.min_size = int(config.get("min_size", 1))

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch

        variant = self.model_name
        if "sam" in variant:
            backbone_type = variant.replace("cellvit_sam_", "").upper()  # B, L, H
            self._model = self._build_cellvit_sam(backbone_type)
        else:
            self._model = self._build_cellvit_256()

        if self.weight_path and Path(self.weight_path).exists():
            state = torch.load(self.weight_path, map_location=self.device)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            result = self._model.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logger.warning(
                    f"CellViT checkpoint mismatch: missing={result.missing_keys}, "
                    f"unexpected={result.unexpected_keys}"
                )
            logger.info(f"Loaded CellViT weights from {self.weight_path}")
        else:
            logger.warning(
                f"CellViT pretrained weights not found at {self.weight_path!r}; "
                "using random init."
            )

        self._model.to(self.device)
        self._model.eval()

    def _build_cellvit_256(self):
        # Both CellViT256 and CellViTSAM live in cellvit.py, not cellvit_sam.py.
        # model256_path is stored by the constructor but not used during __init__;
        # the full model weights are loaded separately via load_state_dict().
        from cellvit.models.segmentation.cell_segmentation.cellvit import CellViT256
        return CellViT256(
            model256_path=None,
            num_nuclei_classes=self.num_classes,
            num_tissue_classes=self.num_tissue_classes,
        )

    def _build_cellvit_sam(self, backbone: str = "H"):
        # CellViTSAM is defined in cellvit.py alongside CellViT256.
        # vit_structure must be "SAM-B", "SAM-L", or "SAM-H".
        # model_path is stored by the constructor but not used during __init__;
        # the full model weights are loaded separately via load_state_dict().
        from cellvit.models.segmentation.cell_segmentation.cellvit import CellViTSAM
        return CellViTSAM(
            model_path=None,
            num_nuclei_classes=self.num_classes,
            num_tissue_classes=self.num_tissue_classes,
            vit_structure=f"SAM-{backbone.upper()}",
        )

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Tiled inference at patch_size×patch_size with 50% overlap.

        Predictions from overlapping tiles are averaged in raw (unsoftmaxed)
        logit space before post-processing, eliminating boundary artefacts.

        Precision: fp32, matching CellViT's own inference.  cell_detection.py builds
        run_conf by unflattening the checkpoint's bundled `config` dict and then reads
        `run_conf["training"].get("mixed_precision", False)`.  The released checkpoints
        carry no `training.mixed_precision` key, so that resolves to False and the
        reference infers in fp32.  Do not add autocast here.
        """
        self._ensure_loaded()
        import torch

        img = self.preprocess(image)   # → float32 (H, W, 3)
        H, W = img.shape[:2]
        ps = self.patch_size  # 256
        stride = ps // 2       # 128 = 50% overlap

        ys = _tile_starts(H, ps, stride)
        xs = _tile_starts(W, ps, stride)

        hv_acc: np.ndarray | None = None
        bin_acc: np.ndarray | None = None
        weight_acc = np.zeros((H, W), dtype=np.float32)

        for y in ys:
            for x in xs:
                ty = min(y + ps, H)
                tx = min(x + ps, W)
                tile = img[y:ty, x:tx]
                th, tw = tile.shape[:2]

                # Zero-pad to ps×ps for edge tiles
                if th < ps or tw < ps:
                    pad = np.zeros((ps, ps, 3), dtype=np.float32)
                    pad[:th, :tw] = tile
                    tile = pad

                tile_f = _cellvit_normalize(tile)
                tensor = torch.from_numpy(tile_f).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self._model(tensor)

                if not isinstance(out, dict):
                    continue
                hv = out.get("hv_map", out.get("hv", None))
                binary = out.get("nuclei_binary_map", out.get("binary_map", None))
                if hv is None or binary is None:
                    continue

                hv_np = hv[0].cpu().float().numpy()       # (2, ps, ps)
                bin_np = binary[0].cpu().float().numpy()  # (C, ps, ps)

                if hv_acc is None:
                    hv_acc = np.zeros((hv_np.shape[0], H, W), dtype=np.float32)
                    bin_acc = np.zeros((bin_np.shape[0], H, W), dtype=np.float32)

                # Accumulate only the valid (non-padded) portion
                hv_acc[:, y:y+th, x:x+tw] += hv_np[:, :th, :tw]
                bin_acc[:, y:y+th, x:x+tw] += bin_np[:, :th, :tw]
                weight_acc[y:y+th, x:x+tw] += 1.0

        if hv_acc is None:
            return np.zeros((H, W), dtype=np.uint16)

        # Average overlapping predictions (raw logit space)
        w = np.maximum(weight_acc, 1.0)[np.newaxis]
        hv_avg = hv_acc / w      # (2, H, W)
        bin_avg = bin_acc / w    # (C, H, W)

        try:
            instance_map = self._hv_postprocess(hv_avg, bin_avg)
            return instance_map.astype(np.uint16)
        except Exception as e:
            logger.warning(f"CellViT postprocessing failed: {e}")
            return np.zeros((H, W), dtype=np.uint16)

    # NOTE — CellViT has no post-processing threshold to calibrate.
    # The reference derives the foreground mask by ARGMAX over the 2-class
    # nuclei_binary_map head — there is no numeric threshold in CellViT's
    # post-processing at all, it ships no optimisation facility, and the paper
    # discusses no such parameter.  A previous revision introduced a softmax
    # threshold (0.5, bit-identical to argmax) and calibrated it; both were
    # removed, so this matches the reference.

    def _hv_postprocess(self, hv: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """Instance map from the averaged binary + HV maps, via CellViT's own
        HoVer post-processing (``_proc_np_hv``).

        Args:
            hv:     (2, H, W) averaged horizontal/vertical regression maps.
            binary: (C, H, W) averaged 2-class nuclei_binary_map head.

        Foreground is the argmax over the binary head (reference convention, no
        numeric threshold); touching nuclei are separated by the Sobel ridges of
        the H/V maps inside ``_proc_np_hv``.
        """
        fg = binary.argmax(0).astype(np.float32)        # (H, W), 0/1
        pred = np.stack([fg, hv[0], hv[1]], axis=-1)     # (H, W, 3): [np, h, v]
        return _proc_np_hv(pred, min_size=self.min_size, ksize=_CELLVIT_KSIZE)

    def _freeze_cellvit_encoder(self):
        """Freeze CellViT encoder.

        CellViT stores the SAM/ViT image encoder as self.encoder (both CellViT
        base class and CellViTSAM subclass — confirmed in CellViT-main source:
        models/segmentation/cell_segmentation/cellvit.py).
        Falls back to keyword matching for forward compatibility.
        """
        total = sum(p.numel() for p in self._model.parameters())
        frozen = 0

        if hasattr(self._model, "encoder"):
            for param in self._model.encoder.parameters():
                param.requires_grad = False
                frozen += param.numel()
        else:
            # Fallback: keyword-based (handles any future SAM attribute layout)
            _ENC_KEYS = ("patch_embed", "blocks", "pos_embed", "cls_token",
                         "image_encoder", "encoder", "neck")
            for name, param in self._model.named_parameters():
                if any(k in name for k in _ENC_KEYS):
                    param.requires_grad = False
                    frozen += param.numel()

        logger.info(f"CellViT encoder frozen: {frozen:,}/{total:,} parameters frozen")

    def _unfreeze_all(self):
        """Restore all parameters to trainable."""
        for param in self._model.parameters():
            param.requires_grad = True
        logger.info("CellViT encoder unfrozen (all parameters trainable restored)")

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        """Fine-tune CellViT segmentation heads in fp32.

        All hyperparameters from the official CellViT SAM-H PanNuke config
        (CellViT-main/logs_paper/PanNuke/CellViTHV/SAM-H/Fold-1/config.yaml
        in the upstream CellViT repository):

        Optimizer: AdamW, betas=(0.85, 0.95), weight_decay=1e-4
        Scheduler: ExponentialLR, gamma=0.85
        freeze_encoder=True  (frozen_encoder mode): encoder frozen for all epochs.
        freeze_encoder=False (full_model mode):     encoder trainable from epoch 1.
        Note: the original paper trains with encoder frozen for 25 epochs then
        unfrozen — this two-phase schedule is NOT implemented here.
        Loss NP:   FocalTverskyLoss×1 + Dice×1   (alpha_t=0.7, beta_t=0.3, gamma_f=4/3)
        Loss HV:   MSE×2.5 + MSGE×8              (Sobel 5×5 gradient, foreground-masked)
        Precision: fp32, matching the x20 checkpoints this benchmark uses.  CellViT
                   reads `training.mixed_precision` from the run config with a default
                   of False (experiment_cellvit_pannuke.py); every published x20 config
                   omits the key, while every x40 config sets it true.  Do not add
                   autocast here unless the checkpoint is switched to x40.
        """
        self._ensure_loaded()
        import torch
        import torch.optim as optim
        from .base import split_into_patches

        n_epochs       = kwargs.get("n_epochs", 130)
        learning_rate  = kwargs.get("learning_rate", 3e-4)
        weight_decay   = kwargs.get("weight_decay", 1e-4)
        batch_size     = int(kwargs.get("batch_size", 8))
        freeze_encoder = kwargs.get("freeze_encoder", True)
        val_images     = kwargs.get("val_images", [])
        val_masks      = kwargs.get("val_masks", [])
        patience       = int(kwargs.get("patience", 20))

        # Tile support images into patch_size×patch_size patches (matches inference).
        patched_images: list = []
        patched_masks:  list = []
        for img, msk in zip(support_images, support_masks):
            for img_p, msk_p in split_into_patches(
                self.preprocess(img), msk.astype(np.uint16), self.patch_size
            ):
                patched_images.append(img_p)
                patched_masks.append(msk_p)
        n_support_images = len(support_images)   # original image count (before tiling)
        support_images = patched_images
        support_masks  = patched_masks
        n_shots        = len(support_images)     # patch count (used for training loop)
        logger.info(f"CellViT finetune: {n_support_images} images → {n_shots} training patches")

        if freeze_encoder:
            self._freeze_cellvit_encoder()

        self._model.train()

        # AdamW, betas=(0.85, 0.95) — from CellViT SAM-H PanNuke config
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.85, 0.95),
        )
        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # captured while the freeze flags are active (restored before return).
        _n_trainable_params = int(sum(p.numel() for p in self._model.parameters() if p.requires_grad))
        _n_frozen_params    = int(sum(p.numel() for p in self._model.parameters() if not p.requires_grad))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

        # Pre-process val images: split each into patch_size×patch_size patches
        # (same as training) so the val loss covers the full image, not just a
        # single crop.  For 512×512 HE val images this gives 4 patches (100%)
        # instead of one random 256×256 crop (25%).
        from .base import _EarlyStopper
        val_tensors = []
        for vimg, vmsk in zip(val_images, val_masks):
            vf32 = self.preprocess(vimg)
            for vc, vm in split_into_patches(vf32, vmsk.astype(np.uint16), self.patch_size):
                vf = _cellvit_normalize(vc)
                vt = torch.from_numpy(vf).permute(2, 0, 1).unsqueeze(0).to(self.device)
                val_tensors.append((vt, vm))

        stopper    = _EarlyStopper(patience)
        train_losses = []
        val_losses   = []
        best_epoch   = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_steps = 0

            # Shuffle support set each epoch, then process in batches.
            perm = np.random.permutation(n_shots)

            for b_start in range(0, n_shots, batch_size):
                batch_idx = perm[b_start:b_start + batch_size]

                tensors: List[torch.Tensor] = []
                aug_masks: List[np.ndarray] = []

                for idx in batch_idx:
                    # Patches are patch_size×patch_size; CellViT-style augmentation per step.
                    aug_img, aug_mask = _cellvit_augment(
                        support_images[idx], support_masks[idx]
                    )
                    img_f = _cellvit_normalize(aug_img)
                    tensors.append(torch.from_numpy(img_f).permute(2, 0, 1))
                    aug_masks.append(aug_mask)

                batch_tensor = torch.stack(tensors).to(self.device)  # (B, 3, ps, ps)

                out = self._model(batch_tensor)

                # Batch-wise loss over the whole batch (matches the reference
                # calculate_loss — region losses pooled across all B images).
                loss = self._compute_loss(out, aug_masks)

                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().float().item()
                    n_steps += 1

            avg = epoch_loss / max(n_steps, 1)
            train_losses.append(avg)
            scheduler.step()
            ep = epoch + 1
            if ep % max(1, n_epochs // 5) == 0 or epoch == 0:
                logger.info(f"  CellViT finetune epoch {ep}/{n_epochs}  loss={avg:.4f}")
            if n_steps == 0:
                logger.warning(
                    f"  CellViT finetune: loss was None for every sample in epoch {ep} — "
                    "check _compute_loss output keys and mask shapes."
                )

            # ── Early stopping ─────────────────────────────────────────────
            if val_tensors and patience > 0:
                self._model.eval()
                with torch.no_grad():
                    _vl_items = [
                        self._compute_loss(self._model(vt), [vm])
                        for vt, vm in val_tensors
                    ]
                    _valid = [x.detach().float().item() for x in _vl_items if x is not None]
                    # If ALL val samples fail loss computation fall back to inf so the
                    # early stopper does not mistake the absence of a signal for improvement.
                    vl = float(sum(_valid) / len(_valid)) if _valid else float("inf")
                self._model.train()
                val_losses.append(vl)
                stop = stopper.step(vl, self._model.state_dict())
                if stopper.counter == 0:  # val_loss improved → update best_epoch
                    best_epoch = ep
                if stop:
                    logger.info(
                        f"  CellViT early stop at epoch {ep}  "
                        f"best_val={stopper.best:.4f} (epoch {best_epoch})"
                    )
                    break
            else:
                best_epoch = ep

        stopper.restore_best(self._model)
        if freeze_encoder:
            self._unfreeze_all()
        self._model.eval()
        logger.info(
            f"CellViT fine-tuning complete: {len(train_losses)}/{n_epochs} epochs, "
            f"best epoch: {best_epoch}, {n_support_images} images ({n_shots} patches)."
        )

        return {
            "n_epochs_run":        len(train_losses),
            "n_epochs_max":        n_epochs,
            "best_epoch":          best_epoch,
            "early_stopping_mode": "patience_restore_best_weights",
            "n_train_images":      n_support_images,   # original images, not patches
            "n_train_patches":     n_shots,
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
            "train_losses":        train_losses,
            "val_losses":          val_losses,
        }

    def _compute_loss(self, model_output, masks):
        """Batch-wise CellViT training loss, matching the reference
        ``trainer_cellvit.py::calculate_loss`` (weights from the SAM-H PanNuke
        config): ``FocalTversky×1 + Dice×1`` on nuclei_binary_map and
        ``MSE×2.5 + MSGE×8`` on hv_map, each computed **once over the whole
        batch** (region losses pooled across all images, not per-image averaged).

        ``masks`` is a list of B instance masks (one per image in the batch);
        the HV target and one-hot foreground are assembled per mask and stacked
        into batched (B, …) tensors before the losses are applied.  The
        nuclei_binary_map logits are softmax'd here to mirror the reference
        ``unpack_predictions`` (FocalTversky then softmaxes a second time —
        reproduced in ``_focaltversky_loss``).

        The nuclei_type_map and tissue_types branches are intentionally omitted:
        CellBinDB has no per-nucleus type or tissue-type labels.
        """
        import torch
        import torch.nn.functional as F

        if not isinstance(model_output, dict):
            return None

        try:
            device = self.device
            # ── Assemble batched targets from the list of masks ──
            fg_list, hv_list = [], []
            for m in masks:
                fg_list.append(torch.from_numpy((m > 0).astype(np.int64)))          # (H, W)
                hv_list.append(torch.from_numpy(
                    _gen_hv_targets(m.astype(np.int32)).transpose(2, 0, 1)))          # (2, H, W)
            fg = torch.stack(fg_list).to(device)                                     # (B, H, W)
            gt_hv = torch.stack(hv_list).float().to(device)                          # (B, 2, H, W)

            loss = None

            # ── Binary (NP) branch: FocalTversky×1 + Dice×1 (both classes) ──
            binary_out = model_output.get(
                "nuclei_binary_map", model_output.get("binary_map", None)
            )
            if binary_out is not None:
                if fg.shape[-2:] != binary_out.shape[-2:]:
                    fg = F.interpolate(
                        fg.float().unsqueeze(1), size=binary_out.shape[-2:], mode="nearest"
                    ).squeeze(1).long()
                gt_onehot = F.one_hot(
                    fg, num_classes=binary_out.shape[1]
                ).permute(0, 3, 1, 2).float()                                        # (B, 2, H, W)
                nbm_soft = torch.softmax(binary_out, dim=1)   # mirrors unpack_predictions
                loss = _focaltversky_loss(nbm_soft, gt_onehot) + _dice_loss(nbm_soft, gt_onehot)

            # ── HV map branch: MSE×2.5 + MSGE×8 ──
            hv_out = model_output.get("hv_map", model_output.get("hv", None))
            if hv_out is not None:
                if gt_hv.shape[-2:] != hv_out.shape[-2:]:
                    gt_hv = F.interpolate(
                        gt_hv, size=hv_out.shape[-2:], mode="bilinear", align_corners=False,
                    )
                focus = fg.float()  # (B, H, W) foreground mask
                loss_hv = F.mse_loss(hv_out, gt_hv) * 2.5 \
                    + _sobel_msge_loss(hv_out, gt_hv, focus, device) * 8.0
                loss = loss_hv if loss is None else loss + loss_hv

            return loss
        except Exception as e:
            logger.warning(f"  CellViT _compute_loss failed: {e}")
            return None

    def get_param_count(self) -> int:
        if self._model is None:
            return -1
        try:
            return sum(p.numel() for p in self._model.parameters())
        except Exception:
            return -1

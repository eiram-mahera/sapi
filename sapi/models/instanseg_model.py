"""
InstanSeg model wrapper.

InstanSeg is a fast, embedding-based instance segmentation algorithm optimised
for accurate and efficient cell/nuclei segmentation in histology and fluorescence
microscopy.  It uses a UNet backbone with coordinate-based embedding learning:
each pixel votes for the instance centre it belongs to, and instances are
extracted by peak detection + watershed on those votes.

Paper:
    Goldsborough et al. (2024), "InstanSeg: An Embedding-Based Instance
    Segmentation Algorithm Optimized for Accurate, Efficient and Multimodal
    Cell Segmentation". arXiv: 2408.15954.

Setup:
    pip install instanseg-torch

Checkpoint path priority:
    YAML 'checkpoint_path'  →  INSTANSEG_CHECKPOINT env var  →  default
    The YAML wins; the env var is only a fallback when the YAML key is absent.

Checkpoint: single_channel_nuclei/instanseg.pt (BioImage Model Zoo).
    - Type: TorchScript scripted InstanSeg_Torchscript module
    - in_channels=1 (single-channel nuclei model)
    - cells_and_nuclei=False (nuclei output only, 5-channel head)
    - n_sigma=2, dim_coords=2, dim_seeds=1, mlp_width=5
    - No AdaptorNet, no multihead

Architecture auto-detection:
    For state-dict checkpoints (.pth), all architecture parameters are
    auto-detected from the checkpoint keys:
      in_channels:      encoder.0.conv0.0.weight.shape[1]
      cells_and_nuclei: multihead, or final_block count in decoders.0 (>3 → True)
      multihead:        'decoders.1.' present in any key
      mlp_width:        pixel_classifier.fc1.weight.shape[0]

Fine-tuning:
    Two-phase schedule matching instanseg/scripts/train.py, with the reference's
    own defaults throughout:
      Phase 1 (hotstart): dice_loss + binary_xloss for fast warm-up.
      Phase 2 (main):     lovasz_hinge + l1_distance, window_size=128.
    Both losses are swapped in for the hotstart and both swapped back after, as
    train.py does; with hotstart_epochs=0 the model trains on the main losses
    from the first epoch, again matching the reference.
    Architecture is always auto-detected from checkpoint so the UNet matches.

Input preprocessing:
    All datasets (including H&E) receive the standard 1-channel float32 image
    from the dataset loader: fluorescence images are normalised directly;
    H&E images are stain-separated to the Hematoxylin channel and normalised.
    The single_channel_nuclei checkpoint (in_channels=1) takes this 1-channel
    input directly — no RGB averaging needed.

Inference:
    target_segmentation=[1,0] requests nuclei only.  For cells_and_nuclei=False
    this is a no-op; for cells_and_nuclei=True it skips the cells decoder.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseSegmentationModel, calibrate_thresholds_on_support

logger = logging.getLogger(__name__)

_DEFAULT_CKPT = Path.home() / ".instanseg" / "instanseg.pt"

# ---------------------------------------------------------------------------
# Training configuration — the reference's own defaults
# ---------------------------------------------------------------------------
# From instanseg/scripts/train.py's argparse defaults, which is what
# InstanSeg(...) is constructed with:
#   --binary_loss_fn  lovasz_hinge   (line 51)
#   --seed_loss_fn    l1_distance    (line 52)
#   --window_size     128            (line 56)
#
# `l1_distance` rescales the seed channel, so seed_threshold is calibrated on
# the support set after fine-tuning rather than kept at its pretrained value.
#
# `window_size` is the crop taken around each instance when computing the loss.
# The reference deliberately uses different values for training and inference
# (trains at 128, infers at 32).
_MAIN_BINARY_LOSS  = "lovasz_hinge"
_MAIN_SEED_LOSS    = "l1_distance"
_TRAIN_WINDOW_SIZE = 128


def _filter_min_size(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop every predicted instance whose area is strictly less than min_size.

    This is the benchmark's definition, applied here in the
    wrapper because InstanSeg's own `min_size` does NOT implement it: that filter
    lives in merge_sparse_predictions (the sliding-window path), so images
    <=1024 px never reach it, and its operator gates candidates rather than final
    rasterised area.  Every other model drops `area < min_size`; without this,
    InstanSeg would silently under-filter for any dataset setting min_size > 1.

    Area is counted per LABEL (all pixels sharing an id), matching how the metrics
    pool an instance — not per connected component.

    min_size <= 1 is a strict no-op: no instance can have an area below 1, and the
    mask is returned untouched (not even copied), so the current dataset value of 1
    cannot change any prediction.
    """
    if min_size <= 1:
        return mask
    ids, counts = np.unique(mask, return_counts=True)
    small = ids[(ids > 0) & (counts < min_size)]
    if small.size == 0:
        return mask
    out = mask.copy()
    out[np.isin(out, small)] = 0
    return out


def _load_state_dict_from_path(path: str) -> dict:
    """
    Load a flat state_dict from a checkpoint file.

    Handles both:
    - Training checkpoint dict (.pth) with optional 'model_state_dict' key
    - TorchScript file (.pt) returned by torch.jit.load → call .state_dict()

    Never returns a Module object — always returns {str: Tensor}.
    """
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        sd = raw.get("model_state_dict", raw)
        return {k.replace("module.", ""): v for k, v in sd.items()}
    if hasattr(raw, "state_dict"):
        # TorchScript module (RecursiveScriptModule)
        return raw.state_dict()
    raise ValueError(f"Unrecognized checkpoint format: {type(raw)}")


def _instanseg_augment(img: np.ndarray, mask: np.ndarray):
    """InstanSeg's native default ("minimal") fluorescence augmentation.

    The reference pipeline is
        to_tensor -> normalize -> torch_rescale -> channel_subsample -> flips -> rotate
    (augmentation_config.py::get_augmentation_dict, augmentation_type="minimal").
    Of those, only flips and rotate are stochastic geometry and are reproduced here:
      * to_tensor / normalize  -> done in _to_tensors via the reference's own
        percentile_normalize.
      * channel_subsample      -> dispatched with probability 0 unless the model is
        channel-invariant, i.e. never applied for our checkpoints.
      * torch_rescale          -> a DETERMINISTIC rescale to the model's canonical
        pixel size, NOT augmentation: the dispatcher never passes `amount`, so its
        jitter term is `1 + (rand-0.5)*0` == 1.  It rescales only when a pixel size
        is available (`metadata["pixel_size"]`, else `current_pixel_size` stays None
        and the branch is skipped).  CellBinDB publishes no pixel size metadata,
        so the reference itself would no-op here.  Do NOT add scale jitter to
        "restore" this step — the reference applies none.

    flips: h/v independently at p=0.5 (Augmentations.flips hardcodes amount=0.5).
    rotate: uniform over {0,90,180,270} (Augmentations.rotate).
    Implemented with numpy (np.flip / np.rot90) to avoid a torchvision dependency."""
    import random
    # Reshape to channel-first (C, H, W) for uniform axis handling
    was_2d = img.ndim == 2
    arr = (img[None] if was_2d else img.transpose(2, 0, 1)).astype(np.float32)
    msk = mask[None].astype(np.int32)

    if random.random() > 0.5:
        arr = np.flip(arr, axis=2).copy()
        msk = np.flip(msk, axis=2).copy()
    if random.random() > 0.5:
        arr = np.flip(arr, axis=1).copy()
        msk = np.flip(msk, axis=1).copy()
    k = int(np.random.choice([0, 1, 2, 3]))
    if k:
        arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
        msk = np.rot90(msk, k=k, axes=(1, 2)).copy()

    a_img = arr[0] if was_2d else arr.transpose(1, 2, 0)
    return a_img.astype(np.float32), msk[0].astype(np.uint16)


class InstanSegModel(BaseSegmentationModel):
    """InstanSeg instance segmentation wrapper."""

    supports_finetune = True
    num_channels      = 1           # single_channel_nuclei: 1-channel input for all modalities
    needs_raw_he_image = False      # H&E: use stain-separated hematoxylin channel (same as other 1-ch models)

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        self.min_size = int(config.get("min_size", 1))
        self.model_name = config.get("variant", "instanseg")

        # Checkpoint path priority: YAML → env var → default.
        _yaml_path = config.get("checkpoint_path")
        _env_path  = os.environ.get("INSTANSEG_CHECKPOINT")
        _raw = _yaml_path or _env_path
        self.checkpoint_path: str | None = (
            os.path.expanduser(_raw) if _raw else
            str(_DEFAULT_CKPT) if _DEFAULT_CKPT.exists() else None
        )

        self.pixel_size: float | None = config.get("pixel_size") or None

        # Architecture parameters — fallback values used when auto-detection
        # from the checkpoint fails.  For single_channel_nuclei: in_channels=1.
        self.model_str        = config.get("model_str", "InstanSeg_UNet")
        self.in_channels      = config.get("in_channels", 1)
        self.n_sigma          = config.get("n_sigma", 2)
        self.dim_coords       = config.get("dim_coords", 2)
        self.dim_seeds        = config.get("dim_seeds", 1)
        self.cells_and_nuclei = config.get("cells_and_nuclei", False)
        self.feat_eng_fn: str = str(config.get("feature_engineering", "0"))

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _inspect_state_dict(state_dict: dict) -> dict:
        """
        Auto-detect architecture from state_dict keys (any prefix depth).

        Returns:
          in_channels     — int: UNet encoder input channels (from conv0.0.weight.shape[1])
          cells_and_nuclei— bool: nuclei+cells (True) or nuclei-only (False)
          multihead       — bool: two separate decoders
          mlp_width       — int or None: pixel_classifier width
        """
        info: dict = {}

        # in_channels: first encoder conv0 weight shape[1]
        info["in_channels"] = None
        for k, v in state_dict.items():
            if re.search(r"encoder\.0\.conv0\.0\.weight$", k):
                info["in_channels"] = int(v.shape[1])
                break

        # multihead: second decoder present
        info["multihead"] = any("decoders.1." in k for k in state_dict)

        # cells_and_nuclei: count final_block indices in decoder 0
        fb_idx: set = set()
        for k in state_dict:
            m = re.search(r"decoders\.0\.final_block\.(\d+)\.", k)
            if m:
                fb_idx.add(int(m.group(1)))
        info["cells_and_nuclei"] = info["multihead"] or len(fb_idx) > 3

        # mlp_width: from pixel_classifier.fc1.weight (any prefix)
        info["mlp_width"] = None
        for k, v in state_dict.items():
            if k.endswith("pixel_classifier.fc1.weight"):
                info["mlp_width"] = int(v.shape[0])
                break

        return info

    @staticmethod
    def _infer_mlp_width(pc_state_dict: dict, default: int = 10) -> int:
        """Infer ProbabilityNet width from stripped state dict (fc1.weight.shape[0])."""
        w = pc_state_dict.get("fc1.weight")
        return int(w.shape[0]) if w is not None else default

    def _arch_dict(
        self,
        multihead: bool = False,
        cells_and_nuclei: bool | None = None,
        in_channels: int | None = None,
    ) -> dict:
        """Build architecture dictionary for build_model_from_dict."""
        c_n = self.cells_and_nuclei if cells_and_nuclei is None else cells_and_nuclei
        n_ch = self.in_channels if in_channels is None else in_channels
        return {
            "model_str":            self.model_str,
            "dim_in":               n_ch,
            "n_sigma":              self.n_sigma,
            "dim_coords":           self.dim_coords,
            "dim_seeds":            self.dim_seeds,
            "cells_and_nuclei":     c_n,
            "layers":               [32, 64, 128, 256],
            "norm":                 "BATCH",
            "dropprob":             0.0,
            "multihead":            multihead,
            "feature_engineering":  self.feat_eng_fn,
            "adaptor_net_str":      "1",
            "channel_invariant":    False,
            "only_positive_labels": True,
        }

    # ── Model loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if self.checkpoint_path is None:
            raise FileNotFoundError(
                "InstanSeg checkpoint not found. "
                "Set 'checkpoint_path' in configs/models/instanseg.yaml or "
                "the INSTANSEG_CHECKPOINT env var."
            )

        import torch

        # ── Path 1: TorchScript (.pt) ──────────────────────────────────────
        try:
            self._model = torch.jit.load(
                self.checkpoint_path, map_location=self.device
            )
            self._model.eval()
            logger.info(f"InstanSeg loaded as TorchScript from {self.checkpoint_path}")
            return
        except Exception as exc:
            logger.info(
                f"TorchScript load failed ({exc}); trying state-dict path for "
                f"{self.checkpoint_path}"
            )

        # ── Path 2: Training checkpoint dict (.pth) ────────────────────────
        from instanseg.utils.model_loader import build_model_from_dict
        from instanseg.utils.loss.instanseg_loss import (
            InstanSeg as InstanSegLoss,
            InstanSeg_Torchscript,
        )

        state_dict = _load_state_dict_from_path(self.checkpoint_path)

        has_pixel_clf = any("pixel_classifier" in k for k in state_dict)
        has_adaptor   = any("AdaptorNet"        in k for k in state_dict)
        arch_info     = self._inspect_state_dict(state_dict)
        multihead        = arch_info["multihead"]
        cells_and_nuclei = arch_info["cells_and_nuclei"]
        det_in_ch        = arch_info["in_channels"] or self.in_channels

        logger.info(
            f"InstanSeg state-dict: in_channels={det_in_ch}, "
            f"cells_and_nuclei={cells_and_nuclei}, multihead={multihead}, "
            f"has_pixel_clf={has_pixel_clf}, has_adaptor={has_adaptor}"
        )

        arch_dict = self._arch_dict(
            multihead=multihead, cells_and_nuclei=cells_and_nuclei, in_channels=det_in_ch
        )
        nn_model = build_model_from_dict(arch_dict)

        if has_pixel_clf:
            mlp_width = arch_info["mlp_width"] or int(self.config.get("mlp_width", 10))
            loss_obj = InstanSegLoss(
                n_sigma=self.n_sigma,
                dim_coords=self.dim_coords,
                dim_seeds=self.dim_seeds,
                cells_and_nuclei=cells_and_nuclei,
                feature_engineering_function=self.feat_eng_fn,
            )
            nn_model = loss_obj.initialize_pixel_classifier(nn_model, MLP_width=mlp_width)

        if has_adaptor:
            from instanseg.utils.models.ChannelInvariantNet import AdaptorNetWrapper
            nn_model = AdaptorNetWrapper(
                nn_model,
                norm=arch_dict["norm"],
                adaptor_net_str=arch_dict["adaptor_net_str"],
            )

        try:
            nn_model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model_sd = nn_model.state_dict()
            compat = {
                k: v for k, v in state_dict.items()
                if k in model_sd and v.shape == model_sd[k].shape
            }
            nn_model.load_state_dict(compat, strict=False)
            logger.warning(
                f"InstanSeg: shape-compatible load: {len(compat)}/{len(model_sd)} tensors."
            )

        nn_model.eval()
        ts_wrapper = InstanSeg_Torchscript(
            model=nn_model,
            cells_and_nuclei=cells_and_nuclei,
            pixel_size=self.pixel_size or 0.0,
            n_sigma=self.n_sigma,
            dim_coords=self.dim_coords,
            dim_seeds=self.dim_seeds,
            backbone_dim_in=det_in_ch,
            feature_engineering_function=self.feat_eng_fn,
        )
        self._model = ts_wrapper.to(self.device).eval()
        self.in_channels = det_in_ch
        n_params = sum(p.numel() for p in nn_model.parameters())
        logger.info(
            f"InstanSeg loaded from state-dict ({self.checkpoint_path}), "
            f"{n_params:,} params."
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Post-processing thresholds
    # ------------------------------------------------------------------

    # InstanSeg_Torchscript.forward() takes these as optional arguments and falls
    # back to `self.default_mask_threshold` / `self.default_seed_threshold` when
    # they are None.  Those defaults are NOT fixed library constants: they come
    # from `params` baked into the checkpoint at trace time
    # (`self.params.get('mask_threshold', 0.53)`), so a released checkpoint can
    # carry its own tuned values.
    #
    # Passing the generic 0.53/0.7 unconditionally therefore OVERRIDES whatever
    # the checkpoint specifies — a silent regression, not a default.  So None
    # means "defer to this model's own value", and an explicit number is only
    # ever set by calibrate_thresholds().
    #
    # None is listed FIRST in each grid, and the shared helper keeps the first
    # combo on a tie, so calibration can only move away from the checkpoint's
    # own values when doing so measurably helps.
    _MASK_GRID = (None, 0.40, 0.45, 0.53, 0.60, 0.70)
    _SEED_GRID = (None, 0.50, 0.60, 0.70, 0.80)

    def _thresholds(self) -> dict:
        """Explicit thresholds, or None to defer to the model's own defaults."""
        return {
            "mask_threshold": self.config.get("mask_threshold"),
            "seed_threshold": self.config.get("seed_threshold"),
        }

    def calibrate_thresholds(self, cal_images, cal_masks) -> dict:
        """Re-fit mask/seed thresholds on the SUPPORT set after fine-tuning."""
        self._ensure_loaded()
        result = calibrate_thresholds_on_support(
            predict_fn=lambda img, **kw: self._predict_with(img, **kw),
            images=cal_images,
            masks=cal_masks,
            grid={
                "mask_threshold": list(self._MASK_GRID),
                "seed_threshold": list(self._SEED_GRID),
            },
            model_name=self.model_name,
        )
        self.config["mask_threshold"] = result["mask_threshold"]
        self.config["seed_threshold"] = result["seed_threshold"]
        return result

    def _predict_with(self, image, **thresholds):
        """predict_single with explicit thresholds (used by calibration)."""
        saved = {k: self.config.get(k) for k in thresholds}
        self.config.update(thresholds)
        try:
            return self.predict_single(image)
        finally:
            for k, v in saved.items():
                if v is None:
                    self.config.pop(k, None)
                else:
                    self.config[k] = v

    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Inference on a single image.

        Channel handling (self.in_channels=1 for single_channel_nuclei):
          Fluorescence and H&E (already reduced to 1-ch Hematoxylin by the
          dataset loader, needs_raw_he_image=False) both pass through as-is
          (no expansion).

        target_segmentation=[1,0] requests nuclei only (no-op for
        cells_and_nuclei=False, skips cells decoder for True).

        Returns uint16 instance label map (H, W).
        """
        import torch
        from instanseg.utils.pytorch_utils import _to_tensor_float32
        from instanseg.utils.utils import percentile_normalize

        img_t = _to_tensor_float32(image)   # (C, H, W)
        H, W = img_t.shape[-2], img_t.shape[-1]

        # Channel adaptation: match self.in_channels (1 for this checkpoint)
        if img_t.shape[0] == 1 and self.in_channels > 1:
            img_t = img_t.expand(self.in_channels, -1, -1).contiguous()
        elif img_t.shape[0] > 1 and self.in_channels == 1:
            img_t = img_t.mean(dim=0, keepdim=True)
        elif img_t.shape[0] > 1 and img_t.shape[0] != self.in_channels:
            img_t = img_t[:self.in_channels]  # trim extra channels

        img_norm = percentile_normalize(img_t)   # (C, H, W), CPU

        target_seg = torch.tensor([1, 0])   # nuclei only
        use_cuda = str(self.device).startswith("cuda")

        # Dataset-level minimum instance size (configs/datasets/<name>.yaml), which
        # the benchmark defines as "drop instances with area < min_size".
        #
        # M-1 is passed, not M.  For M=1 this yields 0, which trips InstanSeg's own
        # `if min_size > 0` guard and skips its filter entirely — exactly right, since
        # "drop area < 1" is a no-op.  This is the only value the benchmark currently
        # uses, and it is verified correct.
        #
        # WARNING for M > 1: InstanSeg's min_size is NOT an output-area filter and
        # does NOT honour the benchmark's definition.  Measured on the released
        # checkpoint at 256x256: passing 2 still leaves 12 instances with area < 3,
        # passing 5 still leaves 78 with area < 6.  It changes the instance count, so
        # it gates something (seeds/candidates), but not final rasterised area.  The
        # `area > min_size` line in instanseg_loss.py lives in merge_sparse_predictions,
        # i.e. the sliding-window path, which images <=1024 px never reach.
        #
        # RESOLVED: the benchmark's definition is now enforced in this wrapper by
        # _filter_min_size() on the returned instance map (see the return below), so
        # InstanSeg drops `area < min_size` on every path exactly like every other
        # model.  This parameter is still passed because it is semantically correct
        # where it does apply (`keep area > M-1` == `drop area < M`); the wrapper-side
        # filter is the authoritative one and makes it redundant, not wrong.
        _MIN_SIZE = self.min_size - 1

        if max(H, W) <= 1024:
            img_batch = img_norm.unsqueeze(0).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    result = self._model(
                        img_batch, target_segmentation=target_seg,
                        min_size=_MIN_SIZE, **self._thresholds(),
                    )
        else:
            from instanseg.utils.tiling import _sliding_window_inference
            result = _sliding_window_inference(
                img_norm,
                self._model,
                window_size=(512, 512),
                sw_device=self.device,
                device="cpu",
                batch_size=1,
                output_channels=1,
                show_progress=False,
                instanseg_kwargs={
                    "target_segmentation": target_seg,
                    "min_size": _MIN_SIZE,
                    **self._thresholds(),
                },
            ).float()

        return _filter_min_size(self._to_instance_map(result, H, W), self.min_size)

    @staticmethod
    def _to_instance_map(result, orig_h: int, orig_w: int) -> np.ndarray:
        """Convert InstanSeg output tensor to uint16 HxW instance map."""
        import torch

        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, torch.Tensor):
            t = result.cpu()
            while t.ndim > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim == 3:
                t = t[0]
            arr = t.numpy()
        else:
            arr = np.array(result)
        if arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.uint16)

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        return self._run_inference(image)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 1) -> List[np.ndarray]:
        self._ensure_loaded()
        return [self._run_inference(img) for img in images]

    # ── Fine-tuning ────────────────────────────────────────────────────────────

    def _build_nn_model_from_checkpoint(self):
        """
        Build trainable nn.Module UNet with pretrained weights.

        Architecture (in_channels, cells_and_nuclei, multihead, mlp_width) is
        always auto-detected from checkpoint keys to ensure correctness.

        Returns:
            (nn_model, pc_state_dict, detected_cells_and_nuclei, detected_in_channels)
        """
        import torch
        from instanseg.utils.model_loader import build_model_from_dict

        if not self.checkpoint_path:
            logger.warning("InstanSeg: no checkpoint_path; fine-tuning from random init.")
            arch = self._arch_dict()
            return build_model_from_dict(arch), {}, self.cells_and_nuclei, self.in_channels

        # ── TorchScript checkpoint ─────────────────────────────────────────
        try:
            ts = torch.jit.load(self.checkpoint_path, map_location="cpu")
            sd = ts.state_dict()

            # Detect from 'fcn.*' subtree (the traced UNet)
            fcn_sd     = {k[4:]: v for k, v in sd.items() if k.startswith("fcn.")}
            arch_info  = self._inspect_state_dict(fcn_sd)
            multihead  = arch_info["multihead"]
            det_c_n    = arch_info["cells_and_nuclei"]
            det_in_ch  = arch_info["in_channels"] or self.in_channels

            # UNet weights: full 'fcn.*' subtree with prefix stripped.
            # strict=False silently skips 'pixel_classifier.*' inside fcn
            # (those are the same weights stored separately as 'pixel_classifier.*').
            unet_sd = fcn_sd
            pc_sd   = {k[len("pixel_classifier."):]: v
                       for k, v in sd.items()
                       if k.startswith("pixel_classifier.")}

            arch_dict = self._arch_dict(
                multihead=multihead, cells_and_nuclei=det_c_n, in_channels=det_in_ch
            )
            nn_model = build_model_from_dict(arch_dict)

            # Shape-compatible load: avoid RuntimeError on shape mismatches
            model_sd  = nn_model.state_dict()
            compat_sd = {
                k: v for k, v in unet_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape
            }
            nn_model.load_state_dict(compat_sd, strict=False)
            n_loaded = len(compat_sd)
            n_total  = len(model_sd)
            logger.info(
                f"InstanSeg finetune (TorchScript): loaded {n_loaded}/{n_total} UNet tensors. "
                f"in_channels={det_in_ch}, cells_and_nuclei={det_c_n}, multihead={multihead}."
            )
            if pc_sd:
                w = pc_sd.get("fc1.weight")
                logger.info(
                    f"InstanSeg finetune: pixel_classifier found "
                    f"(fc1 shape {tuple(w.shape) if w is not None else '?'})."
                )
            return nn_model, pc_sd, det_c_n, det_in_ch

        except Exception as exc:
            logger.warning(
                f"InstanSeg finetune: TorchScript load failed ({exc}); "
                "trying generic state-dict path."
            )

        # ── Generic state-dict path (.pth or .pt via torch.load) ──────────
        # _load_state_dict_from_path handles both dict and ScriptModule returns.
        sd = _load_state_dict_from_path(self.checkpoint_path)

        arch_info = self._inspect_state_dict(sd)
        multihead  = arch_info["multihead"]
        det_c_n    = arch_info["cells_and_nuclei"]
        det_in_ch  = arch_info["in_channels"] or self.in_channels

        pc_sd = {}
        for k, v in sd.items():
            if "pixel_classifier." in k:
                suffix = k[k.index("pixel_classifier.") + len("pixel_classifier."):]
                pc_sd[suffix] = v

        arch_dict = self._arch_dict(
            multihead=multihead, cells_and_nuclei=det_c_n, in_channels=det_in_ch
        )
        nn_model = build_model_from_dict(arch_dict)

        model_sd  = nn_model.state_dict()
        compat_sd = {
            k: v for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        nn_model.load_state_dict(compat_sd, strict=False)
        logger.info(
            f"InstanSeg finetune (generic): loaded {len(compat_sd)}/{len(model_sd)} tensors. "
            f"in_channels={det_in_ch}, cells_and_nuclei={det_c_n}, multihead={multihead}."
        )
        return nn_model, pc_sd, det_c_n, det_in_ch

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        """
        Fine-tune InstanSeg using the original two-phase training schedule.

        Phase 1 — hotstart (`hotstart_epochs`): dice_loss + binary_xloss.
            Warms up the embedding before the harder Lovász optimisation so the
            model doesn't start lovasz_hinge from a random embedding — matching
            the hotstart used in instanseg/scripts/train.py.

        Phase 2 — main training (up to `n_epochs`): the reference's own main-phase
            losses, `lovasz_hinge` + `l1_distance`, restored exactly as
            train.py:377-378 restores both after its hotstart.  `l1_distance`
            rescales the seed channel, so seed_threshold is calibrated on the
            support set afterwards.

        Each epoch is one natural pass through all n_train support images
        (shuffled, fresh augmentation each time).  The authors' 1000-patch epoch
        was a large-dataset normalisation constant — for 1–9 images the natural
        epoch is exactly n_train steps.

        No LR scheduler (constant Adam lr) — matches train.py default.
        """
        import torch
        from instanseg.utils.loss.instanseg_loss import (
            InstanSeg as InstanSegLoss,
            InstanSeg_Torchscript,
        )
        from instanseg.utils.pytorch_utils import _to_tensor_float32
        from instanseg.utils.utils import percentile_normalize

        n_epochs        = int(kwargs.get("n_epochs",        300))
        lr              = float(kwargs.get("learning_rate",  1e-3))
        wd              = float(kwargs.get("weight_decay",   0.0))
        batch_size      = int(kwargs.get("batch_size",       8))
        freeze_enc      = bool(kwargs.get("freeze_encoder",  True))
        val_images_np   = kwargs.get("val_images",           [])
        val_masks_np    = kwargs.get("val_masks",            [])
        patience        = int(kwargs.get("patience",         20))
        hotstart_epochs = int(kwargs.get("hotstart_epochs",  10))
        n_train         = len(support_images)

        # ── 1. Build trainable UNet ────────────────────────────────────────
        nn_model, pc_sd, ckpt_c_n, ckpt_in_ch = self._build_nn_model_from_checkpoint()

        if ckpt_c_n != self.cells_and_nuclei:
            logger.warning(
                f"InstanSeg: YAML cells_and_nuclei={self.cells_and_nuclei} "
                f"≠ checkpoint ({ckpt_c_n}). Using checkpoint value."
            )

        # ── 2. Attach pixel classifier (Phase 1 loss: dice + binary_xloss) ─
        mlp_width = self._infer_mlp_width(
            pc_sd, default=int(self.config.get("mlp_width", 10))
        )
        # window_size=32 matches the inference TorchScript default
        # (InstanSeg_Torchscript.default_window_size=32).  Using the same crop
        # size during fine-tuning ensures the pixel_classifier training signal
        # comes from the same spatial context it sees at inference.
        # Constructed with the MAIN-phase losses, exactly as instanseg/scripts/
        # train.py does (`InstanSeg(binary_loss_fn_str=args.binary_loss_fn,
        # seed_loss_fn=args.seed_loss_fn, ...)`); the hotstart block below swaps
        # in dice_loss + binary_xloss and swaps them back afterwards.  This also
        # makes hotstart_epochs=0 behave exactly like the reference with
        # --hotstart 0, where the update_* calls never run at all.
        loss_obj = InstanSegLoss(
            n_sigma=self.n_sigma,
            dim_coords=self.dim_coords,
            dim_seeds=self.dim_seeds,
            cells_and_nuclei=ckpt_c_n,
            binary_loss_fn_str=_MAIN_BINARY_LOSS,
            seed_loss_fn=_MAIN_SEED_LOSS,
            device=self.device,
            feature_engineering_function=self.feat_eng_fn,
            window_size=_TRAIN_WINDOW_SIZE,
        )
        nn_model = loss_obj.initialize_pixel_classifier(nn_model, MLP_width=mlp_width)

        # ── 3. Load pretrained pixel_classifier weights ────────────────────
        if pc_sd:
            try:
                nn_model.pixel_classifier.load_state_dict(pc_sd, strict=True)
                loss_obj.pixel_classifier = nn_model.pixel_classifier
                logger.info(
                    f"InstanSeg: loaded pretrained pixel_classifier "
                    f"(width={mlp_width})."
                )
            except RuntimeError as exc:
                logger.warning(
                    f"InstanSeg: pixel_classifier load failed ({exc}); "
                    "using random init."
                )

        nn_model = nn_model.to(self.device).train()

        # ── 4. Freeze encoder ──────────────────────────────────────────────
        if freeze_enc and hasattr(nn_model, "encoder"):
            for p in nn_model.encoder.parameters():
                p.requires_grad_(False)
            logger.info("InstanSeg: encoder frozen.")

        trainable = [p for p in nn_model.parameters() if p.requires_grad]
        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # captured while the freeze flags are active (restored before return).
        _n_trainable_params = int(sum(p.numel() for p in trainable))
        _n_frozen_params    = int(sum(p.numel() for p in nn_model.parameters())) - _n_trainable_params
        # Constant LR Adam — matches train.py default (scheduler=None for UNet).
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)

        # ── 5. Tensor-conversion helper (channel adapt + percentile norm) ──
        def _to_tensors(img_np: np.ndarray, mask_np: np.ndarray):
            t = _to_tensor_float32(img_np)
            if t.shape[0] == 1 and ckpt_in_ch > 1:
                t = t.expand(ckpt_in_ch, -1, -1).contiguous()
            elif t.shape[0] > 1 and ckpt_in_ch == 1:
                t = t.mean(dim=0, keepdim=True)
            elif t.shape[0] > 1 and t.shape[0] != ckpt_in_ch:
                t = t[:ckpt_in_ch]
            return (
                percentile_normalize(t),
                torch.tensor(mask_np.astype(np.int32)).unsqueeze(0),
            )

        # ── 6. Pre-process val images once (fixed throughout training) ─────
        val_tensors = (
            [_to_tensors(i, m) for i, m in zip(val_images_np, val_masks_np)]
            if val_images_np else []
        )

        # ── 7. Per-epoch helpers ───────────────────────────────────────────
        def _run_epoch(current_loss_obj) -> float:
            """One pass through all n_train images (InstanSeg-style augmentation per
            step; InstanSeg percentile-normalises internally).  ceil(n_train/batch_size)
            gradient steps per epoch."""
            nn_model.train()
            # With frozen encoder, keep its BatchNorm layers in eval mode so their
            # running stats do not drift on the tiny support set (same fix as MicroSAM).
            if freeze_enc and hasattr(nn_model, "encoder"):
                nn_model.encoder.eval()
            epoch_loss = 0.0
            count = 0
            indices = np.random.permutation(n_train).tolist()
            for b_start in range(0, n_train, batch_size):
                idx = indices[b_start : b_start + batch_size]
                imgs_b, masks_b = [], []
                for i in idx:
                    a_img, a_msk = _instanseg_augment(support_images[i], support_masks[i])
                    t, m = _to_tensors(a_img, a_msk.astype(np.int32))
                    imgs_b.append(t)
                    masks_b.append(m)
                img_b = torch.stack(imgs_b).to(self.device)
                msk_b = torch.stack(masks_b).to(self.device)

                # fp32, matching InstanSeg's own training loop (utils/AI_utils.py::
                # train_epoch), which uses no autocast and no GradScaler.  Do not add
                # autocast here to mirror predict_single: InstanSeg is deliberately
                # asymmetric — fp32 to train, fp16 to infer.
                pred = nn_model(img_b)
                # .clone() matches the reference's `train_loss_fn(output, labels.clone())`.
                # It is defensive for a real reason: InstanSegLoss.forward runs
                # `instance[instance < 0] = 0` where `instance` is a VIEW of the
                # label tensor (basic indexing + unsqueeze), so it writes back into
                # the caller's tensor.
                loss = current_loss_obj.forward(pred.float(), msk_b.clone())
                if not isinstance(loss, torch.Tensor) or loss.numel() == 0:
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=20.0)
                optimizer.step()
                epoch_loss += loss.item()
                count += 1
            return epoch_loss / max(count, 1)

        train_losses: list = []
        val_losses:   list = []

        # ── 8. Phase 1: Hotstart (dice_loss + binary_xloss) ───────────────
        # train.py:371-373 — both losses are swapped for the hotstart.
        if hotstart_epochs > 0:
            loss_obj.update_seed_loss("binary_xloss")
            loss_obj.update_binary_loss("dice_loss")
            logger.info(
                f"InstanSeg hotstart: {hotstart_epochs} epoch(s) "
                f"× {n_train} images  [dice_loss + binary_xloss]"
            )
            for epoch in range(hotstart_epochs):
                mean_loss = _run_epoch(loss_obj)
                train_losses.append(mean_loss)
                ep = epoch + 1
                if ep % 10 == 0 or ep == hotstart_epochs:
                    logger.info(
                        f"InstanSeg hotstart {ep}/{hotstart_epochs}: "
                        f"loss={mean_loss:.4f}"
                    )

        # ── 9. Phase 2: Main training (reference main-phase losses) ──────
        # train.py:377-378 restores BOTH losses after the hotstart.  l1_distance
        # rescales the seed channel, so seed_threshold is calibrated on the
        # support set after fine-tuning to give it a fitted threshold rather
        # than a stale one.
        # No-ops when hotstart_epochs == 0 (the losses are already these).
        loss_obj.update_seed_loss(_MAIN_SEED_LOSS)
        loss_obj.update_binary_loss(_MAIN_BINARY_LOSS)
        logger.info(
            f"InstanSeg main training: up to {n_epochs} epoch(s) "
            f"× {n_train} images  [{_MAIN_BINARY_LOSS} + {_MAIN_SEED_LOSS}]"
        )

        def _val_loss() -> float:
            """Val loss using the same loss_obj as training (whatever the main phase
            is currently set to — lovasz_hinge + l1_distance by default).
            Consistent with _EarlyStopper convention: lower is better."""
            nn_model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for vit, vmt in val_tensors:
                    pred = nn_model(vit.unsqueeze(0).to(self.device))
                    # .clone() as in training — and it matters more here: val_tensors
                    # are built ONCE and reused every epoch, so an in-place write would
                    # persist.  On CUDA the .to() copy already shields them; on CPU it
                    # returns the same view and would not.
                    loss = loss_obj.forward(
                        pred.float(), vmt.unsqueeze(0).to(self.device).clone()
                    )
                    if isinstance(loss, torch.Tensor) and loss.numel() > 0:
                        total += loss.item()
                        count += 1
            nn_model.train()
            if freeze_enc and hasattr(nn_model, "encoder"):
                nn_model.encoder.eval()
            return total / max(count, 1) if count > 0 else float("inf")

        from .base import _EarlyStopper
        stopper   = _EarlyStopper(patience)
        best_epoch = hotstart_epochs

        for epoch in range(n_epochs):
            ep_total = hotstart_epochs + epoch + 1  # 1-indexed overall
            mean_loss = _run_epoch(loss_obj)
            train_losses.append(mean_loss)

            if val_tensors and patience > 0:
                vl = _val_loss()
                val_losses.append(vl)

                if stopper.step(vl, nn_model.state_dict()):
                    logger.info(
                        f"InstanSeg early stop ep={ep_total}  "
                        f"best val_loss={stopper.best:.4f} @ ep={best_epoch}"
                    )
                    break
                if stopper.counter == 0:
                    best_epoch = ep_total
                if ep_total % 10 == 0:
                    logger.info(
                        f"InstanSeg ep={ep_total} (main {epoch + 1}/{n_epochs}): "
                        f"loss={mean_loss:.4f}  val_loss={vl:.4f}  "
                        f"best={stopper.best:.4f}"
                    )
            else:
                best_epoch = ep_total
                if ep_total % 10 == 0:
                    logger.info(
                        f"InstanSeg epoch {ep_total} "
                        f"(main {epoch + 1}/{n_epochs}): loss={mean_loss:.4f}"
                    )

        if val_tensors and patience > 0:
            stopper.restore_best(nn_model)

        if freeze_enc and hasattr(nn_model, "encoder"):
            for p in nn_model.encoder.parameters():
                p.requires_grad_(True)

        # ── 10. Re-wrap in InstanSeg_Torchscript ──────────────────────────
        nn_model.cpu().eval()
        # Deliberately NOT wrapped in try/except.  A bare InstanSeg_UNet.forward()
        # takes only `x`, while predict_single calls the model with
        # target_segmentation=, min_size= and the post-processing thresholds, so
        # falling back to the unwrapped net would only fail later.  A failed wrap
        # is fatal; let it say so here.
        # Carry the pretrained checkpoint's own post-processing defaults across to
        # the re-wrapped model.  InstanSeg_Torchscript(params=None) sets
        # self.params = {} and every default_* falls back to the library's generic
        # constants, so without this the fine-tuned model would be evaluated in a
        # different threshold regime from the checkpoint it started from — for no
        # stated reason.  Read defensively: if the loaded module does not expose
        # them, we pass nothing and behaviour is unchanged.
        #
        # min_size is deliberately NOT carried over: it is a dataset property
        # and the benchmark's value must win.
        _prev_params = {}
        for _key in ("mask_threshold", "seed_threshold"):
            _val = getattr(self._model, f"default_{_key}", None)
            if _val is not None:
                _prev_params[_key] = float(_val)
        if _prev_params:
            logger.info(
                f"InstanSeg: carrying checkpoint post-processing defaults into the "
                f"re-wrapped model: {_prev_params}"
            )

        ts_wrapper = InstanSeg_Torchscript(
            model=nn_model,
            cells_and_nuclei=ckpt_c_n,
            pixel_size=self.pixel_size or 0.0,
            n_sigma=self.n_sigma,
            dim_coords=self.dim_coords,
            dim_seeds=self.dim_seeds,
            backbone_dim_in=ckpt_in_ch,
            feature_engineering_function=self.feat_eng_fn,
            params=_prev_params or None,
        )
        self._model = ts_wrapper.to(self.device).eval()
        logger.info("InstanSeg: wrapped fine-tuned model in InstanSeg_Torchscript.")

        # Re-fit the mask/seed thresholds for the fine-tuned weights, on the
        # SUPPORT set only (never the test set — that would leak test labels).
        cal = self.calibrate_thresholds(support_images, support_masks)

        return {
            "threshold_calibration": cal,
            "n_epochs_run":        len(train_losses),
            "n_epochs_max":        hotstart_epochs + n_epochs,
            "best_epoch":          best_epoch,
            "early_stopping_mode": "patience_restore_best_weights",
            "val_metric":          "val_loss",
            "best_val_loss":       stopper.best if (val_tensors and patience > 0) else float("inf"),
            "n_train_images":      n_train,
            "n_train_patches":     n_train,   # whole images, one pass/epoch (no tiling)
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
            "train_losses":        train_losses,
            "val_losses":          val_losses,
            "hotstart_epochs":     hotstart_epochs,
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_param_count(self) -> int:
        if self._model is None:
            return -1
        try:
            inner = getattr(self._model, "fcn", self._model)
            return sum(p.numel() for p in inner.parameters())
        except Exception:
            try:
                return sum(p.numel() for p in self._model.parameters())
            except Exception:
                return -1

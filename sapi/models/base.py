"""Abstract base class and shared training utilities for model wrappers.

The benchmark passes each model raw, Layer-A-corrected images (see
sapi/datasets/cellbindb.py).  Each model is responsible for its own
normalisation and augmentation.  The helpers here are lightweight building
blocks the wrappers may use inside their own training loops:

  * _EarlyStopper          - patience-based early stopping with best-weight restore
  * split_into_patches     - tile an image+mask into fixed-size patches

Augmentation is NOT here: each model applies its own native augmentation inside
its wrapper (Cellpose/StarDist native augmenters; MEDIAR via MONAI; HoverNet and
CellViT via albumentations; MicroSAM via kornia flips; InstanSeg via numpy).
"""

from __future__ import annotations

import abc
import copy
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class _EarlyStopper:
    """Track validation loss; stop when it stops improving and restore best weights.

    patience=0 disables early stopping (step() always returns False).
    """

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self._best_state: dict | None = None

    def step(self, val_loss: float, state_dict=None) -> bool:
        """Call once per epoch. Returns True when training should stop.

        state_dict is optional; pass it only when restore_best() will be used
        instead of a separate best-weight snapshot.
        """
        if self.patience == 0:
            return False
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            if state_dict is not None:
                self._best_state = copy.deepcopy(state_dict)
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model) -> None:
        """Reload the best observed weights into model (in-place)."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def split_into_patches(image: np.ndarray, mask: np.ndarray, patch_size: int) -> list:
    """Tile image+mask into patch_size x patch_size patches covering the image.

    Stride = patch_size; the last position per axis is anchored at
    (dim - patch_size) so the final patch reaches the edge (minimal overlap when
    the dimension is not a multiple of patch_size).  Images smaller than
    patch_size are centre-padded with zeros to a single patch.
    """
    H, W = image.shape[:2]
    ps = patch_size
    if H < ps or W < ps:
        return [_pad_to_patch_size(image, mask, ps)]

    def _positions(length: int) -> list:
        pos = list(range(0, length - ps + 1, ps))
        last = length - ps
        if last > 0 and last not in pos:
            pos.append(last)
        return pos

    return [
        (image[y:y + ps, x:x + ps].copy(), mask[y:y + ps, x:x + ps].copy())
        for y in _positions(H)
        for x in _positions(W)
    ]


def _pad_to_patch_size(image: np.ndarray, mask: np.ndarray, patch_size: int):
    """Centre-pad image and mask to patch_size x patch_size with zeros."""
    H, W = image.shape[:2]
    ps = patch_size
    ph, pw = max(0, ps - H), max(0, ps - W)
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    pad = ((pt, pb), (pl, pr)) if image.ndim == 2 else ((pt, pb), (pl, pr), (0, 0))
    image = np.pad(image, pad, mode="constant", constant_values=0)
    mask = np.pad(mask, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)
    return image, mask


def gen_instance_hv_map(mask: np.ndarray) -> np.ndarray:
    """Horizontal/vertical distance maps for HoVer-style targets.

    Faithful port of the reference per-instance loop, which is **byte-identical**
    between HoVer-Net (`models/hovernet/targets.py::gen_instance_hv_map`) and
    CellViT (`cell_segmentation/datasets/pannuke.py::gen_instance_hv_map`).  Used
    by both wrappers so the target convention cannot drift between them.

    Returns float32 (H, W, 2): channel 0 = horizontal offset, 1 = vertical.  For
    each instance: signed distance from the **int-rounded** centroid
    (``int(com + 0.5)``) on a **1-based** pixel grid (``arange(1, N+1) - com``),
    then **independent half-normalisation** (negative side ÷ -min, positive side
    ÷ max) so both extremes reach ±1 even for off-centre cells.

    This replicates the reference's **per-instance bounding-box-crop** form
    exactly (compute the grid and centroid inside each instance's 2px-expanded
    bbox crop) rather than a full-image grid — the two are algebraically equal
    except at ``int(com + 0.5)`` rounding boundaries, where ``center_of_mass``
    on the crop vs the full image can differ in the last floating-point bit and
    flip the rounded centroid by a pixel.  The reference's HoVer-Net-training-only
    steps (``fix_mirror_padding``, ``remove_small_objects(min_size=30)``) are
    intentionally omitted: the benchmark keeps every GT instance
    (min_size=1).
    """
    from scipy.ndimage import center_of_mass

    def _bbox(binary):
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [rmin, rmax + 1, cmin, cmax + 1]

    orig = mask
    H, W = orig.shape[:2]
    x_map = np.zeros((H, W), dtype=np.float32)
    y_map = np.zeros((H, W), dtype=np.float32)

    for inst_id in np.unique(orig):
        if inst_id == 0:
            continue
        inst_full = np.array(orig == inst_id, np.uint8)
        box = _bbox(inst_full)
        # expand the box by 2px (guarded to stay in-image), as the reference does
        if box[0] >= 2:
            box[0] -= 2
        if box[2] >= 2:
            box[2] -= 2
        if box[1] <= H - 2:
            box[1] += 2
        if box[3] <= W - 2:
            box[3] += 2
        inst = inst_full[box[0]:box[1], box[2]:box[3]]
        if inst.shape[0] < 2 or inst.shape[1] < 2:
            continue

        com = list(center_of_mass(inst))               # centroid within the crop
        com[0] = int(com[0] + 0.5)
        com[1] = int(com[1] + 0.5)

        x_range = np.arange(1, inst.shape[1] + 1) - com[1]
        y_range = np.arange(1, inst.shape[0] + 1) - com[0]
        inst_x, inst_y = np.meshgrid(x_range, y_range)
        inst_x[inst == 0] = 0
        inst_y[inst == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        x_map[box[0]:box[1], box[2]:box[3]][inst > 0] = inst_x[inst > 0]
        y_map[box[0]:box[1], box[2]:box[3]][inst > 0] = inst_y[inst > 0]

    return np.stack([x_map, y_map], axis=-1)   # (H, W, 2): [horizontal, vertical]


# ---------------------------------------------------------------------------
# Post-processing threshold calibration
# ---------------------------------------------------------------------------

def calibrate_thresholds_on_support(
    predict_fn,
    images: list,
    masks: list,
    grid: dict,
    model_name: str = "",
) -> dict:
    """Grid-search post-processing thresholds on labelled SUPPORT images.

    Every model in the benchmark turns a continuous network output into instances
    by comparing it against fixed numeric thresholds (cell probability, flow
    error, seed/centre distance, ...).  Those published defaults are tuned for
    the PRETRAINED checkpoint.  Fine-tuning shifts the output distribution while
    the training loss — which is threshold-free — barely registers it, so the
    thresholded instance set can change enormously for a model that still fits
    its targets perfectly.  Left uncorrected this measures threshold staleness
    rather than model quality (measured on MicroSAM: AJI+ 0.50 -> 0.08).

    Calibrate on the SUPPORT set only.  Those are the only labels a few-shot
    model is entitled to see, which makes this part of adaptation; using test
    images would leak test labels into the evaluation.

    Args:
        predict_fn: ``fn(image, **params) -> uint16 instance mask``.
        images:     support images, in the format the model's predict path takes.
        masks:      matching ground-truth instance masks.
        grid:       ``{param_name: [candidate, ...]}``; the full cross-product is
                    searched, so keep it small — this re-runs inference per combo.
        model_name: for the log line only.

    Returns:
        ``{**best_params, "cal_aji": float, "n_cal_images": int,
           "n_combos_searched": int}``.  On a tie the first combo in
        cross-product order wins, so list each parameter's default first to make
        the search default-preserving.
    """
    from itertools import product

    from sapi.metrics import compute_all_metrics

    names = list(grid)
    combos = [dict(zip(names, values)) for values in product(*(grid[n] for n in names))]

    best_params, best_aji = combos[0], -1.0
    for params in combos:
        preds = [np.asarray(predict_fn(img, **params), dtype=np.uint16) for img in images]
        aji = compute_all_metrics(preds, masks).summary_dict()["aji_plus"]
        if aji > best_aji:
            best_aji, best_params = aji, params

    logger.info(
        "[%s] thresholds calibrated on %d support images  %s  cal_AJI=%.4f  "
        "(%d combos searched)",
        model_name or "model",
        len(images),
        "  ".join(f"{k}={v}" for k, v in best_params.items()),
        best_aji,
        len(combos),
    )
    return {
        **best_params,
        "cal_aji": float(best_aji),
        "n_cal_images": len(images),
        "n_combos_searched": len(combos),
    }


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseSegmentationModel(abc.ABC):
    """Uniform interface for every model in the benchmark.

    Subclasses implement predict_single (and optionally finetune).  Each model
    normalises and augments its own inputs; the benchmark supplies raw,
    Layer-A-corrected float32 images.

    Attributes:
        model_name          human-readable identifier
        supports_finetune   whether fine-tuning is implemented
        num_channels        expected input channels (1 or 3)
        needs_raw_he_image  True for models trained on full RGB H&E
    """

    model_name: str = "base"
    supports_finetune: bool = False
    num_channels: int = 1
    needs_raw_he_image: bool = False

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device
        self._model = None  # loaded lazily

    @abc.abstractmethod
    def predict_single(self, image: np.ndarray) -> np.ndarray:
        """Segment one image -> uint16 instance label map (0 = background)."""

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Segment a list of images. Override for true GPU batching."""
        return [self.predict_single(img) for img in images]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Match the model's expected channel count (replicate gray->RGB or
        luminance RGB->gray). Intensity is left untouched."""
        if image.ndim == 2:
            gray = image
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = (
                0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            ).astype(image.dtype)
            rgb = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        return gray if self.num_channels == 1 else rgb

    def finetune(self, support_images, support_masks, **kwargs) -> dict:
        raise NotImplementedError(f"{self.model_name} does not support fine-tuning.")

    def get_param_count(self) -> int:
        """Return the model's parameter count, or -1 if unavailable.

        The -1 sentinel indicates the model couldn't be loaded or doesn't expose
        parameter count information. Downstream code (e.g., SAPI scoring) checks
        `if param_count > 0` before using the value.
        """
        return -1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, finetune={self.supports_finetune})"

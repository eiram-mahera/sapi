"""
Instance segmentation metrics for cell segmentation benchmarking.

Only the three metrics that constitute the SAPI performance term *P* are
computed and stored:

AJI+   Aggregated Jaccard Index, plus variant (Graham et al., 2019) — AJI with
       optimal one-to-one (Hungarian) matching between predicted and ground-truth
       instances, which removes the over-penalisation of the original greedy AJI
       (Kumar et al., 2017). No IoU threshold.
PQ     Panoptic Quality = SQ × RQ (Kirillov et al., 2019). Instances are paired
       at a strict IoU > 0.5.
NS     Normalised Surface Dice at tolerance τ = 2 px (Nikolov et al., 2021),
       abbreviated NSD in the code. Computed on the MERGED foreground of each
       mask — every instance is combined into a single binary region, so there is
       NO instance pairing and NO IoU threshold. It scores agreement of the
       foreground outline only and is by construction insensitive to instance
       splits and merges; those are captured by AJI+ and by PQ's RQ term. See
       compute_nsd() for the full statement of this property.

All three satisfy 0 <= x <= 1 with larger = better, which is what makes the
harmonic mean below well defined.

Required aggregation order
--------------------------
Metrics are computed PER IMAGE, then each metric is averaged over the dataset
INDEPENDENTLY (macro-average — every image carries equal weight regardless of
its cell count), and only then is the composite score P formed, as the harmonic
mean of the three DATASET-level means:

    A = mean_i(a_i),  Q = mean_i(q_i),  N = mean_i(n_i)      over i in I
    P = 0                                  if min(A, Q, N) = 0
    P = 3 / (1/A + 1/Q + 1/N)              otherwise

where I is the set of included images. P is computed by
compute_performance_score() in sapi/metrics/sapi_score.py.

PROHIBITED (do not "optimise" the code into any of these):
  * computing a per-image P and averaging it — HM(mean a, mean q, mean n) is NOT
    mean_i HM(a_i, q_i, n_i), and only the former is correct here;
  * forming the harmonic mean before all three dataset means exist;
  * pooling pixels, matched instances, intersections, unions, TP/FP/FN across the
    dataset before computing the metrics (that would weight cell-dense images
    more heavily — a micro-average, which is a different analysis);
  * computing the three means from different subsets of images;
  * dropping valid zero-valued metrics, or adding an epsilon to avoid a zero
    harmonic mean.

Instance ids
------------
Each positive integer is taken to identify exactly ONE instance: a label's pixels
are pooled into a single instance and paired as one. Ground truth satisfies this
(all 102,480 CellBinDB instance labels are a single 8-connected component, after
repairing a handful of malformed samples), but predictions do not always —
Cellpose emits a label covering disconnected regions for roughly 0.4% of
predicted instances, whose fragments are then pooled and paired as one instance.
This is NOT enforced or corrected here, by explicit decision — think carefully
before adding a relabelling step.

Empty masks (resolved BEFORE any metric function is called, so an expected empty
case can never produce NaN)
--------------------------------------------------------------------------------
    GT has instances, prediction empty    AJI+ = PQ = NS = 0.0, image included.
                                          A valid model failure, not missing data.
    GT empty, prediction has instances    AJI+ = PQ = NS = 0.0, image included.
                                          A false-positive prediction.
    GT and prediction both non-empty      AJI+ = 0.0 and PQ = 0.0 (no true
      but no valid instance matches         positives); NS is computed from its
                                            own definition and is NOT forced to 0.
    both empty                            image marked both_empty and EXCLUDED
                                            from all three dataset means; the
                                            count is recorded in num_both_empty.
                                          NOT scored 1.0 — a perfect score would
                                            let trivially empty images inflate
                                            reported performance. No CellBinDB
                                            mask is empty, so this branch is
                                            implemented but should not be reached.

Zero is a valid result and is never softened: a 0.0 is NOT replaced by an
epsilon, NOT replaced by NaN, NOT dropped before averaging, and does NOT cause
the image to be removed. At dataset level min(A, Q, N) = 0 implies P = 0.

NaN / infinite values
---------------------
Never silently coerced to 0. Expected empty-mask cases are handled above and
cannot reach the metric functions, so on two valid non-empty masks a NaN, an
infinity, or a materially out-of-range value indicates an implementation or data
error (invalid label map, unhandled zero denominator, overflow, failed matching).
Such a value raises MetricError, which identifies the image and the metric and
carries the offending masks, and stops aggregation for the run — so no
dataset-level score is written from a broken image set. Only the failed metric or
the failed image is NEVER dropped: all three means always come from one image set.

Range validation
----------------
Every computed metric is checked to be finite and within [0, 1]. A deviation of
at most _RANGE_TOL (1e-12) is treated as floating-point drift and clipped (e.g.
1 + 1e-15 -> 1.0); anything beyond that is materially invalid and raises.

Per-image records
-----------------
compute_all_metrics() returns one record per evaluated image (image id, instance
counts, the three metrics, inclusion status, exclusion reason). Callers add the
fields the metrics module cannot know — model, variant, dataset, condition, mode,
seed, tissue group.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import surface_distance as _sd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

#: Tolerance for floating-point drift in the [0, 1] range check.  Used ONLY to
#: clip drift (1 + 1e-15 -> 1.0); a materially invalid value must never be
#: clipped into range, it must raise.
_RANGE_TOL: float = 1e-12


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MetricError(RuntimeError):
    """An invalid metric value or mask on a pair that should have been valid.

    Raised instead of coercing to 0.0 or NaN.  Expected empty-mask cases are
    resolved before the metric functions are called, so an invalid value here
    means an implementation or data error — it must stop the run rather than
    silently enter an aggregate.

    The offending masks are attached so a caller or debugger can inspect them
    without re-running the model.
    """

    def __init__(
        self,
        message: str,
        *,
        image_id: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[float] = None,
        gt_mask: Optional[np.ndarray] = None,
        pred_mask: Optional[np.ndarray] = None,
    ):
        super().__init__(message)
        self.image_id = image_id
        self.metric = metric
        self.value = value
        self.gt_mask = gt_mask
        self.pred_mask = pred_mask


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerImageMetrics:
    """One image-level record.

    The caller adds the identity fields the metrics module cannot know (model,
    variant, dataset, condition, mode, seed, tissue group).

    metric_error is always None in a completed run: an invalid metric raises
    MetricError and the run is aborted, so a failed image never reaches a stored
    record. The field exists so the record schema states that explicitly.
    """
    image_id: str = ""
    n_gt_instances: int = 0
    n_pred_instances: int = 0
    aji_plus: float = 0.0
    pq: float = 0.0
    nsd: float = 0.0
    included: bool = True
    exclusion_reason: Optional[str] = None
    metric_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstanceSegMetrics:
    """Run-level summary over one model-dataset-condition-mode-seed test set."""
    # Dataset-level means over the INCLUDED images (A, Q, N in the formula above)
    aji_plus: float = 0.0
    pq: float = 0.0
    nsd: float = 0.0
    # Standard deviations over the same included images
    aji_plus_std: float = 0.0
    pq_std: float = 0.0
    nsd_std: float = 0.0
    # Image accounting.  num_included + num_both_empty + num_failed == num_images.
    # num_failed is always 0 in a completed run (an invalid metric raises), and is
    # reported so the summary states that rather than leaving it implicit.
    num_images: int = 0        # number of test images evaluated
    num_included: int = 0      # images contributing to all three means
    num_both_empty: int = 0    # excluded: GT and prediction both empty
    num_failed: int = 0
    per_image: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_dict(self) -> dict:
        """Return the run-level summary without the per-image list (for logging)."""
        d = self.to_dict()
        d.pop("per_image", None)
        return d


# ---------------------------------------------------------------------------
# Mask validation
# ---------------------------------------------------------------------------

def _validate_label_map(mask, role: str, image_id: Optional[str]) -> np.ndarray:
    """Return *mask* as a validated 2-D integer instance-label array.

    A valid mask is two-dimensional, has background = 0, and labels each instance
    with a distinct positive integer.  Anything else raises MetricError rather
    than being silently coerced.  Integer masks pass through untouched; integral
    float/bool masks are converted to int64 — wide enough that a large label ID
    cannot wrap (a narrow cast such as uint16 would wrap above 65,535).
    """
    arr = np.asarray(mask)

    if arr.ndim != 2:
        raise MetricError(
            f"{role} mask for image {image_id!r} is not a 2-D instance-label "
            f"image (ndim={arr.ndim}, shape={arr.shape}).",
            image_id=image_id, metric="label_map",
        )

    if arr.dtype.kind == "b":
        arr = arr.astype(np.int64)
    elif arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise MetricError(
                f"{role} mask for image {image_id!r} contains NaN or infinite "
                f"values; an instance-label map must be finite integers.",
                image_id=image_id, metric="label_map",
            )
        if not np.all(arr == np.rint(arr)):
            raise MetricError(
                f"{role} mask for image {image_id!r} has non-integer labels; "
                f"each instance must be identified by a positive integer.",
                image_id=image_id, metric="label_map",
            )
        arr = np.rint(arr).astype(np.int64)
    elif arr.dtype.kind not in "iu":
        raise MetricError(
            f"{role} mask for image {image_id!r} has unsupported dtype "
            f"{arr.dtype}; expected an integer instance-label map.",
            image_id=image_id, metric="label_map",
        )

    if arr.size and int(arr.min()) < 0:
        raise MetricError(
            f"{role} mask for image {image_id!r} contains negative labels "
            f"(min={int(arr.min())}); background must be 0 and instances positive.",
            image_id=image_id, metric="label_map",
        )
    return arr


def _check_metric_value(
    value: float,
    metric: str,
    image_id: Optional[str],
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    """Validate a computed metric: finite, in [0, 1]; clip drift only."""
    v = float(value)

    if not math.isfinite(v):
        raise MetricError(
            f"{metric} = {v} for image {image_id!r}, whose ground truth and "
            f"prediction are both non-empty. An expected empty-mask case cannot "
            f"reach here, so this is an implementation or data error (invalid "
            f"label map, unhandled zero denominator, overflow, or a failed "
            f"matching procedure), not a model failure. Aggregation stopped; no "
            f"dataset-level score written for this run.",
            image_id=image_id, metric=metric, value=v,
            gt_mask=gt_mask, pred_mask=pred_mask,
        )

    if v < -_RANGE_TOL or v > 1.0 + _RANGE_TOL:
        raise MetricError(
            f"{metric} = {v!r} for image {image_id!r} lies outside [0, 1] by more "
            f"than the floating-point tolerance {_RANGE_TOL:g}. A materially "
            f"out-of-range value is not clipped — it indicates an implementation "
            f"error. Aggregation stopped; no dataset-level score written.",
            image_id=image_id, metric=metric, value=v,
            gt_mask=gt_mask, pred_mask=pred_mask,
        )

    # Within tolerance: floating-point drift only (e.g. 1 + 1e-15 -> 1.0).
    return min(1.0, max(0.0, v))


# ---------------------------------------------------------------------------
# Core overlap utilities
# ---------------------------------------------------------------------------

def _get_instance_ids(mask: np.ndarray) -> np.ndarray:
    """Return unique positive integer cell IDs from an instance mask."""
    ids = np.unique(mask)
    return ids[ids > 0]


def _instance_areas(mask: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return the pixel area of each instance in *ids*, aligned to *ids*."""
    if len(ids) == 0:
        return np.zeros(0, dtype=np.float64)
    uniq, counts = np.unique(mask, return_counts=True)
    lookup = {int(u): int(c) for u, c in zip(uniq, counts)}
    return np.array([lookup[int(i)] for i in ids], dtype=np.float64)


@dataclass
class _Overlap:
    """Pairwise overlap between the GT and predicted instances of one image.

    All matrices are (n_gt, n_pred). Computed ONCE per image and shared by AJI+
    and PQ: both need the same pairwise quantities, so computing them separately
    would repeat the work and — worse — leave two independent definitions of the
    same numbers free to drift apart.
    """
    inter: np.ndarray       # (n_gt, n_pred) intersection areas
    union: np.ndarray       # (n_gt, n_pred) union areas
    iou: np.ndarray         # (n_gt, n_pred) intersection over union
    gt_ids: np.ndarray
    pred_ids: np.ndarray
    gt_areas: np.ndarray    # (n_gt,)
    pred_areas: np.ndarray  # (n_pred,)

    @property
    def n_gt(self) -> int:
        return len(self.gt_ids)

    @property
    def n_pred(self) -> int:
        return len(self.pred_ids)


def _pairwise_overlap(pred_mask: np.ndarray, gt_mask: np.ndarray) -> _Overlap:
    """Compute all pairwise intersections, unions and IoUs for one image."""
    gt_ids = _get_instance_ids(gt_mask)
    pred_ids = _get_instance_ids(pred_mask)
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    gt_areas = _instance_areas(gt_mask, gt_ids)
    pred_areas = _instance_areas(pred_mask, pred_ids)

    inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt == 0 or n_pred == 0:
        return _Overlap(
            inter=inter,
            union=np.zeros((n_gt, n_pred), dtype=np.float64),
            iou=np.zeros((n_gt, n_pred), dtype=np.float64),
            gt_ids=gt_ids,
            pred_ids=pred_ids,
            gt_areas=gt_areas,
            pred_areas=pred_areas,
        )

    # Intersections in one pass: label every co-occupied pixel with a unique
    # (gt, pred) key and count the keys, instead of re-scanning the masks once
    # per instance pair.
    both = (gt_mask > 0) & (pred_mask > 0)
    if both.any():
        gt_pos = {int(g): i for i, g in enumerate(gt_ids)}
        pred_pos = {int(p): j for j, p in enumerate(pred_ids)}
        stride = int(pred_ids.max()) + 1
        keys, counts = np.unique(
            gt_mask[both].astype(np.int64) * stride + pred_mask[both].astype(np.int64),
            return_counts=True,
        )
        for key, count in zip(keys, counts):
            inter[gt_pos[int(key) // stride], pred_pos[int(key) % stride]] = count

    union = gt_areas[:, None] + pred_areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return _Overlap(
        inter=inter,
        union=union,
        iou=iou,
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_areas=gt_areas,
        pred_areas=pred_areas,
    )


def _greedy_match(
    iou_matrix: np.ndarray, iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy pairing of predictions to GT by descending IoU. Used by PQ only —
    AJI+ pairs by optimal assignment and NS does not pair at all.

    A pair is matched only when IoU > iou_threshold (STRICT, not >=). At the
    0.5 level this is the canonical Panoptic Quality rule (Kirillov et al.,
    2019): IoU > 0.5 admits at most one pairing per instance, so the result is
    unique regardless of iteration order. For thresholds below 0.5 the pairing
    is genuinely greedy and order-dependent — the uniqueness guarantee does not
    carry over.

    Args:
        iou_matrix: (n_pred, n_gt) IoU matrix.

    Returns:
        matched_pairs:  list of (pred_idx, gt_idx)
        unmatched_pred: indices of unmatched predictions
        unmatched_gt:   indices of unmatched GT
    """
    n_pred, n_gt = iou_matrix.shape
    matched_pred: set = set()
    matched_gt: set = set()
    matched_pairs: List[Tuple[int, int]] = []

    if iou_matrix.size:
        # Only pairs above the threshold can ever match, so rank those alone
        # rather than sorting every pair in the matrix.
        cand = np.argwhere(iou_matrix > iou_threshold)
        order = np.argsort(-iou_matrix[cand[:, 0], cand[:, 1]], kind="stable")
        for i, j in cand[order]:
            i, j = int(i), int(j)
            if i not in matched_pred and j not in matched_gt:
                matched_pairs.append((i, j))
                matched_pred.add(i)
                matched_gt.add(j)

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]
    return matched_pairs, unmatched_pred, unmatched_gt


# ---------------------------------------------------------------------------
# Aggregated Jaccard Index, plus variant (AJI+)
# ---------------------------------------------------------------------------

def _aji_plus_from_overlap(ov: _Overlap) -> float:
    """AJI+ from a precomputed _Overlap. See compute_aji_plus() for the definition."""
    if ov.n_gt == 0 and ov.n_pred == 0:
        return float("nan")  # undefined; the caller excludes the image
    if ov.n_gt == 0 or ov.n_pred == 0:
        return 0.0

    # Optimal one-to-one assignment (maximise total IoU).
    row_ind, col_ind = linear_sum_assignment(-ov.iou)
    keep = ov.iou[row_ind, col_ind] > 0.0
    row_ind, col_ind = row_ind[keep], col_ind[keep]

    overall_inter = ov.inter[row_ind, col_ind].sum()
    overall_union = ov.union[row_ind, col_ind].sum()

    paired_gt = set(row_ind.tolist())
    paired_pred = set(col_ind.tolist())
    for i in range(ov.n_gt):
        if i not in paired_gt:
            overall_union += ov.gt_areas[i]
    for j in range(ov.n_pred):
        if j not in paired_pred:
            overall_union += ov.pred_areas[j]

    return float(overall_inter / overall_union) if overall_union > 0 else 0.0


def compute_aji_plus(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    AJI+ — Aggregated Jaccard Index with maximal unique pairing (Graham et al.,
    2019), the over-penalisation-free variant of the original AJI (Kumar et al.,
    2017).

        AJI+ = ( Σ_(g,p)∈M |g ∩ p| ) / ( Σ_(g,p)∈M |g ∪ p| + Σ_(unpaired) |·| )

    where M is the optimal one-to-one assignment between GT and predicted
    instances that maximises total IoU (Hungarian algorithm). Every prediction
    is paired with at most one GT instance, unlike plain AJI where a prediction
    may be claimed by several GT instances (inflating the union). Areas of all
    unpaired GT and predicted instances are added to the union. No IoU threshold.

    Two non-empty masks whose instances do not overlap at all score 0.0.

    Returns:
        nan  if both masks are empty (undefined; compute_all_metrics excludes
             such images rather than calling this).
        0.0  if exactly one mask is empty.
        value in [0.0, 1.0] otherwise.
    """
    gt = _validate_label_map(gt_mask, "ground-truth", None)
    pred = _validate_label_map(pred_mask, "predicted", None)
    _check_same_shape(pred, gt, None)
    return _aji_plus_from_overlap(_pairwise_overlap(pred, gt))


# ---------------------------------------------------------------------------
# Panoptic Quality (PQ = SQ × RQ)
# ---------------------------------------------------------------------------

def _pq_from_overlap(ov: _Overlap, iou_threshold: float = 0.5) -> float:
    """PQ from a precomputed _Overlap. See compute_pq() for the definition."""
    if ov.n_gt == 0 and ov.n_pred == 0:
        return float("nan")  # undefined; the caller excludes the image
    if ov.n_gt == 0 or ov.n_pred == 0:
        return 0.0

    # _Overlap matrices are (n_gt, n_pred); _greedy_match works in (n_pred, n_gt).
    iou_pred_gt = ov.iou.T
    matched_pairs, unmatched_pred, unmatched_gt = _greedy_match(
        iou_pred_gt, iou_threshold
    )

    TP = len(matched_pairs)
    FP = len(unmatched_pred)
    FN = len(unmatched_gt)
    matched_ious = [iou_pred_gt[i, j] for i, j in matched_pairs]

    SQ = float(np.mean(matched_ious)) if matched_ious else 0.0
    RQ = TP / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) > 0 else 0.0
    return SQ * RQ


def compute_pq(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """
    Panoptic Quality = SQ × RQ (Kirillov et al., 2019), paired at IoU > 0.5.

    SQ = mean IoU of matched pairs; RQ = TP / (TP + ½FP + ½FN).
    SQ and RQ are computed internally but only their product PQ is returned/stored.

    iou_threshold is exposed for completeness but the benchmark always uses the
    canonical 0.5. Below 0.5 the pairing becomes order-dependent — see
    _greedy_match().

    Two non-empty masks with no pair above the threshold score 0.0 (TP = 0), as
    does an empty prediction against a non-empty ground truth: both are valid
    model outputs, not missing values.

    Returns:
        nan  if both masks are empty (undefined; compute_all_metrics excludes
             such images rather than calling this).
        0.0  if exactly one mask is empty.
        value in [0.0, 1.0] otherwise.
    """
    gt = _validate_label_map(gt_mask, "ground-truth", None)
    pred = _validate_label_map(pred_mask, "predicted", None)
    _check_same_shape(pred, gt, None)
    return _pq_from_overlap(_pairwise_overlap(pred, gt), iou_threshold)


# ---------------------------------------------------------------------------
# Normalised Surface Dice (NS / NSD) via the DeepMind surface-distance library
# ---------------------------------------------------------------------------

def compute_nsd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: float = 2.0,
) -> float:
    """
    Normalised Surface Dice — the ORIGINAL semantic definition (Nikolov et al.,
    2021), computed once on the whole foreground surfaces.

    Each mask's foreground (all instances merged, label > 0) is treated as a
    single binary region. NSD is the fraction of surface elements of each
    boundary lying within `tolerance_px` of the other boundary, via DeepMind's
    official surface-distance library:
        compute_surface_distances() → compute_surface_dice_at_tolerance()
    spacing_mm=[1.0, 1.0] so tolerance_mm == tolerance in pixels.

    NO instance pairing and NO IoU threshold (unlike PQ). This measures the
    foreground boundary agreement of the whole image. Note the consequence for
    instance segmentation: the internal boundary between two correctly-separated
    touching cells is not part of the merged foreground surface, so semantic NSD
    is insensitive to such instance splits/merges — a prediction that merges two
    touching cells into one scores NSD = 1.0. Detection and instance errors are
    captured by PQ (RQ) and AJI+ instead; NSD scores the foreground outline.

    When both masks are non-empty but no instances match, NSD is NOT forced to 0
    — it is computed from this definition like any other image.

    Returns:
      nan  if both masks have no foreground (undefined; compute_all_metrics
           excludes such images rather than calling this), or if a foreground has
           no surface elements (a single-pixel-only region). The latter is
           undefined rather than 0 and is treated as an error by
           compute_all_metrics().
      0.0  if exactly one mask has foreground (surfaces fully disjoint).
      value in [0.0, 1.0] otherwise.
    """
    gt = _validate_label_map(gt_mask, "ground-truth", None)
    pred = _validate_label_map(pred_mask, "predicted", None)
    _check_same_shape(pred, gt, None)

    gt_bin = gt > 0
    pred_bin = pred > 0
    gt_has = bool(gt_bin.any())
    pred_has = bool(pred_bin.any())

    if not gt_has and not pred_has:
        return float("nan")
    if not gt_has or not pred_has:
        return 0.0

    surface_dists = _sd.compute_surface_distances(gt_bin, pred_bin, spacing_mm=[1.0, 1.0])
    # No surface elements (e.g. single-pixel-only foreground): undefined, not 0.
    if (len(surface_dists["surfel_areas_gt"]) == 0 or
            len(surface_dists["surfel_areas_pred"]) == 0):
        return float("nan")
    return float(_sd.compute_surface_dice_at_tolerance(surface_dists, tolerance_px))


# ---------------------------------------------------------------------------
# Per-image and dataset-level aggregation
# ---------------------------------------------------------------------------

def _check_same_shape(pred: np.ndarray, gt: np.ndarray, image_id: Optional[str]) -> None:
    if pred.shape != gt.shape:
        raise MetricError(
            f"Prediction and ground-truth shapes differ for image {image_id!r}: "
            f"{pred.shape} vs {gt.shape}.",
            image_id=image_id, metric="label_map",
        )


def compute_per_image_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    image_id: str = "",
) -> PerImageMetrics:
    """Compute the three P-constituent metrics for one image.

    Both masks are validated and the empty-mask cases are resolved BEFORE any
    metric function is called, so an expected empty case never produces NaN. An
    unexpected invalid value on two valid non-empty masks raises MetricError.
    """
    gt = _validate_label_map(gt_mask, "ground-truth", image_id)
    pred = _validate_label_map(pred_mask, "predicted", image_id)
    _check_same_shape(pred, gt, image_id)

    n_gt = len(_get_instance_ids(gt))
    n_pred = len(_get_instance_ids(pred))

    # ── Empty-mask handling, before the metric functions ────────────────────
    if n_gt == 0 and n_pred == 0:
        return PerImageMetrics(
            image_id=image_id, n_gt_instances=0, n_pred_instances=0,
            aji_plus=float("nan"), pq=float("nan"), nsd=float("nan"),
            included=False, exclusion_reason="both_empty",
        )
    if n_gt == 0 or n_pred == 0:
        # A valid model failure (or a false-positive prediction): a real 0.0,
        # not missing data.
        return PerImageMetrics(
            image_id=image_id, n_gt_instances=n_gt, n_pred_instances=n_pred,
            aji_plus=0.0, pq=0.0, nsd=0.0, included=True,
        )

    # ── Both masks non-empty ────────────────────────────────────────────────
    ov = _pairwise_overlap(pred, gt)
    aji_plus = _check_metric_value(_aji_plus_from_overlap(ov), "aji_plus", image_id, gt, pred)
    pq = _check_metric_value(_pq_from_overlap(ov), "pq", image_id, gt, pred)
    try:
        nsd_raw = compute_nsd(pred, gt)
    except MetricError:
        raise
    except Exception as exc:  # surface-distance failure: an error, not a 0
        raise MetricError(
            f"NSD computation raised for image {image_id!r}, whose ground truth "
            f"and prediction are both non-empty: {exc}. Aggregation stopped; no "
            f"dataset-level score written for this run.",
            image_id=image_id, metric="nsd", gt_mask=gt, pred_mask=pred,
        ) from exc
    nsd = _check_metric_value(nsd_raw, "nsd", image_id, gt, pred)

    return PerImageMetrics(
        image_id=image_id, n_gt_instances=n_gt, n_pred_instances=n_pred,
        aji_plus=aji_plus, pq=pq, nsd=nsd, included=True,
    )


def compute_all_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    image_ids: Optional[Sequence[str]] = None,
) -> InstanceSegMetrics:
    """
    Compute AJI+, PQ and NS per image, then the dataset-level macro-average of
    each metric independently over the included images.

    The three means always come from ONE image set: an image is either included
    in all three or excluded from all three (both_empty). P is NOT computed here
    — see compute_performance_score() in sapi_score.py, which takes the harmonic
    mean of the three means returned by this function.

    Raises:
        MetricError: on an invalid mask or an invalid metric value; aggregation
            stops and no dataset-level score is produced for the run.
        ValueError: on mismatched input lengths.
    """
    if len(pred_masks) != len(gt_masks):
        raise ValueError(
            f"Mismatched pred/GT count: {len(pred_masks)} predictions "
            f"vs {len(gt_masks)} ground-truth masks"
        )
    if image_ids is None:
        image_ids = [str(i) for i in range(len(pred_masks))]
    elif len(image_ids) != len(pred_masks):
        raise ValueError(
            f"Mismatched image_ids count: {len(image_ids)} ids "
            f"vs {len(pred_masks)} images"
        )

    per_image = [
        compute_per_image_metrics(pred, gt, image_id=str(iid))
        for pred, gt, iid in zip(pred_masks, gt_masks, image_ids)
    ]

    included = [m for m in per_image if m.included]
    num_both_empty = sum(1 for m in per_image if m.exclusion_reason == "both_empty")
    if num_both_empty:
        logger.warning(
            f"  {num_both_empty} image(s) excluded from the dataset means: "
            f"ground truth and prediction both empty."
        )

    def _mean(attr: str) -> float:
        vals = [getattr(m, attr) for m in included]
        return float(np.mean(vals)) if vals else float("nan")

    def _std(attr: str) -> float:
        vals = [getattr(m, attr) for m in included]
        return float(np.std(vals)) if vals else float("nan")

    return InstanceSegMetrics(
        aji_plus=_mean("aji_plus"),
        pq=_mean("pq"),
        nsd=_mean("nsd"),
        aji_plus_std=_std("aji_plus"),
        pq_std=_std("pq"),
        nsd_std=_std("nsd"),
        num_images=len(per_image),
        num_included=len(included),
        num_both_empty=num_both_empty,
        num_failed=0,  # an invalid metric raises, so a completed run has none
        per_image=[m.to_dict() for m in per_image],
    )

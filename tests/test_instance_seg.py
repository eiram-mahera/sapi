"""
Validation tests for the segmentation-performance metrics.

Covers the twelve cases required by the metrics implementation guideline, in
order:

  1. identical non-empty masks -> every metric at its maximum
  2. non-empty ground truth, empty prediction
  3. empty ground truth, non-empty prediction
  4. two empty masks
  5. two non-empty masks with no valid matches
  6. a valid image with one metric equal to zero
  7. an invalid label map
  8. an unexpected NaN
  9. a metric slightly above one through floating-point drift
 10. a materially out-of-range metric
 11. a dataset in which one dataset-level mean equals zero
 12. HM(mean a, mean q, mean n) != mean_i HM(a_i, q_i, n_i) in general

Plus the aggregation-order contract itself and the value-identity of AJI+/PQ
against an independent brute-force oracle.

Run:  python -m pytest tests/test_instance_seg.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sapi.metrics.instance_seg import (
    MetricError,
    compute_aji_plus,
    compute_all_metrics,
    compute_nsd,
    compute_per_image_metrics,
    compute_pq,
    _check_metric_value,
)
from sapi.metrics.sapi_score import compute_performance_score, harmonic_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 64


def empty_mask(size: int = SIZE) -> np.ndarray:
    return np.zeros((size, size), np.uint16)


def two_cells(size: int = SIZE) -> np.ndarray:
    """Two well-separated square instances."""
    m = np.zeros((size, size), np.uint16)
    m[8:20, 8:20] = 1
    m[36:48, 36:48] = 2
    return m


def brute_force_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Independent naive oracle for pairwise IoU, shape (n_gt, n_pred)."""
    g_ids = [i for i in np.unique(gt) if i > 0]
    p_ids = [i for i in np.unique(pred) if i > 0]
    out = np.zeros((len(g_ids), len(p_ids)))
    for a, g in enumerate(g_ids):
        for b, p in enumerate(p_ids):
            G, P = gt == g, pred == p
            union = (G | P).sum()
            out[a, b] = (G & P).sum() / union if union else 0.0
    return out


# ---------------------------------------------------------------------------
# 1. Identical non-empty masks -> maximum on every metric
# ---------------------------------------------------------------------------

def test_01_identical_masks_score_maximum():
    gt = two_cells()
    rec = compute_per_image_metrics(gt.copy(), gt, image_id="identical")
    assert rec.aji_plus == pytest.approx(1.0)
    assert rec.pq == pytest.approx(1.0)
    assert rec.nsd == pytest.approx(1.0)
    assert rec.included is True
    assert rec.exclusion_reason is None
    assert rec.n_gt_instances == 2 and rec.n_pred_instances == 2
    assert compute_performance_score(
        {"aji_plus": rec.aji_plus, "pq": rec.pq, "nsd": rec.nsd}
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Non-empty ground truth, empty prediction -> a valid zero, image included
# ---------------------------------------------------------------------------

def test_02_empty_prediction_scores_zero_and_is_included():
    rec = compute_per_image_metrics(empty_mask(), two_cells(), image_id="pred_empty")
    assert (rec.aji_plus, rec.pq, rec.nsd) == (0.0, 0.0, 0.0)
    # A model failure is a real result, never missing data.
    assert not any(math.isnan(v) for v in (rec.aji_plus, rec.pq, rec.nsd))
    assert rec.included is True
    assert rec.exclusion_reason is None
    assert rec.n_gt_instances == 2 and rec.n_pred_instances == 0


# ---------------------------------------------------------------------------
# 3. Empty ground truth, non-empty prediction -> false positive, scores zero
# ---------------------------------------------------------------------------

def test_03_empty_gt_with_prediction_scores_zero_and_is_included():
    rec = compute_per_image_metrics(two_cells(), empty_mask(), image_id="gt_empty")
    assert (rec.aji_plus, rec.pq, rec.nsd) == (0.0, 0.0, 0.0)
    assert rec.included is True
    assert rec.n_gt_instances == 0 and rec.n_pred_instances == 2


# ---------------------------------------------------------------------------
# 4. Two empty masks -> both_empty, excluded, never scored 1.0
# ---------------------------------------------------------------------------

def test_04_both_empty_is_excluded_not_perfect():
    rec = compute_per_image_metrics(empty_mask(), empty_mask(), image_id="both_empty")
    assert rec.included is False
    assert rec.exclusion_reason == "both_empty"
    assert all(math.isnan(v) for v in (rec.aji_plus, rec.pq, rec.nsd))
    # Explicitly NOT a perfect score: that would let empty images inflate results.
    assert not any(v == 1.0 for v in (rec.aji_plus, rec.pq, rec.nsd))


def test_04b_both_empty_excluded_from_dataset_means_and_counted():
    gt = [two_cells(), two_cells(), empty_mask()]
    pred = [two_cells(), two_cells(), empty_mask()]
    seg = compute_all_metrics(pred, gt, image_ids=["a", "b", "c"])
    assert seg.num_images == 3
    assert seg.num_included == 2
    assert seg.num_both_empty == 1
    assert seg.num_failed == 0
    # Mean is over the 2 included images only, not 3.
    assert seg.aji_plus == pytest.approx(1.0)
    assert seg.num_included + seg.num_both_empty + seg.num_failed == seg.num_images


# ---------------------------------------------------------------------------
# 5. Two non-empty masks with no valid matches
# ---------------------------------------------------------------------------

def test_05_no_valid_matches_zeroes_aji_and_pq_but_not_ns():
    gt = np.zeros((SIZE, SIZE), np.uint16)
    gt[8:20, 8:20] = 1
    pred = np.zeros((SIZE, SIZE), np.uint16)
    pred[40:52, 40:52] = 1  # disjoint from GT

    rec = compute_per_image_metrics(pred, gt, image_id="no_match")
    assert rec.aji_plus == 0.0
    assert rec.pq == 0.0
    # NS follows its own definition here; it is NOT forced to zero.
    assert not math.isnan(rec.nsd)
    assert 0.0 <= rec.nsd <= 1.0
    assert rec.included is True


# ---------------------------------------------------------------------------
# 6. A valid image with one metric equal to zero
# ---------------------------------------------------------------------------

def test_06_single_zero_metric_is_retained_and_collapses_p():
    # Predicting a merged blob over two touching cells: PQ = 0 (no IoU > 0.5
    # pair), while AJI+ and NS stay positive.
    gt = np.zeros((SIZE, SIZE), np.uint16)
    gt[10:30, 8:20] = 1
    gt[10:30, 20:32] = 2
    pred = np.zeros((SIZE, SIZE), np.uint16)
    pred[10:30, 8:32] = 1

    rec = compute_per_image_metrics(pred, gt, image_id="merged")
    assert rec.pq == 0.0
    assert rec.aji_plus > 0.0
    assert rec.nsd > 0.0
    assert rec.included is True  # a zero never removes an image

    p = compute_performance_score({"aji_plus": rec.aji_plus, "pq": rec.pq, "nsd": rec.nsd})
    assert p == 0.0  # min(A, Q, N) = 0 => P = 0, with no epsilon rescue


# ---------------------------------------------------------------------------
# 7. An invalid label map
# ---------------------------------------------------------------------------

def test_07a_invalid_label_map_not_2d():
    gt = np.zeros((8, 8, 3), np.uint16)
    with pytest.raises(MetricError, match="not a 2-D instance-label image"):
        compute_per_image_metrics(np.zeros((8, 8), np.uint16), gt, image_id="3d")


def test_07b_invalid_label_map_negative_labels():
    gt = two_cells().astype(np.int32)
    gt[0, 0] = -5
    with pytest.raises(MetricError, match="negative labels"):
        compute_per_image_metrics(two_cells(), gt, image_id="negative")


def test_07c_invalid_label_map_non_integer_labels():
    gt = two_cells().astype(np.float32)
    gt[0, 0] = 1.5
    with pytest.raises(MetricError, match="non-integer labels"):
        compute_per_image_metrics(two_cells(), gt, image_id="fractional")


def test_07d_invalid_label_map_shape_mismatch():
    with pytest.raises(MetricError, match="shapes differ"):
        compute_per_image_metrics(empty_mask(32), two_cells(64), image_id="mismatch")


def test_07e_integral_float_and_large_labels_are_valid():
    # An integral float map is valid, and labels above the uint16 range must not
    # wrap (a narrow cast would silently corrupt them).
    gt = two_cells().astype(np.float64) * 70000
    rec = compute_per_image_metrics(gt.copy(), gt, image_id="large_labels")
    assert rec.aji_plus == pytest.approx(1.0)
    assert rec.n_gt_instances == 2


# ---------------------------------------------------------------------------
# 8. An unexpected NaN -> explicit error, never coerced to zero
# ---------------------------------------------------------------------------

def test_08_unexpected_nan_raises_with_context(monkeypatch):
    import sapi.metrics.instance_seg as m

    monkeypatch.setattr(m, "_aji_plus_from_overlap", lambda ov: float("nan"))
    with pytest.raises(MetricError) as excinfo:
        m.compute_per_image_metrics(two_cells(), two_cells(), image_id="img_042")

    err = excinfo.value
    assert err.metric == "aji_plus"
    assert err.image_id == "img_042"
    assert math.isnan(err.value)
    # The offending masks travel with the error so it can be diagnosed.
    assert err.gt_mask is not None and err.pred_mask is not None
    assert "implementation or data error" in str(err)


def test_08b_unexpected_nan_aborts_the_whole_run(monkeypatch):
    import sapi.metrics.instance_seg as m

    monkeypatch.setattr(m, "_pq_from_overlap", lambda ov, t=0.5: float("inf"))
    # No dataset-level score may be produced from a broken image set.
    with pytest.raises(MetricError):
        m.compute_all_metrics([two_cells()], [two_cells()], image_ids=["x"])


# ---------------------------------------------------------------------------
# 9. Floating-point drift above one -> clipped
# ---------------------------------------------------------------------------

def test_09_floating_point_drift_is_clipped():
    gt = pred = two_cells()
    assert _check_metric_value(1.0 + 1e-15, "aji_plus", "drift", gt, pred) == 1.0
    assert _check_metric_value(-1e-15, "pq", "drift", gt, pred) == 0.0


# ---------------------------------------------------------------------------
# 10. A materially out-of-range metric -> error, never clipped
# ---------------------------------------------------------------------------

def test_10_materially_out_of_range_raises():
    gt = pred = two_cells()
    for bad in (1.5, -0.2, 2.0):
        with pytest.raises(MetricError, match="outside \\[0, 1\\]"):
            _check_metric_value(bad, "nsd", "bad", gt, pred)


def test_10b_out_of_range_just_beyond_tolerance_raises():
    gt = pred = two_cells()
    with pytest.raises(MetricError):
        _check_metric_value(1.0 + 1e-9, "nsd", "bad", gt, pred)


# ---------------------------------------------------------------------------
# 11. A dataset whose dataset-level mean is zero -> P = 0
# ---------------------------------------------------------------------------

def test_11_zero_dataset_mean_gives_zero_p():
    # Every prediction is empty => all three dataset means are exactly 0.
    gt = [two_cells(), two_cells(), two_cells()]
    pred = [empty_mask(), empty_mask(), empty_mask()]
    seg = compute_all_metrics(pred, gt, image_ids=["a", "b", "c"])
    assert seg.aji_plus == 0.0 and seg.pq == 0.0 and seg.nsd == 0.0
    assert seg.num_included == 3  # zeros are never dropped
    assert compute_performance_score(seg.summary_dict()) == 0.0


def test_11b_one_zero_component_collapses_p():
    assert compute_performance_score({"aji_plus": 0.9, "pq": 0.0, "nsd": 0.8}) == 0.0
    assert compute_performance_score({"aji_plus": 0.0, "pq": 0.7, "nsd": 0.8}) == 0.0


# ---------------------------------------------------------------------------
# 12. HM of means != mean of per-image HMs
# ---------------------------------------------------------------------------

def test_12_hm_of_means_differs_from_mean_of_hms():
    a = [0.9, 0.1, 0.5]
    q = [0.2, 0.8, 0.5]
    n = [0.7, 0.3, 0.5]

    hm_of_means = harmonic_mean([float(np.mean(a)), float(np.mean(q)), float(np.mean(n))])
    mean_of_hms = float(np.mean([harmonic_mean([ai, qi, ni]) for ai, qi, ni in zip(a, q, n)]))

    assert hm_of_means != pytest.approx(mean_of_hms)
    # The implementation must follow the first expression.
    assert compute_performance_score(
        {"aji_plus": float(np.mean(a)), "pq": float(np.mean(q)), "nsd": float(np.mean(n))}
    ) == pytest.approx(hm_of_means, abs=1e-6)


def test_12b_zero_in_one_image_does_not_zero_the_dataset_score():
    # A per-image HM would be 0 for the zero-PQ image and drag the average; the
    # required order (mean first, then HM) does not collapse.
    a, q, n = [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    p = compute_performance_score(
        {"aji_plus": float(np.mean(a)), "pq": float(np.mean(q)), "nsd": float(np.mean(n))}
    )
    assert p > 0.0
    assert p == pytest.approx(harmonic_mean([1.0, 0.5, 1.0]), abs=1e-6)


# ---------------------------------------------------------------------------
# Aggregation-order contract
# ---------------------------------------------------------------------------

def test_p_requires_all_three_components():
    # A two-component P is a different quantity and must never be returned as P.
    assert math.isnan(compute_performance_score({"aji_plus": 0.8, "pq": 0.6}))
    assert math.isnan(compute_performance_score({"aji_plus": 0.8, "pq": 0.6, "nsd": float("nan")}))
    assert math.isnan(compute_performance_score({"aji_plus": 0.8, "pq": 0.6, "nsd": None}))


def test_p_matches_the_closed_form():
    A, Q, N = 0.8, 0.6, 0.9
    assert compute_performance_score({"aji_plus": A, "pq": Q, "nsd": N}) == pytest.approx(
        3 / (1 / A + 1 / Q + 1 / N), abs=1e-6
    )


def test_means_are_macro_averages_with_equal_image_weight():
    # A cell-dense image must not outweigh a sparse one (that would be a
    # micro-average over pooled instances).
    dense = np.zeros((SIZE, SIZE), np.uint16)
    for i in range(1, 17):
        r, c = divmod(i - 1, 4)
        dense[r * 8:r * 8 + 6, c * 8:c * 8 + 6] = i
    sparse = np.zeros((SIZE, SIZE), np.uint16)
    sparse[8:20, 8:20] = 1

    # dense predicted perfectly, sparse missed entirely.
    seg = compute_all_metrics([dense.copy(), empty_mask()], [dense, sparse],
                              image_ids=["dense", "sparse"])
    # Equal weight => (1.0 + 0.0) / 2, not weighted by the 16:1 cell ratio.
    assert seg.aji_plus == pytest.approx(0.5)
    assert seg.num_included == 2


def test_per_image_records_are_complete():
    seg = compute_all_metrics(
        [two_cells(), empty_mask()], [two_cells(), two_cells()],
        image_ids=["img_a", "img_b"],
    )
    assert [r["image_id"] for r in seg.per_image] == ["img_a", "img_b"]
    for rec in seg.per_image:
        for key in ("image_id", "n_gt_instances", "n_pred_instances", "aji_plus",
                    "pq", "nsd", "included", "exclusion_reason", "metric_error"):
            assert key in rec
    assert seg.per_image[1]["n_pred_instances"] == 0
    assert seg.per_image[0]["n_gt_instances"] == 2


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError, match="Mismatched pred/GT count"):
        compute_all_metrics([two_cells()], [two_cells(), two_cells()])
    with pytest.raises(ValueError, match="Mismatched image_ids count"):
        compute_all_metrics([two_cells()], [two_cells()], image_ids=["a", "b"])


# ---------------------------------------------------------------------------
# Value identity: AJI+/PQ against an independent oracle
# ---------------------------------------------------------------------------

def test_iou_matches_brute_force_oracle():
    from sapi.metrics.instance_seg import _pairwise_overlap

    rng = np.random.default_rng(7)
    for _ in range(20):
        def make(n):
            m = np.zeros((48, 48), np.uint16)
            for i in range(1, n + 1):
                y, x = rng.integers(0, 40, 2)
                m[y:y + 8, x:x + 8] = i
            return m

        gt, pred = make(int(rng.integers(1, 8))), make(int(rng.integers(1, 8)))
        ov = _pairwise_overlap(pred, gt)
        ref = brute_force_iou(pred, gt)
        assert ov.iou.shape == ref.shape
        np.testing.assert_allclose(ov.iou, ref, atol=1e-9)


def test_aji_plus_pairs_one_to_one():
    # Two GT instances over one prediction: optimal 1-to-1 pairing claims the
    # prediction once, so the union is not double-counted (plain AJI would
    # over-penalise here).
    gt = np.zeros((10, 10), np.uint16)
    gt[2:4, 2:8] = 1
    gt[4:6, 2:8] = 2
    pred = np.zeros((10, 10), np.uint16)
    pred[2:6, 2:8] = 1
    assert compute_aji_plus(pred, gt) == pytest.approx(1 / 3, abs=1e-4)


def test_pq_uses_strict_iou_threshold():
    # A pair at exactly IoU = 0.5 must NOT match (strict >, not >=).
    gt = np.zeros((20, 20), np.uint16)
    gt[0:4, 0:4] = 1              # area 16
    pred = np.zeros((20, 20), np.uint16)
    pred[0:4, 0:8] = 1            # area 32, intersection 16 -> IoU = 16/32 = 0.5
    assert compute_pq(pred, gt) == 0.0


def test_nsd_is_semantic_not_instance_paired():
    # Merging two touching cells leaves the foreground outline identical, so NS
    # is 1.0 while PQ collapses. NS scores the outline; splits/merges are AJI+/PQ's job.
    gt = np.zeros((40, 40), np.uint16)
    gt[10:30, 8:20] = 1
    gt[10:30, 20:32] = 2
    pred = np.zeros((40, 40), np.uint16)
    pred[10:30, 8:32] = 1
    assert compute_nsd(pred, gt) == pytest.approx(1.0)
    assert compute_pq(pred, gt) == 0.0

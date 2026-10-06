"""
SAPI (Sustainability-Aware Performance Index).

FUNDAMENTAL ENERGY EQUATIONS
=============================

The profiler measures total IT equipment energy as:
    E_IT = E_GPU + E_CPU + E_RAM   (in millijoules, mJ)

This total is divided by the number of images to get per-image energy:
    Ei  = E_IT / N_test             (inference energy per test image)
    Et  = E_IT / N_support          (fine-tuning energy per training image)

All energy values in SAPI components are in millijoules (mJ/image).

Raw component computation for one experiment result.
The final SAPI score is computed per-value using log-transformed components —
no cross-model normalisation is required, so SAPI is absolute and stable:
adding or removing models from the comparison field does not change any
individual model's score.

Components stored in result JSONs
----------------------------------
P      Performance          harmonic_mean(AJI+, PQ, NSD)                 in [0,1]
Ei     Inference energy     E_IT / N_test (mJ/image)                     lower=better, CI-independent
Et     Training energy      E_IT / N_support (mJ/image)                  lower=better, CI-independent, supplementary only
S      Sustainability       Actual CO₂ per image (gCO₂eq)               lower=better, CI-dependent
C      Complexity           log10(n_params)                              lower=better

Note on time: inference and training time are implicitly captured in Ei and Et
respectively.  Energy = Power × Time, so a slower model uses more energy under
similar power draw, and is penalised through the energy terms.  Time is not an
independent formula component.

SAPI formula
------------
Two sub-scores share identical outer weights (P=1/2, E=1/3, C=1/6); the only
difference is how E is defined.

SAPI_ZS  (zero-shot — pretrained inference only, no training cost)

    SAPI_ZS = P^(1/2) × (1/log10(Ei_mJ))^(1/3) × (1/m_raw)^(1/6)

    E = Ei (IT inference energy per image, mJ).
    Applies to: the `inference` (zero-shot) experiment type — ALL models,
    including inference-only ones (DeepCell, Mesmer, CellSAM).

SAPI_LC  (lifecycle — fine-tuning + inference, fewshot_finetuning only)

    E_LC_mJ = 0.8 × Ei_mJ + 0.2 × Et_mJ   (lifecycle composite energy)
    SAPI_LC  = P^(1/2) × (1/log10(E_LC_mJ))^(1/3) × (1/m_raw)^(1/6)

    E = E_LC (lifecycle composite of inference + training energy).
    Applies to: the `fewshot_finetuning` experiment type only. fewshot_finetuning is
    a single fixed-epoch fine-tuning run with no early stopping (patience=0),
    so both Ei and Et are clean, unambiguous per-run measurements suitable for
    a lifecycle score.

    The 0.8/0.2 split reflects the empirical consensus from the AI energy
    literature that inference dominates a model's total deployment energy
    (Schneider Electric 2023: ~80% inference; Nvidia/AWS 2019: 80-90%;
    Patterson et al. 2021: ~60%; de Vries/Meta 2023: ~65%; Kaack et al. 2022:
    inference accumulates more energy over the deployment lifetime). Central
    estimate across sources: 80% inference / 20% training.

Weights: P = 1/2,  E = 1/3,  C = 1/6    (sum = 1.0) — identical for both
sub-scores; only the E term's definition differs. There is no combined SAPI
score — SAPI_ZS and SAPI_LC are reported separately, for different
deployment questions (zero-annotation use vs. tissue-adapted use).

Outer weight rationale  (P = 1/2,  E = 1/3,  C = 1/6)
-------------------------------------------------------
P = 1/2   Performance is the primary objective.  A computationally efficient
          model that fails at segmentation provides no practical value.  The
          harmonic mean structure of P already collapses to near-zero when any
          single constituent metric fails, so wP = 1/2 ensures that performance
          is the necessary condition for a model to rank well.

E = 1/3   Energy is the dominant sustainability dimension of the index.  Direct
          energy measurement subsumes inference and training time (Energy =
          Power × Time), so runtime efficiency is fully captured within this
          term.  The 2:1 ratio wE:wC reflects that measured energy is a more
          direct and complete sustainability signal than the proxy provided by
          parameter count.

C = 1/6   Model complexity (log10 of parameter count) captures deployment
          feasibility — peak VRAM, storage footprint, portability to edge
          hardware — that is not fully captured by energy on a specific server
          GPU.  It receives the smallest weight because energy already partially
          penalises large models, and parameter count is a secondary
          sustainability consideration relative to direct energy measurement.

Weight ratio P:E:C = 3:2:1  reflects the hierarchy
performance > energy > complexity.

where:
    P        = harmonic_mean(AJI+, PQ, NSD)                   already in [0,1]
    Ei_mJ    = ei_score_mj_per_image                         IT inference energy in mJ/image
    m_raw    = log10(n_params)                                model complexity

    Note: energy is expressed in millijoules (mJ), the SI-derived unit, for the
    log-transform. The unit inside log10 is a genuine design parameter (log is not
    scale-invariant), and mJ keeps real per-image energies (~360–36000 mJ) safely
    in the log10 ≈ 2.6–4.6 range, well clear of the log10(E)=0 singularity at
    E=1 mJ. Plain joules would push the most efficient models below 1 J (log10 < 0),
    reintroducing that singularity, so mJ (not J) is used.

Interpretation:
    lower E   → smaller log10(E_mJ) → larger 1/log10(E_mJ) → higher SAPI
    fewer params → smaller m_raw     → larger 1/m_raw          → higher SAPI
    log10(E_mJ) must be > 0 (i.e. E > 1 mJ) — holds for all parametric models.

Applicability:
    SAPI_ZS  — all models (including inference-only: DeepCell, Mesmer, CellSAM)
    SAPI_LC  — fewshot_finetuning results only, fine-tunable models only
    Any model with an undefined complexity term (m_raw = 0 → log10(0) undefined)
    is excluded from both.

Note on AJI+: P uses the AJI+ variant (Graham et al., 2019) — AJI with optimal
one-to-one (Hungarian) matching — rather than the original greedy AJI (Kumar
et al., 2017), because greedy AJI over-penalises by allowing one prediction to
be claimed by several GT instances (inflating the union). AJI+ occupies the same
[0, 1] range. Only AJI+, PQ, and NSD are computed and stored; no other
segmentation metric is produced by the benchmark.

Carbon reference (fallback for CI measurements)
-----------------------------------------------
CI_REFERENCE_G_PER_KWH = 309.0  (Germany 2025, Umweltbundesamt — static fallback)
"""

from __future__ import annotations

import math

CI_REFERENCE_G_PER_KWH: float = 309.0

# ── Outer SAPI weights (shared by SAPI_ZS and SAPI_LC) ─────────────────────────
_W_P = 1 / 2   # performance
_W_E = 1 / 3   # energy (Ei for SAPI_ZS; E_LC for SAPI_LC)
_W_C = 1 / 6   # model complexity
assert abs(_W_P + _W_E + _W_C - 1.0) < 1e-9, (
    f"SAPI outer weights must sum to 1.0, got {_W_P + _W_E + _W_C:.6f}"
)

# ── SAPI_LC lifecycle energy composite (fewshot_finetuning only) ───────────────────
_W_Ei_LC = 0.8   # inference energy share of the lifecycle composite
_W_Et_LC = 0.2   # training energy share of the lifecycle composite
assert abs(_W_Ei_LC + _W_Et_LC - 1.0) < 1e-9, (
    f"SAPI_LC lifecycle split must sum to 1.0, got {_W_Ei_LC + _W_Et_LC:.6f}"
)


def harmonic_mean(values: list) -> float:
    """Harmonic mean of non-negative floats.

    Returns 0.0 if any value is exactly 0 — a model that completely fails on
    one metric (e.g. AJI=0) scores P=0.
    Returns nan if any value is nan: NaN in any required component means the
    overall score is NaN too, rather than silently dropping that component
    and computing a harmonic mean over fewer (and therefore not comparable)
    values.
    """
    if any(math.isnan(v) for v in values):
        return float("nan")
    vals = [v for v in values if v >= 0.0]
    if not vals:
        return float("nan")
    if any(v == 0.0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def compute_performance_score(perf_metrics: dict) -> float:
    """P = HM(A, Q, N) over the three DATASET-level metric means.

        P = 0                        if min(A, Q, N) = 0
        P = 3 / (1/A + 1/Q + 1/N)    otherwise

    AJI+ — aggregated Jaccard index with optimal 1-to-1 matching (no IoU threshold)
    PQ   — panoptic quality = SQ × RQ (shape × detection at IoU > 0.5)
    NS   — normalised surface dice at tau=2px (stored under the key "nsd")

    Only these three segmentation metrics are computed and stored; they are the
    sole constituents of the SAPI performance term.

    Input is the dataset-level means produced by compute_all_metrics(), NOT
    per-image values. P is deliberately not computed per image and averaged:
    HM(mean a, mean q, mean n) != mean_i HM(a_i, q_i, n_i) in general, and only
    the former is the benchmark's definition of P.

    All three components are REQUIRED: a missing, non-numeric or NaN component
    yields NaN rather than a harmonic mean over the remaining two, because a
    two-component P is a different quantity and must never be reported as P.

    A zero component is a valid result: it is not dropped and no epsilon is
    added — it collapses P to 0, which is the intended "no compensation across
    dimensions" behaviour.
    """
    vals = []
    for key in ("aji_plus", "pq", "nsd"):
        v = perf_metrics.get(key)
        if v is None:
            return float("nan")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(fv):
            return float("nan")
        vals.append(max(0.0, min(1.0, fv)))
    return round(harmonic_mean(vals), 6)


def compute_sapi_zs(p: float, ei_mj: float, m_raw: float) -> float:
    """Compute SAPI_ZS for one (model, dataset) observation.

    SAPI_ZS = P^(1/2) × (1/log10(Ei_mJ))^(1/3) × (1/m_raw)^(1/6)

    ``ei_mj`` is the inference energy per image in millijoules (mJ).

    Returns NaN if any guard condition is violated:
        log10(Ei_mJ) must be > 0  (Ei_mJ > 1)
        m_raw must be > 0           (n_params > 1)
        p must be >= 0
        (all three also fail closed if NaN — see note below)

    NaN inputs are checked explicitly with math.isnan() rather than relied on
    to propagate through `<= 0` comparisons: NaN is never `<= 0` in Python/
    IEEE-754 (all ordered comparisons involving NaN are False), so a NaN
    ei_mj/m_raw would silently skip the intended guard and only happen to
    still produce NaN because log10/division/exponentiation of NaN also
    yields NaN. That accidental propagation is fragile — any future change to
    this arithmetic could break it silently — so the NaN case is guarded for
    directly instead.
    """
    if p < 0 or math.isnan(p) or math.isnan(ei_mj) or math.isnan(m_raw):
        return float("nan")
    log_ei = math.log10(max(ei_mj, 1e-20))
    if log_ei <= 0 or m_raw <= 0:
        return float("nan")
    return (p ** _W_P) * ((1.0 / log_ei) ** _W_E) * ((1.0 / m_raw) ** _W_C)


def compute_sapi_lc(p: float, ei_mj: float, et_mj: float, m_raw: float) -> float:
    """Compute SAPI_LC for one fewshot_finetuning (model, dataset) observation.

    E_LC_mJ = 0.8 × Ei_mJ + 0.2 × Et_mJ   (lifecycle composite energy)
    SAPI_LC  = P^(1/2) × (1/log10(E_LC_mJ))^(1/3) × (1/m_raw)^(1/6)

    ``ei_mj``/``et_mj`` are the inference and training energies per image in
    millijoules (mJ).

    Only meaningful for fewshot_finetuning results (fixed epoch budget, no early
    stopping — both Ei and Et are clean per-run measurements).

    Returns NaN if any guard condition is violated:
        E_LC_mJ must be > 1 mJ (log10(E_LC) must be > 0 for real-valued exponentiation)
        m_raw must be > 0      (n_params > 1)
        p must be >= 0
        (all also fail closed if NaN — see compute_sapi_zs() for why NaN is
        checked explicitly rather than relied on to propagate through `<= 0`)
    """
    if (
        p < 0 or math.isnan(p)
        or math.isnan(ei_mj) or math.isnan(et_mj) or math.isnan(m_raw)
    ):
        return float("nan")
    e_lc_mj = _W_Ei_LC * ei_mj + _W_Et_LC * et_mj
    log_e_lc = math.log10(max(e_lc_mj, 1e-20))
    if log_e_lc <= 0 or m_raw <= 0:
        return float("nan")
    return (p ** _W_P) * ((1.0 / log_e_lc) ** _W_E) * ((1.0 / m_raw) ** _W_C)


def compute_raw_components(
    perf_metrics: dict,
    inf_profiler: dict,
    n_test_images: int,
    train_profiler: dict | None = None,
    n_train_images: int = 0,
    n_params: int = -1,
) -> dict:
    """Compute all raw SAPI components for one experiment result.

    Args:
        perf_metrics:      performance_metrics dict from the experiment result
        inf_profiler:      ProfilerResult.to_dict() for the inference phase
        n_test_images:     number of images in the test/eval set
        train_profiler:    ProfilerResult.to_dict() for training (None for
                           inference-only experiments; sets et_score = 0.0)
        n_train_images:    number of images actually used for training
        n_params:          model parameter count (-1 if unavailable)

    Returns dict with:
        p_score                  harmonic_mean(AJI+, PQ, NSD)
        ei_score_mj_per_image    IT inference energy per image (mJ)      [Ei in SAPI_ZS]
        et_score_mj_per_image    IT training energy per training image (mJ)  [Et in SAPI_LC]
                                   = total_training_it_energy × 3.6e9 / n_train_images
                                   (normalised by support set size only, NOT by n_epochs, so that
                                    training energy per image is comparable across different epoch budgets)
                                   (supplementary only; 0.0 for inference-only)
        s_score_co2_per_image    actual CO₂ per image (gCO₂eq) — CI-dependent
        m_raw                    log10(n_params), 0.0 for non-parametric models
    """
    # ── P: segmentation performance ──────────────────────────────────────────
    p_score = compute_performance_score(perf_metrics)

    # ── Ei: IT inference energy per image in mJ (CI-independent) ─────────────
    # ENERGY EQUATION: Ei = E_IT / N_test
    # where E_IT = E_GPU + E_CPU + E_RAM (total IT equipment energy in mJ)
    # and N_test = number of test/inference images
    # Profiler returns mJ; fall back to kWh for older result files
    if "it_energy_mj" in inf_profiler:
        e_it_total_mj = float(inf_profiler.get("it_energy_mj", 0.0) or 0.0)
    else:
        e_it_total_kwh = float(inf_profiler.get("it_energy_kwh", 0.0) or 0.0)
        e_it_total_mj = e_it_total_kwh * 3.6e9  # kWh → mJ: ×3.6e9
    if n_test_images <= 0:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"n_test_images <= 0 (got {n_test_images}); ei_score set to nan."
        )
        ei_score = float("nan")
    else:
        ei_score = e_it_total_mj / n_test_images

    # ── Et: IT training energy per training image in mJ (supplementary) ──────
    # ENERGY EQUATION: Et = E_IT / N_support
    # where E_IT = E_GPU + E_CPU + E_RAM (total IT equipment energy in mJ)
    # and N_support = number of training/support images used
    et_score = 0.0
    if train_profiler:
        if n_train_images <= 0:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "train_profiler was provided but n_train_images <= 0 "
                f"(got {n_train_images}); et_score set to nan."
            )
            et_score = float("nan")
        else:
            # Profiler returns mJ; fall back to kWh for older result files
            if "it_energy_mj" in train_profiler:
                e_train_it_mj = float(train_profiler.get("it_energy_mj", 0.0) or 0.0)
            else:
                e_train_it_kwh = float(train_profiler.get("it_energy_kwh", 0.0) or 0.0)
                e_train_it_mj = e_train_it_kwh * 3.6e9  # kWh → mJ: ×3.6e9
            n_tr = n_train_images
            et_score = e_train_it_mj / n_tr

    # ── S: actual CO₂ per image at live carbon intensity ─────────────────────
    s_score = float(inf_profiler.get("co2_eq_g_per_image", 0.0) or 0.0)

    # ── M: model complexity (log10 parameter count) ───────────────────────────
    m_raw = math.log10(n_params) if n_params > 0 else 0.0

    return {
        "p_score":                 round(p_score,      6),
        "ei_score_mj_per_image":   round(ei_score,     12),
        "et_score_mj_per_image":   round(et_score,     12),
        "s_score_co2_per_image":   round(s_score,      10),
        "m_raw":                   round(m_raw,        4),
    }

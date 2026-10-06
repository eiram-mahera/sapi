"""
Base classes for all experiment types.

Adding a new experiment type
----------------------------
Create one module under sapi/experiments/ with a BaseExperiment subclass, then
add a single line to EXPERIMENT_REGISTRY in __init__.py.  The subclass is
self-describing: it declares its identity and behaviour via class attributes and
two small methods, so neither the CLI (run_experiment.py) nor the result logger
(ResultLogger) needs to hardcode anything about it.

Declare on the subclass:
    experiment_type          str   — registry key / directory segment.
    measures_energy          bool  — True if the run profiles energy/CO₂ so the
                                       CLI prints efficiency and downstream SAPI
                                       tooling knows a profiler ran.
    yaml_section_chain       tuple — model-YAML training sections to read, in
                                       priority order (first present wins).

Override:
    experiment_subtype(config)     — the <experiment_subtype> path segment for the
                                       run (full_model / frozen_encoder / default).
    summarize(result) -> str       — human-readable one-block summary for the CLI.

The two experiment types
-------------------------
inference             Run the pretrained model on ALL samples. No training, no
                       split. Measures energy. Source of SAPI_ZS.

fewshot_finetuning    Fine-tune once on a tissue-stratified support set (one
                       image per tissue group); fixed epoch budget, no early
                       stopping. Both full_model and frozen_encoder modes in one
                       execution. Measures energy. Source of SAPI_LC.

Each experiment runs exactly once. The date and time are recorded in the
result so experiments run at different times of day can be compared.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redaction — keep secrets and identifying info out of stored results.
# Results/logs are intended to be shareable (e.g. with a journal), so the
# stored config/hardware snapshots are scrubbed of API credentials, the
# machine hostname, and the running user's home-directory prefix in paths.
# This affects ONLY the stored snapshot; the live ExperimentConfig used at
# runtime is untouched.
# ---------------------------------------------------------------------------

# Config fields that hold secrets — blanked in the stored snapshot.
_SECRET_CONFIG_KEYS = ("electricitymaps_api_token", "electricitymaps_proxy")
# Config fields that hold filesystem paths — home dir anonymised to "~".
_PATH_CONFIG_KEYS = ("data_root", "output_dir", "ci_timeline_path")
# hardware_info fields that identify the machine — blanked.
_IDENTIFYING_HW_KEYS = ("hostname",)


def _anonymise_path(value: str) -> str:
    """Replace the running user's home-directory prefix with '~' in *value*."""
    if not isinstance(value, str) or not value:
        return value
    home = str(Path.home())
    if home and value.startswith(home):
        return "~" + value[len(home):]
    return value


def _deep_anonymise(value):
    """Recursively anonymise home-dir prefixes in any nested str/dict/list value."""
    if isinstance(value, str):
        return _anonymise_path(value)
    if isinstance(value, dict):
        return {k: _deep_anonymise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_deep_anonymise(v) for v in value)
    return value


def redact_config(cfg: dict) -> dict:
    """Return a copy of a config dict with secrets blanked and paths anonymised.

    Path anonymisation is recursive so that filesystem paths nested inside
    ``model_config`` (e.g. ``weights_path``, ``mediar_root``, ``checkpoint_path``)
    are also stripped of the running user's home-directory prefix — not just the
    top-level ``_PATH_CONFIG_KEYS``.
    """
    out = {k: _deep_anonymise(v) for k, v in cfg.items()}
    for k in _SECRET_CONFIG_KEYS:
        if out.get(k):
            out[k] = ""
    return out


def redact_hardware_info(hw: dict) -> dict:
    """Return a copy of hardware_info with identifying fields blanked."""
    out = dict(hw)
    for k in _IDENTIFYING_HW_KEYS:
        if k in out:
            out[k] = ""
    return out


# ---------------------------------------------------------------------------
# Failure classification — coarse, queryable category for a raised exception.
# Lets analysis distinguish infrastructure failures (CUDA OOM) and genuinely
# unsupported configurations from real model/code errors without string-matching
# the free-text message at analysis time.
# ---------------------------------------------------------------------------

def classify_failure(exc: BaseException) -> str:
    """Return a coarse failure category: 'oom' | 'unsupported' | 'error'."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "out of memory" in msg or "outofmemory" in name.lower():
        return "oom"
    if isinstance(exc, NotImplementedError):
        return "unsupported"
    return "error"


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All parameters that fully define a benchmark execution."""

    # ── Identity ────────────────────────────────────────────────────────────
    # inference | fewshot_finetuning
    experiment_type: str = "inference"
    model_name: str = "cellpose_cyto3"
    dataset_name: str = "DAPI"

    # ── Data ────────────────────────────────────────────────────────────────
    data_root: str = "data/CellBinDB"
    seed: int = 42

    # ── Training hyperparameters (few-shot fine-tuning) ─────────────────────
    n_epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 8
    weight_decay: float = 1e-5
    freeze_encoder: bool = True
    patience: int = 20
    """Early-stopping patience (epochs without val-loss improvement before stopping).
    0 disables early stopping (run all n_epochs)."""

    # ── Hardware ────────────────────────────────────────────────────────────
    device: str = "cuda"
    gpu_index: int = 0

    # ── Energy & sustainability ──────────────────────────────────────────────
    pue: float = 1.3  # Power Usage Effectiveness: 1.0 = laptop/workstation, 1.3 = data centre.
    baseline_duration: float = 2.0  # Seconds to sample idle GPU power before each timed block (for process isolation).
    electricitymaps_api_token: str = ""  # Optional Electricity Maps API token for real-time carbon intensity.
    electricitymaps_zone: str = "DE"  # Electricity Maps zone code (default 'DE' = Germany whole-country average).
    electricitymaps_proxy: str = ""  # HTTP/HTTPS proxy URL for HPC nodes with no direct internet.
    carbon_intensity_override: float = 0.0  # If > 0, use this gCO₂eq/kWh value directly and skip the API call (pre-fetch on login node).

    # Path to configs/ci_timeline.csv produced by collect_carbon_intensity.py.
    # When set, the nearest-timestamp entry is used instead of an API call.
    # Recommended for HPC compute nodes with no internet access.
    ci_timeline_path: str = ""

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: str = "results"
    # Write predicted instance masks to <run_dir>/predictions/<sample>.npz.
    # Off by default: the masks are large and are not needed to compute any metric
    # or SAPI score (everything downstream reads result.json / master_log.csv).
    # Turn on per run with `--save_predictions`; the experiments still attach the
    # masks to the result either way, so nothing else changes.
    save_predictions: bool = False

    # GPU display name, auto-detected (torch.cuda.get_device_name, with a
    # pynvml fallback) when not set explicitly. Stored in the config snapshot
    # for reference; NOT part of the result/log directory path (see ResultLogger).
    gpu_type: str = ""

    # ── Model-specific overrides ─────────────────────────────────────────────
    model_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Complete record of a single experiment execution.

    Every experiment type produces exactly one ExperimentResult per execution.

    Note: performance_metrics_per_image is written to per_image_metrics.csv and
    excluded from result.json to keep it compact.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    experiment_id: str = ""          # shared across all results in one execution
    # inference | fewshot_finetuning
    experiment_type: str = ""
    model_name: str = ""
    dataset_name: str = ""
    experiment_datetime: str = ""    # ISO 8601 with UTC offset, e.g. 2024-05-01T14:32:00+02:00

    # ── Config snapshot (for exact reproducibility) ─────────────────────────
    config: dict = field(default_factory=dict)

    # ── Hardware & software environment ────────────────────────────────────
    hardware_info: dict = field(default_factory=dict)

    # ── Segmentation performance ─────────────────────────────────────────
    performance_metrics: dict = field(default_factory=dict)

    # One image-level record per evaluated image: image_id, tissue_group,
    # instance counts, AJI+/PQ/NS, inclusion status, exclusion reason (see
    # sapi/metrics/instance_seg.py::PerImageMetrics, enriched by
    # BaseExperiment.build_per_image_records). Written to result.json and to
    # per_image_metrics.csv, excluded from master_log.csv (only
    # performance_metrics is flattened) and stripped from master_log.jsonl
    # (bulky) — see ResultLogger.
    performance_metrics_per_image: list = field(default_factory=list)

    # ── Resource usage during inference ────────────────────────────────────
    inference_efficiency: dict = field(default_factory=dict)

    # ── Resource usage during training/fine-tuning ─────────────────────────
    # Empty dict for inference experiments.
    training_efficiency: dict = field(default_factory=dict)

    # ── Training details (epochs, loss curve, etc.) ─────────────────────────
    training_metadata: dict = field(default_factory=dict)

    # ── Dataset sizes ───────────────────────────────────────────────────────
    num_test_images: int = 0
    num_train_images: int = 0
    n_dataset_images: int = 0   # total images in the dataset

    # ── Training epochs ─────────────────────────────────────────────────────
    # 0 for inference experiments (no training).
    num_epochs: int = 0       # actual epochs trained (with early stopping)
    num_epochs_max: int = 0   # configured maximum from YAML / CLI

    # ── Few-shot specific ───────────────────────────────────────────────────
    # 0 for inference experiments.
    num_shots: int = 0                        # actual number of support samples

    # ── Error / status ────────────────────────────────────────────────────────
    # status is the structured, queryable outcome: "success" on a clean run,
    # "failed" when execute() caught an exception. failure_stage records which
    # phase raised (build_model | load_data | finetuning | inference | metrics |
    # zero_shot); failure_type is the coarse category from classify_failure().
    # On a successful run: status="success", the other three are None/unset.
    status: str = "success"
    failure_stage: Optional[str] = None
    failure_type: Optional[str] = None
    error: Optional[str] = None

    def mark_failed(self, exc: BaseException, stage: str = "") -> None:
        """Record a caught exception as a structured failure on this result."""
        self.status = "failed"
        self.failure_stage = stage or None
        self.failure_type = classify_failure(exc)
        self.error = str(exc)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Exclude performance_metrics_per_image from result.json (written to CSV instead)
        d.pop("performance_metrics_per_image", None)
        return d


# ---------------------------------------------------------------------------
# BaseExperiment
# ---------------------------------------------------------------------------

class BaseExperiment:
    """
    Abstract base class for all experiment types.

    Subclasses implement execute() which returns a list with exactly one
    ExperimentResult, and declare the self-describing attributes/methods below
    so the CLI and result logger never need to special-case them.
    """

    # ── Self-describing metadata (override in subclasses) ────────────────────
    experiment_type: str = "base"

    #: True if execute() profiles energy/CO₂ (so the CLI prints efficiency and
    #: SAPI tooling knows a profiler ran).  False for pure performance runs.
    measures_energy: bool = True

    #: Model-YAML training sections to consult, highest priority first.  Empty
    #: means the experiment reads no training hyperparameters (e.g. inference).
    yaml_section_chain: tuple[str, ...] = ()

    #: True if the output directory is further split into full_model /
    #: frozen_encoder subdirectories (set by ExperimentConfig.freeze_encoder).
    splits_by_freeze_encoder: bool = False

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = str(uuid.uuid4())[:8]

    def execute(self) -> List[ExperimentResult]:
        """Run the experiment and return [result]. Override in subclasses."""
        raise NotImplementedError

    # ── Output layout (override only if the default is not wanted) ───────────

    @classmethod
    def experiment_subtype(cls, config: ExperimentConfig) -> str:
        """Return the <experiment_subtype> path segment for this run.

        Used in the per-image CSV's experiment_subtype identity column and, for
        experiment types ResultLogger doesn't special-case, as the last segment
        of its fallback directory layout
        (results/<family>/<variant>/<dataset>/<experiment_type>/<experiment_subtype>/<runid>/):
        'full_model' / 'frozen_encoder' for an experiment class that overrides
        ``splits_by_freeze_encoder = True`` and runs each freeze_encoder mode as
        its own separate execution/result; 'default' otherwise. Neither current
        experiment type sets that flag, so both return 'default' — including
        fewshot_finetuning, which runs both freeze_encoder modes together in one
        unified execution and stores them side by side in a single result.json
        rather than in separate subtype directories. Note that ResultLogger
        does not use this segment at all for inference or fewshot_finetuning —
        see ResultLogger.__init__.
        """
        if cls.splits_by_freeze_encoder:
            return "frozen_encoder" if config.freeze_encoder else "full_model"
        return "default"

    # ── CLI summary (override in subclasses) ─────────────────────────────────

    def summarize(self, result: "ExperimentResult") -> str:
        """Return a human-readable summary block for the CLI. Override."""
        if result.error:
            return f"{self.experiment_type} failed: {result.error}"
        return f"{self.experiment_type} completed."

    # ── Dataset helpers ───────────────────────────────────────────────────

    def _load_all_data(self):
        """Load ALL samples in the dataset.

        Used directly by inference; fewshot_finetuning also calls this and
        then splits the result into support/test via tissue_stratified_split.
        """
        from sapi.datasets import load_dataset
        return load_dataset(
            dataset_name=self.config.dataset_name,
            data_root=self.config.data_root,
        )

    # ── Per-image records ─────────────────────────────────────────────────

    def build_per_image_records(self, seg, stage: Optional[str] = None) -> List[dict]:
        """Enrich the metrics module's per-image records with the tissue group.

        The metrics module knows the image id, instance counts, metric values
        and inclusion status; the tissue group comes from the dataset. The
        remaining identity fields (model, variant, dataset, condition, mode,
        seed) are added by ResultLogger when it writes per_image_metrics.csv —
        they are constant for a run and already stored in result.json.

        Args:
            seg: an InstanceSegMetrics from compute_all_metrics().
            stage: evaluation stage, for experiments that evaluate the test set
                more than once (e.g. "zero_shot" / "ep100"); omitted otherwise.
        """
        from .tissue_split import load_tissue_labels
        tissue_map = load_tissue_labels(self.config.dataset_name)
        records = []
        for rec in seg.per_image:
            row = dict(rec)
            row["tissue_group"] = tissue_map.get(row.get("image_id"))
            if stage is not None:
                row["stage"] = stage
            records.append(row)
        return records

    # ── Model / profiler builders ──────────────────────────────────────────

    def _build_model(self):
        from sapi.models import get_model
        return get_model(
            self.config.model_name,
            device=self.config.device,
            extra_config=self.config.model_config or {},
        )

    def _build_profiler(self):
        from sapi.efficiency import EfficiencyProfiler
        return EfficiencyProfiler(
            device=self.config.device,
            gpu_index=self.config.gpu_index,
            pue=self.config.pue,
            baseline_duration_s=self.config.baseline_duration,
            electricitymaps_api_token=self.config.electricitymaps_api_token,
            electricitymaps_zone=self.config.electricitymaps_zone,
            electricitymaps_proxy=self.config.electricitymaps_proxy,
            carbon_intensity_override=self.config.carbon_intensity_override,
            ci_timeline_path=self.config.ci_timeline_path,
        )

    # ── Reproducibility ───────────────────────────────────────────────────

    def _seed_everything(self, seed: int):
        """Set random seeds for all frameworks present in the current environment."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except (ImportError, AttributeError):
            pass

    # ── Result skeleton ───────────────────────────────────────────────────

    def _collect_hardware_info(self) -> dict:
        from sapi.utils.hardware_info import collect_hardware_info
        return collect_hardware_info()

    def _make_result_base(self, hardware_info: dict) -> ExperimentResult:
        """Return an ExperimentResult pre-filled with identity and environment fields.

        The stored config and hardware snapshots are redacted (API credentials
        blanked, hostname blanked, home-dir paths anonymised) so that result
        files are safe to share — see redact_config()/redact_hardware_info().
        """
        return ExperimentResult(
            experiment_id=self.experiment_id,
            experiment_type=self.experiment_type,
            model_name=self.config.model_name,
            dataset_name=self.config.dataset_name,
            # ISO 8601 with local timezone offset so time-of-day is preserved.
            experiment_datetime=datetime.now().astimezone().isoformat(),
            config=redact_config(self.config.to_dict()),
            hardware_info=redact_hardware_info(hardware_info),
        )

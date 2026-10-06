"""
Result logger for the benchmarking framework.

Output structure
----------------
Each run gets one self-contained, timestamped directory:

    results/                       (or test_results/ for smoke-tests)
      <model_family>/              e.g. cellpose  stardist  mediar
        <model_variant>/           e.g. cyto3  fluo  mediar
          <dataset>/                e.g. DAPI  HE  mIF
            <experiment_dir>/       zero_shot_inference | fewshot_finetuning
              <runid>/               timestamp, e.g. 20260709T142301_837421
                result.json      ← full machine-readable record (analysis scripts read this)
                config.json      ← ExperimentConfig snapshot
                metrics.json     ← performance metrics + SAPI components + efficiency
                per_image_metrics.csv ← one self-identifying row per evaluated image
                experiment.log   ← this run's log (no separate logs/ tree)
                predictions/      ← predicted instance masks (<sample>.npz), when available

<experiment_dir> is a literal directory name chosen per experiment_type
("inference" → "zero_shot_inference", "fewshot_finetuning" → "fewshot_finetuning");
for any other experiment_type it falls back to
<experiment_type>/<experiment_subtype> (BaseExperiment.experiment_subtype) —
see ResultLogger.__init__.  The <runid> is a microsecond-precision local
timestamp — every run is preserved (re-running the same model/dataset/type/seed
never overwrites a prior run), which supports re-running at different times of
day to sample carbon intensity.  GPU type is NOT part of the path; the full
hardware environment is stored inside result.json / metrics.json.

For test scripts the same structure applies but under test_results/.

Note: cross-run master_metrics.csv aggregation tooling (scripts/vizualizations/,
scripts/analysis/ in the source project) is out of scope for this minimal
fork; use the per-result result.json/metrics.json files directly.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from sapi.experiments.base import ExperimentResult, ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path-building helpers
# ---------------------------------------------------------------------------

# Model families: mapping prefix → family name.
# The variant is whatever follows the prefix + underscore.
_MODEL_FAMILIES = frozenset({
    "cellpose",
    "stardist",
    "cellvit",
    "hovernet",
    "microsam",    # microsam_vit_b_lm → (microsam, vit_b_lm)
    "pathosamvit", # pathosamvit_b     → (pathosamvit, b)
    "cellsam",     # cellsam_general   → (cellsam, general)  [if variants added]
})


def split_model_name(model_name: str) -> tuple[str, str]:
    """Return (family, variant) from a model registry name.

    Examples
    --------
    >>> split_model_name("cellpose_cyto2")  → ("cellpose", "cyto2")
    >>> split_model_name("cellvit_sam_h")  → ("cellvit",  "sam_h")
    >>> split_model_name("mediar")         → ("mediar",   "mediar")
    """
    for family in _MODEL_FAMILIES:
        prefix = family + "_"
        if model_name == family:
            return family, family
        if model_name.startswith(prefix):
            return family, model_name[len(prefix):]
    # Single-word models (mediar, instanseg, deepcell, mesmer)
    return model_name, model_name


def _experiment_subtype(config: ExperimentConfig) -> str:
    """Return the <experiment_subtype> segment for *config* from its experiment class."""
    from sapi.experiments import EXPERIMENT_REGISTRY
    exp_cls = EXPERIMENT_REGISTRY.get(config.experiment_type)
    if exp_cls is None:
        return "default"
    return exp_cls.experiment_subtype(config)


def _nan_to_none(row: dict) -> dict:
    """Replace NaN float values with None so csv.DictWriter leaves the cell
    blank (its behavior for None) instead of writing the literal string "nan"
    (its behavior for float('nan')), which is indistinguishable from real data."""
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.items()
    }


def _create_run_dir(parent_dir: Path) -> Path:
    """Atomically create and return a unique, timestamp-named run directory.

    The directory name is the microsecond timestamp (YYYYMMDDTHHMMSS_ffffff).
    Creation relies on ``mkdir(exist_ok=False)`` as the uniqueness guarantee —
    there is NO separate exists() check, so two parallel runs writing into the
    same parent (e.g. different seeds of the same model/dataset/mode, which
    share a parent because the seed is not part of the path) can never both
    claim the same directory.  On the astronomically rare same-microsecond
    clash the first mkdir raises FileExistsError; we then append a short random
    suffix and retry, so a run is never lost and never overwrites another.
    """
    parent_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    candidate = base
    for _ in range(10000):
        run_dir = parent_dir / candidate
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            # Fresh random suffix each retry so two colliding processes don't
            # both deterministically retry to the same name.
            candidate = f"{base}_{secrets.token_hex(3)}"
    raise RuntimeError(f"Could not create a unique run directory under {parent_dir}")


def _safe_name(name: str) -> str:
    """Sanitise a sample name for use as a filename component."""
    return re.sub(r"[^\w\-.]", "_", str(name)) or "sample"


# ---------------------------------------------------------------------------
# ResultLogger
# ---------------------------------------------------------------------------

class ResultLogger:
    """
    Writes ExperimentResult objects to disk as self-contained run directories.

    Usage
    -----
        result_logger = ResultLogger(config, experiment_id)
        log_path = result_logger.log_file   # attach FileHandler before experiment
        result_logger.log(results)           # after experiment completes

    Predictions
    -----------
    If an experiment attaches a ``predictions`` attribute to its result — a dict
    mapping a relative path (e.g. ``"cell_001"`` or ``"full_model/cell_001"``)
    to a NumPy mask — the masks are written as compressed ``predictions/<rel>.npz``
    (array key ``"mask"``).  Experiments that do not set it simply produce no
    predictions/ folder.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_id: str,
        output_dir: Optional[str] = None,
    ):
        self.config        = config
        self.experiment_id = experiment_id
        self.results_root  = Path(output_dir or config.output_dir)

        # ── Compose the run directory ───────────────────────────────────────
        # Run directory layout:
        #   Zero-shot:  results/<family>/<variant>/<dataset>/zero_shot_inference/<timestamp>/
        #   Fewshot:    results/<family>/<variant>/<dataset>/fewshot_finetuning/<timestamp>/
        # (No experiment_subtype segment; subtype is part of experiment_type directory name)
        family, variant = split_model_name(config.model_name)

        if config.experiment_type == "inference":
            parent = (
                self.results_root
                / family / variant
                / config.dataset_name / "zero_shot_inference"
            )
        elif config.experiment_type == "fewshot_finetuning":
            parent = (
                self.results_root
                / family / variant
                / config.dataset_name / "fewshot_finetuning"
            )
        else:
            # Fallback for unknown experiment types (shouldn't occur in practice)
            subtype = _experiment_subtype(config)
            parent = (
                self.results_root
                / family / variant
                / config.dataset_name / config.experiment_type / subtype
            )
        # Create the run directory atomically — mkdir(exist_ok=False) is the
        # uniqueness guarantee (no check-then-create race for parallel runs that
        # share this parent). Name stays the plain microsecond timestamp.
        self.exp_dir = _create_run_dir(parent)
        self.run_id  = self.exp_dir.name

        # Per-run log lives inside the run directory (no separate logs/ tree).
        self.log_file = self.exp_dir / "experiment.log"

        logger.info(f"ResultLogger: run dir → {self.exp_dir}")

    # ── Public API ──────────────────────────────────────────────────────────

    def log(self, results: List[ExperimentResult]) -> Path:
        """Write all per-run files and append to the master logs.
        Returns the run directory path.
        """
        for result in results:
            data = result.to_dict()
            self._write_json(data, self.exp_dir / "result.json")
            self._write_json(data.get("config", {}), self.exp_dir / "config.json")
            self._write_json(
                {
                    "performance_metrics":  data.get("performance_metrics", {}),
                    "inference_efficiency": data.get("inference_efficiency", {}),
                    "training_efficiency":  data.get("training_efficiency", {}),
                },
                self.exp_dir / "metrics.json",
            )
            self._write_per_image_csv(result)
            self._write_predictions(result)
            logger.info(f"  Logged run → {self.exp_dir}")
        return self.exp_dir

    # ── Internal helpers ────────────────────────────────────────────────────

    def _write_json(self, data: dict, path: Path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_serialise)

    def _write_per_image_csv(self, result: ExperimentResult):
        """Write one self-identifying row per evaluated image.

        For fewshot_finetuning: pivots metrics across stages (zero_shot, full_model,
        frozen_encoder) into separate columns with stage prefix (zs_, fm_, fe_).

        For other experiment types: writes one row per image with all available fields.

        The run's identity is denormalised onto every row so the file can be
        concatenated across runs and analysed directly.
        """
        records = getattr(result, "performance_metrics_per_image", None)
        if not records:
            return

        cfg = result.config or {}
        family, variant = split_model_name(result.model_name)
        identity = {
            "run_id":             self.run_id,
            "model_name":         result.model_name,
            "model_family":       family,
            "model_variant":      variant,
            "dataset_name":       result.dataset_name,
            "experiment_type":    result.experiment_type,
            "experiment_subtype": _experiment_subtype(self.config),
            "seed":               cfg.get("seed"),
        }

        # For fewshot_finetuning: pivot metrics across stages into separate columns
        if result.experiment_type == "fewshot_finetuning":
            rows = self._pivot_fewshot_metrics(records, identity)
        else:
            # For other experiment types: flat structure (one row per stage per image)
            rows = [{**identity, **rec} for rec in records]

        # NaN (e.g. aji_plus/pq/nsd on a both_empty image) must render as a
        # blank cell, matching the derived _p column's existing blank-for-missing
        # convention -- csv.DictWriter writes the literal string "nan" for a
        # raw float('nan'), which is indistinguishable from a real numeric
        # string and inconsistent with every other "missing" cell in this file.
        rows = [_nan_to_none(row) for row in rows]

        # Build fieldnames
        fieldnames = list(identity.keys())
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        out = self.exp_dir / "per_image_metrics.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  Wrote {len(rows)} per-image record(s) → {out}")

    def _pivot_fewshot_metrics(self, records: list, identity: dict) -> list:
        """Pivot fewshot_finetuning metrics: one row per image with all stages' metrics.

        Input: records with 'stage' field (zero_shot, full_model, frozen_encoder)
        Output: one row per image with prefixed metric columns (zs_, fm_, fe_)
        """
        # Group by image_id
        by_image = {}
        for rec in records:
            img_id = rec.get("image_id")
            stage = rec.get("stage", "default")
            if img_id not in by_image:
                by_image[img_id] = {}
            by_image[img_id][stage] = rec

        rows = []
        stage_prefix_map = {"zero_shot": "zs", "full_model": "fm", "frozen_encoder": "fe"}

        for img_id, stage_data in by_image.items():
            row = {**identity}
            row["image_id"] = img_id

            # Copy dataset-level info from any stage (they're the same)
            first_stage = next(iter(stage_data.values()))
            for key in ["n_gt_instances", "n_pred_instances", "tissue_group"]:
                if key in first_stage:
                    row[key] = first_stage[key]

            # Pivot metric columns per stage
            for stage, stage_rec in stage_data.items():
                prefix = stage_prefix_map.get(stage, stage)
                for key, val in stage_rec.items():
                    if key not in ["image_id", "stage", "tissue_group", "n_gt_instances", "n_pred_instances", "included", "exclusion_reason", "metric_error"]:
                        if key in ["aji_plus", "pq", "nsd"]:
                            row[f"{prefix}_{key}"] = val

            # Compute performance score (harmonic mean) for each stage. A
            # component that is exactly 0.0 is a valid result and must
            # collapse the score to 0.0 -- it must never be silently dropped
            # (see the "zero is never softened" contract in instance_seg.py /
            # sapi_score.py). Only a genuinely missing/NaN component (e.g. a
            # both_empty image, where aji_plus/pq/nsd are NaN) leaves the
            # column unset, since a two-component P is not P.
            for stage, stage_rec in stage_data.items():
                prefix = stage_prefix_map.get(stage, stage)
                aji = stage_rec.get("aji_plus")
                pq = stage_rec.get("pq")
                nsd = stage_rec.get("nsd")
                vals = (aji, pq, nsd)
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
                    continue
                if any(v <= 0 for v in vals):
                    row[f"{prefix}_p"] = 0.0
                else:
                    row[f"{prefix}_p"] = 3.0 / (1.0/aji + 1.0/pq + 1.0/nsd)

            rows.append(row)

        return rows

    def _write_predictions(self, result: ExperimentResult):
        """Write predicted masks to predictions/, if enabled and the experiment attached any.

        Handles both flat and hierarchical predictions structures:
        - Flat: {name: mask, ...}
        - Hierarchical (fewshot_finetuning): {mode: {name: mask, ...}, ...}

        Off unless ``ExperimentConfig.save_predictions`` is True (CLI:
        ``--save_predictions``).  This is the single gate: experiments always attach
        their masks to the result, so turning it back on needs no change to them.
        Nothing downstream reads predictions/ — every metric and SAPI score comes
        from result.json / master_log.csv.
        """
        if not result.config.get("save_predictions", False):
            return
        preds = getattr(result, "predictions", None)
        if not preds:
            return
        pred_dir = self.exp_dir / "predictions"
        n = 0

        # Handle both flat and hierarchical structures
        for key, value in preds.items():
            if isinstance(value, dict):
                # Hierarchical: {mode: {name: mask, ...}, ...}
                mode_name = key
                for rel, mask in value.items():
                    if mask is None:
                        continue
                    # Validate mask type to prevent triple-nested dicts or corrupt data
                    if isinstance(mask, dict):
                        raise TypeError(f"Predictions too deeply nested: {mode_name}/{rel} is a dict, expected numpy array")
                    if not isinstance(mask, np.ndarray):
                        raise TypeError(f"Prediction mask must be numpy array, got {type(mask).__name__} at {mode_name}/{rel}")
                    relp = Path(rel)
                    out = pred_dir / mode_name / relp.parent / f"{_safe_name(relp.name)}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, mask=np.asarray(mask))
                    n += 1
            else:
                # Flat: {name: mask, ...}
                mask = value
                if mask is None:
                    continue
                # Validate mask type
                if isinstance(mask, dict):
                    raise TypeError(f"Predictions too deeply nested: {key} is a dict, expected numpy array")
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"Prediction mask must be numpy array, got {type(mask).__name__} at {key}")
                relp = Path(key)
                out = pred_dir / relp.parent / f"{_safe_name(relp.name)}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, mask=np.asarray(mask))
                n += 1

        if n:
            logger.info(f"  Wrote {n} prediction mask(s) → {pred_dir}")


# ---------------------------------------------------------------------------
# JSON serialiser for NumPy types
# ---------------------------------------------------------------------------

def _json_serialise(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

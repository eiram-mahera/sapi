#!/usr/bin/env python3
"""
Cell Instance Segmentation Sustainability Benchmark – main runner.

Runs ONE experiment: 1 model × 1 dataset × 1 experiment type.
The date and time are recorded in each result file so experiments run
at different times of day (with different grid carbon intensities) can
be directly compared.

Experiment types
----------------
inference                Run the pretrained model on ALL images in the
                          dataset. No training required.

fewshot_finetuning            Fine-tune once on a tissue-stratified support set
                          (one image per tissue group); fixed epoch budget,
                          no early stopping (patience=0).

Usage examples
--------------
python run_experiment.py \\
    --model cellpose_cyto3 \\
    --dataset DAPI \\
    --experiment_type inference

python run_experiment.py \\
    --model cellpose_cyto3 \\
    --dataset mIF \\
    --experiment_type fewshot_finetuning \\
    --n_epochs 50 \\
    --electricitymaps_token YOUR_TOKEN_HERE

Available models
----------------
  cellpose_cyto2, cellpose_cyto3, cellpose_sam
  stardist_fluo, stardist_he
  deepcell                                ← DeepCell NuclearSegmentation (TF, inference-only)
  mesmer
  cellvit_256, cellvit_sam_h              ← CellViT (Hörst et al. 2024)
  mediar                                  ← NeurIPS 2022 winner; all modalities
  hovernet                                ← H&E specialist (HE, 10xGenomics_HE only)
  instanseg                               ← embedding-based, all modalities
  cellsam                                 ← SAM + CellFinder foundation model, inference-only
  microsam_vit_t_lm, microsam_vit_b_lm, microsam_vit_l_lm
  pathosamvit_b, pathosamvit_l, pathosamvit_h ← H&E-only

Available datasets
------------------
  DAPI, HE, mIF, ssDNA, 10xGenomics_DAPI, 10xGenomics_HE
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cell instance segmentation sustainability benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment identity
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. cellpose_cyto3, stardist_fluo, deepcell)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset: DAPI | HE | mIF | ssDNA | 10xGenomics_DAPI | 10xGenomics_HE")
    parser.add_argument("--experiment_type", required=True,
                        choices=["inference", "fewshot_finetuning"],
                        help="Experiment type")

    # Data
    parser.add_argument("--data_root", default=None,
                        help="Path to the CellBinDB root directory. Default: "
                             "<project_root>/data/CellBinDB (resolved relative to "
                             "this script's location, not the current working "
                             "directory).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible data splits and model init")

    # Training hyperparameters (few-shot only)
    # Defaults are None so the model YAML value takes priority when not set on CLI.
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Training epochs. Default: from model YAML "
                             "(fewshot_finetuning section, falls back to fewshot), "
                             "else 100.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate. Default: from model YAML, else 1e-4.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference. Fewshot training batch size is "
                             "capped automatically and not controlled by this flag: "
                             "max(1, min(n_support, 8)) for fewshot_finetuning.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay. Default: from model YAML, else 1e-5.")
    parser.add_argument("--save_predictions", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Write predicted instance masks to "
                             "<run_dir>/predictions/<sample>.npz. Default: off — the "
                             "masks are large and no metric or SAPI score needs them "
                             "(everything downstream reads result.json / master_log.csv).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience (epochs without val-loss improvement). "
                             "0 disables early stopping. "
                             "Default: from model YAML (fewshot_finetuning section, falls "
                             "back to fewshot), else 20. "
                             "NOTE: has no effect — fewshot_finetuning forces patience=0 "
                             "unconditionally (no validation image). Retained as the "
                             "wired-up entry point for future early-stopping use with real "
                             "validation images.")

    # Hardware & sustainability
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--pue", type=float, default=1.3,
                        help="Power Usage Effectiveness: 1.0=laptop, 1.3=data centre")
    parser.add_argument("--baseline_duration", type=float, default=2.0,
                        help="Seconds to measure idle GPU power before each timed block")
    parser.add_argument("--electricitymaps_token", default="",
                        metavar="TOKEN",
                        help="Electricity Maps API token for real-time carbon intensity. "
                             "Free tier: https://electricitymaps.com/free-tier")
    parser.add_argument("--electricitymaps_zone", default="",
                        metavar="ZONE",
                        help="Electricity Maps zone code (e.g. 'DE', 'US-CAL-CISO'). "
                             "Default: the 'zone' field in configs/electricity_maps.json, "
                             "else 'DE'. Takes priority over the config file even when "
                             "--electricitymaps_token is also passed on the CLI.")
    parser.add_argument("--electricitymaps_proxy", default="",
                        metavar="URL",
                        help="HTTP/HTTPS proxy for reaching the Electricity Maps API from "
                             "HPC compute nodes with no direct internet access, "
                             "e.g. http://proxy.hpc.edu:3128. "
                             "If unset, HTTP_PROXY / HTTPS_PROXY env vars are used.")
    parser.add_argument("--electricitymaps_intensity", type=float, default=0.0,
                        metavar="G_CO2_PER_KWH",
                        help="Pre-fetched carbon intensity in gCO2eq/kWh. When provided "
                             "(non-zero), the timeline file and API call are skipped. "
                             "Default 0 = use timeline file, then API, then static fallback. "
                             "Intended for HPC compute nodes with no internet access: "
                             "fetch the value on the login node before submitting your job.")
    parser.add_argument("--ci_timeline", default="",
                        metavar="PATH",
                        help="Path to a carbon intensity timeline CSV produced by "
                             "collect_carbon_intensity.py. The entry whose timestamp is "
                             "closest to experiment start time is used, and no API call is "
                             "made. If not provided, the script auto-detects "
                             "configs/ci_timeline.csv in the project root. "
                             "Recommended for HPC nodes with no internet access.")

    # Output
    parser.add_argument("--output_dir", default=None,
                        help="Root directory for result files. Default: "
                             "<project_root>/results (resolved relative to this "
                             "script's location, not the current working "
                             "directory).")
    parser.add_argument("--gpu_type", default="",
                        metavar="GPU_NAME",
                        help="GPU display name recorded in the result/config JSON "
                             "(e.g. 'Tesla_V100-PCIE-32GB'), for later filtering/"
                             "grouping of results by hardware. Not used in the "
                             "results directory path. Auto-detected from "
                             "torch.cuda.get_device_name(gpu_index) when not "
                             "provided.")

    # Model-specific overrides
    parser.add_argument("--model_config", nargs="*", metavar="KEY=VALUE",
                        help="Override model config, e.g. --model_config diameter=30")

    # Logging
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _detect_gpu_type(device: str, gpu_index: int = 0) -> str:
    """Return a sanitised GPU display name for use in directory paths.

    Falls back to 'CPU' if no GPU is available or detectable.
    """
    if device == "cpu":
        return "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(gpu_index)
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(h)
        pynvml.nvmlShutdown()
        return name.decode("utf-8") if isinstance(name, bytes) else name
    except Exception:
        pass
    return "GPU"


def _setup_logging(level: str) -> None:
    """Configure stdout-only logging for startup messages.

    A structured FileHandler is added later, once ResultLogger has determined
    the experiment's log directory.
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class _RedactionFilter(logging.Filter):
    """Scrub secrets and the user's home path from every log record.

    Results and logs are meant to be shareable, so we make sure no API
    credential (even inside a stray message or traceback) and no absolute
    home-directory path (which reveals the OS username) ever reaches the log.
    """

    def __init__(self, secrets):
        super().__init__()
        # Only redact non-empty, non-trivial secret strings.
        self._secrets = sorted({s for s in secrets if s and len(str(s)) >= 4},
                               key=len, reverse=True)
        from pathlib import Path as _P
        self._home = str(_P.home())

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        original = msg
        for s in self._secrets:
            if s in msg:
                msg = msg.replace(s, "[REDACTED]")
        if self._home and self._home in msg:
            msg = msg.replace(self._home, "~")
        if msg != original:
            record.msg = msg
            record.args = ()
        return True


def _check_cuda(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logging.getLogger(__name__).warning(
                    "CUDA not available; falling back to CPU."
                )
                return "cpu"
            return "cuda"
        except ImportError:
            pass
        # PyTorch not present (e.g. tf215env/tf28env) — check via TensorFlow
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices("GPU"):
                return "cuda"
        except Exception:
            pass
        logging.getLogger(__name__).warning(
            "CUDA not available (no PyTorch or TF GPU found); falling back to CPU."
        )
        return "cpu"
    return device


def _load_electricitymaps_token(cli_token: str, cli_zone: str = "") -> tuple[str, str]:
    """
    Return (api_token, zone) for Electricity Maps.

    Token priority:
    1. --electricitymaps_token CLI flag (if non-empty)
    2. configs/electricity_maps.json in the project root
    3. Empty string (profiler will fall back to static Germany average)

    Zone priority (resolved independently of the token, so passing a CLI token
    does not silently discard a non-default zone configured in the JSON file):
    1. --electricitymaps_zone CLI flag (if non-empty)
    2. configs/electricity_maps.json in the project root
    3. 'DE' (Germany) as a last-resort default
    """
    config_token, config_zone = "", ""
    config_path = _PROJECT_ROOT / "configs" / "electricity_maps.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            config_token = data.get("api_token", "")
            config_zone = data.get("zone", "")
            if config_token and not cli_token:
                logging.getLogger(__name__).info(
                    f"Loaded Electricity Maps token from {config_path}"
                )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"Could not read {config_path}: {exc}"
            )

    token = cli_token or config_token
    zone = cli_zone or config_zone or "DE"
    return token, zone


def _load_deepcell_token() -> str:
    """
    Return the DeepCell access token.

    Priority:
    1. DEEPCELL_ACCESS_TOKEN environment variable (already set)
    2. configs/deepcell.json in the project root
    3. Empty string (DeepCell will attempt unauthenticated download)
    """
    import os
    if os.environ.get("DEEPCELL_ACCESS_TOKEN"):
        return os.environ["DEEPCELL_ACCESS_TOKEN"]

    config_path = _PROJECT_ROOT / "configs" / "deepcell.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            token = data.get("access_token", "")
            if token:
                logging.getLogger(__name__).info(
                    f"Loaded DeepCell token from {config_path}"
                )
                return token
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"Could not read {config_path}: {exc}"
            )

    return ""


def _parse_model_config(pairs: list | None) -> dict:
    if not pairs:
        return {}
    config: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid model_config entry '{item}'; expected KEY=VALUE")
        key, val = item.split("=", 1)
        if val.lower() in ("null", "none"):
            val = None
        elif val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        config[key.strip()] = val
    return config


# ---------------------------------------------------------------------------
# Model YAML helpers
# ---------------------------------------------------------------------------

def _load_model_yaml(model_name: str) -> dict:
    """Load configs/models/{model_name}.yaml. Returns {} if the file is absent."""
    import yaml
    yaml_path = _PROJECT_ROOT / "configs" / "models" / f"{model_name}.yaml"
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path) as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            f"Could not load model YAML {yaml_path}: {exc}"
        )
        return {}


def _yaml_training_section_key(experiment_type: str, model_yaml: dict) -> str:
    """Return the model YAML section key holding training defaults for
    experiment_type.

    The candidate sections (highest priority first) are declared on the
    experiment class itself as ``yaml_section_chain`` — so this resolver and
    _build_model_extra_config both read from the same source of truth and a new
    experiment type needs no edit here.  The first section that is present in the
    model YAML wins; if none is present, the lowest-priority section in the chain
    is used (or the experiment_type itself when the chain is empty, e.g.
    inference, which reads no training hyperparameters).
    """
    from sapi.experiments import EXPERIMENT_REGISTRY
    exp_cls = EXPERIMENT_REGISTRY.get(experiment_type)
    chain = tuple(getattr(exp_cls, "yaml_section_chain", ())) if exp_cls else ()
    for section in chain:
        if model_yaml.get(section):
            return section
    return chain[-1] if chain else experiment_type


def _resolve_training_params(args: argparse.Namespace, model_yaml: dict) -> dict:
    """
    Resolve n_epochs / learning_rate / weight_decay with priority:
        CLI (explicitly set, i.e. not None) > model YAML > built-in defaults.

    The model YAML is expected to have per-experiment-type sections, e.g.::

        fewshot:
          n_epochs: 10
          learning_rate: 0.05
          weight_decay: 1.0e-5

    For inference experiments these sections are not consulted (no training).
    """
    BUILT_IN = {"n_epochs": 100, "learning_rate": 1e-4, "weight_decay": 1e-5,
                "freeze_encoder": True, "patience": 20}
    yaml_section = model_yaml.get(
        _yaml_training_section_key(args.experiment_type, model_yaml), {}
    )
    resolved = {}
    for key in BUILT_IN:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            resolved[key] = cli_val
        elif key in yaml_section:
            resolved[key] = yaml_section[key]
        else:
            resolved[key] = BUILT_IN[key]
    return resolved


def _inject_dataset_min_size(model_cfg: dict, dataset_name: str) -> dict:
    """Put the dataset's ``min_size`` into model_config so every model reads it.

    ``min_size`` is a property of the DATASET (its cell-size distribution), not of
    the model: configs/datasets/<name>.yaml declares it, no model YAML does.  It
    means "drop every predicted instance whose area is strictly less than this
    value" — so 1 keeps everything.  Each wrapper applies it inside whatever
    post-processing its reference implementation uses; see the per-model notes in
    sapi/models/*.py, and note that InstanSeg's operator is flipped so it
    receives ``min_size - 1``.

    Defaults to 1 (keep everything) when the dataset YAML is absent or omits it.
    """
    from sapi.datasets.cellbindb import _load_dataset_config

    ds_cfg = _load_dataset_config(dataset_name)
    min_size = ds_cfg.get("min_size", 1)
    if min_size is None:
        min_size = 1
    min_size = int(min_size)
    if min_size < 1:
        raise ValueError(
            f"min_size must be >= 1 (configs/datasets/{dataset_name}.yaml has "
            f"{min_size}); 1 means 'keep every instance'."
        )
    model_cfg["min_size"] = min_size
    return model_cfg


def _build_model_extra_config(model_yaml: dict, cli_overrides: dict,
                               exp_type: str = "") -> dict:
    """
    Extract model-specific (non-training) fields from the YAML to pass as
    model_config. Excludes meta-fields and per-experiment training sections.
    CLI --model_config KEY=VALUE overrides take final precedence.

    For fewshot_finetuning experiments, model-specific keys that live inside the
    experiment-type YAML section (e.g. InstanSeg's ``hotstart_epochs`` inside
    ``fewshot_finetuning:``) are also merged in, so they reach
    ``model.finetune()`` via ``self.config.model_config``.  Standard
    training keys (n_epochs, learning_rate, …) are excluded because they are
    already resolved into ExperimentConfig by ``_resolve_training_params``.
    """
    _EXCLUDE = {
        "name", "description", "notes", "compatible_datasets",
        "fewshot", "fewshot_finetuning", "inference",
        # Training fields (these belong in the fewshot/inference sections):
        "n_epochs", "learning_rate", "weight_decay", "batch_size", "freeze_encoder",
        "patience",
        # min_size is a DATASET property (configs/datasets/<name>.yaml), injected by
        # _inject_dataset_min_size().  Excluded here so a stray model YAML key can
        # never shadow the dataset's value.
        "min_size",
    }
    # Keys that belong to ExperimentConfig / the training loop, not to the model.
    _TRAINING_KEYS = {
        "n_epochs", "learning_rate", "weight_decay", "freeze_encoder",
        "patience", "batch_size",
    }
    model_cfg = {k: v for k, v in model_yaml.items() if k not in _EXCLUDE}

    # Merge model-specific sub-keys from the experiment-type section, using the
    # exact same fallback chain as _resolve_training_params (_yaml_training_section_key)
    # so the two functions can never read from different sections for the same run.
    # Only experiments that declare a yaml_section_chain (i.e. the training ones)
    # have a section to merge; inference has an empty chain and is skipped.
    from sapi.experiments import EXPERIMENT_REGISTRY
    _exp_cls = EXPERIMENT_REGISTRY.get(exp_type)
    if getattr(_exp_cls, "yaml_section_chain", ()):
        section_key = _yaml_training_section_key(exp_type, model_yaml)
        for k, v in model_yaml.get(section_key, {}).items():
            if k not in _TRAINING_KEYS and k not in model_cfg:
                model_cfg[k] = v

    model_cfg.update(cli_overrides)
    return model_cfg



def _print_hardware(result, logger):
    hw = result.hardware_info
    if not hw:
        return
    logger.info("\n── HARDWARE ENVIRONMENT ──────────────────────────────────────")
    logger.info(f"  OS:          {hw.get('os_name', '?')}")
    logger.info(f"  CPU:         {hw.get('cpu_model', '?')}")
    logger.info(f"               {hw.get('cpu_physical_cores', '?')} physical / "
                f"{hw.get('cpu_logical_cores', '?')} logical cores")
    logger.info(f"  RAM:         {hw.get('ram_total_gb', '?')} GB total")
    gpus = hw.get("gpus", [])
    for g in gpus:
        logger.info(f"  GPU {g.get('index', 0)}:       {g.get('name', '?')}  "
                    f"{g.get('vram_total_gb', '?')} GB VRAM")
    logger.info(f"  CUDA:        {hw.get('cuda_version', 'n/a')}  "
                f"driver {hw.get('gpu_driver_version', 'n/a')}")
    sw = hw.get("software_versions", {})
    if sw:
        versions = "  ".join(f"{k}={v}" for k, v in sw.items())
        logger.info(f"  Packages:    {versions}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("  Cell Instance Segmentation Sustainability Benchmark")
    logger.info("=" * 70)
    logger.info(f"  Model:           {args.model}")
    logger.info(f"  Dataset:         {args.dataset}")
    logger.info(f"  Experiment type: {args.experiment_type}")
    logger.info(f"  Seed:            {args.seed}")

    # Resolve data_root / output_dir: CLI flag > project-root-relative default.
    # (Defaults are anchored to _PROJECT_ROOT, not the current working directory,
    # so the script behaves the same whether invoked from the repo root or via an
    # absolute path from elsewhere, e.g. a sweep script or cron job.)
    data_root = args.data_root or str(_PROJECT_ROOT / "data" / "CellBinDB")
    output_dir = args.output_dir or str(_PROJECT_ROOT / "results")

    # Resolve CI timeline path: CLI flag > auto-detect default location
    ci_timeline_path = args.ci_timeline
    if not ci_timeline_path:
        default_timeline = _PROJECT_ROOT / "configs" / "ci_timeline.csv"
        if default_timeline.exists():
            ci_timeline_path = str(default_timeline)
            logger.info(f"  CI timeline:     {ci_timeline_path} (auto-detected)")
    elif not Path(ci_timeline_path).exists():
        logger.warning(f"  CI timeline file not found: {ci_timeline_path}")
        ci_timeline_path = ""

    # Detect GPU type (recorded in the result/config JSON; not used in the path)
    device   = _check_cuda(args.device)
    gpu_type = args.gpu_type or _detect_gpu_type(device, gpu_index=args.gpu_index)
    logger.info(f"  GPU type:        {gpu_type}")
    em_token, em_zone = _load_electricitymaps_token(
        args.electricitymaps_token, args.electricitymaps_zone
    )

    # Load model YAML and resolve training hyperparameters
    model_yaml = _load_model_yaml(args.model)
    training = _resolve_training_params(args, model_yaml)
    model_extra_cfg = _build_model_extra_config(
        model_yaml, _parse_model_config(args.model_config),
        exp_type=args.experiment_type,
    )
    _inject_dataset_min_size(model_extra_cfg, args.dataset)

    # Set DeepCell access token so cellsam's first-time weight download
    # succeeds — but only when the weights aren't already cached locally.
    # Only cellsam actually consumes DEEPCELL_ACCESS_TOKEN; loading/logging
    # it for every other model, or when weights are already present, is
    # pure noise.
    import os
    deepcell_token = ""
    if args.model == "cellsam":
        checkpoint_path = model_yaml.get("checkpoint_path") or os.environ.get("CELLSAM_CHECKPOINT")
        if checkpoint_path:
            weights_present = Path(checkpoint_path).expanduser().exists()
        else:
            model_variant = model_yaml.get("model_variant", "cellsam_general")
            default_ckpt = Path.home() / ".deepcell" / "models" / "cellsam_v1.2" / f"{model_variant}.pt"
            weights_present = default_ckpt.exists()
        if not weights_present:
            deepcell_token = _load_deepcell_token()
            if deepcell_token:
                os.environ["DEEPCELL_ACCESS_TOKEN"] = deepcell_token

    if args.experiment_type != "inference":
        logger.info(
            f"  Training params: n_epochs={training['n_epochs']}  "
            f"lr={training['learning_rate']}  "
            f"wd={training['weight_decay']}  "
            f"batch_size={args.batch_size}"
        )
        logger.info(
            f"  Fewshot finetuning:  seed={args.seed}  "
            f"(1 image per tissue group, no validation image, patience=0; "
            f"runs both full_model and frozen_encoder fine-tuning phases "
            f"unconditionally, in one execution)"
        )

    from sapi.experiments.base import ExperimentConfig
    config = ExperimentConfig(
        experiment_type=args.experiment_type,
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=data_root,
        seed=args.seed,
        n_epochs=training["n_epochs"],
        learning_rate=training["learning_rate"],
        batch_size=args.batch_size,
        weight_decay=training["weight_decay"],
        freeze_encoder=training["freeze_encoder"],
        patience=training["patience"],
        device=device,
        gpu_index=args.gpu_index,
        pue=args.pue,
        baseline_duration=args.baseline_duration,
        electricitymaps_api_token=em_token,
        electricitymaps_zone=em_zone,
        electricitymaps_proxy=args.electricitymaps_proxy,
        carbon_intensity_override=args.electricitymaps_intensity,
        ci_timeline_path=ci_timeline_path,
        output_dir=output_dir,
        save_predictions=args.save_predictions,
        gpu_type=gpu_type,
        model_config=model_extra_cfg,
    )

    # Instantiate experiment
    from sapi.experiments import get_experiment
    ExperimentClass = get_experiment(args.experiment_type)
    experiment = ExperimentClass(config)

    # Instantiate logger — this creates the timestamped run directory (and its
    # in-directory experiment.log) before the experiment starts.
    from sapi.logging_utils import ResultLogger
    result_logger = ResultLogger(
        config=config,
        experiment_id=experiment.experiment_id,
        output_dir=output_dir,
    )

    # Attach a FileHandler to the root logger so all subsequent messages are
    # captured in the structured per-experiment log file.
    file_handler = logging.FileHandler(str(result_logger.log_file), mode="w")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)

    # Scrub secrets and the home-dir path from EVERY handler (file + stdout), so
    # neither the log file nor console output can leak the API token/proxy or the
    # OS username. Secrets are the resolved token, proxy, and DeepCell token.
    _redaction = _RedactionFilter(
        secrets=[em_token, args.electricitymaps_token,
                 args.electricitymaps_proxy, deepcell_token]
    )
    for _h in logging.getLogger().handlers:
        _h.addFilter(_redaction)
    logger.info(f"  Log file:        {result_logger.log_file}")
    logger.info(f"  Run directory:   {result_logger.exp_dir}")

    # Run experiment
    logger.info(f"\nStarting experiment (id={experiment.experiment_id})...\n")
    t_start = time.perf_counter()

    results = experiment.execute()

    elapsed = (time.perf_counter() - t_start) / 60.0
    logger.info(f"\nExperiment completed in {elapsed:.1f} min")

    # Log results
    result_logger.log(results)

    # Print summaries — each experiment class formats its own summary block.
    if results:
        _print_hardware(results[0], logger)
        logger.info("\n" + experiment.summarize(results[0]))

    logger.info(f"\nResults directory: {result_logger.exp_dir}")

    # Signal failure to the caller (e.g. sweep scripts) with a non-zero exit
    # code when an experiment failed but was caught internally (result.error is
    # set). The result.json / experiment.log already record the full error; this
    # only makes the process exit status reflect the failure so batch runners
    # count it as failed instead of OK.
    errored = [r for r in results if getattr(r, "status", "success") == "failed"]
    if errored:
        logger.error(
            f"Experiment finished with an error: {errored[0].error} "
            "(exiting with status 1)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

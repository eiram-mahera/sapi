#!/usr/bin/env bash
# =============================================================================
# run_fewshot_finetuning_benchmark.sh
#
# Runs the tissue-stratified benchmark for ONE (seed, epoch count, GPU),
# optionally narrowed to a single model family, variant, and/or dataset.
# Invoke multiple times with different --seed or --epochs values to build the
# variance estimate used for publication tables.
#
# Arguments (all optional — every one has a default)
# --------------------------------------------------
#   --gpu     N   Physical GPU index; CUDA_VISIBLE_DEVICES=N for PyTorch AND TF
#                 (default 0)
#   --seed    N   RNG seed — controls tissue sample selection and model init
#                 (default 0)
#   --epochs  N   Fine-tuning epoch budget (all models; patience=0, no early
#                 stop) (default 10)
#   --model   S   Model family to run, e.g. cellpose, stardist, microsam.
#                 "all" = every model family (default all)
#   --variant S   Model variant to run, e.g. cyto3, fluo, vit_b_lm.
#                 "all" = every variant of the selected family/families
#                 (default all)
#   --dataset S   Dataset modality, e.g. DAPI, HE, mIF. "all" = every dataset
#                 compatible with the selected model(s) (default all)
#   --pue     F   Power Usage Effectiveness (default 1.3 for data centre)
#   --save-predictions   Save predicted instance masks to predictions/ (off by default)
#
# The family/variant naming matches split_model_name() in
# sapi/logging_utils/result_logger.py — e.g. registry name
# "cellpose_cyto3" is family=cellpose, variant=cyto3; "mediar" is
# family=mediar, variant=mediar.
#
# Experiment sequence
# -------------------
# Per (dataset × model) in the main sweep:
#   fewshot_finetuning — ONE unified run per model × dataset that includes:
#     • zero-shot evaluation on the test set (baseline)
#     • fine-tuning of full_model (all parameters unfrozen)
#     • fine-tuning of frozen_encoder (encoder frozen, decoder only)
#     • post-fine-tuning evaluation for both modes
#
# Patience is always 0. fewshot_finetuning passes val_images=[], patience=0
# to model.finetune() so every model trains for exactly EPOCHS epochs with no
# early stopping — regardless of the model's default patience setting.
#
# Dataset order (before randomization):  mIF  10xGenomics_DAPI  10xGenomics_HE  ssDNA  DAPI  HE
# Model order (before randomization):    smallest → largest by parameter count
#
# IMPORTANT: The above is only the order experiments are built in. The full
# (dataset × model) list is then shuffled with Python's random.shuffle(),
# seeded with --seed, before execution — so actual run order is a seeded
# random permutation, not the sequence listed above. See "Randomize
# experiment order" below.
#
# Results layout (designed for cross-seed mean / std computation)
# ---------------------------------------------------------------
#   results/
#     <family>/<variant>/<dataset>/fewshot_finetuning/
#       <timestamp_runid>/result.json   (+ config.json, metrics.json,
#                                        experiment.log; also predictions/
#                                        when --save-predictions is passed)
#
# Runs land directly under results/. Every run gets its own timestamped
# directory (runs are never overwritten). Each result.json contains results
# for all three variants (zero_shot, full_model, frozen_encoder).
# The seed and epoch budget are recorded inside each result.json
# (config.seed, config.n_epochs), so cross-seed / cross-epoch aggregation
# reads them from the JSON, not the path.
# Glob pattern (e.g. stardist_fluo, DAPI):
#   results/stardist/fluo/DAPI/fewshot_finetuning/*/result.json
#
# GPU cooling between experiments
# ---------------------------------
# Each Python subprocess exits cleanly, freeing GPU memory automatically.
# After subprocess exit, this script polls nvidia-smi every 10 s until the
# GPU temperature falls to ≤ 45°C (5-minute hard timeout). This is a
# separate, sweep-level cooldown from — and not identical to — the
# wait_for_gpu_cooldown() helper in sapi/efficiency/profiler.py, which each
# experiment subprocess calls internally (mandatory 60 s sleep, then 30 s
# polling with no timeout). Both share the same 45°C target and serve the
# same purpose: preventing residual GPU heat from contaminating the idle
# power baseline of the next measurement.
#
# Usage
# -----
#   # Everything (all models, variants, datasets) with the defaults:
#   bash scripts/run_fewshot_finetuning_benchmark.sh
#
#   # One model family / variant / dataset on GPU 1, seed 42, 50 epochs:
#   bash scripts/run_fewshot_finetuning_benchmark.sh \
#       --gpu 1 --seed 42 --epochs 50 \
#       --model cellpose --variant cyto3 --dataset DAPI
#
#   # A whole family (all its variants) across all compatible datasets:
#   bash scripts/run_fewshot_finetuning_benchmark.sh --model stardist
#
#   # Five seeds, same epoch budget:
#   for s in 42 84810 15592 4278 98196; do
#       bash scripts/run_fewshot_finetuning_benchmark.sh --seed $s
#   done
# =============================================================================

set -uo pipefail

# Enable unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# ── Argument parsing ──────────────────────────────────────────────────────────
# =============================================================================

GPU=0
SEED=0
EPOCHS=10
MODEL_FILTER="all"
VARIANT_FILTER="all"
DATASET_FILTER="all"
PUE=1.3
SAVE_PREDICTIONS=""

_usage() {
    printf 'Usage: %s [--gpu N] [--seed N] [--epochs N]\n' "$0"
    printf '          [--model S] [--variant S] [--dataset S] [--pue F]\n'
    printf '          [--save-predictions]\n\n'
    printf '  --gpu               N   Physical GPU index (CUDA_VISIBLE_DEVICES=N)          [default 0]\n'
    printf '  --seed              N   RNG seed for tissue sample selection + model init    [default 0]\n'
    printf '  --epochs            N   Fine-tuning epochs (patience=0; exactly N per model) [default 10]\n'
    printf '  --model             S   Model family, or "all"                              [default all]\n'
    printf '  --variant           S   Model variant, or "all"                             [default all]\n'
    printf '  --dataset           S   Dataset modality, or "all" (compatible only)        [default all]\n'
    printf '  --pue               F   Power Usage Effectiveness                          [default 1.3]\n'
    printf '  --save-predictions      Save predicted instance masks to predictions/\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)              GPU="$2";            shift 2 ;;
        --seed)             SEED="$2";           shift 2 ;;
        --epochs)           EPOCHS="$2";         shift 2 ;;
        --model)            MODEL_FILTER="$2";   shift 2 ;;
        --variant)          VARIANT_FILTER="$2"; shift 2 ;;
        --dataset)          DATASET_FILTER="$2"; shift 2 ;;
        --pue)              PUE="$2";            shift 2 ;;
        --save-predictions) SAVE_PREDICTIONS="1"; shift ;;
        -h|--help)          _usage; exit 0 ;;
        *)
            printf 'Unknown argument: %s\n\n' "$1"
            _usage
            exit 1 ;;
    esac
done

# GPU isolation — exported so every subprocess (PyTorch + TensorFlow) inherits.
# CUDA_DEVICE_ORDER=PCI_BUS_ID ensures GPU N matches row N in `nvidia-smi -L`.
export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# PyTorch CUDA allocator: expandable_segments reduces memory fragmentation
# (reserved-but-unallocated cache), reclaiming ~1 GB on large fp32 fine-tuning
# runs.  Exported here so every experiment subprocess inherits it.  Note: this
# defragments the cache but does NOT raise the memory ceiling — a run whose true
# peak exceeds the GPU (e.g. cellpose_sam full_model on ssDNA at batch 8, fp32)
# still needs gradient accumulation (grad_accum_micro_batch in the model YAML)
# to lower the peak itself.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Runs are written directly under results/; ResultLogger builds the tree
# (family/variant/type/subtype/dataset/timestamped-runid) underneath, and each
# run's per-experiment log lives inside its own run directory. The epoch budget
# is recorded inside each result.json (config.n_epochs) rather than in the path.
# LOG_ROOT below holds only this sweep script's own bookkeeping logs (which
# experiments ran / failed), not the per-experiment logs.
OUT_ROOT="${BENCH_ROOT}/results"
LOG_ROOT="${BENCH_ROOT}/logs"

source "${BENCH_ROOT}/configs/cluster_config.sh"

# =============================================================================
# ── Dataset order and compatibility groups ────────────────────────────────────
# =============================================================================

DATASET_ORDER="mIF 10xGenomics_DAPI 10xGenomics_HE ssDNA DAPI HE"

# Word-boundary padding so pattern matching is unambiguous.
_FLUOR=" mIF 10xGenomics_DAPI ssDNA DAPI "
_HE=" 10xGenomics_HE HE "

_compat() {   # usage: _compat GROUP DATASET — returns 0 if compatible
    case "$1" in
        ALL)   return 0 ;;
        FLUOR) [[ "$_FLUOR" == *" $2 "* ]] ;;
        HE)    [[ "$_HE"    == *" $2 "* ]] ;;
        *)     return 1 ;;
    esac
}

# Validate an explicit --dataset value against the known modalities.
if [[ "$DATASET_FILTER" != "all" && " $DATASET_ORDER " != *" $DATASET_FILTER "* ]]; then
    printf 'ERROR: unknown --dataset "%s". Choose one of: %s (or "all").\n' \
        "$DATASET_FILTER" "$DATASET_ORDER"
    exit 1
fi

# Split a registry model name into (family, variant) — mirrors
# split_model_name() in sapi/logging_utils/result_logger.py.
# Sets globals _FAMILY and _VARIANT.
_MODEL_FAMILIES="cellpose stardist cellvit hovernet microsam pathosamvit cellsam"
_split_model() {
    local name="$1" fam
    for fam in $_MODEL_FAMILIES; do
        if [[ "$name" == "$fam" ]]; then
            _FAMILY="$fam"; _VARIANT="$fam"; return
        fi
        if [[ "$name" == "${fam}_"* ]]; then
            _FAMILY="$fam"; _VARIANT="${name#"${fam}"_}"; return
        fi
    done
    # Single-word models (mediar, instanseg, deepcell, mesmer)
    _FAMILY="$name"; _VARIANT="$name"
}

# =============================================================================
# ── Model registry ────────────────────────────────────────────────────────────
# Format: "model_name:finetune:compat_group"
#   finetune     : yes | no  (no = inference-only; no fewshot_finetuning steps)
#   compat_group : ALL | FLUOR | HE
# Ordered smallest → largest param count.
# The conda env of each model comes from MODEL_ENV in configs/cluster_config.sh.
# =============================================================================

MODELS=(
    "stardist_fluo:yes:FLUOR"        #  1.4 M
    "stardist_he:yes:HE"             #  1.4 M
    "instanseg:yes:ALL"              #  3.8 M
    "cellpose_cyto2:yes:ALL"         #  6.6 M
    "cellpose_cyto3:yes:ALL"         #  6.6 M
    "microsam_vit_t_lm:yes:FLUOR"    # 21.2 M
    "mesmer:no:ALL"                  # 26.0 M — inference-only
    "hovernet:yes:HE"                # 45.0 M
    "cellvit_256:yes:HE"             # 46.8 M
    "microsam_vit_b_lm:yes:FLUOR"    #103.3 M
    "pathosamvit_b:yes:HE"           #104.8 M
    "deepcell:no:ALL"                #119.4 M — inference-only
    "mediar:yes:ALL"                 #121.3 M
    "cellsam:no:ALL"                 #289.6 M — inference-only
    "cellpose_sam:yes:ALL"           #304.6 M
    "microsam_vit_l_lm:yes:FLUOR"    #321.9 M
    "pathosamvit_l:yes:HE"           #323.4 M
    "pathosamvit_h:yes:HE"           #652.1 M
    "cellvit_sam_h:yes:HE"           #699.7 M
)

# Every model must have an env in MODEL_ENV (configs/cluster_config.sh).
_missing_env=()
for _entry in "${MODELS[@]}"; do
    _m="${_entry%%:*}"
    [[ -n "${MODEL_ENV[$_m]:-}" ]] || _missing_env+=("$_m")
done
if (( ${#_missing_env[@]} > 0 )); then
    printf 'ERROR: no MODEL_ENV entry in configs/cluster_config.sh for: %s\n' "${_missing_env[*]}"
    exit 1
fi

# =============================================================================
# ── Build ordered experiment list ─────────────────────────────────────────────
# Each entry: "model:env:dataset"
#
# Each experiment is a single fewshot_finetuning run that internally computes
# results for all three variants (zero_shot, full_model, frozen_encoder).
# Inference-only models (mesmer, deepcell, cellsam) are skipped entirely.
#
# Loop: dataset (outer) → model by param count (inner)
# =============================================================================

# Inference-only models: no fewshot_finetuning steps, zero-shot inference only.
_INFERENCE_ONLY="mesmer deepcell cellsam"
_is_inference_only() {   # returns 0 if $1 is an inference-only model
    [[ " $_INFERENCE_ONLY " == *" $1 "* ]]
}

EXPS=()

# ── Build experiment list: all fine-tunable models (or single --model) ─────────
# Note: inference-only models are skipped; fewshot_finetuning now includes
# built-in zero-shot evaluation on the same test set before fine-tuning.
for _ds in $DATASET_ORDER; do
    # Dataset filter
    [[ "$DATASET_FILTER" == "all" || "$DATASET_FILTER" == "$_ds" ]] || continue
    for _entry in "${MODELS[@]}"; do
        IFS=':' read -r _m _ft _grp <<< "$_entry"
        _e="${MODEL_ENV[$_m]}"
        # Skip inference-only models — only run fine-tunable models
        if _is_inference_only "$_m"; then
            continue
        fi
        # Skip models that don't support fine-tuning
        if [[ "$_ft" != "yes" ]]; then
            continue
        fi
        _split_model "$_m"
        # Model-family filter
        [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "$_FAMILY" ]] || continue
        # Variant filter
        [[ "$VARIANT_FILTER" == "all" || "$VARIANT_FILTER" == "$_VARIANT" ]] || continue
        # Dataset ↔ model compatibility
        _compat "$_grp" "$_ds" || continue
        # Add ONE fewshot_finetuning experiment (unified execution internally
        # computes both full_model and frozen_encoder modes)
        EXPS+=("${_m}:${_e}:${_ds}")
    done
done
N_TOTAL=${#EXPS[@]}

if (( N_TOTAL == 0 )); then
    printf 'ERROR: no experiments match the given filters:\n'
    printf '  --model=%s  --variant=%s  --dataset=%s\n' \
        "$MODEL_FILTER" "$VARIANT_FILTER" "$DATASET_FILTER"
    printf 'Check that the family/variant exist and the dataset is compatible.\n'
    exit 1
fi

# =============================================================================
# ── Randomize experiment order using seed ──────────────────────────────────────
# Uses Python's random module seeded with the same seed passed to the script,
# ensuring reproducible shuffling across runs.
# =============================================================================

# Shuffle EXPS array using seed
mapfile -t EXPS < <(python3 -c "
import random
import sys
random.seed($SEED)
items = sys.argv[1:]
random.shuffle(items)
for item in items:
    print(item)
" "${EXPS[@]}")

# =============================================================================
# ── Logging setup ─────────────────────────────────────────────────────────────
# =============================================================================

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
_TS="$(date '+%Y%m%d_%H%M%S')"
SCRIPT_LOG="${LOG_ROOT}/sweep_gpu${GPU}_${_TS}.log"
FAILED_LOG="${LOG_ROOT}/sweep_gpu${GPU}_${_TS}_failed.txt"
exec > >(tee -a "$SCRIPT_LOG") 2>&1
: > "$FAILED_LOG"

# =============================================================================
# ── Banner ────────────────────────────────────────────────────────────────────
# =============================================================================

printf '=%.0s' {1..72}; printf '\n'
printf '  Tissue Sweep (with pre-fine-tuning zero-shot) — %s\n' "$(date)"
printf '  seed=%-8s  epochs=%-5s  gpu=%s  pue=%s\n' \
    "$SEED" "$EPOCHS" "$GPU" "$PUE"
printf '  Filters:           model=%s  variant=%s  dataset=%s\n' \
    "$MODEL_FILTER" "$VARIANT_FILTER" "$DATASET_FILTER"
printf '  Total experiments: %d (fewshot_finetuning unified execution)\n' "$N_TOTAL"
printf '  Execution order:   RANDOMIZED (using seed=%s)\n' "$SEED"
printf '  Dataset order:     %s (before randomization)\n' "$DATASET_ORDER"
printf '  Model order:       smallest → largest param count (before randomization)\n'
printf '  Patience:          0 (all models train for exactly %s epochs)\n' "$EPOCHS"
printf '  Each run includes:  zero-shot eval → fine-tuning → post-FT eval\n'
printf '\n'
printf '  Results: %s\n' "$OUT_ROOT"
printf '  Logs:    %s\n' "$LOG_ROOT"
printf '  Script log: %s\n' "$SCRIPT_LOG"
printf '=%.0s' {1..72}; printf '\n\n'

# Print randomized execution order
printf '=%.0s' {1..72}; printf '\n'
printf '  RANDOMIZED EXECUTION ORDER (seed=%s):\n' "$SEED"
printf '=%.0s' {1..72}; printf '\n\n'
for (( i=0; i<N_TOTAL; i++ )); do
    IFS=':' read -r _m _e _d <<< "${EXPS[$i]}"
    printf '  [%3d/%d]  %-30s  dataset: %-20s  env: %s\n' \
        $(( i + 1 )) "$N_TOTAL" "$_m" "$_d" "$_e"
done
printf '\n'
printf '=%.0s' {1..72}; printf '\n\n'

# =============================================================================
# ── Pre-flight: environment check ─────────────────────────────────────────────
# =============================================================================

declare -A ENV_OK=()
for _env in "${!ENV_PYTHON[@]}"; do
    if [[ -x "${ENV_PYTHON[$_env]}" ]]; then
        ENV_OK[$_env]=true
    else
        ENV_OK[$_env]=false
        printf 'WARNING: %-16s  Python not found at %s\n' \
            "$_env" "${ENV_PYTHON[$_env]}"
    fi
done
printf '\n'

[[ -f "${BENCH_ROOT}/configs/tissue_labels.json" ]] || {
    printf 'ERROR: configs/tissue_labels.json not found.\n'
    printf '       This file ships with the repository — check your checkout.\n'
    exit 1
}

# =============================================================================
# ── GPU cooldown ──────────────────────────────────────────────────────────────
# Polls nvidia-smi every 10 s until GPU temperature ≤ 45°C (5-min timeout).
# GPU memory is freed automatically when the Python subprocess exits.
# Temperature polling ensures the next experiment's idle power baseline is not
# contaminated by residual heat from the previous run.
# =============================================================================

_COOL_TARGET=45    # °C
_COOL_POLL_S=10    # poll interval
_COOL_TIMEOUT_S=300

_gpu_cooldown() {
    local elapsed=0 temp
    while (( elapsed < _COOL_TIMEOUT_S )); do
        temp=$(nvidia-smi --query-gpu=temperature.gpu \
               --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null \
               | head -1 | tr -d '[:space:]')
        # Stop polling if nvidia-smi is unavailable (CPU-only node)
        if [[ -z "$temp" || ! "$temp" =~ ^[0-9]+$ ]]; then
            break
        fi
        if (( temp <= _COOL_TARGET )); then
            printf '   [cooldown] GPU %d°C — ready\n' "$temp"
            break
        fi
        printf '   [cooldown] GPU %d°C > %d°C, waiting %ds...\n' \
            "$temp" "$_COOL_TARGET" "$_COOL_POLL_S"
        sleep "$_COOL_POLL_S"
        (( elapsed += _COOL_POLL_S )) || true
    done
}


# =============================================================================
# ── run_one MODEL ENV DATASET ────────────────────────────────────────────────
# Runs unified fewshot_finetuning (includes zero_shot, full_model, frozen_encoder)
# =============================================================================

N_OK=0
N_FAIL=0
FAILED_LABELS=()

run_one() {
    local model="$1" env="$2" dataset="$3"
    local label="${model} / ${dataset}"
    printf '── %s  [%s]\n' "$label" "$(date '+%H:%M:%S')"

    # ── Environment check ─────────────────────────────────────────────────────
    if [[ "${ENV_OK[$env]:-false}" != "true" ]]; then
        printf '   SKIP — env %s not available\n' "$env"
        (( N_FAIL++ )) || true
        FAILED_LABELS+=("SKIP(env): $label")
        printf 'SKIP(env): %s\n' "$label" >> "$FAILED_LOG"
        return 0
    fi

    # ── Unified fewshot_finetuning ────────────────────────────────────────────
    # Note: fewshot_finetuning now includes zero-shot evaluation on the test set
    # before fine-tuning, followed by full_model and frozen_encoder fine-tuning,
    # all in a single unified run.
    local exp_type=fewshot_finetuning
    local -a extra=(--n_epochs "$EPOCHS" --patience 0)
    [[ -n "$SAVE_PREDICTIONS" ]] && extra+=(--save_predictions)

    local device="cuda"

    # ── Execute ───────────────────────────────────────────────────────────────
    local t0; t0=$(date +%s)
    if LD_LIBRARY_PATH="${ENV_LIB[$env]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
       "${ENV_PYTHON[$env]}" -u "${BENCH_ROOT}/run_experiment.py" \
           --model           "$model"    \
           --dataset         "$dataset"  \
           --experiment_type "$exp_type" \
           --seed            "$SEED"     \
           --device          "$device"   \
           --pue             "$PUE"      \
           --output_dir      "$OUT_ROOT" \
           --data_root       "${BENCH_ROOT}/data/CellBinDB" \
           "${extra[@]}"; then
        local elapsed=$(( $(date +%s) - t0 ))
        printf '   [OK]   %dm %02ds\n' $(( elapsed/60 )) $(( elapsed%60 ))
        (( N_OK++ )) || true
    else
        local elapsed=$(( $(date +%s) - t0 ))
        printf '   [FAIL] %dm %02ds\n' $(( elapsed/60 )) $(( elapsed%60 ))
        (( N_FAIL++ )) || true
        FAILED_LABELS+=("$label")
        printf '%s\n' "$label" >> "$FAILED_LOG"
    fi
}

# =============================================================================
# ── Main sweep ────────────────────────────────────────────────────────────────
# =============================================================================

for (( idx=0; idx<N_TOTAL; idx++ )); do
    # Run each unified fewshot_finetuning experiment
    # Each EXPS entry is "model:env:dataset"
    IFS=':' read -r _m _e _d <<< "${EXPS[$idx]}"

    printf '\n'

    # Run experiment (unified execution includes zero_shot, full_model, frozen_encoder)
    printf '[%d/%d] %s / %s  ' "$(( idx + 1 ))" "$N_TOTAL" "$_m" "$_d"
    run_one "$_m" "$_e" "$_d"

    # GPU cooldown between model/dataset pairs (skip after the last pair)
    if (( idx + 1 < N_TOTAL )); then
        _gpu_cooldown
    fi

    # ── Progress ──────────────────────────────────────────────────────────────
    n_done=$(( N_OK + N_FAIL ))
    printf '\n'
    printf '  Completed %d / %d   (OK: %d  failed: %d)\n' \
        "$n_done" "$N_TOTAL" "$N_OK" "$N_FAIL"

    # ── Next experiment details ───────────────────────────────────────────────
    if (( idx + 1 < N_TOTAL )); then
        IFS=':' read -r nm ne nd <<< "${EXPS[$(( idx + 1 ))]}"
        printf '  Next  [%d/%d]:\n' "$(( idx + 2 ))" "$N_TOTAL"
        printf '    model   = %s\n'    "$nm"
        printf '    dataset = %s\n'    "$nd"
        printf '    epochs  = %s\n'    "$EPOCHS"
        printf '    seed    = %s\n'    "$SEED"
        printf '    env     = %s\n'    "$ne"
        printf '    gpu     = %s\n'    "$GPU"
    fi
done

# =============================================================================
# ── Summary ───────────────────────────────────────────────────────────────────
# =============================================================================

printf '\n'
printf '=%.0s' {1..72}; printf '\n'
printf '  Tissue Sweep complete — %s\n' "$(date)"
printf '  seed=%-8s  epochs=%-5s  gpu=%s\n' "$SEED" "$EPOCHS" "$GPU"
printf '  Total: %d  |  OK: %d  |  Failed: %d\n' "$N_TOTAL" "$N_OK" "$N_FAIL"
printf '\n'
printf '  Results: %s\n' "$OUT_ROOT"
printf '  Log:     %s\n' "$SCRIPT_LOG"

if [[ ${#FAILED_LABELS[@]} -gt 0 ]]; then
    printf '\n  Failed / skipped (%d):\n' "${#FAILED_LABELS[@]}"
    for _f in "${FAILED_LABELS[@]}"; do
        printf '    %s\n' "$_f"
    done
    printf '\n  (also written to: %s)\n' "$FAILED_LOG"
fi

printf '=%.0s' {1..72}; printf '\n'

# Analysis hint
printf '\n  Each run includes zero-shot baseline + post-fine-tuning results.\n'
printf '\n  SAPI_ZS and SAPI_LC are already computed and stored in each result.json.\n'

[[ $N_FAIL -eq 0 ]]

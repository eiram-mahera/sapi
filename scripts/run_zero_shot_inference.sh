#!/usr/bin/env bash
# =============================================================================
# scripts/run_zero_shot_inference.sh
#
# Zero-shot (pretrained, no training) inference sweep — the sole source of
# SAPI_ZS. Runs every registered model/variant (all 19 — including
# inference-only models: DeepCell, Mesmer, CellSAM) on every dataset it is
# compatible with, for ONE seed/GPU combo.
#
# Each run is a plain `--experiment_type inference` call to run_experiment.py,
# which already measures and logs everything (performance metrics, energy,
# CO2, timing, hardware info) into the run's own self-contained directory —
# this script adds no logging of its own beyond a sweep-level bookkeeping log.
#
# IMPORTANT: Every run generates a new timestamped directory (never overwrites),
# allowing re-sampling at different times of day to capture carbon intensity
# variation. Predicted instance segmentation masks are never saved to disk by
# this script (--save_predictions is intentionally not passed to
# run_experiment.py) — only result.json/metrics.json are needed for SAPI_ZS.
#
# Output layout:
#   results_zero_shot/<model_family>/<model_variant>/<dataset>/zero_shot_inference/<runid>/
#       result.json         Machine-readable result snapshot
#       config.json         Experiment configuration
#       metrics.json        Performance metrics + efficiency scores
#       experiment.log      Per-experiment log
#
# Arguments (all optional with defaults):
#   --seed N          RNG seed (default 0)
#   --gpu N           CUDA device index (default 0)
#   --pue F           Power Usage Effectiveness multiplier (default 1.3)
#   --output_dir DIR  results root (default "<repo>/results_zero_shot")
#   --dry-run         Print the experiment list and exit without running anything
#
# Usage:
#   ./scripts/run_zero_shot_inference.sh --seed 42 --gpu 0
#   ./scripts/run_zero_shot_inference.sh --seed 42 --gpu 0 --pue 1.0
#   ./scripts/run_zero_shot_inference.sh --seed 42 --gpu 0 --dry-run
# =============================================================================

set -uo pipefail

# Enable unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
SEED=0
GPU=0
PUE=1.3
OUT_ROOT="${BENCH_ROOT}/results_zero_shot"
DRY_RUN=0

_usage() {
    printf 'Usage: %s [--seed N] [--gpu N] [--pue F]\n' "$0"
    printf '          [--output_dir DIR] [--dry-run]\n\n'
    printf '  --seed            N   RNG seed for reproducibility              [default 0]\n'
    printf '  --gpu             N   CUDA device index (nvidia-smi numbering)  [default 0]\n'
    printf '  --pue             F   Power Usage Effectiveness                [default 1.3]\n'
    printf '  --output_dir      DIR Results root directory                   [default results_zero_shot]\n'
    printf '  --dry-run             Print experiment list and exit without running\n'
    printf '\n  Predicted instance masks are never written to disk (saving is always off).\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)           SEED="$2"; shift 2 ;;
        --gpu)            GPU="$2"; shift 2 ;;
        --pue)            PUE="$2"; shift 2 ;;
        --output_dir)     OUT_ROOT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        _usage; exit 0 ;;
        *)                printf 'Unknown argument: %s\n\n' "$1"; _usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

LOG_ROOT="${BENCH_ROOT}/logs/zero_shot"

source "${BENCH_ROOT}/configs/cluster_config.sh"

# ── Dataset sweep order ────────────────────────────────────────────────────────
DATASET_ORDER="mIF 10xGenomics_DAPI 10xGenomics_HE ssDNA DAPI HE"

# ── Compatibility-group routing (mirrors run_fewshot_finetuning_benchmark.sh) ──
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

# ── Full model registry: "model_name:compat_group" ───────────────────────────
# Ordered smallest -> largest by param count (all 19 registered models/variants;
# includes inference-only models: mesmer, deepcell, cellsam).
# The conda env of each model comes from MODEL_ENV in configs/cluster_config.sh.
MODELS=(
    "stardist_fluo:FLUOR"         #  1.4 M
    "stardist_he:HE"              #  1.4 M
    "instanseg:ALL"               #  3.8 M
    "cellpose_cyto2:ALL"          #  6.6 M
    "cellpose_cyto3:ALL"          #  6.6 M
    "microsam_vit_t_lm:FLUOR"     # 21.2 M
    "mesmer:ALL"                  # 26.0 M
    "hovernet:HE"                 # 45.0 M
    "cellvit_256:HE"              # 46.8 M
    "microsam_vit_b_lm:FLUOR"     #103.3 M
    "pathosamvit_b:HE"            #104.8 M
    "deepcell:ALL"                #119.4 M
    "mediar:ALL"                  #121.3 M
    "cellsam:ALL"                 #289.6 M
    "cellpose_sam:ALL"            #304.6 M
    "microsam_vit_l_lm:FLUOR"     #321.9 M
    "pathosamvit_l:HE"            #323.4 M
    "pathosamvit_h:HE"            #652.1 M
    "cellvit_sam_h:HE"            #699.7 M
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

# ── Build the experiment list ─────────────────────────────────────────────────
EXPS=()
for _ds in $DATASET_ORDER; do
    for _entry in "${MODELS[@]}"; do
        IFS=':' read -r _m _grp <<< "$_entry"
        _e="${MODEL_ENV[$_m]}"
        _compat "$_grp" "$_ds" || continue
        EXPS+=("${_m}:${_e}:${_ds}")
    done
done
N_TOTAL=${#EXPS[@]}

# ── Logging setup ──────────────────────────────────────────────────────────────
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
_TS="$(date '+%Y%m%d_%H%M%S')"
SCRIPT_LOG="${LOG_ROOT}/sweep_gpu${GPU}_${_TS}.log"
FAILED_LOG="${LOG_ROOT}/sweep_gpu${GPU}_${_TS}_failed.txt"
exec > >(tee -a "$SCRIPT_LOG") 2>&1
: > "$FAILED_LOG"

echo "============================================================"
echo " Zero-shot inference sweep (SAPI_ZS source)"
echo "============================================================"
echo " Seed:          $SEED"
echo " GPU:           $GPU"
echo " PUE:           $PUE"
echo " Results root:  $OUT_ROOT"
echo " Sweep log:     $SCRIPT_LOG"
echo " Dataset order: $DATASET_ORDER"
echo " Total runs:    $N_TOTAL"
echo " Re-run policy: Always create new timestamped runs (never skip)"
echo "============================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    printf '%s\n' "${EXPS[@]}"
    echo "(dry run — no experiments executed)"
    exit 0
fi

# ── Pre-flight: check every conda env python exists ──────────────────────────
declare -A ENV_OK=()
for _env in "${!ENV_PYTHON[@]}"; do
    if [[ -x "${ENV_PYTHON[$_env]}" ]]; then
        ENV_OK[$_env]=true
    else
        ENV_OK[$_env]=false
        printf 'WARNING: %-16s  Python not found at %s\n' "$_env" "${ENV_PYTHON[$_env]}"
    fi
done

# ── GPU cooldown (avoid measuring a hot GPU baseline between runs) ────────────
_COOL_TARGET=45    # degC
_COOL_POLL_S=10    # poll interval
_COOL_TIMEOUT_S=300

_gpu_cooldown() {
    local elapsed=0 temp
    while (( elapsed < _COOL_TIMEOUT_S )); do
        temp=$(nvidia-smi --query-gpu=temperature.gpu \
               --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null \
               | head -1 | tr -d '[:space:]')
        if [[ -z "$temp" || ! "$temp" =~ ^[0-9]+$ ]]; then
            break
        fi
        if (( temp <= _COOL_TARGET )); then
            printf '   [cooldown] GPU %d°C — ready\n' "$temp"
            break
        fi
        printf '   [cooldown] GPU %d°C > %d°C, waiting %ds...\n' "$temp" "$_COOL_TARGET" "$_COOL_POLL_S"
        sleep "$_COOL_POLL_S"
        (( elapsed += _COOL_POLL_S )) || true
    done
}


# ── Run executor ────────────────────────────────────────────────────────────
N_OK=0
N_FAIL=0
FAILED_LABELS=()

run_one() {
    local model="$1" env="$2" dataset="$3"
    local label="${model} / ${dataset}"
    printf '── %s  [%s]\n' "$label" "$(date '+%H:%M:%S')"

    if [[ "${ENV_OK[$env]:-false}" != "true" ]]; then
        printf '   SKIP — env %s not available\n' "$env"
        (( N_FAIL++ )) || true
        FAILED_LABELS+=("SKIP(env): $label")
        printf 'SKIP(env): %s\n' "$label" >> "$FAILED_LOG"
        return 0
    fi

    local device="cuda"

    local t0; t0=$(date +%s)
    if LD_LIBRARY_PATH="${ENV_LIB[$env]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
       "${ENV_PYTHON[$env]}" -u "${BENCH_ROOT}/run_experiment.py" \
           --model           "$model"    \
           --dataset         "$dataset"  \
           --experiment_type inference   \
           --seed            "$SEED"     \
           --device          "$device"   \
           --pue             "$PUE"      \
           --output_dir      "$OUT_ROOT" \
           --data_root       "${BENCH_ROOT}/data/CellBinDB"; then
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

# ── Main sweep loop ────────────────────────────────────────────────────────────
_i=0
for _exp in "${EXPS[@]}"; do
    (( _i++ )) || true
    IFS=':' read -r _model _env _dataset <<< "$_exp"
    printf '\n[%d/%d] ' "$_i" "$N_TOTAL"
    run_one "$_model" "$_env" "$_dataset"
    if (( _i < N_TOTAL )); then
        _gpu_cooldown
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Zero-shot inference sweep complete"
echo "============================================================"
echo " Total:   $(( N_OK + N_FAIL ))"
echo " OK:      $N_OK"
echo " Failed:  $N_FAIL"
if (( N_FAIL > 0 )); then
    echo " Failed runs (see $FAILED_LOG):"
    printf '   - %s\n' "${FAILED_LABELS[@]}"
fi
echo ""
echo " Results stored in: ${OUT_ROOT}/"
echo " SAPI_ZS is already computed in each result.json (performance_metrics.sapi_zs)"
echo "============================================================"

[[ $N_FAIL -eq 0 ]]

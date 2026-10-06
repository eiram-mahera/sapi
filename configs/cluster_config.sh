#!/usr/bin/env bash
# =============================================================================
# configs/cluster_config.sh — CFADS cluster configuration (single source of truth)
#
# Source this file AFTER setting BENCH_ROOT:
#
#   BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # or equivalent
#   source "${BENCH_ROOT}/configs/cluster_config.sh"
#
# Defines:
#   ENV_PYTHON[name]  — absolute path to Python interpreter per conda env
#   ENV_LIB[name]     — lib/ directory path for LD_LIBRARY_PATH per conda env
#   _ENV_*            — canonical conda env name constants (for conda run -n)
#   MODEL_ENV[model]  — conda env each model runs in (used by the sweep scripts)
#   HOVERNET_ROOT, MEDIAR_ROOT, CELLSAM_ROOT           (exported for model wrappers)
#   HOVERNET_WEIGHTS, MEDIAR_WEIGHTS, ...              (exported; env-var override ok)
#
# Conda environment → model mapping (defined in MODEL_ENV below):
#   tf215env      → stardist_fluo, stardist_he
#   tf28env       → deepcell, mesmer
#   cellpose3_env → cellpose_cyto2, cellpose_cyto3
#   cellpose4_env → cellpose_sam
#   mediar_env    → mediar
#   hovernet_env  → hovernet
#   microsam_env  → microsam_vit_t_lm, microsam_vit_b_lm, microsam_vit_l_lm,
#                   pathosamvit_b, pathosamvit_l, pathosamvit_h
#   cellvitpp_env → cellvit_256, cellvit_sam_h
#   instanseg_env → instanseg
#   cellsam_env   → cellsam
# =============================================================================

# ── Conda roots ───────────────────────────────────────────────────────────────
_TORCH_ROOT="/opt/dacam/user_envs/shei_ei/envs2021.05"
_TF_ROOT="/opt/dacam/user_envs/shei_ei/envs2021.05gpu"

# ── Canonical conda env names (use these constants, never bare strings) ────────
_ENV_TF215="tf215env"
_ENV_TF28="tf28env"
_ENV_CP3="cellpose3_env"
_ENV_CP4="cellpose4_env"
_ENV_MEDIAR="mediar_env"
_ENV_HOVERNET="hovernet_env"
_ENV_MICROSAM="microsam_env"
_ENV_CELLVITPP="cellvitpp_env"
_ENV_INSTANSEG="instanseg_env"
_ENV_CELLSAM="cellsam_env"

# ── Python interpreter per conda env ──────────────────────────────────────────
declare -A ENV_PYTHON=(
    [tf215env]="${_TF_ROOT}/tf215env/bin/python"
    [tf28env]="${_TF_ROOT}/tf28env/bin/python"
    [cellpose3_env]="${_TORCH_ROOT}/cellpose3_env/bin/python"
    [cellpose4_env]="${_TORCH_ROOT}/cellpose4_env/bin/python"
    [mediar_env]="${_TORCH_ROOT}/mediar_env/bin/python"
    [hovernet_env]="${_TORCH_ROOT}/hovernet_env/bin/python"
    [microsam_env]="${_TORCH_ROOT}/microsam_env/bin/python"
    [cellvitpp_env]="${_TORCH_ROOT}/cellvitpp_env/bin/python"
    [instanseg_env]="${_TORCH_ROOT}/instanseg_env/bin/python"
    [cellsam_env]="${_TORCH_ROOT}/cellsam_env/bin/python"
)

# ── lib/ path per conda env (for LD_LIBRARY_PATH) ─────────────────────────────
declare -A ENV_LIB=(
    [tf215env]="${_TF_ROOT}/tf215env/lib"
    [tf28env]="${_TF_ROOT}/tf28env/lib"
    [cellpose3_env]="${_TORCH_ROOT}/cellpose3_env/lib"
    [cellpose4_env]="${_TORCH_ROOT}/cellpose4_env/lib"
    [mediar_env]="${_TORCH_ROOT}/mediar_env/lib"
    [hovernet_env]="${_TORCH_ROOT}/hovernet_env/lib"
    [microsam_env]="${_TORCH_ROOT}/microsam_env/lib"
    [cellvitpp_env]="${_TORCH_ROOT}/cellvitpp_env/lib"
    [instanseg_env]="${_TORCH_ROOT}/instanseg_env/lib"
    [cellsam_env]="${_TORCH_ROOT}/cellsam_env/lib"
)

# ── Conda env per model (registry name → env name) ────────────────────────────
# Used by both sweep scripts. If you merge or split envs, edit only this table
# (and ENV_PYTHON / ENV_LIB above for any new env name).
declare -A MODEL_ENV=(
    [stardist_fluo]="$_ENV_TF215"
    [stardist_he]="$_ENV_TF215"
    [deepcell]="$_ENV_TF28"
    [mesmer]="$_ENV_TF28"
    [cellpose_cyto2]="$_ENV_CP3"
    [cellpose_cyto3]="$_ENV_CP3"
    [cellpose_sam]="$_ENV_CP4"
    [mediar]="$_ENV_MEDIAR"
    [hovernet]="$_ENV_HOVERNET"
    [microsam_vit_t_lm]="$_ENV_MICROSAM"
    [microsam_vit_b_lm]="$_ENV_MICROSAM"
    [microsam_vit_l_lm]="$_ENV_MICROSAM"
    [pathosamvit_b]="$_ENV_MICROSAM"
    [pathosamvit_l]="$_ENV_MICROSAM"
    [pathosamvit_h]="$_ENV_MICROSAM"
    [cellvit_256]="$_ENV_CELLVITPP"
    [cellvit_sam_h]="$_ENV_CELLVITPP"
    [instanseg]="$_ENV_INSTANSEG"
    [cellsam]="$_ENV_CELLSAM"
)

# ── Model source code paths ────────────────────────────────────────────────────
# HoverNet, MEDIAR and CellSAM have no installable pip package, so their source
# must be vendored under third_party/. CellViT, InstanSeg, DeepCell and
# micro-sam/patho-sam are installed via pip (see requirements.txt) and need no
# path here.
# BENCH_ROOT must be set before sourcing this file.
# Exported so model wrappers pick them up via os.environ.
export HOVERNET_ROOT="${BENCH_ROOT}/third_party/hover_net-master"
export MEDIAR_ROOT="${BENCH_ROOT}/third_party/MEDIAR-main"
export CELLSAM_ROOT="${BENCH_ROOT}/third_party/cellSAM-master"

# ── Model weight / checkpoint paths ───────────────────────────────────────────
# Honour a pre-existing env var override; otherwise use the canonical location.
export HOVERNET_WEIGHTS="${HOVERNET_WEIGHTS:-${HOME}/.hovernet/hovernet_original_kumar_notype_pytorch.tar}"
export MEDIAR_WEIGHTS="${MEDIAR_WEIGHTS:-${HOME}/.mediar/finetuned/from_phase2.pth}"
export INSTANSEG_CHECKPOINT="${INSTANSEG_CHECKPOINT:-${HOME}/.instanseg/instanseg.pt}"
export CELLSAM_CHECKPOINT="${CELLSAM_CHECKPOINT:-${HOME}/.deepcell/models/cellsam_v1.2/cellsam_general.pt}"
# No CELLVIT_WEIGHTS default is set here: unlike the models above, CellViT has
# two registered variants (cellvit_256, cellvit_sam_h) with two different
# checkpoint files, each already given its own weight_path in
# configs/models/cellvit_256.yaml / cellvit_sam_h.yaml. A single global
# default would silently point both variants at the same file. Set
# CELLVIT_WEIGHTS in the run's own environment to override a specific run.

#!/usr/bin/env bash
set -euo pipefail

# Sequentially evaluate every run folder under BASE that contains STEP.tar.
# Writes per-folder eval_result_<STEP>.txt and BASE/batch_eval_<STEP>_summary.{txt,csv}.
#
# Example:
#   BASE=./logs/Quan_aux-muon \
#   STEP=200000 \
#   EVAL_SCRIPT=./eval_nerf_checkpoint_safe.py \
#   PROJECT_ROOT=. \
#   NERF_MODULE=run_nerf_ranksched \
#   GPU=0 \
#   bash scripts/run_batch_eval_nerf_tar.sh

BASE=${BASE:?"Please set BASE, e.g. BASE=./logs/Quan_aux-muon"}
STEP=${STEP:-200000}
EVAL_SCRIPT=${EVAL_SCRIPT:-./eval_nerf_checkpoint_safe.py}
PROJECT_ROOT=${PROJECT_ROOT:-.}
NERF_MODULE=${NERF_MODULE:-run_nerf_ranksched}
PYTHON=${PYTHON:-python}
SSIM_IMPL=${SSIM_IMPL:-torch}
GPU=${GPU:-}
INCLUDE_GLOB=${INCLUDE_GLOB:-*}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
OVERWRITE=${OVERWRITE:-1}
SAVE_PNG=${SAVE_PNG:-0}
SAVE_FLOAT_NPY=${SAVE_FLOAT_NPY:-0}
DETERMINISTIC=${DETERMINISTIC:-0}
COMPUTE_LPIPS=${COMPUTE_LPIPS:-0}
LPIPS_NET=${LPIPS_NET:-alex}
SEED=${SEED:-}
BATCH_SCRIPT=${BATCH_SCRIPT:-scripts/batch_eval_nerf_checkpoints.py}

args=(
  --base "$BASE"
  --step "$STEP"
  --eval_script "$EVAL_SCRIPT"
  --project_root "$PROJECT_ROOT"
  --nerf_module "$NERF_MODULE"
  --python "$PYTHON"
  --ssim_impl "$SSIM_IMPL"
  --include_glob "$INCLUDE_GLOB"
)

if [[ -n "$GPU" ]]; then
  args+=(--gpu "$GPU")
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  args+=(--continue_on_error)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "$SAVE_PNG" == "1" ]]; then
  args+=(--save_png)
fi
if [[ "$SAVE_FLOAT_NPY" == "1" ]]; then
  args+=(--save_float_npy)
fi
if [[ "$DETERMINISTIC" == "1" ]]; then
  args+=(--deterministic)
fi
if [[ "$COMPUTE_LPIPS" == "1" ]]; then
  args+=(--compute_lpips --lpips_net "$LPIPS_NET")
fi
if [[ -n "$SEED" ]]; then
  args+=(--seed "$SEED")
fi

"$PYTHON" "$BATCH_SCRIPT" "${args[@]}" "$@"

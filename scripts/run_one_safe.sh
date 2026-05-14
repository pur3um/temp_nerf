#!/usr/bin/env bash
set -euo pipefail

# Minimal single-run wrapper. Override variables from the shell.
# Example:
#   OPTIMIZER=mofa_split EXP=my_mofa DATADIR=./data/nerf_synthetic/lego bash scripts/run_one_safe.sh

OPTIMIZER=${OPTIMIZER:-mofa_split}
EXP=${EXP:-safe_${OPTIMIZER}_$(date +%Y%m%d_%H%M%S)}
DATASET_TYPE=${DATASET_TYPE:-blender}
DATADIR=${DATADIR:-./data/nerf_synthetic/lego}
BASEDIR=${BASEDIR:-./logs_mofasgd_safe}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-0}
N_ITERS=${N_ITERS:-100000}

EXTRA_DATASET_ARGS=()
if [[ "$DATASET_TYPE" == "blender" ]]; then
  EXTRA_DATASET_ARGS+=(--half_res --white_bkgd)
fi

python run_nerf_mofasgd_safe.py \
  --optimizer "$OPTIMIZER" \
  --expname "$EXP" \
  --dataset_type "$DATASET_TYPE" \
  --datadir "$DATADIR" \
  --basedir "$BASEDIR" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --use_viewdirs \
  --N_samples 64 \
  --N_importance 128 \
  --N_rand 4096 \
  --N_iters "$N_ITERS" \
  --i_print 1000 \
  --i_weights 50000 \
  --i_video 0 \
  --i_testset 0 \
  --no_reload \
  "${EXTRA_DATASET_ARGS[@]}"

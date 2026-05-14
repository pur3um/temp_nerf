#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, next to run_nerf_mofasgd_safe.py.
# Example:
#   DATADIR=./data/llff/fern N_ITERS=100000 bash scripts/run_compare_llff_safe.sh

DATADIR=${DATADIR:-./data/llff/fern}
SCENE=${SCENE:-$(basename "$DATADIR")}
BASEDIR=${BASEDIR:-./logs_mofasgd_safe}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EXP_PREFIX=${EXP_PREFIX:-${SCENE}_safe_${RUN_ID}}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-0}
N_ITERS=${N_ITERS:-100000}
N_RAND=${N_RAND:-4096}
N_SAMPLES=${N_SAMPLES:-64}
N_IMPORTANCE=${N_IMPORTANCE:-128}
CHUNK=${CHUNK:-32768}
NETCHUNK=${NETCHUNK:-65536}
I_PRINT=${I_PRINT:-1000}
I_WEIGHTS=${I_WEIGHTS:-50000}
LRATE=${LRATE:-5e-4}
SGD_LRATE=${SGD_LRATE:-5e-4}
MOFA_LRATE=${MOFA_LRATE:-5e-4}
MOFA_RANK=${MOFA_RANK:-16}
MOFA_BETA=${MOFA_BETA:-0.9}
MOFA_ETA1=${MOFA_ETA1:-1.0}
MOFA_ETA2=${MOFA_ETA2:-0.0}
FACTOR=${FACTOR:-8}
LLFFHOLD=${LLFFHOLD:-8}

COMMON_ARGS=(
  --dataset_type llff
  --datadir "$DATADIR"
  --basedir "$BASEDIR"
  --device "$DEVICE"
  --seed "$SEED"
  --factor "$FACTOR"
  --llffhold "$LLFFHOLD"
  --use_viewdirs
  --N_samples "$N_SAMPLES"
  --N_importance "$N_IMPORTANCE"
  --N_rand "$N_RAND"
  --N_iters "$N_ITERS"
  --lrate "$LRATE"
  --chunk "$CHUNK"
  --netchunk "$NETCHUNK"
  --i_print "$I_PRINT"
  --i_weights "$I_WEIGHTS"
  --i_video 0
  --i_testset 0
  --no_reload
)

python run_nerf_mofasgd_safe.py \
  "${COMMON_ARGS[@]}" \
  --optimizer adam \
  --expname "${EXP_PREFIX}_adam"

python run_nerf_mofasgd_safe.py \
  "${COMMON_ARGS[@]}" \
  --optimizer sgd_split \
  --sgd_lrate "$SGD_LRATE" \
  --sgd_momentum 0.9 \
  --expname "${EXP_PREFIX}_sgd_split"

python run_nerf_mofasgd_safe.py \
  "${COMMON_ARGS[@]}" \
  --optimizer mofa_split \
  --mofa_lrate "$MOFA_LRATE" \
  --mofa_rank "$MOFA_RANK" \
  --mofa_beta "$MOFA_BETA" \
  --mofa_eta1 "$MOFA_ETA1" \
  --mofa_eta2 "$MOFA_ETA2" \
  --expname "${EXP_PREFIX}_mofa_split"

python scripts/collect_metrics_safe.py \
  "$BASEDIR/${EXP_PREFIX}_adam/metrics_final.json" \
  "$BASEDIR/${EXP_PREFIX}_sgd_split/metrics_final.json" \
  "$BASEDIR/${EXP_PREFIX}_mofa_split/metrics_final.json" \
  | tee "$BASEDIR/${EXP_PREFIX}_summary.csv"

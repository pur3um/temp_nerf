#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, next to run_nerf_mofasgd_safe.py.
#
# This script uses 8 GPUs by launching 8 independent seed chains in parallel.
# Each GPU runs: Adam -> SGD-split -> MoFaSGD-split sequentially for one seed.
#
# Example:
#   DATADIR=./data/llff/fern N_ITERS=100000 \
#   GPU_IDS="0 1 2 3 4 5 6 7" SEEDS="0 1 2 3 4 5 6 7" \
#   bash scripts/run_compare_llff_safe_8gpu.sh
#
# Important: run_nerf_mofasgd_safe.py is single-process/single-device code.
# This script does NOT make one training job use 8 GPUs via DDP. It uses 8 GPUs
# for 8 independent runs, which is usually the right setup for optimizer
# comparison across seeds.

DATADIR=${DATADIR:-./data/llff/fern}
SCENE=${SCENE:-$(basename "$DATADIR")}
BASEDIR=${BASEDIR:-./logs_mofasgd_safe}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
EXP_PREFIX=${EXP_PREFIX:-${SCENE}_safe_${RUN_ID}}

# Space-separated physical GPU ids and seeds. Keep the same count unless you
# intentionally want only the first min(counts) pairs to run.
GPU_IDS_STR=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
SEEDS_STR=${SEEDS:-"0 1 2 3 4 5 6 7"}
read -r -a GPU_IDS_ARR <<< "$GPU_IDS_STR"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"

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
MOFA_EPS=${MOFA_EPS:-1e-4}
MOFA_UV_EPS=${MOFA_UV_EPS:-0.0}
MOFA_MAX_VALUE=${MOFA_MAX_VALUE:-5.0}
FACTOR=${FACTOR:-8}
LLFFHOLD=${LLFFHOLD:-8}

if [[ ${#GPU_IDS_ARR[@]} -eq 0 ]]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 1
fi
if [[ ${#SEEDS_ARR[@]} -eq 0 ]]; then
  echo "ERROR: SEEDS is empty." >&2
  exit 1
fi
if [[ ${#GPU_IDS_ARR[@]} -ne ${#SEEDS_ARR[@]} ]]; then
  echo "ERROR: GPU_IDS and SEEDS must have the same number of entries." >&2
  echo "  GPU_IDS=$GPU_IDS_STR" >&2
  echo "  SEEDS=$SEEDS_STR" >&2
  exit 1
fi

mkdir -p "$BASEDIR"
mkdir -p "$BASEDIR/${EXP_PREFIX}_launcher_logs"

COMMON_ARGS_BASE=(
  --dataset_type llff
  --datadir "$DATADIR"
  --basedir "$BASEDIR"
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

run_one() {
  local gpu="$1"
  local seed="$2"
  local optimizer="$3"
  local suffix="$4"
  shift 4

  local expname="${EXP_PREFIX}_seed${seed}_${suffix}"
  local logfile="$BASEDIR/${EXP_PREFIX}_launcher_logs/${expname}.gpu${gpu}.log"

  echo "[$(date '+%F %T')] START gpu=${gpu} seed=${seed} optimizer=${optimizer} exp=${expname}"

  # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0 inside this process.
  CUDA_VISIBLE_DEVICES="$gpu" python run_nerf_mofasgd_safe.py \
    "${COMMON_ARGS_BASE[@]}" \
    --device cuda \
    --seed "$seed" \
    --optimizer "$optimizer" \
    --expname "$expname" \
    "$@" \
    2>&1 | tee "$logfile"

  echo "[$(date '+%F %T')] DONE  gpu=${gpu} seed=${seed} optimizer=${optimizer} exp=${expname}"
}

run_seed_chain() {
  local gpu="$1"
  local seed="$2"

  run_one "$gpu" "$seed" adam "adam"

  run_one "$gpu" "$seed" sgd_split "sgd_split" \
    --sgd_lrate "$SGD_LRATE" \
    --sgd_momentum 0.9

  run_one "$gpu" "$seed" mofa_split "mofa_split" \
    --mofa_lrate "$MOFA_LRATE" \
    --mofa_rank "$MOFA_RANK" \
    --mofa_beta "$MOFA_BETA" \
    --mofa_eta1 "$MOFA_ETA1" \
    --mofa_eta2 "$MOFA_ETA2" \
    --mofa_eps "$MOFA_EPS" \
    --mofa_uv_eps "$MOFA_UV_EPS" \
    --mofa_max_value "$MOFA_MAX_VALUE"

  python scripts/collect_metrics_safe.py \
    "$BASEDIR/${EXP_PREFIX}_seed${seed}_adam/metrics_final.json" \
    "$BASEDIR/${EXP_PREFIX}_seed${seed}_sgd_split/metrics_final.json" \
    "$BASEDIR/${EXP_PREFIX}_seed${seed}_mofa_split/metrics_final.json" \
    | tee "$BASEDIR/${EXP_PREFIX}_seed${seed}_summary.csv"
}

pids=()
for idx in "${!GPU_IDS_ARR[@]}"; do
  gpu="${GPU_IDS_ARR[$idx]}"
  seed="${SEEDS_ARR[$idx]}"
  (
    set -euo pipefail
    run_seed_chain "$gpu" "$seed"
  ) &
  pids+=("$!")
  echo "Launched seed=${seed} on physical GPU=${gpu}, pid=${pids[-1]}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "ERROR: at least one GPU worker failed. Check $BASEDIR/${EXP_PREFIX}_launcher_logs/*.log" >&2
  exit "$status"
fi

metric_files=()
for seed in "${SEEDS_ARR[@]}"; do
  metric_files+=("$BASEDIR/${EXP_PREFIX}_seed${seed}_adam/metrics_final.json")
  metric_files+=("$BASEDIR/${EXP_PREFIX}_seed${seed}_sgd_split/metrics_final.json")
  metric_files+=("$BASEDIR/${EXP_PREFIX}_seed${seed}_mofa_split/metrics_final.json")
done

python scripts/collect_metrics_safe.py "${metric_files[@]}" \
  | tee "$BASEDIR/${EXP_PREFIX}_summary_all_seeds.csv"

echo "All done. Combined summary: $BASEDIR/${EXP_PREFIX}_summary_all_seeds.csv"

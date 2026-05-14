# bash wsd_ab.sh configs/lego.txt --datadir ./data/nerf_synthetic/lego --dataset_type blender --white_bkgd

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash wsd_ab.sh [config_path] [extra args...]
# Example:
#   bash wsd_ab.sh configs/lego.txt \
#     --datadir ./data/nerf_synthetic/lego --dataset_type blender --white_bkgd

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-99_run_nerf.py}
CONFIG=${1:-configs/lego.txt}
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

BASEDIR=${BASEDIR:-logs/scheduler_compare_4gpu}
mkdir -p "${BASEDIR}/_stdout"

# Shared training settings
N_ITERS=${N_ITERS:-200000}
LR_ADAM=${LR_ADAM:-5e-4}
LR_MUON=${LR_MUON:-5e-4}
LRATE_DECAY=${LRATE_DECAY:-250}
SEED=${SEED:-0}

# Auto-cos-inc rank schedule
LOWRANK_RANK_START=${LOWRANK_RANK_START:-150}
LOWRANK_RANK_END=${LOWRANK_RANK_END:-250}
LOWRANK_SCHEDULE_STEPS=${LOWRANK_SCHEDULE_STEPS:-100000}
LOWRANK_OVERSAMPLE=${LOWRANK_OVERSAMPLE:-4}
LOWRANK_NS_STEPS=${LOWRANK_NS_STEPS:-5}
AUTO_INIT_RANK_START=${AUTO_INIT_RANK_START:-1}

# Rank-WSD knobs for full-rank Muon
FULLRANK_DECAY_START_FRAC=${FULLRANK_DECAY_START_FRAC:-0.2}
FULLRANK_DECAY_END_FRAC=${FULLRANK_DECAY_END_FRAC:-0.6}
SCHED_MIN_LR_RATIO=${SCHED_MIN_LR_RATIO:-0.1}
SCHED_WARMUP_FRAC=${SCHED_WARMUP_FRAC:-0.01}

# Logging/checkpoint cadence
I_PRINT=${I_PRINT:-2000}
I_WEIGHTS=${I_WEIGHTS:-100000}
I_TESTSET=${I_TESTSET:-100000}
I_VIDEO=${I_VIDEO:-100000}
I_VALSET=${I_VALSET:-0}

COMMON_ARGS=(
  --config "${CONFIG}"
  --basedir "${BASEDIR}"
  --no_reload
  --seed "${SEED}"
  --deterministic
  --N_iters "${N_ITERS}"
  --lrate "${LR_ADAM}"
  --muon_lrate "${LR_MUON}"
  --lrate_decay "${LRATE_DECAY}"
  --lowrank_rank_start "${LOWRANK_RANK_START}"
  --lowrank_rank_end "${LOWRANK_RANK_END}"
  --lowrank_schedule_steps "${LOWRANK_SCHEDULE_STEPS}"
  --lowrank_oversample "${LOWRANK_OVERSAMPLE}"
  --lowrank_ns_steps "${LOWRANK_NS_STEPS}"
  --sched_min_lr_ratio "${SCHED_MIN_LR_RATIO}"
  --sched_warmup_frac "${SCHED_WARMUP_FRAC}"
  --i_print "${I_PRINT}"
  --i_weights "${I_WEIGHTS}"
  --i_testset "${I_TESTSET}"
  --i_video "${I_VIDEO}"
  --i_valset "${I_VALSET}"
)

AUTO_ARGS=()
if [[ "${AUTO_INIT_RANK_START}" == "1" ]]; then
  AUTO_ARGS+=(--lowrank_auto_init_rank_start)
fi

PIDS=()

run_one() {
  local gpu="$1"
  local expname="$2"
  shift 2

  echo "[launch] GPU=${gpu} expname=${expname}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --expname "${expname}" \
    "$@" \
    "${EXTRA_ARGS[@]}" \
    > "${BASEDIR}/_stdout/${expname}.out" 2>&1 &
  PIDS+=("$!")
}

# 1) Muon + rank-aware WSD
run_one 0 "muon_rank_wsd" \
  --optimizer aux-muon \
  --train_scheduler rank_wsd \
  --rank_wsd_fullrank_decay_start_frac "${FULLRANK_DECAY_START_FRAC}" \
  --rank_wsd_fullrank_decay_end_frac "${FULLRANK_DECAY_END_FRAC}"

# 2) Muon + original NeRF exponential decay
run_one 1 "muon_exp_decay" \
  --optimizer aux-muon \
  --train_scheduler exp_decay

# 3) Auto-cos-inc + rank-aware WSD
run_one 2 "autocosinc_rank_wsd" \
  --optimizer aux-sign-auto-cos-inc \
  --train_scheduler rank_wsd \
  "${AUTO_ARGS[@]}"

# 4) Auto-cos-inc + original NeRF exponential decay
run_one 3 "autocosinc_exp_decay" \
  --optimizer aux-sign-auto-cos-inc \
  --train_scheduler exp_decay \
  "${AUTO_ARGS[@]}"

printf '[running] PIDs: %s\n' "${PIDS[*]}"

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[done] all runs finished. Logs: ${BASEDIR}/_stdout"
#!/usr/bin/env bash
set -euo pipefail

# Run the same 4 optimizer/scheduler experiments on two datasets at once:
#   Dataset 1 -> GPUs 0,1,2,3
#   Dataset 2 -> GPUs 4,5,6,7
#
# Edit the DATA*_ lines below for your paths.
# Extra command-line args passed to this script are appended to ALL 8 runs.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHON_BIN=${PYTHON_BIN:-python}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-99_run_nerf.py}

# ===================== Dataset 1: GPUs 0,1,2,3 =====================
DATA1_NAME=${DATA1_NAME:-lego}
DATA1_CONFIG=${DATA1_CONFIG:-configs/lego.txt}
DATA1_DATADIR=${DATA1_DATADIR:-./data/nerf_synthetic/lego}
DATA1_DATASET_TYPE=${DATA1_DATASET_TYPE:-blender}
DATA1_BASEDIR=${DATA1_BASEDIR:-logs/scheduler_compare/lego}
DATA1_EXTRA_ARGS=(--white_bkgd)

# ===================== Dataset 2: GPUs 4,5,6,7 =====================
DATA2_NAME=${DATA2_NAME:-chair}
DATA2_CONFIG=${DATA2_CONFIG:-configs/chair.txt}
DATA2_DATADIR=${DATA2_DATADIR:-./data/nerf_synthetic/chair}
DATA2_DATASET_TYPE=${DATA2_DATASET_TYPE:-blender}
DATA2_BASEDIR=${DATA2_BASEDIR:-logs/scheduler_compare/chair}
DATA2_EXTRA_ARGS=(--white_bkgd)

# Any arguments passed after the script name are applied to both datasets.
# Example: bash run_8gpu_two_datasets_scheduler_compare.sh --N_rand 2048 --half_res
GLOBAL_EXTRA_ARGS=("$@")

# ===================== Shared training settings =====================
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

PIDS=()

AUTO_ARGS=()
if [[ "${AUTO_INIT_RANK_START}" == "1" ]]; then
  AUTO_ARGS+=(--lowrank_auto_init_rank_start)
fi

run_one() {
  local gpu="$1"
  local basedir="$2"
  local expname="$3"
  shift 3

  mkdir -p "${basedir}/_stdout"
  echo "[launch] GPU=${gpu} basedir=${basedir} expname=${expname}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    "$@" \
    --expname "${expname}" \
    > "${basedir}/_stdout/${expname}.out" 2>&1 &

  PIDS+=("$!")
}

launch_suite() {
  local dataset_name="$1"
  local config="$2"
  local datadir="$3"
  local dataset_type="$4"
  local basedir="$5"
  local gpu0="$6"
  local gpu1="$7"
  local gpu2="$8"
  local gpu3="$9"
  shift 9
  local dataset_extra_args=("$@")

  local common_args=(
    --config "${config}"
    --datadir "${datadir}"
    --dataset_type "${dataset_type}"
    --basedir "${basedir}"
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
    "${dataset_extra_args[@]}"
    "${GLOBAL_EXTRA_ARGS[@]}"
  )

  # 1) Muon + rank-aware WSD
  run_one "${gpu0}" "${basedir}" "${dataset_name}_muon_rank_wsd" \
    "${common_args[@]}" \
    --optimizer aux-muon \
    --train_scheduler rank_wsd \
    --rank_wsd_fullrank_decay_start_frac "${FULLRANK_DECAY_START_FRAC}" \
    --rank_wsd_fullrank_decay_end_frac "${FULLRANK_DECAY_END_FRAC}"

  # 2) Muon + original NeRF exponential decay
  run_one "${gpu1}" "${basedir}" "${dataset_name}_muon_exp_decay" \
    "${common_args[@]}" \
    --optimizer aux-muon \
    --train_scheduler exp_decay

  # 3) Auto-cos-inc + rank-aware WSD
  run_one "${gpu2}" "${basedir}" "${dataset_name}_autocosinc_rank_wsd" \
    "${common_args[@]}" \
    --optimizer aux-sign-auto-cos-inc \
    --train_scheduler rank_wsd \
    "${AUTO_ARGS[@]}"

  # 4) Auto-cos-inc + original NeRF exponential decay
  run_one "${gpu3}" "${basedir}" "${dataset_name}_autocosinc_exp_decay" \
    "${common_args[@]}" \
    --optimizer aux-sign-auto-cos-inc \
    --train_scheduler exp_decay \
    "${AUTO_ARGS[@]}"
}

launch_suite \
  "${DATA1_NAME}" "${DATA1_CONFIG}" "${DATA1_DATADIR}" "${DATA1_DATASET_TYPE}" "${DATA1_BASEDIR}" \
  0 1 2 3 \
  "${DATA1_EXTRA_ARGS[@]}"

launch_suite \
  "${DATA2_NAME}" "${DATA2_CONFIG}" "${DATA2_DATADIR}" "${DATA2_DATASET_TYPE}" "${DATA2_BASEDIR}" \
  4 5 6 7 \
  "${DATA2_EXTRA_ARGS[@]}"

printf '[running] PIDs: %s\n' "${PIDS[*]}"

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[done] all 8 runs finished."
echo "[logs] ${DATA1_BASEDIR}/_stdout"
echo "[logs] ${DATA2_BASEDIR}/_stdout"

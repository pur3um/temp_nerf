#!/bin/bash

SCRIPT="00_run_nerf_ranksched_final.py"
BASEDIR="logs/auto/rankwsd"
OPTIM="aux-sign-auto-cos-inc"
SCHED="rank_wsd"
N_ITERS=100000
MUON_LR="3e-3"
SWF="5"

CUDA_VISIBLE_DEVICES=0 python ${SCRIPT} --basedir ${BASEDIR} --config configs/flower.txt   --expname flower_100k_swf${SWF}_mlr${MUON_LR}   --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=1 python ${SCRIPT} --basedir ${BASEDIR} --config configs/fern.txt     --expname fern_100k_swf${SWF}_mlr${MUON_LR}     --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=2 python ${SCRIPT} --basedir ${BASEDIR} --config configs/orchids.txt  --expname orchids_100k_swf${SWF}_mlr${MUON_LR}  --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=3 python ${SCRIPT} --basedir ${BASEDIR} --config configs/leaves.txt   --expname leaves_100k_swf${SWF}_mlr${MUON_LR}   --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=4 python ${SCRIPT} --basedir ${BASEDIR} --config configs/lego.txt     --expname lego_100k_swf${SWF}_mlr${MUON_LR}     --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=5 python ${SCRIPT} --basedir ${BASEDIR} --config configs/mic.txt     --expname mic_100k_swf${SWF}_mlr${MUON_LR}     --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=6 python ${SCRIPT} --basedir ${BASEDIR} --config configs/materials.txt    --expname materials_100k_swf${SWF}_mlr${MUON_LR}    --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &
CUDA_VISIBLE_DEVICES=7 python ${SCRIPT} --basedir ${BASEDIR} --config configs/drums.txt --expname drums_100k_swf${SWF}_mlr${MUON_LR} --optimizer ${OPTIM} --train_scheduler ${SCHED} --muon_lrate ${MUON_LR} --sched_warmup_frac ${SWF} --N_iters ${N_ITERS} &

wait
echo "All 8 scenes completed."
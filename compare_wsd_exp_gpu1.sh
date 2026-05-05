# synthetic lego drum mic materials
# CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/lego.txt --expname lego_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000;
# CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/drums.txt --expname drums_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000;
# CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/mic.txt --expname mic_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000;
# CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/materials.txt --expname materials_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000;
# wait
#!/bin/bash

GPU=1
SCRIPT="00_run_nerf_ranksched_final.py"
BASEDIR="logs/aux-muon/rankwsd"
OPTIM="aux-muon"
SCHED="rank_wsd"
N_ITERS=100000

CONFIGS=("lego" "drums" "mic" "materials")
MUON_LR_LIST=("4e-3" "2e-3" "3e-3")
SWF_LIST=("5" "5e-1" "5e-2" "5e-3")

for mll in "${MUON_LR_LIST[@]}"; do
    for swf in "${SWF_LIST[@]}"; do
      for scene in "${CONFIGS[@]}"; do
        expname="${scene}_100k_swf${swf}_mlr${mll}"
    
        echo "========================================"
        echo "Running: scene=${scene}, swf=${swf}"
        echo "expname=${expname}"
        echo "========================================"
    
        CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} \
          --basedir ${BASEDIR} \
          --config configs/${scene}.txt \
          --expname ${expname} \
          --optimizer ${OPTIM} \
          --train_scheduler ${SCHED} \
          --muon_lrate ${mll} \
          --sched_warmup_frac ${swf} \
          --N_iters ${N_ITERS}
      done
    done
done
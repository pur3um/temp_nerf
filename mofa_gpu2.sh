#!/bin/bash
# flower fern fortress horns leaves orchids room trex
SCRIPT="run_nerf_mofasgd_safe.py"
BASEDIR="logs/mofa_safe"
N_ITERS=100000
MOFA_LR="3e-3"

CUDA_VISIBLE_DEVICES=0 python ${SCRIPT} --basedir ${BASEDIR} --config configs/flower.txt  --dataset_type llff --datadir ./data/nerf_llff_data/flower --expname flower_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=1 python ${SCRIPT} --basedir ${BASEDIR} --config configs/fern.txt  --dataset_type llff  --datadir ./data/nerf_llff_data/fern  --expname fern_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95  --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=2 python ${SCRIPT} --basedir ${BASEDIR} --config configs/fortress.txt  --dataset_type llff --datadir ./data/nerf_llff_data/fortress --expname fortress_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=3 python ${SCRIPT} --basedir ${BASEDIR} --config configs/horns.txt  --dataset_type llff --datadir ./data/nerf_llff_data/horns --expname horns_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=4 python ${SCRIPT} --basedir ${BASEDIR} --config configs/leaves.txt  --dataset_type llff --datadir ./data/nerf_llff_data/leaves --expname leaves_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=5 python ${SCRIPT} --basedir ${BASEDIR} --config configs/orchids.txt  --dataset_type llff --datadir ./data/nerf_llff_data/orchids --expname orchids_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=6 python ${SCRIPT} --basedir ${BASEDIR} --config configs/room.txt  --dataset_type llff --datadir ./data/nerf_llff_data/room --expname room_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
CUDA_VISIBLE_DEVICES=7 python ${SCRIPT} --basedir ${BASEDIR} --config configs/trex.txt  --dataset_type llff --datadir ./data/nerf_llff_data/trex --expname trex_100k_mflr${MOFA_LR}  --mofa_lrate ${MOFA_LR} --mofa_rank 64 --mofa_beta 0.95 --lrate 5e-4 --N_iters ${N_ITERS} --no_reload --N_importance 128&
wait

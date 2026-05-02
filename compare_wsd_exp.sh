
CUDA_VISIBLE_DEVICES=0 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_200k --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/lego.txt --expname lego_200k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 200000&
wait
CUDA_VISIBLE_DEVICES=0 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_200k --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/flower.txt --expname flower_200k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 200000&
wait
CUDA_VISIBLE_DEVICES=0 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/chair.txt --expname chair_200k --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/chair.txt --expname chair_200k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 200000&
wait
CUDA_VISIBLE_DEVICES=0 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_200k --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --N_iters 200000&
CUDA_VISIBLE_DEVICES=1 python 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/fern.txt --expname fern_200k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 200000&
wait

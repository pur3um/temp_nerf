====================================================================================================
saved_time: 2026-05-04 04:38:21
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: orchids_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0083943252
current_train_psnr: 24.559078
testset_mean_loss: 0.0068193201
testset_mean_psnr: 21.803437
testset_mean_ssim: 0.754069
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0077830818, psnr=21.088484, ssim=0.717602, lpips=unavailable
  image_001: loss=0.0041713524, psnr=23.797231, ssim=0.828005, lpips=unavailable
  image_002: loss=0.0080994805, psnr=20.915428, ssim=0.734412, lpips=unavailable
  image_003: loss=0.0072233658, psnr=21.412604, ssim=0.736255, lpips=unavailable

====================================================================================================
saved_time: 2026-05-05 07:05:47
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-3_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: orchids_100k_swf5e-3_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0079663415
current_train_psnr: 24.856905
testset_mean_loss: 0.0068468865
testset_mean_psnr: 21.780306
testset_mean_ssim: 0.751476
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0078224000, psnr=21.066600, ssim=0.716286, lpips=unavailable
  image_001: loss=0.0042221565, psnr=23.744657, ssim=0.825348, lpips=unavailable
  image_002: loss=0.0079915943, psnr=20.973666, ssim=0.731533, lpips=unavailable
  image_003: loss=0.0073513952, psnr=21.336302, ssim=0.732735, lpips=unavailable

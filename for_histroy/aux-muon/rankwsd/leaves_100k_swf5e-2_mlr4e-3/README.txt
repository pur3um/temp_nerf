====================================================================================================
saved_time: 2026-05-03 23:41:05
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: leaves_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0097153187
current_train_psnr: 23.661167
testset_mean_loss: 0.0057120614
testset_mean_psnr: 22.503674
testset_mean_ssim: 0.801924
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0074829943, psnr=21.259246, ssim=0.749154, lpips=unavailable
  image_001: loss=0.0050488240, psnr=22.968098, ssim=0.826450, lpips=unavailable
  image_002: loss=0.0056354743, psnr=22.490695, ssim=0.810114, lpips=unavailable
  image_003: loss=0.0046809530, psnr=23.296657, ssim=0.821977, lpips=unavailable

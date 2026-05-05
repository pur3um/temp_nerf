====================================================================================================
saved_time: 2026-05-04 07:57:24
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: flower_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0038836563
current_train_psnr: 27.545942
testset_mean_loss: 0.0022634218
testset_mean_psnr: 26.532501
testset_mean_ssim: 0.826811
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0024591994, psnr=26.092062, ssim=0.804699, lpips=unavailable
  image_001: loss=0.0023523546, psnr=26.284972, ssim=0.827715, lpips=unavailable
  image_002: loss=0.0027555132, psnr=25.597975, ssim=0.799340, lpips=unavailable
  image_003: loss=0.0022177936, psnr=26.540789, ssim=0.845679, lpips=unavailable
  image_004: loss=0.0015322483, psnr=28.146708, ssim=0.856623, lpips=unavailable

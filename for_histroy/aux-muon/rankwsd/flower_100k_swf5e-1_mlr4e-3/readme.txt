====================================================================================================
saved_time: 2026-05-03 12:06:33
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: flower_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0022227215
current_train_psnr: 30.214138
testset_mean_loss: 0.0014571087
testset_mean_psnr: 28.548068
testset_mean_ssim: 0.883615
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0014850546, psnr=28.282575, ssim=0.870879, lpips=unavailable
  image_001: loss=0.0014836654, psnr=28.286640, ssim=0.888083, lpips=unavailable
  image_002: loss=0.0020797837, psnr=26.819818, ssim=0.849604, lpips=unavailable
  image_003: loss=0.0014185009, psnr=28.481704, ssim=0.895486, lpips=unavailable
  image_004: loss=0.0008185391, psnr=30.869605, ssim=0.914022, lpips=unavailable

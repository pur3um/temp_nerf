====================================================================================================
saved_time: 2026-05-05 03:47:45
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-3_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: flower_100k_swf5e-3_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0020772833
current_train_psnr: 30.534777
testset_mean_loss: 0.0014249969
testset_mean_psnr: 28.668691
testset_mean_ssim: 0.889075
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0014021794, psnr=28.531964, ssim=0.876000, lpips=unavailable
  image_001: loss=0.0014222596, psnr=28.470211, ssim=0.894455, lpips=unavailable
  image_002: loss=0.0020768621, psnr=26.825923, ssim=0.853624, lpips=unavailable
  image_003: loss=0.0014555483, psnr=28.369733, ssim=0.901606, lpips=unavailable
  image_004: loss=0.0007681352, psnr=31.145623, ssim=0.919692, lpips=unavailable

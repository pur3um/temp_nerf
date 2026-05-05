====================================================================================================
saved_time: 2026-05-03 05:30:00
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: flower_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0038071503
current_train_psnr: 27.934355
testset_mean_loss: 0.0023070462
testset_mean_psnr: 26.444909
testset_mean_ssim: 0.821271
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0025231438, psnr=25.980580, ssim=0.799407, lpips=unavailable
  image_001: loss=0.0023951719, psnr=26.206633, ssim=0.822854, lpips=unavailable
  image_002: loss=0.0028510685, psnr=25.449923, ssim=0.792449, lpips=unavailable
  image_003: loss=0.0021436219, psnr=26.688518, ssim=0.840878, lpips=unavailable
  image_004: loss=0.0016222249, psnr=27.898889, ssim=0.850764, lpips=unavailable

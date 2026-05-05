====================================================================================================
saved_time: 2026-05-03 17:03:32
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: leaves_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0100643262
current_train_psnr: 23.211607
testset_mean_loss: 0.0058888213
testset_mean_psnr: 22.369453
testset_mean_ssim: 0.794669
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0077330000, psnr=21.116520, ssim=0.739249, lpips=unavailable
  image_001: loss=0.0051577613, psnr=22.875388, ssim=0.821388, lpips=unavailable
  image_002: loss=0.0057292050, psnr=22.419056, ssim=0.805568, lpips=unavailable
  image_003: loss=0.0049353191, psnr=23.066848, ssim=0.812473, lpips=unavailable

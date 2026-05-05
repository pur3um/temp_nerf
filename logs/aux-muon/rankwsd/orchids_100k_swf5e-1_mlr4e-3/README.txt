====================================================================================================
saved_time: 2026-05-03 15:24:43
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: orchids_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0087060146
current_train_psnr: 24.481092
testset_mean_loss: 0.0069995753
testset_mean_psnr: 21.704289
testset_mean_ssim: 0.744729
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0080024162, psnr=20.967789, ssim=0.709792, lpips=unavailable
  image_001: loss=0.0041532745, psnr=23.816094, ssim=0.822619, lpips=unavailable
  image_002: loss=0.0082858941, psnr=20.816606, ssim=0.721454, lpips=unavailable
  image_003: loss=0.0075567164, psnr=21.216669, ssim=0.725050, lpips=unavailable

====================================================================================================
saved_time: 2026-05-04 22:49:46
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5e-2_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: fern_100k_swf5e-2_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0037797077
current_train_psnr: 28.274191
testset_mean_loss: 0.0018968664
testset_mean_psnr: 27.260613
testset_mean_ssim: 0.860229
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0022458488, psnr=26.486195, ssim=0.845991, lpips=unavailable
  image_001: loss=0.0018337592, psnr=27.366577, ssim=0.868392, lpips=unavailable
  image_002: loss=0.0016109913, psnr=27.929068, ssim=0.866304, lpips=unavailable

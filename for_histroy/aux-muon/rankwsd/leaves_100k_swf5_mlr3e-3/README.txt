====================================================================================================
saved_time: 2026-05-04 12:55:00
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: leaves_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0152131114
current_train_psnr: 21.404018
testset_mean_loss: 0.0083277684
testset_mean_psnr: 20.843000
testset_mean_ssim: 0.711156
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0105124405, psnr=19.782964, ssim=0.651418, lpips=unavailable
  image_001: loss=0.0072015342, psnr=21.425750, ssim=0.745999, lpips=unavailable
  image_002: loss=0.0080219051, psnr=20.957225, ssim=0.725325, lpips=unavailable
  image_003: loss=0.0075751939, psnr=21.206062, ssim=0.721880, lpips=unavailable

====================================================================================================
saved_time: 2026-05-04 06:17:33
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: leaves_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0096429773
current_train_psnr: 23.653067
testset_mean_loss: 0.0056845333
testset_mean_psnr: 22.521192
testset_mean_ssim: 0.802947
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0073938500, psnr=21.311294, ssim=0.750791, lpips=unavailable
  image_001: loss=0.0050714002, psnr=22.948721, ssim=0.826508, lpips=unavailable
  image_002: loss=0.0056158672, psnr=22.505832, ssim=0.811218, lpips=unavailable
  image_003: loss=0.0046570157, psnr=23.318923, ssim=0.823271, lpips=unavailable

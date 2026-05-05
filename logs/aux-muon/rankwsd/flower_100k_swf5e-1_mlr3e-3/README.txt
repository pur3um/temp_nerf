====================================================================================================
saved_time: 2026-05-04 14:34:04
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-1_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: flower_100k_swf5e-1_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0021109467
current_train_psnr: 30.570887
testset_mean_loss: 0.0014706664
testset_mean_psnr: 28.509978
testset_mean_ssim: 0.882018
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0015004840, psnr=28.237686, ssim=0.867846, lpips=unavailable
  image_001: loss=0.0014998484, psnr=28.239526, ssim=0.886169, lpips=unavailable
  image_002: loss=0.0021067285, psnr=26.763914, ssim=0.848101, lpips=unavailable
  image_003: loss=0.0014213044, psnr=28.473129, ssim=0.894772, lpips=unavailable
  image_004: loss=0.0008249666, psnr=30.835636, ssim=0.913203, lpips=unavailable

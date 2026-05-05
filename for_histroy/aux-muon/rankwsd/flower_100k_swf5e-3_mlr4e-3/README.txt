====================================================================================================
saved_time: 2026-05-04 01:20:19
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: flower_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0020260604
current_train_psnr: 30.681709
testset_mean_loss: 0.0013918842
testset_mean_psnr: 28.768986
testset_mean_ssim: 0.889837
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0013700649, psnr=28.632588, ssim=0.878546, lpips=unavailable
  image_001: loss=0.0014078896, psnr=28.514314, ssim=0.895088, lpips=unavailable
  image_002: loss=0.0020689331, psnr=26.842535, ssim=0.854369, lpips=unavailable
  image_003: loss=0.0013423878, psnr=28.721220, ssim=0.901636, lpips=unavailable
  image_004: loss=0.0007701455, psnr=31.134272, ssim=0.919547, lpips=unavailable

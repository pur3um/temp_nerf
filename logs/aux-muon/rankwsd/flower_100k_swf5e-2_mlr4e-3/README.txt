====================================================================================================
saved_time: 2026-05-03 18:43:06
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: flower_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0020304795
current_train_psnr: 30.692797
testset_mean_loss: 0.0014025600
testset_mean_psnr: 28.730645
testset_mean_ssim: 0.888854
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0014111643, psnr=28.504224, ssim=0.875727, lpips=unavailable
  image_001: loss=0.0014273039, psnr=28.454835, ssim=0.894847, lpips=unavailable
  image_002: loss=0.0020663417, psnr=26.847978, ssim=0.853335, lpips=unavailable
  image_003: loss=0.0013275908, psnr=28.769357, ssim=0.901470, lpips=unavailable
  image_004: loss=0.0007803991, psnr=31.076832, ssim=0.918890, lpips=unavailable

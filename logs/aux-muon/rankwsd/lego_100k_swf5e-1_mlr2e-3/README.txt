====================================================================================================
saved_time: 2026-05-04 22:32:31
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5e-1_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: lego_100k_swf5e-1_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0027901407
current_train_psnr: 32.625748
testset_mean_loss: 0.0007794085
testset_mean_psnr: 31.336120
testset_mean_ssim: 0.962207
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0005807597, psnr=32.360034, ssim=0.966646, lpips=unavailable
  image_001: loss=0.0015591986, psnr=28.070985, ssim=0.952750, lpips=unavailable
  image_002: loss=0.0004483381, psnr=33.483943, ssim=0.970161, lpips=unavailable
  image_003: loss=0.0005493510, psnr=32.601500, ssim=0.968045, lpips=unavailable
  image_004: loss=0.0007686359, psnr=31.142793, ssim=0.962024, lpips=unavailable
  image_005: loss=0.0005429006, psnr=32.652796, ssim=0.963674, lpips=unavailable
  image_006: loss=0.0008390723, psnr=30.762006, ssim=0.962055, lpips=unavailable
  image_007: loss=0.0005036726, psnr=32.978516, ssim=0.961870, lpips=unavailable
  image_008: loss=0.0005871989, psnr=32.312147, ssim=0.968415, lpips=unavailable
  image_009: loss=0.0011380881, psnr=29.438241, ssim=0.963936, lpips=unavailable
  image_010: loss=0.0015691319, psnr=28.043405, ssim=0.949161, lpips=unavailable
  image_011: loss=0.0012079645, psnr=29.179458, ssim=0.951742, lpips=unavailable
  image_012: loss=0.0008026647, psnr=30.954658, ssim=0.963031, lpips=unavailable
  image_013: loss=0.0007405492, psnr=31.304460, ssim=0.963191, lpips=unavailable
  image_014: loss=0.0006068627, psnr=32.169095, ssim=0.963127, lpips=unavailable
  image_015: loss=0.0007146402, psnr=31.459125, ssim=0.964558, lpips=unavailable
  image_016: loss=0.0007686412, psnr=31.142763, ssim=0.972164, lpips=unavailable
  image_017: loss=0.0007270270, psnr=31.384494, ssim=0.964232, lpips=unavailable
  image_018: loss=0.0006982248, psnr=31.560047, ssim=0.962350, lpips=unavailable
  image_019: loss=0.0006036902, psnr=32.191858, ssim=0.958705, lpips=unavailable
  image_020: loss=0.0004339108, psnr=33.625995, ssim=0.964697, lpips=unavailable
  image_021: loss=0.0006360127, psnr=31.965341, ssim=0.963114, lpips=unavailable
  image_022: loss=0.0008590978, psnr=30.659573, ssim=0.959187, lpips=unavailable
  image_023: loss=0.0008530408, psnr=30.690302, ssim=0.957596, lpips=unavailable
  image_024: loss=0.0007465393, psnr=31.269473, ssim=0.958748, lpips=unavailable

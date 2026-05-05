====================================================================================================
saved_time: 2026-05-04 02:13:20
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: mic_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0010960856
current_train_psnr: 37.673553
testset_mean_loss: 0.0004750731
testset_mean_psnr: 33.424513
testset_mean_ssim: 0.979523
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004541508, psnr=33.427998, ssim=0.979255, lpips=unavailable
  image_001: loss=0.0002904455, psnr=35.369353, ssim=0.983770, lpips=unavailable
  image_002: loss=0.0004880891, psnr=33.115008, ssim=0.978173, lpips=unavailable
  image_003: loss=0.0004270355, psnr=33.695359, ssim=0.977370, lpips=unavailable
  image_004: loss=0.0004372437, psnr=33.592763, ssim=0.974884, lpips=unavailable
  image_005: loss=0.0003821428, psnr=34.177742, ssim=0.979317, lpips=unavailable
  image_006: loss=0.0003506835, psnr=34.550846, ssim=0.990479, lpips=unavailable
  image_007: loss=0.0004362071, psnr=33.603071, ssim=0.982785, lpips=unavailable
  image_008: loss=0.0005596852, psnr=32.520561, ssim=0.970364, lpips=unavailable
  image_009: loss=0.0004432487, psnr=33.533524, ssim=0.975510, lpips=unavailable
  image_010: loss=0.0003047962, psnr=35.159904, ssim=0.982177, lpips=unavailable
  image_011: loss=0.0002969271, psnr=35.273501, ssim=0.984666, lpips=unavailable
  image_012: loss=0.0011406937, psnr=29.428309, ssim=0.977106, lpips=unavailable
  image_013: loss=0.0006348221, psnr=31.973479, ssim=0.983866, lpips=unavailable
  image_014: loss=0.0005260403, psnr=32.789809, ssim=0.982339, lpips=unavailable
  image_015: loss=0.0004781552, psnr=33.204310, ssim=0.980501, lpips=unavailable
  image_016: loss=0.0004042404, psnr=33.933602, ssim=0.978242, lpips=unavailable
  image_017: loss=0.0004516672, psnr=33.451814, ssim=0.976146, lpips=unavailable
  image_018: loss=0.0003229659, psnr=34.908432, ssim=0.984904, lpips=unavailable
  image_019: loss=0.0005161878, psnr=32.871922, ssim=0.989184, lpips=unavailable
  image_020: loss=0.0005475212, psnr=32.615990, ssim=0.973496, lpips=unavailable
  image_021: loss=0.0005341794, psnr=32.723128, ssim=0.972278, lpips=unavailable
  image_022: loss=0.0005396656, psnr=32.678752, ssim=0.973658, lpips=unavailable
  image_023: loss=0.0005452439, psnr=32.634091, ssim=0.976041, lpips=unavailable
  image_024: loss=0.0003647902, psnr=34.379567, ssim=0.981562, lpips=unavailable

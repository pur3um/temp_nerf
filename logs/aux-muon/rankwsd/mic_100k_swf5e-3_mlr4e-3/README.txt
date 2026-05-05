====================================================================================================
saved_time: 2026-05-04 10:21:28
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: mic_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0015009879
current_train_psnr: 33.602226
testset_mean_loss: 0.0004721244
testset_mean_psnr: 33.418746
testset_mean_ssim: 0.979456
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004581715, psnr=33.389718, ssim=0.978336, lpips=unavailable
  image_001: loss=0.0002959974, psnr=35.287120, ssim=0.983715, lpips=unavailable
  image_002: loss=0.0005117468, psnr=32.909448, ssim=0.978045, lpips=unavailable
  image_003: loss=0.0004306607, psnr=33.658647, ssim=0.977094, lpips=unavailable
  image_004: loss=0.0004266317, psnr=33.699468, ssim=0.974781, lpips=unavailable
  image_005: loss=0.0003967642, psnr=34.014674, ssim=0.978456, lpips=unavailable
  image_006: loss=0.0003563148, psnr=34.481661, ssim=0.990308, lpips=unavailable
  image_007: loss=0.0004434878, psnr=33.531182, ssim=0.982252, lpips=unavailable
  image_008: loss=0.0005453073, psnr=32.633586, ssim=0.970731, lpips=unavailable
  image_009: loss=0.0004554060, psnr=33.416012, ssim=0.975234, lpips=unavailable
  image_010: loss=0.0003091366, psnr=35.098495, ssim=0.982158, lpips=unavailable
  image_011: loss=0.0002824379, psnr=35.490768, ssim=0.985354, lpips=unavailable
  image_012: loss=0.0009915230, psnr=30.036971, ssim=0.979372, lpips=unavailable
  image_013: loss=0.0006396518, psnr=31.940563, ssim=0.983238, lpips=unavailable
  image_014: loss=0.0005356609, psnr=32.711099, ssim=0.982024, lpips=unavailable
  image_015: loss=0.0004621162, psnr=33.352487, ssim=0.980422, lpips=unavailable
  image_016: loss=0.0004128415, psnr=33.842165, ssim=0.977743, lpips=unavailable
  image_017: loss=0.0004591699, psnr=33.380265, ssim=0.975527, lpips=unavailable
  image_018: loss=0.0003359511, psnr=34.737238, ssim=0.984494, lpips=unavailable
  image_019: loss=0.0005128909, psnr=32.899749, ssim=0.988717, lpips=unavailable
  image_020: loss=0.0005612238, psnr=32.508638, ssim=0.973531, lpips=unavailable
  image_021: loss=0.0005332725, psnr=32.730507, ssim=0.972744, lpips=unavailable
  image_022: loss=0.0005369164, psnr=32.700933, ssim=0.974289, lpips=unavailable
  image_023: loss=0.0005455343, psnr=32.631779, ssim=0.976103, lpips=unavailable
  image_024: loss=0.0003642949, psnr=34.385468, ssim=0.981730, lpips=unavailable

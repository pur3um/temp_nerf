====================================================================================================
saved_time: 2026-05-02 15:13:24
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_200k_swf1e-1 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 1e-1 --N_iters 200000
expname: lego_200k_swf1e-1
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 4 hour 3 min
current_train_loss: 0.0023927486
current_train_psnr: 34.495346
testset_mean_loss: 0.0005847319
testset_mean_psnr: 32.603630
testset_mean_ssim: 0.972479
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004358896, psnr=33.606234, ssim=0.976083, lpips=unavailable
  image_001: loss=0.0010735036, psnr=29.691964, ssim=0.964387, lpips=unavailable
  image_002: loss=0.0003558572, psnr=34.487241, ssim=0.977341, lpips=unavailable
  image_003: loss=0.0004114477, psnr=33.856852, ssim=0.976410, lpips=unavailable
  image_004: loss=0.0005899937, psnr=32.291525, ssim=0.971259, lpips=unavailable
  image_005: loss=0.0003702984, psnr=34.314480, ssim=0.976602, lpips=unavailable
  image_006: loss=0.0005776102, psnr=32.383651, ssim=0.973848, lpips=unavailable
  image_007: loss=0.0003720073, psnr=34.294484, ssim=0.975090, lpips=unavailable
  image_008: loss=0.0004100471, psnr=33.871662, ssim=0.978941, lpips=unavailable
  image_009: loss=0.0009003259, psnr=30.456002, ssim=0.972874, lpips=unavailable
  image_010: loss=0.0012584699, psnr=29.001571, ssim=0.960345, lpips=unavailable
  image_011: loss=0.0009054354, psnr=30.431425, ssim=0.962416, lpips=unavailable
  image_012: loss=0.0006103734, psnr=32.144043, ssim=0.971819, lpips=unavailable
  image_013: loss=0.0005848687, psnr=32.329415, ssim=0.971806, lpips=unavailable
  image_014: loss=0.0004874110, psnr=33.121046, ssim=0.970913, lpips=unavailable
  image_015: loss=0.0005532088, psnr=32.571108, ssim=0.973845, lpips=unavailable
  image_016: loss=0.0005927004, psnr=32.271647, ssim=0.979722, lpips=unavailable
  image_017: loss=0.0005682866, psnr=32.454325, ssim=0.973088, lpips=unavailable
  image_018: loss=0.0005181849, psnr=32.855152, ssim=0.973089, lpips=unavailable
  image_019: loss=0.0004722918, psnr=33.257895, ssim=0.970347, lpips=unavailable
  image_020: loss=0.0002801245, psnr=35.526488, ssim=0.978642, lpips=unavailable
  image_021: loss=0.0004492249, psnr=33.475361, ssim=0.974981, lpips=unavailable
  image_022: loss=0.0006454691, psnr=31.901245, ssim=0.969900, lpips=unavailable
  image_023: loss=0.0006425778, psnr=31.920742, ssim=0.968785, lpips=unavailable
  image_024: loss=0.0005526887, psnr=32.575193, ssim=0.969451, lpips=unavailable

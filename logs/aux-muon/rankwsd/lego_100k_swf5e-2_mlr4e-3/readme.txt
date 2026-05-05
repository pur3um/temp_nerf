====================================================================================================
saved_time: 2026-05-03 22:08:54
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: lego_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0031996814
current_train_psnr: 33.635490
testset_mean_loss: 0.0006854279
testset_mean_psnr: 31.872409
testset_mean_ssim: 0.966924
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0005097161, psnr=32.926716, ssim=0.970231, lpips=unavailable
  image_001: loss=0.0011199261, psnr=29.508106, ssim=0.959733, lpips=unavailable
  image_002: loss=0.0004124886, psnr=33.845880, ssim=0.973276, lpips=unavailable
  image_003: loss=0.0004928893, psnr=33.072505, ssim=0.971485, lpips=unavailable
  image_004: loss=0.0006847805, psnr=31.644486, ssim=0.965626, lpips=unavailable
  image_005: loss=0.0004487920, psnr=33.479548, ssim=0.970448, lpips=unavailable
  image_006: loss=0.0007123644, psnr=31.472977, ssim=0.968706, lpips=unavailable
  image_007: loss=0.0004556901, psnr=33.413303, ssim=0.968185, lpips=unavailable
  image_008: loss=0.0005430796, psnr=32.651365, ssim=0.971852, lpips=unavailable
  image_009: loss=0.0009891535, psnr=30.047363, ssim=0.968569, lpips=unavailable
  image_010: loss=0.0014476945, psnr=28.393230, ssim=0.954327, lpips=unavailable
  image_011: loss=0.0010875742, psnr=29.635411, ssim=0.955650, lpips=unavailable
  image_012: loss=0.0007469628, psnr=31.267010, ssim=0.966036, lpips=unavailable
  image_013: loss=0.0006918626, psnr=31.599801, ssim=0.966360, lpips=unavailable
  image_014: loss=0.0005493732, psnr=32.601325, ssim=0.966978, lpips=unavailable
  image_015: loss=0.0006280732, psnr=32.019896, ssim=0.969615, lpips=unavailable
  image_016: loss=0.0006784734, psnr=31.684671, ssim=0.975523, lpips=unavailable
  image_017: loss=0.0006594200, psnr=31.808378, ssim=0.968326, lpips=unavailable
  image_018: loss=0.0006076121, psnr=32.163735, ssim=0.967294, lpips=unavailable
  image_019: loss=0.0005208376, psnr=32.832976, ssim=0.965695, lpips=unavailable
  image_020: loss=0.0003750418, psnr=34.259202, ssim=0.970463, lpips=unavailable
  image_021: loss=0.0005674243, psnr=32.460920, ssim=0.967600, lpips=unavailable
  image_022: loss=0.0007653689, psnr=31.161292, ssim=0.964453, lpips=unavailable
  image_023: loss=0.0007605632, psnr=31.188646, ssim=0.963318, lpips=unavailable
  image_024: loss=0.0006805353, psnr=31.671493, ssim=0.963353, lpips=unavailable

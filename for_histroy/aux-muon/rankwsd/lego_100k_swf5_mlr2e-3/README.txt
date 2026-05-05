====================================================================================================
saved_time: 2026-05-04 14:25:54
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5 --N_iters 100000
expname: lego_100k_swf5_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0043581720
current_train_psnr: 29.423815
testset_mean_loss: 0.0012922260
testset_mean_psnr: 29.035974
testset_mean_ssim: 0.942732
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0010799826, psnr=29.665832, ssim=0.946537, lpips=unavailable
  image_001: loss=0.0021481116, psnr=26.679431, ssim=0.935729, lpips=unavailable
  image_002: loss=0.0008803423, psnr=30.553484, ssim=0.951359, lpips=unavailable
  image_003: loss=0.0010628096, psnr=29.735445, ssim=0.947689, lpips=unavailable
  image_004: loss=0.0013363305, psnr=28.740861, ssim=0.941623, lpips=unavailable
  image_005: loss=0.0010386832, psnr=29.835168, ssim=0.942064, lpips=unavailable
  image_006: loss=0.0012691851, psnr=28.964750, ssim=0.942847, lpips=unavailable
  image_007: loss=0.0009259603, psnr=30.334076, ssim=0.941092, lpips=unavailable
  image_008: loss=0.0011051840, psnr=29.565654, ssim=0.948704, lpips=unavailable
  image_009: loss=0.0017960991, psnr=27.456697, ssim=0.944655, lpips=unavailable
  image_010: loss=0.0023127731, psnr=26.358669, ssim=0.928003, lpips=unavailable
  image_011: loss=0.0018156359, psnr=27.409712, ssim=0.930553, lpips=unavailable
  image_012: loss=0.0012426201, psnr=29.056616, ssim=0.946980, lpips=unavailable
  image_013: loss=0.0011382323, psnr=29.437691, ssim=0.947909, lpips=unavailable
  image_014: loss=0.0010844641, psnr=29.647848, ssim=0.946164, lpips=unavailable
  image_015: loss=0.0012094799, psnr=29.174013, ssim=0.947027, lpips=unavailable
  image_016: loss=0.0013780259, psnr=28.607426, ssim=0.954615, lpips=unavailable
  image_017: loss=0.0012700036, psnr=28.961950, ssim=0.947822, lpips=unavailable
  image_018: loss=0.0011655579, psnr=29.334661, ssim=0.942573, lpips=unavailable
  image_019: loss=0.0008936746, psnr=30.488205, ssim=0.939821, lpips=unavailable
  image_020: loss=0.0008611522, psnr=30.649200, ssim=0.941415, lpips=unavailable
  image_021: loss=0.0011398037, psnr=29.431699, ssim=0.941209, lpips=unavailable
  image_022: loss=0.0014882185, psnr=28.273333, ssim=0.936455, lpips=unavailable
  image_023: loss=0.0014321354, psnr=28.440159, ssim=0.935650, lpips=unavailable
  image_024: loss=0.0012311850, psnr=29.096766, ssim=0.939799, lpips=unavailable

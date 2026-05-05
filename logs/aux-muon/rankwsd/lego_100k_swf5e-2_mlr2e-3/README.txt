====================================================================================================
saved_time: 2026-05-05 06:40:57
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5e-2_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: lego_100k_swf5e-2_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0028150710
current_train_psnr: 32.977825
testset_mean_loss: 0.0006953060
testset_mean_psnr: 31.842868
testset_mean_ssim: 0.966352
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004985789, psnr=33.022660, ssim=0.970706, lpips=unavailable
  image_001: loss=0.0013355539, psnr=28.743385, ssim=0.957891, lpips=unavailable
  image_002: loss=0.0004093065, psnr=33.879513, ssim=0.973190, lpips=unavailable
  image_003: loss=0.0005015550, psnr=32.996814, ssim=0.971498, lpips=unavailable
  image_004: loss=0.0007181261, psnr=31.437992, ssim=0.965714, lpips=unavailable
  image_005: loss=0.0004545649, psnr=33.424040, ssim=0.969442, lpips=unavailable
  image_006: loss=0.0007327956, psnr=31.350171, ssim=0.966788, lpips=unavailable
  image_007: loss=0.0004530616, psnr=33.438427, ssim=0.967178, lpips=unavailable
  image_008: loss=0.0005187050, psnr=32.850795, ssim=0.972136, lpips=unavailable
  image_009: loss=0.0010206416, psnr=29.911267, ssim=0.967883, lpips=unavailable
  image_010: loss=0.0014608804, psnr=28.353853, ssim=0.953697, lpips=unavailable
  image_011: loss=0.0011133191, psnr=29.533803, ssim=0.954570, lpips=unavailable
  image_012: loss=0.0007131189, psnr=31.468380, ssim=0.966561, lpips=unavailable
  image_013: loss=0.0006626668, psnr=31.787047, ssim=0.966868, lpips=unavailable
  image_014: loss=0.0005389137, psnr=32.684807, ssim=0.967001, lpips=unavailable
  image_015: loss=0.0006473585, psnr=31.888551, ssim=0.968277, lpips=unavailable
  image_016: loss=0.0006655600, psnr=31.768127, ssim=0.976179, lpips=unavailable
  image_017: loss=0.0006384540, psnr=31.948703, ssim=0.968133, lpips=unavailable
  image_018: loss=0.0006175282, psnr=32.093432, ssim=0.966018, lpips=unavailable
  image_019: loss=0.0005418601, psnr=32.661128, ssim=0.963356, lpips=unavailable
  image_020: loss=0.0003765039, psnr=34.242304, ssim=0.969774, lpips=unavailable
  image_021: loss=0.0005523122, psnr=32.578153, ssim=0.967540, lpips=unavailable
  image_022: loss=0.0007818855, psnr=31.068568, ssim=0.963258, lpips=unavailable
  image_023: loss=0.0007657930, psnr=31.158885, ssim=0.961936, lpips=unavailable
  image_024: loss=0.0006636058, psnr=31.780898, ssim=0.963196, lpips=unavailable

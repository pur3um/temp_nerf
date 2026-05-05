====================================================================================================
saved_time: 2026-05-04 20:30:53
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5 --N_iters 100000
expname: materials_100k_swf5_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0052194521
current_train_psnr: 27.216578
testset_mean_loss: 0.0018270723
testset_mean_psnr: 27.489239
testset_mean_ssim: 0.942618
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0028414587, psnr=25.464586, ssim=0.925279, lpips=unavailable
  image_001: loss=0.0030755040, psnr=25.120837, ssim=0.922980, lpips=unavailable
  image_002: loss=0.0025287932, psnr=25.970867, ssim=0.930393, lpips=unavailable
  image_003: loss=0.0020999957, psnr=26.777816, ssim=0.937163, lpips=unavailable
  image_004: loss=0.0019336283, psnr=27.136270, ssim=0.940807, lpips=unavailable
  image_005: loss=0.0018808413, psnr=27.256478, ssim=0.936266, lpips=unavailable
  image_006: loss=0.0014210535, psnr=28.473895, ssim=0.944547, lpips=unavailable
  image_007: loss=0.0018839424, psnr=27.249324, ssim=0.928873, lpips=unavailable
  image_008: loss=0.0017501864, psnr=27.569157, ssim=0.933634, lpips=unavailable
  image_009: loss=0.0014681844, psnr=28.332194, ssim=0.953032, lpips=unavailable
  image_010: loss=0.0012619472, psnr=28.989588, ssim=0.953970, lpips=unavailable
  image_011: loss=0.0013427497, psnr=28.720049, ssim=0.956826, lpips=unavailable
  image_012: loss=0.0016414426, psnr=27.847743, ssim=0.955117, lpips=unavailable
  image_013: loss=0.0014754712, psnr=28.310692, ssim=0.959012, lpips=unavailable
  image_014: loss=0.0015444106, psnr=28.112372, ssim=0.955669, lpips=unavailable
  image_015: loss=0.0016840413, psnr=27.736472, ssim=0.945170, lpips=unavailable
  image_016: loss=0.0014366130, psnr=28.426602, ssim=0.956902, lpips=unavailable
  image_017: loss=0.0018557580, psnr=27.314786, ssim=0.936393, lpips=unavailable
  image_018: loss=0.0016533255, psnr=27.816416, ssim=0.939065, lpips=unavailable
  image_019: loss=0.0015313987, psnr=28.149117, ssim=0.939897, lpips=unavailable
  image_020: loss=0.0019931484, psnr=27.004604, ssim=0.933087, lpips=unavailable
  image_021: loss=0.0016468067, psnr=27.833573, ssim=0.947029, lpips=unavailable
  image_022: loss=0.0017032721, psnr=27.687159, ssim=0.947369, lpips=unavailable
  image_023: loss=0.0019917365, psnr=27.007681, ssim=0.943043, lpips=unavailable
  image_024: loss=0.0020310991, psnr=26.922689, ssim=0.943938, lpips=unavailable

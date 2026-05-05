====================================================================================================
saved_time: 2026-05-04 12:23:40
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: materials_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0032891622
current_train_psnr: 30.014420
testset_mean_loss: 0.0010462751
testset_mean_psnr: 29.984813
testset_mean_ssim: 0.962398
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017192211, psnr=27.646682, ssim=0.950372, lpips=unavailable
  image_001: loss=0.0020281679, psnr=26.928961, ssim=0.945213, lpips=unavailable
  image_002: loss=0.0015715976, psnr=28.036586, ssim=0.953870, lpips=unavailable
  image_003: loss=0.0011255882, psnr=29.486204, ssim=0.961982, lpips=unavailable
  image_004: loss=0.0010385208, psnr=29.835847, ssim=0.964420, lpips=unavailable
  image_005: loss=0.0009623407, psnr=30.166711, ssim=0.961650, lpips=unavailable
  image_006: loss=0.0006329579, psnr=31.986251, ssim=0.968238, lpips=unavailable
  image_007: loss=0.0012578899, psnr=29.003573, ssim=0.945277, lpips=unavailable
  image_008: loss=0.0011438591, psnr=29.416274, ssim=0.952546, lpips=unavailable
  image_009: loss=0.0008199178, psnr=30.862296, ssim=0.968149, lpips=unavailable
  image_010: loss=0.0007463804, psnr=31.270397, ssim=0.967446, lpips=unavailable
  image_011: loss=0.0008118147, psnr=30.905430, ssim=0.970316, lpips=unavailable
  image_012: loss=0.0011884876, psnr=29.250053, ssim=0.964950, lpips=unavailable
  image_013: loss=0.0009071471, psnr=30.423222, ssim=0.971747, lpips=unavailable
  image_014: loss=0.0008647732, psnr=30.630977, ssim=0.971331, lpips=unavailable
  image_015: loss=0.0011882026, psnr=29.251094, ssim=0.959452, lpips=unavailable
  image_016: loss=0.0009159631, psnr=30.381220, ssim=0.971026, lpips=unavailable
  image_017: loss=0.0011786072, psnr=29.286309, ssim=0.958413, lpips=unavailable
  image_018: loss=0.0009400438, psnr=30.268519, ssim=0.959856, lpips=unavailable
  image_019: loss=0.0008216448, psnr=30.853158, ssim=0.959737, lpips=unavailable
  image_020: loss=0.0011817808, psnr=29.274630, ssim=0.951658, lpips=unavailable
  image_021: loss=0.0007621774, psnr=31.179439, ssim=0.969761, lpips=unavailable
  image_022: loss=0.0006546779, psnr=31.839722, ssim=0.972326, lpips=unavailable
  image_023: loss=0.0008525598, psnr=30.692751, ssim=0.969475, lpips=unavailable
  image_024: loss=0.0008425568, psnr=30.744007, ssim=0.970748, lpips=unavailable

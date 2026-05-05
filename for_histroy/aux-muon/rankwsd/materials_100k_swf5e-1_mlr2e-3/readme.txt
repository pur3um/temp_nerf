====================================================================================================
saved_time: 2026-05-05 04:38:55
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5e-1_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: materials_100k_swf5e-1_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0035013831
current_train_psnr: 29.570082
testset_mean_loss: 0.0011369076
testset_mean_psnr: 29.614567
testset_mean_ssim: 0.958995
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018540396, psnr=27.318810, ssim=0.946630, lpips=unavailable
  image_001: loss=0.0021586125, psnr=26.658253, ssim=0.940986, lpips=unavailable
  image_002: loss=0.0016750509, psnr=27.759720, ssim=0.949625, lpips=unavailable
  image_003: loss=0.0012629125, psnr=28.986267, ssim=0.957874, lpips=unavailable
  image_004: loss=0.0011783037, psnr=29.287427, ssim=0.960046, lpips=unavailable
  image_005: loss=0.0010656901, psnr=29.723690, ssim=0.957253, lpips=unavailable
  image_006: loss=0.0006955154, psnr=31.576932, ssim=0.965016, lpips=unavailable
  image_007: loss=0.0013863111, psnr=28.581393, ssim=0.939505, lpips=unavailable
  image_008: loss=0.0012448304, psnr=29.048898, ssim=0.947427, lpips=unavailable
  image_009: loss=0.0009278180, psnr=30.325372, ssim=0.964707, lpips=unavailable
  image_010: loss=0.0008310017, psnr=30.803981, ssim=0.965374, lpips=unavailable
  image_011: loss=0.0008767933, psnr=30.571027, ssim=0.968753, lpips=unavailable
  image_012: loss=0.0012263295, psnr=29.113928, ssim=0.964709, lpips=unavailable
  image_013: loss=0.0009678166, psnr=30.142069, ssim=0.969685, lpips=unavailable
  image_014: loss=0.0009420575, psnr=30.259225, ssim=0.968801, lpips=unavailable
  image_015: loss=0.0012680246, psnr=28.968723, ssim=0.955965, lpips=unavailable
  image_016: loss=0.0009720021, psnr=30.123328, ssim=0.968378, lpips=unavailable
  image_017: loss=0.0012496330, psnr=29.032175, ssim=0.954571, lpips=unavailable
  image_018: loss=0.0009970276, psnr=30.012928, ssim=0.956825, lpips=unavailable
  image_019: loss=0.0009020512, psnr=30.447688, ssim=0.956124, lpips=unavailable
  image_020: loss=0.0013363047, psnr=28.740945, ssim=0.945725, lpips=unavailable
  image_021: loss=0.0008528474, psnr=30.691286, ssim=0.966463, lpips=unavailable
  image_022: loss=0.0007207310, psnr=31.422267, ssim=0.970410, lpips=unavailable
  image_023: loss=0.0009289511, psnr=30.320071, ssim=0.965753, lpips=unavailable
  image_024: loss=0.0009020354, psnr=30.447764, ssim=0.968279, lpips=unavailable

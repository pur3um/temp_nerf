====================================================================================================
saved_time: 2026-05-03 18:05:00
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: mic_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0012421443
current_train_psnr: 36.964813
testset_mean_loss: 0.0004932491
testset_mean_psnr: 33.266006
testset_mean_ssim: 0.978851
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004797849, psnr=33.189534, ssim=0.977697, lpips=unavailable
  image_001: loss=0.0003045133, psnr=35.163935, ssim=0.983290, lpips=unavailable
  image_002: loss=0.0005318076, psnr=32.742454, ssim=0.977294, lpips=unavailable
  image_003: loss=0.0004483528, psnr=33.483800, ssim=0.976618, lpips=unavailable
  image_004: loss=0.0004361823, psnr=33.603319, ssim=0.974409, lpips=unavailable
  image_005: loss=0.0003897424, psnr=34.092222, ssim=0.979094, lpips=unavailable
  image_006: loss=0.0003695890, psnr=34.322809, ssim=0.990073, lpips=unavailable
  image_007: loss=0.0004565251, psnr=33.405352, ssim=0.982110, lpips=unavailable
  image_008: loss=0.0005571500, psnr=32.540278, ssim=0.970300, lpips=unavailable
  image_009: loss=0.0004699507, psnr=33.279476, ssim=0.974224, lpips=unavailable
  image_010: loss=0.0003187802, psnr=34.965085, ssim=0.981371, lpips=unavailable
  image_011: loss=0.0003125889, psnr=35.050263, ssim=0.984111, lpips=unavailable
  image_012: loss=0.0012197325, psnr=29.137354, ssim=0.977032, lpips=unavailable
  image_013: loss=0.0006660217, psnr=31.765115, ssim=0.982824, lpips=unavailable
  image_014: loss=0.0005388311, psnr=32.685473, ssim=0.981846, lpips=unavailable
  image_015: loss=0.0004827446, psnr=33.162825, ssim=0.980118, lpips=unavailable
  image_016: loss=0.0004243891, psnr=33.722357, ssim=0.977695, lpips=unavailable
  image_017: loss=0.0004380964, psnr=33.584303, ssim=0.976063, lpips=unavailable
  image_018: loss=0.0003369272, psnr=34.724638, ssim=0.984475, lpips=unavailable
  image_019: loss=0.0005156646, psnr=32.876326, ssim=0.988513, lpips=unavailable
  image_020: loss=0.0005640291, psnr=32.486984, ssim=0.972606, lpips=unavailable
  image_021: loss=0.0005568364, psnr=32.542723, ssim=0.971474, lpips=unavailable
  image_022: loss=0.0005620354, psnr=32.502362, ssim=0.972849, lpips=unavailable
  image_023: loss=0.0005669354, psnr=32.464663, ssim=0.974747, lpips=unavailable
  image_024: loss=0.0003840161, psnr=34.156505, ssim=0.980435, lpips=unavailable

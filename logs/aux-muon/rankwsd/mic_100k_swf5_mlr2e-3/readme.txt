====================================================================================================
saved_time: 2026-05-04 18:29:10
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5 --N_iters 100000
expname: mic_100k_swf5_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0024395334
current_train_psnr: 33.020603
testset_mean_loss: 0.0008231830
testset_mean_psnr: 30.896863
testset_mean_ssim: 0.970946
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0007123637, psnr=31.472981, ssim=0.972294, lpips=unavailable
  image_001: loss=0.0006663085, psnr=31.763246, ssim=0.974194, lpips=unavailable
  image_002: loss=0.0009479061, psnr=30.232347, ssim=0.967555, lpips=unavailable
  image_003: loss=0.0008458781, psnr=30.726922, ssim=0.966924, lpips=unavailable
  image_004: loss=0.0007863754, psnr=31.043700, ssim=0.966881, lpips=unavailable
  image_005: loss=0.0007874412, psnr=31.037818, ssim=0.970138, lpips=unavailable
  image_006: loss=0.0007749299, psnr=31.107375, ssim=0.980486, lpips=unavailable
  image_007: loss=0.0007936252, psnr=31.003845, ssim=0.972848, lpips=unavailable
  image_008: loss=0.0008323266, psnr=30.797062, ssim=0.963983, lpips=unavailable
  image_009: loss=0.0007795520, psnr=31.081548, ssim=0.967573, lpips=unavailable
  image_010: loss=0.0006107861, psnr=32.141108, ssim=0.974192, lpips=unavailable
  image_011: loss=0.0006688304, psnr=31.746839, ssim=0.976156, lpips=unavailable
  image_012: loss=0.0012641198, psnr=28.982117, ssim=0.973579, lpips=unavailable
  image_013: loss=0.0010368337, psnr=29.842908, ssim=0.977325, lpips=unavailable
  image_014: loss=0.0008716087, psnr=30.596784, ssim=0.974873, lpips=unavailable
  image_015: loss=0.0007343123, psnr=31.341191, ssim=0.973375, lpips=unavailable
  image_016: loss=0.0007930009, psnr=31.007263, ssim=0.970060, lpips=unavailable
  image_017: loss=0.0007841068, psnr=31.056247, ssim=0.969761, lpips=unavailable
  image_018: loss=0.0007041297, psnr=31.523473, ssim=0.974990, lpips=unavailable
  image_019: loss=0.0009080105, psnr=30.419091, ssim=0.979700, lpips=unavailable
  image_020: loss=0.0008166929, psnr=30.879412, ssim=0.964857, lpips=unavailable
  image_021: loss=0.0008762146, psnr=30.573894, ssim=0.962573, lpips=unavailable
  image_022: loss=0.0008794387, psnr=30.557944, ssim=0.964265, lpips=unavailable
  image_023: loss=0.0009805219, psnr=30.085427, ssim=0.964145, lpips=unavailable
  image_024: loss=0.0007242622, psnr=31.401041, ssim=0.970930, lpips=unavailable

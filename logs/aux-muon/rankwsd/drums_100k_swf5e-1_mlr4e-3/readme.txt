====================================================================================================
saved_time: 2026-05-03 16:02:43
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: drums_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0053185513
current_train_psnr: 27.407927
testset_mean_loss: 0.0029872369
testset_mean_psnr: 25.507173
testset_mean_ssim: 0.929447
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018172297, psnr=27.405902, ssim=0.931030, lpips=unavailable
  image_001: loss=0.0022552530, psnr=26.468047, ssim=0.927297, lpips=unavailable
  image_002: loss=0.0014665365, psnr=28.337071, ssim=0.945800, lpips=unavailable
  image_003: loss=0.0022263511, psnr=26.524063, ssim=0.939894, lpips=unavailable
  image_004: loss=0.0019362275, psnr=27.130436, ssim=0.936850, lpips=unavailable
  image_005: loss=0.0018794097, psnr=27.259785, ssim=0.940975, lpips=unavailable
  image_006: loss=0.0049804086, psnr=23.027350, ssim=0.904431, lpips=unavailable
  image_007: loss=0.0036329688, psnr=24.397383, ssim=0.919347, lpips=unavailable
  image_008: loss=0.0030799222, psnr=25.114602, ssim=0.931128, lpips=unavailable
  image_009: loss=0.0036461605, psnr=24.381642, ssim=0.931067, lpips=unavailable
  image_010: loss=0.0034871788, psnr=24.575258, ssim=0.934541, lpips=unavailable
  image_011: loss=0.0044120518, psnr=23.553594, ssim=0.915538, lpips=unavailable
  image_012: loss=0.0037850484, psnr=24.219286, ssim=0.928545, lpips=unavailable
  image_013: loss=0.0027518517, psnr=25.603750, ssim=0.934415, lpips=unavailable
  image_014: loss=0.0026166399, psnr=25.822560, ssim=0.937867, lpips=unavailable
  image_015: loss=0.0039361818, psnr=24.049248, ssim=0.932479, lpips=unavailable
  image_016: loss=0.0020659480, psnr=26.848806, ssim=0.950520, lpips=unavailable
  image_017: loss=0.0029373013, psnr=25.320515, ssim=0.931653, lpips=unavailable
  image_018: loss=0.0031392637, psnr=25.031722, ssim=0.928731, lpips=unavailable
  image_019: loss=0.0053458852, psnr=22.719804, ssim=0.891884, lpips=unavailable
  image_020: loss=0.0034980699, psnr=24.561715, ssim=0.915918, lpips=unavailable
  image_021: loss=0.0036517926, psnr=24.374939, ssim=0.915189, lpips=unavailable
  image_022: loss=0.0025715742, psnr=25.898009, ssim=0.931541, lpips=unavailable
  image_023: loss=0.0020000322, psnr=26.989630, ssim=0.936417, lpips=unavailable
  image_024: loss=0.0015616351, psnr=28.064204, ssim=0.943115, lpips=unavailable

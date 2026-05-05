====================================================================================================
saved_time: 2026-05-04 00:10:46
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: drums_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0053061559
current_train_psnr: 27.418659
testset_mean_loss: 0.0028815336
testset_mean_psnr: 25.650258
testset_mean_ssim: 0.931334
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017662380, psnr=27.529507, ssim=0.933538, lpips=unavailable
  image_001: loss=0.0020340981, psnr=26.916281, ssim=0.930488, lpips=unavailable
  image_002: loss=0.0013851408, psnr=28.585060, ssim=0.947574, lpips=unavailable
  image_003: loss=0.0022082026, psnr=26.559611, ssim=0.939106, lpips=unavailable
  image_004: loss=0.0019391661, psnr=27.123850, ssim=0.936426, lpips=unavailable
  image_005: loss=0.0018369083, psnr=27.359125, ssim=0.942982, lpips=unavailable
  image_006: loss=0.0044958950, psnr=23.471838, ssim=0.908678, lpips=unavailable
  image_007: loss=0.0035709704, psnr=24.472137, ssim=0.921051, lpips=unavailable
  image_008: loss=0.0030033013, psnr=25.224011, ssim=0.932683, lpips=unavailable
  image_009: loss=0.0037162784, psnr=24.298918, ssim=0.931233, lpips=unavailable
  image_010: loss=0.0033284037, psnr=24.777640, ssim=0.937047, lpips=unavailable
  image_011: loss=0.0042641698, psnr=23.701655, ssim=0.918155, lpips=unavailable
  image_012: loss=0.0035612311, psnr=24.483998, ssim=0.930354, lpips=unavailable
  image_013: loss=0.0027663342, psnr=25.580953, ssim=0.935359, lpips=unavailable
  image_014: loss=0.0025183375, psnr=25.988861, ssim=0.940284, lpips=unavailable
  image_015: loss=0.0037811084, psnr=24.223809, ssim=0.932619, lpips=unavailable
  image_016: loss=0.0019616683, psnr=27.073744, ssim=0.952731, lpips=unavailable
  image_017: loss=0.0026935325, psnr=25.696778, ssim=0.933857, lpips=unavailable
  image_018: loss=0.0031434272, psnr=25.025966, ssim=0.930377, lpips=unavailable
  image_019: loss=0.0048043677, psnr=23.183638, ssim=0.900288, lpips=unavailable
  image_020: loss=0.0035452514, psnr=24.503529, ssim=0.917793, lpips=unavailable
  image_021: loss=0.0036272740, psnr=24.404196, ssim=0.914964, lpips=unavailable
  image_022: loss=0.0025619157, psnr=25.914352, ssim=0.933446, lpips=unavailable
  image_023: loss=0.0020005112, psnr=26.988590, ssim=0.937509, lpips=unavailable
  image_024: loss=0.0015246092, psnr=28.168414, ssim=0.944799, lpips=unavailable

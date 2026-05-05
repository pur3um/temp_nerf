====================================================================================================
saved_time: 2026-05-03 11:57:49
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: materials_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0057589309
current_train_psnr: 26.446283
testset_mean_loss: 0.0019669401
testset_mean_psnr: 27.189514
testset_mean_ssim: 0.941297
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0031138575, psnr=25.067013, ssim=0.924502, lpips=unavailable
  image_001: loss=0.0033930761, psnr=24.694064, ssim=0.919200, lpips=unavailable
  image_002: loss=0.0028318029, psnr=25.479370, ssim=0.927625, lpips=unavailable
  image_003: loss=0.0024079385, psnr=26.183546, ssim=0.935168, lpips=unavailable
  image_004: loss=0.0019620811, psnr=27.072830, ssim=0.942051, lpips=unavailable
  image_005: loss=0.0018680628, psnr=27.286085, ssim=0.938363, lpips=unavailable
  image_006: loss=0.0015675245, psnr=28.047856, ssim=0.942026, lpips=unavailable
  image_007: loss=0.0019956960, psnr=26.999056, ssim=0.926007, lpips=unavailable
  image_008: loss=0.0018932567, psnr=27.227905, ssim=0.931107, lpips=unavailable
  image_009: loss=0.0014920854, psnr=28.262063, ssim=0.952220, lpips=unavailable
  image_010: loss=0.0012331731, psnr=29.089759, ssim=0.954658, lpips=unavailable
  image_011: loss=0.0014153579, psnr=28.491337, ssim=0.956835, lpips=unavailable
  image_012: loss=0.0018033653, psnr=27.439163, ssim=0.952855, lpips=unavailable
  image_013: loss=0.0014770927, psnr=28.305922, ssim=0.958864, lpips=unavailable
  image_014: loss=0.0016329235, psnr=27.870341, ssim=0.953862, lpips=unavailable
  image_015: loss=0.0018437154, psnr=27.343061, ssim=0.943493, lpips=unavailable
  image_016: loss=0.0015663096, psnr=28.051224, ssim=0.955402, lpips=unavailable
  image_017: loss=0.0019019005, psnr=27.208122, ssim=0.936910, lpips=unavailable
  image_018: loss=0.0016847909, psnr=27.734540, ssim=0.939823, lpips=unavailable
  image_019: loss=0.0016822632, psnr=27.741060, ssim=0.937316, lpips=unavailable
  image_020: loss=0.0022354759, psnr=26.506300, ssim=0.928703, lpips=unavailable
  image_021: loss=0.0019059515, psnr=27.198881, ssim=0.944493, lpips=unavailable
  image_022: loss=0.0019311772, psnr=27.141778, ssim=0.944769, lpips=unavailable
  image_023: loss=0.0020400393, psnr=26.903614, ssim=0.943510, lpips=unavailable
  image_024: loss=0.0022945860, psnr=26.392956, ssim=0.942650, lpips=unavailable

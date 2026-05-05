====================================================================================================
saved_time: 2026-05-03 20:06:58
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: materials_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0036685900
current_train_psnr: 29.183113
testset_mean_loss: 0.0011018187
testset_mean_psnr: 29.748409
testset_mean_ssim: 0.960299
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018051855, psnr=27.434781, ssim=0.948040, lpips=unavailable
  image_001: loss=0.0021207535, psnr=26.735098, ssim=0.943321, lpips=unavailable
  image_002: loss=0.0016576994, psnr=27.804942, ssim=0.951712, lpips=unavailable
  image_003: loss=0.0012039499, psnr=29.193915, ssim=0.959116, lpips=unavailable
  image_004: loss=0.0011081133, psnr=29.554158, ssim=0.962296, lpips=unavailable
  image_005: loss=0.0010255717, psnr=29.890339, ssim=0.959199, lpips=unavailable
  image_006: loss=0.0006663374, psnr=31.763057, ssim=0.966476, lpips=unavailable
  image_007: loss=0.0012979057, psnr=28.867568, ssim=0.944056, lpips=unavailable
  image_008: loss=0.0011812246, psnr=29.276675, ssim=0.949046, lpips=unavailable
  image_009: loss=0.0008821563, psnr=30.544544, ssim=0.965716, lpips=unavailable
  image_010: loss=0.0008268924, psnr=30.825510, ssim=0.964555, lpips=unavailable
  image_011: loss=0.0008972039, psnr=30.471088, ssim=0.967091, lpips=unavailable
  image_012: loss=0.0012308522, psnr=29.097941, ssim=0.963874, lpips=unavailable
  image_013: loss=0.0009666370, psnr=30.147365, ssim=0.969783, lpips=unavailable
  image_014: loss=0.0008969691, psnr=30.472225, ssim=0.969771, lpips=unavailable
  image_015: loss=0.0012272260, psnr=29.110754, ssim=0.956910, lpips=unavailable
  image_016: loss=0.0009260187, psnr=30.333802, ssim=0.969967, lpips=unavailable
  image_017: loss=0.0012228623, psnr=29.126224, ssim=0.956422, lpips=unavailable
  image_018: loss=0.0009920283, psnr=30.034759, ssim=0.958041, lpips=unavailable
  image_019: loss=0.0008874492, psnr=30.518564, ssim=0.957694, lpips=unavailable
  image_020: loss=0.0012020058, psnr=29.200934, ssim=0.950157, lpips=unavailable
  image_021: loss=0.0008193034, psnr=30.865552, ssim=0.967560, lpips=unavailable
  image_022: loss=0.0007144414, psnr=31.460334, ssim=0.970403, lpips=unavailable
  image_023: loss=0.0009022835, psnr=30.446569, ssim=0.967125, lpips=unavailable
  image_024: loss=0.0008843967, psnr=30.533529, ssim=0.969134, lpips=unavailable

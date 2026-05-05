====================================================================================================
saved_time: 2026-05-04 04:15:41
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/materials.txt --expname materials_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: materials_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0035611098
current_train_psnr: 29.240938
testset_mean_loss: 0.0010708523
testset_mean_psnr: 29.888822
testset_mean_ssim: 0.961270
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017689993, psnr=27.522723, ssim=0.948984, lpips=unavailable
  image_001: loss=0.0021005892, psnr=26.776588, ssim=0.942317, lpips=unavailable
  image_002: loss=0.0015682477, psnr=28.045853, ssim=0.952504, lpips=unavailable
  image_003: loss=0.0011400094, psnr=29.430915, ssim=0.960328, lpips=unavailable
  image_004: loss=0.0010424525, psnr=29.819437, ssim=0.963634, lpips=unavailable
  image_005: loss=0.0009826520, psnr=30.076002, ssim=0.960769, lpips=unavailable
  image_006: loss=0.0006446624, psnr=31.906676, ssim=0.967564, lpips=unavailable
  image_007: loss=0.0013486702, psnr=28.700942, ssim=0.943173, lpips=unavailable
  image_008: loss=0.0011703584, psnr=29.316811, ssim=0.949312, lpips=unavailable
  image_009: loss=0.0009049645, psnr=30.433684, ssim=0.965363, lpips=unavailable
  image_010: loss=0.0007707309, psnr=31.130972, ssim=0.965870, lpips=unavailable
  image_011: loss=0.0008301672, psnr=30.808344, ssim=0.969899, lpips=unavailable
  image_012: loss=0.0012090374, psnr=29.175602, ssim=0.965085, lpips=unavailable
  image_013: loss=0.0009188203, psnr=30.367694, ssim=0.971546, lpips=unavailable
  image_014: loss=0.0008634318, psnr=30.637719, ssim=0.971236, lpips=unavailable
  image_015: loss=0.0012424047, psnr=29.057369, ssim=0.958170, lpips=unavailable
  image_016: loss=0.0008954029, psnr=30.479814, ssim=0.971087, lpips=unavailable
  image_017: loss=0.0011972330, psnr=29.218213, ssim=0.956893, lpips=unavailable
  image_018: loss=0.0009350555, psnr=30.291626, ssim=0.959883, lpips=unavailable
  image_019: loss=0.0008504533, psnr=30.703495, ssim=0.959233, lpips=unavailable
  image_020: loss=0.0012377097, psnr=29.073812, ssim=0.950230, lpips=unavailable
  image_021: loss=0.0007935198, psnr=31.004422, ssim=0.968699, lpips=unavailable
  image_022: loss=0.0006574016, psnr=31.821692, ssim=0.971805, lpips=unavailable
  image_023: loss=0.0008503972, psnr=30.703781, ssim=0.968176, lpips=unavailable
  image_024: loss=0.0008479357, psnr=30.716370, ssim=0.969984, lpips=unavailable

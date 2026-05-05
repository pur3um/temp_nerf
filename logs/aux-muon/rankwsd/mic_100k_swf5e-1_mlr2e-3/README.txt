====================================================================================================
saved_time: 2026-05-05 02:36:37
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5e-1_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: mic_100k_swf5e-1_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0012773772
current_train_psnr: 37.119633
testset_mean_loss: 0.0004975489
testset_mean_psnr: 33.216636
testset_mean_ssim: 0.978462
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004561612, psnr=33.408815, ssim=0.978714, lpips=unavailable
  image_001: loss=0.0003029921, psnr=35.185686, ssim=0.982938, lpips=unavailable
  image_002: loss=0.0005613558, psnr=32.507617, ssim=0.975553, lpips=unavailable
  image_003: loss=0.0004434214, psnr=33.531833, ssim=0.976108, lpips=unavailable
  image_004: loss=0.0004542460, psnr=33.427088, ssim=0.973274, lpips=unavailable
  image_005: loss=0.0003945109, psnr=34.039409, ssim=0.978525, lpips=unavailable
  image_006: loss=0.0003828963, psnr=34.169187, ssim=0.989085, lpips=unavailable
  image_007: loss=0.0004555166, psnr=33.414957, ssim=0.981310, lpips=unavailable
  image_008: loss=0.0005498211, psnr=32.597785, ssim=0.970699, lpips=unavailable
  image_009: loss=0.0004885885, psnr=33.110566, ssim=0.974907, lpips=unavailable
  image_010: loss=0.0003121656, psnr=35.056149, ssim=0.981638, lpips=unavailable
  image_011: loss=0.0003069573, psnr=35.129219, ssim=0.984528, lpips=unavailable
  image_012: loss=0.0011533633, psnr=29.380338, ssim=0.977315, lpips=unavailable
  image_013: loss=0.0006835322, psnr=31.652410, ssim=0.983301, lpips=unavailable
  image_014: loss=0.0005451738, psnr=32.634649, ssim=0.981386, lpips=unavailable
  image_015: loss=0.0005005884, psnr=33.005191, ssim=0.979361, lpips=unavailable
  image_016: loss=0.0004221718, psnr=33.745107, ssim=0.977336, lpips=unavailable
  image_017: loss=0.0004654474, psnr=33.321293, ssim=0.975584, lpips=unavailable
  image_018: loss=0.0003548201, psnr=34.499916, ssim=0.983589, lpips=unavailable
  image_019: loss=0.0005600129, psnr=32.518019, ssim=0.987056, lpips=unavailable
  image_020: loss=0.0005692149, psnr=32.447237, ssim=0.972056, lpips=unavailable
  image_021: loss=0.0005638595, psnr=32.488290, ssim=0.970548, lpips=unavailable
  image_022: loss=0.0005493985, psnr=32.601125, ssim=0.972749, lpips=unavailable
  image_023: loss=0.0005812083, psnr=32.356681, ssim=0.974043, lpips=unavailable
  image_024: loss=0.0003812994, psnr=34.187338, ssim=0.979951, lpips=unavailable

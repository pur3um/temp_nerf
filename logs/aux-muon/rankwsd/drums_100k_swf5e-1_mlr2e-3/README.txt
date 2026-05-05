====================================================================================================
saved_time: 2026-05-05 00:34:53
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5e-1_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: drums_100k_swf5e-1_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0050618267
current_train_psnr: 28.098696
testset_mean_loss: 0.0030103025
testset_mean_psnr: 25.488147
testset_mean_ssim: 0.928579
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0018124366, psnr=27.417371, ssim=0.929596, lpips=unavailable
  image_001: loss=0.0021991516, psnr=26.577448, ssim=0.925824, lpips=unavailable
  image_002: loss=0.0014494199, psnr=28.388058, ssim=0.946547, lpips=unavailable
  image_003: loss=0.0022604153, psnr=26.458117, ssim=0.938370, lpips=unavailable
  image_004: loss=0.0019561793, psnr=27.085913, ssim=0.936104, lpips=unavailable
  image_005: loss=0.0018357926, psnr=27.361764, ssim=0.940818, lpips=unavailable
  image_006: loss=0.0048221089, psnr=23.167630, ssim=0.904965, lpips=unavailable
  image_007: loss=0.0038858326, psnr=24.105159, ssim=0.917022, lpips=unavailable
  image_008: loss=0.0031563174, psnr=25.008193, ssim=0.929048, lpips=unavailable
  image_009: loss=0.0036610800, psnr=24.363908, ssim=0.929626, lpips=unavailable
  image_010: loss=0.0034394183, psnr=24.635150, ssim=0.935543, lpips=unavailable
  image_011: loss=0.0042313458, psnr=23.735215, ssim=0.915747, lpips=unavailable
  image_012: loss=0.0036422785, psnr=24.386268, ssim=0.928342, lpips=unavailable
  image_013: loss=0.0028087408, psnr=25.514883, ssim=0.934384, lpips=unavailable
  image_014: loss=0.0025382186, psnr=25.954710, ssim=0.938637, lpips=unavailable
  image_015: loss=0.0040836958, psnr=23.889466, ssim=0.931969, lpips=unavailable
  image_016: loss=0.0020540969, psnr=26.873791, ssim=0.950640, lpips=unavailable
  image_017: loss=0.0029507682, psnr=25.300649, ssim=0.930307, lpips=unavailable
  image_018: loss=0.0030700497, psnr=25.128546, ssim=0.927322, lpips=unavailable
  image_019: loss=0.0058899433, psnr=22.298889, ssim=0.889701, lpips=unavailable
  image_020: loss=0.0034728628, psnr=24.593124, ssim=0.914996, lpips=unavailable
  image_021: loss=0.0038708227, psnr=24.121967, ssim=0.910394, lpips=unavailable
  image_022: loss=0.0025211619, psnr=25.983992, ssim=0.932849, lpips=unavailable
  image_023: loss=0.0020496310, psnr=26.883243, ssim=0.934095, lpips=unavailable
  image_024: loss=0.0015957942, psnr=27.970231, ssim=0.941638, lpips=unavailable

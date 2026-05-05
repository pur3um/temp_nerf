====================================================================================================
saved_time: 2026-05-03 09:55:37
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/mic.txt --expname mic_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: mic_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0025809659
current_train_psnr: 32.117550
testset_mean_loss: 0.0008988267
testset_mean_psnr: 30.501914
testset_mean_ssim: 0.969358
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0008037944, psnr=30.948550, ssim=0.970896, lpips=unavailable
  image_001: loss=0.0007538106, psnr=31.227377, ssim=0.972377, lpips=unavailable
  image_002: loss=0.0009677995, psnr=30.142146, ssim=0.966247, lpips=unavailable
  image_003: loss=0.0009412562, psnr=30.262921, ssim=0.965274, lpips=unavailable
  image_004: loss=0.0009106811, psnr=30.406336, ssim=0.964176, lpips=unavailable
  image_005: loss=0.0008365913, psnr=30.774866, ssim=0.969271, lpips=unavailable
  image_006: loss=0.0008075067, psnr=30.928538, ssim=0.981072, lpips=unavailable
  image_007: loss=0.0008846178, psnr=30.532443, ssim=0.972152, lpips=unavailable
  image_008: loss=0.0008603907, psnr=30.653042, ssim=0.961730, lpips=unavailable
  image_009: loss=0.0008678422, psnr=30.615592, ssim=0.965968, lpips=unavailable
  image_010: loss=0.0007201203, psnr=31.425949, ssim=0.971752, lpips=unavailable
  image_011: loss=0.0008007834, psnr=30.964849, ssim=0.972885, lpips=unavailable
  image_012: loss=0.0013523152, psnr=28.689220, ssim=0.972008, lpips=unavailable
  image_013: loss=0.0010772428, psnr=29.676864, ssim=0.975172, lpips=unavailable
  image_014: loss=0.0009958423, psnr=30.018094, ssim=0.972395, lpips=unavailable
  image_015: loss=0.0008247602, psnr=30.836723, ssim=0.971737, lpips=unavailable
  image_016: loss=0.0008743021, psnr=30.583384, ssim=0.968057, lpips=unavailable
  image_017: loss=0.0008442322, psnr=30.735380, ssim=0.967589, lpips=unavailable
  image_018: loss=0.0007971607, psnr=30.984541, ssim=0.974308, lpips=unavailable
  image_019: loss=0.0009410778, psnr=30.263744, ssim=0.979818, lpips=unavailable
  image_020: loss=0.0009122741, psnr=30.398746, ssim=0.963077, lpips=unavailable
  image_021: loss=0.0009355194, psnr=30.289472, ssim=0.960345, lpips=unavailable
  image_022: loss=0.0009541288, psnr=30.203929, ssim=0.962380, lpips=unavailable
  image_023: loss=0.0010408477, psnr=29.826128, ssim=0.963082, lpips=unavailable
  image_024: loss=0.0007657700, psnr=31.159016, ssim=0.970180, lpips=unavailable

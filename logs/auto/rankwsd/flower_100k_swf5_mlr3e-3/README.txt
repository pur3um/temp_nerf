====================================================================================================
saved_time: 2026-05-05 12:36:47
script_path: /data2/greenx9/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/auto/rankwsd --config configs/flower.txt --expname flower_100k_swf5_mlr3e-3 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: flower_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 46 min
current_train_loss: 0.0040579648
current_train_psnr: 27.113171
testset_mean_loss: 0.0023341344
testset_mean_psnr: 26.402759
testset_mean_ssim: 0.832291
testset_mean_lpips: 0.116761
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0024802960, psnr=26.054965, ssim=0.808067, lpips=0.155235
  image_001: loss=0.0024208899, psnr=26.160249, ssim=0.835808, lpips=0.119294
  image_002: loss=0.0028488399, psnr=25.453319, ssim=0.805552, lpips=0.123480
  image_003: loss=0.0023677964, psnr=26.256556, ssim=0.852207, lpips=0.086464
  image_004: loss=0.0015528499, psnr=28.088705, ssim=0.859821, lpips=0.099331

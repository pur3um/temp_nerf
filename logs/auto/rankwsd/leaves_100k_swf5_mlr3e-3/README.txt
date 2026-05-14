====================================================================================================
saved_time: 2026-05-05 12:31:58
script_path: /data2/greenx9/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/auto/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5_mlr3e-3 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: leaves_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 41 min
current_train_loss: 0.0157572981
current_train_psnr: 21.319096
testset_mean_loss: 0.0084139579
testset_mean_psnr: 20.786513
testset_mean_ssim: 0.732191
testset_mean_lpips: 0.193978
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0103403777, psnr=19.854636, ssim=0.675060, lpips=0.258630
  image_001: loss=0.0074272715, psnr=21.291707, ssim=0.764022, lpips=0.161915
  image_002: loss=0.0080369776, psnr=20.949072, ssim=0.747739, lpips=0.172177
  image_003: loss=0.0078512048, psnr=21.050637, ssim=0.741945, lpips=0.183190

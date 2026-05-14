====================================================================================================
saved_time: 2026-05-05 12:34:16
script_path: /data2/greenx9/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/auto/rankwsd --config configs/fern.txt --expname fern_100k_swf5_mlr3e-3 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: fern_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 43 min
current_train_loss: 0.0062284712
current_train_psnr: 25.462826
testset_mean_loss: 0.0028613571
testset_mean_psnr: 25.486964
testset_mean_ssim: 0.803662
testset_mean_lpips: 0.173276
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0035024064, psnr=24.556334, ssim=0.781296, lpips=0.195775
  image_001: loss=0.0026186865, psnr=25.819165, ssim=0.817593, lpips=0.156317
  image_002: loss=0.0024629785, psnr=26.085394, ssim=0.812097, lpips=0.167734

====================================================================================================
saved_time: 2026-05-05 12:33:55
script_path: /data2/greenx9/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/auto/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5_mlr3e-3 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: orchids_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 43 min
current_train_loss: 0.0154004144
current_train_psnr: 21.679256
testset_mean_loss: 0.0090294305
testset_mean_psnr: 20.529819
testset_mean_ssim: 0.685097
testset_mean_lpips: 0.199821
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0098642725, psnr=20.059349, ssim=0.653225, lpips=0.212175
  image_001: loss=0.0061766119, psnr=22.092497, ssim=0.758464, lpips=0.153306
  image_002: loss=0.0101690404, psnr=19.927200, ssim=0.667458, lpips=0.204580
  image_003: loss=0.0099077970, psnr=20.040229, ssim=0.661242, lpips=0.229224

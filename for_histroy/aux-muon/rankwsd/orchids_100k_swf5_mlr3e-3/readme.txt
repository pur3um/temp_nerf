====================================================================================================
saved_time: 2026-05-04 11:15:40
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: orchids_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0145552661
current_train_psnr: 21.873335
testset_mean_loss: 0.0087684329
testset_mean_psnr: 20.654275
testset_mean_ssim: 0.674409
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0096238600, psnr=20.166507, ssim=0.644870, lpips=unavailable
  image_001: loss=0.0060383989, psnr=22.190782, ssim=0.748109, lpips=unavailable
  image_002: loss=0.0097967163, psnr=20.089195, ssim=0.653472, lpips=unavailable
  image_003: loss=0.0096147563, psnr=20.170617, ssim=0.651186, lpips=unavailable

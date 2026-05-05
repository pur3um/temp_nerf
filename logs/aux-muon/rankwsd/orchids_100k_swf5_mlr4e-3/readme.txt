====================================================================================================
saved_time: 2026-05-03 08:48:13
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: orchids_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0147057334
current_train_psnr: 21.683018
testset_mean_loss: 0.0090159958
testset_mean_psnr: 20.538387
testset_mean_ssim: 0.666641
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0098055368, psnr=20.085286, ssim=0.637415, lpips=unavailable
  image_001: loss=0.0061414260, psnr=22.117308, ssim=0.741585, lpips=unavailable
  image_002: loss=0.0102531128, psnr=19.891443, ssim=0.644206, lpips=unavailable
  image_003: loss=0.0098639075, psnr=20.059510, ssim=0.643360, lpips=unavailable

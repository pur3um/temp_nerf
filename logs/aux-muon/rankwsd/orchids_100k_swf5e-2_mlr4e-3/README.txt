====================================================================================================
saved_time: 2026-05-03 22:01:29
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-2_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: orchids_100k_swf5e-2_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0082301404
current_train_psnr: 24.833923
testset_mean_loss: 0.0068693064
testset_mean_psnr: 21.778715
testset_mean_ssim: 0.751243
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0077330293, psnr=21.116503, ssim=0.716432, lpips=unavailable
  image_001: loss=0.0041339323, psnr=23.836366, ssim=0.827035, lpips=unavailable
  image_002: loss=0.0081718424, psnr=20.876800, ssim=0.730066, lpips=unavailable
  image_003: loss=0.0074384217, psnr=21.285192, ssim=0.731437, lpips=unavailable

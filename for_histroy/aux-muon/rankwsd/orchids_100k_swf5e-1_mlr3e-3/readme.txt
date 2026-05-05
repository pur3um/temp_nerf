====================================================================================================
saved_time: 2026-05-04 17:52:08
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-1_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: orchids_100k_swf5e-1_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0088939779
current_train_psnr: 24.091595
testset_mean_loss: 0.0069786228
testset_mean_psnr: 21.710325
testset_mean_ssim: 0.743046
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0079950551, psnr=20.971785, ssim=0.707298, lpips=unavailable
  image_001: loss=0.0041994140, psnr=23.768113, ssim=0.820026, lpips=unavailable
  image_002: loss=0.0082344124, psnr=20.843674, ssim=0.721196, lpips=unavailable
  image_003: loss=0.0074856095, psnr=21.257728, ssim=0.723664, lpips=unavailable

====================================================================================================
saved_time: 2026-05-03 00:23:11
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/orchids.txt --expname orchids_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000
expname: orchids_100k
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0095141847
current_train_psnr: 24.084595
testset_mean_loss: 0.0074553264
testset_mean_psnr: 21.413018
testset_mean_ssim: 0.732149
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0086662918, psnr=20.621667, ssim=0.692897, lpips=unavailable
  image_001: loss=0.0045855073, psnr=23.386126, ssim=0.807872, lpips=unavailable
  image_002: loss=0.0086794896, psnr=20.615058, ssim=0.713700, lpips=unavailable
  image_003: loss=0.0078900168, psnr=21.029221, ssim=0.714126, lpips=unavailable

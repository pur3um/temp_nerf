====================================================================================================
saved_time: 2026-05-05 00:28:53
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/orchids.txt --expname orchids_100k_swf5e-2_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: orchids_100k_swf5e-2_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0081421863
current_train_psnr: 24.719103
testset_mean_loss: 0.0068268991
testset_mean_psnr: 21.813130
testset_mean_ssim: 0.751753
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0078938520, psnr=21.027110, ssim=0.714399, lpips=unavailable
  image_001: loss=0.0040509645, psnr=23.924416, ssim=0.828345, lpips=unavailable
  image_002: loss=0.0080458736, psnr=20.944268, ssim=0.731332, lpips=unavailable
  image_003: loss=0.0073169065, psnr=21.356725, ssim=0.732936, lpips=unavailable

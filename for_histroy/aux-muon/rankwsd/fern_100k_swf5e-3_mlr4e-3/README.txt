====================================================================================================
saved_time: 2026-05-04 02:59:13
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: fern_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0039167507
current_train_psnr: 28.435463
testset_mean_loss: 0.0018813706
testset_mean_psnr: 27.294562
testset_mean_ssim: 0.861465
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0022133212, psnr=26.549555, ssim=0.848374, lpips=unavailable
  image_001: loss=0.0018357058, psnr=27.361969, ssim=0.869496, lpips=unavailable
  image_002: loss=0.0015950847, psnr=27.972162, ssim=0.866526, lpips=unavailable

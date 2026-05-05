====================================================================================================
saved_time: 2026-05-05 02:08:29
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5e-2_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: leaves_100k_swf5e-2_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0092861382
current_train_psnr: 23.638622
testset_mean_loss: 0.0057103977
testset_mean_psnr: 22.500799
testset_mean_ssim: 0.802982
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0074193040, psnr=21.296368, ssim=0.750750, lpips=unavailable
  image_001: loss=0.0050849635, psnr=22.937122, ssim=0.826920, lpips=unavailable
  image_002: loss=0.0056464467, psnr=22.482248, ssim=0.811205, lpips=unavailable
  image_003: loss=0.0046908767, psnr=23.287460, ssim=0.823052, lpips=unavailable

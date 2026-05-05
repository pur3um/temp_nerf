====================================================================================================
saved_time: 2026-05-03 07:09:11
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: fern_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0067952285
current_train_psnr: 25.126352
testset_mean_loss: 0.0028704056
testset_mean_psnr: 25.457182
testset_mean_ssim: 0.799790
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0034013330, psnr=24.683508, ssim=0.778709, lpips=unavailable
  image_001: loss=0.0026839622, psnr=25.712236, ssim=0.812421, lpips=unavailable
  image_002: loss=0.0025259217, psnr=25.975801, ssim=0.808241, lpips=unavailable

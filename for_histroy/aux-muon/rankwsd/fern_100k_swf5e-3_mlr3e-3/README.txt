====================================================================================================
saved_time: 2026-05-05 05:27:02
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5e-3_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: fern_100k_swf5e-3_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0038377014
current_train_psnr: 28.431273
testset_mean_loss: 0.0018855815
testset_mean_psnr: 27.289810
testset_mean_ssim: 0.861636
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0022405931, psnr=26.496370, ssim=0.847679, lpips=unavailable
  image_001: loss=0.0018327610, psnr=27.368941, ssim=0.869291, lpips=unavailable
  image_002: loss=0.0015833904, psnr=28.004120, ssim=0.867940, lpips=unavailable

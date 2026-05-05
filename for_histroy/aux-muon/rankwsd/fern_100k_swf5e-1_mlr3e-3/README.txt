====================================================================================================
saved_time: 2026-05-04 16:13:05
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5e-1_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: fern_100k_swf5e-1_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0043430980
current_train_psnr: 27.567919
testset_mean_loss: 0.0019859477
testset_mean_psnr: 27.064405
testset_mean_ssim: 0.852236
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0023680227, psnr=26.256141, ssim=0.836123, lpips=unavailable
  image_001: loss=0.0019101202, psnr=27.189393, ssim=0.861532, lpips=unavailable
  image_002: loss=0.0016797002, psnr=27.747682, ssim=0.859052, lpips=unavailable

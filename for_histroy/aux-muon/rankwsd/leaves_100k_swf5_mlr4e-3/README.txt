====================================================================================================
saved_time: 2026-05-03 10:27:32
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: leaves_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0158166233
current_train_psnr: 21.251095
testset_mean_loss: 0.0084839495
testset_mean_psnr: 20.762955
testset_mean_ssim: 0.706862
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0107677691, psnr=19.678743, ssim=0.645346, lpips=unavailable
  image_001: loss=0.0074780257, psnr=21.262130, ssim=0.739535, lpips=unavailable
  image_002: loss=0.0080444291, psnr=20.945048, ssim=0.723815, lpips=unavailable
  image_003: loss=0.0076455739, psnr=21.165899, ssim=0.718752, lpips=unavailable

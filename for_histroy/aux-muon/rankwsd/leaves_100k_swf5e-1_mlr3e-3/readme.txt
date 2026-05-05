====================================================================================================
saved_time: 2026-05-04 19:31:19
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/leaves.txt --expname leaves_100k_swf5e-1_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: leaves_100k_swf5e-1_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0096282475
current_train_psnr: 23.439318
testset_mean_loss: 0.0059388686
testset_mean_psnr: 22.335193
testset_mean_ssim: 0.792309
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0078466907, psnr=21.053135, ssim=0.735438, lpips=unavailable
  image_001: loss=0.0052088764, psnr=22.832559, ssim=0.819571, lpips=unavailable
  image_002: loss=0.0057309372, psnr=22.417743, ssim=0.803581, lpips=unavailable
  image_003: loss=0.0049689701, psnr=23.037336, ssim=0.810648, lpips=unavailable

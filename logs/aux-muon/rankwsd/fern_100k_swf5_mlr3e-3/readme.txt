====================================================================================================
saved_time: 2026-05-04 09:36:34
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5 --N_iters 100000
expname: fern_100k_swf5_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0069695120
current_train_psnr: 24.931774
testset_mean_loss: 0.0028424736
testset_mean_psnr: 25.497882
testset_mean_ssim: 0.802954
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0033533522, psnr=24.745208, ssim=0.783482, lpips=unavailable
  image_001: loss=0.0026701286, psnr=25.734678, ssim=0.816144, lpips=unavailable
  image_002: loss=0.0025039401, psnr=26.013760, ssim=0.809237, lpips=unavailable

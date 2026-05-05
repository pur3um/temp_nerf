====================================================================================================
saved_time: 2026-05-03 13:45:47
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/fern.txt --expname fern_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: fern_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0042499756
current_train_psnr: 27.645933
testset_mean_loss: 0.0019724309
testset_mean_psnr: 27.094202
testset_mean_ssim: 0.854059
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0023465573, psnr=26.295688, ssim=0.839000, lpips=unavailable
  image_001: loss=0.0019107459, psnr=27.187970, ssim=0.863093, lpips=unavailable
  image_002: loss=0.0016599895, psnr=27.798946, ssim=0.860084, lpips=unavailable

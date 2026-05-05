====================================================================================================
saved_time: 2026-05-03 02:02:22
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/leaves.txt --expname leaves_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000
expname: leaves_100k
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0105041312
current_train_psnr: 23.145769
testset_mean_loss: 0.0062161281
testset_mean_psnr: 22.133161
testset_mean_ssim: 0.788077
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0081325648, psnr=20.897725, ssim=0.733031, lpips=unavailable
  image_001: loss=0.0055373292, psnr=22.566996, ssim=0.812519, lpips=unavailable
  image_002: loss=0.0060420725, psnr=22.188141, ssim=0.798721, lpips=unavailable
  image_003: loss=0.0051525459, psnr=22.879781, ssim=0.808036, lpips=unavailable

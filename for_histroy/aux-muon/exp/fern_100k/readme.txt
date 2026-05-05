====================================================================================================
saved_time: 2026-05-02 22:43:46
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/fern.txt --expname fern_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000
expname: fern_100k
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 39 min
current_train_loss: 0.0044067590
current_train_psnr: 27.446817
testset_mean_loss: 0.0020797435
testset_mean_psnr: 26.859488
testset_mean_ssim: 0.851335
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0024584820, psnr=26.093329, ssim=0.836149, lpips=unavailable
  image_001: loss=0.0020053349, psnr=26.978131, ssim=0.859901, lpips=unavailable
  image_002: loss=0.0017754135, psnr=27.507005, ssim=0.857956, lpips=unavailable

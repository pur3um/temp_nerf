====================================================================================================
saved_time: 2026-05-02 21:04:12
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/exp --config configs/flower.txt --expname flower_100k --optimizer aux-muon --train_scheduler exp_decay --muon_lrate 3e-3 --N_iters 100000
expname: flower_100k
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0024905521
current_train_psnr: 29.878130
testset_mean_loss: 0.0015971626
testset_mean_psnr: 28.118168
testset_mean_ssim: 0.877435
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016488591, psnr=27.828164, ssim=0.862165, lpips=unavailable
  image_001: loss=0.0016395645, psnr=27.852715, ssim=0.881580, lpips=unavailable
  image_002: loss=0.0022038550, psnr=26.568170, ssim=0.844251, lpips=unavailable
  image_003: loss=0.0015459072, psnr=28.108165, ssim=0.891980, lpips=unavailable
  image_004: loss=0.0009476271, psnr=30.233625, ssim=0.907200, lpips=unavailable

====================================================================================================
saved_time: 2026-05-02 18:32:51
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_200k_swf5e-2 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 200000
expname: flower_200k_swf5e-2
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 3 hour 18 min
current_train_loss: 0.0021250315
current_train_psnr: 30.947226
testset_mean_loss: 0.0013364799
testset_mean_psnr: 28.964455
testset_mean_ssim: 0.898771
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0012735727, psnr=28.949762, ssim=0.888135, lpips=unavailable
  image_001: loss=0.0013283087, psnr=28.767009, ssim=0.905562, lpips=unavailable
  image_002: loss=0.0019665039, psnr=27.063052, ssim=0.862307, lpips=unavailable
  image_003: loss=0.0014133652, psnr=28.497456, ssim=0.909768, lpips=unavailable
  image_004: loss=0.0007006490, psnr=31.544994, ssim=0.928083, lpips=unavailable

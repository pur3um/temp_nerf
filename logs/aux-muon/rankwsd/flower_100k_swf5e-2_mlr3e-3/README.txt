====================================================================================================
saved_time: 2026-05-04 21:10:23
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_100k_swf5e-2_mlr3e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 100000
expname: flower_100k_swf5e-2_mlr3e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 1 hour 38 min
current_train_loss: 0.0020604399
current_train_psnr: 30.580933
testset_mean_loss: 0.0014202538
testset_mean_psnr: 28.677712
testset_mean_ssim: 0.888986
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0013929917, psnr=28.560514, ssim=0.875785, lpips=unavailable
  image_001: loss=0.0014046513, psnr=28.524314, ssim=0.895352, lpips=unavailable
  image_002: loss=0.0020693049, psnr=26.841755, ssim=0.852942, lpips=unavailable
  image_003: loss=0.0014579865, psnr=28.362465, ssim=0.901796, lpips=unavailable
  image_004: loss=0.0007763343, psnr=31.099512, ssim=0.919054, lpips=unavailable

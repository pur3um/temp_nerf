====================================================================================================
saved_time: 2026-05-03 14:00:11
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5e-1_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-1 --N_iters 100000
expname: lego_100k_swf5e-1_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0029262579
current_train_psnr: 33.616909
testset_mean_loss: 0.0007332357
testset_mean_psnr: 31.579140
testset_mean_ssim: 0.964675
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0005425242, psnr=32.655808, ssim=0.968233, lpips=unavailable
  image_001: loss=0.0013013717, psnr=28.855986, ssim=0.957338, lpips=unavailable
  image_002: loss=0.0004267157, psnr=33.698612, ssim=0.971794, lpips=unavailable
  image_003: loss=0.0005344939, psnr=32.720572, ssim=0.969208, lpips=unavailable
  image_004: loss=0.0007581330, psnr=31.202545, ssim=0.962813, lpips=unavailable
  image_005: loss=0.0004972732, psnr=33.034049, ssim=0.967213, lpips=unavailable
  image_006: loss=0.0007670404, psnr=31.151817, ssim=0.966233, lpips=unavailable
  image_007: loss=0.0004770755, psnr=33.214128, ssim=0.965955, lpips=unavailable
  image_008: loss=0.0005811712, psnr=32.356958, ssim=0.970206, lpips=unavailable
  image_009: loss=0.0010435961, psnr=29.814675, ssim=0.967038, lpips=unavailable
  image_010: loss=0.0014791973, psnr=28.299739, ssim=0.953345, lpips=unavailable
  image_011: loss=0.0011545540, psnr=29.375857, ssim=0.953767, lpips=unavailable
  image_012: loss=0.0007672755, psnr=31.150486, ssim=0.964002, lpips=unavailable
  image_013: loss=0.0007231406, psnr=31.407772, ssim=0.964962, lpips=unavailable
  image_014: loss=0.0005727099, psnr=32.420652, ssim=0.965021, lpips=unavailable
  image_015: loss=0.0006740637, psnr=31.712990, ssim=0.967108, lpips=unavailable
  image_016: loss=0.0007056284, psnr=31.514239, ssim=0.974265, lpips=unavailable
  image_017: loss=0.0006773231, psnr=31.692040, ssim=0.966492, lpips=unavailable
  image_018: loss=0.0006359184, psnr=31.965985, ssim=0.965054, lpips=unavailable
  image_019: loss=0.0005828162, psnr=32.344683, ssim=0.962251, lpips=unavailable
  image_020: loss=0.0004152814, psnr=33.816574, ssim=0.967962, lpips=unavailable
  image_021: loss=0.0006095984, psnr=32.149561, ssim=0.965181, lpips=unavailable
  image_022: loss=0.0008429242, psnr=30.742114, ssim=0.960875, lpips=unavailable
  image_023: loss=0.0008436514, psnr=30.738369, ssim=0.959128, lpips=unavailable
  image_024: loss=0.0007174154, psnr=31.442293, ssim=0.961432, lpips=unavailable

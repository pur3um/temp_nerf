====================================================================================================
saved_time: 2026-05-03 05:52:20
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: lego_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 1 min
current_train_loss: 0.0042090761
current_train_psnr: 29.979288
testset_mean_loss: 0.0013729117
testset_mean_psnr: 28.768559
testset_mean_ssim: 0.942186
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0011110723, psnr=29.542577, ssim=0.945443, lpips=unavailable
  image_001: loss=0.0022239669, psnr=26.528717, ssim=0.935989, lpips=unavailable
  image_002: loss=0.0008897418, psnr=30.507360, ssim=0.951138, lpips=unavailable
  image_003: loss=0.0011245247, psnr=29.490309, ssim=0.947140, lpips=unavailable
  image_004: loss=0.0013230952, psnr=28.784089, ssim=0.941226, lpips=unavailable
  image_005: loss=0.0010653648, psnr=29.725016, ssim=0.943924, lpips=unavailable
  image_006: loss=0.0013844388, psnr=28.587262, ssim=0.943979, lpips=unavailable
  image_007: loss=0.0009950355, psnr=30.021614, ssim=0.943036, lpips=unavailable
  image_008: loss=0.0012964343, psnr=28.872494, ssim=0.946406, lpips=unavailable
  image_009: loss=0.0019496329, psnr=27.100471, ssim=0.943149, lpips=unavailable
  image_010: loss=0.0023783187, psnr=26.237299, ssim=0.927519, lpips=unavailable
  image_011: loss=0.0019395562, psnr=27.122976, ssim=0.928237, lpips=unavailable
  image_012: loss=0.0013689924, psnr=28.635989, ssim=0.943413, lpips=unavailable
  image_013: loss=0.0011306922, psnr=29.466556, ssim=0.947957, lpips=unavailable
  image_014: loss=0.0011283561, psnr=29.475538, ssim=0.945226, lpips=unavailable
  image_015: loss=0.0012772692, psnr=28.937175, ssim=0.947978, lpips=unavailable
  image_016: loss=0.0014645276, psnr=28.343024, ssim=0.952925, lpips=unavailable
  image_017: loss=0.0012980830, psnr=28.866975, ssim=0.947513, lpips=unavailable
  image_018: loss=0.0011906293, psnr=29.242234, ssim=0.945595, lpips=unavailable
  image_019: loss=0.0010412743, psnr=29.824348, ssim=0.941503, lpips=unavailable
  image_020: loss=0.0009368722, psnr=30.283196, ssim=0.942096, lpips=unavailable
  image_021: loss=0.0012754054, psnr=28.943517, ssim=0.938231, lpips=unavailable
  image_022: loss=0.0015931980, psnr=27.977302, ssim=0.934545, lpips=unavailable
  image_023: loss=0.0015966108, psnr=27.968009, ssim=0.932486, lpips=unavailable
  image_024: loss=0.0013397001, psnr=28.729924, ssim=0.937995, lpips=unavailable

====================================================================================================
saved_time: 2026-05-04 16:27:19
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5_mlr2e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 2e-3 --sched_warmup_frac 5 --N_iters 100000
expname: drums_100k_swf5_mlr2e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0072738337
current_train_psnr: 25.576365
testset_mean_loss: 0.0038826271
testset_mean_psnr: 24.258146
testset_mean_ssim: 0.911932
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0027161660, psnr=25.660437, ssim=0.913127, lpips=unavailable
  image_001: loss=0.0030043314, psnr=25.222522, ssim=0.911127, lpips=unavailable
  image_002: loss=0.0023022790, psnr=26.378420, ssim=0.927795, lpips=unavailable
  image_003: loss=0.0030758737, psnr=25.120315, ssim=0.924253, lpips=unavailable
  image_004: loss=0.0027126714, psnr=25.666028, ssim=0.920833, lpips=unavailable
  image_005: loss=0.0028844993, psnr=25.399296, ssim=0.919502, lpips=unavailable
  image_006: loss=0.0053992728, psnr=22.676647, ssim=0.888312, lpips=unavailable
  image_007: loss=0.0046884050, psnr=23.289749, ssim=0.898633, lpips=unavailable
  image_008: loss=0.0046843044, psnr=23.293549, ssim=0.903490, lpips=unavailable
  image_009: loss=0.0048163328, psnr=23.172835, ssim=0.911570, lpips=unavailable
  image_010: loss=0.0046427725, psnr=23.332226, ssim=0.914602, lpips=unavailable
  image_011: loss=0.0048882179, psnr=23.108494, ssim=0.902397, lpips=unavailable
  image_012: loss=0.0043913648, psnr=23.574005, ssim=0.915043, lpips=unavailable
  image_013: loss=0.0038442893, psnr=24.151839, ssim=0.917909, lpips=unavailable
  image_014: loss=0.0036163917, psnr=24.417245, ssim=0.918996, lpips=unavailable
  image_015: loss=0.0046392623, psnr=23.335511, ssim=0.919909, lpips=unavailable
  image_016: loss=0.0029346712, psnr=25.324405, ssim=0.932162, lpips=unavailable
  image_017: loss=0.0035175146, psnr=24.537641, ssim=0.914219, lpips=unavailable
  image_018: loss=0.0043351408, psnr=23.629968, ssim=0.906234, lpips=unavailable
  image_019: loss=0.0060063233, psnr=22.213913, ssim=0.881638, lpips=unavailable
  image_020: loss=0.0047139330, psnr=23.266166, ssim=0.894971, lpips=unavailable
  image_021: loss=0.0045983037, psnr=23.374023, ssim=0.898549, lpips=unavailable
  image_022: loss=0.0033698557, psnr=24.723887, ssim=0.917411, lpips=unavailable
  image_023: loss=0.0028310961, psnr=25.480454, ssim=0.919248, lpips=unavailable
  image_024: loss=0.0024524042, psnr=26.104079, ssim=0.926358, lpips=unavailable

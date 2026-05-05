====================================================================================================
saved_time: 2026-05-04 08:19:06
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: drums_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0047502806
current_train_psnr: 28.238796
testset_mean_loss: 0.0028191295
testset_mean_psnr: 25.715246
testset_mean_ssim: 0.932094
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017648364, psnr=27.532955, ssim=0.934124, lpips=unavailable
  image_001: loss=0.0021695045, psnr=26.636394, ssim=0.929487, lpips=unavailable
  image_002: loss=0.0013669983, psnr=28.642320, ssim=0.948301, lpips=unavailable
  image_003: loss=0.0022422899, psnr=26.493082, ssim=0.941272, lpips=unavailable
  image_004: loss=0.0019551364, psnr=27.088229, ssim=0.937610, lpips=unavailable
  image_005: loss=0.0018273945, psnr=27.381677, ssim=0.942627, lpips=unavailable
  image_006: loss=0.0039864322, psnr=23.994156, ssim=0.911761, lpips=unavailable
  image_007: loss=0.0034628357, psnr=24.605681, ssim=0.922265, lpips=unavailable
  image_008: loss=0.0029051092, psnr=25.368375, ssim=0.933583, lpips=unavailable
  image_009: loss=0.0035413431, psnr=24.508320, ssim=0.933032, lpips=unavailable
  image_010: loss=0.0034065761, psnr=24.676819, ssim=0.936529, lpips=unavailable
  image_011: loss=0.0038023833, psnr=24.199441, ssim=0.921979, lpips=unavailable
  image_012: loss=0.0035832364, psnr=24.457245, ssim=0.931190, lpips=unavailable
  image_013: loss=0.0025872779, psnr=25.871569, ssim=0.937140, lpips=unavailable
  image_014: loss=0.0025383129, psnr=25.954548, ssim=0.939793, lpips=unavailable
  image_015: loss=0.0037541592, psnr=24.254873, ssim=0.934308, lpips=unavailable
  image_016: loss=0.0020393049, psnr=26.905178, ssim=0.951160, lpips=unavailable
  image_017: loss=0.0026173552, psnr=25.821373, ssim=0.935279, lpips=unavailable
  image_018: loss=0.0030769664, psnr=25.118772, ssim=0.931170, lpips=unavailable
  image_019: loss=0.0046583316, psnr=23.317696, ssim=0.897463, lpips=unavailable
  image_020: loss=0.0033786746, psnr=24.712536, ssim=0.918482, lpips=unavailable
  image_021: loss=0.0037271974, psnr=24.286176, ssim=0.915732, lpips=unavailable
  image_022: loss=0.0025338216, psnr=25.962239, ssim=0.934160, lpips=unavailable
  image_023: loss=0.0020196436, psnr=26.947253, ssim=0.938868, lpips=unavailable
  image_024: loss=0.0015331157, psnr=28.144250, ssim=0.945028, lpips=unavailable

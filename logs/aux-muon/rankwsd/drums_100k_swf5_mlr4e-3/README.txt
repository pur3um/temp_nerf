====================================================================================================
saved_time: 2026-05-03 07:54:15
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/drums.txt --expname drums_100k_swf5_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5 --N_iters 100000
expname: drums_100k_swf5_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0080103260
current_train_psnr: 25.811817
testset_mean_loss: 0.0040179070
testset_mean_psnr: 24.106880
testset_mean_ssim: 0.907135
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0028069296, psnr=25.517685, ssim=0.909102, lpips=unavailable
  image_001: loss=0.0031499795, psnr=25.016923, ssim=0.905673, lpips=unavailable
  image_002: loss=0.0025677669, psnr=25.904444, ssim=0.921398, lpips=unavailable
  image_003: loss=0.0030710273, psnr=25.127163, ssim=0.920801, lpips=unavailable
  image_004: loss=0.0027399934, psnr=25.622505, ssim=0.915239, lpips=unavailable
  image_005: loss=0.0029336102, psnr=25.325976, ssim=0.914100, lpips=unavailable
  image_006: loss=0.0052498369, psnr=22.798542, ssim=0.883244, lpips=unavailable
  image_007: loss=0.0049241837, psnr=23.076657, ssim=0.892168, lpips=unavailable
  image_008: loss=0.0050559770, psnr=22.961949, ssim=0.895338, lpips=unavailable
  image_009: loss=0.0047435262, psnr=23.238987, ssim=0.908176, lpips=unavailable
  image_010: loss=0.0049348157, psnr=23.067291, ssim=0.908717, lpips=unavailable
  image_011: loss=0.0050815595, psnr=22.940030, ssim=0.898230, lpips=unavailable
  image_012: loss=0.0045843259, psnr=23.387245, ssim=0.912602, lpips=unavailable
  image_013: loss=0.0039500552, psnr=24.033968, ssim=0.914307, lpips=unavailable
  image_014: loss=0.0036972254, psnr=24.321241, ssim=0.914399, lpips=unavailable
  image_015: loss=0.0049654800, psnr=23.040388, ssim=0.914299, lpips=unavailable
  image_016: loss=0.0030576028, psnr=25.146189, ssim=0.928455, lpips=unavailable
  image_017: loss=0.0038040997, psnr=24.197481, ssim=0.910019, lpips=unavailable
  image_018: loss=0.0043596094, psnr=23.605524, ssim=0.901477, lpips=unavailable
  image_019: loss=0.0065085883, psnr=21.865132, ssim=0.872015, lpips=unavailable
  image_020: loss=0.0048433766, psnr=23.148518, ssim=0.890573, lpips=unavailable
  image_021: loss=0.0043824343, psnr=23.582846, ssim=0.897766, lpips=unavailable
  image_022: loss=0.0035573870, psnr=24.488689, ssim=0.912739, lpips=unavailable
  image_023: loss=0.0028632558, psnr=25.431398, ssim=0.916766, lpips=unavailable
  image_024: loss=0.0026150292, psnr=25.825234, ssim=0.920768, lpips=unavailable

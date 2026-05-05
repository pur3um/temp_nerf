====================================================================================================
saved_time: 2026-05-04 06:17:23
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_100k_swf5e-3_mlr4e-3 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 4e-3 --sched_warmup_frac 5e-3 --N_iters 100000
expname: lego_100k_swf5e-3_mlr4e-3
iter: 100000
global_step: 99999
elapsed_time_from_train_start: 2 hour 0 min
current_train_loss: 0.0028963599
current_train_psnr: 33.811089
testset_mean_loss: 0.0006882898
testset_mean_psnr: 31.870506
testset_mean_ssim: 0.966911
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004976429, psnr=33.030821, ssim=0.970536, lpips=unavailable
  image_001: loss=0.0012267811, psnr=29.112329, ssim=0.958365, lpips=unavailable
  image_002: loss=0.0004083219, psnr=33.889972, ssim=0.973706, lpips=unavailable
  image_003: loss=0.0005104889, psnr=32.920136, ssim=0.971002, lpips=unavailable
  image_004: loss=0.0007312579, psnr=31.359294, ssim=0.965287, lpips=unavailable
  image_005: loss=0.0004597252, psnr=33.375016, ssim=0.970233, lpips=unavailable
  image_006: loss=0.0006815264, psnr=31.665173, ssim=0.969344, lpips=unavailable
  image_007: loss=0.0004308047, psnr=33.657195, ssim=0.969116, lpips=unavailable
  image_008: loss=0.0005234292, psnr=32.811420, ssim=0.972406, lpips=unavailable
  image_009: loss=0.0010059628, psnr=29.974180, ssim=0.968503, lpips=unavailable
  image_010: loss=0.0014439699, psnr=28.404418, ssim=0.953166, lpips=unavailable
  image_011: loss=0.0010867299, psnr=29.638783, ssim=0.955812, lpips=unavailable
  image_012: loss=0.0007068060, psnr=31.506997, ssim=0.966513, lpips=unavailable
  image_013: loss=0.0006681393, psnr=31.751329, ssim=0.966840, lpips=unavailable
  image_014: loss=0.0005415377, psnr=32.663713, ssim=0.967293, lpips=unavailable
  image_015: loss=0.0006467896, psnr=31.892369, ssim=0.968980, lpips=unavailable
  image_016: loss=0.0006874378, psnr=31.627665, ssim=0.975612, lpips=unavailable
  image_017: loss=0.0006475497, psnr=31.887268, ssim=0.968296, lpips=unavailable
  image_018: loss=0.0006074004, psnr=32.165249, ssim=0.967270, lpips=unavailable
  image_019: loss=0.0005191733, psnr=32.846875, ssim=0.965417, lpips=unavailable
  image_020: loss=0.0003733727, psnr=34.278574, ssim=0.971390, lpips=unavailable
  image_021: loss=0.0005757192, psnr=32.397892, ssim=0.967798, lpips=unavailable
  image_022: loss=0.0007745444, psnr=31.109536, ssim=0.964048, lpips=unavailable
  image_023: loss=0.0007700801, psnr=31.134640, ssim=0.962728, lpips=unavailable
  image_024: loss=0.0006820545, psnr=31.661809, ssim=0.963113, lpips=unavailable

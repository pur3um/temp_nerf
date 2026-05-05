====================================================================================================
saved_time: 2026-05-02 15:13:41
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/lego.txt --expname lego_200k_swf5e-2 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 5e-2 --N_iters 200000
expname: lego_200k_swf5e-2
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 4 hour 4 min
current_train_loss: 0.0022022617
current_train_psnr: 34.490520
testset_mean_loss: 0.0005924702
testset_mean_psnr: 32.559146
testset_mean_ssim: 0.972220
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004293127, psnr=33.672261, ssim=0.976075, lpips=unavailable
  image_001: loss=0.0010671727, psnr=29.717652, ssim=0.964116, lpips=unavailable
  image_002: loss=0.0003437996, psnr=34.636944, ssim=0.977838, lpips=unavailable
  image_003: loss=0.0004192378, psnr=33.775395, ssim=0.976368, lpips=unavailable
  image_004: loss=0.0006188742, psnr=32.083975, ssim=0.970403, lpips=unavailable
  image_005: loss=0.0003837464, psnr=34.159556, ssim=0.975704, lpips=unavailable
  image_006: loss=0.0005698806, psnr=32.442161, ssim=0.974016, lpips=unavailable
  image_007: loss=0.0003703655, psnr=34.313694, ssim=0.974841, lpips=unavailable
  image_008: loss=0.0004209398, psnr=33.757799, ssim=0.978315, lpips=unavailable
  image_009: loss=0.0009434613, psnr=30.252759, ssim=0.972156, lpips=unavailable
  image_010: loss=0.0012940231, psnr=28.880579, ssim=0.959288, lpips=unavailable
  image_011: loss=0.0009442638, psnr=30.249066, ssim=0.961714, lpips=unavailable
  image_012: loss=0.0006008313, psnr=32.212474, ssim=0.971774, lpips=unavailable
  image_013: loss=0.0005917008, psnr=32.278978, ssim=0.971328, lpips=unavailable
  image_014: loss=0.0004752802, psnr=33.230502, ssim=0.971948, lpips=unavailable
  image_015: loss=0.0005542181, psnr=32.563192, ssim=0.973468, lpips=unavailable
  image_016: loss=0.0006043791, psnr=32.186905, ssim=0.979590, lpips=unavailable
  image_017: loss=0.0005729321, psnr=32.418968, ssim=0.972760, lpips=unavailable
  image_018: loss=0.0005274982, psnr=32.777789, ssim=0.973142, lpips=unavailable
  image_019: loss=0.0004855044, psnr=33.138067, ssim=0.970503, lpips=unavailable
  image_020: loss=0.0002795531, psnr=35.535355, ssim=0.978212, lpips=unavailable
  image_021: loss=0.0004473353, psnr=33.493667, ssim=0.975004, lpips=unavailable
  image_022: loss=0.0006381015, psnr=31.951101, ssim=0.970291, lpips=unavailable
  image_023: loss=0.0006590115, psnr=31.811069, ssim=0.968001, lpips=unavailable
  image_024: loss=0.0005703308, psnr=32.438731, ssim=0.968645, lpips=unavailable

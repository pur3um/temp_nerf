====================================================================================================
saved_time: 2026-05-07 01:45:25
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/chair.txt --datadir ./data/nerf_synthetic/chair --dataset_type blender --basedir logs/scheduler_compare/chair --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-muon --train_scheduler exp_decay --expname chair_muon_exp_decay
expname: chair_muon_exp_decay
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 12 hour 54 min
current_train_loss: 0.0011710861
current_train_psnr: 34.453590
testset_mean_loss: 0.0004971679
testset_mean_psnr: 33.240589
testset_mean_ssim: 0.972482
testset_mean_lpips: 0.023441
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001881568, psnr=37.254798, ssim=0.990490, lpips=0.008835
  image_001: loss=0.0002561807, psnr=35.914535, ssim=0.984835, lpips=0.010018
  image_002: loss=0.0005090333, psnr=32.932537, ssim=0.974454, lpips=0.022091
  image_003: loss=0.0005319930, psnr=32.740940, ssim=0.972636, lpips=0.025473
  image_004: loss=0.0005817812, psnr=32.352402, ssim=0.970558, lpips=0.025450
  image_005: loss=0.0006275955, psnr=32.023201, ssim=0.968687, lpips=0.023155
  image_006: loss=0.0006362059, psnr=31.964022, ssim=0.966029, lpips=0.024634
  image_007: loss=0.0006753579, psnr=31.704659, ssim=0.961436, lpips=0.031934
  image_008: loss=0.0005013074, psnr=32.998958, ssim=0.968572, lpips=0.024719
  image_009: loss=0.0005606021, psnr=32.513452, ssim=0.968385, lpips=0.023573
  image_010: loss=0.0007215677, psnr=31.417229, ssim=0.965776, lpips=0.030960
  image_011: loss=0.0004844507, psnr=33.147503, ssim=0.972592, lpips=0.027748
  image_012: loss=0.0003547378, psnr=34.500924, ssim=0.977047, lpips=0.026186
  image_013: loss=0.0004320869, psnr=33.644288, ssim=0.973359, lpips=0.029445
  image_014: loss=0.0003555556, psnr=34.490924, ssim=0.978349, lpips=0.021931
  image_015: loss=0.0005030161, psnr=32.984180, ssim=0.972725, lpips=0.026146
  image_016: loss=0.0004349328, psnr=33.615777, ssim=0.976721, lpips=0.020427
  image_017: loss=0.0006156844, psnr=32.106418, ssim=0.970190, lpips=0.024245
  image_018: loss=0.0006441287, psnr=31.910273, ssim=0.968091, lpips=0.024076
  image_019: loss=0.0006401986, psnr=31.936852, ssim=0.963829, lpips=0.026166
  image_020: loss=0.0005558400, psnr=32.550501, ssim=0.965505, lpips=0.029137
  image_021: loss=0.0004557677, psnr=33.412564, ssim=0.969942, lpips=0.024208
  image_022: loss=0.0004889535, psnr=33.107324, ssim=0.969984, lpips=0.025141
  image_023: loss=0.0004300244, psnr=33.665068, ssim=0.975819, lpips=0.019943
  image_024: loss=0.0002440390, psnr=36.125405, ssim=0.986037, lpips=0.010383

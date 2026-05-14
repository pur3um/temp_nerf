====================================================================================================
saved_time: 2026-05-07 02:13:54
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/lego.txt --datadir ./data/nerf_synthetic/lego --dataset_type blender --basedir logs/scheduler_compare/lego --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --lowrank_auto_init_rank_start --expname lego_autocosinc_rank_wsd
expname: lego_autocosinc_rank_wsd
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 13 hour 22 min
current_train_loss: 0.0027220822
current_train_psnr: 32.064713
testset_mean_loss: 0.0008662112
testset_mean_psnr: 30.871982
testset_mean_ssim: 0.957294
testset_mean_lpips: 0.026357
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0006392332, psnr=31.943406, ssim=0.963085, lpips=0.024411
  image_001: loss=0.0017938350, psnr=27.462175, ssim=0.948998, lpips=0.032409
  image_002: loss=0.0004824848, psnr=33.165162, ssim=0.968396, lpips=0.018760
  image_003: loss=0.0005888739, psnr=32.299776, ssim=0.965269, lpips=0.024359
  image_004: loss=0.0008963943, psnr=30.475009, ssim=0.957461, lpips=0.027791
  image_005: loss=0.0006492738, psnr=31.875721, ssim=0.957056, lpips=0.026498
  image_006: loss=0.0009122270, psnr=30.398970, ssim=0.951237, lpips=0.030687
  image_007: loss=0.0005990359, psnr=32.225471, ssim=0.952295, lpips=0.024055
  image_008: loss=0.0006155928, psnr=32.107064, ssim=0.965168, lpips=0.019901
  image_009: loss=0.0012208050, psnr=29.133537, ssim=0.960642, lpips=0.031247
  image_010: loss=0.0016997503, psnr=27.696148, ssim=0.945871, lpips=0.038311
  image_011: loss=0.0012852898, psnr=28.909989, ssim=0.948456, lpips=0.033424
  image_012: loss=0.0008580266, psnr=30.664992, ssim=0.959610, lpips=0.025487
  image_013: loss=0.0008734901, psnr=30.587420, ssim=0.958063, lpips=0.025036
  image_014: loss=0.0006663396, psnr=31.763043, ssim=0.959079, lpips=0.021167
  image_015: loss=0.0008028148, psnr=30.953846, ssim=0.960418, lpips=0.027969
  image_016: loss=0.0008419266, psnr=30.747257, ssim=0.970346, lpips=0.024226
  image_017: loss=0.0008211979, psnr=30.855521, ssim=0.960367, lpips=0.027869
  image_018: loss=0.0008518454, psnr=30.696392, ssim=0.952021, lpips=0.027753
  image_019: loss=0.0007105808, psnr=31.483865, ssim=0.946647, lpips=0.026910
  image_020: loss=0.0004831560, psnr=33.159125, ssim=0.957252, lpips=0.022049
  image_021: loss=0.0006606897, psnr=31.800024, ssim=0.959663, lpips=0.020015
  image_022: loss=0.0009320943, psnr=30.305401, ssim=0.955334, lpips=0.027992
  image_023: loss=0.0009594334, psnr=30.179851, ssim=0.953682, lpips=0.027656
  image_024: loss=0.0008108894, psnr=30.910383, ssim=0.955931, lpips=0.022939

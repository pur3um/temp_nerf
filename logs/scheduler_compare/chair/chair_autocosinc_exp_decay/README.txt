====================================================================================================
saved_time: 2026-05-07 02:07:33
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/chair.txt --datadir ./data/nerf_synthetic/chair --dataset_type blender --basedir logs/scheduler_compare/chair --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-sign-auto-cos-inc --train_scheduler exp_decay --lowrank_auto_init_rank_start --expname chair_autocosinc_exp_decay
expname: chair_autocosinc_exp_decay
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 13 hour 16 min
current_train_loss: 0.0011409390
current_train_psnr: 34.726257
testset_mean_loss: 0.0005024586
testset_mean_psnr: 33.198467
testset_mean_ssim: 0.972277
testset_mean_lpips: 0.023401
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001914620, psnr=37.179171, ssim=0.990414, lpips=0.009123
  image_001: loss=0.0002588778, psnr=35.869049, ssim=0.984686, lpips=0.010252
  image_002: loss=0.0005211708, psnr=32.830198, ssim=0.974411, lpips=0.022490
  image_003: loss=0.0005387017, psnr=32.686516, ssim=0.972207, lpips=0.024601
  image_004: loss=0.0005849782, psnr=32.328602, ssim=0.970304, lpips=0.026273
  image_005: loss=0.0006319912, psnr=31.992889, ssim=0.968510, lpips=0.023834
  image_006: loss=0.0006424831, psnr=31.921382, ssim=0.965892, lpips=0.023985
  image_007: loss=0.0006998436, psnr=31.549990, ssim=0.960517, lpips=0.031149
  image_008: loss=0.0005160050, psnr=32.873460, ssim=0.968028, lpips=0.026325
  image_009: loss=0.0005619602, psnr=32.502944, ssim=0.968263, lpips=0.022748
  image_010: loss=0.0007031037, psnr=31.529806, ssim=0.966112, lpips=0.031749
  image_011: loss=0.0004734994, psnr=33.246805, ssim=0.973935, lpips=0.025733
  image_012: loss=0.0003517291, psnr=34.537915, ssim=0.977379, lpips=0.024277
  image_013: loss=0.0004246018, psnr=33.720181, ssim=0.973239, lpips=0.029417
  image_014: loss=0.0003531954, psnr=34.519849, ssim=0.978250, lpips=0.021983
  image_015: loss=0.0005050507, psnr=32.966649, ssim=0.972423, lpips=0.026960
  image_016: loss=0.0004384463, psnr=33.580835, ssim=0.976759, lpips=0.020060
  image_017: loss=0.0006209332, psnr=32.069550, ssim=0.970069, lpips=0.024665
  image_018: loss=0.0006595930, psnr=31.807239, ssim=0.967838, lpips=0.024509
  image_019: loss=0.0006602768, psnr=31.802739, ssim=0.962949, lpips=0.025692
  image_020: loss=0.0005855671, psnr=32.324232, ssim=0.964421, lpips=0.027889
  image_021: loss=0.0004646336, psnr=33.328893, ssim=0.969375, lpips=0.025936
  image_022: loss=0.0004899241, psnr=33.098711, ssim=0.969573, lpips=0.026131
  image_023: loss=0.0004390492, psnr=33.574867, ssim=0.975327, lpips=0.019070
  image_024: loss=0.0002443872, psnr=36.119213, ssim=0.986045, lpips=0.010173

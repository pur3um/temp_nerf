====================================================================================================
saved_time: 2026-05-07 02:13:42
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/chair.txt --datadir ./data/nerf_synthetic/chair --dataset_type blender --basedir logs/scheduler_compare/chair --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-sign-auto-cos-inc --train_scheduler rank_wsd --lowrank_auto_init_rank_start --expname chair_autocosinc_rank_wsd
expname: chair_autocosinc_rank_wsd
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 13 hour 22 min
current_train_loss: 0.0010267638
current_train_psnr: 35.429062
testset_mean_loss: 0.0004503299
testset_mean_psnr: 33.671111
testset_mean_ssim: 0.974952
testset_mean_lpips: 0.019693
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001679615, psnr=37.747900, ssim=0.991280, lpips=0.008241
  image_001: loss=0.0002287765, psnr=36.405885, ssim=0.986450, lpips=0.008260
  image_002: loss=0.0004575838, psnr=33.395292, ssim=0.977148, lpips=0.019819
  image_003: loss=0.0004792725, psnr=33.194174, ssim=0.975175, lpips=0.019851
  image_004: loss=0.0005178143, psnr=32.858259, ssim=0.973760, lpips=0.020363
  image_005: loss=0.0005587398, psnr=32.527903, ssim=0.972140, lpips=0.017916
  image_006: loss=0.0005669581, psnr=32.464490, ssim=0.969583, lpips=0.020157
  image_007: loss=0.0006046228, psnr=32.185154, ssim=0.965308, lpips=0.027347
  image_008: loss=0.0004601628, psnr=33.370884, ssim=0.971213, lpips=0.022030
  image_009: loss=0.0005037884, psnr=32.977518, ssim=0.970326, lpips=0.021794
  image_010: loss=0.0006697329, psnr=31.740983, ssim=0.967886, lpips=0.028583
  image_011: loss=0.0004516102, psnr=33.452362, ssim=0.974539, lpips=0.023894
  image_012: loss=0.0003235647, psnr=34.900388, ssim=0.979433, lpips=0.021287
  image_013: loss=0.0004115725, psnr=33.855535, ssim=0.975873, lpips=0.023040
  image_014: loss=0.0003196964, psnr=34.952621, ssim=0.980656, lpips=0.016916
  image_015: loss=0.0004679085, psnr=33.298390, ssim=0.975046, lpips=0.022696
  image_016: loss=0.0003900302, psnr=34.089017, ssim=0.979017, lpips=0.017394
  image_017: loss=0.0005501599, psnr=32.595110, ssim=0.973460, lpips=0.021559
  image_018: loss=0.0005806003, psnr=32.361226, ssim=0.971584, lpips=0.019780
  image_019: loss=0.0005791381, psnr=32.372178, ssim=0.966994, lpips=0.022451
  image_020: loss=0.0005067702, psnr=32.951889, ssim=0.968543, lpips=0.022674
  image_021: loss=0.0004094463, psnr=33.878030, ssim=0.971873, lpips=0.019739
  image_022: loss=0.0004396738, psnr=33.568693, ssim=0.971886, lpips=0.020779
  image_023: loss=0.0003924710, psnr=34.061923, ssim=0.977305, lpips=0.016676
  image_024: loss=0.0002201925, psnr=36.571973, ssim=0.987314, lpips=0.009072

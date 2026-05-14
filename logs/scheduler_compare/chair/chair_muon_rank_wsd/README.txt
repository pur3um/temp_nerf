====================================================================================================
saved_time: 2026-05-07 01:42:00
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/chair.txt --datadir ./data/nerf_synthetic/chair --dataset_type blender --basedir logs/scheduler_compare/chair --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-muon --train_scheduler rank_wsd --rank_wsd_fullrank_decay_start_frac 0.2 --rank_wsd_fullrank_decay_end_frac 0.6 --expname chair_muon_rank_wsd
expname: chair_muon_rank_wsd
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 12 hour 50 min
current_train_loss: 0.0011416988
current_train_psnr: 34.810188
testset_mean_loss: 0.0004776756
testset_mean_psnr: 33.420318
testset_mean_ssim: 0.973448
testset_mean_lpips: 0.021954
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001760530, psnr=37.543562, ssim=0.991083, lpips=0.008046
  image_001: loss=0.0002410111, psnr=36.179628, ssim=0.985553, lpips=0.009443
  image_002: loss=0.0004874876, psnr=33.120363, ssim=0.975684, lpips=0.021931
  image_003: loss=0.0005047590, psnr=32.969159, ssim=0.973883, lpips=0.021873
  image_004: loss=0.0005485817, psnr=32.607586, ssim=0.971845, lpips=0.023891
  image_005: loss=0.0006016267, psnr=32.206728, ssim=0.969488, lpips=0.021288
  image_006: loss=0.0006088955, psnr=32.154572, ssim=0.966907, lpips=0.024429
  image_007: loss=0.0006488815, psnr=31.878345, ssim=0.962710, lpips=0.029601
  image_008: loss=0.0004836062, psnr=33.155081, ssim=0.969583, lpips=0.023221
  image_009: loss=0.0005346194, psnr=32.719552, ssim=0.969112, lpips=0.023716
  image_010: loss=0.0006927941, psnr=31.593958, ssim=0.967207, lpips=0.031247
  image_011: loss=0.0004813946, psnr=33.174987, ssim=0.973491, lpips=0.025933
  image_012: loss=0.0003415594, psnr=34.665336, ssim=0.978715, lpips=0.022110
  image_013: loss=0.0004207995, psnr=33.759246, ssim=0.974133, lpips=0.027274
  image_014: loss=0.0003428647, psnr=34.648771, ssim=0.979271, lpips=0.019871
  image_015: loss=0.0004983008, psnr=33.025084, ssim=0.973988, lpips=0.025318
  image_016: loss=0.0004132222, psnr=33.838162, ssim=0.977707, lpips=0.019112
  image_017: loss=0.0005904825, psnr=32.287929, ssim=0.971274, lpips=0.023783
  image_018: loss=0.0006216405, psnr=32.064606, ssim=0.968910, lpips=0.024057
  image_019: loss=0.0006249323, psnr=32.041670, ssim=0.964499, lpips=0.025831
  image_020: loss=0.0005360938, psnr=32.707591, ssim=0.966416, lpips=0.024434
  image_021: loss=0.0004306874, psnr=33.658378, ssim=0.970579, lpips=0.022887
  image_022: loss=0.0004700181, psnr=33.278854, ssim=0.970741, lpips=0.023058
  image_023: loss=0.0004104449, psnr=33.867450, ssim=0.976683, lpips=0.017344
  image_024: loss=0.0002311344, psnr=36.361353, ssim=0.986746, lpips=0.009161

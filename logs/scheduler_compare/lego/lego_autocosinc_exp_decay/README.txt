====================================================================================================
saved_time: 2026-05-07 02:10:27
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/lego.txt --datadir ./data/nerf_synthetic/lego --dataset_type blender --basedir logs/scheduler_compare/lego --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-sign-auto-cos-inc --train_scheduler exp_decay --lowrank_auto_init_rank_start --expname lego_autocosinc_exp_decay
expname: lego_autocosinc_exp_decay
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 13 hour 19 min
current_train_loss: 0.0031158361
current_train_psnr: 31.250177
testset_mean_loss: 0.0009716296
testset_mean_psnr: 30.353412
testset_mean_ssim: 0.951946
testset_mean_lpips: 0.030575
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0007285252, psnr=31.375554, ssim=0.959108, lpips=0.028020
  image_001: loss=0.0019089160, psnr=27.192131, ssim=0.944653, lpips=0.035394
  image_002: loss=0.0005543527, psnr=32.562137, ssim=0.964529, lpips=0.022692
  image_003: loss=0.0006562087, psnr=31.829580, ssim=0.962125, lpips=0.026963
  image_004: loss=0.0009747270, psnr=30.111170, ssim=0.953359, lpips=0.032014
  image_005: loss=0.0007712989, psnr=31.127772, ssim=0.949138, lpips=0.032193
  image_006: loss=0.0010198668, psnr=29.914565, ssim=0.943605, lpips=0.036248
  image_007: loss=0.0006873034, psnr=31.628514, ssim=0.942531, lpips=0.028518
  image_008: loss=0.0006993545, psnr=31.553026, ssim=0.959696, lpips=0.022832
  image_009: loss=0.0014240182, psnr=28.464844, ssim=0.954987, lpips=0.036789
  image_010: loss=0.0018924114, psnr=27.229844, ssim=0.939964, lpips=0.046374
  image_011: loss=0.0013870191, psnr=28.579175, ssim=0.945013, lpips=0.038760
  image_012: loss=0.0009589356, psnr=30.182105, ssim=0.954541, lpips=0.028161
  image_013: loss=0.0009762326, psnr=30.104467, ssim=0.953448, lpips=0.030483
  image_014: loss=0.0007356289, psnr=31.333411, ssim=0.955452, lpips=0.024491
  image_015: loss=0.0009048817, psnr=30.434081, ssim=0.956200, lpips=0.030134
  image_016: loss=0.0010011728, psnr=29.994909, ssim=0.965362, lpips=0.029543
  image_017: loss=0.0009519223, psnr=30.213985, ssim=0.955537, lpips=0.031110
  image_018: loss=0.0009744728, psnr=30.112302, ssim=0.945044, lpips=0.031715
  image_019: loss=0.0007966613, psnr=30.987262, ssim=0.938414, lpips=0.032752
  image_020: loss=0.0005641579, psnr=32.485992, ssim=0.949563, lpips=0.027555
  image_021: loss=0.0007387209, psnr=31.315195, ssim=0.954802, lpips=0.023030
  image_022: loss=0.0010346720, psnr=29.851973, ssim=0.950415, lpips=0.031079
  image_023: loss=0.0010530241, psnr=29.775617, ssim=0.949509, lpips=0.031934
  image_024: loss=0.0008962542, psnr=30.475687, ssim=0.951668, lpips=0.025595

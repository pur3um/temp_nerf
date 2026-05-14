====================================================================================================
saved_time: 2026-05-07 01:53:20
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/lego.txt --datadir ./data/nerf_synthetic/lego --dataset_type blender --basedir logs/scheduler_compare/lego --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-muon --train_scheduler rank_wsd --rank_wsd_fullrank_decay_start_frac 0.2 --rank_wsd_fullrank_decay_end_frac 0.6 --expname lego_muon_rank_wsd
expname: lego_muon_rank_wsd
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 13 hour 1 min
current_train_loss: 0.0029884758
current_train_psnr: 31.175251
testset_mean_loss: 0.0009208433
testset_mean_psnr: 30.592637
testset_mean_ssim: 0.954345
testset_mean_lpips: 0.028964
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0006965251, psnr=31.570632, ssim=0.960234, lpips=0.027326
  image_001: loss=0.0018918151, psnr=27.231213, ssim=0.945589, lpips=0.035659
  image_002: loss=0.0005190850, psnr=32.847614, ssim=0.966013, lpips=0.021802
  image_003: loss=0.0006318248, psnr=31.994033, ssim=0.962975, lpips=0.026959
  image_004: loss=0.0009523772, psnr=30.211910, ssim=0.954938, lpips=0.031733
  image_005: loss=0.0007068138, psnr=31.506949, ssim=0.953243, lpips=0.029321
  image_006: loss=0.0009674722, psnr=30.143615, ssim=0.947688, lpips=0.032371
  image_007: loss=0.0006484257, psnr=31.881397, ssim=0.947084, lpips=0.027063
  image_008: loss=0.0006560785, psnr=31.830442, ssim=0.962813, lpips=0.021689
  image_009: loss=0.0013044517, psnr=28.845720, ssim=0.957955, lpips=0.032487
  image_010: loss=0.0017366111, psnr=27.602974, ssim=0.943987, lpips=0.044052
  image_011: loss=0.0013854328, psnr=28.584145, ssim=0.945644, lpips=0.035149
  image_012: loss=0.0009081793, psnr=30.418283, ssim=0.956678, lpips=0.028298
  image_013: loss=0.0009362223, psnr=30.286210, ssim=0.955130, lpips=0.029006
  image_014: loss=0.0007117590, psnr=31.476670, ssim=0.956902, lpips=0.023962
  image_015: loss=0.0008195784, psnr=30.864095, ssim=0.959251, lpips=0.028451
  image_016: loss=0.0008928666, psnr=30.492134, ssim=0.968371, lpips=0.026860
  image_017: loss=0.0008841698, psnr=30.534643, ssim=0.957376, lpips=0.029832
  image_018: loss=0.0009175396, psnr=30.373751, ssim=0.949208, lpips=0.029333
  image_019: loss=0.0007557090, psnr=31.216453, ssim=0.941804, lpips=0.030811
  image_020: loss=0.0005329773, psnr=32.732912, ssim=0.952590, lpips=0.025209
  image_021: loss=0.0006965438, psnr=31.570515, ssim=0.957447, lpips=0.021878
  image_022: loss=0.0009835677, psnr=30.071957, ssim=0.952739, lpips=0.030541
  image_023: loss=0.0009951434, psnr=30.021143, ssim=0.950788, lpips=0.029309
  image_024: loss=0.0008899124, psnr=30.506527, ssim=0.952187, lpips=0.024999

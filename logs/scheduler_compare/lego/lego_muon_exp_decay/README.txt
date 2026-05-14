====================================================================================================
saved_time: 2026-05-07 01:48:25
script_path: /data2/greenx9/temp_nerf/99_run_nerf.py
executed_command: 99_run_nerf.py --config configs/lego.txt --datadir ./data/nerf_synthetic/lego --dataset_type blender --basedir logs/scheduler_compare/lego --no_reload --seed 0 --deterministic --N_iters 200000 --lrate 5e-4 --muon_lrate 5e-4 --lrate_decay 250 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_schedule_steps 100000 --lowrank_oversample 4 --lowrank_ns_steps 5 --sched_min_lr_ratio 0.1 --sched_warmup_frac 0.01 --i_print 2000 --i_weights 100000 --i_testset 100000 --i_video 100000 --i_valset 0 --white_bkgd --half_res --N_rand 2048 --optimizer aux-muon --train_scheduler exp_decay --expname lego_muon_exp_decay
expname: lego_muon_exp_decay
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 12 hour 57 min
current_train_loss: 0.0031893132
current_train_psnr: 31.141722
testset_mean_loss: 0.0009694778
testset_mean_psnr: 30.372834
testset_mean_ssim: 0.952493
testset_mean_lpips: 0.030002
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0007441420, psnr=31.283441, ssim=0.959076, lpips=0.026904
  image_001: loss=0.0020312460, psnr=26.922375, ssim=0.944435, lpips=0.036312
  image_002: loss=0.0005411757, psnr=32.666617, ssim=0.964361, lpips=0.021969
  image_003: loss=0.0006574702, psnr=31.821239, ssim=0.961735, lpips=0.026873
  image_004: loss=0.0009472882, psnr=30.235178, ssim=0.953672, lpips=0.031994
  image_005: loss=0.0007609605, psnr=31.186378, ssim=0.949747, lpips=0.032590
  image_006: loss=0.0010188923, psnr=29.918717, ssim=0.944559, lpips=0.034199
  image_007: loss=0.0006905873, psnr=31.607814, ssim=0.943522, lpips=0.028827
  image_008: loss=0.0006919773, psnr=31.599081, ssim=0.960354, lpips=0.022673
  image_009: loss=0.0014024255, psnr=28.531202, ssim=0.956133, lpips=0.035620
  image_010: loss=0.0018694071, psnr=27.282961, ssim=0.942026, lpips=0.045055
  image_011: loss=0.0013932849, psnr=28.559600, ssim=0.945683, lpips=0.036010
  image_012: loss=0.0009545778, psnr=30.201886, ssim=0.956314, lpips=0.027685
  image_013: loss=0.0009722185, psnr=30.122361, ssim=0.954016, lpips=0.029412
  image_014: loss=0.0007558381, psnr=31.215712, ssim=0.955095, lpips=0.025067
  image_015: loss=0.0008778857, psnr=30.565620, ssim=0.957457, lpips=0.028993
  image_016: loss=0.0009532954, psnr=30.207724, ssim=0.966934, lpips=0.026715
  image_017: loss=0.0009213529, psnr=30.355739, ssim=0.956111, lpips=0.030448
  image_018: loss=0.0009438758, psnr=30.250851, ssim=0.946743, lpips=0.031413
  image_019: loss=0.0008116568, psnr=30.906275, ssim=0.938667, lpips=0.031966
  image_020: loss=0.0005602429, psnr=32.516236, ssim=0.949718, lpips=0.026907
  image_021: loss=0.0007361454, psnr=31.330364, ssim=0.954819, lpips=0.024005
  image_022: loss=0.0010380363, psnr=29.837874, ssim=0.950351, lpips=0.031006
  image_023: loss=0.0010671838, psnr=29.717607, ssim=0.949000, lpips=0.031617
  image_024: loss=0.0008957784, psnr=30.477994, ssim=0.951808, lpips=0.025801

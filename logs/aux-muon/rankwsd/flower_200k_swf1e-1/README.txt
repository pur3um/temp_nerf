====================================================================================================
saved_time: 2026-05-02 18:33:19
script_path: /workspace/temp_nerf/00_run_nerf_ranksched_final.py
executed_command: 00_run_nerf_ranksched_final.py --basedir logs/aux-muon/rankwsd --config configs/flower.txt --expname flower_200k_swf1e-1 --optimizer aux-muon --train_scheduler rank_wsd --muon_lrate 3e-3 --sched_warmup_frac 1e-1 --N_iters 200000
expname: flower_200k_swf1e-1
iter: 200000
global_step: 199999
elapsed_time_from_train_start: 3 hour 18 min
current_train_loss: 0.0022038806
current_train_psnr: 30.806890
testset_mean_loss: 0.0013508191
testset_mean_psnr: 28.933051
testset_mean_ssim: 0.898717
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0012740784, psnr=28.948038, ssim=0.887968, lpips=unavailable
  image_001: loss=0.0013920987, psnr=28.563299, ssim=0.905325, lpips=unavailable
  image_002: loss=0.0019896615, psnr=27.012208, ssim=0.861848, lpips=unavailable
  image_003: loss=0.0014135106, psnr=28.497009, ssim=0.910060, lpips=unavailable
  image_004: loss=0.0006847464, psnr=31.644702, ssim=0.928384, lpips=unavailable

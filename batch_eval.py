import os
import subprocess

# synth: chair, drums, ficus, hotdog, lego, materials, mic, ship
base_path = "/home/greenx9/data2/temp_nerf/logs/mofa"
for folder_name in os.listdir(base_path):
    absolute_path = os.path.join(base_path, folder_name)
    subprocess.call(f'CUDA_VISIBLE_DEVICES=4 python eval_nerf_checkpoint_safe.py \
            --config {absolute_path}/config.txt \
            --checkpoint {absolute_path}/100000.tar \
            --out_dir {absolute_path}/eval_ckpt_100000 \
            --project_root . \
            --nerf_module run_mofa \
            --device cuda \
            --ssim_impl torch \
            --no_save_png \
            --overwrite',
            shell=True)

"""
python eval_nerf_checkpoint_safe.py \
  --config ./logs/mofa/fern_100k_mflr3e-3/config.txt \
  --checkpoint ./logs/mofa/fern_100k_mflr3e-3/100000.tar \
  --out_dir ./logs/mofa/fern_100k_mflr3e-3/eval_ckpt_100000 \
  --project_root . \
  --nerf_module run_mofa \
  --device cuda \
  --ssim_impl torch \
  --no_save_png \
  --overwrite
"""
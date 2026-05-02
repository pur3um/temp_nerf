# 00_run_nerf_ranksched_final.py
#!/usr/bin/env python3
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as skimage_structural_similarity
except ImportError:
    skimage_structural_similarity = None

try:
    import lpips
except ImportError:
    lpips = None

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

#//========== Optim / Scheduler ================
def parse_pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected 2 values for betas, got {value}")
        return float(value[0]), float(value[1])
    parts = [p.strip() for p in str(value).split(',')]
    if len(parts) != 2:
        raise ValueError(f"Expected a comma-separated pair like '0.9,0.95', got {value!r}")
    return float(parts[0]), float(parts[1])


from optims.muon import SingleDeviceMuonWithAuxAdam
from optims.lr_sign10_rsclF import SingleDeviceSign10RsclFWithAuxAdam
from optims.auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam
from optims.rank_wsd_scheduler import RankAwareWarmupStableLinearScheduler
from optims.warmup_cosine_scheduler import WarmupCosineScheduler
#//=================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def seed_everything(seed: int = 0, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / CUDA deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch에서 가능한 deterministic 알고리즘 강제
        torch.use_deterministic_algorithms(True, warn_only=True)

        # CUDA matmul 재현성 강화
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def _to_numpy_image(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    return np.asarray(img)

_LPIPS_MODEL_CACHE = {}

def _sanitize_image_for_metrics(img):
    img = _to_numpy_image(img)[..., :3].astype(np.float32)
    return np.clip(img, 0.0, 1.0)

def _image_to_torch_chw(img, target_device=None, value_range='0_1'):
    if target_device is None:
        target_device = device
    img = _sanitize_image_for_metrics(img)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(target_device)
    if value_range == 'minus1_1':
        tensor = tensor * 2.0 - 1.0
    return tensor

def _build_ssim_window(window_size=11, sigma=1.5, channels=3, target_device=None):
    if target_device is None:
        target_device = device
    coords = torch.arange(window_size, dtype=torch.float32, device=target_device) - (window_size // 2)
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_2d = torch.outer(gauss, gauss)
    window = window_2d.unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()

def _compute_ssim_torch(pred_img, gt_img, window_size=11, sigma=1.5):
    pred = _image_to_torch_chw(pred_img, target_device=torch.device('cpu'), value_range='0_1')
    gt = _image_to_torch_chw(gt_img, target_device=torch.device('cpu'), value_range='0_1')

    channels = pred.shape[1]
    window = _build_ssim_window(window_size=window_size, sigma=sigma, channels=channels, target_device=pred.device)

    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_gt = F.conv2d(gt, window, padding=window_size // 2, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channels) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channels) - mu_pred_gt

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)
    denominator = (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())

def compute_ssim(pred_img, gt_img):
    pred_img = _sanitize_image_for_metrics(pred_img)
    gt_img = _sanitize_image_for_metrics(gt_img)

    if pred_img.shape != gt_img.shape:
        raise ValueError(f'SSIM shape mismatch: pred={pred_img.shape}, gt={gt_img.shape}')

    if skimage_structural_similarity is not None:
        try:
            return float(skimage_structural_similarity(gt_img, pred_img, channel_axis=-1, data_range=1.0))
        except TypeError:
            return float(skimage_structural_similarity(gt_img, pred_img, multichannel=True, data_range=1.0))

    return _compute_ssim_torch(pred_img, gt_img)

def get_lpips_model(net='alex'):
    cache_key = (net, device.type)
    if cache_key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[cache_key]

    if lpips is None:
        raise ImportError(
            'LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. '
            'If torchvision pretrained weights are not cached, the first run may also download the backbone weights.'
        )

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    _LPIPS_MODEL_CACHE[cache_key] = model
    return model

def compute_lpips(pred_img, gt_img, net='alex'):
    pred = _image_to_torch_chw(pred_img, target_device=device, value_range='minus1_1')
    gt = _image_to_torch_chw(gt_img, target_device=device, value_range='minus1_1')
    lpips_model = get_lpips_model(net=net)

    with torch.no_grad():
        score = lpips_model(pred, gt)

    return float(score.mean().item())

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path_with_metrics(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, lpips_net='alex'):
    rgbs, disps = render_path(
        render_poses, hwf, K, chunk, render_kwargs,
        gt_imgs=gt_imgs, savedir=savedir, render_factor=render_factor
    )

    metrics = {
        'mse_per_image': [],
        'psnr_per_image': [],
        'ssim_per_image': [],
        'lpips_per_image': [],
        'mean_mse': None,
        'mean_psnr': None,
        'mean_ssim': None,
        'mean_lpips': None,
        'lpips_status': 'not_requested',
        'lpips_error': None,
        'lpips_net': lpips_net,
    }

    if gt_imgs is not None and render_factor == 0:
        lpips_available = True
        lpips_failed = False

        for pred_img, gt_img in zip(rgbs, gt_imgs):
            pred_img = _sanitize_image_for_metrics(pred_img)
            gt_img = _sanitize_image_for_metrics(gt_img)

            mse = float(np.mean(np.square(pred_img - gt_img)))
            psnr = float(-10.0 * np.log10(mse + 1e-10))
            ssim = compute_ssim(pred_img, gt_img)

            if lpips_available:
                try:
                    lpips_score = compute_lpips(pred_img, gt_img, net=lpips_net)
                except Exception as e:
                    lpips_available = False
                    lpips_failed = True
                    lpips_score = None
                    metrics['lpips_status'] = 'unavailable'
                    metrics['lpips_error'] = str(e)
                    print(f'[WARN] LPIPS computation skipped: {e}')
            else:
                lpips_score = None

            metrics['mse_per_image'].append(mse)
            metrics['psnr_per_image'].append(psnr)
            metrics['ssim_per_image'].append(ssim)
            metrics['lpips_per_image'].append(lpips_score)

        if len(metrics['mse_per_image']) > 0:
            metrics['mean_mse'] = float(np.mean(metrics['mse_per_image']))
            metrics['mean_psnr'] = float(np.mean(metrics['psnr_per_image']))
            metrics['mean_ssim'] = float(np.mean(metrics['ssim_per_image']))

            valid_lpips = [v for v in metrics['lpips_per_image'] if v is not None]
            if len(valid_lpips) > 0:
                metrics['mean_lpips'] = float(np.mean(valid_lpips))
                if lpips_failed:
                    metrics['lpips_status'] = 'partial'
                else:
                    metrics['lpips_status'] = 'ok'
            elif metrics['lpips_status'] == 'not_requested':
                metrics['lpips_status'] = 'unavailable'
                metrics['lpips_error'] = 'LPIPS scores were not computed.'

    return rgbs, disps, metrics


def render_path_mean_psnr(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, render_factor=0):
    """Render a pose set and compute mean PSNR only.

    This is intentionally lighter than render_path_with_metrics for HPO validation:
    it does not compute SSIM or LPIPS and does not save images.
    """
    rgbs, _ = render_path(
        render_poses, hwf, K, chunk, render_kwargs,
        gt_imgs=None, savedir=None, render_factor=render_factor
    )

    metrics = {
        'mse_per_image': [],
        'psnr_per_image': [],
        'mean_mse': None,
        'mean_psnr': None,
    }

    if gt_imgs is None or render_factor != 0:
        return metrics

    for pred_img, gt_img in zip(rgbs, gt_imgs):
        pred_img = _sanitize_image_for_metrics(pred_img)
        gt_img = _sanitize_image_for_metrics(gt_img)
        mse = float(np.mean(np.square(pred_img - gt_img)))
        psnr = float(-10.0 * np.log10(mse + 1e-10))
        metrics['mse_per_image'].append(mse)
        metrics['psnr_per_image'].append(psnr)

    if len(metrics['psnr_per_image']) > 0:
        metrics['mean_mse'] = float(np.mean(metrics['mse_per_image']))
        metrics['mean_psnr'] = float(np.mean(metrics['psnr_per_image']))

    return metrics


def format_elapsed_time_hm(elapsed_seconds):
    elapsed_seconds = int(elapsed_seconds)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    return hours, minutes

def save_testset_metrics(report_path, args, i, global_step, loss, psnr, elapsed_seconds, test_metrics=None):
    elapsed_hours, elapsed_minutes = format_elapsed_time_hm(elapsed_seconds)
    current_loss = float(loss.item()) if torch.is_tensor(loss) else float(loss)
    current_psnr = float(psnr.item()) if torch.is_tensor(psnr) else float(psnr)
    executed_command = ' '.join(sys.argv)

    with open(report_path, 'w') as f:
        f.write('=' * 100 + '\n')
        f.write(f"saved_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"script_path: {os.path.abspath(sys.argv[0])}\n")
        f.write(f"executed_command: {executed_command}\n")
        f.write(f"expname: {args.expname}\n")
        f.write(f"iter: {i}\n")
        f.write(f"global_step: {global_step}\n")
        f.write(f"elapsed_time_from_train_start: {elapsed_hours} hour {elapsed_minutes} min\n")
        f.write(f"current_train_loss: {current_loss:.10f}\n")
        f.write(f"current_train_psnr: {current_psnr:.6f}\n")

        if test_metrics is not None and len(test_metrics.get('psnr_per_image', [])) > 0:
            f.write(f"testset_mean_loss: {test_metrics['mean_mse']:.10f}\n")
            f.write(f"testset_mean_psnr: {test_metrics['mean_psnr']:.6f}\n")
            f.write(f"testset_mean_ssim: {test_metrics['mean_ssim']:.6f}\n")

            if test_metrics.get('mean_lpips') is not None:
                f.write(f"testset_mean_lpips: {test_metrics['mean_lpips']:.6f}\n")
            else:
                f.write('testset_mean_lpips: unavailable\n')

            if test_metrics.get('lpips_net') is not None:
                f.write(f"testset_lpips_net: {test_metrics['lpips_net']}\n")
            if test_metrics.get('lpips_status') is not None:
                f.write(f"testset_lpips_status: {test_metrics['lpips_status']}\n")
            if test_metrics.get('lpips_error'):
                f.write(f"testset_lpips_error: {test_metrics['lpips_error']}\n")

            f.write('testset_metrics_per_image:\n')
            for idx, (mse_val, psnr_val, ssim_val, lpips_val) in enumerate(zip(
                test_metrics['mse_per_image'],
                test_metrics['psnr_per_image'],
                test_metrics['ssim_per_image'],
                test_metrics['lpips_per_image'],
            )):
                lpips_str = f'{lpips_val:.6f}' if lpips_val is not None else 'unavailable'
                f.write(
                    f"  image_{idx:03d}: loss={mse_val:.10f}, psnr={psnr_val:.6f}, "
                    f"ssim={ssim_val:.6f}, lpips={lpips_str}\n"
                )


def save_testset_metrics_to_readmes(report_dir, args, i, global_step, loss, psnr, elapsed_seconds, test_metrics=None):
    """Save test metrics to README.txt and readme.txt for compatibility."""
    os.makedirs(report_dir, exist_ok=True)
    main_report_path = os.path.join(report_dir, 'README.txt')
    legacy_report_path = os.path.join(report_dir, 'readme.txt')

    save_testset_metrics(
        report_path=main_report_path,
        args=args,
        i=i,
        global_step=global_step,
        loss=loss,
        psnr=psnr,
        elapsed_seconds=elapsed_seconds,
        test_metrics=test_metrics,
    )

    # Keep the historical lowercase filename too, but make README.txt the primary report.
    if legacy_report_path != main_report_path:
        with open(main_report_path, 'r') as src, open(legacy_report_path, 'w') as dst:
            dst.write(src.read())

    return main_report_path


def _json_safe_metrics(metrics):
    """Convert metric payload to plain Python values suitable for json.dump."""
    if metrics is None:
        return {}
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            safe[key] = value.item()
        elif isinstance(value, list):
            converted = []
            for item in value:
                if isinstance(item, (np.floating, np.integer)):
                    converted.append(item.item())
                else:
                    converted.append(item)
            safe[key] = converted
        else:
            safe[key] = value
    return safe

def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def init_results_log_muon(results_path, args, optimizer_muon, optimizer_adam, start):
    muon_tensor_count = sum(len(group['params']) for group in optimizer_muon.param_groups)
    adam_tensor_count = sum(len(group['params']) for group in optimizer_adam.param_groups)
    muon_total_params = sum(p.numel() for group in optimizer_muon.param_groups for p in group['params'])
    adam_total_params = sum(p.numel() for group in optimizer_adam.param_groups for p in group['params'])
    with open(results_path, 'a') as f:
        f.write('=' * 100 + '\n')
        f.write(f"run_start_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("script: run_nerf_ranksched.py\n")
        f.write(f"expname: {args.expname}\n")
        f.write(f"dataset_type: {args.dataset_type}\n")
        f.write(f"datadir: {args.datadir}\n")
        f.write(f"resume_start_step: {start}\n")
        f.write("optimizer_setup:\n")
        f.write("  optimizer_main: Muon\n")
        f.write(f"  muon_lr_init: {args.muon_lrate}\n")
        f.write(f"  muon_weight_decay: {args.muon_decay}\n")
        f.write(f"  muon_momentum: {args.muon_momentum}\n")
        f.write("  optimizer_aux: Adam\n")
        f.write(f"  aux_adam_lr_init: {args.lrate}\n")
        f.write("  aux_adam_betas: (0.9, 0.999)\n")
        f.write(f"  lrate_decay_ksteps: {args.lrate_decay}\n")
        f.write(f"  muon_tensor_count: {muon_tensor_count}\n")
        f.write(f"  adam_tensor_count: {adam_tensor_count}\n")
        f.write(f"  muon_total_params: {muon_total_params}\n")
        f.write(f"  adam_total_params: {adam_total_params}\n")
        f.write("train_psnr_log:\n")

def init_results_log_optim(results_path, args, optimizer, start):
    optim_tensor_count = sum(len(group['params']) for group in optimizer.param_groups)
    optim_total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    with open(results_path, 'a') as f:
        f.write('=' * 100 + '\n')
        f.write(f"run_start_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("script: run_nerf_ranksched.py\n")
        f.write(f"expname: {args.expname}\n")
        f.write(f"dataset_type: {args.dataset_type}\n")
        f.write(f"datadir: {args.datadir}\n")
        f.write(f"resume_start_step: {start}\n")
        f.write("optimizer_setup:\n")
        f.write("  optimizer_main: Muon\n")
        f.write(f"  muon_lr_init: {args.muon_lrate}\n")
        f.write(f"  muon_weight_decay: {args.muon_decay}\n")
        f.write(f"  muon_momentum: {args.muon_momentum}\n")
        f.write("  optimizer_aux: Adam\n")
        f.write(f"  aux_adam_lr_init: {args.lrate}\n")
        f.write("  aux_adam_betas: (0.9, 0.999)\n")
        f.write(f"  lrate_decay_ksteps: {args.lrate_decay}\n")
        f.write(f"  muon_tensor_count: {optim_tensor_count}\n")
        f.write(f"  muon_total_params: {optim_total_params}\n")
        f.write("scheduler_setup:\n")
        f.write(f"  scheduler_name: {getattr(args, 'train_scheduler', 'rank_wsd')}\n")
        if getattr(args, 'train_scheduler', 'rank_wsd') == 'exp_decay':
            f.write(f"  lrate_decay_ksteps: {getattr(args, 'lrate_decay', 250)}\n")
            f.write("  decay_rate: 0.1\n")
            f.write(f"  decay_steps: {float(getattr(args, 'lrate_decay', 250)) * 1000.0:.1f}\n")
        else:
            f.write(f"  warmup_steps: {getattr(args, '_resolved_sched_warmup_steps', 0)}\n")
            f.write(f"  min_lr_ratio: {getattr(args, 'sched_min_lr_ratio', 0.1)}\n")
            if getattr(args, 'train_scheduler', 'rank_wsd') == 'rank_wsd':
                f.write(f"  decay_start_step: {getattr(args, '_resolved_decay_start_step', args.N_iters)}\n")
        if getattr(args, 'optimizer', '') == 'aux-lowrank-svd':
            f.write("  lowrank_schedule:\n")
            f.write(f"    rank_start: {args.lowrank_rank_start}\n")
            f.write(f"    rank_end: {args.lowrank_rank_end}\n")
            f.write(f"    schedule: {args.lowrank_schedule}\n")
            f.write(f"    schedule_steps: {getattr(args, '_effective_lowrank_schedule_steps', args.lowrank_schedule_steps if args.lowrank_schedule_steps > 0 else args.N_iters)}\n")
            f.write(f"    oversample: {args.lowrank_oversample}\n")
            f.write(f"    subspace_iters: {args.lowrank_subspace_iters}\n")
            f.write(f"    ns_steps: {args.lowrank_ns_steps}\n")
            f.write(f"    min_dim: {args.lowrank_min_dim}\n")
            f.write(f"    max_rank_ratio: {args.lowrank_max_rank_ratio}\n")
            f.write(f"    scale_mode: {args.lowrank_scale_mode}\n")
        f.write("train_psnr_log:\n")
        if getattr(args, 'optimizer', '') == 'aux-sign-auto-cos-inc':
            f.write("  lowrank_schedule:\n")
            f.write(f"    rank_start: {args.lowrank_rank_start}\n")
            f.write(f"    rank_end: {args.lowrank_rank_end}\n")
            f.write("    schedule: cosine_increase_closed_form\n")
            f.write(f"    schedule_steps: {getattr(args, '_effective_lowrank_schedule_steps', args.lowrank_schedule_steps if args.lowrank_schedule_steps > 0 else args.N_iters)}\n")
            f.write(f"    oversample: {args.lowrank_oversample}\n")
            f.write(f"    subspace_iters: {args.lowrank_subspace_iters}\n")
            f.write(f"    ns_steps: {args.lowrank_ns_steps}\n")
            f.write(f"    scale_mode: {args.lowrank_scale_mode}\n")
            if args.lowrank_auto_init_rank_start:
                f.write("  lowrank_auto_init_start:\n")
                f.write(f"    enabled: {args.lowrank_auto_init_rank_start}\n")
                f.write(f"    probe_steps: {args.lowrank_init_probe_steps}\n")
                f.write(f"    energy_tau: {args.lowrank_init_energy}\n")
                f.write(f"    round_multiple: {args.lowrank_init_round_multiple}\n")


def append_results_log_muon(results_path, i, global_step, loss, psnr, optimizer_muon, optimizer_adam):
    muon_lr = optimizer_muon.param_groups[0]['lr']
    adam_lr = optimizer_adam.param_groups[0]['lr']
    with open(results_path, 'a') as f:
        f.write(
            f"iter={i} global_step={global_step} "
            f"loss={loss.item():.10f} psnr={psnr.item():.6f} "
            f"muon_lr={muon_lr:.8e} adam_lr={adam_lr:.8e}\n"
        )

def append_results_log_optim(results_path, i, global_step, loss, psnr, optimizer):
    muon_lr = optimizer.param_groups[0]['lr']
    adam_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
    current_rank = optimizer.param_groups[0].get('current_rank', None)
    current_target_rank = optimizer.param_groups[0].get('current_target_rank', None)
    current_method = optimizer.param_groups[0].get('current_method', None)
    rank_log = ""
    if current_rank is not None:
        rank_log += f" muon_rank={current_rank}"
    if current_target_rank is not None:
        rank_log += f" muon_target_rank={current_target_rank}"
    if current_method is not None:
        rank_log += f" muon_method={current_method}"
    with open(results_path, 'a') as f:
        f.write(
            f"iter={i} global_step={global_step} "
            f"loss={loss.item():.10f} psnr={psnr.item():.6f} "
            f"muon_lr={muon_lr:.8e} adam_lr={adam_lr:.8e}{rank_log}\n"
        )

def split_nerf_params(net):
    muon_params = []
    adam_params = []

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue

        # bias / norm-like / scalar-vector params
        if p.ndim < 2:
            adam_params.append(p)
            continue

        # final output heads는 일단 Adam으로 두기
        if (
            "output_linear" in name
            or "rgb_linear" in name
            or "alpha_linear" in name
        ):
            adam_params.append(p)
            continue

        # 나머지 2D weight는 Muon
        muon_params.append(p)

    return muon_params, adam_params


def _optimizer_supports_progressive_rank(optimizer_name):
    return optimizer_name in {'aux-sign-auto-cos-inc', 'aux-sign10-rsclF'}


def _resolve_warmup_steps(args, total_iters):
    total_iters = int(total_iters)
    explicit_steps = int(getattr(args, 'sched_warmup_steps', 0) or 0)
    if explicit_steps > 0:
        return min(total_iters - 1, explicit_steps)
    warmup_frac = float(getattr(args, 'sched_warmup_frac', 0.0) or 0.0)
    if warmup_frac <= 0.0:
        return 0
    return min(total_iters - 1, int(round(total_iters * warmup_frac)))


def _resolve_ns_steps(args):
    explicit = int(getattr(args, 'lowrank_ns_steps', 0) or 0)
    if explicit > 0:
        return explicit
    if args.optimizer == 'aux-sign10-rsclF':
        return 10
    return 5


def _resolve_lowrank_rescale(args):
    mode = str(getattr(args, 'lowrank_scale_mode', 'default'))
    if mode == 'sqrt':
        return True
    if mode == 'none':
        return False
    # 'default': preserve each optimizer's native behavior.
    if args.optimizer == 'aux-sign':
        return True
    return False


def _resolve_fixed_rank(args):
    source = str(getattr(args, 'fixed_rank_source', 'end'))
    if source == 'start':
        return int(args.lowrank_rank_start)
    return int(args.lowrank_rank_end)


def _resolve_lowrank_schedule_steps(args, total_iters):
    total_iters = int(total_iters)
    if not _optimizer_supports_progressive_rank(args.optimizer):
        return 1

    explicit = int(getattr(args, 'lowrank_schedule_steps', 0) or 0)
    if explicit > 0:
        return min(total_iters, explicit)

    default_frac = float(getattr(args, 'rank_schedule_default_frac', 0.8) or 0.8)
    resolved = max(1, int(round(total_iters * default_frac)))
    return min(total_iters, resolved)


def _resolve_decay_start_step(args, total_iters, warmup_steps, rank_schedule_steps=None):
    total_iters = int(total_iters)
    warmup_steps = int(warmup_steps)

    explicit_steps = int(getattr(args, 'sched_decay_start_step', 0) or 0)
    if explicit_steps > 0:
        resolved = explicit_steps
    else:
        decay_frac = float(getattr(args, 'sched_decay_start_frac', 0.0) or 0.0)
        if decay_frac > 0.0:
            resolved = int(round(total_iters * decay_frac))
        elif (
            getattr(args, 'train_scheduler', 'rank_wsd') == 'rank_wsd'
            and _optimizer_supports_progressive_rank(args.optimizer)
            and rank_schedule_steps is not None
        ):
            resolved = int(rank_schedule_steps)
        else:
            default_frac = float(getattr(args, 'sched_default_decay_start_frac', 0.8) or 0.8)
            resolved = int(round(total_iters * default_frac))

    resolved = max(warmup_steps, resolved)
    resolved = min(total_iters - 1, resolved)
    return resolved


def _sync_optimizer_schedule_config(optimizer, args, reset_rank_fields=False):
    if optimizer is None:
        return

    effective_rank_schedule_steps = int(getattr(args, '_effective_lowrank_schedule_steps', args.N_iters))
    resolved_ns_steps = int(getattr(args, '_resolved_lowrank_ns_steps', _resolve_ns_steps(args)))
    resolved_rescale = bool(getattr(args, '_resolved_lowrank_rescale', _resolve_lowrank_rescale(args)))

    for group in optimizer.param_groups:
        if not group.get('use_muon', False):
            continue
        
        if args.optimizer == 'aux-muon' or args.optimizer == "muon":
            group['ns_steps'] = int(resolved_ns_steps)

        if args.optimizer in {'aux-sign-auto-cos-inc', 'aux-sign10-rsclF'}:
            group['warmup_steps'] = effective_rank_schedule_steps
            group['rank'] = int(args.lowrank_rank_start)
            if reset_rank_fields or ('rank_start' not in group):
                group['rank_start'] = int(args.lowrank_rank_start)
                group['current_rank'] = int(args.lowrank_rank_start)
            group['rank_end'] = int(args.lowrank_rank_end)
            group['current_target_rank'] = int(max(args.lowrank_rank_start, args.lowrank_rank_end))
            group['oversample'] = int(args.lowrank_oversample)
            group['ns_steps'] = int(resolved_ns_steps)
            group['lowrank_rescale'] = bool(resolved_rescale)
            group['auto_init_rank_start'] = bool(args.lowrank_auto_init_rank_start)
            group['init_probe_steps'] = min(int(args.lowrank_init_probe_steps), effective_rank_schedule_steps)
            group['init_energy'] = float(args.lowrank_init_energy)
            group['init_round_multiple'] = int(args.lowrank_init_round_multiple)
            if reset_rank_fields:
                group['current_method'] = (
                    'cosine_inc_auto_start' if bool(args.lowrank_auto_init_rank_start) else 'cosine_inc_closed_form'
                )



class OriginalExpDecayScheduler:
    """Original NeRF exponential learning-rate decay.

    Original NeRF updates the LR as:
        lr = base_lr * (0.1 ** (global_step / (lrate_decay * 1000)))

    This wrapper returns the same dictionary interface as the rank_wsd and
    warmup_cosine schedulers. For the Muon+AuxAdam optimizer, the same
    multiplicative decay ratio is applied to both Muon and Adam branches.
    N_rand is kept constant, matching the original scheduler behavior.
    """

    def __init__(self, base_lr_adam, base_lr_muon, lrate_decay, base_N_rand):
        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.lrate_decay = float(lrate_decay)
        self.base_N_rand = int(base_N_rand)
        self.decay_rate = 0.1
        self.decay_steps = max(1.0, self.lrate_decay * 1000.0)

    def step(self, global_step):
        # The original nerf-pytorch code updates the optimizer LR after
        # optimizer.step() using the current global_step. This training loop
        # applies schedulers before optimizer.step(), so use global_step - 1
        # to match the LR that the original code would have assigned for the
        # current optimization step. The difference is only one iteration, but
        # this keeps exp_decay faithful to the nerf-pytorch default timing.
        global_step = max(0, int(global_step))
        effective_decay_step = max(0, global_step - 1)
        lr_ratio = float(self.decay_rate ** (effective_decay_step / self.decay_steps))
        return {
            "lr_muon": self.base_lr_muon * lr_ratio,
            "lr_adam": self.base_lr_adam * lr_ratio,
            "N_rand": self.base_N_rand,
            "phase_name": "exp_decay",
            "lr_ratio": lr_ratio,
            "effective_decay_step": effective_decay_step,
        }

    def describe(self):
        return (
            "OriginalExpDecayScheduler("
            f"base_lr_adam={self.base_lr_adam:.6g}, "
            f"base_lr_muon={self.base_lr_muon:.6g}, "
            f"decay_rate={self.decay_rate}, "
            f"lrate_decay={self.lrate_decay:.6g}, "
            f"decay_steps={self.decay_steps:.1f}, "
            f"base_N_rand={self.base_N_rand}"
            ")"
        )

def _build_lr_scheduler(args):
    if args.train_scheduler == 'rank_wsd':
        return RankAwareWarmupStableLinearScheduler(
            base_lr_adam=args.lrate,
            base_lr_muon=args.muon_lrate,
            total_iters=args._effective_total_iters,
            base_N_rand=args.N_rand,
            warmup_steps=args._resolved_sched_warmup_steps,
            decay_start_step=args._resolved_decay_start_step,
            min_lr_ratio=args.sched_min_lr_ratio,
        )

    if args.train_scheduler == 'warmup_cosine':
        return WarmupCosineScheduler(
            base_lr_adam=args.lrate,
            base_lr_muon=args.muon_lrate,
            total_iters=args._effective_total_iters,
            base_N_rand=args.N_rand,
            warmup_steps=args._resolved_sched_warmup_steps,
            min_lr_ratio=args.sched_min_lr_ratio,
        )

    if args.train_scheduler == 'exp_decay':
        return OriginalExpDecayScheduler(
            base_lr_adam=args.lrate,
            base_lr_muon=args.muon_lrate,
            lrate_decay=args.lrate_decay,
            base_N_rand=args.N_rand,
        )

    raise ValueError(
        f"Unsupported train_scheduler {args.train_scheduler!r}. "
        "Supported options are: 'rank_wsd', 'warmup_cosine', 'exp_decay'."
    )


def _apply_optimizer_lrs(optimizer, muon_lr, adam_lr):
    for param_group in optimizer.param_groups:
        if param_group.get('use_muon', False):
            param_group['lr'] = muon_lr
        else:
            param_group['lr'] = adam_lr

#@ Optimizer in HERE!!
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    effective_total_iters = int(getattr(args, "_effective_total_iters", args.N_iters))
    effective_lowrank_schedule_steps = int(getattr(args, "_effective_lowrank_schedule_steps", effective_total_iters))

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    """
    NeRF(
    (pts_linears): ModuleList(
        (0): Linear(in_features=63, out_features=256, bias=True)
        (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
        (5): Linear(in_features=319, out_features=256, bias=True)
        (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
    )
    (views_linears): ModuleList(
        (0): Linear(in_features=283, out_features=128, bias=True)
    )
    (feature_linear): Linear(in_features=256, out_features=256, bias=True)
    (alpha_linear): Linear(in_features=256, out_features=1, bias=True)
    (rgb_linear): Linear(in_features=128, out_features=3, bias=True)
    )
    """
    #@ Muon INR: split_muon_like_named_params
    total_grad_vars = list(model.parameters())
    muon_params, adam_params = split_nerf_params(model)

    # from optims.run_utils import split_muon_like_named_params
    # hiddenp, otherp = split_muon_like_named_params(model, "nerf")
    # import pdb; pdb.set_trace()

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        total_grad_vars += list(model_fine.parameters())
        muon_fine_params, adam_fine_params = split_nerf_params(model_fine)
        muon_params += muon_fine_params
        adam_params += adam_fine_params

    grad_vars = muon_params + adam_params

    print(f'Param split: Muon={len(muon_params)} tensors, Adam={len(adam_params)} tensors')

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk
        )

    #//======== Create optimizers ========//#
    aux_param_groups = [
        dict(
            params=muon_params,
            use_muon=True,
            lr=args.muon_lrate,
            weight_decay=args.muon_decay,
            momentum=args.muon_momentum,
        ),
        dict(
            params=adam_params,
            use_muon=False,
            lr=args.lrate,
            betas=parse_pair(args.muon_aux_betas),
            eps=args.muon_aux_eps,
            weight_decay=args.muon_aux_weight_decay,
        ),
    ]

    if args.optimizer in ('ori-adam', 'adam'):
        optimizer = torch.optim.Adam(params=list(total_grad_vars), lr=args.lrate, betas=(0.9, 0.999))
    elif args.optimizer == 'aux-muon' or args.optimizer == "muon":
        if SingleDeviceMuonWithAuxAdam is None:
            raise ImportError(
                "optimizer aux-muon requires optims/muon.py, but that file is missing. "
            )
        optimizer = SingleDeviceMuonWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Aux sign optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}.'
        )
    elif args.optimizer == 'aux-sign-auto-cos-inc':
        aux_param_groups[0].update(
            dict(
                rank=args.lowrank_rank_start,
                rank_start=args.lowrank_rank_start,
                rank_end=args.lowrank_rank_end,
                warmup_steps=args._effective_lowrank_schedule_steps,
                oversample=args.lowrank_oversample,
                ns_steps=int(getattr(args, '_resolved_lowrank_ns_steps', _resolve_ns_steps(args))),
                lowrank_rescale=bool(getattr(args, '_resolved_lowrank_rescale', _resolve_lowrank_rescale(args))),
                auto_init_rank_start=args.lowrank_auto_init_rank_start,
                init_probe_steps=min(args.lowrank_init_probe_steps, args._effective_lowrank_schedule_steps),
                init_energy=args.lowrank_init_energy,
                init_round_multiple=args.lowrank_init_round_multiple,
            )
        )
        optimizer = SingleDeviceAutoCosIncWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Sign Auto Cos increase optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}.'
        )
    elif args.optimizer == 'aux-sign10-rsclF':
        aux_param_groups[0].update(
            dict(
                rank=args.lowrank_rank_start,
                rank_start=args.lowrank_rank_start,
                rank_end=args.lowrank_rank_end,
                warmup_steps=args._effective_lowrank_schedule_steps,
                oversample=args.lowrank_oversample,
                ns_steps=int(getattr(args, '_resolved_lowrank_ns_steps', _resolve_ns_steps(args))),
                lowrank_rescale=bool(getattr(args, '_resolved_lowrank_rescale', _resolve_lowrank_rescale(args))),
                auto_init_rank_start=args.lowrank_auto_init_rank_start,
                init_probe_steps=min(args.lowrank_init_probe_steps, args._effective_lowrank_schedule_steps),
                init_energy=args.lowrank_init_energy,
                init_round_multiple=args.lowrank_init_round_multiple,
            )
        )
        optimizer = SingleDeviceSign10RsclFWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Sign10RsclF optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}.'
        )
    else:
        raise ValueError(
            f"Unsupported optimizer {args.optimizer!r}. Supported options are: "
            "'adam', 'aux-sign', 'aux-sign-auto-cos-inc', 'aux-sign10-rsclF'."
        )
    #//===================================//#

    _sync_optimizer_schedule_config(optimizer, args, reset_rank_fields=True)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        ckpt_opt_name = ckpt.get('optimizer_name')

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None and ckpt.get('network_fine_state_dict') is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        # if 'optimizer_state_dict' in ckpt:
        #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #     print('Loaded optimizer_state_dict from checkpoint.')
        # elif 'optimizer_muon_state_dict' in ckpt or 'optimizer_adam_state_dict' in ckpt:
        #     print('Legacy split optimizer checkpoint detected: skipping optimizer state reload because the current code uses a unified optimizer state_dict.')
        if 'optimizer_state_dict' in ckpt and ckpt_opt_name == args.optimizer:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            _sync_optimizer_schedule_config(optimizer, args, reset_rank_fields=False)
            print('Loaded optimizer_state_dict from checkpoint.')
        else:
            print('Skipped optimizer_state_dict reload (optimizer mismatch or missing optimizer_name).')

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer_muon, optimizer_adam
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--lpips_net", type=str, default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='backbone network used for LPIPS evaluation on the testset')
    parser.add_argument("--eval_testset_only", action='store_true',
                        help='Load the saved checkpoint, evaluate the test set once, save metrics, and exit without training.')
    parser.add_argument("--test_out_json", type=str, default=None,
                        help='Output JSON path for --eval_testset_only. Default: <basedir>/<expname>/test_metrics_eval.json')
    parser.add_argument("--test_out_dir", type=str, default=None,
                        help='Output rendered-image directory for --eval_testset_only. Default: <basedir>/<expname>/testset_eval')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=2000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=100000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_valset", type=int, default=100000,
                        help='frequency of full validation-set mean PSNR evaluation for HPO')
    parser.add_argument("--i_video",   type=int, default=100000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--N_iters", type=int, default=100000,
                        help='total number of training iterations')
    
    # Muon optimizer options 0.02
    parser.add_argument("--muon_lrate",   type=float, default=5e-4,)
    parser.add_argument("--muon_decay",   type=float, default=0.0,) #=muon_weight_decay
    parser.add_argument("--muon_momentum",   type=float, default=0.90,)
    # NEW
    parser.add_argument('--muon_aux_eps', type=float, default=5e-4, help='Epsilon for Muon auxiliary Adam branch.')
    parser.add_argument('--muon_aux_weight_decay', type=float, default=0.0, help='Weight decay for auxiliary Adam in Muon.')
    parser.add_argument('--muon_aux_betas', type=str, default='0.9,0.95', help='Betas for Muon auxiliary Adam branch.')
    parser.add_argument('--lowrank_rank_start', type=int, default=150,
                        help='Starting rank for low-rank Muon schedule.')
    parser.add_argument('--lowrank_rank_end', type=int, default=250,
                        help='Ending rank for low-rank Muon schedule.')
    parser.add_argument('--lowrank_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine'],
                        help='Rank schedule for low-rank Muon branch.')
    parser.add_argument('--lowrank_schedule_steps', type=int, default=0,
                        help='Number of optimizer steps over which to apply the rank schedule. 0 means use N_iters.')
    parser.add_argument('--lowrank_oversample', type=int, default=4,
                        help='Oversampling for randomized low-rank subspace estimation.')
    parser.add_argument('--lowrank_subspace_iters', type=int, default=1,
                        help='Number of subspace power iterations for randomized low-rank Muon.')
    parser.add_argument('--lowrank_ns_steps', type=int, default=0,
                        help='Number of Newton-Schulz steps. 0 means use the optimizer-specific default (5 for auto_cos/lr_sign, 10 for lr_sign10_rsclF).')
    parser.add_argument('--lowrank_min_dim', type=int, default=256,
                        help='Use Newton-Schulz instead of low-rank approximation when min(weight.shape) is smaller than this value.')
    parser.add_argument('--lowrank_max_rank_ratio', type=float, default=1.0,
                        help='Optional cap: effective rank <= lowrank_max_rank_ratio * min(weight.shape).')
    parser.add_argument('--lowrank_scale_mode', type=str, default='default',
                        choices=['default', 'sqrt', 'none'],
                        help="How to rescale truncated low-rank updates. 'default' preserves each optimizer's native behavior.")
    parser.add_argument("--optimizer",
                        type=str,
                        default='aux-sign-auto-cos-inc',
                        choices=['adam', 'ori-adam', 'aux-muon', 'muon', 'aux-sign', 'aux-sign-auto-cos-inc', 'aux-sign10-rsclF'],
                        )
    
    # 재현성
    parser.add_argument("--seed", type=int, default=0,
                    help="random seed for full reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                    help="use deterministic CUDA/PyTorch behavior")

    parser.add_argument(
        "--lowrank_auto_init_rank_start",
        action="store_true",
        help="If set, estimate rank_start from the first few Muon search matrices "
            "using a Frobenius-energy probe, then continue with the original cosine increase schedule.",
    )
    parser.add_argument(
        "--lowrank_init_probe_steps",
        type=int,
        default=8,
        help="Number of early steps used to estimate rank_start when --lowrank_auto_init_rank_start is enabled.",
    )
    parser.add_argument(
        "--lowrank_init_energy",
        type=float,
        # default=0.90,
        default=0.999,
        help="Energy threshold tau for choosing rank_start from the sketched spectrum.",
    )
    parser.add_argument(
        "--lowrank_init_round_multiple",
        type=int,
        default=8,
        help="Round the auto-selected rank_start up to this multiple.",
    )

    # Scheduler options: main candidate vs strong baseline
    parser.add_argument("--train_scheduler", "--train-scheduler", dest="train_scheduler", type=str, default='rank_wsd',
                        choices=['rank_wsd', 'warmup_cosine', 'exp_decay'],
                        help=("LR scheduler family. 'rank_wsd' = micro-warmup -> stable plateau -> late linear decay; "
                              "'warmup_cosine' = micro-warmup -> cosine decay; "
                              "'exp_decay' = original NeRF exponential LR decay using --lrate_decay."))
    parser.add_argument("--sched_warmup_steps", type=int, default=0,
                        help="Optional explicit warmup steps. If > 0, overrides sched_warmup_frac.")
    parser.add_argument("--sched_warmup_frac", type=float, default=0.01,
                        help="Warmup fraction when sched_warmup_steps == 0. Default 1%% as a micro-warmup for INR.")
    parser.add_argument("--sched_min_lr_ratio", type=float, default=0.1,
                        help="Final LR ratio relative to the base LR for both schedulers.")
    parser.add_argument("--sched_decay_start_step", type=int, default=0,
                        help="Explicit decay-start step for rank_wsd. If 0, it is inferred from rank saturation or sched_decay_start_frac.")
    parser.add_argument("--sched_decay_start_frac", type=float, default=0.0,
                        help="Explicit decay-start fraction for rank_wsd. Ignored if sched_decay_start_step > 0.")
    parser.add_argument("--sched_default_decay_start_frac", type=float, default=0.8,
                        help="Fallback decay-start fraction for rank_wsd when no explicit step/fraction is given and the optimizer has no progressive rank schedule.")
    parser.add_argument("--rank_schedule_default_frac", type=float, default=0.8,
                        help="When lowrank_schedule_steps == 0 for a progressive-rank optimizer, use this fraction of total iters as the default rank-growth horizon.")
    parser.add_argument("--fixed_rank_source", type=str, default='end',
                        choices=['start', 'end'],
                        help="For fixed-rank lr_sign, which user rank to apply: lowrank_rank_start or lowrank_rank_end.")

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)

    # Resolve scheduler and optimizer-side rank horizons *before* constructing
    # the optimizer so that any internal progressive-rank logic sees the final
    # schedule-aligned training horizon.
    args._effective_total_iters = int(args.N_iters)
    args._resolved_sched_warmup_steps = _resolve_warmup_steps(args, args._effective_total_iters)
    args._resolved_lowrank_ns_steps = _resolve_ns_steps(args)
    args._resolved_lowrank_rescale = _resolve_lowrank_rescale(args)
    args._resolved_fixed_rank = _resolve_fixed_rank(args)
    args._effective_lowrank_schedule_steps = _resolve_lowrank_schedule_steps(args, args._effective_total_iters)
    args._resolved_decay_start_step = _resolve_decay_start_step(
        args,
        args._effective_total_iters,
        args._resolved_sched_warmup_steps,
        args._effective_lowrank_schedule_steps if _optimizer_supports_progressive_rank(args.optimizer) else None,
    )
    train_scheduler = _build_lr_scheduler(args)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file. In eval-only mode, avoid
    # overwriting the original training args.txt/config.txt from the HPO trial.
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    args_log_name = 'eval_args.txt' if args.eval_testset_only else 'args.txt'
    config_log_name = 'eval_config.txt' if args.eval_testset_only else 'config.txt'
    f = os.path.join(basedir, expname, args_log_name)
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, config_log_name)
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    #? CHANGE: Create nerf model
    # render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer_muon, optimizer_adam = create_nerf(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    results_path = os.path.join(basedir, expname, 'results.txt')
    # init_results_log_muon(results_path, args, optimizer_muon, optimizer_adam, start)
    if not args.eval_testset_only:
        init_results_log_optim(results_path, args, optimizer, start)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.eval_testset_only:
        eval_start_time = time.time()
        testsavedir = args.test_out_dir or os.path.join(basedir, expname, 'testset_eval')
        out_json = args.test_out_json or os.path.join(basedir, expname, 'test_metrics_eval.json')
        os.makedirs(testsavedir, exist_ok=True)
        out_json_dir = os.path.dirname(out_json)
        if out_json_dir:
            os.makedirs(out_json_dir, exist_ok=True)

        print('EVAL TESTSET ONLY')
        print('test poses shape', poses[i_test].shape)
        print('test output dir', testsavedir)
        print('test metrics json', out_json)

        with torch.no_grad():
            _, _, test_metrics = render_path_with_metrics(
                torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk,
                render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir,
                lpips_net=args.lpips_net
            )

        metrics_payload = _json_safe_metrics(test_metrics)
        metrics_payload.update({
            'checkpoint_step': int(start),
            'num_test_views': int(len(i_test)),
            'dataset_type': args.dataset_type,
            'expname': args.expname,
            'basedir': args.basedir,
            'test_out_dir': testsavedir,
            'readme_path': os.path.join(basedir, expname, 'README.txt'),
        })

        with open(out_json, 'w') as f:
            json.dump(metrics_payload, f, indent=2)

        elapsed_seconds = time.time() - eval_start_time
        report_path = save_testset_metrics_to_readmes(
            report_dir=os.path.join(basedir, expname),
            args=args,
            i=int(start),
            global_step=int(start),
            loss=0.0,
            psnr=0.0,
            elapsed_seconds=elapsed_seconds,
            test_metrics=test_metrics,
        )

        print('=' * 60)
        print('Saved test metrics to:', out_json)
        print('Saved rendered test images to:', testsavedir)
        print('Saved test README to:', report_path)
        print('mean_psnr :', metrics_payload.get('mean_psnr'))
        print('mean_ssim :', metrics_payload.get('mean_ssim'))
        print('mean_lpips:', metrics_payload.get('mean_lpips'))
        mean_psnr = metrics_payload.get('mean_psnr')
        mean_ssim = metrics_payload.get('mean_ssim')
        mean_lpips = metrics_payload.get('mean_lpips')
        if mean_psnr is not None and mean_ssim is not None:
            print(
                f"[TEST_METRICS] mean_psnr: {mean_psnr:.6f} "
                f"mean_ssim: {mean_ssim:.6f} "
                f"mean_lpips: {mean_lpips}"
            )
        else:
            print(
                f"[TEST_METRICS] mean_psnr: {mean_psnr} "
                f"mean_ssim: {mean_ssim} mean_lpips: {mean_lpips}"
            )
        return

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    print(train_scheduler.describe())
    N_iters = int(getattr(args, '_effective_total_iters', args.N_iters)) + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    train_start_time = time.time()
    for i in trange(start, N_iters):
        time0 = time.time()

        # Apply the selected LR schedule *before* sampling so the current step
        # uses the correct LR and the rank-schedule-aware decay onset.
        sched = train_scheduler.step(global_step)
        new_muon_lrate = sched["lr_muon"]
        new_adam_lrate = sched["lr_adam"]
        N_rand = int(sched["N_rand"])
        current_phase = sched["phase_name"]
        current_lr_ratio = float(sched.get("lr_ratio", 1.0))

        _apply_optimizer_lrs(optimizer, new_muon_lrate, new_adam_lrate)

        # Sample random ray batch
        if use_batching:
            # Ensure the variable batch size always gets a full slice.
            if i_batch + N_rand > rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                replace = bool(N_rand > coords.shape[0])
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=replace)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        # optimizer_muon.zero_grad()
        # optimizer_adam.zero_grad()
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if "shampoo" in args.optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grad_vars, max_norm=1.0)
            optimizer.step()
        else: 
            loss.backward()
            # optimizer_muon.step()
            # optimizer_adam.step()
            optimizer.step()


        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'optimizer_name': args.optimizer,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                # 'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                # 'optimizer_adam_state_dict': optimizer_adam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'effective_total_iters': int(getattr(args, '_effective_total_iters', args.N_iters)),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            # with torch.no_grad():
            #     render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            with torch.no_grad():
                _, _, test_metrics = render_path_with_metrics(
                    torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk,
                    render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir,
                    lpips_net=args.lpips_net
                )
            
            elapsed_seconds = time.time() - train_start_time
            report_path = save_testset_metrics_to_readmes(
                report_dir=os.path.join(basedir, expname),
                args=args,
                i=i,
                global_step=global_step,
                loss=loss,
                psnr=psnr,
                elapsed_seconds=elapsed_seconds,
                test_metrics=test_metrics,
            )
            print('Saved test set')
            print('Saved test README to', report_path)
            if test_metrics is not None and test_metrics.get('mean_psnr') is not None:
                print(
                    f"[TEST_METRICS] Iter: {i} "
                    f"mean_psnr: {test_metrics['mean_psnr']:.6f} "
                    f"mean_ssim: {test_metrics['mean_ssim']:.6f} "
                    f"mean_lpips: {test_metrics.get('mean_lpips')}"
                )

        # Full validation-set evaluation for HPO.
        # This is the metric that the revised BoTorch script parses.
        # It is different from the single random-image [VAL] log below.
        if getattr(args, 'i_valset', 0) > 0 and i % args.i_valset == 0 and i > 0:
            print('val poses shape', poses[i_val].shape)
            with torch.no_grad():
                val_metrics = render_path_mean_psnr(
                    poses[i_val].to(device), hwf, K, args.chunk,
                    render_kwargs_test, gt_imgs=images[i_val],
                )

            if val_metrics['mean_psnr'] is None:
                raise RuntimeError('Full validation-set mean PSNR could not be computed.')

            hpo_val_path = os.path.join(basedir, expname, 'hpo_val_metrics.json')
            with open(hpo_val_path, 'w') as f:
                json.dump({
                    'iter': int(i),
                    'global_step': int(global_step),
                    'valset_mean_mse': val_metrics['mean_mse'],
                    'valset_mean_psnr': val_metrics['mean_psnr'],
                    'valset_psnr_per_image': val_metrics['psnr_per_image'],
                }, f, indent=2)

            tqdm.write(
                f"[HPO_VAL] Iter: {i} mean_psnr: {val_metrics['mean_psnr']:.6f} "
                f"mean_mse: {val_metrics['mean_mse']:.10f}"
            )


        #// val
        if i % args.i_print==0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  "
                f"scheduler={args.train_scheduler} phase={current_phase} N_rand={N_rand} "
                f"lr_ratio={current_lr_ratio:.4f} lr_muon={new_muon_lrate:.3e} lr_adam={new_adam_lrate:.3e}"
            )
            if "aux" in args.optimizer:
                append_results_log_optim(results_path, i, global_step, loss, psnr, optimizer)
            else:
                # append_results_log_muon(results_path, i, global_step, loss, psnr, optimizer_muon, optimizer_adam)
                append_results_log_muon(results_path, i, global_step, loss, psnr, optimizer, optimizer)
            

            img_i = np.random.choice(i_val)
            target_val = torch.Tensor(images[img_i]).to(device)
            pose_val = poses[img_i, :3, :4]

            with torch.no_grad():
                rgb_val, disp_val, acc_val, extras_val = render(
                    H, W, K, chunk=args.chunk, c2w=pose_val, **render_kwargs_test
                )

            val_loss = img2mse(rgb_val, target_val)
            val_psnr = mse2psnr(val_loss)
            tqdm.write(f"[VAL]   Iter: {i} PSNR: {val_psnr.item()}")
        
        global_step += 1

    if "aux" in args.optimizer:
        append_results_log_optim(results_path, i, global_step, loss, psnr, optimizer)
    else:
        # append_results_log_muon(results_path, i, global_step, loss, psnr, optimizer_muon, optimizer_adam)
        append_results_log_muon(results_path, i, global_step, loss, psnr, optimizer, optimizer)
        


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()


"""
--optimizer aux-sign10-rsclF

# 1) 메인 후보: micro-warmup -> stable -> late linear decay
CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py \
  --config configs/lego.txt \
  --expname lego_rankwsd_auto \
  --optimizer aux-sign-auto-cos-inc \
  --train-scheduler rank_wsd \
  --muon_lrate 3e-3 \
  --lowrank_rank_start 150 \
  --lowrank_rank_end 250 \
  --lowrank_auto_init_rank_start
  
# 2) 강한 baseline: micro-warmup -> cosine decay
CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py \
  --config configs/lego.txt \
  --expname lego_cosine_auto \
  --optimizer aux-sign-auto-cos-inc \
  --train-scheduler warmup_cosine \
  --muon_lrate 3e-3 \
  --lowrank_rank_start 150 \
  --lowrank_rank_end 250 \
  --lowrank_auto_init_rank_start


# 3) Original NeRF exponential decay scheduler
CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched_final.py \
  --config configs/lego.txt \
  --expname lego_exp_decay_auto \
  --optimizer aux-sign-auto-cos-inc \
  --train-scheduler exp_decay \
  --muon_lrate 3e-3 \
  --lrate 5e-4 \
  --lrate_decay 250 \
  --lowrank_rank_start 150 \
  --lowrank_rank_end 250 \
  --lowrank_auto_init_rank_start

CUDA_VISIBLE_DEVICES=1 python run_nerf_ranksched.py --basedir logs/sched/rankwsd --config configs/flower.txt --expname flower_auto_200k --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000
CUDA_VISIBLE_DEVICES=2 python run_nerf_ranksched.py --basedir logs/sched/cosine --config configs/fern.txt --expname fern_auto_200k --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000
CUDA_VISIBLE_DEVICES=3 python run_nerf_ranksched.py --basedir logs/sched/cosine --config configs/drums.txt --expname drums_auto_200k --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 200000


CUDA_VISIBLE_DEVICES=0 python run_nerf_ranksched.py --basedir logs/sched/rankwsd --config configs/room.txt --expname room_auto --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 3e-3 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000
"""
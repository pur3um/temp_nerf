"""Safe NeRF training/evaluation entry point for MoFaSGD comparisons.

Place this file at the root of the NeRF/MoFaSGD repository, next to
``run_nerf_helpers.py`` and the dataset loaders.  It supports three optimizer
modes under the same rendering, sampling, checkpoint, logging, and evaluation
code:

* adam       : all parameters with Adam, standard NeRF baseline.
* sgd_split  : hidden 2D weights with SGD momentum, auxiliary parameters with Adam.
* mofa_split : hidden 2D weights with MoFaSGD, auxiliary parameters with Adam.

The implementation avoids unsafe/default-device assumptions and is designed for
fair optimizer comparisons rather than quick prototype rendering.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from run_nerf_helpers import *  # noqa: F401,F403 - keep compatibility with the original NeRF codebase
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from optims.mofasgd_safe import MomentumFactorizedSGD


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


# -----------------------------------------------------------------------------
# Reproducibility, device, and checkpoint helpers
# -----------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return dev


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def as_float_tensor(x: Any, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(x, device=device, dtype=dtype)


def clone_intrinsics_np(K: Any) -> np.ndarray:
    if torch.is_tensor(K):
        return K.detach().cpu().numpy().astype(np.float32).copy()
    return np.array(K, dtype=np.float32, copy=True)


def scale_intrinsics(K: Any, scale: int) -> np.ndarray:
    K_scaled = clone_intrinsics_np(K)
    if scale != 0:
        K_scaled[:2, :] /= float(scale)
    return K_scaled


def safe_torch_load(path: str, map_location: torch.device, allow_unsafe: bool = False) -> Mapping[str, Any]:
    """Load a checkpoint with PyTorch's weights-only mode unless explicitly disabled.

    ``weights_only=True`` can load dictionaries containing tensors and simple
    Python containers, which is enough for the checkpoints saved by this script.
    Passing ``--allow_unsafe_checkpoint_load`` re-enables full pickle loading and
    should only be used for checkpoints you created yourself.
    """
    if allow_unsafe:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not expose torch.load(..., weights_only=True). "
            "Upgrade PyTorch, or use --allow_unsafe_checkpoint_load only for a checkpoint "
            "you fully trust."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Safe checkpoint loading failed for {path}. Do not load untrusted .tar/.pth files. "
            "For a trusted checkpoint created by you, rerun with --allow_unsafe_checkpoint_load."
        ) from exc


def cuda_peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# -----------------------------------------------------------------------------
# Device-aware ray generation and hierarchical sampling
# -----------------------------------------------------------------------------


def get_rays_torch_safe(H: int, W: int, K: Any, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device-aware version of the original NeRF get_rays."""
    if c2w.dim() == 2:
        c2w_mat = c2w[:3, :4]
    else:
        raise ValueError(f"c2w must be [3,4] or [4,4], got {tuple(c2w.shape)}")

    device = c2w.device
    dtype = c2w.dtype
    K_t = as_float_tensor(K, device=device, dtype=dtype)

    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )
    dirs = torch.stack(
        [
            (i - K_t[0, 2]) / K_t[0, 0],
            -(j - K_t[1, 2]) / K_t[1, 1],
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w_mat[:3, :3], dim=-1)
    rays_o = c2w_mat[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def get_rays_np_safe(H: int, W: int, K: Any, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy version used before moving the precomputed ray batch to GPU."""
    K_np = clone_intrinsics_np(K)
    c2w_np = np.array(c2w, dtype=np.float32, copy=False)[:3, :4]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack(
        [
            (i - K_np[0, 2]) / K_np[0, 0],
            -(j - K_np[1, 2]) / K_np[1, 1],
            -np.ones_like(i, dtype=np.float32),
        ],
        axis=-1,
    )
    rays_d = np.sum(dirs[..., None, :] * c2w_np[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w_np[:3, 3], rays_d.shape)
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def sample_pdf_safe(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    pytest: bool = False,
) -> torch.Tensor:
    """Device-aware hierarchical sampling copied from NeRF's sample_pdf logic."""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True).clamp_min(1e-10)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    sample_shape = list(cdf.shape[:-1]) + [N_samples]
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device, dtype=bins.dtype)
        u = u.expand(sample_shape).contiguous()
    else:
        u = torch.rand(sample_shape, device=bins.device, dtype=bins.dtype)

    if pytest:
        np.random.seed(0)
        if det:
            u_np = np.linspace(0.0, 1.0, N_samples, dtype=np.float32)
            u_np = np.broadcast_to(u_np, sample_shape)
        else:
            u_np = np.random.rand(*sample_shape).astype(np.float32)
        u = torch.as_tensor(u_np, device=bins.device, dtype=bins.dtype)

    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), dim=2, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), dim=2, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def batchify(fn, chunk: Optional[int]):
    """Construct a version of fn that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepare inputs and apply network fn."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret: Dict[str, List[torch.Tensor]] = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for key, value in ret.items():
            all_ret.setdefault(key, []).append(value)

    return {key: torch.cat(value, dim=0) for key, value in all_ret.items()}


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays or a full image."""
    H, W = int(H), int(W)

    if c2w is not None:
        c2w = as_float_tensor(c2w, device=DEVICE)
        rays_o, rays_d = get_rays_torch_safe(H, W, K, c2w)
    else:
        rays_o, rays_d = rays
        rays_o = as_float_tensor(rays_o, device=DEVICE)
        rays_d = as_float_tensor(rays_d, device=DEVICE)

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            c2w_staticcam = as_float_tensor(c2w_staticcam, device=DEVICE)
            rays_o, rays_d = get_rays_torch_safe(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).clamp_min(1e-10)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    if ndc:
        K_t = as_float_tensor(K, device=rays_d.device, dtype=rays_d.dtype)
        rays_o, rays_d = ndc_rays(H, W, K_t[0, 0], 1.0, rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t = float(near) * torch.ones_like(rays_d[..., :1])
    far_t = float(far) * torch.ones_like(rays_d[..., :1])
    rays_cat = torch.cat([rays_o, rays_d, near_t, far_t], dim=-1)
    if use_viewdirs:
        rays_cat = torch.cat([rays_cat, viewdirs], dim=-1)

    all_ret = batchify_rays(rays_cat, chunk, **kwargs)
    for key in all_ret:
        k_shape = list(sh[:-1]) + list(all_ret[key].shape[1:])
        all_ret[key] = torch.reshape(all_ret[key], k_shape)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[key] for key in k_extract]
    ret_dict = {key: all_ret[key] for key in all_ret if key not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    K_render = clone_intrinsics_np(K)

    if render_factor != 0:
        H = H // int(render_factor)
        W = W // int(render_factor)
        focal = float(focal) / float(render_factor)
        K_render = scale_intrinsics(K_render, int(render_factor))

    rgbs = []
    disps = []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses, desc="render_path")):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K_render, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            filename = os.path.join(savedir, f"{i:03d}.png")
            imageio.imwrite(filename, to8b(rgbs[-1]))

    return np.stack(rgbs, axis=0), np.stack(disps, axis=0)


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False, pytest=False):
    """Transform raw NeRF predictions into RGB, disparity, opacity, weights, depth."""
    raw2alpha = lambda raw_sigma, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw_sigma) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = torch.zeros_like(raw[..., 3])
    if raw_noise_std > 0.0:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise_np = np.random.rand(*list(raw[..., 3].shape)).astype(np.float32) * raw_noise_std
            noise = torch.as_tensor(noise_np, device=raw.device, dtype=raw.dtype)

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    trans_init = torch.ones((alpha.shape[0], 1), device=alpha.device, dtype=alpha.dtype)
    weights = alpha * torch.cumprod(torch.cat([trans_init, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    denom = acc_map.clamp_min(1e-10)
    disp_map = 1.0 / torch.clamp(depth_map / denom, min=1e-10)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
):
    """Volumetric rendering for a batch of rays."""
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=ray_batch.device, dtype=ray_batch.dtype)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)

        if pytest:
            np.random.seed(0)
            t_rand_np = np.random.rand(*list(z_vals.shape)).astype(np.float32)
            t_rand = torch.as_tensor(t_rand_np, device=z_vals.device, dtype=z_vals.dtype)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf_safe(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)

    if DEBUG:
        for key, value in ret.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"! [Numerical Error] {key} contains nan or inf.")

    return ret


# -----------------------------------------------------------------------------
# Optimizer setup and diagnostics
# -----------------------------------------------------------------------------


def split_nerf_params(net) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split NeRF parameters into MoFa/SGD hidden matrices and Adam auxiliaries."""
    main_params: List[torch.nn.Parameter] = []
    aux_params: List[torch.nn.Parameter] = []

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim < 2:
            aux_params.append(p)
            continue

        if "output_linear" in name or "rgb_linear" in name or "alpha_linear" in name:
            aux_params.append(p)
            continue

        main_params.append(p)

    return main_params, aux_params


def collect_param_splits(models: Sequence[Optional[torch.nn.Module]]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    main_params: List[torch.nn.Parameter] = []
    aux_params: List[torch.nn.Parameter] = []
    for model in models:
        if model is None:
            continue
        m, a = split_nerf_params(model)
        main_params.extend(m)
        aux_params.extend(a)
    return main_params, aux_params


def count_params(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def zero_optimizers(optimizers: Mapping[str, Optional[torch.optim.Optimizer]]) -> None:
    for optimizer in optimizers.values():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: Mapping[str, Optional[torch.optim.Optimizer]]) -> None:
    for optimizer in optimizers.values():
        if optimizer is not None:
            optimizer.step()


def set_optimizer_lrs(args, optimizers: Mapping[str, Optional[torch.optim.Optimizer]], global_step: int) -> Dict[str, float]:
    decay_rate = 0.1
    decay_steps = max(1, int(args.lrate_decay) * 1000)
    decay = decay_rate ** (float(global_step) / float(decay_steps))

    lr_map: Dict[str, float] = {}
    if args.optimizer == "adam":
        lr_map["adam"] = float(args.lrate) * decay
    elif args.optimizer == "sgd_split":
        lr_map["main"] = float(args.sgd_lrate) * decay
        lr_map["aux"] = float(args.lrate) * decay
    elif args.optimizer == "mofa_split":
        lr_map["main"] = float(args.mofa_lrate) * decay
        lr_map["aux"] = float(args.lrate) * decay
    else:
        raise ValueError(f"Unknown optimizer mode: {args.optimizer}")

    for name, optimizer in optimizers.items():
        if optimizer is None:
            continue
        lr = lr_map.get(name)
        if lr is None:
            continue
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr_map


def optimizer_state_dicts(optimizers: Mapping[str, Optional[torch.optim.Optimizer]]) -> Dict[str, Any]:
    return {name: opt.state_dict() for name, opt in optimizers.items() if opt is not None}


def load_optimizer_state_dicts(
    ckpt: Mapping[str, Any],
    optimizers: Mapping[str, Optional[torch.optim.Optimizer]],
    expected_mode: str,
) -> None:
    ckpt_mode = ckpt.get("optimizer_mode")
    if ckpt_mode is not None and ckpt_mode != expected_mode:
        print(f"[WARN] Checkpoint optimizer mode is {ckpt_mode!r}, current mode is {expected_mode!r}; skipping optimizer states.")
        return

    if "optimizer_state_dicts" in ckpt:
        state_dicts = ckpt["optimizer_state_dicts"]
        for name, optimizer in optimizers.items():
            if optimizer is not None and name in state_dicts:
                optimizer.load_state_dict(state_dicts[name])
        return

    # Backward-compatible path for the prototype split checkpoint format.
    if expected_mode == "mofa_split" and "optimizer_mofa_state_dict" in ckpt and "optimizer_adam_state_dict" in ckpt:
        if optimizers.get("main") is not None:
            optimizers["main"].load_state_dict(ckpt["optimizer_mofa_state_dict"])
        if optimizers.get("aux") is not None:
            optimizers["aux"].load_state_dict(ckpt["optimizer_adam_state_dict"])
        return

    if expected_mode == "adam" and "optimizer_state_dict" in ckpt and optimizers.get("adam") is not None:
        optimizers["adam"].load_state_dict(ckpt["optimizer_state_dict"])
        return

    print("[WARN] No compatible optimizer state found; model weights were loaded but optimizer states were not.")


def optimizer_diagnostics(optimizers: Mapping[str, Optional[torch.optim.Optimizer]]) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}
    for name, optimizer in optimizers.items():
        if optimizer is None:
            continue
        if hasattr(optimizer, "get_diagnostics"):
            diag[name] = optimizer.get_diagnostics()
        else:
            diag[name] = {"num_param_groups": len(optimizer.param_groups)}
    return diag


# -----------------------------------------------------------------------------
# Logging and evaluation
# -----------------------------------------------------------------------------


def init_results_log(results_path: str, args, optimizers: Mapping[str, Optional[torch.optim.Optimizer]], start_iter: int) -> None:
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"run_start_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("script: run_nerf_mofasgd_safe.py\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"expname: {args.expname}\n")
        f.write(f"dataset_type: {args.dataset_type}\n")
        f.write(f"datadir: {args.datadir}\n")
        f.write(f"resume_start_iter: {start_iter}\n")
        f.write(f"optimizer_mode: {args.optimizer}\n")
        f.write("optimizer_hparams:\n")
        f.write(f"  adam_lr_init: {args.lrate}\n")
        f.write(f"  lrate_decay_ksteps: {args.lrate_decay}\n")
        if args.optimizer == "mofa_split":
            f.write(f"  mofa_lr_init: {args.mofa_lrate}\n")
            f.write(f"  mofa_rank: {args.mofa_rank}\n")
            f.write(f"  mofa_beta: {args.mofa_beta}\n")
            f.write(f"  mofa_weight_decay: {args.mofa_decay}\n")
            f.write(f"  mofa_eta1: {args.mofa_eta1}\n")
            f.write(f"  mofa_eta2: {args.mofa_eta2}\n")
            f.write(f"  mofa_eps_reciprocal: {args.mofa_eps}\n")
            f.write(f"  mofa_uv_eps: {args.mofa_uv_eps}\n")
            f.write(f"  mofa_max_value: {args.mofa_max_value}\n")
            f.write(f"  mofa_use_reciprocal_s: {args.mofa_use_reciprocal_s}\n")
            f.write(f"  mofa_use_current_projection: {args.mofa_use_current_projection}\n")
            f.write(f"  mofa_nesterov_like: {args.mofa_nesterov}\n")
        if args.optimizer == "sgd_split":
            f.write(f"  sgd_lr_init: {args.sgd_lrate}\n")
            f.write(f"  sgd_momentum: {args.sgd_momentum}\n")
            f.write(f"  sgd_weight_decay: {args.sgd_decay}\n")
            f.write(f"  sgd_nesterov: {args.sgd_nesterov}\n")
        f.write("optimizer_diagnostics_at_start:\n")
        f.write(json.dumps(optimizer_diagnostics(optimizers), indent=2) + "\n")
        f.write("train_log:\n")


def append_results_log(
    results_path: str,
    iter_i: int,
    global_step: int,
    loss: torch.Tensor,
    psnr: torch.Tensor,
    lr_map: Mapping[str, float],
    optimizers: Mapping[str, Optional[torch.optim.Optimizer]],
    iter_time_sec: float,
) -> None:
    peak = cuda_peak_mb(DEVICE)
    with open(results_path, "a", encoding="utf-8") as f:
        lr_text = " ".join([f"{name}_lr={lr:.8e}" for name, lr in sorted(lr_map.items())])
        mem_text = f" cuda_peak_mb={peak:.3f}" if peak is not None else ""
        f.write(
            f"iter={iter_i} global_step={global_step} loss={loss.item():.10f} "
            f"psnr={psnr.item():.6f} {lr_text} iter_time_sec={iter_time_sec:.6f}{mem_text}\n"
        )
        diag = optimizer_diagnostics(optimizers)
        if diag:
            f.write("  optimizer_diag=" + json.dumps(diag, sort_keys=True) + "\n")


def compute_ssim_torch(img1, img2, data_range=1.0, window_size=11, sigma=1.5):
    """Compute mean RGB SSIM for two H x W x C tensors in [0, 1]."""
    img1 = img1.clamp(0.0, 1.0).float()
    img2 = img2.clamp(0.0, 1.0).float()

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(f"SSIM expects H x W x C images, got {img1.shape} and {img2.shape}")
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM image shape mismatch: {img1.shape} vs {img2.shape}")

    h, w, c = img1.shape
    window_size = min(int(window_size), h, w)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return torch.tensor(1.0 if torch.allclose(img1, img2) else 0.0, device=img1.device, dtype=img1.dtype)

    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)

    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, window_size, window_size)
    kernel = kernel.expand(c, 1, window_size, window_size).contiguous()

    padding = window_size // 2
    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    y_pad = F.pad(y, (padding, padding, padding, padding), mode="reflect")

    mu_x = F.conv2d(x_pad, kernel, padding=0, groups=c)
    mu_y = F.conv2d(y_pad, kernel, padding=0, groups=c)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x_pad * x_pad, kernel, padding=0, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y_pad * y_pad, kernel, padding=0, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x_pad * y_pad, kernel, padding=0, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean()


def append_final_eval(
    results_path: str,
    final_iter: int,
    global_step: int,
    mean_psnr: float,
    mean_ssim: float,
    per_view_metrics: Sequence[Mapping[str, float]],
    elapsed_sec: float,
) -> None:
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("final_eval:\n")
        f.write(f"  final_iter: {final_iter}\n")
        f.write(f"  global_step: {global_step}\n")
        f.write(f"  mean_psnr: {mean_psnr:.6f}\n")
        f.write(f"  mean_ssim: {mean_ssim:.6f}\n")
        f.write(f"  num_test_views: {len(per_view_metrics)}\n")
        f.write(f"  elapsed_sec: {elapsed_sec:.3f}\n")
        f.write("  per_view:\n")
        for metric in per_view_metrics:
            f.write(f"    index={metric['index']} psnr={metric['psnr']:.6f} ssim={metric['ssim']:.6f}\n")


def write_final_eval_json(
    output_path: str,
    args,
    final_iter: int,
    global_step: int,
    mean_psnr: float,
    mean_ssim: float,
    per_view_metrics: Sequence[Mapping[str, float]],
    elapsed_sec: float,
    optimizers: Mapping[str, Optional[torch.optim.Optimizer]],
) -> None:
    payload = {
        "expname": args.expname,
        "optimizer": args.optimizer,
        "dataset_type": args.dataset_type,
        "datadir": args.datadir,
        "seed": args.seed,
        "final_iter": final_iter,
        "global_step": global_step,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "num_test_views": len(per_view_metrics),
        "per_view": list(per_view_metrics),
        "elapsed_sec": elapsed_sec,
        "cuda_peak_mb": cuda_peak_mb(DEVICE),
        "optimizer_diagnostics": optimizer_diagnostics(optimizers),
        "args": vars(args),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def evaluate_testset(images, poses, i_test, hwf, K, chunk, render_kwargs_test):
    per_view_metrics = []
    psnrs = []
    ssims = []

    with torch.no_grad():
        for img_i in tqdm(i_test, desc="Final eval testset"):
            target = as_float_tensor(images[int(img_i)], device=DEVICE)
            pose = as_float_tensor(poses[int(img_i), :3, :4], device=DEVICE)
            rgb, disp, acc, extras = render(hwf[0], hwf[1], K, chunk=chunk, c2w=pose, **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))
            ssim = compute_ssim_torch(rgb, target)

            psnr_value = float(psnr.detach().cpu().item())
            ssim_value = float(ssim.detach().cpu().item())
            psnrs.append(psnr_value)
            ssims.append(ssim_value)
            per_view_metrics.append({"index": int(img_i), "psnr": psnr_value, "ssim": ssim_value})

    return float(np.mean(psnrs)), float(np.mean(ssims)), per_view_metrics


# -----------------------------------------------------------------------------
# Model creation and checkpoint loading
# -----------------------------------------------------------------------------


def create_nerf(args):
    """Instantiate NeRF model(s), render kwargs, and requested optimizer(s)."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(DEVICE)

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(DEVICE)

    all_models = [model, model_fine]
    grad_vars = [p for m in all_models if m is not None for p in m.parameters() if p.requires_grad]
    main_params, aux_params = collect_param_splits(all_models)

    print(
        "Param split: "
        f"main_hidden_2d={len(main_params)} tensors/{count_params(main_params)} params, "
        f"aux={len(aux_params)} tensors/{count_params(aux_params)} params, "
        f"all={len(grad_vars)} tensors/{count_params(grad_vars)} params"
    )

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    optimizers: Dict[str, Optional[torch.optim.Optimizer]]
    if args.optimizer == "adam":
        optimizers = {
            "adam": torch.optim.Adam(params=list(grad_vars), lr=args.lrate, betas=(0.9, 0.999)),
        }
    elif args.optimizer == "sgd_split":
        if len(main_params) == 0:
            raise RuntimeError("sgd_split selected but no hidden 2D parameters were found.")
        optimizers = {
            "main": torch.optim.SGD(
                params=list(main_params),
                lr=args.sgd_lrate,
                momentum=args.sgd_momentum,
                nesterov=args.sgd_nesterov,
                weight_decay=args.sgd_decay,
            ),
            "aux": torch.optim.Adam(params=list(aux_params), lr=args.lrate, betas=(0.9, 0.999)) if aux_params else None,
        }
    elif args.optimizer == "mofa_split":
        if len(main_params) == 0:
            raise RuntimeError("mofa_split selected but no hidden 2D parameters were found.")
        optimizers = {
            "main": MomentumFactorizedSGD(
                params=list(main_params),
                lr=args.mofa_lrate,
                rank=args.mofa_rank,
                beta=args.mofa_beta,
                eta1=args.mofa_eta1,
                eta2=args.mofa_eta2,
                use_current_projection=args.mofa_use_current_projection,
                use_ones_for_nonzero_s=not args.mofa_use_reciprocal_s,
                eps=args.mofa_eps,
                uv_eps=args.mofa_uv_eps,
                nesterov=args.mofa_nesterov,
                max_value=args.mofa_max_value,
                weight_decay=args.mofa_decay,
                init_from_grad=True,
            ),
            "aux": torch.optim.Adam(params=list(aux_params), lr=args.lrate, betas=(0.9, 0.999)) if aux_params else None,
        }
    else:
        raise ValueError(f"Unknown optimizer mode: {args.optimizer}")

    start_iter = 0
    global_step = 0
    basedir = args.basedir
    expname = args.expname

    exp_dir = os.path.join(basedir, expname)
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    elif os.path.isdir(exp_dir):
        ckpts = [os.path.join(exp_dir, f) for f in sorted(os.listdir(exp_dir)) if f.endswith(".tar")]
    else:
        ckpts = []

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = safe_torch_load(ckpt_path, map_location=DEVICE, allow_unsafe=args.allow_unsafe_checkpoint_load)

        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

        if not args.no_reload_optimizer:
            load_optimizer_state_dicts(ckpt, optimizers, args.optimizer)

        if "iter_i" in ckpt:
            start_iter = int(ckpt["iter_i"]) + 1
            global_step = int(ckpt.get("global_step", start_iter))
        elif "global_step" in ckpt:
            # Backward-compatible interpretation for old checkpoints.  The old
            # script saved global_step before incrementing it, so this is only an
            # approximation; the safer format above is used for new checkpoints.
            start_iter = int(ckpt["global_step"]) + 1
            global_step = start_iter
        print(f"Resume start_iter={start_iter}, global_step={global_step}")

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = dict(render_kwargs_train)
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start_iter, global_step, grad_vars, optimizers


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------


def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, required=True, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="request deterministic torch algorithms when possible")

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument("--netdepth_fine", type=int, default=8, help="layers in fine network")
    parser.add_argument("--netwidth_fine", type=int, default=256, help="channels per layer in fine network")
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4, help="batch size: random rays per gradient step")
    parser.add_argument("--N_iters", type=int, default=100000, help="total number of training iterations")
    parser.add_argument("--lrate", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential LR decay in 1000-step units")
    parser.add_argument("--chunk", type=int, default=1024 * 32, help="rays processed in parallel")
    parser.add_argument("--netchunk", type=int, default=1024 * 64, help="points sent through network in parallel")
    parser.add_argument("--no_batching", action="store_true", help="only take random rays from one image at a time")
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument("--no_reload_optimizer", action="store_true", help="reload model weights but skip optimizer states")
    parser.add_argument("--allow_unsafe_checkpoint_load", action="store_true", help="allow full pickle checkpoint load; only for trusted files")
    parser.add_argument("--ft_path", type=str, default=None, help="specific checkpoint to reload")

    # optimizer options
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mofa_split",
        choices=["adam", "sgd_split", "mofa_split"],
        help="optimizer comparison mode",
    )

    parser.add_argument("--sgd_lrate", type=float, default=5e-4, help="SGD learning rate for sgd_split main branch")
    parser.add_argument("--sgd_momentum", type=float, default=0.9, help="SGD momentum for sgd_split main branch")
    parser.add_argument("--sgd_decay", type=float, default=0.0, help="SGD weight decay for sgd_split main branch")
    parser.add_argument("--sgd_nesterov", action="store_true", help="use PyTorch SGD Nesterov in sgd_split")

    parser.add_argument("--mofa_lrate", type=float, default=5e-4, help="MoFaSGD learning rate for hidden 2D branch")
    parser.add_argument("--mofa_rank", type=int, default=16, help="target rank for factorized momentum")
    parser.add_argument("--mofa_beta", type=float, default=0.9, help="momentum decay for MoFaSGD")
    parser.add_argument("--mofa_decay", type=float, default=0.0, help="AdamW-style weight decay for MoFaSGD branch")
    parser.add_argument("--mofa_eta1", type=float, default=1.0, help="scale for low-rank MoFaSGD update")
    parser.add_argument("--mofa_eta2", type=float, default=0.0, help="scale for orthogonal complement gradient update")
    parser.add_argument("--mofa_eps", type=float, default=1e-4, help="epsilon for reciprocal-S variant only")
    parser.add_argument("--mofa_uv_eps", type=float, default=0.0, help="near-zero mask for default U V^T variant")
    parser.add_argument("--mofa_max_value", type=float, default=5.0, help="clip value for reciprocal singular values")
    parser.add_argument("--mofa_use_current_projection", action="store_true", help="use updated U/V for eta2 complement")
    parser.add_argument("--mofa_use_reciprocal_s", action="store_true", help="use U diag(1/S) V^T instead of default U V^T")
    parser.add_argument("--mofa_nesterov", action="store_true", help="apply nonstandard Nesterov-like second factor update")

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help="coarse samples per ray")
    parser.add_argument("--N_importance", type=int, default=0, help="additional fine samples per ray")
    parser.add_argument("--perturb", type=float, default=1.0, help="0 for no jitter, 1 for stratified jitter")
    parser.add_argument("--use_viewdirs", action="store_true", help="use full 5D input")
    parser.add_argument("--i_embed", type=int, default=0, help="0 for positional encoding, -1 for none")
    parser.add_argument("--multires", type=int, default=10, help="log2 max freq for 3D location")
    parser.add_argument("--multires_views", type=int, default=4, help="log2 max freq for view direction")
    parser.add_argument("--raw_noise_std", type=float, default=0.0, help="stddev of density noise")

    parser.add_argument("--render_only", action="store_true", help="do not optimize, render from checkpoint")
    parser.add_argument("--render_test", action="store_true", help="render test set instead of render_poses path")
    parser.add_argument("--render_factor", type=int, default=0, help="downsampling factor for preview rendering")

    # cropping
    parser.add_argument("--precrop_iters", type=int, default=0, help="steps for central crops")
    parser.add_argument("--precrop_frac", type=float, default=0.5, help="fraction of image for central crops")

    # dataset options
    parser.add_argument("--dataset_type", type=str, default="llff", help="llff / blender / deepvoxels / LINEMOD")
    parser.add_argument("--testskip", type=int, default=8, help="load 1/N images from test/val sets")
    parser.add_argument("--shape", type=str, default="greek", help="deepvoxels shape")
    parser.add_argument("--white_bkgd", action="store_true", help="render synthetic data on white background")
    parser.add_argument("--half_res", action="store_true", help="load blender synthetic at 400x400")
    parser.add_argument("--factor", type=int, default=8, help="LLFF downsample factor")
    parser.add_argument("--no_ndc", action="store_true", help="disable normalized device coordinates")
    parser.add_argument("--lindisp", action="store_true", help="sample linearly in inverse depth")
    parser.add_argument("--spherify", action="store_true", help="LLFF spherical 360 scenes")
    parser.add_argument("--llffhold", type=int, default=8, help="take every 1/N image as LLFF test set")

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=2000, help="console/result log frequency")
    parser.add_argument("--i_img", type=int, default=100000, help="unused legacy image logging frequency")
    parser.add_argument("--i_weights", type=int, default=100000, help="checkpoint saving frequency")
    parser.add_argument("--i_testset", type=int, default=100000, help="testset render saving frequency")
    parser.add_argument("--i_video", type=int, default=100000, help="render_poses video frequency")

    return parser


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train():
    global DEVICE

    parser = config_parser()
    args = parser.parse_args()
    DEVICE = resolve_device(args.device)
    set_random_seed(args.seed, args.deterministic)

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    K = None
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=0.75, spherify=args.spherify
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = float(np.ndarray.min(bds) * 0.9)
            far = float(np.ndarray.max(bds) * 1.0)
        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.0
        far = 6.0
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "LINEMOD":
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip
        )
        print(f"Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}")
        print(f"[CHECK HERE] near: {near}, far: {far}.")
        i_train, i_val, i_test = i_split
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape, basedir=args.datadir, testskip=args.testskip)
        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = float(hemi_R - 1.0)
        far = float(hemi_R + 1.0)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, float(focal)]

    if K is None:
        K = np.array([[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        K = clone_intrinsics_np(K)

    if args.render_test:
        render_poses = np.array(poses[i_test])

    basedir = args.basedir
    expname = args.expname
    exp_dir = os.path.join(basedir, expname)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.txt"), "w", encoding="utf-8") as f:
        for arg in sorted(vars(args)):
            f.write(f"{arg} = {getattr(args, arg)}\n")
    if args.config is not None:
        with open(os.path.join(exp_dir, "config.txt"), "w", encoding="utf-8") as f:
            with open(args.config, "r", encoding="utf-8") as cfg:
                f.write(cfg.read())

    render_kwargs_train, render_kwargs_test, start_iter, global_step, grad_vars, optimizers = create_nerf(args)

    results_path = os.path.join(exp_dir, "results.txt")
    init_results_log(results_path, args, optimizers, start_iter)

    bds_dict = {"near": near, "far": far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    render_poses = as_float_tensor(render_poses, device=DEVICE)

    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            gt_images = images[i_test] if args.render_test else None
            testsavedir = os.path.join(
                exp_dir, "renderonly_{}_{:06d}".format("test" if args.render_test else "path", start_iter)
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)
            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=gt_images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8)
        return

    N_rand = args.N_rand
    use_batching = not args.no_batching
    rays_rgb = None
    i_batch = 0

    if use_batching:
        print("get rays")
        rays = np.stack([get_rays_np_safe(H, W, K, p) for p in poses[:, :3, :4]], axis=0)
        print("done, concats")
        rays_rgb_np = np.concatenate([rays, images[:, None]], axis=1)
        rays_rgb_np = np.transpose(rays_rgb_np, [0, 2, 3, 1, 4])
        rays_rgb_np = np.stack([rays_rgb_np[i] for i in i_train], axis=0)
        rays_rgb_np = np.reshape(rays_rgb_np, [-1, 3, 3]).astype(np.float32)
        print("shuffle rays")
        rng = np.random.default_rng(args.seed)
        rng.shuffle(rays_rgb_np, axis=0)
        print("done")
        rays_rgb = as_float_tensor(rays_rgb_np, device=DEVICE)
        images = as_float_tensor(images, device=DEVICE)

    poses = as_float_tensor(poses, device=DEVICE)

    if start_iter >= args.N_iters:
        print(f"start_iter={start_iter} >= N_iters={args.N_iters}; skipping training and running final evaluation.")

    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    last_loss = None
    last_psnr = None
    last_iter = start_iter - 1

    for iter_i in trange(start_iter, args.N_iters, desc="train"):
        sync_if_cuda(DEVICE)
        time0 = time.time()
        lr_map = set_optimizer_lrs(args, optimizers, global_step)

        if use_batching:
            if rays_rgb is None:
                raise RuntimeError("rays_rgb was not initialized")
            batch = rays_rgb[i_batch : i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0], device=rays_rgb.device)
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            img_i = int(np.random.choice(i_train))
            target = as_float_tensor(images[img_i], device=DEVICE)
            pose = poses[img_i, :3, :4]

            if N_rand is None:
                raise RuntimeError("N_rand=None is not supported in this comparison script.")

            rays_o, rays_d = get_rays_torch_safe(H, W, K, pose)

            if iter_i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=DEVICE),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=DEVICE),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                if iter_i == start_iter:
                    print(f"[Config] Center cropping {2 * dH} x {2 * dW} enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(
                    torch.meshgrid(
                        torch.arange(H, device=DEVICE, dtype=torch.float32),
                        torch.arange(W, device=DEVICE, dtype=torch.float32),
                        indexing="ij",
                    ),
                    dim=-1,
                )

            coords = torch.reshape(coords, [-1, 2])
            select_inds = torch.randperm(coords.shape[0], device=DEVICE)[:N_rand]
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays = torch.stack([rays_o, rays_d], dim=0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]

        rgb, disp, acc, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=iter_i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        zero_optimizers(optimizers)
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        step_optimizers(optimizers)
        global_step += 1
        last_loss = loss.detach()
        last_psnr = psnr.detach()
        last_iter = iter_i

        sync_if_cuda(DEVICE)
        iter_time = time.time() - time0

        if args.i_weights > 0 and global_step % args.i_weights == 0:
            path = os.path.join(exp_dir, f"{iter_i:06d}.tar")
            torch.save(
                {
                    "iter_i": iter_i,
                    "global_step": global_step,
                    "optimizer_mode": args.optimizer,
                    "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                    "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict()
                    if render_kwargs_train["network_fine"] is not None
                    else None,
                    "optimizer_state_dicts": optimizer_state_dicts(optimizers),
                    "args": vars(args),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if args.i_video > 0 and global_step % args.i_video == 0:
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(exp_dir, f"{expname}_spiral_{iter_i:06d}_")
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=30, quality=8)
            denom = np.max(disps)
            imageio.mimwrite(moviebase + "disp.mp4", to8b(disps / max(denom, 1e-10)), fps=30, quality=8)

        if args.i_testset > 0 and global_step % args.i_testset == 0:
            testsavedir = os.path.join(exp_dir, f"testset_{iter_i:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                render_path(poses[i_test], hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print("Saved test set")

        if args.i_print > 0 and (global_step % args.i_print == 0 or iter_i == start_iter):
            tqdm.write(f"[TRAIN] Iter: {iter_i} Global: {global_step} Loss: {loss.item()} PSNR: {psnr.item()}")
            append_results_log(results_path, iter_i, global_step, loss, psnr, lr_map, optimizers, iter_time)

            img_i = int(np.random.choice(i_val))
            target_val = as_float_tensor(images[img_i], device=DEVICE)
            pose_val = poses[img_i, :3, :4]
            with torch.no_grad():
                rgb_val, disp_val, acc_val, extras_val = render(H, W, K, chunk=args.chunk, c2w=pose_val, **render_kwargs_test)
            val_loss = img2mse(rgb_val, target_val)
            val_psnr = mse2psnr(val_loss)
            tqdm.write(f"[VAL]   Iter: {iter_i} Global: {global_step} PSNR: {val_psnr.item()}")

    if last_loss is not None and last_psnr is not None:
        append_results_log(
            results_path,
            last_iter,
            global_step,
            last_loss,
            last_psnr,
            set_optimizer_lrs(args, optimizers, global_step),
            optimizers,
            0.0,
        )

    final_eval_start = time.time()
    final_mean_psnr, final_mean_ssim, final_per_view_metrics = evaluate_testset(
        images=images,
        poses=poses,
        i_test=i_test,
        hwf=hwf,
        K=K,
        chunk=args.chunk,
        render_kwargs_test=render_kwargs_test,
    )
    final_eval_elapsed = time.time() - final_eval_start
    append_final_eval(
        results_path=results_path,
        final_iter=last_iter,
        global_step=global_step,
        mean_psnr=final_mean_psnr,
        mean_ssim=final_mean_ssim,
        per_view_metrics=final_per_view_metrics,
        elapsed_sec=final_eval_elapsed,
    )
    write_final_eval_json(
        output_path=os.path.join(exp_dir, "metrics_final.json"),
        args=args,
        final_iter=last_iter,
        global_step=global_step,
        mean_psnr=final_mean_psnr,
        mean_ssim=final_mean_ssim,
        per_view_metrics=final_per_view_metrics,
        elapsed_sec=final_eval_elapsed,
        optimizers=optimizers,
    )
    tqdm.write(
        f"[FINAL EVAL] Iter: {last_iter} Global: {global_step} "
        f"mean PSNR: {final_mean_psnr:.6f} mean SSIM: {final_mean_ssim:.6f}"
    )


if __name__ == "__main__":
    train()

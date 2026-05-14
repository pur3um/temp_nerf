#!/usr/bin/env python3
"""
Checkpoint-based NeRF test-set evaluator.

Purpose
-------
Evaluate an existing NeRF checkpoint such as 200000.tar on the test split and
write PSNR/SSIM summaries without re-training.

This file is designed to work with NeRF-style modules that expose:
  - config_parser()
  - get_embedder()
  - NeRF
  - run_network()

Typical usage from the repository root:

  python eval_nerf_checkpoint_safe.py \
    --config ./logs/Quan_aux-muon/chair_aux-muon_mlr3e-3/config.txt \
    --checkpoint ./logs/Quan_aux-muon/chair_aux-muon_mlr3e-3/200000.tar \
    --out_dir ./logs/Quan_aux-muon/chair_aux-muon_mlr3e-3/eval_ckpt_200000 \
    --project_root . \
    --nerf_module run_nerf_ranksched \
    --ssim_impl torch \
    --overwrite \
    --no_save_png

Notes
-----
1. Metrics are computed from float RGB renders before PNG quantization.
2. This evaluator loads model weights only. It does not construct/load optimizer state.
3. The renderer below avoids global torch.set_default_tensor_type() and creates tensors
   on the active ray/device explicitly.
4. Only evaluate checkpoints that you trust. PyTorch .tar checkpoints are pickle-based
   in older workflows. This script tries torch.load(..., weights_only=True) first.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore


# --------------------------------------------------------------------------------------
# General utilities
# --------------------------------------------------------------------------------------


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def parse_checkpoint_file_step(path: Path) -> Optional[int]:
    return int(path.stem) if re.fullmatch(r"\d+", path.stem) else None


def to8b(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (255.0 * np.clip(x, 0.0, 1.0)).astype(np.uint8)


def sanitize_rgb01(img: Any, *, name: str) -> np.ndarray:
    arr = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    if arr.ndim != 3:
        raise RuntimeError(f"{name} must be HxWxC, got shape={arr.shape}")
    if arr.shape[-1] < 3:
        raise RuntimeError(f"{name} must have at least 3 channels, got shape={arr.shape}")
    arr = arr[..., :3].astype(np.float32, copy=False)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"{name} contains non-finite values. count={bad}, shape={arr.shape}")
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * np.log10(float(mse) + 1e-10))


def resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={device_arg}, but CUDA is not available.")
    return torch.device(device_arg)


def set_eval_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def safe_torch_load_checkpoint(path: Path, device: torch.device, allow_unsafe: bool) -> Dict[str, Any]:
    """Load checkpoint. Prefer weights_only=True when the installed PyTorch supports it."""
    kwargs = {"map_location": device}
    supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters

    if supports_weights_only:
        try:
            return torch.load(str(path), weights_only=True, **kwargs)
        except Exception as exc:
            if not allow_unsafe:
                raise RuntimeError(
                    "torch.load(..., weights_only=True) failed. If this is your own trusted "
                    "checkpoint and you accept pickle loading risk, rerun with "
                    "--allow_unsafe_checkpoint_load.\n"
                    f"checkpoint={path}\nerror={exc}"
                ) from exc
            print("[WARN] weights_only=True load failed; falling back to unsafe pickle load because "
                  "--allow_unsafe_checkpoint_load was set.")
            return torch.load(str(path), **kwargs)

    # Older PyTorch has no weights_only argument. This is necessarily pickle loading.
    if not allow_unsafe:
        print("[WARN] Installed PyTorch has no torch.load(weights_only=...). "
              "Loading uses legacy pickle behavior. Only use trusted checkpoints.")
    return torch.load(str(path), **kwargs)


# --------------------------------------------------------------------------------------
# Checkpoint/config path handling
# --------------------------------------------------------------------------------------


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_checkpoint_path(
    checkpoint_arg: str,
    nerf_args: argparse.Namespace,
    config_path: Path,
    project_root: Path,
) -> Path:
    if not checkpoint_arg or checkpoint_arg.strip() in {"", "None", "none", "null"}:
        raise ValueError("--checkpoint is required and cannot be empty.")

    raw = Path(checkpoint_arg).expanduser()
    candidates: List[Path] = [raw]

    if not raw.is_absolute():
        candidates += [Path.cwd() / raw, project_root / raw, config_path.parent / raw]

    # If only '200000' or '200000.tar' is given, also search basedir/expname.
    basedir = getattr(nerf_args, "basedir", None)
    expname = getattr(nerf_args, "expname", None)
    if basedir and expname:
        basedir_path = Path(str(basedir)).expanduser()
        exp_dirs = [basedir_path / str(expname)]
        if not basedir_path.is_absolute():
            exp_dirs += [
                Path.cwd() / basedir_path / str(expname),
                project_root / basedir_path / str(expname),
                config_path.parent / basedir_path / str(expname),
            ]
        for exp_dir in exp_dirs:
            candidates.append(exp_dir / raw)
            if raw.suffix == "":
                candidates.append(exp_dir / f"{raw.name}.tar")
                if raw.name.isdigit():
                    candidates.append(exp_dir / f"{int(raw.name):06d}.tar")
            if raw.suffix == ".tar" and raw.stem.isdigit():
                candidates.append(exp_dir / f"{int(raw.stem):06d}.tar")

    for candidate in unique_paths(candidates):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    pretty = "\n  ".join(str(p) for p in unique_paths(candidates))
    raise FileNotFoundError("Could not resolve --checkpoint. Checked:\n  " + pretty)


def prepare_render_dir(render_dir: Optional[Path], out_dir: Path, overwrite: bool) -> Optional[Path]:
    if render_dir is None:
        return None
    render_dir = render_dir.resolve()
    out_dir = out_dir.resolve()
    if render_dir.exists() and any(render_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Render directory already exists and is not empty: {render_dir}\n"
                "Use --overwrite or choose a new --out_dir/--render_dir."
            )
        if not is_relative_to(render_dir, out_dir):
            raise RuntimeError(
                f"Refusing to delete --render_dir outside --out_dir.\n"
                f"render_dir={render_dir}\nout_dir={out_dir}"
            )
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    return render_dir


# --------------------------------------------------------------------------------------
# Dataset loading with the same split/near/far/background logic as standard NeRF scripts
# --------------------------------------------------------------------------------------


def load_data_like_train(args: argparse.Namespace) -> Dict[str, Any]:
    K = None

    if args.dataset_type == "llff":
        from load_llff import load_llff_data
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]
        if args.no_ndc:
            near = float(np.ndarray.min(bds) * 0.9)
            far = float(np.ndarray.max(bds) * 1.0)
        else:
            near = 0.0
            far = 1.0

    elif args.dataset_type == "blender":
        from load_blender import load_blender_data
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir,
            args.half_res,
            args.testskip,
        )
        _i_train, _i_val, i_test = i_split
        near, far = 2.0, 6.0
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "LINEMOD":
        from load_LINEMOD import load_LINEMOD_data
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir,
            args.half_res,
            args.testskip,
        )
        _i_train, _i_val, i_test = i_split
        near, far = float(near), float(far)
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":
        from load_deepvoxels import load_dv_data
        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape,
            basedir=args.datadir,
            testskip=args.testskip,
        )
        _i_train, _i_val, i_test = i_split
        hemi_R = float(np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1)))
        near, far = hemi_R - 1.0, hemi_R + 1.0

    else:
        raise ValueError(
            f"Unsupported dataset_type={args.dataset_type!r}. "
            "Supported: llff, blender, LINEMOD, deepvoxels."
        )

    images = np.asarray(images).astype(np.float32)
    if images.ndim != 4:
        raise RuntimeError(f"Loaded images must be NxHxWxC, got shape={images.shape}")
    if images.shape[-1] > 3:
        images = images[..., :3]
    if not np.isfinite(images).all():
        raise RuntimeError("Loaded GT images contain non-finite values.")
    if images.max(initial=0.0) > 1.5:
        images = images / 255.0
    images = np.clip(images, 0.0, 1.0).astype(np.float32, copy=False)

    H, W, focal = hwf
    H, W, focal = int(H), int(W), float(focal)
    hwf = [H, W, focal]

    if K is None:
        K = np.array(
            [[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    else:
        K = np.asarray(K, dtype=np.float32)

    i_test = np.asarray(i_test, dtype=np.int64)
    if i_test.size == 0:
        raise RuntimeError("Test split is empty. Check dataset config/testskip/llffhold.")

    return {
        "images": images,
        "poses": np.asarray(poses, dtype=np.float32),
        "render_poses": np.asarray(render_poses, dtype=np.float32),
        "hwf": hwf,
        "K": K,
        "i_test": i_test,
        "near": float(near),
        "far": float(far),
    }


# --------------------------------------------------------------------------------------
# Safe renderer: no global default CUDA tensor dependency
# --------------------------------------------------------------------------------------


def get_rays_safe(H: int, W: int, K: Any, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = c2w.device
    dtype = c2w.dtype
    K_t = torch.as_tensor(K, device=device, dtype=dtype)

    i, j = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="ij",
    )
    i = i.t()
    j = j.t()

    dirs = torch.stack(
        [
            (i - K_t[0, 2]) / K_t[0, 0],
            -(j - K_t[1, 2]) / K_t[1, 1],
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def ndc_rays_safe(
    H: int,
    W: int,
    focal: float,
    near: float,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -(2.0 * focal / W) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -(2.0 * focal / H) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -(2.0 * focal / W) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -(2.0 * focal / H) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)
    return rays_o, rays_d


def sample_pdf_safe(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    pytest: bool = False,
) -> torch.Tensor:
    device = weights.device
    dtype = weights.dtype

    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device, dtype=dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device, dtype=dtype)

    if pytest:
        np.random.seed(0)
        u_np = np.random.rand(*list(cdf.shape[:-1]), N_samples)
        u = torch.as_tensor(u_np, device=device, dtype=dtype)

    u = u.contiguous()
    inds = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def raw2outputs_safe(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
    pytest: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw2alpha = lambda raw_sigma, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw_sigma) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])

    noise: Any = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn_like(raw[..., 3]) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise_np = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.as_tensor(noise_np, device=raw.device, dtype=raw.dtype)

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    trans = torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha.shape[0], 1), device=alpha.device, dtype=alpha.dtype),
                1.0 - alpha + 1e-10,
            ],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    denom = acc_map.clamp_min(1e-10)
    disp_map = 1.0 / torch.clamp(depth_map / denom, min=1e-10)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays_safe(
    ray_batch: torch.Tensor,
    network_fn: torch.nn.Module,
    network_query_fn: Any,
    N_samples: int,
    retraw: bool = False,
    lindisp: bool = False,
    perturb: float = 0.0,
    N_importance: int = 0,
    network_fine: Optional[torch.nn.Module] = None,
    white_bkgd: bool = False,
    raw_noise_std: float = 0.0,
    verbose: bool = False,
    pytest: bool = False,
) -> Dict[str, torch.Tensor]:
    del verbose
    N_rays = ray_batch.shape[0]
    device = ray_batch.device
    dtype = ray_batch.dtype

    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device, dtype=dtype)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device, dtype=dtype)
        if pytest:
            np.random.seed(0)
            t_rand = torch.as_tensor(np.random.rand(*list(z_vals.shape)), device=device, dtype=dtype)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_safe(
        raw,
        z_vals,
        rays_d,
        raw_noise_std=raw_noise_std,
        white_bkgd=white_bkgd,
        pytest=pytest,
    )

    ret: Dict[str, torch.Tensor] = {
        "rgb_map": rgb_map,
        "disp_map": disp_map,
        "acc_map": acc_map,
    }

    if retraw:
        ret["raw"] = raw

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf_safe(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        ).detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs_safe(
            raw,
            z_vals,
            rays_d,
            raw_noise_std=raw_noise_std,
            white_bkgd=white_bkgd,
            pytest=pytest,
        )
        ret.update(
            {
                "rgb_map": rgb_map,
                "disp_map": disp_map,
                "acc_map": acc_map,
                "rgb0": rgb_map_0,
                "disp0": disp_map_0,
                "acc0": acc_map_0,
                "z_std": torch.std(z_samples, dim=-1, unbiased=False),
            }
        )

    if retraw:
        ret["raw"] = raw

    for k, v in ret.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise RuntimeError(f"Numerical error: render output {k} contains NaN or Inf.")

    return ret


def batchify_rays_safe(rays_flat: torch.Tensor, chunk: int, **kwargs: Any) -> Dict[str, torch.Tensor]:
    all_ret: Dict[str, List[torch.Tensor]] = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_safe(rays_flat[i : i + chunk], **kwargs)
        for k, v in ret.items():
            all_ret.setdefault(k, []).append(v)
    return {k: torch.cat(v, dim=0) for k, v in all_ret.items()}


def render_safe(
    H: int,
    W: int,
    K: Any,
    chunk: int = 1024 * 32,
    rays: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    c2w: Optional[torch.Tensor] = None,
    ndc: bool = True,
    near: float = 0.0,
    far: float = 1.0,
    use_viewdirs: bool = False,
    c2w_staticcam: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if c2w is not None:
        c2w = c2w.to(dtype=torch.float32)
        rays_o, rays_d = get_rays_safe(H, W, K, c2w)
    else:
        if rays is None:
            raise ValueError("Either c2w or rays must be provided.")
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays_safe(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape

    if ndc:
        focal = float(np.asarray(K)[0, 0])
        rays_o, rays_d = ndc_rays_safe(H, W, focal, 1.0, rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near_t = torch.full_like(rays_d[..., :1], float(near))
    far_t = torch.full_like(rays_d[..., :1], float(far))
    rays_flat = torch.cat([rays_o, rays_d, near_t, far_t], dim=-1)

    if use_viewdirs:
        rays_flat = torch.cat([rays_flat, viewdirs], dim=-1)

    all_ret = batchify_rays_safe(rays_flat, chunk, **kwargs)

    for k in list(all_ret.keys()):
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    ret_list = [all_ret[k] for k in ["rgb_map", "disp_map", "acc_map"]]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in ["rgb_map", "disp_map", "acc_map"]}
    return ret_list[0], ret_list[1], ret_list[2], ret_dict


def render_path_safe(
    render_poses: torch.Tensor,
    hwf: Sequence[Any],
    K: Any,
    chunk: int,
    render_kwargs: Dict[str, Any],
    savedir: Optional[str] = None,
    render_factor: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W, focal = hwf
    H, W, focal = int(H), int(W), float(focal)
    K_eff = np.asarray(K, dtype=np.float32).copy()

    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
        K_eff[:2, :] /= float(render_factor)

    if savedir is not None:
        Path(savedir).mkdir(parents=True, exist_ok=True)

    rgbs: List[np.ndarray] = []
    disps: List[np.ndarray] = []

    for i, c2w in enumerate(render_poses):
        print(f"[RENDER] {i + 1}/{len(render_poses)}")
        rgb, disp, acc, _ = render_safe(
            H,
            W,
            K_eff,
            chunk=chunk,
            c2w=c2w[:3, :4],
            **render_kwargs,
        )
        rgb_np = rgb.detach().cpu().numpy().astype(np.float32)
        disp_np = disp.detach().cpu().numpy().astype(np.float32)
        rgbs.append(rgb_np)
        disps.append(disp_np)

        if savedir is not None:
            imageio.imwrite(os.path.join(savedir, f"{i:03d}.png"), to8b(rgb_np))

    return np.stack(rgbs, axis=0), np.stack(disps, axis=0)


# --------------------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------------------


def gaussian_kernel(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()


def ssim_torch_cpu(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = sanitize_rgb01(pred, name="pred_for_ssim")
    gt = sanitize_rgb01(gt, name="gt_for_ssim")
    if pred.shape != gt.shape:
        raise RuntimeError(f"SSIM shape mismatch: pred={pred.shape}, gt={gt.shape}")

    x = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    y = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()
    channels = x.shape[1]
    window_size = min(11, x.shape[-2], x.shape[-1])
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return 1.0 if np.allclose(pred, gt) else 0.0

    kernel = gaussian_kernel(window_size=window_size, sigma=1.5, channels=channels, device=x.device)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


def compute_ssim(nerf_mod: Any, pred: np.ndarray, gt: np.ndarray, impl: str, device: torch.device) -> float:
    if impl == "train":
        if hasattr(nerf_mod, "compute_ssim"):
            return float(nerf_mod.compute_ssim(pred, gt))
        if hasattr(nerf_mod, "compute_ssim_torch"):
            pred_t = torch.from_numpy(sanitize_rgb01(pred, name="pred_train_ssim")).to(device)
            gt_t = torch.from_numpy(sanitize_rgb01(gt, name="gt_train_ssim")).to(device)
            return float(nerf_mod.compute_ssim_torch(pred_t, gt_t).detach().cpu().item())
        raise RuntimeError(
            "--ssim_impl train was requested, but the NeRF module has neither "
            "compute_ssim() nor compute_ssim_torch(). Use --ssim_impl torch."
        )

    if impl == "torch":
        return ssim_torch_cpu(pred, gt)

    if impl == "skimage":
        try:
            from skimage.metrics import structural_similarity
        except ImportError as exc:
            raise RuntimeError("--ssim_impl skimage requires scikit-image.") from exc
        try:
            return float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0))
        except TypeError:  # older skimage
            return float(structural_similarity(gt, pred, multichannel=True, data_range=1.0))

    raise ValueError(f"Unknown SSIM implementation: {impl}")


# --------------------------------------------------------------------------------------
# NeRF construction and checkpoint load
# --------------------------------------------------------------------------------------


def create_nerf_for_eval(
    nerf_mod: Any,
    args: argparse.Namespace,
    checkpoint_path: Path,
    device: torch.device,
    allow_unsafe_checkpoint_load: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    required = ["get_embedder", "NeRF", "run_network"]
    missing = [name for name in required if not hasattr(nerf_mod, name)]
    if missing:
        raise RuntimeError(f"NeRF module is missing required symbols: {missing}")

    embed_fn, input_ch = nerf_mod.get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = nerf_mod.get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = nerf_mod.NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(device)

    model_fine = None
    if args.N_importance > 0:
        model_fine = nerf_mod.NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(device)

    network_query_fn = lambda inputs, viewdirs, network_fn: nerf_mod.run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    print(f"[LOAD] checkpoint = {checkpoint_path}")
    ckpt = safe_torch_load_checkpoint(checkpoint_path, device, allow_unsafe_checkpoint_load)

    if "network_fn_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint has no network_fn_state_dict: {checkpoint_path}")

    ckpt_fine = ckpt.get("network_fine_state_dict", None)
    if model_fine is None and ckpt_fine is not None:
        raise RuntimeError(
            "Checkpoint contains network_fine_state_dict, but current args.N_importance == 0. "
            "The eval config likely does not match the train config."
        )
    if model_fine is not None and ckpt_fine is None:
        raise RuntimeError(
            "Current args.N_importance > 0, but checkpoint has no network_fine_state_dict. "
            "The eval config likely does not match the train config."
        )

    model.load_state_dict(ckpt["network_fn_state_dict"], strict=True)
    if model_fine is not None:
        model_fine.load_state_dict(ckpt_fine, strict=True)

    model.eval()
    if model_fine is not None:
        model_fine.eval()

    render_kwargs_test = {
        "network_query_fn": network_query_fn,
        "perturb": False,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": 0.0,
    }

    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_test["ndc"] = False
        render_kwargs_test["lindisp"] = args.lindisp

    metadata = {
        "checkpoint_global_step": json_safe(ckpt.get("global_step", None)),
        "checkpoint_optimizer_name": json_safe(ckpt.get("optimizer_name", None)),
        "checkpoint_effective_total_iters": json_safe(ckpt.get("effective_total_iters", None)),
        "checkpoint_keys": sorted(str(k) for k in ckpt.keys()),
    }

    return render_kwargs_test, metadata


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and evaluate a NeRF checkpoint on the configured test split."
    )
    parser.add_argument("--config", required=True, type=str, help="Path to config.txt or original config file.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint .tar or step name.")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for metrics.")
    parser.add_argument("--project_root", default=".", type=str, help="Repo root containing run_nerf*.py and loaders.")
    parser.add_argument("--nerf_module", default="run_nerf_ranksched", type=str, help="Python module name without .py.")
    parser.add_argument("--device", default="auto", type=str, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--render_dir", default=None, type=str, help="Where to save rendered PNGs.")
    parser.add_argument("--no_save_png", action="store_true", help="Do not save rendered PNGs.")
    parser.add_argument("--save_float_npy", action="store_true", help="Save float32 rendered RGBs as .npy.")
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse/deletion of non-empty render_dir under out_dir.")
    parser.add_argument("--expected_step", default=None, type=int, help="Optional check against checkpoint filename step.")
    parser.add_argument("--strict_global_step", action="store_true", help="Require checkpoint['global_step'] == expected_step.")
    parser.add_argument(
        "--ssim_impl",
        default="torch",
        choices=["torch", "train", "skimage"],
        help="SSIM implementation. Default torch is self-contained.",
    )
    parser.add_argument("--compute_lpips", action="store_true", help="Also compute LPIPS if the NeRF module exposes compute_lpips().")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--scene", default="", type=str)
    parser.add_argument("--variant", default="", type=str)
    parser.add_argument("--seed", default=None, type=int, help="Override eval seed. Default uses config seed if present, else 0.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--allow_unsafe_checkpoint_load",
        action="store_true",
        help="Allow fallback to torch pickle loading if weights_only=True fails. Use only for trusted checkpoints.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    eval_parser = build_eval_parser()
    eval_args, forwarded_nerf_args = eval_parser.parse_known_args(argv)

    config_path = Path(eval_args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_root = Path(eval_args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    old_cwd = Path.cwd()
    os.chdir(project_root)

    try:
        nerf_mod = importlib.import_module(eval_args.nerf_module)
        if not hasattr(nerf_mod, "config_parser"):
            raise RuntimeError(f"Module {eval_args.nerf_module!r} has no config_parser().")

        nerf_parser = nerf_mod.config_parser()
        nerf_cli = ["--config", str(config_path)] + list(forwarded_nerf_args)
        nerf_args = nerf_parser.parse_args(nerf_cli)

        device = resolve_device(eval_args.device)
        if hasattr(nerf_mod, "device"):
            nerf_mod.device = device

        checkpoint_path = resolve_checkpoint_path(
            eval_args.checkpoint,
            nerf_args,
            config_path,
            project_root,
        )
        checkpoint_file_step = parse_checkpoint_file_step(checkpoint_path)
        if eval_args.expected_step is not None and checkpoint_file_step != int(eval_args.expected_step):
            raise RuntimeError(
                f"--expected_step={eval_args.expected_step}, but checkpoint filename step is "
                f"{checkpoint_file_step}. checkpoint={checkpoint_path}"
            )

        seed = int(eval_args.seed if eval_args.seed is not None else getattr(nerf_args, "seed", 0))
        deterministic = bool(eval_args.deterministic or getattr(nerf_args, "deterministic", False))
        set_eval_seed(seed, deterministic)

        out_dir = Path(eval_args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if eval_args.no_save_png:
            render_dir = None
        else:
            render_dir = (
                Path(eval_args.render_dir).expanduser().resolve()
                if eval_args.render_dir
                else out_dir / f"renders_ckpt_{checkpoint_path.stem}"
            )
            render_dir = prepare_render_dir(render_dir, out_dir, eval_args.overwrite)

        print("=" * 100)
        print("[EVAL CONFIG]")
        print(f"  original_cwd      = {old_cwd}")
        print(f"  project_root      = {project_root}")
        print(f"  nerf_module       = {eval_args.nerf_module}")
        print(f"  config            = {config_path}")
        print(f"  checkpoint        = {checkpoint_path}")
        print(f"  checkpoint_step   = {checkpoint_file_step}")
        print(f"  out_dir           = {out_dir}")
        print(f"  render_dir        = {render_dir}")
        print(f"  device            = {device}")
        print(f"  ssim_impl         = {eval_args.ssim_impl}")
        print("=" * 100)

        data = load_data_like_train(nerf_args)
        images = data["images"]
        poses = data["poses"]
        hwf = data["hwf"]
        K = data["K"]
        i_test = data["i_test"]
        near, far = float(data["near"]), float(data["far"])

        gt_images = images[i_test].astype(np.float32, copy=False)
        gt_images = np.stack(
            [sanitize_rgb01(img, name=f"gt[{idx}]") for idx, img in enumerate(gt_images)],
            axis=0,
        )
        render_poses_test = torch.as_tensor(poses[i_test], device=device, dtype=torch.float32)

        render_kwargs_test, ckpt_meta = create_nerf_for_eval(
            nerf_mod,
            nerf_args,
            checkpoint_path,
            device,
            allow_unsafe_checkpoint_load=eval_args.allow_unsafe_checkpoint_load,
        )
        render_kwargs_test.update({"near": near, "far": far})

        ckpt_global_step = ckpt_meta.get("checkpoint_global_step", None)
        if checkpoint_file_step is not None and ckpt_global_step is not None and checkpoint_file_step != ckpt_global_step:
            print(
                f"[WARN] checkpoint filename step ({checkpoint_file_step}) != "
                f"checkpoint['global_step'] ({ckpt_global_step})."
            )
        if eval_args.strict_global_step and eval_args.expected_step is not None:
            if ckpt_global_step != int(eval_args.expected_step):
                raise RuntimeError(
                    f"--strict_global_step requires checkpoint['global_step'] == "
                    f"{eval_args.expected_step}, but got {ckpt_global_step}."
                )

        with torch.no_grad():
            rgbs, disps = render_path_safe(
                render_poses_test,
                hwf,
                K,
                int(nerf_args.chunk),
                render_kwargs_test,
                savedir=str(render_dir) if render_dir is not None else None,
                render_factor=0,
            )

        if rgbs.shape[0] != gt_images.shape[0]:
            raise RuntimeError(f"Rendered count mismatch: rendered={rgbs.shape[0]}, gt={gt_images.shape[0]}")

        if eval_args.compute_lpips and not hasattr(nerf_mod, "compute_lpips"):
            raise RuntimeError("--compute_lpips was set, but the NeRF module has no compute_lpips().")

        rows: List[Dict[str, Any]] = []
        mses: List[float] = []
        psnrs: List[float] = []
        ssims: List[float] = []
        lpips_values: List[float] = []

        for idx, (pred_raw, gt_raw) in enumerate(zip(rgbs, gt_images)):
            pred = sanitize_rgb01(pred_raw, name=f"pred[{idx}]")
            gt = sanitize_rgb01(gt_raw, name=f"gt[{idx}]")
            if pred.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch at index={idx}: pred={pred.shape}, gt={gt.shape}")
            mse = float(np.mean(np.square(pred - gt)))
            psnr = psnr_from_mse(mse)
            ssim = compute_ssim(nerf_mod, pred, gt, impl=eval_args.ssim_impl, device=device)
            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
            row = {
                "index": idx,
                "test_index": int(i_test[idx]),
                "image": f"{idx:03d}.png",
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            }
            if eval_args.compute_lpips:
                lpips_value = float(nerf_mod.compute_lpips(pred, gt, net=eval_args.lpips_net))
                lpips_values.append(lpips_value)
                row["lpips"] = lpips_value
            rows.append(row)

        mean_mse = float(np.mean(mses))
        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))
        mean_lpips = float(np.mean(lpips_values)) if lpips_values else None
        ckpt_label = checkpoint_path.stem

        per_image_csv = out_dir / f"per_image_metrics_{ckpt_label}.csv"
        with open(per_image_csv, "w", newline="") as f:
            per_image_fields = ["index", "test_index", "image", "mse", "psnr", "ssim"]
            if eval_args.compute_lpips:
                per_image_fields.append("lpips")
            writer = csv.DictWriter(f, fieldnames=per_image_fields)
            writer.writeheader()
            writer.writerows(rows)

        float_npy_path = None
        if eval_args.save_float_npy:
            float_npy_path = out_dir / f"rgbs_float32_{ckpt_label}.npy"
            np.save(float_npy_path, rgbs.astype(np.float32, copy=False))

        summary = {
            "method": eval_args.method,
            "scene": eval_args.scene,
            "variant": eval_args.variant,
            "metric_source": "float32_render_output_before_png_quantization",
            "config": str(config_path),
            "project_root": str(project_root),
            "nerf_module": eval_args.nerf_module,
            "checkpoint": str(checkpoint_path),
            "checkpoint_file_step": checkpoint_file_step,
            **ckpt_meta,
            "dataset_type": getattr(nerf_args, "dataset_type", None),
            "datadir": getattr(nerf_args, "datadir", None),
            "half_res": getattr(nerf_args, "half_res", None),
            "white_bkgd": getattr(nerf_args, "white_bkgd", None),
            "testskip": getattr(nerf_args, "testskip", None),
            "factor": getattr(nerf_args, "factor", None),
            "llffhold": getattr(nerf_args, "llffhold", None),
            "no_ndc": getattr(nerf_args, "no_ndc", None),
            "spherify": getattr(nerf_args, "spherify", None),
            "near": near,
            "far": far,
            "hwf": hwf,
            "K": K,
            "test_indices": i_test,
            "num_images": len(rows),
            "mean_mse": mean_mse,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "ssim_impl": eval_args.ssim_impl,
            "lpips_net": eval_args.lpips_net if eval_args.compute_lpips else None,
            "render_dir": str(render_dir) if render_dir is not None else None,
            "per_image_csv": str(per_image_csv),
            "float_npy": str(float_npy_path) if float_npy_path is not None else None,
            "out_dir": str(out_dir),
            "forwarded_nerf_args": list(forwarded_nerf_args),
            "effective_nerf_args": vars(nerf_args),
            "seed": seed,
            "deterministic": deterministic,
            "device": str(device),
        }

        metrics_json = out_dir / f"metrics_{ckpt_label}.json"
        with open(metrics_json, "w") as f:
            json.dump(json_safe(summary), f, indent=2)

        summary_csv = out_dir / f"summary_metrics_{ckpt_label}.csv"
        with open(summary_csv, "w", newline="") as f:
            fields = [
                "method",
                "scene",
                "variant",
                "num_images",
                "mean_mse",
                "mean_psnr",
                "mean_ssim",
                "mean_lpips",
                "checkpoint",
                "checkpoint_file_step",
                "checkpoint_global_step",
                "config",
                "render_dir",
                "metric_source",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: json_safe(summary.get(key, None)) for key in fields})

        # Simple txt file for direct reading.
        result_txt = out_dir / f"result_{ckpt_label}.txt"
        with open(result_txt, "w") as f:
            f.write(f"status: OK\n")
            f.write(f"checkpoint: {checkpoint_path}\n")
            f.write(f"checkpoint_file_step: {checkpoint_file_step}\n")
            f.write(f"checkpoint_global_step: {ckpt_global_step}\n")
            f.write(f"num_images: {len(rows)}\n")
            f.write(f"mean_mse: {mean_mse:.10f}\n")
            f.write(f"mean_psnr: {mean_psnr:.6f}\n")
            f.write(f"mean_ssim: {mean_ssim:.6f}\n")
            if mean_lpips is not None:
                f.write(f"mean_lpips: {mean_lpips:.6f}\n")
            f.write(f"metrics_json: {metrics_json}\n")
            f.write(f"per_image_csv: {per_image_csv}\n")

        print("=" * 100)
        print(f"[METRIC DONE] {eval_args.method} {eval_args.scene} {eval_args.variant}")
        print(f"  checkpoint      = {checkpoint_path}")
        print(f"  file_step       = {checkpoint_file_step}")
        print(f"  global_step     = {ckpt_global_step}")
        print(f"  num_images      = {len(rows)}")
        print(f"  mean_MSE        = {mean_mse:.10f}")
        print(f"  mean_PSNR       = {mean_psnr:.6f}")
        print(f"  mean_SSIM       = {mean_ssim:.6f}")
        if mean_lpips is not None:
            print(f"  mean_LPIPS      = {mean_lpips:.6f}")
        print(f"  metrics_json    = {metrics_json}")
        print(f"  per_image_csv   = {per_image_csv}")
        print(f"  result_txt      = {result_txt}")
        if render_dir is not None:
            print(f"  rendered_pngs   = {render_dir}")
        print("=" * 100)

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
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
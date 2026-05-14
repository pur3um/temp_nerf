#!/usr/bin/env python3
"""
Safe checkpoint-based NeRF test-set evaluator.

핵심:
1. --checkpoint를 필수로 받고 실제 NeRF weight를 직접 로드합니다.
2. renderonly_test_100000 같은 폴더를 자동 추정하지 않습니다.
3. PNG를 다시 읽어서 metric을 계산하지 않고, 렌더링 직후 float32 RGB로 PSNR/SSIM을 계산합니다.
4. train 코드와 같은 test split, white_bkgd, near/far, K, pose 로직을 사용합니다.
5. 이미지 개수나 shape mismatch는 조용히 넘기지 않고 에러로 막습니다.

기본 module은 run_nerf_ranksched.py 입니다.
이 파일을 프로젝트 root에 두거나, --project_root /path/to/project 를 넘기세요.
"""

from __future__ import annotations

import argparse
import csv
import importlib
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


# --------------------------------------------------------------------------------------
# Utilities
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
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def parse_checkpoint_file_step(path: Path) -> Optional[int]:
    if re.fullmatch(r"\d+", path.stem):
        return int(path.stem)
    return None


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


def psnr_from_mse_train_style(mse: float) -> float:
    # train 코드의 render_path_with_metrics와 동일:
    # -10 * log10(mse + 1e-10)
    return float(-10.0 * np.log10(float(mse) + 1e-10))


# --------------------------------------------------------------------------------------
# SSIM
# 기본값은 train module의 compute_ssim 사용.
# 없거나 강제로 torch/skimage를 쓰고 싶으면 --ssim_impl torch 또는 skimage 사용.
# --------------------------------------------------------------------------------------


def gaussian_kernel(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim_torch_cpu(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = sanitize_rgb01(pred, name="pred_for_ssim")
    gt = sanitize_rgb01(gt, name="gt_for_ssim")

    if pred.shape != gt.shape:
        raise RuntimeError(f"SSIM shape mismatch: pred={pred.shape}, gt={gt.shape}")

    x = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    y = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()

    channels = x.shape[1]
    window_size = 11
    kernel = gaussian_kernel(
        window_size=window_size,
        sigma=1.5,
        channels=channels,
        device=x.device,
    )
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )

    return float(ssim_map.mean().item())


def compute_ssim(nerf_mod: Any, pred: np.ndarray, gt: np.ndarray, impl: str) -> float:
    if impl == "train":
        if not hasattr(nerf_mod, "compute_ssim"):
            raise RuntimeError(
                "--ssim_impl train was requested, but the NeRF module has no compute_ssim(). "
                "Use --ssim_impl torch or --ssim_impl skimage."
            )
        return float(nerf_mod.compute_ssim(pred, gt))

    if impl == "torch":
        return ssim_torch_cpu(pred, gt)

    if impl == "skimage":
        try:
            from skimage.metrics import structural_similarity
        except ImportError as exc:
            raise RuntimeError("--ssim_impl skimage requires scikit-image.") from exc

        try:
            return float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0))
        except TypeError:
            return float(structural_similarity(gt, pred, multichannel=True, data_range=1.0))

    raise ValueError(f"Unknown SSIM implementation: {impl}")


# --------------------------------------------------------------------------------------
# Checkpoint path handling
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
    """
    --checkpoint 허용 형태:
      - /abs/path/to/100000.tar
      - relative/path/to/100000.tar
      - 100000.tar
      - 100000

    100000 또는 100000.tar만 준 경우에는 basedir/expname 아래에서도 찾습니다.
    """

    if not checkpoint_arg or checkpoint_arg.strip() in {"", "None", "none", "null"}:
        raise ValueError("--checkpoint is required and cannot be empty.")

    raw = Path(checkpoint_arg).expanduser()
    candidates: List[Path] = []

    candidates.append(raw)

    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        candidates.append(project_root / raw)
        candidates.append(config_path.parent / raw)

    basedir = getattr(nerf_args, "basedir", None)
    expname = getattr(nerf_args, "expname", None)

    if basedir and expname:
        basedir_path = Path(str(basedir)).expanduser()

        exp_dirs = []
        exp_dirs.append(basedir_path / str(expname))

        if not basedir_path.is_absolute():
            exp_dirs.append(Path.cwd() / basedir_path / str(expname))
            exp_dirs.append(project_root / basedir_path / str(expname))
            exp_dirs.append(config_path.parent / basedir_path / str(expname))

        for exp_dir in exp_dirs:
            candidates.append(exp_dir / raw)

            if raw.suffix == "":
                candidates.append(exp_dir / f"{raw.name}.tar")
                if raw.name.isdigit():
                    candidates.append(exp_dir / f"{int(raw.name):06d}.tar")

            if raw.suffix == ".tar" and raw.stem.isdigit():
                candidates.append(exp_dir / f"{int(raw.stem):06d}.tar")

    candidates = unique_paths(candidates)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    pretty = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve --checkpoint. Checked these locations:\n  " + pretty
    )


def prepare_render_dir(render_dir: Optional[Path], out_dir: Path, overwrite: bool) -> Optional[Path]:
    if render_dir is None:
        return None

    render_dir = render_dir.resolve()
    out_dir = out_dir.resolve()

    if render_dir.exists() and any(render_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Render directory already exists and is not empty: {render_dir}\n"
                "Use a fresh --out_dir/--render_dir, or pass --overwrite."
            )

        if not is_relative_to(render_dir, out_dir):
            raise RuntimeError(
                f"Refusing to delete --render_dir outside --out_dir.\n"
                f"render_dir={render_dir}\n"
                f"out_dir={out_dir}"
            )

        shutil.rmtree(render_dir)

    render_dir.mkdir(parents=True, exist_ok=True)
    return render_dir


# --------------------------------------------------------------------------------------
# Dataset loading: train 코드의 split/background/near-far 로직과 맞춤
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

        near = 2.0
        far = 6.0

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
        near = float(near)
        far = float(far)

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
        near = hemi_R - 1.0
        far = hemi_R + 1.0

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
    H = int(H)
    W = int(W)
    focal = float(focal)
    hwf = [H, W, focal]

    if K is None:
        K = np.array(
            [
                [focal, 0.0, 0.5 * W],
                [0.0, focal, 0.5 * H],
                [0.0, 0.0, 1.0],
            ],
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
        "near": near,
        "far": far,
    }


# --------------------------------------------------------------------------------------
# Eval-only NeRF construction and checkpoint load
# optimizer를 만들거나 optimizer state를 로드하지 않습니다.
# --------------------------------------------------------------------------------------


def create_nerf_for_eval(
    nerf_mod: Any,
    args: argparse.Namespace,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    required = ["get_embedder", "NeRF", "run_network"]

    missing = [name for name in required if not hasattr(nerf_mod, name)]

    if missing:
        raise RuntimeError(f"NeRF module is missing required symbols: {missing}")

    embed_fn, input_ch = nerf_mod.get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = nerf_mod.get_embedder(
            args.multires_views,
            args.i_embed,
        )

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
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    if "network_fn_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint has no network_fn_state_dict: {checkpoint_path}")

    ckpt_fine = ckpt.get("network_fine_state_dict", None)

    if model_fine is None and ckpt_fine is not None:
        raise RuntimeError(
            "Checkpoint contains network_fine_state_dict, but current args.N_importance == 0. "
            "This usually means the eval config does not match the train config."
        )

    if model_fine is not None and ckpt_fine is None:
        raise RuntimeError(
            "Current args.N_importance > 0, but checkpoint has no network_fine_state_dict. "
            "This usually means the eval config does not match the train config."
        )

    model.load_state_dict(ckpt["network_fn_state_dict"], strict=True)

    if model_fine is not None:
        model_fine.load_state_dict(ckpt_fine, strict=True)

    model.eval()

    if model_fine is not None:
        model_fine.eval()

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

    metadata = {
        "checkpoint_global_step": json_safe(ckpt.get("global_step", None)),
        "checkpoint_optimizer_name": json_safe(ckpt.get("optimizer_name", None)),
        "checkpoint_effective_total_iters": json_safe(ckpt.get("effective_total_iters", None)),
        "checkpoint_keys": sorted(str(k) for k in ckpt.keys()),
    }

    return render_kwargs_train, render_kwargs_test, metadata


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely render and evaluate a NeRF checkpoint on the test split. "
            "Unknown args are forwarded to the NeRF config parser."
        )
    )

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument(
        "--project_root",
        default=None,
        type=str,
        help="Project root containing run_nerf_ranksched.py and loaders. Default: this script's directory.",
    )

    parser.add_argument(
        "--nerf_module",
        default="run_nerf_ranksched",
        type=str,
        help="Python module to import. Default: run_nerf_ranksched",
    )

    parser.add_argument(
        "--render_dir",
        default=None,
        type=str,
        help="Where to save rendered PNGs. Default: <out_dir>/renders_ckpt_<checkpoint_stem>.",
    )

    parser.add_argument(
        "--no_save_png",
        action="store_true",
        help="Do not save rendered PNGs. Metrics are still computed from float renders.",
    )

    parser.add_argument(
        "--save_float_npy",
        action="store_true",
        help="Save float32 rendered RGBs as .npy for exact metric rechecking.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deletion/reuse of a non-empty render_dir under out_dir.",
    )

    parser.add_argument(
        "--expected_step",
        default=None,
        type=int,
        help="Optional safety check against checkpoint filename step, e.g. 100000.",
    )

    parser.add_argument(
        "--strict_global_step",
        action="store_true",
        help=(
            "Also require checkpoint['global_step'] == --expected_step. "
            "Usually leave off because some train loops save global_step off by one."
        ),
    )

    parser.add_argument(
        "--ssim_impl",
        default="train",
        choices=["train", "torch", "skimage"],
        help="Default 'train' uses the train module's compute_ssim().",
    )

    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Also compute LPIPS if the train module provides compute_lpips().",
    )

    parser.add_argument(
        "--lpips_net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
    )

    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--scene", default="", type=str)
    parser.add_argument("--variant", default="", type=str)

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional override for eval seed. If omitted, uses NeRF config seed if present.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic CUDA/PyTorch behavior where possible.",
    )

    return parser


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


def main(argv: Optional[Sequence[str]] = None) -> None:
    eval_parser = build_eval_parser()
    eval_args, forwarded_nerf_args = eval_parser.parse_known_args(argv)

    config_path = Path(eval_args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    project_root = (
        Path(eval_args.project_root).expanduser().resolve()
        if eval_args.project_root
        else Path(__file__).resolve().parent
    )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 상대 datadir/basedir가 config 안에 있을 때 원래 train 실행 환경과 맞추기 위해 project_root로 이동.
    old_cwd = Path.cwd()
    os.chdir(project_root)

    try:
        nerf_mod = importlib.import_module(eval_args.nerf_module)

        if not hasattr(nerf_mod, "config_parser"):
            raise RuntimeError(f"Module {eval_args.nerf_module!r} has no config_parser().")

        if not hasattr(nerf_mod, "render_path"):
            raise RuntimeError(f"Module {eval_args.nerf_module!r} has no render_path().")

        nerf_parser = nerf_mod.config_parser()

        # eval script가 모르는 args는 NeRF parser로 forwarding합니다.
        # 예: --basedir, --expname, --optimizer, --train_scheduler, --N_iters 등
        nerf_cli = ["--config", str(config_path)] + list(forwarded_nerf_args)
        nerf_args = nerf_parser.parse_args(nerf_cli)

        checkpoint_path = resolve_checkpoint_path(
            eval_args.checkpoint,
            nerf_args,
            config_path,
            project_root,
        )

        checkpoint_file_step = parse_checkpoint_file_step(checkpoint_path)

        if eval_args.expected_step is not None:
            if checkpoint_file_step != int(eval_args.expected_step):
                raise RuntimeError(
                    f"--expected_step={eval_args.expected_step}, "
                    f"but checkpoint filename step is {checkpoint_file_step}. "
                    f"checkpoint={checkpoint_path}"
                )

        # metadata 명확화를 위해 설정.
        # create_nerf()는 호출하지 않지만, summary에 남습니다.
        nerf_args.ft_path = str(checkpoint_path)
        nerf_args.no_reload = False
        nerf_args.render_only = False
        nerf_args.render_test = True
        nerf_args.render_factor = 0

        seed = int(
            eval_args.seed
            if eval_args.seed is not None
            else getattr(nerf_args, "seed", 0)
        )

        deterministic = bool(
            eval_args.deterministic
            or getattr(nerf_args, "deterministic", False)
        )

        set_eval_seed(seed, deterministic)

        device = getattr(
            nerf_mod,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        if not isinstance(device, torch.device):
            device = torch.device(str(device))

        # 원래 train script는 __main__에서 CUDA FloatTensor를 default로 설정합니다.
        # render 함수 내부의 torch.Tensor([...]) device mismatch를 피하기 위해 맞춥니다.
        if device.type == "cuda" and torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

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
        print("[SAFE EVAL CONFIG]")
        print(f"  original_cwd      = {old_cwd}")
        print(f"  project_root      = {project_root}")
        print(f"  nerf_module       = {eval_args.nerf_module}")
        print(f"  config            = {config_path}")
        print(f"  checkpoint        = {checkpoint_path}")
        print(f"  checkpoint_step   = {checkpoint_file_step}")
        print(f"  out_dir           = {out_dir}")
        print(f"  render_dir        = {render_dir}")
        print(
            f"  forwarded_args    = "
            f"{' '.join(forwarded_nerf_args) if forwarded_nerf_args else '(none)'}"
        )
        print("=" * 100)

        data = load_data_like_train(nerf_args)

        images = data["images"]
        poses = data["poses"]
        hwf = data["hwf"]
        K = data["K"]
        i_test = data["i_test"]
        near = float(data["near"])
        far = float(data["far"])

        gt_images = images[i_test].astype(np.float32, copy=False)
        gt_images = np.stack(
            [
                sanitize_rgb01(img, name=f"gt[{idx}]")
                for idx, img in enumerate(gt_images)
            ],
            axis=0,
        )

        render_poses_test = torch.Tensor(poses[i_test]).to(device)

        _, render_kwargs_test, ckpt_meta = create_nerf_for_eval(
            nerf_mod,
            nerf_args,
            checkpoint_path,
            device,
        )

        render_kwargs_test.update({"near": near, "far": far})

        ckpt_global_step = ckpt_meta.get("checkpoint_global_step", None)

        if (
            checkpoint_file_step is not None
            and ckpt_global_step is not None
            and checkpoint_file_step != ckpt_global_step
        ):
            print(
                f"[WARN] checkpoint filename step ({checkpoint_file_step}) "
                f"!= checkpoint['global_step'] ({ckpt_global_step}). "
                "This can happen if the train loop saved global_step before incrementing it. "
                "Metrics will be labeled with both."
            )

        if eval_args.strict_global_step and eval_args.expected_step is not None:
            if ckpt_global_step != int(eval_args.expected_step):
                raise RuntimeError(
                    f"--strict_global_step requires checkpoint['global_step'] "
                    f"== {eval_args.expected_step}, but got {ckpt_global_step}."
                )

        # 실제 렌더링.
        # metric은 saved PNG를 다시 읽지 않고 rgbs float32로 계산합니다.
        with torch.no_grad():
            rgbs, disps = nerf_mod.render_path(
                render_poses_test,
                hwf,
                K,
                nerf_args.chunk,
                render_kwargs_test,
                gt_imgs=gt_images,
                savedir=str(render_dir) if render_dir is not None else None,
                render_factor=0,
            )

        rgbs = np.asarray(rgbs, dtype=np.float32)

        if rgbs.shape[0] != gt_images.shape[0]:
            raise RuntimeError(
                f"Rendered count mismatch: rendered={rgbs.shape[0]}, gt={gt_images.shape[0]}"
            )

        rows: List[Dict[str, Any]] = []
        mses: List[float] = []
        psnrs: List[float] = []
        ssims: List[float] = []
        lpips_scores: List[Optional[float]] = []

        if eval_args.compute_lpips and not hasattr(nerf_mod, "compute_lpips"):
            raise RuntimeError(
                "--compute_lpips was set, but the NeRF module has no compute_lpips()."
            )

        for idx, (pred_raw, gt_raw) in enumerate(zip(rgbs, gt_images)):
            pred = sanitize_rgb01(pred_raw, name=f"pred[{idx}]")
            gt = sanitize_rgb01(gt_raw, name=f"gt[{idx}]")

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch at index={idx}: pred={pred.shape}, gt={gt.shape}"
                )

            mse = float(np.mean(np.square(pred - gt)))
            psnr = psnr_from_mse_train_style(mse)
            ssim = compute_ssim(nerf_mod, pred, gt, impl=eval_args.ssim_impl)

            lpips_value: Optional[float] = None

            if eval_args.compute_lpips:
                lpips_value = float(
                    nerf_mod.compute_lpips(pred, gt, net=eval_args.lpips_net)
                )

            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
            lpips_scores.append(lpips_value)

            row = {
                "index": idx,
                "test_index": int(i_test[idx]),
                "image": f"{idx:03d}.png",
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
            }

            if eval_args.compute_lpips:
                row["lpips"] = lpips_value

            rows.append(row)

        mean_mse = float(np.mean(mses))
        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))

        mean_lpips = None

        if eval_args.compute_lpips:
            valid_lpips = [v for v in lpips_scores if v is not None]
            mean_lpips = float(np.mean(valid_lpips)) if valid_lpips else None

        ckpt_label = checkpoint_path.stem

        per_image_csv = out_dir / f"per_image_metrics_{ckpt_label}.csv"

        fieldnames = ["index", "test_index", "image", "mse", "psnr", "ssim"]

        if eval_args.compute_lpips:
            fieldnames.append("lpips")

        with open(per_image_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        }

        metrics_json = out_dir / f"metrics_{ckpt_label}.json"

        with open(metrics_json, "w") as f:
            json.dump(json_safe(summary), f, indent=2)

        summary_csv = out_dir / f"summary_metrics_{ckpt_label}.csv"

        with open(summary_csv, "w", newline="") as f:
            summary_fields = [
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

            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerow(
                {
                    key: json_safe(summary.get(key, None))
                    for key in summary_fields
                }
            )

        print("=" * 100)
        print(f"[METRIC DONE] {eval_args.method} {eval_args.scene} {eval_args.variant}")
        print(f"  checkpoint      = {checkpoint_path}")
        print(f"  file_step       = {checkpoint_file_step}")
        print(f"  global_step     = {ckpt_global_step}")
        print(f"  num_images      = {len(rows)}")
        print(f"  mean_MSE        = {mean_mse:.10f}")
        print(f"  mean_PSNR       = {mean_psnr:.6f}")
        print(f"  mean_SSIM       = {mean_ssim:.6f}")

        if eval_args.compute_lpips:
            if mean_lpips is not None:
                print(f"  mean_LPIPS      = {mean_lpips:.6f}")
            else:
                print("  mean_LPIPS      = unavailable")

        print(f"  metrics_json    = {metrics_json}")
        print(f"  per_image_csv   = {per_image_csv}")

        if render_dir is not None:
            print(f"  rendered_pngs   = {render_dir}")

        if float_npy_path is not None:
            print(f"  float_npy       = {float_npy_path}")

        print("=" * 100)

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
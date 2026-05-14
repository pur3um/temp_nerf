#!/usr/bin/env python3
"""
Batch NeRF checkpoint evaluator.

What it does
------------
Given a base directory, it recursively finds run folders that contain a target
checkpoint such as 200000.tar. For each run folder, it calls an existing
single-checkpoint evaluator, then records PSNR/SSIM into:

  1. <run_dir>/eval_result_<step>.txt
  2. <base>/batch_eval_<step>_summary.csv
  3. <base>/batch_eval_<step>_summary.txt

Expected folder layout example
------------------------------
BASE/
  chair_aux-muon_mlr3e-3/
    config.txt
    args.txt
    200000.tar
  drums_aux-muon_mlr3e-3/
    config.txt
    200000.tar

Typical command
---------------
python scripts/batch_eval_nerf_checkpoints.py \
  --base ./logs/Quan_aux-muon \
  --step 200000 \
  --eval_script ./eval_nerf_checkpoint_safe.py \
  --project_root . \
  --nerf_module run_nerf_ranksched \
  --ssim_impl torch \
  --overwrite \
  --continue_on_error

Any unknown arguments are forwarded to the single-checkpoint evaluator.
For example, if your NeRF parser supports additional flags:

python scripts/batch_eval_nerf_checkpoints.py ... -- --chunk 16384
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def checkpoint_names_from_args(step: Optional[int], checkpoint_name: Optional[str]) -> List[str]:
    names: List[str] = []

    if checkpoint_name:
        names.append(checkpoint_name)

    if step is not None:
        names.append(f"{int(step)}.tar")
        names.append(f"{int(step):06d}.tar")

    # preserve order, remove duplicates
    out: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)

    if not out:
        raise ValueError("Either --step or --checkpoint_name must be provided.")

    return out


def looks_like_run_dir(path: Path, config_names: Sequence[str]) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in config_names)


def find_run_checkpoints(
    base: Path,
    checkpoint_names: Sequence[str],
    config_names: Sequence[str],
    recursive: bool,
    include_glob: str,
    exclude_dir_keywords: Sequence[str],
) -> List[Tuple[Path, Path, Path]]:
    """
    Returns list of (run_dir, checkpoint_path, config_path).
    """

    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"--base must be a directory: {base}")

    candidates: List[Tuple[Path, Path, Path]] = []
    seen_dirs = set()

    iterator: Iterable[Path]
    if recursive:
        # Search by checkpoint filename. This handles BASE/category/run_dir/200000.tar.
        paths: List[Path] = []
        for name in checkpoint_names:
            paths.extend(base.rglob(name))
        iterator = sorted(paths)
    else:
        paths = []
        for child in sorted(base.glob(include_glob)):
            if child.is_dir():
                for name in checkpoint_names:
                    p = child / name
                    if p.is_file():
                        paths.append(p)
        iterator = sorted(paths)

    for ckpt in iterator:
        if not ckpt.is_file():
            continue

        run_dir = ckpt.parent

        if run_dir in seen_dirs:
            # If both 200000.tar and 200000.tar duplicate somehow, keep first.
            continue

        rel_parts_lower = [p.lower() for p in run_dir.relative_to(base).parts]
        if any(any(key.lower() in part for key in exclude_dir_keywords) for part in rel_parts_lower):
            continue

        if include_glob and not run_dir.name.startswith("."):
            # include_glob is primarily applied to the run folder name.
            # Path.match has surprising semantics for nested paths, so use fnmatch.
            import fnmatch
            if not fnmatch.fnmatch(run_dir.name, include_glob):
                continue

        config_path = None
        for name in config_names:
            maybe = run_dir / name
            if maybe.is_file():
                config_path = maybe
                break

        if config_path is None:
            print(f"[SKIP] checkpoint exists but no config file found: {ckpt}", file=sys.stderr)
            continue

        candidates.append((run_dir.resolve(), ckpt.resolve(), config_path.resolve()))
        seen_dirs.add(run_dir)

    return candidates


# --------------------------------------------------------------------------------------
# Evaluation invocation
# --------------------------------------------------------------------------------------


def build_eval_command(
    python_bin: str,
    eval_script: Path,
    config_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    project_root: Path,
    nerf_module: str,
    method: str,
    scene: str,
    variant: str,
    ssim_impl: str,
    expected_step: Optional[int],
    strict_global_step: bool,
    no_save_png: bool,
    save_float_npy: bool,
    overwrite: bool,
    deterministic: bool,
    compute_lpips: bool,
    lpips_net: str,
    seed: Optional[int],
    forwarded_args: Sequence[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(eval_script),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--out_dir",
        str(out_dir),
        "--project_root",
        str(project_root),
        "--nerf_module",
        nerf_module,
        "--method",
        method,
        "--scene",
        scene,
        "--variant",
        variant,
        "--ssim_impl",
        ssim_impl,
    ]

    if expected_step is not None:
        cmd += ["--expected_step", str(int(expected_step))]
    if strict_global_step:
        cmd += ["--strict_global_step"]
    if no_save_png:
        cmd += ["--no_save_png"]
    if save_float_npy:
        cmd += ["--save_float_npy"]
    if overwrite:
        cmd += ["--overwrite"]
    if deterministic:
        cmd += ["--deterministic"]
    if compute_lpips:
        cmd += ["--compute_lpips", "--lpips_net", lpips_net]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]

    cmd += list(forwarded_args)
    return cmd


def run_streaming_command(cmd: Sequence[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        log_f.write(f"# started_at: {now_str()}\n")
        log_f.write(f"# command: {shell_join(cmd)}\n")
        log_f.write("# " + "=" * 100 + "\n")
        log_f.flush()

        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)

        rc = proc.wait()
        log_f.write("# " + "=" * 100 + "\n")
        log_f.write(f"# finished_at: {now_str()}\n")
        log_f.write(f"# return_code: {rc}\n")

    return int(rc)


def find_metrics_json(out_dir: Path, checkpoint_path: Path) -> Optional[Path]:
    expected = out_dir / f"metrics_{checkpoint_path.stem}.json"
    if expected.is_file():
        return expected

    candidates = sorted(out_dir.glob("metrics_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------------------
# Result writing
# --------------------------------------------------------------------------------------


def format_per_folder_result(row: Dict[str, Any]) -> str:
    lines = [
        "=" * 100,
        "NeRF checkpoint evaluation result",
        "=" * 100,
        f"status: {row.get('status', '')}",
        f"evaluated_at: {row.get('evaluated_at', '')}",
        f"run_dir: {row.get('run_dir', '')}",
        f"relative_run_dir: {row.get('relative_run_dir', '')}",
        f"config: {row.get('config', '')}",
        f"checkpoint: {row.get('checkpoint', '')}",
        f"out_dir: {row.get('out_dir', '')}",
        f"log_file: {row.get('log_file', '')}",
        "-" * 100,
        f"num_images: {row.get('num_images', '')}",
        f"mean_mse: {row.get('mean_mse', '')}",
        f"mean_psnr: {row.get('mean_psnr', '')}",
        f"mean_ssim: {row.get('mean_ssim', '')}",
        f"mean_lpips: {row.get('mean_lpips', '')}",
        "-" * 100,
        f"metrics_json: {row.get('metrics_json', '')}",
        f"per_image_csv: {row.get('per_image_csv', '')}",
        f"render_dir: {row.get('render_dir', '')}",
        "-" * 100,
        f"command: {row.get('command', '')}",
    ]

    if row.get("error"):
        lines += ["-" * 100, f"error: {row.get('error')}"]

    lines.append("")
    return "\n".join(str(x) for x in lines)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "status",
        "relative_run_dir",
        "run_dir",
        "method",
        "scene",
        "variant",
        "num_images",
        "mean_mse",
        "mean_psnr",
        "mean_ssim",
        "mean_lpips",
        "checkpoint_file_step",
        "checkpoint_global_step",
        "checkpoint",
        "config",
        "metrics_json",
        "per_image_csv",
        "render_dir",
        "out_dir",
        "log_file",
        "error",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in fields})


def write_summary_txt(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    fail_rows = [r for r in rows if r.get("status") != "OK"]

    lines = [
        "=" * 100,
        "Batch NeRF checkpoint evaluation summary",
        "=" * 100,
        f"total: {len(rows)}",
        f"ok: {len(ok_rows)}",
        f"failed: {len(fail_rows)}",
        "",
        "[OK results]",
    ]

    for row in ok_rows:
        lines.append(
            f"{row.get('relative_run_dir')} | "
            f"PSNR={row.get('mean_psnr')} | "
            f"SSIM={row.get('mean_ssim')} | "
            f"N={row.get('num_images')}"
        )

    if fail_rows:
        lines += ["", "[Failed results]"]
        for row in fail_rows:
            lines.append(
                f"{row.get('relative_run_dir')} | "
                f"error={row.get('error')} | "
                f"log={row.get('log_file')}"
            )

    lines.append("")
    write_text(path, "\n".join(lines))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse run folders under --base, evaluate each <step>.tar checkpoint, "
            "and write PSNR/SSIM results as txt/csv. Unknown args are forwarded "
            "to the single-checkpoint evaluator."
        )
    )

    parser.add_argument("--base", required=True, type=str, help="Base directory containing run folders.")
    parser.add_argument("--step", default=200000, type=int, help="Checkpoint step, e.g. 200000.")
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Explicit checkpoint filename. If omitted, uses <step>.tar and zero-padded <step>.tar.",
    )
    parser.add_argument(
        "--eval_script",
        default="./eval_nerf_checkpoint_safe.py",
        type=str,
        help="Single-checkpoint evaluator script path.",
    )
    parser.add_argument(
        "--project_root",
        default=".",
        type=str,
        help="Project root containing NeRF module and loader files.",
    )
    parser.add_argument(
        "--nerf_module",
        default="run_nerf_ranksched",
        type=str,
        help="Python module imported by the evaluator.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        type=str,
        help="Python executable used to run the evaluator.",
    )
    parser.add_argument(
        "--config_names",
        default="config.txt,args.txt",
        type=str,
        help="Comma-separated config candidates inside each run folder. Default: config.txt,args.txt.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively find checkpoints under --base. Default: true.",
    )
    parser.add_argument(
        "--include_glob",
        default="*",
        type=str,
        help="Only evaluate run folders whose folder name matches this glob. Default: *.",
    )
    parser.add_argument(
        "--exclude_dir_keywords",
        default="testset_,renderonly_,eval_ckpt_,renders_ckpt_,launcher_logs",
        type=str,
        help="Comma-separated keywords; paths containing these are ignored.",
    )
    parser.add_argument(
        "--out_dir_name",
        default=None,
        type=str,
        help="Output dir name inside each run folder. Default: eval_ckpt_<checkpoint_stem>.",
    )
    parser.add_argument(
        "--result_txt_name",
        default=None,
        type=str,
        help="Per-folder txt filename. Default: eval_result_<checkpoint_stem>.txt.",
    )
    parser.add_argument(
        "--summary_prefix",
        default=None,
        type=str,
        help="Summary filename prefix in --base. Default: batch_eval_<checkpoint_or_step>.",
    )
    parser.add_argument(
        "--ssim_impl",
        default="torch",
        choices=["train", "torch", "skimage"],
        help="SSIM implementation passed to evaluator. Default: torch.",
    )
    parser.add_argument("--strict_global_step", action="store_true")
    parser.add_argument("--save_png", action="store_true", help="Save rendered PNGs. Default: do not save PNGs.")
    parser.add_argument("--save_float_npy", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow evaluator to overwrite render dir under out_dir.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument(
        "--gpu",
        default=None,
        type=str,
        help="Optional physical GPU id. Sets CUDA_VISIBLE_DEVICES for all evaluations, e.g. --gpu 0.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue evaluating other folders even if one folder fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print discovered commands; do not execute.",
    )

    return parser


def infer_labels(base: Path, run_dir: Path) -> Tuple[str, str, str]:
    """
    Conservative label inference. All metrics still include the exact folder path.
    - scene: first token before '_' in the run folder name
    - method: parent folder name if nested, otherwise run folder name
    - variant: full run folder name
    """
    try:
        rel = run_dir.relative_to(base)
    except ValueError:
        rel = Path(run_dir.name)

    variant = run_dir.name
    scene = variant.split("_")[0] if "_" in variant else variant
    method = rel.parts[0] if len(rel.parts) > 1 else variant
    return method, scene, variant


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)

    # If the user uses "--" before forwarded args, argparse leaves it in unknown args.
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    base = Path(args.base).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    if not eval_script.is_file():
        raise FileNotFoundError(f"Evaluator script not found: {eval_script}")
    if not project_root.is_dir():
        raise NotADirectoryError(f"Project root not found: {project_root}")

    checkpoint_names = checkpoint_names_from_args(args.step, args.checkpoint_name)
    config_names = [x.strip() for x in args.config_names.split(",") if x.strip()]
    exclude_keywords = [x.strip() for x in args.exclude_dir_keywords.split(",") if x.strip()]

    runs = find_run_checkpoints(
        base=base,
        checkpoint_names=checkpoint_names,
        config_names=config_names,
        recursive=args.recursive,
        include_glob=args.include_glob,
        exclude_dir_keywords=exclude_keywords,
    )

    print("=" * 100)
    print("[BATCH EVAL DISCOVERY]")
    print(f"base             = {base}")
    print(f"checkpoint_names = {checkpoint_names}")
    print(f"config_names     = {config_names}")
    print(f"eval_script      = {eval_script}")
    print(f"project_root     = {project_root}")
    print(f"nerf_module      = {args.nerf_module}")
    print(f"num_runs         = {len(runs)}")
    print("=" * 100)

    if not runs:
        print("No matching run folders found.", file=sys.stderr)
        return 2

    env = os.environ.copy()
    if args.gpu is not None and args.gpu.strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[ENV] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    rows: List[Dict[str, Any]] = []

    for idx, (run_dir, checkpoint_path, config_path) in enumerate(runs, start=1):
        rel_run_dir = run_dir.relative_to(base)
        method, scene, variant = infer_labels(base, run_dir)

        ckpt_stem = checkpoint_path.stem
        out_dir_name = args.out_dir_name or f"eval_ckpt_{ckpt_stem}"
        result_txt_name = args.result_txt_name or f"eval_result_{ckpt_stem}.txt"
        out_dir = run_dir / out_dir_name
        log_file = out_dir / f"eval_stdout_{ckpt_stem}.log"
        result_txt = run_dir / result_txt_name

        cmd = build_eval_command(
            python_bin=args.python,
            eval_script=eval_script,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            project_root=project_root,
            nerf_module=args.nerf_module,
            method=method,
            scene=scene,
            variant=variant,
            ssim_impl=args.ssim_impl,
            expected_step=args.step,
            strict_global_step=args.strict_global_step,
            no_save_png=not args.save_png,
            save_float_npy=args.save_float_npy,
            overwrite=args.overwrite,
            deterministic=args.deterministic,
            compute_lpips=args.compute_lpips,
            lpips_net=args.lpips_net,
            seed=args.seed,
            forwarded_args=forwarded_args,
        )

        row: Dict[str, Any] = {
            "status": "PENDING",
            "evaluated_at": now_str(),
            "index": idx,
            "relative_run_dir": str(rel_run_dir),
            "run_dir": str(run_dir),
            "method": method,
            "scene": scene,
            "variant": variant,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "out_dir": str(out_dir),
            "log_file": str(log_file),
            "command": shell_join(cmd),
        }

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(runs)}] {rel_run_dir}")
        print(f"checkpoint = {checkpoint_path}")
        print(f"config     = {config_path}")
        print(f"out_dir    = {out_dir}")
        print(f"command    = {shell_join(cmd)}")
        print("=" * 100)

        if args.dry_run:
            row["status"] = "DRY_RUN"
            rows.append(row)
            write_text(result_txt, format_per_folder_result(row))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        rc = run_streaming_command(cmd, log_file, env=env)

        if rc != 0:
            row["status"] = "FAIL"
            row["error"] = f"Evaluator returned non-zero exit code: {rc}"
            rows.append(row)
            write_text(result_txt, format_per_folder_result(row))
            if not args.continue_on_error:
                break
            continue

        metrics_json = find_metrics_json(out_dir, checkpoint_path)
        if metrics_json is None:
            row["status"] = "FAIL"
            row["error"] = f"Evaluator finished but metrics_*.json was not found in {out_dir}"
            rows.append(row)
            write_text(result_txt, format_per_folder_result(row))
            if not args.continue_on_error:
                break
            continue

        try:
            metrics = read_json(metrics_json)
        except Exception as exc:
            row["status"] = "FAIL"
            row["error"] = f"Could not read metrics json: {metrics_json}: {exc}"
            rows.append(row)
            write_text(result_txt, format_per_folder_result(row))
            if not args.continue_on_error:
                break
            continue

        row.update(
            {
                "status": "OK",
                "metrics_json": str(metrics_json),
                "per_image_csv": metrics.get("per_image_csv", ""),
                "render_dir": metrics.get("render_dir", ""),
                "num_images": metrics.get("num_images", ""),
                "mean_mse": metrics.get("mean_mse", ""),
                "mean_psnr": metrics.get("mean_psnr", ""),
                "mean_ssim": metrics.get("mean_ssim", ""),
                "mean_lpips": metrics.get("mean_lpips", ""),
                "checkpoint_file_step": metrics.get("checkpoint_file_step", ""),
                "checkpoint_global_step": metrics.get("checkpoint_global_step", ""),
            }
        )

        rows.append(row)
        write_text(result_txt, format_per_folder_result(row))

        print(
            f"[OK] {rel_run_dir} | "
            f"PSNR={row.get('mean_psnr')} | "
            f"SSIM={row.get('mean_ssim')} | "
            f"txt={result_txt}"
        )

    summary_base = args.summary_prefix
    if summary_base is None:
        label = args.checkpoint_name.replace(".tar", "") if args.checkpoint_name else str(args.step)
        summary_base = f"batch_eval_{label}"

    summary_csv = base / f"{summary_base}_summary.csv"
    summary_txt = base / f"{summary_base}_summary.txt"
    write_summary_csv(summary_csv, rows)
    write_summary_txt(summary_txt, rows)

    ok = sum(1 for r in rows if r.get("status") == "OK")
    failed = sum(1 for r in rows if r.get("status") == "FAIL")

    print("\n" + "=" * 100)
    print("[BATCH EVAL DONE]")
    print(f"total_written = {len(rows)}")
    print(f"ok            = {ok}")
    print(f"failed        = {failed}")
    print(f"summary_csv   = {summary_csv}")
    print(f"summary_txt   = {summary_txt}")
    print("=" * 100)

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python batch_eval_nerf_checkpoints.py \
  --base logs/mofa \
  --step 100000 \
  --eval_script ./eval_nerf_checkpoint_safe.py \
  --project_root . \
  --nerf_module run_nerf_ranksched \
  --gpu 7 \
  --ssim_impl torch \
  --overwrite \
  --continue_on_error
"""
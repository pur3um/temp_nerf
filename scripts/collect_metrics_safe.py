#!/usr/bin/env python3
"""Collect final NeRF comparison metrics into CSV.

Usage:
    python scripts/collect_metrics_safe.py logs/exp_a/metrics_final.json logs/exp_b/metrics_final.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: collect_metrics_safe.py <metrics_final.json> [<metrics_final.json> ...]", file=sys.stderr)
        return 2

    rows = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARN] missing: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "expname": payload.get("expname"),
                "optimizer": payload.get("optimizer"),
                "dataset_type": payload.get("dataset_type"),
                "datadir": payload.get("datadir"),
                "seed": payload.get("seed"),
                "final_iter": payload.get("final_iter"),
                "global_step": payload.get("global_step"),
                "mean_psnr": payload.get("mean_psnr"),
                "mean_ssim": payload.get("mean_ssim"),
                "elapsed_sec": payload.get("elapsed_sec"),
                "cuda_peak_mb": payload.get("cuda_peak_mb"),
                "metrics_path": str(path),
            }
        )

    fieldnames = [
        "expname",
        "optimizer",
        "dataset_type",
        "datadir",
        "seed",
        "final_iter",
        "global_step",
        "mean_psnr",
        "mean_ssim",
        "elapsed_sec",
        "cuda_peak_mb",
        "metrics_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Safe MoFaSGD NeRF comparison code

## Files

- `run_nerf_mofasgd_safe.py`: safe NeRF training/evaluation entry point with optimizer modes `adam`, `sgd_split`, and `mofa_split`.
- `optims/mofasgd_safe.py`: safer MoFaSGD implementation.
- `scripts/run_compare_blender_safe.sh`: runs Adam, split-SGD, and split-MoFaSGD on a Blender scene.
- `scripts/run_compare_llff_safe.sh`: runs Adam, split-SGD, and split-MoFaSGD on an LLFF scene.
- `scripts/run_one_safe.sh`: single-run wrapper.
- `scripts/collect_metrics_safe.py`: collects `metrics_final.json` files into CSV.

## Install / placement

Copy the files into the root of the MoFaSGD/NeRF repository:

```bash
cp run_nerf_mofasgd_safe.py /path/to/MoFaSGD/
cp -r optims /path/to/MoFaSGD/
cp -r scripts /path/to/MoFaSGD/
cd /path/to/MoFaSGD
```

The repository still needs the original files such as `run_nerf_helpers.py`, `load_blender.py`, `load_llff.py`, and the dataset directories.

## Blender comparison example

```bash
DATADIR=./data/nerf_synthetic/lego \
N_ITERS=100000 \
SEED=0 \
bash scripts/run_compare_blender_safe.sh
```

This creates three experiments in `./logs_mofasgd_safe`:

1. `*_adam`
2. `*_sgd_split`
3. `*_mofa_split`

and writes a CSV summary such as:

```text
logs_mofasgd_safe/lego_safe_YYYYMMDD_HHMMSS_summary.csv
```

## LLFF comparison example

```bash
DATADIR=./data/llff/fern \
N_ITERS=100000 \
SEED=0 \
bash scripts/run_compare_llff_safe.sh
```

## Single MoFaSGD run

```bash
OPTIMIZER=mofa_split \
EXP=lego_mofa_rank16 \
DATADIR=./data/nerf_synthetic/lego \
N_ITERS=100000 \
bash scripts/run_one_safe.sh
```

## Important comparison notes

- Use the same `SEED`, `N_ITERS`, `N_rand`, `N_samples`, `N_importance`, `chunk`, and `netchunk` for every optimizer.
- The `sgd_split` run is included as a control for the parameter split itself. Without this control, the comparison may conflate MoFaSGD with the decision to keep biases/output heads on Adam.
- Check `results.txt` and `metrics_final.json` for `optimizer_diagnostics`. If MoFaSGD fallback counts are high, wall-clock claims should be interpreted carefully.
- The script uses `torch.load(..., weights_only=True)` by default. Use `--allow_unsafe_checkpoint_load` only for checkpoints you created yourself.

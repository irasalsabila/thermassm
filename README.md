# ThermaSSM

Structurally Stable Energy-Balance State Space Models for Long-Horizon Temperature Modeling.

See [`plans/prd.md`](plans/prd.md) for the full (revised) specification and [`plans/todo.md`](plans/todo.md) for the task list.

## Layout

```
thermassm/            # Python package
  config.py           # Physics / model / data / train config
  data/               # insolation, ERA5 download, synthetic gen, dataset
  models/             # S4D, EBM prior, PhysSSM, baselines
  losses.py           # composite + ablation losses
  metrics.py          # RMSE, MAE, drift, PSD, CSI95
  rollout.py          # autoregressive multi-year rollout
  train.py            # training loop
  experiment.py       # orchestration helpers
  ablations.py        # ablation model + configs
  figures.py          # paper figures
scripts/              # entry-point CLI scripts
plans/                # PRD + todo
results/              # generated metrics, tables, figures (gitignored)
```

## Install

```bash
pip install -r requirements.txt
```

Core requirements: `torch`, `numpy`, `pandas`, `scipy`. For real ERA5 data you additionally need `xarray`, `netCDF4`, `zarr`, `gcsfs` (all in `requirements.txt`).

## Quick start (synthetic data, offline smoke test)

```bash
python3 scripts/train.py --model physssm --synthetic --device cpu --epochs 20
python3 scripts/evaluate.py --models physssm --synthetic --device cpu
python3 scripts/ablate.py --synthetic --device cpu --epochs 20
python3 scripts/figures.py --synthetic --device cpu
```

## Full benchmark (actual results → tables)

Run the four stages (or `--stage all`), then build the tables from the results:

```bash
# Synthetic data (no download needed)
python3 scripts/run_benchmark.py --stage table1 --synthetic --device cpu --epochs 30
python3 scripts/run_benchmark.py --stage table2 --synthetic --device cpu --epochs 30
python3 scripts/run_benchmark.py --stage table3 --synthetic --device cpu --epochs 30
python3 scripts/run_benchmark.py --stage table4 --synthetic --device cpu

# Real ERA5 data
python3 scripts/download_data.py                 # ~700 MB → data/weatherbench_data/
python3 scripts/run_benchmark.py --stage all --device cuda --epochs 50

# Generate markdown + CSV tables from actual results
python3 scripts/make_tables.py
```

Outputs land in `results/`:
- `benchmark_table{1,2,3,4}.json` — raw computed metrics
- `benchmark_tables.md` — all four tables (actual values)
- `table*_*.csv` — CSV copies

> The tables in `plans/prd.md` are **illustrative layouts with `TBD`** — no fabricated numbers. Only `results/benchmark_tables.md` reflects actual computed values.

## HPC / GPU notes

- Use `--device cuda` on a GPU node. `run_benchmark.py --stage table4` reports peak VRAM only on CUDA (`N/A (CPU)` otherwise).
- Increase `--epochs` (50–100) for converged numbers; 30 epochs is a smoke level.
- Each stage writes its own JSON and can be run as a separate Slurm job; `make_tables.py` picks up whichever stages are present.

Example Slurm job:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --time=04:00:00 --mem=16G
module load python
pip install -r requirements.txt
python3 scripts/run_benchmark.py --stage all --device cuda --epochs 50
python3 scripts/make_tables.py
```

## Reproduce a single model

```bash
python3 scripts/train.py --model <name> [--synthetic] [--device cuda] [--epochs 50]
```

`<name>` ∈ `physssm`, `lstm`, `gru`, `rnn`, `pint-lstm`, `pint-gru`, `patchtst`, `vanilla_s4d`, `climode`.

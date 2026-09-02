# ThermaSSM

**PhysSSM**: Physics-Anchored Stable State-Space Models for Long-Horizon Temperature Dynamics.

PhysSSM decomposes daily temperature into a stable insolation-forced thermal anchor
and a bounded S4D-learned residual, and forecasts in direct 90-day → 30-day blocks
before recursive block rollout.

See [`plans/prd.md`](plans/prd.md) for the full specification and [`plans/todo.md`](plans/todo.md)
for the task list. Legacy Stefan-Boltzmann / fast-slow / anomaly-recurrence code is frozen in `old/`.

## Layout

```
thermassm/            # Python package
  config.py           # anchor / model / data / train config + site registry
  data/               # insolation, ERA5 download, synthetic gen, dataset
  models/             # anchor, S4D, PhysSSM, baselines
  losses.py           # MSE-only core loss (no physics penalties)
  metrics.py          # RMSE, MAE, TAC, bias, drift, PSD, variance ratio, ...
  rollout.py          # 30-day block-recursive inference
  train.py            # training loop
  experiment.py       # orchestration helpers
  ablations.py        # A0-A5 ablation variants
  figures.py          # paper figures
scripts/              # entry-point CLI scripts
tests/                # unit tests (gitignored; kept locally)
plans/                # PRD + todo (gitignored; kept locally)
old/                  # frozen legacy code (gitignored; kept locally)
results/              # generated metrics, tables, figures (gitignored)
```

> `old/`, `tests/`, `plans/`, `results/`, `checkpoints/` and `data/` are gitignored and kept
> only as local working artifacts (see `.gitignore`).

## Install

```bash
pip install -r requirements.txt
```

Core requirements: `torch`, `numpy`, `pandas`, `scipy`. For real ERA5 data you additionally
need `xarray`, `netCDF4`, `zarr`, `gcsfs` (all in `requirements.txt`).

## Quick start (synthetic data, offline smoke test)

```bash
python3 scripts/train.py --model physssm --synthetic --device cpu --epochs 20
python3 scripts/evaluate.py --models physssm --synthetic --device cpu
python3 scripts/ablate.py --synthetic --device cpu --epochs 20
python3 scripts/figures.py --synthetic --device cpu
```

## Full benchmark (actual results → tables)

Run the stages (or `--stage all`), then build the tables:

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
- `benchmark_tables.md` — all tables (actual values)
- `table*_*.csv` — CSV copies

## Unit tests

```bash
python3 -m pytest tests/ -q
```

## Reproduce a single model

```bash
python3 scripts/train.py --model <name> [--synthetic] [--device cuda] [--epochs 50]
```

`<name>` ∈ `physssm`, `lstm`, `gru`, `rnn`, `pint-lstm`, `pint-gru`, `patchtst`, `vanilla_s4d`.

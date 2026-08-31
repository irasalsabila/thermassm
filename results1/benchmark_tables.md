# ThermaSSM Benchmark Tables (Actual Results)

_Generated from actual runs (data=era5, epochs=50, device=cuda)_

## Table 1: Main Multi-Year Forecasting Benchmark (WeatherBench 2m-Temperature)

| Model Category | Model Name | 365-Day RMSE (K) | 730-Day RMSE (K) | 1095-Day RMSE (K) | Drift βdrift 730d (K/yr) | Spectral Dist DPSD 730d |
|---|---|---|---|---|---|---|
| Pure Data-Driven | LSTM (Vanilla) | 13.65 | 13.92 | 13.92 | 7.73 | 90.38 |
| Pure Data-Driven | GRU (Vanilla) | 13.65 | 13.93 | 13.93 | 7.76 | 89.50 |
| Pure Data-Driven | RNN (Vanilla) | 13.61 | 13.86 | 13.85 | 7.75 | 91.91 |
| Physics-Informed | PINT-LSTM | 13.61 | 13.87 | 13.86 | 7.75 | 77.11 |
| Physics-Informed | PINT-GRU | 13.66 | 13.93 | 13.93 | 7.74 | 81.62 |
| Pure Data-Driven | PatchTST | 14.12 | 14.25 | 14.22 | 7.83 | 63.33 |
| Pure Data-Driven | Vanilla S4D | 15.86 | 16.38 | 16.44 | 7.47 | 7.85 |
| Physics-Informed | ClimODE 1D | 18.08 | 17.80 | 17.66 | 12.03 | 5.13 |
| Proposed | PhysSSM-EBM (Ours) | 6.08 | 6.02 | 5.68 | -0.95 | 7.84 |
| Baselines | Climatology (30-yr Mean) | 4.66 | 4.45 | 4.29 | -1.15 | 7.81 |
| Baselines | Persistence | 16.05 | 16.49 | 16.53 | 7.69 | 172.50 |

## Table 2: Multi-Climatic Zone Generalization

| Climate Zone | Coordinates | PINT-GRU RMSE (730d) | PatchTST RMSE (730d) | PhysSSM-EBM RMSE (730d) | Δ Improvement |
|---|---|---|---|---|---|
| Tropical / Equatorial | 0.0N, 100.0E (Padang) | 0.82 | 1.25 | 1.39 | -69.7% |
| Temperate Continental | 40.0N, 105.0W (Denver) | 7.00 | 6.89 | 3.28 | +53.2% |
| Subtropical Desert | 24.4N, 54.3E (Abu Dhabi) | 5.52 | 5.38 | 4.57 | +17.2% |
| Polar / Subarctic | 65.0N, 25.0E (Oulu) | 10.15 | 9.61 | 5.55 | +45.3% |

## Table 3: Comprehensive Ablation Study (730-Day Rollout)

| Configuration | Physics Formulation | Stability Constraint (A) | Output Head | 730d RMSE (K) | Drift (K/yr) | Extreme CSI₉₅ |
|---|---|---|---|---|---|---|
| (a) Full Model | Stefan-Boltzmann EBM | Re(A) <= -delta (Lyapunov) | Decoupled (mu_phys + R_theta) | 14.94 | 7.96 | 0.00 |
| (b) Toy Physics | Simple Harmonic (SHO) | Re(A) <= -delta (Lyapunov) | Decoupled | 13.90 | 7.63 | 0.00 |
| (c) No Physics | None | Re(A) <= -delta (Lyapunov) | Monolithic | 13.92 | 7.55 | 0.00 |
| (d) Unconstrained | Stefan-Boltzmann EBM | Unconstrained A | Decoupled | 14.93 | 7.95 | 0.00 |
| (e) Monolithic Head | Stefan-Boltzmann EBM | Re(A) <= -delta (Lyapunov) | Monolithic y = f(h) | 14.32 | 9.03 | 0.00 |

## Table 4: Computational Complexity & Resource Footprint

| Model | Parameters | Training Time (50 Epochs) | Peak VRAM | Inference Speed (Steps/sec) |
|---|---|---|---|---|
| LSTM (Vanilla) | 18 K | 1.00 min | 0.02 GB | 1,655 |
| GRU (Vanilla) | 14 K | 0.93 min | 0.02 GB | 1,838 |
| RNN (Vanilla) | 5 K | 0.88 min | 0.02 GB | 1,919 |
| PINT-LSTM | 18 K | 1.21 min | 0.02 GB | 1,629 |
| PINT-GRU | 14 K | 1.18 min | 0.02 GB | 1,725 |
| PatchTST | 107 K | 2.31 min | 0.02 GB | 933 |
| Vanilla S4D | 25 K | 1.52 min | 0.02 GB | 992 |
| ClimODE 1D | 9 K | 19.46 min | 0.02 GB | 69 |
| PhysSSM-EBM (Ours) | 34 K | 2.52 min | 0.02 GB | 818 |

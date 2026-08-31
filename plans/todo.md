# ThermaSSM — Todo

## Legend
- `[ ]` todo · `[~]` in progress · `[x]` done

## High Priority

- [x] Phase 1: Download ERA5 `t2m` data via xarray (WeatherBench 1, 5.625° NetCDF)
- [x] Phase 1: Compute solar insolation vectors from day-of-year and latitude
- [x] Phase 1: Implement PINT (RNN/LSTM/GRU) baselines
- [x] Phase 1: Implement PatchTST baseline
- [x] Phase 2: Implement Lyapunov-stable S4D layer (`A_k = -softplus(w_k) - δ + i·ν_k`, `Re(A) ≤ -δ < 0`)
- [x] Phase 2: Implement 1D Radiative-Convective EBM prior (Stefan-Boltzmann `C dT/dt = S(1-α) - εσT⁴`)
- [x] Phase 2: Implement temperature-dependent albedo feedback `α(T)`
- [x] Phase 2: Implement vector-accelerated EBM loss `L_EBM`
- [x] Phase 2: Implement composite loss `L_total = L_MSE + λ1·L_EBM + λ2·L_smooth`
- [x] Phase 2: Implement decoupled residual output head `y_pred(t) = μ_phys(t) + R_θ(h(t))`
- [x] Phase 3: Run 365/730/1095-day iterative rollouts across all model variants

## High Priority — Baseline Faithfulness Revision (current focus)

Revise baselines to match their papers exactly, in order: **S4D → PINT → PatchTST**.
(ClimODE is a global spatiotemporal model — deferred, see cross-cutting note.)

### S4D (first) — see plans/vanillas4d.md

- [x] S4D: HiPPO-based `A` init — S4D-Lin `A_n = -1/2 + iπn`, S4D-Inv, S4D-LegS
- [x] S4D: `B` init = ones; `C` init = complex standard normal `N(0,1) + iN(0,1)`
- [x] S4D: `Δ` sampled geometric-uniform in log-space `[log Δ_min, log Δ_max]`
- [~] S4D: `N/2` complex conjugate pairs with `2·Re(K)` real output (via `y.real`, equivalent)
- [x] S4D: GLU activation + residual/norm blocks in the baseline stack
- [x] S4D: "vanilla S4D" uses paper's `A = -exp(a_re)` (Hurwitz) parameterization with HiPPO init

### PINT (second) — see plans/pint.md

- [x] PINT: 2 hidden layers, 64 units, tanh, dropout 0.1
- [x] PINT: 30-day block prediction (not 1-day next-step), autoregressive 30-day blocks
- [x] PINT: standardize data (zero-mean, unit-variance, per-station train stats)
- [x] PINT: fixed `ω = 2π/365` (not learnable)
- [x] PINT: `λ_data = 1.0`, `λ_physics = 0.001`
- [~] PINT: Adam, lr `1e-3` (1000 epochs / full batch set via CLI, not default)
- [x] PINT: analytic sine/cosine linear-regression baseline `β₁cos(ωt) + β₂sin(ωt)`

### PatchTST (third) — see plans/patchtst.md

- [x] PatchTST: instance normalization + output denormalization
- [x] PatchTST: pad with repeated last value (not zeros)
- [x] PatchTST: learnable positional embedding `W_pos ∈ R^(D×N)`
- [x] PatchTST: 3 layers, 16 heads, D=128, FFN 256, GELU, BatchNorm, dropout 0.2
- [~] PatchTST: direct multi-step head (flatten → Linear → T) — block-direct `T=96` for long horizons
- [x] PatchTST: block-direct rollout protocol (predict `T=96`, slide, repeat)

### Cross-cutting

- [x] Benchmark protocol: aligned per baseline (PINT 30-day blocks, PatchTST block-direct, PhysSSM 1-day recurrent); rollout recomputes DOY/insolation correctly
- [x] ClimODE: removed from benchmark (was a toy Neural ODE, not the global ClimODE) — deferred for global/regional experiment

## Medium Priority

- [x] Phase 1: Implement Vanilla S4D and Mamba baselines
- [x] Phase 1: Implement Climatological Mean (30-yr DOY) baseline
- [x] Phase 1: Implement ClimODE 1D baseline (Neural ODE)
- [x] Phase 3: Implement ablation (a) Full Model, (b) Toy Physics (SHO), (c) No Physics
- [x] Phase 3: Implement ablation (d) Unconstrained A, (e) Monolithic Head
- [x] Phase 3: Generate PSD curves (FFT power spectra comparison)
- [x] Phase 3: Generate error drift graphs (`β_drift` vs. forecast horizon)
- [x] Phase 3: Generate metric tables (RMSE, MAE, drift, spectral distance, CSI95)
- [x] Phase 3: Multi-climatic zone generalization (Tropical, Temperate, Desert, Polar)

## Low Priority

- [x] Phase 4: Generate Figure 1 — End-to-End Architecture Pipeline
- [x] Phase 4: Generate Figure 2 — 730-Day Multi-Year Rollout Trajectories
- [x] Phase 4: Generate Figure 3 — Error Accumulation & Secular Drift Curves
- [x] Phase 4: Generate Figure 4 — Power Spectral Density (PSD) Analysis
- [x] Phase 4: Generate Figure 5 — Extreme Value Anomaly Tracking
- [x] Phase 4: Generate Figure 6 — Eigenvalue Complex Plane Distribution
- [x] Phase 4: Compile manuscript (ICLR/NeurIPS format)
- [x] Phase 4: Publish clean GitHub repository with reproducible scripts

## Future Work (post-benchmark)

- [ ] FW-2: Multivariate forecasting — add humidity, precipitation, wind speed as co-forecast variables
- [ ] FW-3: Adaptive physics loss — learnable/annealed `λ_ebm` (and `λ_smooth`) instead of fixed scalars
- [ ] FW-5: Uncertainty quantification — probabilistic residual head (mean + variance) or ensemble rollouts, calibrated exceedance probabilities

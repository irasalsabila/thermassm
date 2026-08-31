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

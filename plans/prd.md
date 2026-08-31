# ThermaSSM: Structurally Stable Energy-Balance State Space Models for Long-Horizon Temperature Modeling

**Target Venues:** NeurIPS / ICLR (main track, after revision) · AGU / Climate Informatics (fallback)

**Status:** Revised specification v2. This version incorporates a critical review of v1. Fabricated benchmark numbers have been removed; all result tables are illustrative layouts with values marked TBD until experiments are run.

---

## Research Question

> **Can architectural stability constraints and a physically grounded energy-balance decomposition prevent secular drift in neural long-horizon temperature models without sacrificing short-timescale variability?**

### Contribution

$$
\boxed{
\text{Energy-balance anchor}
+
\text{provably stable SSM}
+
\text{frequency-separated residual learning}
}
$$

The claim is **not** "we use an SSM for weather" (no longer novel). The claim is **physics-anchored structural stability**: stability is imposed by *architecture*, not encouraged through a physics-loss coefficient.

---

## 1. Problem Statement

Physics-informed time-series models for temperature — notably PINT (arXiv:2502.04018), which uses a simple-harmonic-oscillator (SHO) prior to forecast two years from a 90-day initial sequence — exhibit three weaknesses:

1. **Unrealistic physics prior.** The SHO prior (`u'' + ω²u = 0`) is linear and cannot represent non-linear blackbody cooling (`T⁴`), asymmetric seasons, or solar-insolation forcing.
2. **Unbounded long-horizon drift.** Physics is enforced via soft loss penalties (`L_data + λ·L_phys`) on recurrent architectures (RNN/LSTM), which gives gradient stiffness and multi-year trajectory drift. Stability is *encouraged*, not *guaranteed*.
3. **Spectral smoothing / misplaced sharpness.** A single network is asked to simultaneously learn annual periodicity, radiative equilibrium, synoptic variability, local peculiarities, and long-term stability — and losses that smooth the *full* predicted signal suppress the very extremes we want to preserve.

### Scope disclaimer (important)

**Exact daily weather is not deterministically predictable years in advance** (atmospheric predictability is O(2 weeks); beyond that, prediction is probabilistic and concerns aggregate statistics, slowly varying modes, or seasonal anomalies). We therefore do **not** claim operational daily weather forecasting at 365–1095 days. We model **long-horizon temperature trajectories** and their **climate statistics**. See §7.

---

## 2. Proposed Approach

The central decomposition:

$$
T(t) = \underbrace{\mu_{\mathrm{phys}}(t)}_{\text{slow, forced, seasonal}}
       + \underbrace{R_\theta(h_t)}_{\text{unresolved variability}}
$$

This separates slow physical evolution from learned residual variability instead of asking one recurrent network to learn everything at once.

Three pillars:

1. **Energy-balance anchor.** A 1D effective radiative energy-balance prior provides the deterministic macro drift `μ_phys(t)`.
2. **Provably stable SSM.** A diagonal state-space backbone (S4D) with Lyapunov-stable parameterization supplies the latent `h_t` in `O(L)` time.
3. **Frequency-separated residual learning.** A residual decoder `R_θ` captures high-frequency anomalies with explicit spectral/amplitude constraints.

---

## 3. Physics: Effective Radiative Energy-Balance Prior

### 3.1 Framing (revised)

The prior is an **inductive bias**, not a complete representation of local atmospheric thermodynamics. `T` is 2-m air temperature; `S(1 − α)` is a top-of-atmosphere shortwave budget; `εσT⁴` is effective radiative emission. A real local 2-m budget also involves greenhouse forcing, clouds, latent/sensible heat, advection, land/ocean storage, and soil moisture.

We therefore call this an **effective radiative energy-balance prior** and treat its parameters as *learnable effective coefficients*, following the spirit of statistical emulators such as FaIRGP (Bouabid et al., 2024).

$$
C \frac{dT}{dt} = S(t)\big(1 - \alpha(T)\big) - \varepsilon \sigma T^{4}
$$

- `C`: effective heat capacity (learnable).
- `ε`: effective emissivity / OLR coefficient (learnable).
- `α(T)`: temperature-dependent albedo feedback.
- `σ = 5.67 × 10⁻⁸ W/m²K⁴`.

Future strengthening: a **two-layer EBM** or a **learned OLR parameterization** `OLR_θ(T)` replacing `εσT⁴`.

### 3.2 Insolation (corrected)

The data are **daily aggregates**, so we use the analytical **daily-mean top-of-atmosphere insolation** with the sunset hour angle `h₀ = arccos(−tan φ tan δ)`:

$$
S(t) = \frac{S_0}{\pi}\Big[h_0 \sin\delta\sin\phi + \cos\delta\cos\phi\sin h_0\Big]
$$

with solar declination `δ = 23.44° · sin(2π(284 + n)/365)`. The instantaneous diurnal-cosine form is **not** used for daily data.

### 3.3 Albedo feedback

$$
\alpha(T) = \alpha_{\text{land}} + (\alpha_{\text{ice}} - \alpha_{\text{land}}) \cdot \sigma\!\left(\frac{T_{\text{freeze}} - T}{\Delta T}\right)
$$

where `σ(·)` is the logistic sigmoid.

---

## 4. Stability: Structural Stability of the SSM

### 4.1 Parameterization

Continuous latent dynamics:

$$
\frac{dh}{dt} = A h + B x, \qquad y_{\text{res}} = C h
$$

with diagonal `A` parameterized as:

$$
A_k = -\operatorname{softplus}(w_k) - \delta + i\,\nu_k
  \quad\Longrightarrow\quad \Re(\lambda_k) \le -\delta < 0
$$

### 4.2 Stability condition (corrected)

For **complex** `A`, the Lyapunov/Hermitian condition is on the Hermitian part:

$$
A + A^{*} \prec 0
$$

(where `A*` is the conjugate transpose), not `A + Aᵀ`. For the diagonal parameterization this reduces to `Re(A_k) ≤ −δ < 0`, so the autonomous system `ḣ = Ah` is asymptotically stable with equilibrium at `h = 0`.

### 4.3 The result we actually need to prove (revised)

A stable `A` guarantees stability of `ḣ = Ah` and, under bounded-input assumptions, of `ḣ = Ah + Bu`. It does **not** by itself guarantee boundedness of the closed loop:

$$
\hat T_t \to x_t \to h_{t+1} \to R_\theta(h) \to \hat T_{t+1}
$$

The nonlinear decoder and autoregressive feedback matter. The target theorem is therefore:

> **Theorem (closed-loop boundedness).** If (i) the SSM is asymptotically stable, (ii) the exogenous forcing `S(t)` and albedo map are bounded, and (iii) the residual decoder `R_θ` is bounded and Lipschitz, then the autoregressive rollout `{T̂_t}` is bounded for all horizons.

This turns the contribution from "physics-informed S4" into **structural stability imposed by architecture**.

---

## 5. Architecture

```
              Input: [T(t), S(t), DOY, Lat/Lon]
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌───────────────────┐              ┌─────────────────────┐
│ Energy-Balance    │              │ Stable SSM Backbone │
│ Prior (effective) │              │ Diagonal S4D/LRU    │
│ C dT/dt =         │              │ Re(A) ≤ −δ < 0      │
│  S(1−α) − εσT⁴    │              │ A + A* ≺ 0          │
└─────────┬─────────┘              └─────────┬───────────┘
          │ μ_phys(t)                        │ h(t)
          │                                  ▼
          │                         ┌─────────────────────┐
          │                         │ Residual Decoder    │
          │                         │ R_θ (Lipschitz)     │
          │                         └─────────┬───────────┘
          └──────────────┬───────────────────┘
                         ▼
              ŷ(t) = μ_phys(t) + R_θ(h(t))
```

---

## 6. Loss Function (revised)

### 6.1 Data term

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{H}\sum_{t=1}^{H} (T_t - \hat T_t)^2
$$

### 6.2 Physics term (escape hatch removed)

The v1 definition subtracted a flexible `R̂_t`, letting `R̂_t = C·Ṫ − EBM terms` make the constraint vacuous. The fix is to evaluate the EBM residual on the **full prediction but without any learned `R̂` term**:

$$
\mathcal{L}_{\text{EBM}} =
\frac{1}{H}\sum_{t=1}^{H}
\Big\|
C\frac{\hat T_{t+1} - \hat T_t}{\Delta t}
- \big(S_t(1-\alpha(\hat T_t)) - \varepsilon\sigma \hat T_t^{4}\big)
\Big\|^2
$$

Because `T̂ = μ + R_θ` and `μ` satisfies the EBM by construction, this reduces to a **penalty on residual amplitude** `‖(C/Δt)·R_θ‖²`. It keeps the full trajectory near EBM dynamics without allowing `R_θ` to absorb the physics (no escape hatch).

### 6.3 Residual constraints (replaces global smoothing)

The residual `R_θ(h_t)` is constrained to be spectrally separated from the macro branch:

- **Zero-mean** over a seasonal window: `⟨R_θ⟩ ≈ 0`.
- **Amplitude bound:** `‖R_θ‖ ≤ ρ` (enforced via a bounded activation).
- **High-frequency bias:** `‖lowpass(R_θ)‖²` penalized (pushes residual energy to high frequencies).

### 6.4 Smoothness applied to the macro branch only

Smoothing is applied to `μ_phys`, **not** the full `T̂` and **not** the residual (smoothing the residual would suppress the sharp extremes we want to keep):

$$
\mathcal{L}_{\text{smooth}} = \frac{1}{H}\sum_{t=1}^{H} (\Delta^{2}\mu_t)^2
$$

### 6.5 Total

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}
+ \lambda_1 \mathcal{L}_{\text{EBM}}
+ \lambda_2 \mathcal{L}_{\text{res}}
+ \lambda_3 \mathcal{L}_{\text{smooth}}
$$

---

## 7. Experimental Design (two regimes)

### Regime 1 — PINT-compatible benchmark

Replicate the exact 90-day → 730-day problem (and 365/1095 for completeness) for apples-to-apples evidence against PINT. Explicitly labeled **long-horizon temperature trajectory modeling**, not operational weather prediction.

### Regime 2 — Physically meaningful climate evaluation

Evaluate aggregate statistics and risk, not day-by-day accuracy:

- Seasonal means and anomalies
- Climatological phase (timing of annual cycle)
- Variance and spectral content (PSD / D_PSD)
- Exceedance frequency above thresholds
- Extreme spell duration
- Quantiles of the predictive distribution
- Probabilistic extreme risk (calibrated exceedance probabilities)

---

## 8. Data & Protocol

| Parameter | Specification |
|---|---|
| Source | ERA5 WeatherBench 2 (`2m_temperature`, `t2m`) |
| Spatial coverage | Grid points across Tropical, Temperate Continental, Subtropical Desert, Polar/Subpolar |
| Temporal resolution | Daily aggregates (1979–2022) |
| Split | Train 1979–2015 · Val 2016–2018 · Test 2019–2022 |
| Setup | Input window `L = 90` days; horizons `H ∈ {365, 730, 1095}` |

---

## 9. Baselines & Metrics

**Baselines:** Climatological mean (30-yr DOY), Persistence, Vanilla LSTM/GRU/RNN, PINT (SHO-regularized RNN/LSTM/GRU), PatchTST, Vanilla S4D (unconstrained), ClimODE 1D (Neural ODE).

**Metrics:**
- Accuracy: RMSE, MAE
- Climate stability: secular drift `β_drift` (target → 0), spectral distance `D_PSD`
- Extreme anomaly: `CSI₉₅` (Regime 1), plus exceedance frequency / spell duration / quantile calibration (Regime 2)

---

## 10. Ablations

1. **Physics formulation:** effective EBM prior vs. SHO prior vs. no physics.
2. **Stability constraint:** Lyapunov-constrained SSM (`Re(A) ≤ −δ`) vs. unconstrained SSM.
3. **Decomposition:** decoupled head (`μ_phys + R_θ`) vs. monolithic end-to-end prediction.
4. **Residual constraint:** with vs. without spectral separation / zero-mean / amplitude bound.
5. **Decoder regularity:** Lipschitz-bounded decoder vs. unconstrained decoder (tests the closed-loop theorem).

---

## 11. Roadmap

- **Phase 1 (Data & baselines):** ERA5 `t2m` extraction, daily-mean insolation, PINT + PatchTST baselines.
- **Phase 2 (Core model):** stable S4D layer, effective EBM prior, frequency-separated residual head, revised loss.
- **Phase 3 (Stability theory):** prove the closed-loop boundedness theorem (bounded forcing + stable SSM + Lipschitz decoder).
- **Phase 4 (Benchmarks):** Regime 1 + Regime 2 evaluations, ablations, PSD/drift/extreme-risk analysis.
- **Phase 5 (Writing & release):** manuscript, reproducible repository.

---

## 12. Illustrative Tables (values TBD)

> **Note:** All numerical values below are **placeholders for the table layout only**. They are **not** experimental results and must not be read as achieved outcomes. Replace with `TBD` / actual numbers after running the benchmark.

### Table 1 — Main multi-year benchmark (Regime 1)

| Category | Model | 365d RMSE (K) | 730d RMSE (K) | 1095d RMSE (K) | Drift (K/yr) | D_PSD |
|---|---|---|---|---|---|---|
| Baselines | Climatology | TBD | TBD | TBD | TBD | TBD |
| | Persistence | TBD | TBD | TBD | TBD | TBD |
| Pure data-driven | LSTM | TBD | TBD | TBD | TBD | TBD |
| | PatchTST | TBD | TBD | TBD | TBD | TBD |
| | Vanilla S4D | TBD | TBD | TBD | TBD | TBD |
| Physics-informed | PINT-LSTM | TBD | TBD | TBD | TBD | TBD |
| | PINT-GRU | TBD | TBD | TBD | TBD | TBD |
| | ClimODE 1D | TBD | TBD | TBD | TBD | TBD |
| Proposed | ThermaSSM (Ours) | TBD | TBD | TBD | TBD | TBD |

### Table 2 — Climate-zone generalization (Regime 1)

| Zone | Coordinates | PINT-GRU RMSE | PatchTST RMSE | ThermaSSM RMSE | Δ |
|---|---|---|---|---|---|
| Tropical | 0.0°N, 100.0°E | TBD | TBD | TBD | TBD |
| Temperate | 40.0°N, 105.0°W | TBD | TBD | TBD | TBD |
| Subtropical desert | 24.4°N, 54.3°E | TBD | TBD | TBD | TBD |
| Polar | 65.0°N, 25.0°E | TBD | TBD | TBD | TBD |

### Table 3 — Ablations (730-day)

| Config | Physics | Stability | Head | RMSE | Drift | CSI₉₅ |
|---|---|---|---|---|---|---|
| (a) Full | EBM | Re(A) ≤ −δ | Decoupled | TBD | TBD | TBD |
| (b) Toy physics | SHO | Re(A) ≤ −δ | Decoupled | TBD | TBD | TBD |
| (c) No physics | None | Re(A) ≤ −δ | Monolithic | TBD | TBD | TBD |
| (d) Unconstrained | EBM | A ∈ ℂ | Decoupled | TBD | TBD | TBD |
| (e) Monolithic head | EBM | Re(A) ≤ −δ | Monolithic | TBD | TBD | TBD |
| (f) No residual constraint | EBM | Re(A) ≤ −δ | Decoupled | TBD | TBD | TBD |
| (g) Unbounded decoder | EBM | Re(A) ≤ −δ | Decoupled | TBD | TBD | TBD |

### Table 4 — Compute footprint

| Model | Parameters | Train time (50 ep) | Peak VRAM | Inference (steps/s) |
|---|---|---|---|---|
| LSTM | TBD | TBD | TBD | TBD |
| PINT-LSTM | TBD | TBD | TBD | TBD |
| ClimODE | TBD | TBD | TBD | TBD |
| PatchTST | TBD | TBD | TBD | TBD |
| ThermaSSM | TBD | TBD | TBD | TBD |

---

## 13. References

- PINT: *Physics-Informed Neural Time Series Models…* arXiv:2502.04018
- S4D: *On the Parameterization and Initialization of Diagonal State Space Models* arXiv:2206.11893
- WSSM: *Geographic-enhanced hierarchical state-space model…* arXiv:2501.11238
- Robertson et al. (2020): *Subseasonal to Seasonal Prediction…* JGR Atmospheres (DOI 10.1029/2018JD029375)
- FaIRGP: Bouabid et al. (2024), *A Bayesian Energy Balance Model for Surface Temperatures Emulation*, JAMES (DOI 10.1029/2023MS003926)

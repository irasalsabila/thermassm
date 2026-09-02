"""Metrics for climate forecasting evaluation (short-range + long-horizon)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(err))


def secular_drift(y_true: np.ndarray, y_pred: np.ndarray, per_year: bool = True) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    t = np.arange(len(err))
    slope = stats.linregress(t, err).slope
    return slope * 365.0 if per_year else slope


def spectral_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return float("nan")

    def norm_psd(x):
        psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
        psd = psd + 1e-12
        return psd / psd.sum()

    p1 = norm_psd(y_true)
    p2 = norm_psd(y_pred)
    return float(stats.wasserstein_distance(np.arange(len(p1)), np.arange(len(p2)), p1, p2))


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def acc(y_true: np.ndarray, y_pred: np.ndarray, clim: np.ndarray) -> float:
    """Anomaly correlation: corr of (pred - clim) vs (true - clim)."""
    return corr(y_true - clim, y_pred - clim)


def tac(y_true: np.ndarray, y_pred: np.ndarray, clim: np.ndarray) -> float:
    """Single-site temporal anomaly correlation (same as `acc`)."""
    return acc(y_true, y_pred, clim)


def variance_ratio(y_true: np.ndarray, y_pred: np.ndarray, clim: np.ndarray) -> float:
    return float(np.std(y_pred - clim) / (np.std(y_true - clim) + 1e-8))


def _annual_harmonic(x: np.ndarray):
    t = np.arange(len(x))
    omega = 2 * np.pi / 365.0
    A = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(x, dtype=np.float64), rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    phase = float(np.arctan2(coef[1], coef[0]))
    return amp, phase


def seasonal_amplitude_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a_true, _ = _annual_harmonic(y_true)
    a_pred, _ = _annual_harmonic(y_pred)
    return a_pred - a_true


def seasonal_phase_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, p_true = _annual_harmonic(y_true)
    _, p_pred = _annual_harmonic(y_pred)
    diff = (p_pred - p_true) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


def acf_distance(y_true: np.ndarray, y_pred: np.ndarray, nlags: int = 30) -> float:
    def acf(x):
        x = np.asarray(x, dtype=np.float64) - x.mean()
        denom = float(np.dot(x, x)) + 1e-12
        return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(nlags)])

    return float(np.mean(np.abs(acf(y_true) - acf(y_pred))))


def quantile_error(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    return float(np.quantile(y_pred, q) - np.quantile(y_true, q))


def exceedance_frequency_error(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    thr = np.quantile(y_true, q)
    return float((y_pred > thr).mean() - (y_true > thr).mean())


def csi_threshold(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    thr = np.quantile(y_true, q)
    hits = np.sum((y_true > thr) & (y_pred > thr))
    misses = np.sum((y_true > thr) & (y_pred <= thr))
    false_alarms = np.sum((y_true <= thr) & (y_pred > thr))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 0.0


def _mean_spell_duration(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    d = np.diff(np.concatenate([[0], np.asarray(mask).astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return float(np.mean(ends - starts))


def hot_spell_duration_error(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.9) -> float:
    thr = np.quantile(y_true, q)
    return _mean_spell_duration(y_pred > thr) - _mean_spell_duration(y_true > thr)


def cold_spell_duration_error(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.1) -> float:
    thr = np.quantile(y_true, q)
    return _mean_spell_duration(y_pred < thr) - _mean_spell_duration(y_true < thr)


def per_lead_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RMSE per lead day across a (N, L) direct forecast block."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))


def summarize_seeds(seed_metrics: list) -> dict:
    """Aggregate a list of per-seed metric dicts into mean/std/n."""
    if not seed_metrics:
        return {}
    keys = seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in seed_metrics if k in m]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[k] = summarize_seeds(vals)
        else:
            arr = np.array([float(v) for v in vals])
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}
    return out


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, clim: np.ndarray | None = None) -> dict:
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": mean_bias(y_true, y_pred),
        "drift": secular_drift(y_true, y_pred),
        "psd": spectral_distance(y_true, y_pred),
        "corr": corr(y_true, y_pred),
    }
    if clim is not None:
        out["tac"] = tac(y_true, y_pred, clim)
        out["var_ratio"] = variance_ratio(y_true, y_pred, clim)
    out["seasonal_amp_err"] = seasonal_amplitude_error(y_true, y_pred)
    out["seasonal_phase_err"] = seasonal_phase_error(y_true, y_pred)
    out["acf_dist"] = acf_distance(y_true, y_pred)
    out["q95_err"] = quantile_error(y_true, y_pred, 0.95)
    out["q99_err"] = quantile_error(y_true, y_pred, 0.99)
    out["exceed_frac_err"] = exceedance_frequency_error(y_true, y_pred, 0.95)
    out["hot_spell_err"] = hot_spell_duration_error(y_true, y_pred, 0.9)
    out["cold_spell_err"] = cold_spell_duration_error(y_true, y_pred, 0.1)
    return out


def evaluate_direct(
    pred: np.ndarray,
    target: np.ndarray,
    clim: np.ndarray | None = None,
    leads: tuple = (1, 7, 14, 30),
) -> dict:
    """Evaluate a (N, 30) direct forecast block at requested lead days."""
    out = {}
    for k in leads:
        p = pred[:, k - 1]
        t = target[:, k - 1]
        row = {"rmse": rmse(t, p), "mae": mae(t, p), "bias": mean_bias(t, p)}
        if clim is not None and k != 1:
            # TAC is not interpreted at H=1 (single-day anomaly corr is degenerate).
            c = clim[:, k - 1] if np.asarray(clim).ndim == 2 else clim
            row["tac"] = tac(t, p, c)
        out[k] = row
    return out

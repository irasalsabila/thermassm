"""Metrics for climate forecasting evaluation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(err)))


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
    return float(
        stats.wasserstein_distance(
            np.arange(len(p1)), np.arange(len(p2)), p1, p2
        )
    )


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def acc(y_true: np.ndarray, y_pred: np.ndarray, clim: np.ndarray) -> float:
    """Anomaly correlation coefficient: corr of (pred - clim) vs (true - clim)."""
    return corr(y_true - clim, y_pred - clim)


def csi_threshold(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    thr = np.quantile(y_true, q)
    hits = np.sum((y_true > thr) & (y_pred > thr))
    misses = np.sum((y_true > thr) & (y_pred <= thr))
    false_alarms = np.sum((y_true <= thr) & (y_pred > thr))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 0.0


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "drift": secular_drift(y_true, y_pred),
        "psd": spectral_distance(y_true, y_pred),
        "csi95": csi_threshold(y_true, y_pred),
        "corr": corr(y_true, y_pred),
    }

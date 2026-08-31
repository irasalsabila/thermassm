"""Autoregressive rollout for long-horizon forecasting."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import torch

from .data.insolation import daily_insolation
from .models.physssm import PhysSSM

TEMP_MIN = 150.0
TEMP_MAX = 400.0


def _to_dates(dates: np.ndarray) -> list:
    return list(dates.astype("datetime64[D]").tolist())


def _doy(d) -> int:
    return d.timetuple().tm_yday


def build_feature_vector(
    t2m: float,
    ins: float,
    doy_sin: float,
    doy_cos: float,
    lat: float,
    lon: float,
) -> np.ndarray:
    return np.array(
        [t2m, ins, doy_sin, doy_cos, lat / 90.0, lon / 180.0], dtype=np.float32
    )


def rollout_physssm(
    model: PhysSSM,
    init_t2m: np.ndarray,
    init_dates: np.ndarray,
    horizon: int,
    lat: float,
    lon: float,
    device: torch.device,
) -> np.ndarray:
    """Roll out horizon days given an input window of init_t2m (len L)."""
    model.eval()
    dates = _to_dates(init_dates)
    state = model.initial_state(1, device)
    history = list(init_t2m)

    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            cur_t = history[-1]
            next_date = dates[-1] + timedelta(days=1)
            n_doy = _doy(next_date)
            n_sin = float(np.sin(2 * np.pi * n_doy / 365.0))
            n_cos = float(np.cos(2 * np.pi * n_doy / 365.0))
            n_ins = float(daily_insolation(lat, np.array([n_doy], dtype=float))[0])
            feat = build_feature_vector(cur_t, n_ins, n_sin, n_cos, lat, lon)
            x_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            y, state = model.step(x_t, state)
            pred = float(np.clip(y.cpu().numpy(), TEMP_MIN, TEMP_MAX))
            preds.append(pred)
            history.append(pred)
            dates.append(next_date)
    return np.array(preds, dtype=np.float32)


def rollout_autoregressive(
    model,
    init_x: np.ndarray,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    """Generic rollout for feed-forward baselines (RNN/PatchTST/S4D/ClimODE)."""
    model.eval()
    x = init_x.copy()
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            y = model(x_t)
            pred = float(np.clip(y[0, -1].cpu().numpy(), TEMP_MIN, TEMP_MAX))
            preds.append(pred)
            next_feat = x[-1].copy()
            next_feat[0] = pred
            x = np.concatenate([x, next_feat[None, :]], axis=0)[-len(x):]
    return np.array(preds, dtype=np.float32)


def rollout_sequence(
    model,
    dates: np.ndarray,
    t2m: np.ndarray,
    features: np.ndarray,
    input_len: int,
    horizon: int,
    lat: float,
    lon: float,
    device: torch.device,
    is_physssm: bool = False,
) -> np.ndarray:
    init_t2m = t2m[:input_len]
    init_dates = dates[:input_len]
    if is_physssm:
        return rollout_physssm(model, init_t2m, init_dates, horizon, lat, lon, device)
    return rollout_autoregressive(model, features[:input_len], horizon, device)

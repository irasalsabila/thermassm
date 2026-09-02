"""Rollout utilities: direct block inference for PhysSSM and baselines."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import torch

from .data.insolation import daily_insolation

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


def _next_feature(prev_feat: np.ndarray, prev_date, next_t: float, lat: float, lon: float):
    next_date = prev_date + timedelta(days=1)
    doy = _doy(next_date)
    ins = float(daily_insolation(lat, np.array([doy], dtype=float))[0])
    s = float(np.sin(2 * np.pi * doy / 365.0))
    c = float(np.cos(2 * np.pi * doy / 365.0))
    feat = prev_feat.copy()
    feat[0] = next_t
    feat[1] = ins
    feat[2] = s
    feat[3] = c
    feat[4] = lat / 90.0
    feat[5] = lon / 180.0
    return feat, next_date


def _block_forcing(last_date, block_len: int, lat: float) -> np.ndarray:
    """Deterministic future forcing [Q, sin(DOY), cos(DOY)] for a block."""
    Q, s, c = [], [], []
    d = last_date
    for _ in range(block_len):
        d = d + timedelta(days=1)
        doy = _doy(d)
        Q.append(float(daily_insolation(lat, np.array([doy], dtype=float))[0]))
        s.append(float(np.sin(2 * np.pi * doy / 365.0)))
        c.append(float(np.cos(2 * np.pi * doy / 365.0)))
    return np.stack([Q, s, c], axis=-1).astype(np.float32)


def rollout_block_physssm(model, init_x, init_dates, horizon, lat, lon, device):
    """PhysSSM: direct 30-day block, append, retain rolling 90-day context."""
    model.eval()
    block_len = getattr(model, "forecast_horizon", 30)
    input_len = init_x.shape[0]
    x = init_x.copy()
    dates = _to_dates(init_dates)
    preds = []
    with torch.no_grad():
        while len(preds) < horizon:
            forcing = _block_forcing(dates[-1], block_len, lat)
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            f_t = torch.tensor(forcing, dtype=torch.float32, device=device).unsqueeze(0)
            y = model(x_t, f_t)[0].cpu().numpy()
            y = np.clip(y, TEMP_MIN, TEMP_MAX)
            take = min(block_len, horizon - len(preds))
            for i in range(take):
                pred = float(y[i])
                preds.append(pred)
                next_date = dates[-1] + timedelta(days=1)
                feat = build_feature_vector(
                    pred, float(forcing[i, 0]), float(forcing[i, 1]), float(forcing[i, 2]), lat, lon
                )
                x = np.concatenate([x, feat[None, :]], axis=0)[-input_len:]
                dates.append(next_date)
    return np.array(preds, dtype=np.float32)


def rollout_next_step(model, init_x, init_dates, horizon, lat, lon, device):
    """Feed-forward baselines: 1-day next-step with sliding window."""
    model.eval()
    x = init_x.copy()
    dates = _to_dates(init_dates)
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            y = model(x_t)
            pred = float(np.clip(y[0, -1].item(), TEMP_MIN, TEMP_MAX))
            preds.append(pred)
            next_feat, next_date = _next_feature(x[-1], dates[-1], pred, lat, lon)
            x = np.concatenate([x, next_feat[None, :]], axis=0)[-len(x):]
            dates.append(next_date)
    return np.array(preds, dtype=np.float32)


def rollout_block(model, init_x, init_dates, horizon, lat, lon, device):
    """PINT/PatchTST-style: predict a fixed block, shift window by block, repeat."""
    model.eval()
    block_len = getattr(model, "block_len", 30)
    x = init_x.copy()
    dates = _to_dates(init_dates)
    preds = []
    with torch.no_grad():
        while len(preds) < horizon:
            x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            y = model(x_t)[0].cpu().numpy()
            if hasattr(model, "t_mean") and hasattr(model, "t_std"):
                y = y * float(model.t_std.cpu().numpy()) + float(model.t_mean.cpu().numpy())
            y = np.clip(y, TEMP_MIN, TEMP_MAX)
            take = min(block_len, horizon - len(preds))
            for v in y[:take]:
                preds.append(float(v))
                next_feat, next_date = _next_feature(x[-1], dates[-1], float(v), lat, lon)
                x = np.concatenate([x, next_feat[None, :]], axis=0)[-len(x):]
                dates.append(next_date)
    return np.array(preds, dtype=np.float32)


def rollout_direct(model, init_x, horizon, device):
    """PatchTST-style: predict the full horizon directly."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(init_x, dtype=torch.float32, device=device).unsqueeze(0)
        y = model(x_t)[0].cpu().numpy()[:horizon]
    return np.clip(y, TEMP_MIN, TEMP_MAX).astype(np.float32)


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
    mode: str = "next",
    is_physssm: bool = False,
) -> np.ndarray:
    init_t2m = t2m[:input_len]
    init_dates = dates[:input_len]
    init_x = features[:input_len]
    if is_physssm:
        return rollout_block_physssm(model, init_x, init_dates, horizon, lat, lon, device)
    if mode == "block":
        return rollout_block(model, init_x, init_dates, horizon, lat, lon, device)
    if mode == "direct":
        return rollout_direct(model, init_x, horizon, device)
    return rollout_next_step(model, init_x, init_dates, horizon, lat, lon, device)

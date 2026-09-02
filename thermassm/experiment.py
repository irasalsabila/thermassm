"""High-level experiment orchestration for PhysSSM."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import ClimateDataset, build_features, load_series
from .losses import baseline_loss, physssm_loss
from .metrics import evaluate_direct, evaluate_forecast
from .models import PhysSSM, build_baseline
from .train import train_model

LONG_HORIZONS = (365, 730)
DIRECT_LEADS = (1, 7, 14, 30)


def make_config(**overrides) -> Config:
    cfg = Config()
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            setattr(getattr(cfg, section), field, value)
        else:
            setattr(cfg, key, value)
    return cfg


def year_mask(dates: np.ndarray, start: int, end: int) -> np.ndarray:
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    return (years >= start) & (years <= end)


def doy_climatology(dates, t2m, train_mask):
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    train_doy = doy[train_mask]
    train_t2m = t2m[train_mask]
    return np.array([train_t2m[train_doy == d].mean() for d in range(1, 366)])


def get_t_stats(t2m: np.ndarray, mask: np.ndarray) -> tuple:
    return float(t2m[mask].mean()), float(t2m[mask].std())


def get_q_stats(features: np.ndarray, mask: np.ndarray) -> tuple:
    q = features[mask, 1]
    return float(q.mean()), float(q.std())


def res_amp_from_train(dates, t2m, train_mask, climo_365) -> float:
    """A_r = q99.9(|T - T_clim|) computed on training data only."""
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    clim = climo_365[np.clip(doy[train_mask], 1, 365) - 1]
    anom = t2m[train_mask] - clim
    return float(np.quantile(np.abs(anom), 0.999))


def load_and_split(cfg):
    dates, t2m = load_series(cfg)
    features = build_features(dates, t2m, cfg.data.lat, cfg.data.lon)
    tr = year_mask(dates, *cfg.data.train_years)
    va = year_mask(dates, *cfg.data.val_years)
    te = year_mask(dates, *cfg.data.test_years)
    cfg._climo_365 = doy_climatology(dates, t2m, tr)
    cfg._t_mean, cfg._t_std = get_t_stats(t2m, tr)
    cfg._q_mean, cfg._q_std = get_q_stats(features, tr)
    cfg._res_amp = res_amp_from_train(dates, t2m, tr, cfg._climo_365)
    return dates, t2m, features, tr, va, te


def model_spec(name: str, cfg):
    if name == "physssm":
        return "direct", cfg.model.forecast_horizon, True, cfg.data.input_len
    if name.startswith("pint"):
        return "block", cfg.model.pint_block, False, cfg.data.input_len
    if name == "patchtst":
        return "block", cfg.model.patch_horizon, False, cfg.data.input_len
    if name == "patchtst336":
        return "block", cfg.model.patch_horizon, False, cfg.model.patch_lookback
    return "next", 1, False, cfg.data.input_len


def make_loaders(cfg, dates, t2m, features, mask, shuffle, mode="next", predict_len=1, lookback=None):
    if lookback is None:
        lookback = cfg.data.input_len
    ds = ClimateDataset(
        dates[mask], t2m[mask], cfg.data.lat, cfg.data.lon, lookback,
        mode=mode, predict_len=predict_len,
    )
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)


def build_model(cfg, name: str, t_mean=0.0, t_std=1.0, q_mean=0.0, q_std=1.0, res_amp=None):
    if name == "physssm":
        return PhysSSM(cfg, res_amp=res_amp, t_mean=t_mean, t_std=t_std, q_mean=q_mean, q_std=q_std)
    _, _, _, lookback = model_spec(name, cfg)
    return build_baseline(
        name, cfg.model.input_dim, cfg.model.d_model,
        input_len=lookback,
        horizon=cfg.model.patch_horizon,
        t_mean=t_mean, t_std=t_std, cfg=cfg,
    )


def _loss_fn(name):
    return physssm_loss if name == "physssm" else baseline_loss


def _stats(cfg):
    return (
        getattr(cfg, "_t_mean", 0.0),
        getattr(cfg, "_t_std", 1.0),
        getattr(cfg, "_q_mean", 0.0),
        getattr(cfg, "_q_std", 1.0),
        getattr(cfg, "_res_amp", None),
    )


def run_training(cfg, name: str):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    mode, predict_len, _, lookback = model_spec(name, cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)
    model = build_model(cfg, name, t_mean, t_std, q_mean, q_std, res_amp)
    history, best_val = train_model(model, train_loader, val_loader, _loss_fn(name), cfg)
    torch.save(model.state_dict(), f"{cfg.train.checkpoint_dir}/{name}.pt")
    return model, history, best_val, (dates, t2m, features, tr, va, te)


def climatology_predict(dates, t2m, train_mask, start_date, horizon):
    climo = doy_climatology(dates, t2m, train_mask)
    start = np.datetime64(start_date).astype("datetime64[D]").tolist()
    preds = []
    for i in range(horizon):
        d = min((start + timedelta(days=i)).timetuple().tm_yday, 365)
        preds.append(climo[d - 1])
    return np.array(preds, dtype=np.float32)


def climatology_trend_predict(dates, t2m, train_mask, start_date, horizon):
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    climo = doy_climatology(dates, t2m, train_mask)
    train_doy = np.clip(doy[train_mask], 1, 365)
    train_years = years[train_mask]
    anomalies = t2m[train_mask] - climo[train_doy - 1]
    slope = float(np.polyfit(train_years, anomalies, 1)[0])
    start = np.datetime64(start_date).astype("datetime64[D]").tolist()
    preds = []
    for i in range(horizon):
        d = start + timedelta(days=i)
        doy_i = min(d.timetuple().tm_yday, 365)
        preds.append(climo[doy_i - 1] + slope * (d.year - start.year))
    return np.array(preds, dtype=np.float32)


def persistence_predict(last_temp, horizon):
    return np.full(horizon, last_temp, dtype=np.float32)


def ar1_predict(dates, t2m, train_mask, climo, start_date, horizon, last_anom):
    """AR(1) on climatological anomalies, recursed forward."""
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    train_doy = np.clip(doy[train_mask], 1, 365)
    train_clim = climo[train_doy - 1]
    z = t2m[train_mask] - train_clim
    X = np.column_stack([z[:-1], np.ones(len(z) - 1)])
    a, b = np.linalg.lstsq(X, z[1:], rcond=None)[0]
    start = np.datetime64(start_date).astype("datetime64[D]").tolist()
    preds = []
    zt = float(last_anom)
    for i in range(horizon):
        d = start + timedelta(days=i)
        doy_i = min(d.timetuple().tm_yday, 365)
        zt = a * zt + b
        preds.append(climo[doy_i - 1] + zt)
    return np.array(preds, dtype=np.float32)


def harmonic_predict(dates, t2m, train_mask, start_date, horizon):
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    train_doy = doy[train_mask]
    train_t2m = t2m[train_mask]
    mean = float(train_t2m.mean())
    std = float(train_t2m.std()) + 1e-8
    z = (train_t2m - mean) / std
    omega = 2 * np.pi / 365.0
    A = np.column_stack([np.cos(omega * train_doy), np.sin(omega * train_doy)])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    start = np.datetime64(start_date).astype("datetime64[D]").tolist()
    preds = []
    for i in range(horizon):
        d = (start + timedelta(days=i)).timetuple().tm_yday
        z_pred = beta[0] * np.cos(omega * d) + beta[1] * np.sin(omega * d)
        preds.append(z_pred * std + mean)
    return np.array(preds, dtype=np.float32)


def _doy_array(dates):
    return np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])


def _clim_window(climo_365, dates, start_idx, horizon):
    doy = _doy_array(dates[start_idx : start_idx + horizon])
    return climo_365[np.clip(doy, 1, 365) - 1]


def run_rollouts(cfg, model, name, data):
    from .rollout import rollout_sequence

    dates, t2m, features, tr, va, te = data
    device = torch.device(cfg.train.device)
    test_start = int(np.argmax(te))
    mode, _, is_physssm, lookback = model_spec(name, cfg)
    if hasattr(model, "forecast_horizon"):
        is_physssm = True
    climo_365 = getattr(cfg, "_climo_365", None)
    results = {}
    for horizon in LONG_HORIZONS:
        preds = rollout_sequence(
            model, dates[test_start:], t2m[test_start:], features[test_start:],
            lookback, horizon, cfg.data.lat, cfg.data.lon, device,
            mode=mode, is_physssm=is_physssm,
        )
        true = t2m[test_start + lookback : test_start + lookback + horizon]
        r = {"pred": preds, "true": true}
        if climo_365 is not None:
            r["clim"] = _clim_window(climo_365, dates, test_start + lookback, horizon)
        results[horizon] = r
    return results


def run_direct_block(cfg, model, data, stride=None):
    """Direct 30-day forecast over test origins; returns (pred, target, clim)."""
    dates, t2m, features, tr, va, te = data
    device = torch.device(cfg.train.device)
    lookback = cfg.data.input_len
    flen = cfg.model.forecast_horizon
    stride = stride or flen
    test_start = int(np.argmax(te))
    origins = list(range(test_start, len(t2m) - lookback - flen, stride))
    if not origins:
        origins = [test_start]

    X = np.stack([features[o : o + lookback] for o in origins])
    F = np.stack([features[o + lookback : o + lookback + flen, 1:4] for o in origins])
    Y = np.stack([t2m[o + lookback : o + lookback + flen] for o in origins])

    model.eval()
    preds = []
    bs = cfg.train.batch_size
    with torch.no_grad():
        for b in range(0, len(X), bs):
            xb = torch.tensor(X[b : b + bs], dtype=torch.float32, device=device)
            fb = torch.tensor(F[b : b + bs], dtype=torch.float32, device=device)
            preds.append(model(xb, fb).cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    clim = None
    if getattr(cfg, "_climo_365", None) is not None:
        doy = _doy_array(dates)
        C = np.stack([cfg._climo_365[np.clip(doy[o + lookback : o + lookback + flen], 1, 365) - 1] for o in origins])
        clim = C
    return pred, Y, clim


def evaluate_direct_block(pred, target, clim=None):
    return evaluate_direct(pred, target, clim, leads=DIRECT_LEADS)


def evaluate_results(results: dict) -> dict:
    out = {}
    for horizon, r in results.items():
        out[horizon] = evaluate_forecast(r["true"], r["pred"], r.get("clim"))
    return out


def save_json(obj, path: str):
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, default=conv, indent=2)

"""High-level experiment orchestration."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import ClimateDataset, build_features, load_series
from .losses import baseline_loss, composite_loss
from .metrics import evaluate_forecast
from .models import PhysSSM, build_baseline
from .train import train_model


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


def load_and_split(cfg):
    dates, t2m = load_series(cfg)
    features = build_features(dates, t2m, cfg.data.lat, cfg.data.lon)
    tr = year_mask(dates, *cfg.data.train_years)
    va = year_mask(dates, *cfg.data.val_years)
    te = year_mask(dates, *cfg.data.test_years)
    return dates, t2m, features, tr, va, te


def model_spec(name: str, cfg):
    if name == "physssm":
        return "next", 1, True
    if name.startswith("pint"):
        return "block", cfg.model.pint_block, False
    if name == "patchtst":
        return "block", cfg.model.patch_horizon, False
    return "next", 1, False


def get_t_stats(t2m: np.ndarray, mask: np.ndarray) -> tuple:
    mean = float(t2m[mask].mean())
    std = float(t2m[mask].std())
    return mean, std


def make_loaders(cfg, dates, t2m, features, mask, shuffle, mode="next", predict_len=1):
    sub_dates = dates[mask]
    sub_t2m = t2m[mask]
    ds = ClimateDataset(
        sub_dates, sub_t2m, cfg.data.lat, cfg.data.lon, cfg.data.input_len,
        mode=mode, predict_len=predict_len,
    )
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)


def build_model(cfg, name: str, t_mean: float = 0.0, t_std: float = 1.0):
    if name == "physssm":
        return PhysSSM(cfg)
    return build_baseline(
        name, cfg.model.input_dim, cfg.model.d_model,
        input_len=cfg.data.input_len,
        horizon=cfg.data.horizons[-1],
        t_mean=t_mean, t_std=t_std, cfg=cfg,
    )


def _loss_fn(name):
    return composite_loss if name == "physssm" else baseline_loss


def run_training(cfg, name: str):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std = get_t_stats(t2m, tr)
    mode, predict_len, _ = model_spec(name, cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len)
    model = build_model(cfg, name, t_mean, t_std)
    history, best_val = train_model(
        model, train_loader, val_loader, _loss_fn(name), cfg
    )
    ckpt = f"{cfg.train.checkpoint_dir}/{name}.pt"
    torch.save(model.state_dict(), ckpt)
    return model, history, best_val, (dates, t2m, features, tr, va, te)


def climatology_predict(dates, t2m, train_mask, start_date, horizon):
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    train_doy = doy[train_mask]
    train_t2m = t2m[train_mask]
    climo = np.array([train_t2m[train_doy == d].mean() for d in range(1, 367)])
    start = np.datetime64(start_date).astype("datetime64[D]").tolist()
    preds = []
    for i in range(horizon):
        d = (start + timedelta(days=i)).timetuple().tm_yday
        preds.append(climo[d - 1])
    return np.array(preds, dtype=np.float32)


def persistence_predict(last_temp, horizon):
    return np.full(horizon, last_temp, dtype=np.float32)


def harmonic_predict(dates, t2m, train_mask, start_date, horizon):
    """Analytic sine/cosine regression baseline (PINT's exact-solution baseline)."""
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


def run_rollouts(cfg, model, name, data):
    from .rollout import rollout_sequence

    dates, t2m, features, tr, va, te = data
    device = torch.device(cfg.train.device)
    test_start_idx = np.argmax(te)
    input_len = cfg.data.input_len
    mode, _, is_physssm = model_spec(name, cfg)
    results = {}
    for horizon in cfg.data.horizons:
        preds = rollout_sequence(
            model, dates[test_start_idx:], t2m[test_start_idx:],
            features[test_start_idx:], input_len, horizon,
            cfg.data.lat, cfg.data.lon, device,
            mode=mode, is_physssm=is_physssm,
        )
        true = t2m[test_start_idx + input_len : test_start_idx + input_len + horizon]
        results[horizon] = {"pred": preds, "true": true}
    return results


def evaluate_results(results: dict) -> dict:
    out = {}
    for horizon, r in results.items():
        out[horizon] = evaluate_forecast(r["true"], r["pred"])
    return out


def save_json(obj, path: str):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, default=conv, indent=2)

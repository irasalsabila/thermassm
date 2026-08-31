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
from .data.insolation import daily_insolation
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


def make_loaders(cfg, dates, t2m, features, mask, shuffle):
    sub_dates = dates[mask]
    sub_t2m = t2m[mask]
    ds = ClimateDataset(sub_dates, sub_t2m, cfg.data.lat, cfg.data.lon, cfg.data.input_len)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)


def build_model(cfg, name: str):
    if name == "physssm":
        return PhysSSM(cfg)
    return build_baseline(name, cfg.model.input_dim, cfg.model.d_model)


def _loss_fn(name):
    return composite_loss if name == "physssm" else baseline_loss


def run_training(cfg, name: str):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False)
    model = build_model(cfg, name)
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


def run_rollouts(cfg, model, name, data):
    from .rollout import rollout_sequence

    dates, t2m, features, tr, va, te = data
    device = torch.device(cfg.train.device)
    test_start_idx = np.argmax(te)
    input_len = cfg.data.input_len
    results = {}
    for horizon in cfg.data.horizons:
        preds = rollout_sequence(
            model, dates[test_start_idx:], t2m[test_start_idx:],
            features[test_start_idx:], input_len, horizon,
            cfg.data.lat, cfg.data.lon, device,
            is_physssm=(name == "physssm"),
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

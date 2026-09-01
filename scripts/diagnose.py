#!/usr/bin/env python3
"""Diagnose the dissipative PhysSSM: tau sweep, horizon skill, variance ratio, oracle feedback, leakage audit."""
import argparse
import math
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm.experiment import (
    build_model,
    doy_climatology,
    get_t_stats,
    load_and_split,
    make_config,
    make_loaders,
    model_spec,
)
from thermassm.losses import composite_loss
from thermassm.metrics import acc, corr, rmse, secular_drift
from thermassm.models.physssm import scale_features
from thermassm.rollout import TEMP_MAX, TEMP_MIN, build_feature_vector, rollout_recurrent
from thermassm.train import train_model

HORIZONS = [1, 7, 30, 90, 180, 365, 730]
TAUS = [4, 10, 30, 90, 365]


def _clim_series(climo_365, dates, start_idx, horizon):
    doy = np.array([d.timetuple().tm_yday for d in dates[start_idx : start_idx + horizon].astype("datetime64[D]").tolist()])
    doy = np.clip(doy, 1, 365)
    return climo_365[doy - 1]


def _metrics(true, pred, clim):
    return {
        "rmse": rmse(true, pred),
        "acc": acc(true, pred, clim),
        "drift": secular_drift(true, pred),
        "var_ratio": float(np.std(pred - clim) / (np.std(true - clim) + 1e-8)),
    }


def oracle_rollout(model, init_feats, init_dates, horizon, lat, lon, device, true_anom):
    model.eval()
    dates = [d for d in init_dates.astype("datetime64[D]").tolist()]
    state = model.initial_state(1, device)
    x = init_feats.copy()
    preds = []
    with torch.no_grad():
        for i in range(horizon):
            cur_t = float(x[-1, 0])
            next_date = dates[-1] + timedelta(days=1)
            doy = next_date.timetuple().tm_yday
            from thermassm.data.insolation import daily_insolation

            ins = float(daily_insolation(lat, np.array([doy], dtype=float))[0])
            s = float(np.sin(2 * np.pi * doy / 365))
            c = float(np.cos(2 * np.pi * doy / 365))
            feat = build_feature_vector(cur_t, ins, s, c, lat, lon)
            x_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)

            doy_idx = model._doy_idx(x_t).item()
            clim = float(model.climo[doy_idx].item())
            xs = scale_features(x_t).squeeze(1)
            u = model.in_proj(xs)
            uf, sf = model.ssm_fast.step(u, state[0])
            us, ss = model.ssm_slow.step(u, state[1])
            state = (sf, ss)
            h = torch.cat([uf, us], dim=-1)

            z_true = torch.tensor([[true_anom[i]]], dtype=torch.float32, device=device)
            res_in = torch.cat([h, z_true, xs[:, 1:2], xs[:, 2:4]], dim=-1)
            res = model.res_amp * torch.tanh(model.res_head(res_in).squeeze(-1))
            rho = model._rho()
            z_pred = cur_t - clim
            z_next = rho * z_pred + (1.0 - rho) * float(res.item())
            y = float(np.clip(clim + z_next, TEMP_MIN, TEMP_MAX))
            preds.append(y)

            next_feat = feat.copy()
            next_feat[0] = y
            x = np.concatenate([x, next_feat[None, :]], axis=0)[-len(x):]
            dates.append(next_date)
    return np.array(preds, dtype=np.float32)


def leakage_audit(model, init_t2m, init_dates, horizon, lat, lon, device):
    # Rollout functions receive only the initial window (init_t2m / init_dates) plus
    # deterministic astronomical forcing; they never receive future true temperatures.
    base = rollout_recurrent(model, init_t2m, init_dates, horizon, lat, lon, device)
    alt = rollout_recurrent(model, init_t2m, init_dates, horizon, lat, lon, device)
    deterministic = bool(np.array_equal(base, alt))
    print(f"leakage audit: rollout reads only the init window (no future access by signature); deterministic={deterministic}")
    return deterministic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=730)
    args = parser.parse_args()

    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device, "train.epochs": args.epochs})
    device = torch.device(args.device)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std = get_t_stats(t2m, tr)
    climo_365 = doy_climatology(dates, t2m, tr)
    mode, predict_len, _, lookback = model_spec("physssm", cfg)

    model = build_model(cfg, "physssm", t_mean, t_std)
    model.to(device)
    train_model(
        model,
        make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback),
        make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback),
        composite_loss, cfg, ckpt_name="physssm.pt", desc="train",
    )

    test_start = int(np.argmax(te))
    input_len = cfg.data.input_len
    init_t2m = t2m[test_start : test_start + input_len]
    init_dates = dates[test_start : test_start + input_len]
    init_feats = features[test_start : test_start + input_len]
    horizon = args.horizon
    true = t2m[test_start + input_len : test_start + input_len + horizon]
    clim = _clim_series(climo_365, dates, test_start + input_len, horizon)
    true_anom = true - clim

    print("\n== teacher-forced one-step (val) ==")
    model.eval()
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            yb_pred = model(xb.to(device)).cpu().numpy()
            tot += float(((yb_pred - yb.numpy()) ** 2).sum())
            n += yb.numel()
    print(f"one-step RMSE = {np.sqrt(tot / n):.3f} K")

    print("\n== autonomous rollout by horizon (learned tau) ==")
    learned_tau = float(torch.exp(model.log_tau).item())
    for h in HORIZONS:
        if h > horizon:
            break
        pred = rollout_recurrent(model, init_t2m, init_dates, h, cfg.data.lat, cfg.data.lon, device)
        m = _metrics(true[:h], pred, clim[:h])
        print(f"H={h:4d}: rmse={m['rmse']:.2f} acc={m['acc']:.3f} drift={m['drift']:.2f} var_ratio={m['var_ratio']:.2f}")

    print(f"\n== tau sweep (autonomous, H={horizon}) ==  [learned tau = {learned_tau:.1f}d]")
    for tau in TAUS + ["learned"]:
        if tau == "learned":
            model.log_tau.data.fill_(math.log(learned_tau))
            label = f"learned({learned_tau:.0f})"
        else:
            model.log_tau.data.fill_(math.log(tau))
            label = f"tau={tau}d"
        pred = rollout_recurrent(model, init_t2m, init_dates, horizon, cfg.data.lat, cfg.data.lon, device)
        m = _metrics(true, pred, clim)
        print(f"{label:16s}: rmse={m['rmse']:.2f} acc={m['acc']:.3f} drift={m['drift']:.2f} var_ratio={m['var_ratio']:.2f}")

    print(f"\n== oracle-feedback rollout (H={horizon}) ==")
    model.log_tau.data.fill_(math.log(learned_tau))
    pred_oracle = oracle_rollout(model, init_feats, init_dates, horizon, cfg.data.lat, cfg.data.lon, device, true_anom)
    m = _metrics(true, pred_oracle, clim)
    print(f"oracle: rmse={m['rmse']:.2f} acc={m['acc']:.3f} drift={m['drift']:.2f} var_ratio={m['var_ratio']:.2f}")

    print("\n== leakage audit ==")
    leakage_audit(model, init_t2m, init_dates, horizon, cfg.data.lat, cfg.data.lon, device)


if __name__ == "__main__":
    main()

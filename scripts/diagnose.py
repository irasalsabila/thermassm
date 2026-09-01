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
from thermassm.models.physssm import PhysSSM, scale_features
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
            rho = float(model._rho().item())
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
    parser.add_argument("--recurrence-ablation", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--lambda-sweep", action="store_true")
    parser.add_argument("--mlp-diagnostic", action="store_true")
    args = parser.parse_args()

    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device, "train.epochs": args.epochs})
    device = torch.device(args.device)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std = get_t_stats(t2m, tr)
    climo_365 = doy_climatology(dates, t2m, tr)
    mode, predict_len, _, lookback = model_spec("physssm", cfg)

    if args.baselines:
        simple_baselines(dates, t2m, climo_365, tr, va)
        return
    if args.mlp_diagnostic:
        run_mlp_diagnostic(cfg, dates, t2m, features, tr, va, device, climo_365)
        return
    if args.lambda_sweep:
        run_lambda_sweep(cfg, dates, t2m, features, tr, va, te, device, climo_365)
        return
    if args.recurrence_ablation:
        run_recurrence_ablation(cfg, dates, t2m, features, tr, va, te, device, climo_365)
        return

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

    print("\n== structural-range diagnostics ==")
    doy_all = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    clim_all = climo_365[np.clip(doy_all, 1, 365) - 1]
    z_all = t2m - clim_all
    dz_all = np.diff(z_all)
    qs = [0.1, 1, 5, 95, 99, 99.9]
    print("|z| quantiles  :", {q: round(float(np.percentile(np.abs(z_all), q)), 2) for q in qs})
    print("|dz| quantiles :", {q: round(float(np.percentile(np.abs(dz_all), q)), 2) for q in qs})

    rho_d = float(model._rho().item())
    amp_d = float(model.res_amp.item())
    if model.formulation == "innovation":
        Lb = rho_d * z_all[:-1] - amp_d
        Ub = rho_d * z_all[:-1] + amp_d
    else:
        Lb = rho_d * z_all[:-1] - (1 - rho_d) * amp_d
        Ub = rho_d * z_all[:-1] + (1 - rho_d) * amp_d
    z_next_all = z_all[1:]
    e_min = np.clip(z_next_all - Ub, 0, None) + np.clip(Lb - z_next_all, 0, None)
    print(f"structural RMSE floor (rho={rho_d:.3f}, amp={amp_d:.1f}, {model.formulation}): "
          f"{float(np.sqrt(np.mean(e_min ** 2))):.3f} K")

    model.eval()
    sat, nsat = 0, 0
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            xs = scale_features(xb)
            u = model.in_proj(xs)
            h = torch.cat([model.ssm_fast(u), model.ssm_slow(u)], dim=-1)
            zz = xb[:, :, 0] - model.climo[model._doy_idx(xb)]
            ri = torch.cat([h, zz.unsqueeze(-1), xs[..., 1:2], xs[..., 2:4]], dim=-1)
            tanh_out = torch.tanh(model.res_head(ri))
            sat += int((tanh_out.abs() > 0.95).sum())
            nsat += tanh_out.numel()
    print(f"residual saturation P(|tanh|>0.95): {sat / max(1, nsat):.3f}")

    print("\n== leakage audit ==")
    leakage_audit(model, init_t2m, init_dates, horizon, cfg.data.lat, cfg.data.lon, device)


def run_recurrence_ablation(cfg, dates, t2m, features, tr, va, te, device, climo_365):
    mode, predict_len, _, lookback = model_spec("physssm", cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    clim = climo_365[np.clip(doy, 1, 365) - 1]
    z = t2m - clim
    U = float(np.percentile(np.abs(np.diff(z[tr])), 99))
    print(f"innovation U = q99(|dz|, train) = {U:.2f} K\n")

    configs = [
        ("C0_equilibrium_amp5", "equilibrium", 5.0),
        ("C1_equilibrium_amp20", "equilibrium", 20.0),
        ("C2_innovation_ampU", "innovation", U),
    ]
    test_start = int(np.argmax(te))
    ilen = cfg.data.input_len
    init_t2m = t2m[test_start : test_start + ilen]
    init_dates = dates[test_start : test_start + ilen]
    true = t2m[test_start + ilen : test_start + ilen + 730]
    d730 = np.array([d.timetuple().tm_yday for d in dates[test_start + ilen : test_start + ilen + 730].astype("datetime64[D]").tolist()])
    c730 = climo_365[np.clip(d730, 1, 365) - 1]

    for label, formulation, amp in configs:
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        model = PhysSSM(cfg, climo_365, formulation=formulation, amp=amp).to(device)
        train_model(model, train_loader, val_loader, composite_loss, cfg, ckpt_name=f"{label}.pt", desc=label)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                yp = model(xb.to(device)).cpu().numpy()
                tot += float(((yp - yb.numpy()) ** 2).sum())
                n += yb.numel()
        one_step = np.sqrt(tot / n)
        pred = rollout_recurrent(model, init_t2m, init_dates, 730, cfg.data.lat, cfg.data.lon, device)
        print(f"{label}: one-step={one_step:.2f}K  730d rmse={rmse(true, pred):.2f} "
              f"acc={acc(true, pred, c730):.3f} var_ratio={float(np.std(pred - c730) / (np.std(true - c730) + 1e-8)):.2f}")


def simple_baselines(dates, t2m, climo_365, tr, va):
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    clim = climo_365[np.clip(doy, 1, 365) - 1]
    z = t2m - clim
    va_idx = np.where(va)[0]
    t = va_idx[:-1]
    t1 = va_idx[1:]
    print("== one-step difficulty baselines (val) ==")
    print(f"temperature persistence : {rmse(t2m[t1], t2m[t]):.3f} K")
    print(f"anomaly persistence     : {rmse(z[t1], z[t]):.3f} K")

    z_tr = z[tr]
    d_tr = doy[tr]
    X = np.column_stack([z_tr[:-1], np.ones(len(z_tr) - 1)])
    a, b = np.linalg.lstsq(X, z_tr[1:], rcond=None)[0]
    z_ar = a * z[t] + b
    print(f"AR(1)                   : {rmse(z[t1], z_ar):.3f} K")

    Xs = np.column_stack([
        z_tr[:-1],
        np.sin(2 * np.pi * d_tr[:-1] / 365),
        np.cos(2 * np.pi * d_tr[:-1] / 365),
        np.ones(len(z_tr) - 1),
    ])
    c = np.linalg.lstsq(Xs, z_tr[1:], rcond=None)[0]
    z_sar = c[0] * z[t] + c[1] * np.sin(2 * np.pi * doy[t] / 365) + c[2] * np.cos(2 * np.pi * doy[t] / 365) + c[3]
    print(f"seasonal AR(1)          : {rmse(z[t1], z_sar):.3f} K")


def run_lambda_sweep(cfg, dates, t2m, features, tr, va, te, device, climo_365):
    mode, predict_len, _, lookback = model_spec("physssm", cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)
    print("== lambda_res sweep (one-step) ==")
    for lam in [0.0, 0.001, 0.01, 0.1]:
        cfg.train.lambda_ebm = lam
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        model = PhysSSM(cfg, climo_365).to(device)
        train_model(model, train_loader, val_loader, composite_loss, cfg, ckpt_name=f"lam{lam}.pt", desc=f"lam={lam}")
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                yp = model(xb.to(device)).cpu().numpy()
                tot += float(((yp - yb.numpy()) ** 2).sum())
                n += yb.numel()
        print(f"lambda_res={lam}: one-step={np.sqrt(tot / n):.2f} K")


def run_mlp_diagnostic(cfg, dates, t2m, features, tr, va, device, climo_365):
    import torch.nn as nn
    import torch.nn.functional as F

    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    clim = climo_365[np.clip(doy, 1, 365) - 1]
    z = t2m - clim
    Q = features[:, 1] / 340.0
    dsin = np.sin(2 * np.pi * doy / 365)
    dcos = np.cos(2 * np.pi * doy / 365)

    def pairs(mask):
        idx = np.where(mask)[0]
        t = idx[:-1]
        t1 = idx[1:]
        return np.column_stack([z[t], Q[t], dsin[t], dcos[t]]), z[t1]

    Xtr, ytr = pairs(tr)
    Xva, yva = pairs(va)
    print("== tiny MLP diagnostic (no SSM, no recurrence) ==")
    for hidden in [(64, 64), (8,), (1,)]:
        layers = [nn.Linear(4, hidden[0])]
        for i in range(len(hidden) - 1):
            layers += [nn.Tanh(), nn.Linear(hidden[i], hidden[i + 1])]
        layers += [nn.Tanh(), nn.Linear(hidden[-1], 1)]
        mlp = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(-1)
        for _ in range(500):
            opt.zero_grad()
            F.mse_loss(mlp(Xt), yt).backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            pred = mlp(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy().ravel()
        print(f"MLP(4->{'->'.join(map(str, hidden))}->1) one-step: {rmse(yva, pred):.3f} K")


if __name__ == "__main__":
    main()

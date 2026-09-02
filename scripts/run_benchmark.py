#!/usr/bin/env python3
"""Run the PhysSSM benchmark and write per-stage JSON results.

Supports multi-seed aggregation via `--seeds` (mean/std/n per metric).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from thermassm.ablations import AblationPhysSSM, ABLATION_CONFIGS
from thermassm.config import SITES
from thermassm.experiment import (
    _stats,
    ar1_predict,
    build_model,
    evaluate_direct_block,
    evaluate_results,
    harmonic_predict,
    load_and_split,
    make_config,
    make_loaders,
    model_spec,
    run_direct_block,
    run_rollouts,
)
from thermassm.losses import ablation_loss, baseline_loss, physssm_loss
from thermassm.metrics import summarize_seeds
from thermassm.train import train_model

OUT = Path("results")

NEURAL_MODELS = ["pint-gru", "vanilla_s4d", "physssm"]

BASELINES = [
    ("climatology", "Climatology"),
    ("persistence", "Persistence"),
    ("harmonic", "Harmonic"),
    ("ar1", "AR(1)"),
]


def _loss_fn(name):
    return physssm_loss if name == "physssm" else baseline_loss


def _train(cfg, name, train_loader, val_loader, t_mean, t_std, q_mean, q_std, res_amp, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg, name, t_mean, t_std, q_mean, q_std, res_amp)
    t0 = time.perf_counter()
    train_model(model, train_loader, val_loader, _loss_fn(name), cfg, ckpt_name=f"{name}.pt", desc=name)
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    per_epoch = elapsed / max(1, cfg.train.epochs)
    return model, n_params, per_epoch


def _direct_origins(cfg, dates, t2m, features, te):
    lookback = cfg.data.input_len
    flen = cfg.model.forecast_horizon
    test_start = int(np.argmax(te))
    origins = list(range(test_start, len(t2m) - lookback - flen, flen))
    if not origins:
        origins = [test_start]
    X = np.stack([features[o : o + lookback] for o in origins])
    Y = np.stack([t2m[o + lookback : o + lookback + flen] for o in origins])
    return origins, X, Y


def _clim_matrix(cfg, dates, origins):
    lookback = cfg.data.input_len
    flen = cfg.model.forecast_horizon
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    return np.stack([cfg._climo_365[np.clip(doy[o + lookback : o + lookback + flen], 1, 365) - 1] for o in origins])


def _pint_direct(cfg, model, X, device):
    preds = []
    bs = cfg.train.batch_size
    model.eval()
    with torch.no_grad():
        for b in range(0, len(X), bs):
            xb = torch.tensor(X[b : b + bs], dtype=torch.float32, device=device)
            y = model(xb).cpu().numpy()
            y = y * float(model.t_std.cpu().numpy()) + float(model.t_mean.cpu().numpy())
            preds.append(y)
    return np.concatenate(preds, axis=0)


def _next_direct(cfg, model, X, origins, dates, lat, lon, device):
    from thermassm.rollout import _next_feature

    lookback = cfg.data.input_len
    flen = cfg.model.forecast_horizon
    dates_list = dates.astype("datetime64[D]").tolist()
    preds = []
    model.eval()
    with torch.no_grad():
        for i, x0 in enumerate(X):
            x = x0.copy()
            d = dates_list[origins[i] + lookback - 1]
            out = []
            for _ in range(flen):
                x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                y = model(x_t)
                p = float(np.clip(y[0, -1].item(), 150.0, 400.0))
                out.append(p)
                nf, d = _next_feature(x[-1], d, p, lat, lon)
                x = np.concatenate([x, nf[None, :]], axis=0)[-lookback:]
            preds.append(out)
    return np.array(preds)


def _stat_pred(cfg, dates, t2m, tr, origins, kind, last_anom):
    flen = cfg.model.forecast_horizon
    lookback = cfg.data.input_len
    climo = cfg._climo_365
    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    rows = []
    for o in origins:
        start = dates[o + lookback]
        if kind == "climatology":
            rows.append(climo[np.clip(doy[o + lookback : o + lookback + flen], 1, 365) - 1])
        elif kind == "persistence":
            rows.append(np.full(flen, t2m[o + lookback - 1], dtype=np.float32))
        elif kind == "harmonic":
            rows.append(harmonic_predict(dates, t2m, tr, str(start), flen))
        elif kind == "ar1":
            rows.append(ar1_predict(dates, t2m, tr, climo, str(start), flen,
                                    t2m[o + lookback - 1] - climo[np.clip(doy[o + lookback - 1], 1, 365) - 1]))
    return np.stack(rows)


def _inference_speed(cfg, model, name, data, device):
    from thermassm.rollout import rollout_sequence

    dates, t2m, features, tr, va, te = data
    test_start = int(np.argmax(te))
    mode, _, is_physssm, lookback = model_spec(name, cfg)
    if hasattr(model, "forecast_horizon"):
        is_physssm = True
    t0 = time.perf_counter()
    rollout_sequence(
        model, dates[test_start:], t2m[test_start:], features[test_start:],
        lookback, 365, cfg.data.lat, cfg.data.lon, device, mode=mode, is_physssm=is_physssm,
    )
    elapsed = time.perf_counter() - t0
    return int(365 / elapsed) if elapsed > 0 else 0


def _peak_vram():
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    return "N/A (CPU)"


def table1(cfg, epochs, seeds):
    cfg.train.epochs = epochs
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    data = (dates, t2m, features, tr, va, te)
    device = torch.device(cfg.train.device)
    origins, X, Y = _direct_origins(cfg, dates, t2m, features, te)
    clim = _clim_matrix(cfg, dates, origins)
    test_start = int(np.argmax(te))
    lookback = cfg.data.input_len

    doy = np.array([d.timetuple().tm_yday for d in dates.astype("datetime64[D]").tolist()])
    last_anom0 = float(t2m[test_start + lookback - 1] - cfg._climo_365[np.clip(doy[test_start + lookback - 1], 1, 365) - 1])

    rows = []

    def add_row(name, category, direct, long=None, params=0, per_epoch_s=None, steps_per_sec=None, peak_vram=None):
        row = {"name": name, "category": category, "params": params,
               "direct": _round_nested(direct), "seeds": len(seeds)}
        if long is not None:
            row["long"] = _round_nested(long)
        if per_epoch_s is not None:
            row["per_epoch_s"] = round(per_epoch_s, 2)
            row["steps_per_sec"] = steps_per_sec
            row["peak_vram"] = peak_vram
        rows.append(row)

    for kind, name in BASELINES:
        pred = _stat_pred(cfg, dates, t2m, tr, origins, kind, last_anom0)
        add_row(name, "Baseline", evaluate_direct_block(pred, Y, clim))

    rollout_save = {}
    for name in tqdm(NEURAL_MODELS, desc="table1"):
        mode, predict_len, _, lookback = model_spec(name, cfg)
        train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
        val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)

        per_seed_direct, per_seed_long = [], []
        n_params = per_epoch = steps_per_sec = peak_vram = None
        for s in seeds:
            model, n_params, per_epoch = _train(cfg, name, train_loader, val_loader, t_mean, t_std, q_mean, q_std, res_amp, s)
            model.to(device)
            if name == "physssm":
                pred, _, _ = run_direct_block(cfg, model, data)
            elif name.startswith("pint"):
                pred = _pint_direct(cfg, model, X, device)
            else:
                pred = _next_direct(cfg, model, X, origins, dates, cfg.data.lat, cfg.data.lon, device)
            rollout_results = run_rollouts(cfg, model, name, data)
            per_seed_direct.append(evaluate_direct_block(pred, Y, clim))
            per_seed_long.append(evaluate_results(rollout_results))
            if s == seeds[0]:
                steps_per_sec = _inference_speed(cfg, model, name, data, device)
                peak_vram = _peak_vram()
                if name in ("physssm", "pint-gru"):
                    rollout_save[f"pred_730_{name}"] = rollout_results[730]["pred"]
                    rollout_save.setdefault("true_730", rollout_results[730]["true"])

        direct = per_seed_direct[0] if len(seeds) == 1 else summarize_seeds(per_seed_direct)
        long = per_seed_long[0] if len(seeds) == 1 else summarize_seeds(per_seed_long)
        cat = "Proposed" if name == "physssm" else ("Physics-Informed" if name.startswith("pint") else "Data-Driven")
        label = "PhysSSM" if name == "physssm" else name.upper()
        add_row(label, cat, direct, long, n_params, per_epoch, steps_per_sec, peak_vram)
        print(f"table1 {name}: 30d RMSE={_lead_rmse(direct, 30):.3f}")

    if rollout_save:
        np.savez(OUT / "rollouts_table1.npz", **rollout_save)

    out = {"meta": _meta(cfg, seeds), "rows": rows}
    (OUT / "benchmark_table1.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table1.json")


def _lead_rmse(direct, lead):
    entry = direct.get(str(lead))
    if entry is None:
        entry = direct.get(lead)
    if entry is None:
        return 0.0
    rmse = entry.get("rmse")
    if isinstance(rmse, dict):
        return rmse.get("mean", 0.0)
    return float(rmse)


def _round_nested(d, depth=0):
    if not isinstance(d, dict):
        return round(d, 3) if isinstance(d, float) else d
    return {k: _round_nested(v, depth + 1) for k, v in d.items()}


def table2(cfg, epochs, seeds):
    cfg.train.epochs = epochs
    rows = []
    for site, (lat, lon) in tqdm(SITES.items(), desc="table2"):
        zcfg = make_config(**{"data.lat": lat, "data.lon": lon, "data.use_synthetic": cfg.data.use_synthetic,
                              "train.device": cfg.train.device, "train.epochs": epochs})
        dates, t2m, features, tr, va, te = load_and_split(zcfg)
        t_mean, t_std, q_mean, q_std, res_amp = _stats(zcfg)
        data = (dates, t2m, features, tr, va, te)
        device = torch.device(zcfg.train.device)
        zone_rmse = {}
        for name in ["pint-gru", "physssm"]:
            mode, predict_len, _, lookback = model_spec(name, zcfg)
            train_loader = make_loaders(zcfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
            val_loader = make_loaders(zcfg, dates, t2m, features, va, False, mode, predict_len, lookback)
            long_rmse = []
            for s in seeds:
                model, _, _ = _train(zcfg, name, train_loader, val_loader, t_mean, t_std, q_mean, q_std, res_amp, s)
                model.to(device)
                long = evaluate_results(run_rollouts(zcfg, model, name, data))
                long_rmse.append(long[730]["rmse"])
            zone_rmse[name] = round(float(np.mean(long_rmse)), 3)
        improvement = round(100.0 * (zone_rmse["pint-gru"] - zone_rmse["physssm"]) / zone_rmse["pint-gru"], 1)
        rows.append({"site": site, "lat": lat, "lon": lon, "pint_gru": zone_rmse["pint-gru"],
                     "physssm": zone_rmse["physssm"], "improvement_pct": improvement})
        print(f"table2 {site}: {rows[-1]}")
    out = {"meta": _meta(cfg, seeds), "rows": rows}
    (OUT / "benchmark_table2.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table2.json")


def table3(cfg, epochs, seeds):
    cfg.train.epochs = epochs
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    data = (dates, t2m, features, tr, va, te)
    mode, predict_len, _, lookback = model_spec("physssm", cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)

    labels = {
        "a0_s4d": ["A0 S4D", "None", "Yes", "No"],
        "a1_bounded_s4d": ["A1 Bounded S4D", "None", "Yes", "Yes"],
        "a2_sho": ["A2 SHO-anchor", "SHO/Harmonic", "Yes", "Yes"],
        "a3_physssm": ["A3 PhysSSM", "Thermal", "Yes", "Yes"],
        "a4_unbounded": ["A4 Unbounded", "Thermal", "Yes", "No"],
        "a5_unconstrained": ["A5 Unconstrained S4D", "Thermal", "No", "Yes"],
    }
    rows = []
    for key, kw in tqdm(ABLATION_CONFIGS.items(), desc="table3"):
        rmse30, rmse730, drift, psd = [], [], [], []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            model = AblationPhysSSM(cfg, res_amp=res_amp, t_mean=t_mean, t_std=t_std, q_mean=q_mean, q_std=q_std, **kw)
            train_model(model, train_loader, val_loader, ablation_loss, cfg, ckpt_name=f"ablation_{key}.pt", desc=key)
            pred, target, clim = run_direct_block(cfg, model, data)
            direct = evaluate_direct_block(pred, target, clim)
            long = evaluate_results(run_rollouts(cfg, model, key, data))
            rmse30.append(direct[30]["rmse"])
            rmse730.append(long[730]["rmse"])
            drift.append(long[730]["drift"])
            psd.append(long[730]["psd"])
        rows.append({
            "config": labels[key][0], "anchor": labels[key][1], "stable": labels[key][2], "bounded": labels[key][3],
            "rmse_30": round(float(np.mean(rmse30)), 3),
            "rmse_30_std": round(float(np.std(rmse30)), 3),
            "rmse_730": round(float(np.mean(rmse730)), 3),
            "rmse_730_std": round(float(np.std(rmse730)), 3),
            "drift": round(float(np.mean(drift)), 3),
            "psd": round(float(np.mean(psd)), 3),
        })
        print(f"table3 {key}: {rows[-1]}")
    out = {"meta": _meta(cfg, seeds), "rows": rows}
    (OUT / "benchmark_table3.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table3.json")


def table4(cfg, epochs, seeds=None):
    data1 = json.loads((OUT / "benchmark_table1.json").read_text())
    rows = []
    for r in data1["rows"]:
        if r["category"] == "Baseline":
            continue
        rows.append({
            "model": r["name"],
            "params": r["params"],
            "train_time_per_epoch_s": r.get("per_epoch_s"),
            "peak_vram": r.get("peak_vram", "N/A (CPU)"),
            "steps_per_sec": r.get("steps_per_sec"),
        })
        print(f"table4 {r['name']}: {rows[-1]}")
    out = {"meta": data1["meta"], "rows": rows}
    (OUT / "benchmark_table4.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table4.json")


def _meta(cfg, seeds):
    return {
        "data_source": "synthetic" if cfg.data.use_synthetic else "era5",
        "lat": cfg.data.lat,
        "lon": cfg.data.lon,
        "epochs": cfg.train.epochs,
        "device": cfg.train.device,
        "input_len": cfg.data.input_len,
        "forecast_len": cfg.model.forecast_horizon,
        "seeds": list(seeds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["table1", "table2", "table3", "table4", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--site", default="Denver", help="site name for table1/table3 (from SITES)")
    args = parser.parse_args()

    seeds = args.seeds or [0]
    lat, lon = SITES.get(args.site, SITES["Denver"])
    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device,
                         "train.epochs": args.epochs, "data.lat": lat, "data.lon": lon})
    OUT.mkdir(parents=True, exist_ok=True)
    stages = ["table1", "table2", "table3", "table4"] if args.stage == "all" else [args.stage]
    for stage in stages:
        {"table1": table1, "table2": table2, "table3": table3, "table4": table4}[stage](cfg, args.epochs, seeds)


if __name__ == "__main__":
    main()

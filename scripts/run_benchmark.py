#!/usr/bin/env python3
"""Run the actual benchmark and write per-stage JSON results."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm.ablations import AblationPhysSSM, ABLATION_CONFIGS
from thermassm.experiment import (
    build_model,
    climatology_predict,
    evaluate_results,
    load_and_split,
    make_config,
    persistence_predict,
    run_rollouts,
)
from thermassm.metrics import evaluate_forecast
from thermassm.train import train_model
from thermassm.losses import ablation_loss

OUT = Path("results")

MODELS = [
    "lstm", "gru", "rnn", "pint-lstm", "pint-gru",
    "patchtst", "vanilla_s4d", "climode", "physssm",
]

ZONES = [
    ("Tropical / Equatorial", 0.0, 100.0, "0.0N, 100.0E (Padang)"),
    ("Temperate Continental", 40.0, -105.0, "40.0N, 105.0W (Denver)"),
    ("Subtropical Desert", 24.4, 54.3, "24.4N, 54.3E (Abu Dhabi)"),
    ("Polar / Subarctic", 65.0, 25.0, "65.0N, 25.0E (Oulu)"),
]


def _time(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def train_with_timing(cfg, name, loader_pair):
    train_loader, val_loader = loader_pair
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    model = build_model(cfg, name)

    def run():
        return train_model(model, train_loader, val_loader, loss_fn(name), cfg, ckpt_name=f"{name}.pt")

    (history, best_val), elapsed = _time(run)
    n_params = sum(p.numel() for p in model.parameters())
    per_epoch = elapsed / max(1, cfg.train.epochs)
    return model, best_val, n_params, per_epoch


def loss_fn(name):
    from thermassm.losses import baseline_loss, composite_loss

    return composite_loss if name == "physssm" else baseline_loss


def table1(cfg, epochs):
    cfg.train.epochs = epochs
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    test_start = int(np.argmax(te))
    input_len = cfg.data.input_len

    from torch.utils.data import DataLoader

    from thermassm.data.dataset import ClimateDataset

    def loader(mask, shuffle):
        ds = ClimateDataset(dates[mask], t2m[mask], cfg.data.lat, cfg.data.lon, input_len)
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(tr, True), loader(va, False)
    data = (dates, t2m, features, tr, va, te)

    rows = []
    rollout_save = {}
    for name in MODELS:
        model, best_val, n_params, per_epoch = train_with_timing(cfg, name, (train_loader, val_loader))
        results = run_rollouts(cfg, model, name, data)
        met = evaluate_results(results)
        steps_per_sec = _inference_speed(cfg, model, name, data)
        if name in ("physssm", "pint-lstm", "patchtst"):
            rollout_save[f"pred_730_{name}"] = results[730]["pred"]
            rollout_save.setdefault("true_730", results[730]["true"])
        rows.append({
            "category": _category(name),
            "name": _pretty_name(name),
            "rmse": {str(h): round(met[h]["rmse"], 3) for h in cfg.data.horizons},
            "drift_730": round(met[730]["drift"], 3),
            "psd_730": round(met[730]["psd"], 3),
            "params": n_params,
            "per_epoch_s": round(per_epoch, 2),
            "steps_per_sec": steps_per_sec,
            "peak_vram": _peak_vram(),
            "val_loss": round(best_val, 4),
        })
        print(f"table1 {name}: rmse={rows[-1]['rmse']} drift={rows[-1]['drift_730']}")

    if rollout_save:
        np.savez(OUT / "rollouts_table1.npz", **rollout_save)

    for name, pretty in [("climatology", "Climatology (30-yr Mean)"), ("persistence", "Persistence")]:
        row = {"category": "Baselines", "name": pretty, "rmse": {}, "params": 0, "per_epoch_s": 0.0}
        for horizon in cfg.data.horizons:
            true = t2m[test_start + input_len : test_start + input_len + horizon]
            if name == "climatology":
                pred = climatology_predict(dates, t2m, tr, str(dates[test_start + input_len]), horizon)
            else:
                pred = persistence_predict(t2m[test_start + input_len - 1], horizon)
            met = evaluate_forecast(true, pred)
            row["rmse"][str(horizon)] = round(met["rmse"], 3)
            if horizon == 730:
                row["drift_730"] = round(met["drift"], 3)
                row["psd_730"] = round(met["psd"], 3)
        rows.append(row)

    out = {"meta": _meta(cfg), "rows": rows}
    (OUT / "benchmark_table1.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table1.json")


def table2(cfg, epochs):
    cfg.train.epochs = epochs
    rows = []
    for zone, lat, lon, coords in ZONES:
        zcfg = make_config(
            **{"data.lat": lat, "data.lon": lon, "data.use_synthetic": cfg.data.use_synthetic,
               "train.device": cfg.train.device, "train.epochs": epochs}
        )
        dates, t2m, features, tr, va, te = load_and_split(zcfg)

        from torch.utils.data import DataLoader

        from thermassm.data.dataset import ClimateDataset

        def loader(mask, shuffle):
            ds = ClimateDataset(dates[mask], t2m[mask], lat, lon, cfg.data.input_len)
            return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)

        data = (dates, t2m, features, tr, va, te)
        zone_rmse = {}
        for name in ["pint-gru", "patchtst", "physssm"]:
            torch.manual_seed(cfg.train.seed)
            np.random.seed(cfg.train.seed)
            model = build_model(zcfg, name)
            train_model(model, loader(tr, True), loader(va, False), loss_fn(name), zcfg)
            results = run_rollouts(zcfg, model, name, data)
            met = evaluate_results(results)
            zone_rmse[name] = round(met[730]["rmse"], 3)
        pint = zone_rmse["pint-gru"]
        phys = zone_rmse["physssm"]
        improvement = round(100.0 * (pint - phys) / pint, 1) if pint != 0 else 0.0
        rows.append({
            "zone": zone, "coords": coords,
            "pint_gru": zone_rmse["pint-gru"],
            "patchtst": zone_rmse["patchtst"],
            "physssm": zone_rmse["physssm"],
            "improvement_pct": improvement,
        })
        print(f"table2 {zone}: {rows[-1]}")
    out = {"meta": _meta(cfg), "rows": rows}
    (OUT / "benchmark_table2.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table2.json")


def table3(cfg, epochs):
    cfg.train.epochs = epochs
    dates, t2m, features, tr, va, te = load_and_split(cfg)

    from torch.utils.data import DataLoader

    from thermassm.data.dataset import ClimateDataset

    def loader(mask, shuffle):
        ds = ClimateDataset(dates[mask], t2m[mask], cfg.data.lat, cfg.data.lon, cfg.data.input_len)
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)

    data = (dates, t2m, features, tr, va, te)
    labels = {
        "a_full": ["(a) Full Model", "Stefan-Boltzmann EBM", "Re(A) <= -delta (Lyapunov)", "Decoupled (mu_phys + R_theta)"],
        "b_sho": ["(b) Toy Physics", "Simple Harmonic (SHO)", "Re(A) <= -delta (Lyapunov)", "Decoupled"],
        "c_nophysics": ["(c) No Physics", "None", "Re(A) <= -delta (Lyapunov)", "Monolithic"],
        "d_unconstrained": ["(d) Unconstrained", "Stefan-Boltzmann EBM", "Unconstrained A", "Decoupled"],
        "e_monolithic": ["(e) Monolithic Head", "Stefan-Boltzmann EBM", "Re(A) <= -delta (Lyapunov)", "Monolithic y = f(h)"],
    }
    rows = []
    for key, kw in ABLATION_CONFIGS.items():
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        model = AblationPhysSSM(cfg, **kw)
        train_model(model, loader(tr, True), loader(va, False), ablation_loss, cfg)
        results = run_rollouts(cfg, model, f"ablation_{key}", data)
        met = evaluate_results(results)[730]
        rows.append({
            "config": labels[key][0],
            "physics": labels[key][1],
            "stability": labels[key][2],
            "head": labels[key][3],
            "rmse_730": round(met["rmse"], 3),
            "drift": round(met["drift"], 3),
            "csi95": round(met["csi95"], 3),
        })
        print(f"table3 {key}: {rows[-1]}")
    out = {"meta": _meta(cfg), "rows": rows}
    (OUT / "benchmark_table3.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table3.json")


def table4(cfg, epochs):
    data1 = json.loads((OUT / "benchmark_table1.json").read_text())
    rows = []
    for r in data1["rows"]:
        if r["name"].startswith(("Climatology", "Persistence")):
            continue
        rows.append({
            "model": r["name"],
            "params": r["params"],
            "train_time_50ep_min": round(r["per_epoch_s"] * 50 / 60.0, 2),
            "peak_vram": r["peak_vram"],
            "steps_per_sec": r["steps_per_sec"],
        })
        print(f"table4 {r['name']}: {rows[-1]}")
    out = {"meta": data1["meta"], "rows": rows}
    (OUT / "benchmark_table4.json").write_text(json.dumps(out, indent=2))
    print("wrote benchmark_table4.json")


def _inference_speed(cfg, model, name, data):
    from thermassm.rollout import rollout_sequence

    horizon = 365
    dates, t2m, features, tr, va, te = data
    test_start = int(np.argmax(te))
    t0 = time.perf_counter()
    rollout_sequence(
        model, dates[test_start:], t2m[test_start:], features[test_start:],
        cfg.data.input_len, horizon, cfg.data.lat, cfg.data.lon,
        torch.device(cfg.train.device), is_physssm=(name == "physssm"),
    )
    elapsed = time.perf_counter() - t0
    return int(horizon / elapsed) if elapsed > 0 else 0


def _peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    return "N/A (CPU)"


def _category(name):
    if name == "physssm":
        return "Proposed"
    if name.startswith("pint") or name == "climode":
        return "Physics-Informed"
    return "Pure Data-Driven"


def _pretty_name(name):
    mapping = {
        "physssm": "PhysSSM-EBM (Ours)",
        "lstm": "LSTM (Vanilla)",
        "gru": "GRU (Vanilla)",
        "rnn": "RNN (Vanilla)",
        "pint-lstm": "PINT-LSTM",
        "pint-gru": "PINT-GRU",
        "patchtst": "PatchTST",
        "vanilla_s4d": "Vanilla S4D",
        "climode": "ClimODE 1D",
    }
    return mapping.get(name, name)


def _meta(cfg):
    return {
        "data_source": "synthetic" if cfg.data.use_synthetic else "era5",
        "lat": cfg.data.lat,
        "lon": cfg.data.lon,
        "epochs": cfg.train.epochs,
        "device": cfg.train.device,
        "input_len": cfg.data.input_len,
        "horizons": list(cfg.data.horizons),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["table1", "table2", "table3", "table4", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = make_config(
        **{"data.use_synthetic": args.synthetic, "train.device": args.device, "train.epochs": args.epochs}
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stages = ["table1", "table2", "table3", "table4"] if args.stage == "all" else [args.stage]
    for stage in stages:
        {"table1": table1, "table2": table2, "table3": table3, "table4": table4}[stage](cfg, args.epochs)


if __name__ == "__main__":
    main()

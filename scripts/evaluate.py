#!/usr/bin/env python3
"""Evaluate trained models: direct 1/7/14/30-day block + 365/730-day rollout."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm.experiment import (
    _stats,
    build_model,
    climatology_predict,
    evaluate_direct_block,
    evaluate_results,
    harmonic_predict,
    load_and_split,
    make_config,
    persistence_predict,
    run_direct_block,
    run_rollouts,
    save_json,
)

MODELS = ["physssm", "lstm", "gru", "rnn", "pint-lstm", "pint-gru", "patchtst", "vanilla_s4d"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["physssm"])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device})
    device = torch.device(args.device)
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    data = (dates, t2m, features, tr, va, te)

    report = {}
    for name in args.models:
        model = build_model(cfg, name, t_mean, t_std, q_mean, q_std, res_amp)
        ckpt = Path(cfg.train.checkpoint_dir) / f"{name}.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        entry = {}
        if name == "physssm":
            pred, target, clim = run_direct_block(cfg, model, data)
            entry["direct"] = evaluate_direct_block(pred, target, clim)
        entry["long"] = evaluate_results(run_rollouts(cfg, model, name, data))
        report[name] = entry

    # Statistical baselines (long horizon only).
    for horizon in (365, 730):
        input_len = cfg.data.input_len
        test_start = int(np.argmax(te))
        true = t2m[test_start + input_len : test_start + input_len + horizon]
        start_str = str(dates[test_start + input_len])
        climo = climatology_predict(dates, t2m, tr, start_str, horizon)
        persist = persistence_predict(t2m[test_start + input_len - 1], horizon)
        harmonic = harmonic_predict(dates, t2m, tr, start_str, horizon)
        report.setdefault("climatology", {})["long"] = report.get("climatology", {}).get("long", {})
        report.setdefault("persistence", {})["long"] = report.get("persistence", {}).get("long", {})
        report.setdefault("harmonic", {})["long"] = report.get("harmonic", {}).get("long", {})
        report["climatology"]["long"][horizon] = {"rmse": float(np.sqrt(np.mean((true - climo) ** 2)))}
        report["persistence"]["long"][horizon] = {"rmse": float(np.sqrt(np.mean((true - persist) ** 2)))}
        report["harmonic"]["long"][horizon] = {"rmse": float(np.sqrt(np.mean((true - harmonic) ** 2)))}

    save_json(report, "results/metrics.json")

    for name, entry in report.items():
        if "direct" in entry:
            leads = "  ".join(f"{k}d={m['rmse']:.2f}" for k, m in entry["direct"].items())
        else:
            leads = ""
        long = "  ".join(f"{h}d={m['rmse']:.2f}" for h, m in entry["long"].items())
        print(f"{name:>14s}: direct[{leads}]  long[{long}]")


if __name__ == "__main__":
    main()

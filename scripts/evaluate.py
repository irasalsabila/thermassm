#!/usr/bin/env python3
"""Evaluate trained models with multi-year rollouts and report metrics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm.experiment import (
    build_model,
    climatology_predict,
    evaluate_results,
    get_t_stats,
    load_and_split,
    make_config,
    persistence_predict,
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

    test_start = int(np.argmax(te))
    t_mean, t_std = get_t_stats(t2m, tr)
    report = {}

    for name in args.models:
        model = build_model(cfg, name, t_mean, t_std)
        ckpt = f"{cfg.train.checkpoint_dir}/{name}.pt"
        if Path(ckpt).exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        data = (dates, t2m, features, tr, va, te)
        results = run_rollouts(cfg, model, name, data)
        report[name] = evaluate_results(results)

    for horizon in cfg.data.horizons:
        input_len = cfg.data.input_len
        true = t2m[test_start + input_len : test_start + input_len + horizon]
        climo = climatology_predict(dates, t2m, tr, str(dates[test_start + input_len]), horizon)
        persist = persistence_predict(t2m[test_start + input_len - 1], horizon)
        report.setdefault("climatology", {})
        report.setdefault("persistence", {})
        report["climatology"][horizon] = {"rmse": float(np.sqrt(np.mean((true - climo) ** 2)))}
        report["persistence"][horizon] = {"rmse": float(np.sqrt(np.mean((true - persist) ** 2)))}

    save_json(report, "results/metrics.json")
    for name, horizons in report.items():
        print(f"{name:>14s}: " + "  ".join(f"{h}d={m['rmse']:.2f}" for h, m in horizons.items()))


if __name__ == "__main__":
    main()

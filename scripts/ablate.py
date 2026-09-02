#!/usr/bin/env python3
"""Run the PhysSSM ablation suite (A0-A5)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm.ablations import AblationPhysSSM, ABLATION_CONFIGS
from thermassm.experiment import (
    _stats,
    evaluate_direct_block,
    evaluate_results,
    load_and_split,
    make_config,
    make_loaders,
    model_spec,
    run_direct_block,
    run_rollouts,
    save_json,
)
from thermassm.losses import ablation_loss
from thermassm.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = make_config(
        **{"data.use_synthetic": args.synthetic, "train.device": args.device, "train.epochs": args.epochs}
    )
    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    data = (dates, t2m, features, tr, va, te)

    mode, predict_len, _, lookback = model_spec("physssm", cfg)
    train_loader = make_loaders(cfg, dates, t2m, features, tr, True, mode, predict_len, lookback)
    val_loader = make_loaders(cfg, dates, t2m, features, va, False, mode, predict_len, lookback)

    report = {}
    for key, kw in ABLATION_CONFIGS.items():
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        model = AblationPhysSSM(
            cfg, res_amp=res_amp, t_mean=t_mean, t_std=t_std, q_mean=q_mean, q_std=q_std, **kw
        )
        train_model(model, train_loader, val_loader, ablation_loss, cfg, ckpt_name=f"ablation_{key}.pt", desc=key)
        pred, target, clim = run_direct_block(cfg, model, data)
        direct = evaluate_direct_block(pred, target, clim)
        long = evaluate_results(run_rollouts(cfg, model, key, data))
        report[key] = {"direct": direct, "long": long}
        leads = "  ".join(f"{k}d={m['rmse']:.2f}" for k, m in direct.items())
        long730 = f"{long[730]['rmse']:.2f}" if 730 in long else "n/a"
        print(f"{key}: direct[{leads}]  730d RMSE={long730}")

    save_json(report, "results/ablations.json")


if __name__ == "__main__":
    main()

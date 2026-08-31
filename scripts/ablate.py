#!/usr/bin/env python3
"""Run the ablation suite (configurations a-e)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from thermassm.ablations import AblationPhysSSM, ABLATION_CONFIGS
from thermassm.experiment import (
    evaluate_results,
    load_and_split,
    make_config,
    run_rollouts,
    save_json,
)
from thermassm.losses import ablation_loss
from thermassm.train import train_model
from torch.utils.data import DataLoader

from thermassm.data.dataset import ClimateDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device, "train.epochs": args.epochs})
    dates, t2m, features, tr, va, te = load_and_split(cfg)

    def loader(mask):
        ds = ClimateDataset(dates[mask], t2m[mask], cfg.data.lat, cfg.data.lon, cfg.data.input_len)
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=mask is tr)

    train_loader = loader(tr)
    val_loader = loader(va)

    report = {}
    for key, kw in ABLATION_CONFIGS.items():
        torch.manual_seed(cfg.train.seed)
        model = AblationPhysSSM(cfg, **kw)
        train_model(model, train_loader, val_loader, ablation_loss, cfg)
        data = (dates, t2m, features, tr, va, te)
        results = run_rollouts(cfg, model, f"ablation_{key}", data)
        report[key] = evaluate_results(results)
        print(f"{key}: " + "  ".join(f"{h}d={m['rmse']:.2f}" for h, m in report[key].items()))

    save_json(report, "results/ablations.json")


if __name__ == "__main__":
    main()

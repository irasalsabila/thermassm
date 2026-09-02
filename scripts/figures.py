#!/usr/bin/env python3
"""Generate the paper figures from benchmark results / checkpoints."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm import figures
from thermassm.experiment import (
    _stats,
    build_model,
    load_and_split,
    make_config,
    run_rollouts,
)

OUT = Path("results/figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device})

    figures.figure1_architecture(str(OUT / "fig1_architecture.png"))

    dates, t2m, features, tr, va, te = load_and_split(cfg)
    t_mean, t_std, q_mean, q_std, res_amp = _stats(cfg)
    data = (dates, t2m, features, tr, va, te)

    physssm = build_model(cfg, "physssm", t_mean, t_std, q_mean, q_std, res_amp)
    ckpt = Path(cfg.train.checkpoint_dir) / "physssm.pt"
    if ckpt.exists():
        physssm.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
    figures.figure6_eigenvalues(physssm, str(OUT / "fig6_eigenvalues.png"))

    npz = Path("results/rollouts_table1.npz")
    if npz.exists():
        d = np.load(npz)
        true = d["true_730"]
        preds = {k[len("pred_730_"):]: d[k] for k in d.files if k.startswith("pred_730_")}
        print("Using saved rollouts from", npz)
    else:
        test_start = int(np.argmax(te))
        input_len = cfg.data.input_len
        horizon = 730
        true = t2m[test_start + input_len : test_start + input_len + horizon]
        preds = {}
        for name in ["physssm", "pint-gru"]:
            model = build_model(cfg, name, t_mean, t_std, q_mean, q_std, res_amp)
            ckpt = Path(cfg.train.checkpoint_dir) / f"{name}.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=args.device))
                model.to(args.device)
                results = run_rollouts(cfg, model, name, data)
                preds[name.replace("-", "_")] = results[horizon]["pred"]

    if preds:
        figures.figure2_rollout(None, true, preds, str(OUT / "fig2_rollout.png"))
        figures.figure3_drift(true, preds, str(OUT / "fig3_drift.png"))
        figures.figure4_psd(true, preds, str(OUT / "fig4_psd.png"))
        figures.figure5_extremes(true, preds, str(OUT / "fig5_extremes.png"))
        print("Figures written to", OUT)
    else:
        print("No rollout data found. Run scripts/run_benchmark.py --stage table1 first.")

    # Optional figures driven by saved benchmark JSON (no retraining).
    t1 = Path("results/benchmark_table1.json")
    if t1.exists():
        import json

        rows = json.loads(t1.read_text())["rows"]
        lead_rmse = {}
        for r in rows:
            if r["category"] == "Baseline" and r["name"] not in ("Climatology", "Harmonic"):
                continue
            d = r.get("direct", {})
            by_lead = {}
            for k, m in d.items():
                rmse = m.get("rmse")
                if isinstance(rmse, dict):
                    rmse = rmse.get("mean", 0.0)
                by_lead[int(k)] = rmse
            lead_rmse[r["name"]] = by_lead
        figures.figure_error_vs_lead(lead_rmse, str(OUT / "fig7_error_vs_lead.png"))

    t3 = Path("results/benchmark_table3.json")
    if t3.exists():
        import json

        rows = json.loads(t3.read_text())["rows"]
        figures.figure_ablation(rows, str(OUT / "fig8_ablation.png"))


if __name__ == "__main__":
    main()

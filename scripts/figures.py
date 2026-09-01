#!/usr/bin/env python3
"""Generate the six paper figures from benchmark results / checkpoints."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from thermassm import figures
from thermassm.experiment import build_model, load_and_split, make_config, run_rollouts

OUT = Path("results/figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = make_config(**{"data.use_synthetic": args.synthetic, "train.device": args.device})

    figures.figure1_architecture(str(OUT / "fig1_architecture.png"))

    # Eigenvalues figure only needs a (fresh) PhysSSM, no training required.
    physssm = build_model(cfg, "physssm")
    ckpt = f"{cfg.train.checkpoint_dir}/physssm.pt"
    if Path(ckpt).exists():
        physssm.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
    figures.figure6_eigenvalues(physssm, str(OUT / "fig6_eigenvalues.png"))

    # Trajectory figures: prefer saved rollouts from run_benchmark, else rerun.
    npz = Path("results/rollouts_table1.npz")
    if npz.exists():
        d = np.load(npz)
        true = d["true_730"]
        preds = {}
        for key in d.files:
            if key.startswith("pred_730_"):
                preds[key[len("pred_730_"):]] = d[key]
        print("Using saved rollouts from", npz)
    else:
        dates, t2m, features, tr, va, te = load_and_split(cfg)
        from thermassm.experiment import get_t_stats

        t_mean, t_std = get_t_stats(t2m, tr)
        data = (dates, t2m, features, tr, va, te)
        horizon = 730
        test_start = int(np.argmax(te))
        input_len = cfg.data.input_len
        true = t2m[test_start + input_len : test_start + input_len + horizon]
        preds = {}
        for name in ["physssm", "pint-lstm", "patchtst"]:
            model = build_model(cfg, name, t_mean, t_std)
            ckpt = f"{cfg.train.checkpoint_dir}/{name}.pt"
            if Path(ckpt).exists():
                model.load_state_dict(torch.load(ckpt, map_location=args.device))
                results = run_rollouts(cfg, model, name, data)
                preds[name] = results[horizon]["pred"]

    if preds:
        figures.figure2_rollout(None, true, preds, str(OUT / "fig2_rollout.png"))
        figures.figure3_drift(true, preds, str(OUT / "fig3_drift.png"))
        figures.figure4_psd(true, preds, str(OUT / "fig4_psd.png"))
        figures.figure5_extremes(true, preds, str(OUT / "fig5_extremes.png"))
        print("Figures written to", OUT)
    else:
        print("No rollout data found. Run scripts/run_benchmark.py --stage table1 first.")


if __name__ == "__main__":
    main()

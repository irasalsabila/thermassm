#!/usr/bin/env python3
"""Generate the six paper figures."""
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
    dates, t2m, features, tr, va, te = load_and_split(cfg)

    figures.figure1_architecture(str(OUT / "fig1_architecture.png"))

    data = (dates, t2m, features, tr, va, te)
    horizon = 730
    test_start = int(np.argmax(te))
    input_len = cfg.data.input_len
    true = t2m[test_start + input_len : test_start + input_len + horizon]
    preds = {}

    model_names = [m for m in ["physssm", "pint-lstm", "patchtst"] if m]
    for name in model_names:
        model = build_model(cfg, name)
        ckpt = f"{cfg.train.checkpoint_dir}/{name}.pt"
        if Path(ckpt).exists():
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
            results = run_rollouts(cfg, model, name, data)
            preds[name] = results[horizon]["pred"]
            if name == "physssm":
                figures.figure6_eigenvalues(model, str(OUT / "fig6_eigenvalues.png"))

    if preds:
        figures.figure2_rollout(None, true, preds, str(OUT / "fig2_rollout.png"))
        figures.figure3_drift(true, preds, str(OUT / "fig3_drift.png"))
        figures.figure4_psd(true, preds, str(OUT / "fig4_psd.png"))
        figures.figure5_extremes(true, preds, str(OUT / "fig5_extremes.png"))
        print("Figures written to", OUT)
    else:
        print("No trained checkpoints found; train models first.")


if __name__ == "__main__":
    main()

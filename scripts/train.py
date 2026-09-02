#!/usr/bin/env python3
"""Train a model (physssm or a baseline)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermassm.experiment import make_config, run_training

MODELS = ["physssm", "lstm", "gru", "rnn", "pint-lstm", "pint-gru", "patchtst", "vanilla_s4d"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = make_config(
        **{"train.epochs": args.epochs, "data.use_synthetic": args.synthetic, "train.device": args.device}
    )
    model, history, best_val, _ = run_training(cfg, args.model)
    print(f"[{args.model}] best val loss = {best_val:.6f}")


if __name__ == "__main__":
    main()

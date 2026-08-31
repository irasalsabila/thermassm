"""Training loop."""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, loss_fn, cfg, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, y, cfg)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / n


def evaluate_epoch(model, loader, loss_fn, cfg, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model, x, y, cfg)
            total += loss.item() * x.shape[0]
            n += x.shape[0]
    return total / n


def train_model(model, train_loader, val_loader, loss_fn, cfg, ckpt_name="best.pt"):
    device = torch.device(cfg.train.device)
    model.to(device)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    best_val = float("inf")
    history = {"train": [], "val": []}
    for epoch in range(cfg.train.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, cfg, device)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, cfg, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{cfg.train.checkpoint_dir}/{ckpt_name}")
    return history, best_val

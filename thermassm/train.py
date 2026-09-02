"""Training loop."""
from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm


def _to_device(batch, device):
    return [b.to(device) if torch.is_tensor(b) else b for b in batch]


def train_epoch(model, loader, optimizer, loss_fn, cfg, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        loss = loss_fn(model, batch, cfg)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch[0].shape[0]
        n += batch[0].shape[0]
    return total / n


def evaluate_epoch(model, loader, loss_fn, cfg, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            loss = loss_fn(model, batch, cfg)
            total += loss.item() * batch[0].shape[0]
            n += batch[0].shape[0]
    return total / n


def train_model(model, train_loader, val_loader, loss_fn, cfg, ckpt_name="best.pt", desc="train"):
    device = torch.device(cfg.train.device)
    model.to(device)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    best_val = float("inf")
    history = {"train": [], "val": []}
    pbar = tqdm(range(cfg.train.epochs), desc=desc, leave=False)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, cfg, device)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, cfg, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        postfix = {"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"}
        summary = getattr(model, "param_summary", None)
        if summary is not None:
            postfix.update(summary())
        pbar.set_postfix(**postfix)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{cfg.train.checkpoint_dir}/{ckpt_name}")
    return history, best_val

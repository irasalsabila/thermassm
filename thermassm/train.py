"""Training loop."""
from __future__ import annotations

import math
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


def _param_groups(model, cfg):
    ebm = getattr(model, "ebm", None)
    if ebm is None:
        return model.parameters()
    ebm_ids = {id(p) for p in ebm.parameters()}
    other = [p for p in model.parameters() if id(p) not in ebm_ids]
    return [
        {"params": other, "lr": cfg.train.lr},
        {"params": list(ebm.parameters()), "lr": cfg.train.lr_ebm},
    ]


def _decay_lambda(cfg, epoch, lambda_init):
    if cfg.train.lambda_ebm_min < 0:
        return
    t = epoch / max(1, cfg.train.epochs - 1)
    cfg.train.lambda_ebm = cfg.train.lambda_ebm_min + 0.5 * (
        lambda_init - cfg.train.lambda_ebm_min
    ) * (1 + math.cos(math.pi * t))


def train_model(model, train_loader, val_loader, loss_fn, cfg, ckpt_name="best.pt", desc="train"):
    device = torch.device(cfg.train.device)
    model.to(device)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    lambda_init = cfg.train.lambda_ebm
    best_val = float("inf")
    history = {"train": [], "val": []}
    pbar = tqdm(range(cfg.train.epochs), desc=desc, leave=False)
    for epoch in pbar:
        _decay_lambda(cfg, epoch, lambda_init)
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, cfg, device)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, cfg, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        postfix = {"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"}
        ebm = getattr(model, "ebm", None)
        if ebm is not None:
            postfix.update(ebm.param_summary())
        pbar.set_postfix(**postfix)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{cfg.train.checkpoint_dir}/{ckpt_name}")
    return history, best_val

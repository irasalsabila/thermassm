"""Loss functions for PhysSSM.

Core PhysSSM objective is MSE-only over the direct 30-day block. No physics
residual / smoothness / frequency-separation penalties.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def physssm_loss(model, batch, cfg) -> torch.Tensor:
    """MSE over the direct forecast block: x -> (Q_future, DOY_future) -> y."""
    x, forcing, y = batch
    pred = model(x, forcing)
    return F.mse_loss(pred, y)


def ablation_loss(model, batch, cfg) -> torch.Tensor:
    """MSE for ablation variants; they share the same (x, forcing, y) interface."""
    x, forcing, y = batch
    pred = model(x, forcing)
    return F.mse_loss(pred, y)


def baseline_loss(model, batch, cfg) -> torch.Tensor:
    """MSE for baselines (no future forcing)."""
    x, y = batch
    pred = model(x)
    if hasattr(model, "t_mean") and hasattr(model, "t_std"):
        y = (y - model.t_mean) / (model.t_std + 1e-8)
    loss = F.mse_loss(pred, y)
    if hasattr(model, "sho_loss"):
        # PINT soft physics regularizer (baseline only, not the PhysSSM core).
        loss = loss + cfg.train.lambda_physics * model.sho_loss(pred)
    return loss

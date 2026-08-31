"""Loss functions for ThermaSSM."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def composite_loss(model, x: torch.Tensor, y_target: torch.Tensor, cfg) -> torch.Tensor:
    y, mu, res = model.forward_full(x)
    mse = F.mse_loss(y, y_target)
    # Physics term = residual amplitude in K^2 (unit-consistent with MSE).
    # Keeps the full trajectory near the EBM macro drift without a C/dt escape-hatch scaling.
    l_ebm = (res ** 2).mean()
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    l_smooth = (d2 ** 2).mean()
    return mse + cfg.train.lambda_ebm * l_ebm + cfg.train.lambda_smooth * l_smooth


def baseline_loss(model, x: torch.Tensor, y_target: torch.Tensor, cfg) -> torch.Tensor:
    y = model(x)
    if hasattr(model, "t_mean") and hasattr(model, "t_std"):
        y_target = (y_target - model.t_mean) / (model.t_std + 1e-8)
    loss = F.mse_loss(y, y_target)
    if hasattr(model, "sho_loss"):
        loss = loss + cfg.train.lambda_physics * model.sho_loss(y)
    return loss


def ablation_loss(model, x: torch.Tensor, y_target: torch.Tensor, cfg) -> torch.Tensor:
    y, mu, res = model.forward_full(x)
    mse = F.mse_loss(y, y_target)
    has_ebm = getattr(model, "physics_formulation", None) == "ebm" and model.ebm is not None
    l_phys = (res ** 2).mean() if has_ebm else torch.zeros((), device=y.device)
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    l_smooth = (d2 ** 2).mean()
    return mse + cfg.train.lambda_ebm * l_phys + cfg.train.lambda_smooth * l_smooth

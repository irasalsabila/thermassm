"""Loss functions for ThermaSSM."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _lowpass(x: torch.Tensor, window: int) -> torch.Tensor:
    x = x.unsqueeze(1)
    x = F.avg_pool1d(x, kernel_size=window, stride=1, padding=window // 2)
    return x.squeeze(1)


def composite_loss(model, x: torch.Tensor, y_target: torch.Tensor, cfg) -> torch.Tensor:
    y, mu, res = model.forward_full(x)
    mse = F.mse_loss(y, y_target)
    # Physics term = residual amplitude in K^2 (unit-consistent with MSE).
    l_ebm = (res ** 2).mean()
    # Smoothness on the macro branch only (never the residual).
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    l_smooth = (d2 ** 2).mean()
    # Frequency separation: push residual energy to high frequencies so the EBM owns the seasonal mode.
    l_freq = (_lowpass(res, cfg.train.freq_window) ** 2).mean()
    return (
        mse
        + cfg.train.lambda_ebm * l_ebm
        + cfg.train.lambda_smooth * l_smooth
        + cfg.train.lambda_freq * l_freq
    )


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

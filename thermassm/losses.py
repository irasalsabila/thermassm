"""Loss functions for ThermaSSM."""
from __future__ import annotations

import torch
import torch.nn.functional as F

DT = 86400.0


def composite_loss(model, x: torch.Tensor, y_target: torch.Tensor, cfg) -> torch.Tensor:
    y, mu, res = model.forward_full(x)
    t_prev = x[..., 0]
    s = x[..., 1]
    mse = F.mse_loss(y, y_target)

    c = model.ebm.heat_capacity()
    flux = s * (1.0 - model.ebm.albedo(t_prev)) - model.ebm.emissivity() * model.ebm.sigma * t_prev ** 4
    dTdt = (y - t_prev) / DT
    ebm_err = c * dTdt - flux
    l_ebm = (ebm_err ** 2).mean()

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
    t_prev = x[..., 0]
    s = x[..., 1]
    mse = F.mse_loss(y, y_target)
    l_phys = torch.zeros((), device=y.device)
    if getattr(model, "physics_formulation", None) == "ebm" and model.ebm is not None:
        c = model.ebm.heat_capacity()
        flux = s * (1.0 - model.ebm.albedo(t_prev)) - model.ebm.emissivity() * model.ebm.sigma * t_prev ** 4
        dTdt = (y - t_prev) / DT
        l_phys = ((c * dTdt - flux) ** 2).mean()
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    l_smooth = (d2 ** 2).mean()
    return mse + cfg.train.lambda_ebm * l_phys + cfg.train.lambda_smooth * l_smooth

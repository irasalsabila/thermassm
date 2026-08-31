"""1D Radiative-Convective Energy Balance Model (EBM) prior."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class EBM(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.sigma = physics.stefan_boltzmann
        self.solar_constant = physics.solar_constant
        self.register_buffer("freeze_temp", torch.tensor(physics.freeze_temp))
        self.log_heat_capacity = nn.Parameter(
            torch.tensor(math.log(physics.init_heat_capacity))
        )
        self.eps_raw = nn.Parameter(torch.tensor(_logit(physics.init_eps)))
        self.albedo_land_raw = nn.Parameter(torch.tensor(_logit(physics.albedo_land)))
        self.albedo_ice_raw = nn.Parameter(torch.tensor(_logit(physics.albedo_ice)))
        self.log_width = nn.Parameter(torch.tensor(math.log(physics.albedo_width)))

    def heat_capacity(self) -> torch.Tensor:
        return torch.exp(self.log_heat_capacity)

    def emissivity(self) -> torch.Tensor:
        return torch.sigmoid(self.eps_raw) * 0.95 + 0.02

    def albedo(self, t: torch.Tensor) -> torch.Tensor:
        a_land = torch.sigmoid(self.albedo_land_raw)
        a_ice = torch.sigmoid(self.albedo_ice_raw)
        width = torch.exp(self.log_width)
        sig = torch.sigmoid((self.freeze_temp - t) / width)
        return a_land + (a_ice - a_land) * sig

    def net_flux(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return s * (1.0 - self.albedo(t)) - self.emissivity() * self.sigma * t ** 4

    def step(self, t: torch.Tensor, s: torch.Tensor, dt: float = 86400.0) -> torch.Tensor:
        return t + (dt / self.heat_capacity()) * self.net_flux(t, s)

    def param_summary(self) -> dict:
        a_land = torch.sigmoid(self.albedo_land_raw).item()
        a_ice = torch.sigmoid(self.albedo_ice_raw).item()
        return {
            "C": f"{self.heat_capacity().item():.2e}",
            "eps": f"{self.emissivity().item():.3f}",
            "a_land": f"{a_land:.3f}",
            "a_ice": f"{a_ice:.3f}",
        }


def _logit(x: float) -> float:
    import numpy as np

    return float(np.log(x / (1.0 - x)))

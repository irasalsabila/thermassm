"""Effective insolation-forced thermal response (thermal anchor).

The anchor models the predictable forced seasonal response with a stable
relaxation equation:

    mu[t+1] = rho * mu[t] + (1 - rho) * (a + b * Q[t])
    rho     = exp(-1 / tau),  tau > 0

Because 0 < rho < 1 and Q is bounded, the anchor response is bounded.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalAnchor(nn.Module):
    def __init__(self, tau_init: float = 30.0, a_init: float = 0.0, b_init: float = 1.0):
        super().__init__()
        # tau > 0 is enforced via softplus so rho = exp(-1/tau) stays in (0, 1).
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau_init, 1e-3)), dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))

    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def rho(self) -> torch.Tensor:
        return torch.exp(-1.0 / self.tau())

    def equilibrium(self, Q: torch.Tensor) -> torch.Tensor:
        """Steady-state response under constant forcing Q."""
        return self.a + self.b * Q

    def simulate(self, Q: torch.Tensor, mu0: torch.Tensor | None = None) -> torch.Tensor:
        """Roll the anchor forward over a forcing sequence Q of shape (B, L).

        Returns mu of shape (B, L). If mu0 is None it starts from the
        equilibrium response to the first forcing value.
        """
        rho = self.rho()
        B, L = Q.shape
        if mu0 is None:
            mu0 = self.equilibrium(Q[:, 0])
        else:
            mu0 = mu0 * torch.ones_like(Q[:, 0])
        mus = []
        mu = mu0
        for k in range(L):
            mu = rho * mu + (1.0 - rho) * self.equilibrium(Q[:, k])
            mus.append(mu)
        return torch.stack(mus, dim=1)

    def param_summary(self) -> dict:
        with torch.no_grad():
            return {
                "tau_d": f"{self.tau().item():.1f}",
                "rho": f"{self.rho().item():.3f}",
                "a_K": f"{self.a.item():.2f}",
                "b": f"{self.b.item():.4f}",
            }

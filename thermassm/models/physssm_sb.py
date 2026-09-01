"""PhysSSM ablation variant: Stefan-Boltzmann EBM prior (development prototype)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .s4d import S4D
from .ebm import EBM
from .physssm import scale_features


class PhysSSM_SB(nn.Module):
    """Prototype with a 0D Stefan-Boltzmann EBM prior; kept as an ablation variant."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.in_proj = nn.Linear(m.input_dim, m.d_model)
        self.ssm = S4D(m.d_model, m.d_state, mode="lyapunov", init="s4d-lin", delta=m.delta)
        self.res_head = nn.Sequential(
            nn.Linear(m.d_model, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, 1),
        )
        self.res_amp = nn.Parameter(torch.tensor(10.0))
        self.ebm = EBM(cfg.physics)

    def _residual(self, h: torch.Tensor) -> torch.Tensor:
        return self.res_amp * torch.tanh(self.res_head(h).squeeze(-1))

    def forward_full(self, x: torch.Tensor):
        t_prev = x[..., 0]
        s = x[..., 1]
        h = self.ssm(self.in_proj(scale_features(x)))
        res = self._residual(h)
        mu = self.ebm.step(t_prev, s)
        y = mu + res
        return y, mu, res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.forward_full(x)
        return y

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        u = self.in_proj(scale_features(x_t.unsqueeze(1)).squeeze(1))
        ssm_out, state = self.ssm.step(u, state)
        res = self._residual(ssm_out)
        mu = self.ebm.step(x_t[:, 0], x_t[:, 1])
        return mu + res, state

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.cfg.model.d_model,
            self.cfg.model.d_state,
            dtype=torch.complex64,
            device=device,
        )

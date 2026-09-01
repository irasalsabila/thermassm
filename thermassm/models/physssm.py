"""PhysSSM-EBM: decoupled EBM prior + Lyapunov-stable S4D residual head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .s4d import S4D
from .ebm import EBM

T_OFFSET = 273.15
T_SCALE = 30.0
S_SCALE = 340.0


def scale_features(x: torch.Tensor) -> torch.Tensor:
    t = (x[..., 0] - T_OFFSET) / T_SCALE
    s = x[..., 1] / S_SCALE
    rest = x[..., 2:]
    return torch.cat([t.unsqueeze(-1), s.unsqueeze(-1), rest], dim=-1)


class PhysSSM(nn.Module):
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

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return scale_features(x)

    def _residual(self, h: torch.Tensor) -> torch.Tensor:
        # Bounded decoder: tanh caps the residual, enabling closed-loop stability.
        return self.res_amp * torch.tanh(self.res_head(h).squeeze(-1))

    def forward_full(self, x: torch.Tensor):
        # x: (B, L, input_dim) raw features [T, S, doy_sin, doy_cos, lat_norm, lon_norm]
        t_prev = x[..., 0]
        s = x[..., 1]
        h = self.ssm(self.in_proj(self._scale(x)))
        res = self._residual(h)
        mu = self.ebm.step(t_prev, s)
        y = mu + res
        return y, mu, res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.forward_full(x)
        return y

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        # x_t: (B, input_dim), state: (B, d_model, d_state) complex
        u = self.in_proj(self._scale(x_t.unsqueeze(1)).squeeze(1))
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

"""PhysSSM: physics-guided stable state-space architecture for long-horizon temperature forecasting."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .s4d import S4D

T_OFFSET = 273.15
T_SCALE = 30.0
S_SCALE = 340.0


def scale_features(x: torch.Tensor) -> torch.Tensor:
    t = (x[..., 0] - T_OFFSET) / T_SCALE
    s = x[..., 1] / S_SCALE
    rest = x[..., 2:]
    return torch.cat([t.unsqueeze(-1), s.unsqueeze(-1), rest], dim=-1)


class PhysSSM(nn.Module):
    """Final proposed model: multiscale stable SSM + dissipative anomaly dynamics + bounded contextual residual."""

    def __init__(self, cfg, climo_365, formulation="equilibrium", amp=5.0):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.formulation = formulation
        self.register_buffer("climo", torch.tensor(climo_365, dtype=torch.float32))

        half_state = max(1, m.d_state // 2)
        self.in_proj = nn.Linear(m.input_dim, m.d_model)
        self.ssm_fast = S4D(
            m.d_model, half_state, mode="lyapunov", init="s4d-lin", delta=m.delta,
            dt_min=0.01, dt_max=0.5,
        )
        self.ssm_slow = S4D(
            m.d_model, half_state, mode="lyapunov", init="s4d-lin", delta=m.delta,
            dt_min=0.0001, dt_max=0.01,
        )
        self.res_head = nn.Sequential(
            nn.Linear(2 * m.d_model + 4, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, 1),
        )
        self.res_amp = nn.Parameter(torch.tensor(amp))
        self.log_tau = nn.Parameter(torch.tensor(math.log(30.0)))

    def _rho(self) -> torch.Tensor:
        tau = torch.exp(self.log_tau)
        return torch.exp(-1.0 / tau)

    def _doy_idx(self, x: torch.Tensor) -> torch.Tensor:
        doy = torch.atan2(x[..., 2], x[..., 3]) * (365.0 / (2 * math.pi))
        doy = doy % 365.0
        return (doy.round().long() - 1).clamp(0, 364)

    def _forward_shared(self, x: torch.Tensor):
        doy_idx = self._doy_idx(x)
        clim = self.climo[doy_idx]
        z = x[..., 0] - clim
        xs = scale_features(x)
        u = self.in_proj(xs)
        h_fast = self.ssm_fast(u)
        h_slow = self.ssm_slow(u)
        h = torch.cat([h_fast, h_slow], dim=-1)
        res_in = torch.cat([h, z.unsqueeze(-1), xs[..., 1:2], xs[..., 2:4]], dim=-1)
        tanh_out = torch.tanh(self.res_head(res_in).squeeze(-1))
        rho = self._rho()
        if self.formulation == "innovation":
            res = self.res_amp * tanh_out
        else:
            res = (1.0 - rho) * self.res_amp * tanh_out
        z_next = rho * z + res
        mu = clim + rho * z
        y = clim + z_next
        return y, mu, res

    def forward_full(self, x: torch.Tensor):
        return self._forward_shared(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self._forward_shared(x)
        return y

    def step(self, x_t: torch.Tensor, state):
        x_t = x_t.unsqueeze(1)
        doy_idx = self._doy_idx(x_t).squeeze(1)
        clim = self.climo[doy_idx]
        z = x_t[..., 0].squeeze(-1) - clim
        xs = scale_features(x_t).squeeze(1)
        u = self.in_proj(xs)
        uf, state_fast = self.ssm_fast.step(u, state[0])
        us, state_slow = self.ssm_slow.step(u, state[1])
        h = torch.cat([uf, us], dim=-1)
        res_in = torch.cat([h, z.unsqueeze(-1), xs[:, 1:2], xs[:, 2:4]], dim=-1)
        tanh_out = torch.tanh(self.res_head(res_in).squeeze(-1))
        rho = self._rho()
        if self.formulation == "innovation":
            res = self.res_amp * tanh_out
        else:
            res = (1.0 - rho) * self.res_amp * tanh_out
        z_next = rho * z + res
        return clim + z_next, (state_fast, state_slow)

    def initial_state(self, batch_size: int, device: torch.device):
        half_state = max(1, self.cfg.model.d_state // 2)
        z = torch.zeros(
            batch_size, self.cfg.model.d_model, half_state,
            dtype=torch.complex64, device=device,
        )
        return (z.clone(), z.clone())

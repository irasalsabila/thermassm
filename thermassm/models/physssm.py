"""PhysSSM: physics-anchored stable state-space model.

Decomposes temperature into a stable insolation-forced thermal anchor and a
bounded S4D-learned residual:

    mu[t+k] = anchor(Q[t+k])          (forced seasonal response)
    r[t+k]  = A_r * tanh(g(h_t, Q_future, DOY_future))
    T_hat[t+k] = mu[t+k] + r[t+k],    k = 1..30

Input: 90-day history [T, Q, sin(DOY), cos(DOY), lat, lon].
Output: direct 30-day forecast block.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .anchor import ThermalAnchor
from .s4d import S4D

# Fixed physical scalers used by the plain baseline backbones (kept for
# baselines.py compatibility).
T_OFFSET = 273.15
T_SCALE = 30.0
S_SCALE = 340.0


def scale_features(x: torch.Tensor) -> torch.Tensor:
    """Fixed-offset scaling for baseline backbones (not the PhysSSM encoder)."""
    t = (x[..., 0] - T_OFFSET) / T_SCALE
    s = x[..., 1] / S_SCALE
    rest = x[..., 2:]
    return torch.cat([t.unsqueeze(-1), s.unsqueeze(-1), rest], dim=-1)


class S4DBlock(nn.Module):
    """Pre-norm S4D block with GLU and residual connection (paper-faithful)."""

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0, mode: str = "s4d", init: str = "s4d-lin"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4D(d_model, d_state, mode=mode, init=init)
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ssm(self.norm(x))
        g = self.glu(h)
        h = g[..., : self.ssm.d_model] * torch.sigmoid(g[..., self.ssm.d_model :])
        return x + self.dropout(h)


class PhysSSM(nn.Module):
    """Proposed model (A3): thermal anchor + stable S4D + bounded residual."""

    def __init__(
        self,
        cfg,
        res_amp: float | None = None,
        t_mean: float = 0.0,
        t_std: float = 1.0,
        q_mean: float = 0.0,
        q_std: float = 1.0,
    ):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.forecast_horizon = m.forecast_horizon

        self.anchor = ThermalAnchor(cfg.anchor.tau_init, cfg.anchor.a_init, cfg.anchor.b_init)
        self.in_proj = nn.Linear(m.input_dim, m.d_model)
        self.blocks = nn.ModuleList(
            [S4DBlock(m.d_model, m.d_state, m.dropout, mode="s4d", init=m.s4d_init) for _ in range(m.s4d_layers)]
        )
        self.norm = nn.LayerNorm(m.d_model)

        # Residual decoder conditions on the last backbone state + future forcing
        # [Q, sin(DOY), cos(DOY)].
        self.decoder = nn.Sequential(
            nn.Linear(m.d_model + 3, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, 1),
        )

        amp = m.res_amp if res_amp is None else res_amp
        self.register_buffer("res_amp", torch.tensor(float(amp), dtype=torch.float32))
        self.register_buffer("t_mean", torch.tensor(float(t_mean), dtype=torch.float32))
        self.register_buffer("t_std", torch.tensor(float(t_std), dtype=torch.float32))
        self.register_buffer("q_mean", torch.tensor(float(q_mean), dtype=torch.float32))
        self.register_buffer("q_std", torch.tensor(float(q_std), dtype=torch.float32))

    def _scale_x(self, x: torch.Tensor) -> torch.Tensor:
        t = (x[..., 0] - self.t_mean) / (self.t_std + 1e-8)
        q = (x[..., 1] - self.q_mean) / (self.q_std + 1e-8)
        return torch.cat([t.unsqueeze(-1), q.unsqueeze(-1), x[..., 2:]], dim=-1)

    def _scale_forcing(self, forcing: torch.Tensor) -> torch.Tensor:
        q = (forcing[..., 0] - self.q_mean) / (self.q_std + 1e-8)
        return torch.cat([q.unsqueeze(-1), forcing[..., 1:]], dim=-1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(self._scale_x(x))
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def _residual(self, h_last: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        F = forcing.shape[1]
        h_last = h_last.unsqueeze(1).expand(-1, F, -1)
        fs = self._scale_forcing(forcing)
        dec_in = torch.cat([h_last, fs], dim=-1)
        raw = self.decoder(dec_in).squeeze(-1)
        return self.res_amp * torch.tanh(raw)

    def _anchor_response(self, x: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        Q_future = forcing[..., 0]
        mu0 = self.anchor.equilibrium(x[:, -1, 1])
        return self.anchor.simulate(Q_future, mu0=mu0)

    def forward_full(self, x: torch.Tensor, forcing: torch.Tensor):
        """Returns (y, mu, r) for diagnostics and ablation losses."""
        h = self._encode(x)
        mu = self._anchor_response(x, forcing)
        r = self._residual(h[:, -1], forcing)
        return mu + r, mu, r

    def forward(self, x: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.forward_full(x, forcing)
        return y

    def param_summary(self) -> dict:
        return self.anchor.param_summary()

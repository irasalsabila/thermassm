"""Hypothesis-driven ablation suite for PhysSSM (A0-A5).

A0  S4D                      -> T_hat            (no anchor, ordinary head)
A1  Bounded S4D              -> A_r * tanh(head) (no anchor)
A2  SHO / harmonic anchor    -> harmonic + bounded residual
A3  PhysSSM                  -> thermal anchor + bounded residual
A4  Unbounded PhysSSM        -> thermal anchor, no tanh bound
A5  Unconstrained S4D        -> thermal anchor, no stable-real-part constraint
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models.anchor import ThermalAnchor
from .models.physssm import S4DBlock


class AblationPhysSSM(nn.Module):
    """Configurable PhysSSM variant sharing the direct 90->30 interface."""

    def __init__(
        self,
        cfg,
        anchor: str = "thermal",
        bounded: bool = True,
        stable: bool = True,
        res_amp: float | None = None,
        t_mean: float = 0.0,
        t_std: float = 1.0,
        q_mean: float = 0.0,
        q_std: float = 1.0,
    ):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.anchor_kind = anchor
        self.bounded = bounded
        self.forecast_horizon = m.forecast_horizon

        if anchor == "thermal":
            self.anchor = ThermalAnchor(cfg.anchor.tau_init, cfg.anchor.a_init, cfg.anchor.b_init)
            self.sho_mean = None
            self.offset = None
        elif anchor == "sho":
            self.anchor = None
            self.sho_mean = nn.Parameter(torch.tensor(280.0, dtype=torch.float32))
            self.sho_a = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.sho_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.offset = None
        else:
            self.anchor = None
            self.sho_mean = None
            self.offset = nn.Parameter(torch.tensor(t_mean, dtype=torch.float32))

        mode = "s4d" if stable else "unconstrained"
        self.in_proj = nn.Linear(m.input_dim, m.d_model)
        self.blocks = nn.ModuleList(
            [S4DBlock(m.d_model, m.d_state, m.dropout, mode=mode, init=m.s4d_init) for _ in range(m.s4d_layers)]
        )
        self.norm = nn.LayerNorm(m.d_model)
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

    def _anchor_response(self, x: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        F = forcing.shape[1]
        if self.anchor_kind == "thermal":
            Q_future = forcing[..., 0]
            return self.anchor.simulate(Q_future, mu0=self.anchor.equilibrium(x[:, -1, 1]))
        if self.anchor_kind == "sho":
            sin = forcing[..., 1]
            cos = forcing[..., 2]
            return self.sho_mean + self.sho_a * sin + self.sho_b * cos
        if self.bounded and self.offset is not None:
            return self.offset.expand(forcing.shape[0], F)
        return torch.zeros(forcing.shape[0], F, device=forcing.device, dtype=forcing.dtype)

    def _residual(self, h_last: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        F = forcing.shape[1]
        h_last = h_last.unsqueeze(1).expand(-1, F, -1)
        fs = self._scale_forcing(forcing)
        raw = self.decoder(torch.cat([h_last, fs], dim=-1)).squeeze(-1)
        if self.bounded:
            return self.res_amp * torch.tanh(raw)
        return raw

    def forward_full(self, x: torch.Tensor, forcing: torch.Tensor):
        h = self._encode(x)
        mu = self._anchor_response(x, forcing)
        r = self._residual(h[:, -1], forcing)
        return mu + r, mu, r

    def forward(self, x: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.forward_full(x, forcing)
        return y


ABLATION_CONFIGS = {
    "a0_s4d": dict(anchor="none", bounded=False, stable=True),
    "a1_bounded_s4d": dict(anchor="none", bounded=True, stable=True),
    "a2_sho": dict(anchor="sho", bounded=True, stable=True),
    "a3_physssm": dict(anchor="thermal", bounded=True, stable=True),
    "a4_unbounded": dict(anchor="thermal", bounded=False, stable=True),
    "a5_unconstrained": dict(anchor="thermal", bounded=True, stable=False),
}

"""Ablation suite for PhysSSM-EBM components."""
from __future__ import annotations

import torch
import torch.nn as nn

from .models.physssm import T_OFFSET, T_SCALE, S_SCALE
from .models.s4d import S4D
from .models.ebm import EBM


class AblationPhysSSM(nn.Module):
    """Configurable PhysSSM variant for ablations."""

    def __init__(self, cfg, physics_formulation="ebm", stability="constrained", head="decoupled"):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.physics_formulation = physics_formulation
        self.head = head

        self.in_proj = nn.Linear(m.input_dim, m.d_model)
        self.ssm = S4D(
            m.d_model, m.d_state, m.delta,
            constrained=(stability == "constrained"),
        )
        self.res_head = nn.Sequential(
            nn.Linear(m.d_model, m.decoder_hidden),
            nn.GELU(),
            nn.Linear(m.decoder_hidden, 1),
        )
        if physics_formulation == "ebm":
            self.ebm = EBM(cfg.physics)
            self.sho_mean = None
        elif physics_formulation == "sho":
            self.ebm = None
            self.sho_mean = nn.Parameter(torch.tensor(285.0))
            self.sho_gain = nn.Parameter(torch.tensor(0.1))
            self.sho_ref = nn.Parameter(torch.tensor(340.0))
        else:
            self.ebm = None
            self.sho_mean = None

    def _scale(self, x):
        t = (x[..., 0] - T_OFFSET) / T_SCALE
        s = x[..., 1] / S_SCALE
        return torch.cat([t.unsqueeze(-1), s.unsqueeze(-1), x[..., 2:]], dim=-1)

    def _phys_prior(self, t_prev, s):
        if self.physics_formulation == "ebm":
            return self.ebm.step(t_prev, s)
        if self.physics_formulation == "sho":
            return self.sho_mean + self.sho_gain * (s - self.sho_ref)
        return torch.zeros_like(t_prev)

    def forward_full(self, x):
        t_prev = x[..., 0]
        s = x[..., 1]
        h = self.ssm(self.in_proj(self._scale(x)))
        res = self.res_head(h).squeeze(-1)
        mu = self._phys_prior(t_prev, s)
        if self.head == "decoupled":
            y = mu + res
        else:
            y = t_prev + res
        return y, mu, res

    def forward(self, x):
        y, _, _ = self.forward_full(x)
        return y


ABLATION_CONFIGS = {
    "a_full": dict(physics_formulation="ebm", stability="constrained", head="decoupled"),
    "b_sho": dict(physics_formulation="sho", stability="constrained", head="decoupled"),
    "c_nophysics": dict(physics_formulation="none", stability="constrained", head="monolithic"),
    "d_unconstrained": dict(physics_formulation="ebm", stability="unconstrained", head="decoupled"),
    "e_monolithic": dict(physics_formulation="ebm", stability="constrained", head="monolithic"),
}

"""Diagonal State Space (S4D) layer with optional Lyapunov stability constraint."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        delta: float = 0.1,
        constrained: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.delta = delta
        self.constrained = constrained

        if constrained:
            self.w = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
            self.nu = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        else:
            self.a_real = nn.Parameter(-0.5 * torch.rand(d_model, d_state))
            self.a_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

        self.b_real = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        log_dt = torch.rand(d_model) * (dt_max - dt_min) + dt_min
        self.log_dt = nn.Parameter(torch.log(log_dt))

    def _discretize(self):
        dt = torch.exp(self.log_dt)
        if self.constrained:
            a = -F.softplus(self.w) - self.delta + 1j * self.nu
        else:
            a = self.a_real + 1j * self.a_imag
        b = self.b_real + 1j * self.b_imag
        c = self.c_real + 1j * self.c_imag
        a_safe = a + 1e-6
        a_bar = torch.exp(dt.unsqueeze(-1) * a)
        b_bar = (a_bar - 1) / a_safe * b
        return a, a_bar, b_bar, c, dt

    def eigenvalues(self) -> torch.Tensor:
        a, _, _, _, _ = self._discretize()
        return a

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, d_model) -> (B, L, d_model)
        _, a_bar, b_bar, c, _ = self._discretize()
        L = u.shape[1]
        ones = torch.ones_like(a_bar.unsqueeze(-1))
        powers = torch.cat(
            [ones, torch.cumprod(a_bar.unsqueeze(-1).expand(-1, -1, L - 1), dim=-1)],
            dim=-1,
        )
        kernel = torch.einsum("ms,msk->mk", c * b_bar, powers)
        u = u.transpose(1, 2)
        uf = torch.fft.fft(u, n=2 * L, dim=-1)
        kf = torch.fft.fft(kernel, n=2 * L, dim=-1)
        y = torch.fft.ifft(uf * kf.unsqueeze(0), n=2 * L, dim=-1)[..., :L]
        return y.real.transpose(1, 2)

    def step(self, u: torch.Tensor, state: torch.Tensor):
        # u: (B, d_model), state: (B, d_model, d_state) complex
        _, a_bar, b_bar, c, _ = self._discretize()
        h = a_bar.unsqueeze(0) * state + b_bar.unsqueeze(0) * u.unsqueeze(-1)
        y = (c.unsqueeze(0) * h).sum(-1)
        return y.real, h

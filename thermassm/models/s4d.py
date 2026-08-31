"""Diagonal State Space (S4D) layer with HiPPO initialization and stability modes."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


def _legs_imag(N: int) -> np.ndarray:
    """Positive imaginary eigenvalues of the HiPPO-LegS normal component."""
    q = np.sqrt(2 * np.arange(N) + 1)
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                S[i, j] = 0.5 * q[i] * q[j] * np.sign(j - i)
    ev = np.imag(np.linalg.eigvals(S))
    return np.sort(ev[ev > 0])


def _hippo_imag(init: str, n_states: int) -> torch.Tensor:
    """Imaginary parts of the HiPPO-based init for `n_states` complex entries."""
    N = 2 * n_states
    n = np.arange(n_states, dtype=np.float64)
    if init == "s4d-lin":
        imag = np.pi * n
    elif init == "s4d-inv":
        imag = (N / np.pi) * (N / (2 * n + 1) - 1)
    elif init == "s4d-legs":
        imag = _legs_imag(N)
    else:
        raise ValueError(f"Unknown init scheme: {init}")
    return torch.tensor(imag, dtype=torch.float32)


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        mode: str = "s4d",
        init: str = "s4d-lin",
        delta: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.mode = mode
        self.delta = delta

        imag_init = _hippo_imag(init, d_state).unsqueeze(0).expand(d_model, -1).contiguous()

        if mode == "s4d":
            # A = -exp(a_re) + i * a_im (Hurwitz, Re < 0, no margin); init Re = -1/2
            self.a_re = nn.Parameter(torch.full((d_model, d_state), math.log(0.5)))
            self.a_im = nn.Parameter(imag_init.clone())
        elif mode == "lyapunov":
            # A = -softplus(w) - delta + i * nu (Re <= -delta); init Re = -1/2
            w_init = _softplus_inverse(0.5 - delta)
            self.w = nn.Parameter(torch.full((d_model, d_state), w_init))
            self.nu = nn.Parameter(imag_init.clone())
        elif mode == "unconstrained":
            self.a_real = nn.Parameter(torch.full((d_model, d_state), -0.5))
            self.a_imag = nn.Parameter(imag_init.clone())
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.b_real = nn.Parameter(torch.ones(d_model, d_state))
        self.b_imag = nn.Parameter(torch.zeros(d_model, d_state))
        self.c_real = nn.Parameter(torch.randn(d_model, d_state))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state))

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def _A(self) -> torch.Tensor:
        if self.mode == "s4d":
            return -torch.exp(self.a_re) + 1j * self.a_im
        if self.mode == "lyapunov":
            return -F.softplus(self.w) - self.delta + 1j * self.nu
        return self.a_real + 1j * self.a_imag

    def _discretize(self):
        dt = torch.exp(self.log_dt)
        a = self._A()
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

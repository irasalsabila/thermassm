"""Baseline models: RNN family (PINT), PatchTST, vanilla S4D, ClimODE."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .s4d import S4D
from .physssm import scale_features


class RNNBaseline(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, cell: str = "lstm"):
        super().__init__()
        self.cell = cell
        if cell == "lstm":
            self.rnn = nn.LSTM(input_dim, d_model, batch_first=True)
        elif cell == "gru":
            self.rnn = nn.GRU(input_dim, d_model, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, d_model, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(scale_features(x))
        return self.head(out).squeeze(-1)


class PINTModel(RNNBaseline):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        cell: str = "lstm",
        omega: float = 2 * math.pi / 365.0,
    ):
        super().__init__(input_dim, d_model, cell)
        self.omega = nn.Parameter(torch.tensor(omega))

    def sho_loss(self, y: torch.Tensor) -> torch.Tensor:
        d2 = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
        omega2 = self.omega ** 2
        return ((d2 + omega2 * y[:, 1:-1]) ** 2).mean()


class PatchTST(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        patch_len: int = 16,
        stride: int = 16,
        n_layers: int = 2,
        n_head: int = 4,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = scale_features(x)
        pad = (self.patch_len - (l % self.patch_len)) % self.patch_len
        xp = torch.cat([x, torch.zeros(b, pad, x.shape[-1], device=x.device)], dim=1)
        lp = xp.shape[1]
        n_patches = (lp - self.patch_len) // self.stride + 1
        patches = torch.stack(
            [xp[:, i * self.stride : i * self.stride + self.patch_len] for i in range(n_patches)],
            dim=1,
        )
        patches = patches.reshape(b, n_patches, -1)
        z = self.patch_embed(patches)
        z = self.encoder(z)
        out = self.head(z)
        out = out.reshape(b, n_patches * self.patch_len)
        return out[:, :l]


class VanillaS4D(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, d_state: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.ssm = S4D(d_model, d_state, constrained=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ssm(self.in_proj(scale_features(x)))
        return self.head(h).squeeze(-1)


class ClimODE(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, d_state: int = 64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_state)
        self.f = nn.Sequential(
            nn.Linear(d_state + input_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_state),
        )
        self.decoder = nn.Linear(d_state, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = scale_features(x)
        b, l, _ = x.shape
        h = self.encoder(x[:, 0])
        outs = []
        for t in range(l):
            h = h + self.f(torch.cat([h, x[:, t]], dim=-1))
            outs.append(self.decoder(h))
        return torch.stack(outs, dim=1).squeeze(-1)


def build_baseline(name: str, input_dim: int, d_model: int = 64) -> nn.Module:
    if name.startswith("pint"):
        cell = name.split("-")[-1] if "-" in name else "lstm"
        return PINTModel(input_dim, d_model, cell)
    if name in ("lstm", "gru", "rnn"):
        return RNNBaseline(input_dim, d_model, name)
    if name == "patchtst":
        return PatchTST(input_dim, d_model)
    if name == "vanilla_s4d":
        return VanillaS4D(input_dim, d_model)
    if name == "climode":
        return ClimODE(input_dim, d_model)
    raise ValueError(f"Unknown baseline: {name}")

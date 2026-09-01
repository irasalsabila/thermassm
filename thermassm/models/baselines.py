"""Baseline models: RNN family (PINT), PatchTST, vanilla S4D."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .s4d import S4D
from .physssm import scale_features, S_SCALE

T_OFFSET = 273.15
T_SCALE = 30.0


def _standardize_scale(x, t_mean, t_std):
    t = (x[..., 0] - t_mean) / (t_std + 1e-8)
    s = x[..., 1] / S_SCALE
    rest = x[..., 2:]
    return torch.cat([t.unsqueeze(-1), s.unsqueeze(-1), rest], dim=-1)


class RNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        cell: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell = cell
        rnn_kwargs = dict(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if cell == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(scale_features(x))
        return self.head(out).squeeze(-1)


class PINTModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        cell: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
        block_len: int = 30,
        t_mean: float = 0.0,
        t_std: float = 1.0,
    ):
        super().__init__()
        self.cell = cell
        self.block_len = block_len
        self.omega = 2 * math.pi / 365.0
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float32))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float32))

        rnn_kwargs = dict(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if cell == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)
        self.head = nn.Linear(d_model, block_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(_standardize_scale(x, self.t_mean, self.t_std))
        return self.head(out[:, -1])

    def sho_loss(self, y: torch.Tensor) -> torch.Tensor:
        d2 = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
        omega2 = self.omega ** 2
        return ((d2 + omega2 * y[:, 1:-1]) ** 2).mean()


class PatchBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x = self.norm1((x + a).transpose(1, 2)).transpose(1, 2)
        x = self.norm2((x + self.ffn(x)).transpose(1, 2)).transpose(1, 2)
        return x


class PatchTST(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_len: int = 90,
        horizon: int = 1095,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_layers: int = 3,
        n_head: int = 16,
        ffn: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.horizon = horizon
        self.block_len = horizon

        padded = input_len + stride
        self.n_patches = (padded - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [PatchBlock(d_model, n_head, ffn, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    def _scale_input(self, x: torch.Tensor) -> torch.Tensor:
        s = x[..., 1] / S_SCALE
        return torch.cat([x[..., 0:1], s.unsqueeze(-1), x[..., 2:]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = self._scale_input(x)

        t_mean = x[:, :, 0].mean(dim=1, keepdim=True)
        t_std = x[:, :, 0].std(dim=1, keepdim=True) + 1e-5
        x_norm = x.clone()
        x_norm[:, :, 0] = (x[:, :, 0] - t_mean) / t_std

        last = x_norm[:, -1:].expand(-1, self.stride, -1)
        xp = torch.cat([x_norm, last], dim=1)
        patches = torch.stack(
            [xp[:, i * self.stride : i * self.stride + self.patch_len] for i in range(self.n_patches)],
            dim=1,
        )
        patches = patches.reshape(b, self.n_patches, -1)
        z = self.patch_embed(patches) + self.pos

        for blk in self.blocks:
            z = blk(z)

        z = z.reshape(b, self.n_patches * self.d_model)
        out = self.head(z)

        t_mean = t_mean.squeeze(1)
        t_std = t_std.squeeze(1)
        return out * t_std.unsqueeze(-1) + t_mean.unsqueeze(-1)


class S4DBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4D(d_model, d_state, mode="s4d", init="s4d-lin")
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ssm(self.norm(x))
        g = self.glu(h)
        h = g[..., : self.ssm.d_model] * torch.sigmoid(g[..., self.ssm.d_model :])
        return x + self.dropout(h)


class VanillaS4D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [S4DBlock(d_model, d_state, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(scale_features(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1)


def build_baseline(
    name: str,
    input_dim: int,
    d_model: int = 64,
    input_len: int = 90,
    horizon: int = 1095,
    t_mean: float = 0.0,
    t_std: float = 1.0,
    cfg=None,
) -> nn.Module:
    if name.startswith("pint"):
        cell = name.split("-")[-1] if "-" in name else "lstm"
        block_len = cfg.model.pint_block if cfg is not None else 30
        return PINTModel(
            input_dim, d_model, cell,
            num_layers=cfg.model.num_layers if cfg is not None else 2,
            dropout=cfg.model.dropout if cfg is not None else 0.1,
            block_len=block_len, t_mean=t_mean, t_std=t_std,
        )
    if name in ("lstm", "gru", "rnn"):
        return RNNBaseline(
            input_dim, d_model, name,
            num_layers=cfg.model.num_layers if cfg is not None else 2,
            dropout=cfg.model.dropout if cfg is not None else 0.1,
        )
    if name in ("patchtst", "patchtst336"):
        if cfg is not None:
            return PatchTST(
                input_dim, input_len,
                horizon=cfg.model.patch_horizon,
                patch_len=cfg.model.patch_len,
                stride=cfg.model.patch_stride,
                d_model=cfg.model.patch_d_model,
                n_layers=cfg.model.patch_layers,
                n_head=cfg.model.patch_heads,
                ffn=cfg.model.patch_ffn,
                dropout=cfg.model.patch_dropout,
            )
        return PatchTST(input_dim, input_len, horizon=horizon)
    if name == "vanilla_s4d":
        n_layers = cfg.model.s4d_layers if cfg is not None else 4
        return VanillaS4D(input_dim, d_model, n_layers=n_layers)
    raise ValueError(f"Unknown baseline: {name}")

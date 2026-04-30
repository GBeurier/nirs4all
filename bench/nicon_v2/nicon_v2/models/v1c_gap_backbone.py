"""NiconV1c — Phase 1c GAP backbone (Cui-Fearn 2018 / DeepSpectra-inspired).

Replaces the strided large-kernel NICON backbone with a 4-block small-kernel
backbone + Global Average Pooling head + linear regression projection.

Architecture (default):

    [in_channels, L]                        # in_channels = 3 if concat-deriv else 1
    Block 1:  Conv1d(in→16, k=7, pad=3)  → Norm → GELU → Dropout → MaxPool(2)
    Block 2:  Conv1d(16→32,  k=5, pad=2) → Norm → GELU → Dropout → MaxPool(2)
    Block 3:  Conv1d(32→64,  k=3, pad=1) → Norm → GELU → Dropout → MaxPool(2)
    Block 4:  Conv1d(64→128, k=3, pad=1) → Norm → GELU → Dropout → MaxPool(2)
    GAP1d                                                              → (N, 128)
    Dropout(p_dense)
    Linear(128 → 1)                       # **linear** output (W1 fix)

`Norm` ∈ {`layer`, `batch`, `group`} — defaults to LayerNorm to be small-batch
friendly. `pool_kind` ∈ {`max`, `avg`} — default `max`. The model is invariant
to spectrum length: GAP averages over whatever timesteps remain, so it never
collapses on short inputs (W3 / W14 fix).

Channel-wise dropout is applied at the input (`spatial_dropout`) and after each
conv block (`p_drop`); standard `Dropout` is applied before the linear head
(`p_dense`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..preprocessing import ConcatDerivatives


def _make_norm(kind: str, num_features: int) -> nn.Module:
    if kind == "layer":
        return nn.GroupNorm(1, num_features)
    if kind == "batch":
        return nn.BatchNorm1d(num_features)
    if kind == "group":
        groups = max(1, min(8, num_features))
        # Pick the largest divisor ≤ 8.
        while num_features % groups != 0:
            groups -= 1
            if groups <= 1:
                groups = 1
                break
        return nn.GroupNorm(groups, num_features)
    raise ValueError(f"unknown norm kind: {kind!r}")


def _make_pool(kind: str) -> nn.Module:
    if kind == "max":
        return nn.MaxPool1d(kernel_size=2, stride=2)
    if kind == "avg":
        return nn.AvgPool1d(kernel_size=2, stride=2)
    raise ValueError(f"unknown pool kind: {kind!r}")


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        norm: str,
        p_drop: float,
        pool: str,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=padding)
        self.norm = _make_norm(norm, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(p=p_drop)
        self.pool = _make_pool(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        if x.shape[-1] >= 2:
            x = self.pool(x)
        return x


class NiconV1c(nn.Module):
    """4-block GAP-head 1-D CNN."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        kernels: tuple[int, ...] = (7, 5, 3, 3),
        channels: tuple[int, ...] = (16, 32, 64, 128),
        norm: str = "layer",
        pool: str = "max",
        spatial_dropout: float = 0.05,
        p_drop: float = 0.1,
        p_dense: float = 0.2,
        use_concat_derivatives: bool = False,
        include_snv_channel: bool = False,
        sg_window_d1: int = 11,
        sg_window_d2: int = 11,
        sg_polyorder: int = 2,
    ) -> None:
        super().__init__()
        in_ch_raw, seq_len = input_shape
        if in_ch_raw != 1 and use_concat_derivatives:
            raise ValueError("concat-derivatives requires a 1-channel raw input")
        if len(kernels) != len(channels):
            raise ValueError(f"kernels and channels must be same length; got {kernels} / {channels}")

        if use_concat_derivatives:
            self.preproc: nn.Module = ConcatDerivatives(
                window_length_d1=sg_window_d1,
                window_length_d2=sg_window_d2,
                polyorder=sg_polyorder,
                include_snv=include_snv_channel,
            )
            stem_in = 4 if include_snv_channel else 3
        else:
            self.preproc = nn.Identity()
            stem_in = in_ch_raw

        self.input_dropout = nn.Dropout1d(p=spatial_dropout)
        blocks: list[nn.Module] = []
        prev = stem_in
        for k, c in zip(kernels, channels):
            blocks.append(_ConvBlock(prev, c, kernel=k, norm=norm, p_drop=p_drop, pool=pool))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dense_dropout = nn.Dropout(p=p_dense)
        self.head = nn.Linear(prev, 1)

        self._n_pools = len(kernels)
        self._stem_channels = stem_in
        self._final_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"NiconV1c expects (N, C, L); got {tuple(x.shape)}")
        x = self.preproc(x)
        x = self.input_dropout(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dense_dropout(x)
        return self.head(x)


def build_nicon_v1c(input_shape: tuple[int, int], params: dict | None = None) -> NiconV1c:
    p = dict(params or {})
    return NiconV1c(
        input_shape=input_shape,
        kernels=tuple(p.get("kernels", (7, 5, 3, 3))),
        channels=tuple(p.get("channels", (16, 32, 64, 128))),
        norm=str(p.get("norm", "layer")),
        pool=str(p.get("pool", "max")),
        spatial_dropout=float(p.get("spatial_dropout", 0.05)),
        p_drop=float(p.get("p_drop", 0.1)),
        p_dense=float(p.get("p_dense", 0.2)),
        use_concat_derivatives=bool(p.get("use_concat_derivatives", False)),
        include_snv_channel=bool(p.get("include_snv_channel", False)),
        sg_window_d1=int(p.get("sg_window_d1", 11)),
        sg_window_d2=int(p.get("sg_window_d2", 11)),
        sg_polyorder=int(p.get("sg_polyorder", 2)),
    )

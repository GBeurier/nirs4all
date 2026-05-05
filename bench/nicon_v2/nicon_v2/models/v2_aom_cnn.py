"""NiconV2A — AOM-Superblock-inspired multi-branch CNN with channel-attention SE blocks.

Design (after Codex round 5 review):

* **Multi-branch front:** strict-linear AOM-extended bank (Identity + 5 SG +
  2 Detrend + FD + NW + Whittaker = 11 branches) plus optional CNN-only
  extras (Gaussian, SNV, MSC). Each branch outputs `(N, 1, L)`.
* **Per-branch RMS normalisation** (`RMSBranchNorm`) — analogous to
  AOM-Ridge `compute_block_scales_from_xt`.
* **Channel concatenation:** branches stacked along channel dim → `(N, B, L)`.
* **Trunk:** 3 residual Conv1D blocks (kernel 7 → 5 → 3, channels 32 → 64 → 96),
  each followed by GELU + LayerNorm + spatial Dropout1d(0.2). MaxPool(2)
  between blocks; the final block uses adaptive avg pool to handle variable
  input length.
* **Squeeze-and-Excite** between blocks (squeeze across channels via GAP →
  bottleneck FC → sigmoid → multiply each channel by its scalar weight).
  This is Codex's recommended AOM-MKL analogue: dataset-level (not
  sample-level) weighting via training.
* **Linear regression head** (no sigmoid, per H1).

The model is length-invariant (GAP head) and supports the cohort length range
{401, 576, 700, 1154, 2151}. Default 3 max-pool blocks; a fourth is enabled
via `extra_pool=True` only when input length ≥ 800.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..operators_torch import (
    RMSBranchNorm,
    aom_compact_branches_torch,
    aom_extended_lowrank_branches_torch,
    aom_extended_strict_linear_branches_torch,
    cnn_only_extra_branches_torch,
    full_branches_torch,
)


def _extended_lowrank_factory(p: int, trainable: bool = True, reg_lambda: float = 0.0,
                              matrix_trainable: bool | None = None,
                              rank: int = 16, **_) -> list[nn.Module]:
    """Adapter for the low-rank bank: forwards `rank` to the factory; ignores the
    `matrix_trainable` flag because low-rank Detrend/Whittaker follow `trainable`."""
    return aom_extended_lowrank_branches_torch(p=p, trainable=trainable, reg_lambda=reg_lambda, rank=rank)


class _SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block (Hu et al. 2018).

    Reduces channels to `channels // reduction`, then expands back. The output
    is multiplied by the input channelwise; weights are dataset-level (a
    function of the trunk activations averaged across L) — matching AOM-MKL's
    fold-local block weighting.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L). Gate produces (N, C); broadcast to (N, C, 1) and multiply.
        weights = self.gate(x).unsqueeze(-1)
        return x * weights


class _ResConvBlock(nn.Module):
    """Residual Conv1D block with LayerNorm and GELU.

    `Conv1D(C_in → C_out)` + `GroupNorm(1, C_out)` + GELU + Dropout1d
    + identity skip (with 1×1 conv if `C_in != C_out`).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 p_drop: float = 0.2, se_reduction: int | None = 4):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(p=p_drop)
        if se_reduction is not None and out_channels >= 4:
            self.se = _SqueezeExcite(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.se(x)
        return x + residual


class _DenseConvBlock(nn.Module):
    """V2Q — DenseNet-style block: concatenate input with conv output.

    Output channels = in_channels + growth_rate. The accumulated feature map
    is fed to the next block, which can re-use earlier representations
    without parameter inflation in the convs themselves (each block grows
    by `growth_rate`, not by total channels). The trunk's GAP head averages
    over wavelengths, so the final dense channel count maps to a small linear.
    """

    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int,
                 p_drop: float = 0.2, se_reduction: int | None = 4):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.norm = nn.GroupNorm(1, growth_rate)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(p=p_drop)
        if se_reduction is not None and growth_rate >= 4:
            self.se = _SqueezeExcite(growth_rate, reduction=se_reduction)
        else:
            self.se = nn.Identity()
        self.out_channels = in_channels + growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(x)
        new_features = self.norm(new_features)
        new_features = self.act(new_features)
        new_features = self.drop(new_features)
        new_features = self.se(new_features)
        return torch.cat([x, new_features], dim=1)


class _MultiKernelStemBlock(nn.Module):
    """V2O — multi-kernel parallel stem (DeepSpectra-Inception style).

    Replaces the first ResConvBlock with 4 parallel `Conv1D(in→out//4, kernel=k)`
    branches at kernels (3, 5, 7, 9), concatenated along channels. Targets
    spectra with features at multiple wavelength scales.

    The remaining trunk blocks are unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int, kernels: tuple[int, ...] = (3, 5, 7, 9),
                 p_drop: float = 0.2, se_reduction: int | None = 4):
        super().__init__()
        if out_channels % len(kernels) != 0:
            raise ValueError(f"out_channels {out_channels} must be divisible by len(kernels) {len(kernels)}")
        per = out_channels // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, per, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        ])
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(p=p_drop)
        if se_reduction is not None and out_channels >= 4:
            self.se = _SqueezeExcite(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        outs = [b(x) for b in self.branches]
        # All same length thanks to symmetric padding; concat along channels.
        x = torch.cat(outs, dim=1)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.se(x)
        return x + residual


class _WavelengthAttentionHead(nn.Module):
    """V2P — single-layer multi-head self-attention over post-trunk wavelength tokens.

    Replaces `AdaptiveAvgPool1d(1) → Flatten → Dropout → Linear(C → 1)` with:

    * Project (N, C, L) → (N, L, d_model) with d_model = C.
    * 1-layer multi-head self-attention (H heads).
    * Mean over the L wavelength axis → (N, d_model).
    * Linear(d_model → 1).

    Attention weights are dataset-level interpretable: pulling them out of the
    attention block gives "which wavelengths drive prediction".
    """

    def __init__(self, in_channels: int, num_heads: int = 4, p_drop: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads,
                                          dropout=p_drop, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(p=p_drop)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L). Treat L as sequence, C as embedding.
        x_seq = x.transpose(1, 2)                       # (N, L, C)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)    # self-attention
        attn_out = self.norm(attn_out + x_seq)          # residual + LN
        pooled = attn_out.mean(dim=1)                   # (N, C)
        return self.fc(self.dropout(pooled))


class _MoEHead(nn.Module):
    """R17 D — Mixture-of-Experts regression head.

    K linear experts + a softmax gate over the GAP feature vector.
    Final prediction = Σₖ gᵢ · headₖ(z), with optional ``out_dim=2`` for the
    heteroscedastic head (μ, log σ²) — orthogonal to MoE.
    """

    def __init__(self, in_features: int, out_dim: int = 1, num_experts: int = 2,
                 p_drop: float = 0.3):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(in_features, out_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
        self.dropout = nn.Dropout(p=p_drop)
        self.num_experts = num_experts
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        z = self.dropout(x)
        weights = torch.softmax(self.gate(z), dim=-1)                  # (N, K)
        outputs = torch.stack([e(z) for e in self.experts], dim=1)     # (N, K, out_dim)
        weighted = (weights.unsqueeze(-1) * outputs).sum(dim=1)        # (N, out_dim)
        return weighted


class _AOMTransformerTrunk(nn.Module):
    """V3 — Transformer encoder over wavelength tokens, after one CNN block.

    Replaces the second + third ResConvBlocks of V2L with:
    1. project (N, prev_C, L) → (N, L, d_model) via 1×1 conv
    2. nn.TransformerEncoder(num_layers, num_heads, d_model, dim_ff)
    3. transpose back to (N, d_model, L) for downstream pooling

    Different from V2P-attnHead (which replaced GAP+Linear): here attention
    is a trunk layer, not a head. Allows long-range wavelength dependencies
    while keeping GAP+Linear at the output.
    """

    def __init__(self, in_channels: int, d_model: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, p_drop: float = 0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=p_drop, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L). Project + reshape to (N, L, d_model) for attention.
        x = self.proj(x)
        x_seq = x.transpose(1, 2)              # (N, L, d_model)
        x_seq = self.encoder(x_seq)
        return x_seq.transpose(1, 2)           # (N, d_model, L)


class _ConformerBlock(nn.Module):
    """R17 I — Conformer block (Gulati 2020) for 1-D spectra.

    Sequence of:
      1. Macaron FFN (×0.5 residual)
      2. Multi-head self-attention (with sinusoidal-ish learnable position via
         rotary or simple LayerNorm — we use plain MHA over LayerNorm for
         simplicity given small `n_train` on this cohort)
      3. Conv module: pointwise conv → GLU → depthwise conv → BN → GELU →
         pointwise conv
      4. Macaron FFN (×0.5 residual)
      5. LayerNorm
    """

    def __init__(self, channels: int, kernel_size: int, num_heads: int = 4,
                 ffn_factor: int = 2, p_drop: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.ffn1 = nn.Sequential(
            nn.Linear(channels, channels * ffn_factor),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(channels * ffn_factor, channels),
            nn.Dropout(p_drop),
        )
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                          dropout=p_drop, batch_first=True)
        self.norm_conv = nn.LayerNorm(channels)
        # Conv module — operates on (N, C, L); we transpose to feed it.
        self.conv_pw1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.conv_dw = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=channels)
        self.conv_bn = nn.BatchNorm1d(channels)
        self.conv_act = nn.GELU()
        self.conv_pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_drop = nn.Dropout1d(p_drop)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn2 = nn.Sequential(
            nn.Linear(channels, channels * ffn_factor),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(channels * ffn_factor, channels),
            nn.Dropout(p_drop),
        )
        self.norm_out = nn.LayerNorm(channels)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L) → transpose to (N, L, C) for the LayerNorm + FFN parts.
        x_seq = x.transpose(1, 2)                          # (N, L, C)
        # Macaron FFN ×0.5
        x_seq = x_seq + 0.5 * self.ffn1(self.norm1(x_seq))
        # Self-attention over wavelengths
        x_norm = self.norm_attn(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        # Conv module (back to (N, C, L))
        x_conv_in = self.norm_conv(x_seq).transpose(1, 2)  # (N, C, L)
        h = self.conv_pw1(x_conv_in)                       # (N, 2C, L)
        h_a, h_b = h.chunk(2, dim=1)
        h = h_a * torch.sigmoid(h_b)                       # GLU
        h = self.conv_dw(h)
        h = self.conv_bn(h)
        h = self.conv_act(h)
        h = self.conv_pw2(h)
        h = self.conv_drop(h)
        x_seq = x_seq + h.transpose(1, 2)
        # Macaron FFN ×0.5
        x_seq = x_seq + 0.5 * self.ffn2(self.norm2(x_seq))
        x_seq = self.norm_out(x_seq)
        return x_seq.transpose(1, 2)                       # (N, C, L)


class _ConformerTrunk(nn.Module):
    """R17 I — full Conformer trunk: 1×1 projection + N Conformer blocks."""

    def __init__(self, in_channels: int, d_model: int = 64, num_blocks: int = 2,
                 num_heads: int = 4, kernel_size: int = 5, p_drop: float = 0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            _ConformerBlock(d_model, kernel_size=kernel_size, num_heads=num_heads,
                            p_drop=p_drop)
            for _ in range(num_blocks)
        ])
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class _KANLayer1D(nn.Module):
    """R17 H — minimal Kolmogorov-Arnold layer for 1-D channel projection.

    Each input channel is passed through a learnable B-spline-like univariate
    function (RBF basis), then summed across input channels to produce each
    output channel. A small linear residual `W·x + b` is added (Liu et al. 2024
    standard recipe).

    Implementation notes:
    * Grid centres are cached as a non-learnable buffer (avoids
      ``torch.linspace`` allocation per forward).
    * The basis × spline_weight contraction is done by reshaping the basis to
      ``(N, in_channels * n_basis)`` and the spline weight to
      ``(in_channels * n_basis, out_channels)``, which uses a single fused
      matmul instead of an einsum.
    """

    def __init__(self, in_channels: int, out_channels: int, num_grid: int = 4,
                 spline_order: int = 3, scale_base: float = 1.0,
                 scale_spline: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_grid = num_grid
        self.spline_order = spline_order
        n_basis = num_grid + spline_order
        self.n_basis = n_basis
        # Cached RBF centres on [-1, 1].
        centres = torch.linspace(-1.0, 1.0, n_basis)
        self.register_buffer("centres", centres)
        self.sigma = 2.0 / max(1, n_basis - 1)
        # Spline weight stored as a fused 2-D matrix: (in*n_basis, out).
        self.spline_weight = nn.Parameter(
            torch.randn(in_channels * n_basis, out_channels) * scale_spline / math.sqrt(in_channels * n_basis)
        )
        self.base_weight = nn.Parameter(
            torch.randn(in_channels, out_channels) * scale_base / math.sqrt(in_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_channels). Flatten leading dims for matmul.
        prefix = x.shape[:-1]
        flat = x.reshape(-1, self.in_channels)                            # (M, in)
        # Linear residual term.
        out_lin = flat @ self.base_weight                                 # (M, out)
        # Compute basis (M, in, n_basis) via Gaussian RBF.
        diff = flat.unsqueeze(-1) - self.centres                          # (M, in, n_basis)
        basis = torch.exp(-(diff / self.sigma) ** 2)
        # Reshape and fused matmul.
        basis_flat = basis.reshape(-1, self.in_channels * self.n_basis)   # (M, in*n_basis)
        out_spline = basis_flat @ self.spline_weight                      # (M, out)
        out = (out_lin + out_spline + self.bias).reshape(*prefix, self.out_channels)
        return out


class _KANTrunk(nn.Module):
    """R17 H — KAN trunk: per-wavelength KAN layers with internal pooling.

    Takes (N, in_channels, L), projects to d_model via a 1×1 conv, then
    alternates KAN-MLP (per-position, channel-wise non-linearity) and a
    depthwise conv with stride-2 pooling. The internal pool keeps compute
    tractable on long-spectrum datasets (L=2151 → ~270 after 3 pools).
    """

    def __init__(self, in_channels: int, d_model: int = 64, num_blocks: int = 2,
                 kernel_size: int = 5, p_drop: float = 0.2,
                 kan_grid: int = 4, kan_order: int = 3,
                 internal_pool: bool = True):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                "kan": _KANLayer1D(d_model, d_model, num_grid=kan_grid, spline_order=kan_order),
                "dw_conv": nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=d_model, bias=False),
                "norm": nn.GroupNorm(1, d_model),
                "drop": nn.Dropout1d(p_drop),
            }))
        self.internal_pool = bool(internal_pool)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if internal_pool else nn.Identity()
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        x = self.proj(x)
        for i, blk in enumerate(self.blocks):
            res = x
            # KAN over channels at each wavelength position.
            x_seq = x.transpose(1, 2)                       # (N, L, C)
            x_seq = blk["kan"](x_seq)
            x = x_seq.transpose(1, 2)                       # (N, C, L)
            # Depthwise conv for wavelength-axis context.
            x = blk["dw_conv"](x)
            x = blk["norm"](x)
            x = blk["drop"](x)
            x = x + res
            if self.internal_pool and i < len(self.blocks) - 1 and x.shape[-1] >= 4:
                x = self.pool(x)
        return x


class _DilatedResConvBlock(nn.Module):
    """V2D variant: 3 parallel dilated convolutions (dilations 1, 2, 4) summed.

    Multi-scale receptive fields without explicit pooling — useful for spectra
    where the relevant features span a range of wavelength scales (e.g. DIESEL
    vibrational features at multiple frequencies).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilations: tuple[int, ...] = (1, 2, 4),
                 p_drop: float = 0.2, se_reduction: int | None = 4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=((kernel_size - 1) // 2) * d,
                      dilation=d, bias=False)
            for d in dilations
        ])
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(p=p_drop)
        if se_reduction is not None and out_channels >= 4:
            self.se = _SqueezeExcite(out_channels, reduction=se_reduction)
        else:
            self.se = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        # Crop each parallel branch to a common length (the largest dilation gives the longest output).
        outs = [b(x) for b in self.branches]
        min_len = min(o.shape[-1] for o in outs)
        outs = [o[..., :min_len] for o in outs]
        x = sum(outs) / len(outs)
        if residual.shape[-1] > min_len:
            residual = residual[..., :min_len]
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.se(x)
        return x + residual


def _compact_plus_cnn_extras_branches(p: int, **kw) -> list[nn.Module]:
    extra_kw = {k: v for k, v in kw.items() if k != "matrix_trainable"}
    return aom_compact_branches_torch(p, **kw) + cnn_only_extra_branches_torch(p, **extra_kw)


class NiconV2A(nn.Module):
    """AOM-superblock CNN with frozen operator branches + SE-block trunk."""

    BANK_FACTORIES = {
        "compact": aom_compact_branches_torch,                       # 9 strict-linear ops
        "extended": aom_extended_strict_linear_branches_torch,       # 11 strict-linear ops
        "compact_plus_cnn_extras": _compact_plus_cnn_extras_branches,
        "full": full_branches_torch,                                 # 14 ops (all)
        "extended_lowrank": _extended_lowrank_factory,               # V2H — low-rank Detrend/Whittaker
    }

    def __init__(
        self,
        input_shape: tuple[int, int],
        bank: str = "extended",
        trainable_ops: bool = False,
        matrix_trainable_ops: bool | None = None,
        operator_reg_lambda: float = 0.0,
        trunk_channels: tuple[int, ...] = (32, 64, 96),
        trunk_kernels: tuple[int, ...] = (7, 5, 3),
        spatial_dropout: float = 0.2,
        head_dropout: float = 0.3,
        se_reduction: int | None = 4,
        block_type: str = "res",                         # "res" | "dilated" — Codex F7
        dilations: tuple[int, ...] = (1, 2, 4),          # used when block_type='dilated'
        branch_se: bool = False,                         # Codex F2 — input-level SE over branches
        lowrank_rank: int = 16,                          # rank for `extended_lowrank` bank (V2H)
        learnable_rms: bool = False,                     # V2L — RMSBranchNorm scale learnable
        rms_init_mode: str = "inverse_rms",              # V2L diagnostic — "inverse_rms" | "unit"
        multi_kernel_stem: bool = False,                 # V2O — first block = parallel-kernel
        multi_kernel_kernels: tuple[int, ...] = (3, 5, 7, 9),
        head_type: str = "gap_linear",                   # "gap_linear" | "attn" (V2P) | "moe" (R17 D)
        attn_heads: int = 4,
        moe_num_experts: int = 2,                        # R17 D — only used when head_type=='moe'
        aux_head_dim: int | None = None,                 # R17 F — aux output dim (None = no aux head)
        boost_signal_dim: int = 0,                       # R17 C-boost — extra signal concat to head input
        tied_global_rms: bool = False,                   # V2L diagnostic — single shared scale across branches
        trunk_type: str = "conv",                        # "conv" | "hybrid_transformer" (V3)
        transformer_d_model: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff: int = 128,
        head_out_dim: int = 1,                           # R17 B — set 2 for heteroscedastic (μ, log σ²)
    ) -> None:
        super().__init__()
        in_ch, seq_len = input_shape
        if in_ch != 1:
            raise ValueError(f"NiconV2A expects 1-channel input; got {in_ch}")
        if bank not in self.BANK_FACTORIES:
            raise ValueError(f"unknown bank {bank!r}; expected one of {list(self.BANK_FACTORIES)}")
        if len(trunk_channels) != len(trunk_kernels):
            raise ValueError("trunk_channels and trunk_kernels must have same length")

        # Operator branches. Round 7 V2G can keep O(p^2) matrix operators
        # frozen while allowing compact convolutional kernels to adapt;
        # round 7 V2H uses low-rank trainable matrix ops via `lowrank_rank`.
        factory = self.BANK_FACTORIES[bank]
        factory_kwargs = dict(p=seq_len, trainable=trainable_ops,
                              reg_lambda=operator_reg_lambda)
        # `matrix_trainable` is supported by AOM strict-linear / compact / full
        # factories; `rank` is supported by the `extended_lowrank` factory only.
        if bank in ("compact", "extended", "compact_plus_cnn_extras", "full"):
            factory_kwargs["matrix_trainable"] = matrix_trainable_ops
        if bank == "extended_lowrank":
            factory_kwargs["rank"] = lowrank_rank
        self.branches = nn.ModuleList(factory(**factory_kwargs))
        self.n_branches = len(self.branches)
        self.bank_name = bank

        # Per-branch RMS normalisation (one per branch — fitted on train).
        if tied_global_rms:
            shared = RMSBranchNorm(learnable=learnable_rms, init_mode=rms_init_mode)
            self.branch_norms = nn.ModuleList([shared for _ in range(self.n_branches)])
        else:
            self.branch_norms = nn.ModuleList([
                RMSBranchNorm(learnable=learnable_rms, init_mode=rms_init_mode)
                for _ in range(self.n_branches)
            ])
        self.tied_global_rms = bool(tied_global_rms)
        self.rms_init_mode = rms_init_mode

        # Codex F2 — optional input-level SE over the n_branches channels (V2C).
        if branch_se:
            self.branch_se: nn.Module = _SqueezeExcite(self.n_branches, reduction=max(1, se_reduction or 4))
        else:
            self.branch_se = nn.Identity()

        # Trunk: residual / dilated / multi-kernel-stem conv blocks (V2*) OR
        # hybrid (1 conv block + transformer encoder over wavelengths) (V3).
        blocks: list[nn.Module] = []
        prev = self.n_branches
        if trunk_type == "hybrid_transformer":
            # V3 — one CNN block to reduce L, then a transformer trunk.
            ch0, k0 = trunk_channels[0], trunk_kernels[0]
            if multi_kernel_stem:
                blocks.append(_MultiKernelStemBlock(prev, ch0, kernels=multi_kernel_kernels,
                                                    p_drop=spatial_dropout, se_reduction=se_reduction))
            else:
                blocks.append(_ResConvBlock(prev, ch0, kernel_size=k0, p_drop=spatial_dropout,
                                            se_reduction=se_reduction))
            blocks.append(_AOMTransformerTrunk(in_channels=ch0,
                                               d_model=transformer_d_model,
                                               num_heads=transformer_heads,
                                               num_layers=transformer_layers,
                                               dim_ff=transformer_ff,
                                               p_drop=spatial_dropout))
            prev = transformer_d_model
        elif trunk_type == "conformer":
            # R17 I — Conformer trunk: 1 conv reduction block + N Conformer blocks.
            ch0, k0 = trunk_channels[0], trunk_kernels[0]
            blocks.append(_ResConvBlock(prev, ch0, kernel_size=k0, p_drop=spatial_dropout,
                                        se_reduction=se_reduction))
            blocks.append(_ConformerTrunk(in_channels=ch0,
                                          d_model=transformer_d_model,
                                          num_blocks=transformer_layers,
                                          num_heads=transformer_heads,
                                          kernel_size=k0,
                                          p_drop=spatial_dropout))
            prev = transformer_d_model
        elif trunk_type == "kan":
            # R17 H — KAN trunk: 1 conv reduction block + N KAN blocks.
            ch0, k0 = trunk_channels[0], trunk_kernels[0]
            blocks.append(_ResConvBlock(prev, ch0, kernel_size=k0, p_drop=spatial_dropout,
                                        se_reduction=se_reduction))
            blocks.append(_KANTrunk(in_channels=ch0,
                                    d_model=transformer_d_model,
                                    num_blocks=transformer_layers,
                                    kernel_size=k0,
                                    p_drop=spatial_dropout))
            prev = transformer_d_model
        else:
            for i, (ch, k) in enumerate(zip(trunk_channels, trunk_kernels)):
                if i == 0 and multi_kernel_stem:
                    blocks.append(_MultiKernelStemBlock(prev, ch, kernels=multi_kernel_kernels,
                                                        p_drop=spatial_dropout, se_reduction=se_reduction))
                elif block_type == "res":
                    blocks.append(_ResConvBlock(prev, ch, kernel_size=k, p_drop=spatial_dropout,
                                                se_reduction=se_reduction))
                elif block_type == "dilated":
                    blocks.append(_DilatedResConvBlock(prev, ch, kernel_size=k,
                                                       dilations=dilations,
                                                       p_drop=spatial_dropout,
                                                       se_reduction=se_reduction))
                else:
                    raise ValueError(f"unknown block_type {block_type!r}; expected res | dilated")
                prev = ch
        self.block_type = block_type
        self.trunk_type = trunk_type
        # Adaptive pooling between blocks (max-pool 2× when seq_len allows).
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Head input channels: `prev` (last trunk output) handles both conv and transformer trunks.
        # R17 C-boost — when boost_signal_dim > 0, the head's Linear input is augmented
        # by `boost_signal_dim` extra features (passed via `forward(..., boost_signal=...)`).
        head_in = prev + max(0, int(boost_signal_dim))
        self.boost_signal_dim = int(boost_signal_dim)
        # Head: GAP+Linear (default) or wavelength-attention (V2P).
        if head_type == "gap_linear":
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=head_dropout)
            self.head: nn.Module = nn.Linear(head_in, head_out_dim)
            self._attn_head = False
        elif head_type == "attn":
            self.gap = nn.Identity()
            self.flatten = nn.Identity()
            self.dropout = nn.Identity()
            self.head = _WavelengthAttentionHead(head_in, num_heads=attn_heads,
                                                  p_drop=head_dropout)
            self._attn_head = True
        elif head_type == "moe":
            # R17 D — Mixture-of-Experts head: GAP + softmax-gated K linear experts.
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.dropout = nn.Identity()
            self.head = _MoEHead(head_in, out_dim=head_out_dim,
                                  num_experts=moe_num_experts, p_drop=head_dropout)
            self._attn_head = False
        else:
            raise ValueError(f"unknown head_type {head_type!r}; expected gap_linear | attn | moe")

        # R17 F — auxiliary multi-task head sharing the GAP feature.
        # When `aux_head_dim` is set, the model exposes ``last_aux_pred`` after
        # ``forward``; the training loop reads it to compute a multi-task MSE.
        self._aux_head_dim = int(aux_head_dim) if aux_head_dim else 0
        if self._aux_head_dim > 0:
            self.aux_head: nn.Module | None = nn.Linear(head_in, self._aux_head_dim)
        else:
            self.aux_head = None
        self.last_aux_pred: torch.Tensor | None = None

        # Pre-fit MSC if present in branches.
        for branch in self.branches:
            if branch.__class__.__name__ == "MSCOperator":
                # MSC is fitted lazily via `fit_branches(X_train)` from outside.
                pass

    def fit_branches(self, X_train: torch.Tensor) -> None:
        """Initialise stateful branches (MSC reference, RMSBranchNorm) from `X_train` (N, 1, L).

        Resets each `RMSBranchNorm.fitted=0` first so the target-domain RMS
        is recomputed even when the model was loaded from a pretrained
        checkpoint (Codex round-11 M2 fix — without this reset, LUCAS-pretrained
        models would inherit LUCAS's RMS scales and never adapt to the target).
        """
        was_training = self.training
        self.train()
        with torch.no_grad():
            # Reset RMSBranchNorm fitted state so target-domain RMS is recomputed.
            for norm in self.branch_norms:
                if hasattr(norm, "fitted"):
                    norm.fitted.fill_(0)
            # Fit MSC operators.
            for branch in self.branches:
                if hasattr(branch, "fit") and callable(branch.fit):
                    if branch.__class__.__name__ == "MSCOperator":
                        branch.fit(X_train)
            # Forward once through every branch + RMSBranchNorm so they fit.
            for branch, norm in zip(self.branches, self.branch_norms):
                out = branch(X_train)
                norm(out)
        if not was_training:
            self.eval()

    def operator_regularisation_loss(self) -> torch.Tensor:
        if not any(getattr(b, "trainable", False) for b in self.branches):
            return next(self.parameters()).new_tensor(0.0)
        total: torch.Tensor | None = None
        for b in self.branches:
            if hasattr(b, "regularisation_loss"):
                val = b.regularisation_loss()
                total = val if total is None else total + val
        if total is None:
            return next(self.parameters()).new_tensor(0.0)
        return total

    def forward(self, x: torch.Tensor, boost_signal: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"NiconV2A expects (N, 1, L); got {tuple(x.shape)}")
        # Apply each branch + per-branch RMS norm; concat along channel.
        feats = []
        for branch, norm in zip(self.branches, self.branch_norms):
            out = branch(x)
            out = norm(out)
            feats.append(out)
        x = torch.cat(feats, dim=1)
        # Codex F2 — optional input-level branch SE (AOM-MKL analogue).
        x = self.branch_se(x)
        # Trunk
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1 and x.shape[-1] >= 2:
                x = self.pool(x)
        # Head — GAP+Linear (default) or wavelength-attention (V2P).
        if self._attn_head:
            self.last_aux_pred = None
            return self.head(x)                          # _WavelengthAttentionHead handles full pipe
        x = self.gap(x)
        x = self.flatten(x)
        x_pooled = x  # GAP feature (used for aux head if present)
        # R17 C-boost — concat boost signal to GAP features.
        if self.boost_signal_dim > 0:
            if boost_signal is None:
                # Fallback to zeros so the model still runs (used during eval if
                # caller forgets to pass the signal — should not happen in
                # production code).
                boost_signal = x_pooled.new_zeros((x_pooled.shape[0], self.boost_signal_dim))
            if boost_signal.dim() == 1:
                boost_signal = boost_signal.unsqueeze(-1)
            if boost_signal.shape[-1] != self.boost_signal_dim:
                raise ValueError(
                    f"boost_signal last-dim {boost_signal.shape[-1]} ≠ boost_signal_dim {self.boost_signal_dim}"
                )
            x_pooled = torch.cat([x_pooled, boost_signal.to(x_pooled.dtype)], dim=-1)
        x = self.dropout(x_pooled)
        # R17 F — auxiliary head shares the GAP features (no dropout for aux).
        if self.aux_head is not None:
            self.last_aux_pred = self.aux_head(x_pooled)
        else:
            self.last_aux_pred = None
        return self.head(x)


def _load_pretrained_compatible(model: NiconV2A, ckpt_path: str | None) -> dict:
    """R15 — load pretrained tensors from ``ckpt_path`` into ``model``.

    Loads with ``strict=False`` and additionally drops keys whose tensor
    shapes don't match. Length-dependent operators (LowRank Detrend /
    Whittaker U/V matrices, MSC mean) at a different sequence length are
    silently skipped, so target-domain models keep their fresh init for
    those parameters and load the pretrained weights for everything else.

    Returns a diagnostic dict with ``loaded_keys``, ``skipped_shape_mismatch``,
    and ``skipped_missing``.
    """
    if not ckpt_path:
        return {"loaded": False}
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained_sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_sd = model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped_shape: list[tuple[str, tuple, tuple]] = []
    skipped_missing: list[str] = []
    for k, v in pretrained_sd.items():
        if k not in model_sd:
            skipped_missing.append(k)
            continue
        if model_sd[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        compatible[k] = v

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": True,
        "ckpt": str(ckpt_path),
        "loaded_keys": len(compatible),
        "skipped_shape_mismatch": skipped_shape,
        "skipped_pretrained_missing_in_model": skipped_missing,
        "model_keys_not_in_ckpt": list(missing),
        "ckpt_keys_unexpected_by_model": list(unexpected),
        "ckpt_meta": {k: v for k, v in (ckpt.items() if isinstance(ckpt, dict) else [])
                      if k != "state_dict" and not isinstance(v, torch.Tensor)},
    }


def build_nicon_v2a(input_shape: tuple[int, int], params: dict | None = None) -> NiconV2A:
    p = dict(params or {})
    model = NiconV2A(
        input_shape=input_shape,
        bank=str(p.get("bank", "extended")),
        trainable_ops=bool(p.get("trainable_ops", False)),
        matrix_trainable_ops=p.get("matrix_trainable_ops", None),
        operator_reg_lambda=float(p.get("operator_reg_lambda", 0.0)),
        trunk_channels=tuple(p.get("trunk_channels", (32, 64, 96))),
        trunk_kernels=tuple(p.get("trunk_kernels", (7, 5, 3))),
        spatial_dropout=float(p.get("spatial_dropout", 0.2)),
        head_dropout=float(p.get("head_dropout", 0.3)),
        se_reduction=p.get("se_reduction", 4),
        block_type=str(p.get("block_type", "res")),
        dilations=tuple(p.get("dilations", (1, 2, 4))),
        branch_se=bool(p.get("branch_se", False)),
        lowrank_rank=int(p.get("lowrank_rank", 16)),
        learnable_rms=bool(p.get("learnable_rms", False)),
        rms_init_mode=str(p.get("rms_init_mode", "inverse_rms")),
        tied_global_rms=bool(p.get("tied_global_rms", False)),
        multi_kernel_stem=bool(p.get("multi_kernel_stem", False)),
        multi_kernel_kernels=tuple(p.get("multi_kernel_kernels", (3, 5, 7, 9))),
        head_type=str(p.get("head_type", "gap_linear")),
        attn_heads=int(p.get("attn_heads", 4)),
        moe_num_experts=int(p.get("moe_num_experts", 2)),
        aux_head_dim=p.get("aux_head_dim", None),
        boost_signal_dim=int(p.get("boost_signal_dim", 0)),
        trunk_type=str(p.get("trunk_type", "conv")),
        transformer_d_model=int(p.get("transformer_d_model", 64)),
        transformer_heads=int(p.get("transformer_heads", 4)),
        transformer_layers=int(p.get("transformer_layers", 2)),
        transformer_ff=int(p.get("transformer_ff", 128)),
        head_out_dim=int(p.get("head_out_dim", 1)),
    )
    pretrained_path = p.get("pretrained_path")
    if pretrained_path:
        diag = _load_pretrained_compatible(model, pretrained_path)
        # Stash for runner-side logging.
        model._pretrained_load_diag = diag  # type: ignore[attr-defined]
    return model

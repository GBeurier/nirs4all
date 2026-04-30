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
    aom_extended_strict_linear_branches_torch,
    cnn_only_extra_branches_torch,
    full_branches_torch,
)


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


class NiconV2A(nn.Module):
    """AOM-superblock CNN with frozen operator branches + SE-block trunk."""

    BANK_FACTORIES = {
        "compact": aom_compact_branches_torch,                       # 9 strict-linear ops
        "extended": aom_extended_strict_linear_branches_torch,       # 11 strict-linear ops
        "compact_plus_cnn_extras": lambda p, **kw: aom_compact_branches_torch(p, **kw)
                                                  + cnn_only_extra_branches_torch(p, **kw),
        "full": full_branches_torch,                                 # 14 ops (all)
    }

    def __init__(
        self,
        input_shape: tuple[int, int],
        bank: str = "extended",
        trainable_ops: bool = False,
        operator_reg_lambda: float = 0.0,
        trunk_channels: tuple[int, ...] = (32, 64, 96),
        trunk_kernels: tuple[int, ...] = (7, 5, 3),
        spatial_dropout: float = 0.2,
        head_dropout: float = 0.3,
        se_reduction: int | None = 4,
    ) -> None:
        super().__init__()
        in_ch, seq_len = input_shape
        if in_ch != 1:
            raise ValueError(f"NiconV2A expects 1-channel input; got {in_ch}")
        if bank not in self.BANK_FACTORIES:
            raise ValueError(f"unknown bank {bank!r}; expected one of {list(self.BANK_FACTORIES)}")
        if len(trunk_channels) != len(trunk_kernels):
            raise ValueError("trunk_channels and trunk_kernels must have same length")

        # Operator branches.
        factory = self.BANK_FACTORIES[bank]
        self.branches = nn.ModuleList(factory(p=seq_len,
                                              trainable=trainable_ops,
                                              reg_lambda=operator_reg_lambda))
        self.n_branches = len(self.branches)
        self.bank_name = bank

        # Per-branch RMS normalisation (one per branch — fitted on train).
        self.branch_norms = nn.ModuleList([RMSBranchNorm() for _ in range(self.n_branches)])

        # Trunk: residual conv blocks with SE.
        blocks = []
        prev = self.n_branches
        for ch, k in zip(trunk_channels, trunk_kernels):
            blocks.append(_ResConvBlock(prev, ch, kernel_size=k, p_drop=spatial_dropout,
                                        se_reduction=se_reduction))
            prev = ch
        # Adaptive pooling between blocks (max-pool 2× when seq_len allows).
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=head_dropout)
        self.head = nn.Linear(trunk_channels[-1], 1)

        # Pre-fit MSC if present in branches.
        for branch in self.branches:
            if branch.__class__.__name__ == "MSCOperator":
                # MSC is fitted lazily via `fit_branches(X_train)` from outside.
                pass

    def fit_branches(self, X_train: torch.Tensor) -> None:
        """Initialise stateful branches (MSC reference, RMSBranchNorm) from `X_train` (N, 1, L)."""
        was_training = self.training
        self.train()
        with torch.no_grad():
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
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for b in self.branches:
            if hasattr(b, "regularisation_loss"):
                total = total + b.regularisation_loss()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"NiconV2A expects (N, 1, L); got {tuple(x.shape)}")
        # Apply each branch + per-branch RMS norm; concat along channel.
        feats = []
        for branch, norm in zip(self.branches, self.branch_norms):
            out = branch(x)
            out = norm(out)
            feats.append(out)
        x = torch.cat(feats, dim=1)
        # Trunk
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1 and x.shape[-1] >= 2:
                x = self.pool(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.head(x)


def build_nicon_v2a(input_shape: tuple[int, int], params: dict | None = None) -> NiconV2A:
    p = dict(params or {})
    return NiconV2A(
        input_shape=input_shape,
        bank=str(p.get("bank", "extended")),
        trainable_ops=bool(p.get("trainable_ops", False)),
        operator_reg_lambda=float(p.get("operator_reg_lambda", 0.0)),
        trunk_channels=tuple(p.get("trunk_channels", (32, 64, 96))),
        trunk_kernels=tuple(p.get("trunk_kernels", (7, 5, 3))),
        spatial_dropout=float(p.get("spatial_dropout", 0.2)),
        head_dropout=float(p.get("head_dropout", 0.3)),
        se_reduction=p.get("se_reduction", 4),
    )

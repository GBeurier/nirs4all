"""NiconV1a — Phase 1a minimal repair of NICON.

This module addresses W1 (sigmoid output saturating regression) and W2 (mixed
SELU/ReLU/ELU activations breaking SELU's self-normalising invariant) without
touching the backbone receptive field, kernel sizes, or downsampling strides.

Three sibling classes for clean attribution (Codex round 2 finding #4):

* :class:`NiconV1aHeadOnly`  — H1 alone. Upstream NICON's activations / norms / dropouts
  preserved; only the final ``Dense(1, sigmoid)`` is replaced with ``Linear(1)``.
* :class:`NiconV1aActivationOnly` — H2 alone. Replaces SELU/ReLU/ELU + BN + AlphaDropout
  with GELU + LayerNorm + Dropout consistently, but keeps the upstream sigmoid output.
* :class:`NiconV1a`           — H1 + H2 combined.

Backbone receptive field, kernel sizes, strides, and channel counts match
``nirs4all.operators.models.pytorch.nicon._build_nicon`` exactly. The block ordering
is `Conv → Norm → Activation → Dropout`, which matches the upstream ordering
`Conv → Activation → BatchNorm` topologically (norm is between conv and the
nonlinearity in V1a, vs after activation upstream). We document the difference
explicitly in the geometry parity test.

The H2 channel-input dropout is `nn.Dropout1d(0.08)` matching the upstream
SpatialDropout1D rate; with 1-channel input this can zero an entire spectrum 8 %
of the time. We expose a flag `input_dropout` to control it (default keeps the
NICON value for parity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _LayerNormChan(nn.Module):
    """LayerNorm over the channel dim for an `(N, C, L)` tensor (== GroupNorm with 1 group)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _conv_out_len(in_len: int, kernel_size: int, stride: int) -> int:
    """Match Keras 'valid' padding (which TF NICON uses): floor((L − k) / s + 1)."""
    return (in_len - kernel_size) // stride + 1


class NiconV1a(nn.Module):
    """1-D CNN with NICON's receptive field but a single-activation discipline + linear head.

    Parameters
    ----------
    input_shape:
        ``(channels, sequence_length)``. Channels is normally 1 for raw spectra.
    spatial_dropout:
        Channel-wise dropout probability applied at the input (default 0.08, same as NICON).
    dropout_rate:
        Standard dropout probability after each conv block (default 0.2).
    dense_units:
        Hidden width of the regression head (default 16, same as NICON).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        spatial_dropout: float = 0.08,
        dropout_rate: float = 0.2,
        dense_units: int = 16,
    ) -> None:
        super().__init__()
        in_ch, seq_len = input_shape

        # Block 1: small filters / large kernel + stride (NICON default).
        kernel1, stride1, ch1 = 15, 5, 8
        kernel2, stride2, ch2 = 21, 3, 64
        kernel3, stride3, ch3 = 5, 3, 32

        if seq_len < kernel1 or _conv_out_len(seq_len, kernel1, stride1) < kernel2:
            raise ValueError(
                f"NiconV1a needs sequence_length ≥ {kernel1 + kernel2 * stride1} for the chosen kernels; got {seq_len}."
            )

        # Channel-wise dropout (drops entire channels — preserves the spectrum's local
        # correlation structure). Implemented via standard `nn.Dropout1d` (PyTorch ≥ 2.0).
        self.input_dropout = nn.Dropout1d(p=spatial_dropout)

        self.conv1 = nn.Conv1d(in_ch, ch1, kernel_size=kernel1, stride=stride1)
        self.norm1 = _LayerNormChan(ch1)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size=kernel2, stride=stride2)
        self.norm2 = _LayerNormChan(ch2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.conv3 = nn.Conv1d(ch2, ch3, kernel_size=kernel3, stride=stride3)
        self.norm3 = _LayerNormChan(ch3)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(p=dropout_rate)

        seq_len_after = _conv_out_len(seq_len, kernel1, stride1)
        seq_len_after = _conv_out_len(seq_len_after, kernel2, stride2)
        seq_len_after = _conv_out_len(seq_len_after, kernel3, stride3)
        if seq_len_after < 1:
            raise ValueError(
                f"NiconV1a backbone collapses sequence_length={seq_len} to {seq_len_after} ≤ 0 after 3 strided convs."
            )
        flat_dim = ch3 * seq_len_after

        self.flatten = nn.Flatten()
        self.head_dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(flat_dim, dense_units),
            nn.GELU(),
            nn.Linear(dense_units, 1),  # **linear** output for regression (W1 fix)
        )

        self._effective_seq_len = seq_len_after
        self._flat_dim = flat_dim

    @property
    def effective_seq_len(self) -> int:
        return self._effective_seq_len

    @property
    def flat_dim(self) -> int:
        return self._flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.drop1(self.act1(self.norm1(self.conv1(x))))
        x = self.drop2(self.act2(self.norm2(self.conv2(x))))
        x = self.drop3(self.act3(self.norm3(self.conv3(x))))
        x = self.flatten(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x


def build_nicon_v1a(input_shape: tuple[int, int], params: dict | None = None) -> NiconV1a:
    """Builder used by the benchmark runner."""
    p = dict(params or {})
    return NiconV1a(
        input_shape=input_shape,
        spatial_dropout=float(p.get("spatial_dropout", 0.08)),
        dropout_rate=float(p.get("dropout_rate", 0.2)),
        dense_units=int(p.get("dense_units", 16)),
    )


# ---------------------------------------------------------------------------
# H1-only ablation: upstream NICON backbone with linear output.
# ---------------------------------------------------------------------------


class NiconV1aHeadOnly(nn.Module):
    """Upstream NICON architecture (SELU/ReLU/ELU + AlphaDropout + BatchNorm + SpatialDropout1D),
    only swapping the final ``Dense(1, sigmoid)`` for a linear projection.

    The hidden ``Dense(dense_units, sigmoid)`` is **kept** — H1 in isolation only changes
    the output activation. This is the cleanest single-change variant.
    """

    def __init__(self, input_shape: tuple[int, int]):
        super().__init__()
        from nirs4all.operators.models.pytorch.nicon import _build_nicon

        backbone = _build_nicon(input_shape, {}, num_classes=1)
        # Upstream `_build_nicon` returns an `nn.Sequential` whose last module is
        # `Dense(1, sigmoid)`. We replace it with `nn.Linear(in_features, 1)`.
        if not isinstance(backbone, nn.Sequential):
            raise RuntimeError("expected upstream nicon to be nn.Sequential")
        # find the last Linear layer; it should be `dense_units → 1` then a Sigmoid
        last_linear_idx = None
        for i, m in enumerate(backbone):
            if isinstance(m, nn.Linear):
                last_linear_idx = i
        if last_linear_idx is None:
            raise RuntimeError("could not find final Linear in upstream nicon")
        in_features = backbone[last_linear_idx].in_features
        backbone[last_linear_idx] = nn.Linear(in_features, 1)
        # Drop any `nn.Sigmoid` modules that follow the final linear.
        keep_idx = [i for i, m in enumerate(backbone) if not (i > last_linear_idx and isinstance(m, nn.Sigmoid))]
        self.body = nn.Sequential(*[backbone[i] for i in keep_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def build_nicon_v1a_head_only(input_shape: tuple[int, int], params: dict | None = None) -> NiconV1aHeadOnly:
    return NiconV1aHeadOnly(input_shape=input_shape)


# ---------------------------------------------------------------------------
# H2-only ablation: GELU + LayerNorm pipeline but with the original sigmoid output.
# ---------------------------------------------------------------------------


class NiconV1aActivationOnly(nn.Module):
    """V1a backbone (GELU + LayerNorm + Dropout) with the upstream **sigmoid** output."""

    def __init__(self, input_shape: tuple[int, int]):
        super().__init__()
        self._inner = NiconV1a(input_shape=input_shape)
        # Replace the inner head's last Linear with Linear → Sigmoid.
        head = self._inner.head
        if not isinstance(head, nn.Sequential):
            raise RuntimeError("V1a head should be nn.Sequential")
        last_linear: nn.Linear | None = None
        for m in reversed(head):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("V1a head has no Linear")
        # Append Sigmoid; copy parameters by reference (no re-init of the linear).
        new_head_layers = list(head) + [nn.Sigmoid()]
        self._inner.head = nn.Sequential(*new_head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner(x)


def build_nicon_v1a_activation_only(input_shape: tuple[int, int], params: dict | None = None) -> NiconV1aActivationOnly:
    return NiconV1aActivationOnly(input_shape=input_shape)

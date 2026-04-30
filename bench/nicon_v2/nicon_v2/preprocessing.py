"""Phase 1b deterministic preprocessing.

The differentiable Savitzky-Golay (SG) derivative is implemented as a
``Conv1d(in=1, out=1, kernel_size=w, bias=False)`` whose weight is the SG
kernel, frozen (``requires_grad=False``). Convolution mode is `"same"`
(reflect-padded), matching ``scipy.signal.savgol_filter`` mode `"interp"` to
within float precision on the interior.

Concat-derivative input front: stacks ``[raw, 1st-SG, 2nd-SG]`` along the
channel dimension to produce a 3-channel tensor (Mishra & Passos 2022).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_coeffs


def savgol_kernel(window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """Return the Savitzky-Golay coefficient vector for a single derivative order.

    Wraps ``scipy.signal.savgol_coeffs`` so we share scipy's exact algorithm.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")
    if deriv < 0 or deriv > polyorder:
        raise ValueError("deriv must satisfy 0 <= deriv <= polyorder")
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    return np.asarray(coeffs, dtype=np.float64)


class FixedSavGol1D(nn.Module):
    """Frozen 1-D SG-derivative convolution.

    The kernel is computed once and stored as a non-trainable buffer.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0, delta: float = 1.0):
        super().__init__()
        kernel = savgol_kernel(window_length, polyorder, deriv=deriv, delta=delta)
        # scipy's savgol_filter applies the coefficients via *convolution* (kernel reversed
        # in the sliding window sense); torch.F.conv1d does *cross-correlation* (kernel
        # not reversed). For symmetric kernels (even-order derivatives) the two coincide;
        # for odd-order derivatives the kernel is antisymmetric so we must flip the
        # kernel here to match scipy's output sign.
        kernel = np.ascontiguousarray(kernel[::-1].copy())
        # nn.Conv1d expects shape (out_channels=1, in_channels=1, kernel_size=w)
        self.register_buffer("kernel", torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1))
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"FixedSavGol1D expects (N, C, L); got {tuple(x.shape)}")
        n, c, length = x.shape
        # Apply per-channel: reshape to (N*C, 1, L) → conv1d → reshape back.
        x_flat = x.reshape(n * c, 1, length)
        pad = self.window_length // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
        out = F.conv1d(x_pad, self.kernel)
        return out.reshape(n, c, length)


class SNVLayer(nn.Module):
    """Standard Normal Variate normalisation (per-spectrum)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SNVLayer expects (N, C, L); got {tuple(x.shape)}")
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp(min=1e-12)
        return (x - mean) / std


class MSCLayer(nn.Module):
    """Multiplicative Scatter Correction.

    The reference spectrum is fitted on the train set (``fit(X_train)``) and
    stored as a non-trainable buffer. At inference, each row is regressed
    against the reference and the per-row offset/slope is removed.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("reference", torch.zeros(num_features))
        self._fitted = False

    def fit(self, X: torch.Tensor | np.ndarray) -> "MSCLayer":
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if X.dim() == 3:
            X = X[:, 0, :]
        if X.shape[1] != self.reference.shape[0]:
            raise ValueError(f"MSC reference width mismatch: {X.shape[1]} vs {self.reference.shape[0]}")
        self.reference.copy_(X.mean(dim=0))
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("MSCLayer must be fitted before forward")
        if x.dim() != 3:
            raise ValueError(f"MSCLayer expects (N, C, L); got {tuple(x.shape)}")
        ref = self.reference
        ref_centered = ref - ref.mean()
        # Per-row least-squares against the reference.
        x_centered = x - x.mean(dim=-1, keepdim=True)
        denom = (ref_centered * ref_centered).sum().clamp(min=1e-12)
        b = (x_centered * ref_centered).sum(dim=-1, keepdim=True) / denom
        a = x.mean(dim=-1, keepdim=True) - b * ref.mean()
        b = b.clamp(min=1e-6)
        return (x - a) / b


class ConcatDerivatives(nn.Module):
    """Concatenate `[raw, 1st-SG-derivative, 2nd-SG-derivative]` (and optionally [SNV])
    along the channel dim.

    Input shape: ``(N, 1, L)``. Output shape: ``(N, 3, L)`` (default) or ``(N, 4, L)``
    when ``include_snv=True``.
    """

    def __init__(
        self,
        window_length_d1: int = 11,
        window_length_d2: int = 11,
        polyorder: int = 2,
        include_snv: bool = False,
    ):
        super().__init__()
        self.deriv0 = nn.Identity()
        self.deriv1 = FixedSavGol1D(window_length_d1, polyorder, deriv=1)
        self.deriv2 = FixedSavGol1D(window_length_d2, polyorder, deriv=2)
        self.include_snv = include_snv
        if include_snv:
            self.snv = SNVLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"ConcatDerivatives expects (N, C, L); got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"ConcatDerivatives expects 1 input channel; got {x.shape[1]}")
        d0 = self.deriv0(x)
        d1 = self.deriv1(x)
        d2 = self.deriv2(x)
        outs = [d0, d1, d2]
        if self.include_snv:
            outs.append(self.snv(x))
        return torch.cat(outs, dim=1)


def n_input_channels_after_preproc(use_concat_derivatives: bool, base_channels: int = 1) -> int:
    return 3 * base_channels if use_concat_derivatives else base_channels

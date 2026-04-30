"""Phase 1b training-time augmentation.

Three operators, all Bjerrum-style scale-aware (Bjerrum, Glahder & Skov 2017):

* offset:        ``x' = x + u`` with ``u ~ U[-σ_u, σ_u]``
* slope:         ``x' = x + s · w(λ)`` with ``s ~ U[-σ_s, σ_s]``, ``w(λ)`` linear
* multiplicative: ``x' = m · x`` with ``m ~ U[1 − σ_m, 1 + σ_m]``

The amplitudes are scaled to per-dataset spectrum statistics: σ_u, σ_s default
to ``0.05 · range(x_train)`` and σ_m defaults to 0.05.

Plus:

* contiguous-band masking (Codex round 1 finding F6),
* mixup (uniform partner),
* C-Mixup (Yao 2022) with a fold-locally-tunable bandwidth.

All operators are deterministic given a torch generator. The augmenter only
runs in `training` mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BjerrumConfig:
    sigma_offset: float = 0.05    # × range(x_train)
    sigma_slope: float = 0.05     # × range(x_train) (slope on normalised wavelength axis)
    sigma_mult: float = 0.05      # additive perturbation around 1.0
    band_mask_prob: float = 0.0   # probability per spectrum to apply a band mask
    band_mask_width_frac: float = 0.05  # fractional width of the band mask
    enabled: bool = True


@dataclass
class CMixupConfig:
    enabled: bool = False
    alpha: float = 0.2           # Beta(α, α) parameter
    sigma_y: float | None = None  # bandwidth (auto if None)


def _per_dataset_amplitude(X_train: np.ndarray) -> float:
    """Spectrum range = max(X) − min(X) (computed once on train)."""
    return float(np.max(X_train) - np.min(X_train))


class BjerrumAugmenter:
    """Stateless apply-on-batch augmenter; uses a torch.Generator for reproducibility."""

    def __init__(self, config: BjerrumConfig, sigma_unit: float, seq_len: int, device: torch.device):
        self.cfg = config
        self.sigma_u = config.sigma_offset * sigma_unit
        self.sigma_s = config.sigma_slope * sigma_unit
        self.sigma_m = config.sigma_mult
        self.seq_len = seq_len
        self.device = device
        # Linear wavelength axis on [-1, 1].
        self._w = torch.linspace(-1.0, 1.0, seq_len, device=device)

    def __call__(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.cfg.enabled:
            return x
        if x.dim() != 3:
            raise ValueError(f"BjerrumAugmenter expects (N, C, L); got {tuple(x.shape)}")
        n, c, length = x.shape
        # Offset (N, 1, 1).
        u = (torch.rand(n, 1, 1, device=x.device, generator=generator) * 2.0 - 1.0) * self.sigma_u
        # Slope (N, 1, 1) → broadcast over wavelength.
        s = (torch.rand(n, 1, 1, device=x.device, generator=generator) * 2.0 - 1.0) * self.sigma_s
        slope_term = s * self._w.view(1, 1, -1)
        # Multiplicative scale (N, 1, 1).
        m = 1.0 + (torch.rand(n, 1, 1, device=x.device, generator=generator) * 2.0 - 1.0) * self.sigma_m
        x_aug = m * x + u + slope_term

        if self.cfg.band_mask_prob > 0.0:
            mask_draws = torch.rand(n, 1, 1, device=x.device, generator=generator)
            mask_apply = mask_draws < self.cfg.band_mask_prob
            if bool(mask_apply.any()):
                width = max(1, int(round(self.cfg.band_mask_width_frac * length)))
                start = torch.randint(0, max(1, length - width), (n,), device=x.device, generator=generator)
                idx = torch.arange(length, device=x.device).view(1, 1, -1)
                bandmask = (
                    (idx >= start.view(-1, 1, 1))
                    & (idx < (start.view(-1, 1, 1) + width))
                ).float()
                # Replace masked region by per-spectrum mean.
                spec_mean = x_aug.mean(dim=-1, keepdim=True)
                x_aug = torch.where(
                    (bandmask.bool() & mask_apply.expand_as(bandmask).bool()),
                    spec_mean.expand_as(x_aug),
                    x_aug,
                )
        return x_aug


def _cmixup_pair_indices(
    y: torch.Tensor,
    sigma_y: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """For each i, sample j from p(j|i) ∝ exp(-((y_i − y_j)² / 2σ²))."""
    n = y.shape[0]
    diff_sq = (y.view(-1, 1) - y.view(1, -1)) ** 2
    log_w = -diff_sq / max(2.0 * sigma_y * sigma_y, 1e-12)
    # Uniform tie-breaking + numeric stability: subtract per-row max.
    log_w = log_w - log_w.max(dim=-1, keepdim=True).values
    w = torch.exp(log_w)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    j = torch.multinomial(w, num_samples=1, replacement=True, generator=generator).squeeze(-1)
    return j


class CMixupAugmenter:
    """Mixup variant where the partner is sampled with a Gaussian kernel on |y_i − y_j|.

    Setting `cfg.sigma_y = float('inf')` (or a very large value) recovers vanilla mixup.
    """

    def __init__(self, config: CMixupConfig):
        self.cfg = config

    def select_sigma_y(self, y_train: torch.Tensor) -> float:
        if self.cfg.sigma_y is not None:
            return float(self.cfg.sigma_y)
        return float(0.5 * y_train.std().item() + 1e-12)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma_y: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enabled:
            return x, y
        n = x.shape[0]
        if n < 2:
            return x, y
        j = _cmixup_pair_indices(y, sigma_y, generator)
        beta = torch.distributions.Beta(self.cfg.alpha, self.cfg.alpha)
        lam = beta.sample().to(x.device).clamp(0.05, 0.95)
        x_mix = lam * x + (1.0 - lam) * x[j]
        y_mix = lam * y + (1.0 - lam) * y[j]
        return x_mix, y_mix


@dataclass
class AugmentationPlan:
    bjerrum: BjerrumConfig
    cmixup: CMixupConfig

    def build(self, X_train: np.ndarray, y_train: np.ndarray, seq_len: int, device: torch.device):
        sigma_unit = _per_dataset_amplitude(X_train)
        bjer = BjerrumAugmenter(self.bjerrum, sigma_unit=sigma_unit, seq_len=seq_len, device=device)
        cmix = CMixupAugmenter(self.cmixup)
        sigma_y = cmix.select_sigma_y(torch.from_numpy(y_train).float()) if self.cmixup.enabled else 0.0
        return bjer, cmix, sigma_y

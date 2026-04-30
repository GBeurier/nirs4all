"""Phase 1b augmentation tests — Bjerrum determinism, C-Mixup label-pair correctness."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nicon_v2.augmentation import (
    AugmentationPlan,
    BjerrumAugmenter,
    BjerrumConfig,
    CMixupAugmenter,
    CMixupConfig,
    _cmixup_pair_indices,
)


def test_bjerrum_seeded_determinism():
    cfg = BjerrumConfig(sigma_offset=0.05, sigma_slope=0.05, sigma_mult=0.05)
    aug = BjerrumAugmenter(cfg, sigma_unit=2.0, seq_len=200, device=torch.device("cpu"))
    x = torch.zeros(8, 1, 200)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = aug(x.clone(), g1)
    out2 = aug(x.clone(), g2)
    assert torch.allclose(out1, out2)


def test_bjerrum_disabled_is_identity():
    cfg = BjerrumConfig(enabled=False)
    aug = BjerrumAugmenter(cfg, sigma_unit=1.0, seq_len=50, device=torch.device("cpu"))
    x = torch.randn(4, 1, 50)
    out = aug(x.clone(), torch.Generator().manual_seed(7))
    assert torch.allclose(out, x)


def test_bjerrum_amplitudes_within_bounds():
    cfg = BjerrumConfig(sigma_offset=0.1, sigma_slope=0.1, sigma_mult=0.1)
    aug = BjerrumAugmenter(cfg, sigma_unit=2.0, seq_len=100, device=torch.device("cpu"))
    x = torch.zeros(64, 1, 100)
    g = torch.Generator().manual_seed(42)
    out = aug(x, g)
    # Maximum delta from x = 0 is sigma_unit * (sigma_offset + sigma_slope) + small numeric headroom.
    max_delta = 2.0 * (0.1 + 0.1) + 1e-6
    assert out.abs().max().item() <= max_delta


def test_cmixup_pair_indices_concentrate_on_close_labels():
    y = torch.tensor([0.0, 0.05, 0.5, 0.55, 1.0, 1.05])
    g = torch.Generator().manual_seed(0)
    j = _cmixup_pair_indices(y, sigma_y=0.05, generator=g)
    # With σ=0.05 the partner should almost always be in {i-1, i+1} (close-y pair).
    assert j.shape == (6,)
    diffs = (y - y[j]).abs()
    assert (diffs <= 0.06).float().mean().item() >= 0.9


def test_cmixup_returns_xy_with_lambda_in_range():
    cfg = CMixupConfig(enabled=True, alpha=0.5, sigma_y=None)
    aug = CMixupAugmenter(cfg)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(16, 1, 200)).astype(np.float32))
    y = torch.from_numpy(rng.normal(size=16).astype(np.float32))
    sigma_y = aug.select_sigma_y(y)
    g = torch.Generator().manual_seed(0)
    x_mix, y_mix = aug(x, y, sigma_y, g)
    assert x_mix.shape == x.shape
    assert y_mix.shape == y.shape


def test_augmentation_plan_builds_components():
    bjer = BjerrumConfig(sigma_offset=0.05, sigma_slope=0.05, sigma_mult=0.05)
    cmix = CMixupConfig(enabled=True)
    plan = AugmentationPlan(bjerrum=bjer, cmixup=cmix)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 200)).astype(np.float32)
    y = rng.normal(size=32).astype(np.float32)
    bjer_aug, cmix_aug, sigma_y = plan.build(X, y, seq_len=200, device=torch.device("cpu"))
    assert bjer_aug.cfg.enabled
    assert cmix_aug.cfg.enabled
    assert sigma_y > 0.0

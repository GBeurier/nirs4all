"""Phase 1b preprocessing tests — SG kernels match scipy; SNV / MSC / concat-deriv shapes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

torch = pytest.importorskip("torch")

from nicon_v2.preprocessing import (
    ConcatDerivatives,
    FixedSavGol1D,
    MSCLayer,
    SNVLayer,
    savgol_kernel,
)


@pytest.mark.parametrize("window_length,polyorder,deriv", [(11, 2, 0), (11, 2, 1), (11, 2, 2), (15, 3, 1)])
def test_savgol_kernel_matches_scipy(window_length, polyorder, deriv):
    rng = np.random.default_rng(0)
    x = rng.normal(size=400).astype(np.float32)
    layer = FixedSavGol1D(window_length=window_length, polyorder=polyorder, deriv=deriv)
    out = layer(torch.from_numpy(x).view(1, 1, -1)).numpy().ravel()
    ref = savgol_filter(x.astype(np.float64), window_length, polyorder, deriv=deriv, mode="interp")
    pad = window_length // 2
    # Compare interior away from the edges (modes differ at boundaries; reflect vs interp).
    assert np.allclose(out[pad:-pad], ref[pad:-pad], atol=1e-3)


def test_snv_per_spectrum_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(8, 1, 200)).astype(np.float32)
    out = SNVLayer()(torch.from_numpy(x)).numpy()
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_msc_fits_train_only_and_corrects_shifts():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200).astype(np.float32)
    # Build a training set: shifted/scaled copies of `base`.
    n = 32
    a = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    b = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    X_train = (a[:, None] + b[:, None] * base).astype(np.float32)
    msc = MSCLayer(num_features=200).fit(X_train)
    x_train_corr = msc(torch.from_numpy(X_train).unsqueeze(1)).squeeze(1).numpy()
    x_train_corr_centered = x_train_corr - x_train_corr.mean(axis=-1, keepdims=True)
    base_centered = base - base.mean()
    # After MSC, each row should align with the reference up to a small residual.
    cor = np.array([
        np.corrcoef(x_train_corr_centered[i], base_centered)[0, 1] for i in range(n)
    ])
    assert (cor > 0.99).all()


def test_concat_derivatives_shape_3channels():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 1, 700)).astype(np.float32)
    out = ConcatDerivatives()(torch.from_numpy(x))
    assert out.shape == (4, 3, 700)
    # Channel 0 must equal raw input.
    assert torch.allclose(out[:, 0, :], torch.from_numpy(x[:, 0, :]))


def test_savgol_kernel_helper_rejects_invalid():
    with pytest.raises(ValueError):
        savgol_kernel(window_length=10, polyorder=2)  # even length
    with pytest.raises(ValueError):
        savgol_kernel(window_length=11, polyorder=11)
    with pytest.raises(ValueError):
        savgol_kernel(window_length=11, polyorder=2, deriv=3)

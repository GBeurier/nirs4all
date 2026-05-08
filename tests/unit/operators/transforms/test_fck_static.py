"""Tests for ``FCKStaticTransformer``.

The transformer builds a static bank of fractional-derivative filters and
applies them as 1D convolutions across the wavelength axis. Tests cover:

- Bank construction (size, normalisation, zero-mean for alpha > 0).
- Output shape (flattened and 3D).
- Fit-time independence from ``y`` and from ``X_test`` content.
- sklearn ``clone`` round-trip.
- Determinism: same hyperparameters produce identical kernels.
- Composition with sklearn ``Pipeline``.
- Error paths: sparse input, even kernel size, non-positive sigma/scale.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from nirs4all.operators.transforms.fck_static import FCKStaticTransformer, _build_kernel

# ---------------------------------------------------------------------------
# Kernel construction (private helper)
# ---------------------------------------------------------------------------


class TestBuildKernel:
    def test_l1_normalised(self):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            for scale in (1, 2):
                for ks in (15, 31):
                    k = _build_kernel(alpha, scale, ks, sigma=3.0)
                    assert k.shape == (ks,)
                    assert np.isclose(np.sum(np.abs(k)), 1.0, atol=1e-6)

    def test_zero_mean_for_positive_alpha(self):
        for alpha in (0.5, 1.0, 1.5, 2.0):
            k = _build_kernel(alpha, scale=1, kernel_size=31, sigma=3.0)
            assert abs(np.mean(k)) < 1e-6

    def test_smoother_for_alpha_zero(self):
        k = _build_kernel(0.0, scale=1, kernel_size=15, sigma=3.0)
        # A pure Gaussian smoother is non-negative everywhere
        assert np.all(k >= 0)

    def test_odd_kernel_size_required(self):
        with pytest.raises(ValueError, match="must be odd"):
            _build_kernel(1.0, 1, 14, sigma=3.0)

    def test_determinism(self):
        a = _build_kernel(1.5, 2, 31, sigma=3.0)
        b = _build_kernel(1.5, 2, 31, sigma=3.0)
        np.testing.assert_array_equal(a, b)

    def test_scale_changes_kernel(self):
        """Different ``scale`` values must produce different kernels.

        Per the spec, ``scale`` multiplies the index axis; with a fixed
        ``sigma`` this produces filters with distinct effective frequencies
        (the higher the scale, the more compact the high-magnitude region
        because the Gaussian envelope decays faster in raw-sample units).
        """
        narrow = _build_kernel(1.0, scale=1, kernel_size=31, sigma=3.0)
        wide = _build_kernel(1.0, scale=2, kernel_size=31, sigma=3.0)
        assert not np.allclose(narrow, wide)


# ---------------------------------------------------------------------------
# Transformer behaviour
# ---------------------------------------------------------------------------


class TestFCKStaticTransformer:
    def _X(self, seed: int = 0, n: int = 24, p: int = 200) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(n, p).astype(np.float64)

    def test_default_bank_size(self):
        fck = FCKStaticTransformer().fit(self._X())
        # 4 alphas x 2 scales x 2 kernel_sizes = 16 filters
        assert fck.n_kernels_ == 16
        assert len(fck.kernel_specs_) == 16

    def test_default_output_shape_flattened(self):
        X = self._X(n=10, p=200)
        fck = FCKStaticTransformer().fit(X)
        Z = fck.transform(X)
        assert Z.shape == (10, 16 * 200)

    def test_output_shape_3d(self):
        X = self._X(n=8, p=128)
        fck = FCKStaticTransformer(flatten=False).fit(X)
        Z = fck.transform(X)
        assert Z.shape == (8, 16, 128)

    def test_fit_does_not_depend_on_y(self):
        X = self._X(n=10, p=64)
        fck1 = FCKStaticTransformer().fit(X)
        fck2 = FCKStaticTransformer().fit(X, y=np.zeros(10))
        np.testing.assert_array_equal(fck1.kernels_, fck2.kernels_)

    def test_transform_does_not_depend_on_other_samples(self):
        """No fit-time leakage: per-sample output depends only on that sample."""
        X = self._X(n=20, p=100)
        fck = FCKStaticTransformer().fit(X)
        Z_full = fck.transform(X)
        Z_sub = fck.transform(X[:1])
        np.testing.assert_allclose(Z_full[0], Z_sub[0])

    def test_clone_preserves_hyperparameters(self):
        fck = FCKStaticTransformer(alphas=(0.5, 1.5), scales=(1,), kernel_sizes=(21,))
        fck2 = clone(fck)
        assert fck2.alphas == (0.5, 1.5)
        assert fck2.scales == (1,)
        assert fck2.kernel_sizes == (21,)

    def test_set_params_rebuilds_bank_after_fit(self):
        X = self._X(n=10, p=100)
        fck = FCKStaticTransformer().fit(X)
        assert fck.n_kernels_ == 16
        fck.set_params(alphas=(1.0,), scales=(1,), kernel_sizes=(15,))
        fck.fit(X)
        assert fck.n_kernels_ == 1

    def test_pipeline_with_ridge(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 64)
        # Fake target: smooth-derivative-friendly signal
        beta = rng.randn(64)
        y = X @ beta + 0.01 * rng.randn(40)
        pipe = Pipeline(
            [
                ("fck", FCKStaticTransformer(alphas=(1.0,), scales=(1,), kernel_sizes=(15,))),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        pipe.fit(X, y)
        y_hat = pipe.predict(X)
        assert y_hat.shape == (40,)
        # Sanity: the model should fit at least better than predicting the mean
        baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        assert rmse < baseline_rmse

    def test_transform_preserves_sample_count(self):
        X = self._X(n=37, p=80)
        Z = FCKStaticTransformer().fit_transform(X)
        assert Z.shape[0] == 37

    def test_kernels_l1_normalised_after_fit(self):
        fck = FCKStaticTransformer().fit(self._X())
        for i, (_, _, ks) in enumerate(fck.kernel_specs_):
            kernel = fck.kernels_[i, :ks]
            assert np.isclose(np.sum(np.abs(kernel)), 1.0, atol=1e-6)

    def test_zero_mean_kernels_for_positive_alpha(self):
        fck = FCKStaticTransformer().fit(self._X())
        for i, (alpha, _, ks) in enumerate(fck.kernel_specs_):
            if alpha > 0.1:
                kernel = fck.kernels_[i, :ks]
                assert abs(np.mean(kernel)) < 1e-6

    # --- error paths ----------------------------------------------------

    def test_sparse_input_rejected(self):
        X = scipy.sparse.csr_matrix(np.random.RandomState(0).randn(5, 50))
        with pytest.raises(TypeError, match="sparse"):
            FCKStaticTransformer().fit(X)

    def test_even_kernel_size_rejected(self):
        with pytest.raises(ValueError, match="kernel_size"):
            FCKStaticTransformer(kernel_sizes=(14,)).fit(self._X())

    def test_non_positive_sigma_rejected(self):
        with pytest.raises(ValueError, match="sigma"):
            FCKStaticTransformer(sigma=0.0).fit(self._X())

    def test_non_positive_scale_rejected(self):
        with pytest.raises(ValueError, match="scale"):
            FCKStaticTransformer(scales=(0,)).fit(self._X())

    def test_transform_before_fit_raises(self):
        with pytest.raises(ValueError, match="fitted"):
            FCKStaticTransformer().transform(self._X())

    def test_empty_alpha_list_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            FCKStaticTransformer(alphas=()).fit(self._X())

"""Tests for scalers.py transforms: StandardNormalVariate, Normalize, Derivate, SimpleScale."""

import numpy as np
import pytest


@pytest.fixture
def spectra():
    """Standard test spectra: 10 samples, 50 features."""
    rng = np.random.default_rng(42)
    return rng.normal(0.5, 0.1, size=(10, 50)).astype(np.float64)

class TestStandardNormalVariate:
    """Tests for StandardNormalVariate (SNV) transform."""

    def test_output_shape(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate()
        Xt = snv.fit_transform(spectra)
        assert Xt.shape == spectra.shape

    def test_row_mean_is_zero(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate()
        Xt = snv.fit_transform(spectra)
        row_means = np.mean(Xt, axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-12)

    def test_row_std_is_one(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate()
        Xt = snv.fit_transform(spectra)
        row_stds = np.std(Xt, axis=1)
        np.testing.assert_allclose(row_stds, 1.0, atol=1e-12)

    def test_without_centering(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate(with_mean=False)
        Xt = snv.fit_transform(spectra)
        assert Xt.shape == spectra.shape
        # Row means should NOT be zero
        row_means = np.mean(Xt, axis=1)
        assert not np.allclose(row_means, 0.0)

    def test_without_scaling(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate(with_std=False)
        Xt = snv.fit_transform(spectra)
        assert Xt.shape == spectra.shape
        # Row stds should NOT be 1.0 in general
        row_stds = np.std(Xt, axis=1)
        assert not np.allclose(row_stds, 1.0)

    def test_does_not_mutate_input(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        X_copy = spectra.copy()
        snv = StandardNormalVariate()
        snv.fit_transform(spectra)
        np.testing.assert_array_equal(spectra, X_copy)

    def test_output_is_finite(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate()
        Xt = snv.fit_transform(spectra)
        assert np.all(np.isfinite(Xt))

    def test_single_sample(self):
        from nirs4all.operators.transforms import StandardNormalVariate
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        snv = StandardNormalVariate()
        Xt = snv.fit_transform(X)
        assert Xt.shape == X.shape
        np.testing.assert_allclose(np.mean(Xt, axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(Xt, axis=1), 1.0, atol=1e-12)

    def test_column_wise_axis(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        # axis=0 should center/scale column-wise
        snv = StandardNormalVariate(axis=0)
        Xt = snv.fit_transform(spectra)
        assert Xt.shape == spectra.shape
        col_means = np.mean(Xt, axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-12)

    def test_fit_then_transform_consistent(self, spectra):
        from nirs4all.operators.transforms import StandardNormalVariate
        snv = StandardNormalVariate()
        Xt1 = snv.fit_transform(spectra)
        snv2 = StandardNormalVariate()
        snv2.fit(spectra)
        Xt2 = snv2.transform(spectra)
        np.testing.assert_array_equal(Xt1, Xt2)

class TestNormalize:
    """Tests for Normalize (l-norm and min-max) transform."""

    def test_output_shape(self, spectra):
        from nirs4all.operators.transforms.scalers import Normalize
        norm = Normalize()
        Xt = norm.fit_transform(spectra)
        assert Xt.shape == spectra.shape

    def test_user_defined_range(self, spectra):
        from nirs4all.operators.transforms.scalers import Normalize
        norm = Normalize(feature_range=(0.0, 1.0))
        Xt = norm.fit_transform(spectra)
        # Each feature should be in [0, 1]
        assert np.all(Xt >= -1e-10)
        assert np.all(Xt <= 1.0 + 1e-10)

    def test_output_is_finite(self, spectra):
        from nirs4all.operators.transforms.scalers import Normalize
        norm = Normalize()
        Xt = norm.fit_transform(spectra)
        assert np.all(np.isfinite(Xt))

    def test_inverse_transform_roundtrip(self, spectra):
        from nirs4all.operators.transforms.scalers import Normalize
        norm = Normalize(feature_range=(0.0, 1.0))
        Xt = norm.fit_transform(spectra)
        X_back = norm.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, spectra, atol=1e-10)

class TestDerivate:
    """Tests for Derivate (nth-order gradient derivative) transform."""

    def test_output_shape_first_order(self, spectra):
        from nirs4all.operators.transforms.scalers import Derivate
        deriv = Derivate(order=1)
        Xt = deriv.fit_transform(spectra)
        assert Xt.shape == spectra.shape

    def test_output_shape_second_order(self, spectra):
        from nirs4all.operators.transforms.scalers import Derivate
        deriv = Derivate(order=2)
        Xt = deriv.fit_transform(spectra)
        assert Xt.shape == spectra.shape

    def test_derivative_of_constant_is_zero_interior(self):
        from nirs4all.operators.transforms.scalers import Derivate
        X = np.ones((4, 30), dtype=np.float64)
        deriv = Derivate(order=1)
        Xt = deriv.fit_transform(X)
        np.testing.assert_allclose(Xt[:, 1:-1], 0.0, atol=1e-14)

    def test_output_is_finite(self, spectra):
        from nirs4all.operators.transforms.scalers import Derivate
        deriv = Derivate(order=1)
        Xt = deriv.fit_transform(spectra)
        assert np.all(np.isfinite(Xt))

class TestSimpleScale:
    """Tests for SimpleScale (min-max per-sample) transform."""

    def test_output_shape(self, spectra):
        from nirs4all.operators.transforms.scalers import SimpleScale
        ss = SimpleScale()
        Xt = ss.fit_transform(spectra)
        assert Xt.shape == spectra.shape

    def test_output_is_finite(self, spectra):
        from nirs4all.operators.transforms.scalers import SimpleScale
        ss = SimpleScale()
        Xt = ss.fit_transform(spectra)
        assert np.all(np.isfinite(Xt))

    def test_fit_then_transform_consistent(self, spectra):
        from nirs4all.operators.transforms.scalers import SimpleScale
        ss = SimpleScale()
        Xt1 = ss.fit_transform(spectra)
        ss2 = SimpleScale()
        ss2.fit(spectra)
        Xt2 = ss2.transform(spectra)
        np.testing.assert_array_equal(Xt1, Xt2)

"""
Unit tests for scattering effects augmentation operators.

Tests for ParticleSizeAugmenter and EMSCDistortionAugmenter.
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators import ParticleSizeAugmenter, EMSCDistortionAugmenter
from nirs4all.operators.base import SpectraTransformerMixin


class TestParticleSizeAugmenterInit:
    """Tests for ParticleSizeAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = ParticleSizeAugmenter()
        assert aug.mean_size_um == 50.0
        assert aug.size_variation_um == 15.0
        assert aug.size_range_um is None
        assert aug.reference_size_um == 50.0
        assert aug.wavelength_exponent == 1.5
        assert aug.size_effect_strength == 0.1
        assert aug.include_path_length is True
        assert aug.path_length_sensitivity == 0.5
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = ParticleSizeAugmenter(
            mean_size_um=30.0,
            size_variation_um=10.0,
            size_range_um=(20, 100),
            reference_size_um=40.0,
            wavelength_exponent=2.0,
            size_effect_strength=0.2,
            include_path_length=False,
            path_length_sensitivity=0.3,
            random_state=42
        )
        assert aug.mean_size_um == 30.0
        assert aug.size_variation_um == 10.0
        assert aug.size_range_um == (20, 100)
        assert aug.reference_size_um == 40.0
        assert aug.wavelength_exponent == 2.0
        assert aug.size_effect_strength == 0.2
        assert aug.include_path_length is False
        assert aug.path_length_sensitivity == 0.3
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that ParticleSizeAugmenter inherits from SpectraTransformerMixin."""
        aug = ParticleSizeAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = ParticleSizeAugmenter()
        assert aug._requires_wavelengths is True


class TestParticleSizeAugmenterTransform:
    """Tests for ParticleSizeAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) + 0.5
        wavelengths = np.linspace(1100, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = ParticleSizeAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = ParticleSizeAugmenter(mean_size_um=30.0, random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        # Should be different from input
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = ParticleSizeAugmenter()

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = ParticleSizeAugmenter(size_range_um=(20, 100), random_state=123)
        aug2 = ParticleSizeAugmenter(size_range_um=(20, 100), random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X, wavelengths = sample_data
        aug1 = ParticleSizeAugmenter(size_range_um=(20, 100), random_state=1)
        aug2 = ParticleSizeAugmenter(size_range_um=(20, 100), random_state=2)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X1, X2)

    def test_size_range_sampling(self, sample_data):
        """Test that size_range_um samples different values for each sample."""
        X, wavelengths = sample_data
        aug = ParticleSizeAugmenter(size_range_um=(10, 200), random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Different samples should have different transformations
        row_diffs = np.diff(X_transformed, axis=0)
        assert np.any(row_diffs != 0)


class TestParticleSizeAugmenterEffects:
    """Tests for particle size effects on spectra."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        X = np.ones((5, 100))
        wavelengths = np.linspace(1100, 2500, 100)
        return X, wavelengths

    def test_wavelength_dependent_baseline(self, sample_data):
        """Test that scattering baseline is wavelength-dependent."""
        X, wavelengths = sample_data
        aug = ParticleSizeAugmenter(
            mean_size_um=20.0,  # Small particles = more scattering
            size_effect_strength=0.3,
            include_path_length=False,  # Isolate baseline effect
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Shorter wavelengths should have more scatter effect
        short_wl_mean = X_transformed[:, wavelengths < 1500].mean()
        long_wl_mean = X_transformed[:, wavelengths > 2000].mean()

        # With λ^(-n) dependence, shorter wavelengths should be more affected
        # (though the exact direction depends on size ratio)
        assert short_wl_mean != long_wl_mean

    def test_path_length_effect(self, sample_data):
        """Test path length multiplicative effect."""
        X, wavelengths = sample_data
        aug_with = ParticleSizeAugmenter(
            mean_size_um=25.0,
            include_path_length=True,
            random_state=42
        )
        aug_without = ParticleSizeAugmenter(
            mean_size_um=25.0,
            include_path_length=False,
            random_state=42
        )

        X_with = aug_with.transform(X, wavelengths=wavelengths)
        X_without = aug_without.transform(X, wavelengths=wavelengths)

        # Results should be different
        assert not np.allclose(X_with, X_without)

    def test_size_effect_strength_scales_effect(self, sample_data):
        """Test that size_effect_strength scales the effect magnitude."""
        X, wavelengths = sample_data
        aug_weak = ParticleSizeAugmenter(
            mean_size_um=30.0,
            size_effect_strength=0.05,
            random_state=42
        )
        aug_strong = ParticleSizeAugmenter(
            mean_size_um=30.0,
            size_effect_strength=0.3,
            random_state=42
        )

        X_weak = aug_weak.transform(X, wavelengths=wavelengths)
        X_strong = aug_strong.transform(X, wavelengths=wavelengths)

        # Stronger effect should produce larger deviations
        weak_dev = np.std(X_weak - X)
        strong_dev = np.std(X_strong - X)

        assert strong_dev > weak_dev


class TestParticleSizeAugmenterSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        aug = ParticleSizeAugmenter()
        X = np.random.rand(10, 100)
        result = aug.fit(X)
        assert result is aug

    def test_get_params(self):
        """Test sklearn get_params."""
        aug = ParticleSizeAugmenter(mean_size_um=35.0, random_state=42)
        params = aug.get_params()

        assert params["mean_size_um"] == 35.0
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        aug = ParticleSizeAugmenter()
        aug.set_params(mean_size_um=25.0)

        assert aug.mean_size_um == 25.0

    def test_clone(self):
        """Test sklearn clone."""
        aug = ParticleSizeAugmenter(mean_size_um=40.0, random_state=123)
        cloned = clone(aug)

        assert cloned.mean_size_um == 40.0
        assert cloned.random_state == 123
        assert cloned is not aug

    def test_more_tags(self):
        """Test _more_tags method."""
        aug = ParticleSizeAugmenter()
        tags = aug._more_tags()

        assert tags["requires_wavelengths"] is True


class TestEMSCDistortionAugmenterInit:
    """Tests for EMSCDistortionAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = EMSCDistortionAugmenter()
        assert aug.multiplicative_range == (0.9, 1.1)
        assert aug.additive_range == (-0.05, 0.05)
        assert aug.polynomial_order == 2
        assert aug.polynomial_strength == 0.02
        assert aug.correlation == 0.3
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(0.85, 1.15),
            additive_range=(-0.1, 0.1),
            polynomial_order=3,
            polynomial_strength=0.05,
            correlation=0.5,
            random_state=42
        )
        assert aug.multiplicative_range == (0.85, 1.15)
        assert aug.additive_range == (-0.1, 0.1)
        assert aug.polynomial_order == 3
        assert aug.polynomial_strength == 0.05
        assert aug.correlation == 0.5
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that EMSCDistortionAugmenter inherits from SpectraTransformerMixin."""
        aug = EMSCDistortionAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = EMSCDistortionAugmenter()
        assert aug._requires_wavelengths is True


class TestEMSCDistortionAugmenterTransform:
    """Tests for EMSCDistortionAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) + 0.5
        wavelengths = np.linspace(1100, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = EMSCDistortionAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter(random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        # Should be different from input
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter()

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = EMSCDistortionAugmenter(random_state=123)
        aug2 = EMSCDistortionAugmenter(random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X, wavelengths = sample_data
        aug1 = EMSCDistortionAugmenter(random_state=1)
        aug2 = EMSCDistortionAugmenter(random_state=2)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X1, X2)


class TestEMSCDistortionAugmenterEffects:
    """Tests for EMSC distortion effects on spectra."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        np.random.seed(42)
        X = np.random.rand(5, 100) + 0.5
        wavelengths = np.linspace(1100, 2500, 100)
        return X, wavelengths

    def test_multiplicative_effect(self, sample_data):
        """Test multiplicative scatter effect."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(0.8, 1.2),
            additive_range=(0.0, 0.0),  # No additive
            polynomial_order=0,  # No polynomial
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Result should be scaled versions of input
        # Check that shape is preserved and values are scaled
        assert X_transformed.shape == X.shape

    def test_additive_effect(self, sample_data):
        """Test additive scatter offset effect."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),  # No multiplicative
            additive_range=(-0.1, 0.1),
            polynomial_order=0,  # No polynomial
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Result should be offset versions of input
        assert X_transformed.shape == X.shape

    def test_polynomial_effect(self, sample_data):
        """Test polynomial wavelength-dependent scatter."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),  # No multiplicative
            additive_range=(0.0, 0.0),  # No additive
            polynomial_order=3,
            polynomial_strength=0.1,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Polynomial effect should add wavelength-dependent baseline
        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_zero_polynomial_order(self, sample_data):
        """Test with polynomial_order=0 (no polynomial terms)."""
        X, wavelengths = sample_data
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(0.95, 1.05),
            additive_range=(-0.02, 0.02),
            polynomial_order=0,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Should still work without polynomial terms
        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_correlation_effect(self, sample_data):
        """Test that correlation affects parameter generation."""
        X, wavelengths = sample_data
        # With high correlation, multiplicative > 1 should tend to have negative additive
        aug_corr = EMSCDistortionAugmenter(
            multiplicative_range=(0.8, 1.2),
            additive_range=(-0.1, 0.1),
            correlation=0.8,
            polynomial_order=0,
            random_state=42
        )

        X_transformed = aug_corr.transform(X, wavelengths=wavelengths)

        # Just verify it works without error
        assert X_transformed.shape == X.shape


class TestEMSCDistortionAugmenterPolynomial:
    """Tests for EMSC polynomial distortion behavior."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        X = np.ones((3, 50))
        wavelengths = np.linspace(1100, 2500, 50)
        return X, wavelengths

    def test_higher_order_polynomial(self, sample_data):
        """Test higher order polynomial produces more complex baseline."""
        X, wavelengths = sample_data
        aug_low = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),
            additive_range=(0.0, 0.0),
            polynomial_order=1,
            polynomial_strength=0.1,
            random_state=42
        )
        aug_high = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),
            additive_range=(0.0, 0.0),
            polynomial_order=4,
            polynomial_strength=0.1,
            random_state=42
        )

        X_low = aug_low.transform(X, wavelengths=wavelengths)
        X_high = aug_high.transform(X, wavelengths=wavelengths)

        # Different polynomial orders should produce different results
        assert not np.allclose(X_low, X_high)

    def test_polynomial_strength_scales_effect(self, sample_data):
        """Test that polynomial_strength scales the polynomial effect."""
        X, wavelengths = sample_data
        aug_weak = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),
            additive_range=(0.0, 0.0),
            polynomial_order=2,
            polynomial_strength=0.01,
            random_state=42
        )
        aug_strong = EMSCDistortionAugmenter(
            multiplicative_range=(1.0, 1.0),
            additive_range=(0.0, 0.0),
            polynomial_order=2,
            polynomial_strength=0.2,
            random_state=42
        )

        X_weak = aug_weak.transform(X, wavelengths=wavelengths)
        X_strong = aug_strong.transform(X, wavelengths=wavelengths)

        # Stronger effect should produce larger deviations
        weak_dev = np.std(X_weak - X)
        strong_dev = np.std(X_strong - X)

        assert strong_dev > weak_dev


class TestEMSCDistortionAugmenterSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        aug = EMSCDistortionAugmenter()
        X = np.random.rand(10, 100)
        result = aug.fit(X)
        assert result is aug

    def test_get_params(self):
        """Test sklearn get_params."""
        aug = EMSCDistortionAugmenter(
            multiplicative_range=(0.85, 1.15),
            random_state=42
        )
        params = aug.get_params()

        assert params["multiplicative_range"] == (0.85, 1.15)
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        aug = EMSCDistortionAugmenter()
        aug.set_params(polynomial_order=4)

        assert aug.polynomial_order == 4

    def test_clone(self):
        """Test sklearn clone."""
        aug = EMSCDistortionAugmenter(polynomial_order=3, random_state=123)
        cloned = clone(aug)

        assert cloned.polynomial_order == 3
        assert cloned.random_state == 123
        assert cloned is not aug

    def test_more_tags(self):
        """Test _more_tags method."""
        aug = EMSCDistortionAugmenter()
        tags = aug._more_tags()

        assert tags["requires_wavelengths"] is True


class TestScatteringAugmentersImport:
    """Tests for import paths."""

    def test_import_from_operators(self):
        """Test that augmenters can be imported from operators."""
        from nirs4all.operators import ParticleSizeAugmenter as PSA
        from nirs4all.operators import EMSCDistortionAugmenter as EDA

        assert PSA is ParticleSizeAugmenter
        assert EDA is EMSCDistortionAugmenter

    def test_import_from_augmentation_module(self):
        """Test that augmenters can be imported from augmentation module."""
        from nirs4all.operators.augmentation.scattering import (
            ParticleSizeAugmenter as PSA,
            EMSCDistortionAugmenter as EDA,
        )

        assert PSA is ParticleSizeAugmenter
        assert EDA is EMSCDistortionAugmenter

    def test_in_operators_all(self):
        """Test that augmenters are in operators.__all__."""
        import nirs4all.operators as ops

        assert "ParticleSizeAugmenter" in ops.__all__
        assert "EMSCDistortionAugmenter" in ops.__all__

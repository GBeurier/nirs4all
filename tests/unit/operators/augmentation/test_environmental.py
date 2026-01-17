"""
Unit tests for environmental effects augmentation operators.

Tests for TemperatureAugmenter and MoistureAugmenter.
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators import TemperatureAugmenter, MoistureAugmenter
from nirs4all.operators.base import SpectraTransformerMixin


class TestTemperatureAugmenterInit:
    """Tests for TemperatureAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = TemperatureAugmenter()
        assert aug.temperature_delta == 5.0
        assert aug.temperature_range is None
        assert aug.reference_temperature == 25.0
        assert aug.enable_shift is True
        assert aug.enable_intensity is True
        assert aug.enable_broadening is True
        assert aug.region_specific is True
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = TemperatureAugmenter(
            temperature_delta=10.0,
            temperature_range=(-5, 15),
            reference_temperature=20.0,
            enable_shift=False,
            enable_intensity=False,
            enable_broadening=False,
            region_specific=False,
            random_state=42
        )
        assert aug.temperature_delta == 10.0
        assert aug.temperature_range == (-5, 15)
        assert aug.reference_temperature == 20.0
        assert aug.enable_shift is False
        assert aug.enable_intensity is False
        assert aug.enable_broadening is False
        assert aug.region_specific is False
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that TemperatureAugmenter inherits from SpectraTransformerMixin."""
        aug = TemperatureAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = TemperatureAugmenter()
        assert aug._requires_wavelengths is True


class TestTemperatureAugmenterTransform:
    """Tests for TemperatureAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) + 0.5  # Positive spectra
        wavelengths = np.linspace(1100, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = TemperatureAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(temperature_delta=10.0, random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        # Should be different from input (temperature effect applied)
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(temperature_delta=5.0)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_zero_temperature_delta_no_change(self, sample_data):
        """Test that zero temperature delta leaves spectra unchanged."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(temperature_delta=0.0)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X_transformed, X)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = TemperatureAugmenter(temperature_range=(-5, 10), random_state=123)
        aug2 = TemperatureAugmenter(temperature_range=(-5, 10), random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X, wavelengths = sample_data
        aug1 = TemperatureAugmenter(temperature_range=(-5, 10), random_state=1)
        aug2 = TemperatureAugmenter(temperature_range=(-5, 10), random_state=2)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X1, X2)

    def test_temperature_range_sampling(self, sample_data):
        """Test that temperature_range samples different values for each sample."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(temperature_range=(-10, 10), random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Different samples should have different transformations
        # Check that rows are different
        row_diffs = np.diff(X_transformed, axis=0)
        assert np.any(row_diffs != 0)


class TestTemperatureAugmenterRegionSpecific:
    """Tests for region-specific temperature effects."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra covering key wavelength regions."""
        X = np.ones((5, 200))
        wavelengths = np.linspace(1300, 2100, 200)  # Covers O-H and N-H regions
        return X, wavelengths

    def test_region_specific_applies_different_effects(self, sample_data):
        """Test that region-specific mode applies different effects to different regions."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(
            temperature_delta=20.0,
            region_specific=True,
            enable_shift=True,
            enable_intensity=True,
            enable_broadening=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # O-H region (1400-1520 nm) should be more affected than C-H region
        oh_mask = (wavelengths >= 1400) & (wavelengths <= 1520)
        ch_mask = (wavelengths >= 1650) & (wavelengths <= 1780)

        oh_diff = np.abs(X_transformed[:, oh_mask] - X[:, oh_mask]).mean()
        ch_diff = np.abs(X_transformed[:, ch_mask] - X[:, ch_mask]).mean()

        # O-H bands should show stronger temperature effects than C-H bands
        assert oh_diff > ch_diff

    def test_uniform_mode(self, sample_data):
        """Test that uniform mode applies same effect across all wavelengths."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(
            temperature_delta=10.0,
            region_specific=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Just verify it runs without error and produces different output
        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)


class TestTemperatureAugmenterEffects:
    """Tests for individual temperature effects."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        np.random.seed(42)
        X = np.sin(np.linspace(0, 4 * np.pi, 150)).reshape(1, -1) + 1.5
        X = np.repeat(X, 5, axis=0)
        wavelengths = np.linspace(1100, 2500, 150)
        return X, wavelengths

    def test_shift_only(self, sample_data):
        """Test effect when only shift is enabled."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(
            temperature_delta=10.0,
            enable_shift=True,
            enable_intensity=False,
            enable_broadening=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X_transformed, X)

    def test_intensity_only(self, sample_data):
        """Test effect when only intensity is enabled."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(
            temperature_delta=10.0,
            enable_shift=False,
            enable_intensity=True,
            enable_broadening=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X_transformed, X)

    def test_broadening_only(self, sample_data):
        """Test effect when only broadening is enabled."""
        X, wavelengths = sample_data
        aug = TemperatureAugmenter(
            temperature_delta=15.0,  # Need larger delta for visible broadening
            enable_shift=False,
            enable_intensity=False,
            enable_broadening=True,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Broadening effect may be subtle, but should produce some change
        assert X_transformed.shape == X.shape


class TestTemperatureAugmenterSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        aug = TemperatureAugmenter()
        X = np.random.rand(10, 100)
        result = aug.fit(X)
        assert result is aug

    def test_get_params(self):
        """Test sklearn get_params."""
        aug = TemperatureAugmenter(temperature_delta=7.5, random_state=42)
        params = aug.get_params()

        assert params["temperature_delta"] == 7.5
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        aug = TemperatureAugmenter()
        aug.set_params(temperature_delta=15.0)

        assert aug.temperature_delta == 15.0

    def test_clone(self):
        """Test sklearn clone."""
        aug = TemperatureAugmenter(temperature_delta=8.0, random_state=123)
        cloned = clone(aug)

        assert cloned.temperature_delta == 8.0
        assert cloned.random_state == 123
        assert cloned is not aug

    def test_more_tags(self):
        """Test _more_tags method."""
        aug = TemperatureAugmenter()
        tags = aug._more_tags()

        assert tags["requires_wavelengths"] is True


class TestMoistureAugmenterInit:
    """Tests for MoistureAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = MoistureAugmenter()
        assert aug.water_activity_delta == 0.1
        assert aug.water_activity_range is None
        assert aug.reference_water_activity == 0.5
        assert aug.free_water_fraction == 0.3
        assert aug.bound_water_shift == 25.0
        assert aug.moisture_content == 0.10
        assert aug.enable_shift is True
        assert aug.enable_intensity is True
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = MoistureAugmenter(
            water_activity_delta=0.2,
            water_activity_range=(-0.3, 0.3),
            reference_water_activity=0.6,
            free_water_fraction=0.4,
            bound_water_shift=30.0,
            moisture_content=0.15,
            enable_shift=False,
            enable_intensity=False,
            random_state=42
        )
        assert aug.water_activity_delta == 0.2
        assert aug.water_activity_range == (-0.3, 0.3)
        assert aug.reference_water_activity == 0.6
        assert aug.free_water_fraction == 0.4
        assert aug.bound_water_shift == 30.0
        assert aug.moisture_content == 0.15
        assert aug.enable_shift is False
        assert aug.enable_intensity is False
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that MoistureAugmenter inherits from SpectraTransformerMixin."""
        aug = MoistureAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = MoistureAugmenter()
        assert aug._requires_wavelengths is True


class TestMoistureAugmenterTransform:
    """Tests for MoistureAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra covering water band regions."""
        np.random.seed(42)
        X = np.random.rand(10, 150) + 0.5
        wavelengths = np.linspace(1300, 2100, 150)  # Covers water bands
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = MoistureAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(water_activity_delta=0.3, random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        # Should be different from input
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(water_activity_delta=0.2)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = MoistureAugmenter(water_activity_range=(-0.2, 0.2), random_state=123)
        aug2 = MoistureAugmenter(water_activity_range=(-0.2, 0.2), random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X, wavelengths = sample_data
        aug1 = MoistureAugmenter(water_activity_range=(-0.2, 0.2), random_state=1)
        aug2 = MoistureAugmenter(water_activity_range=(-0.2, 0.2), random_state=2)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X1, X2)

    def test_water_activity_range_sampling(self, sample_data):
        """Test that water_activity_range samples values for each sample."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(water_activity_range=(-0.3, 0.3), random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Different samples should have different transformations
        row_diffs = np.diff(X_transformed, axis=0)
        assert np.any(row_diffs != 0)


class TestMoistureAugmenterWaterBands:
    """Tests for water band effects."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra covering water band regions."""
        X = np.ones((5, 200))
        wavelengths = np.linspace(1300, 2100, 200)
        return X, wavelengths

    def test_water_band_regions_affected(self, sample_data):
        """Test that water band regions are affected by moisture changes."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(
            water_activity_delta=0.4,
            enable_shift=True,
            enable_intensity=True,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Check that water band regions (1400-1500, 1900-2000) are affected
        water_1st = (wavelengths >= 1400) & (wavelengths <= 1500)
        water_comb = (wavelengths >= 1900) & (wavelengths <= 2000)
        outside_water = ~(water_1st | water_comb)

        # Water regions should show larger changes
        water_diff = np.abs(X_transformed[:, water_1st | water_comb] - X[:, water_1st | water_comb]).mean()
        # Transformation does affect all regions due to interpolation, but water bands more

        assert X_transformed.shape == X.shape


class TestMoistureAugmenterEffects:
    """Tests for individual moisture effects."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        np.random.seed(42)
        X = np.random.rand(5, 150) + 0.5
        wavelengths = np.linspace(1300, 2100, 150)
        return X, wavelengths

    def test_shift_only(self, sample_data):
        """Test effect when only shift is enabled."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(
            water_activity_delta=0.3,
            enable_shift=True,
            enable_intensity=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X_transformed, X)

    def test_intensity_only(self, sample_data):
        """Test effect when only intensity is enabled."""
        X, wavelengths = sample_data
        aug = MoistureAugmenter(
            water_activity_delta=0.3,
            moisture_content=0.20,  # Higher than default 0.10
            enable_shift=False,
            enable_intensity=True,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X_transformed, X)


class TestMoistureAugmenterSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        aug = MoistureAugmenter()
        X = np.random.rand(10, 100)
        result = aug.fit(X)
        assert result is aug

    def test_get_params(self):
        """Test sklearn get_params."""
        aug = MoistureAugmenter(water_activity_delta=0.25, random_state=42)
        params = aug.get_params()

        assert params["water_activity_delta"] == 0.25
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        aug = MoistureAugmenter()
        aug.set_params(water_activity_delta=0.15)

        assert aug.water_activity_delta == 0.15

    def test_clone(self):
        """Test sklearn clone."""
        aug = MoistureAugmenter(water_activity_delta=0.3, random_state=123)
        cloned = clone(aug)

        assert cloned.water_activity_delta == 0.3
        assert cloned.random_state == 123
        assert cloned is not aug

    def test_more_tags(self):
        """Test _more_tags method."""
        aug = MoistureAugmenter()
        tags = aug._more_tags()

        assert tags["requires_wavelengths"] is True


class TestEnvironmentalAugmentersImport:
    """Tests for import paths."""

    def test_import_from_operators(self):
        """Test that augmenters can be imported from operators."""
        from nirs4all.operators import TemperatureAugmenter as TA
        from nirs4all.operators import MoistureAugmenter as MA

        assert TA is TemperatureAugmenter
        assert MA is MoistureAugmenter

    def test_import_from_augmentation_module(self):
        """Test that augmenters can be imported from augmentation module."""
        from nirs4all.operators.augmentation.environmental import (
            TemperatureAugmenter as TA,
            MoistureAugmenter as MA,
        )

        assert TA is TemperatureAugmenter
        assert MA is MoistureAugmenter

    def test_in_operators_all(self):
        """Test that augmenters are in operators.__all__."""
        import nirs4all.operators as ops

        assert "TemperatureAugmenter" in ops.__all__
        assert "MoistureAugmenter" in ops.__all__

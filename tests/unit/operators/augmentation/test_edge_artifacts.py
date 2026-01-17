"""
Unit tests for edge artifacts augmentation operators.

Tests for DetectorRollOffAugmenter, StrayLightAugmenter, EdgeCurvatureAugmenter,
TruncatedPeakAugmenter, and EdgeArtifactsAugmenter.
"""

import numpy as np
import pytest
from sklearn.base import clone

from nirs4all.operators import (
    DetectorRollOffAugmenter,
    StrayLightAugmenter,
    EdgeCurvatureAugmenter,
    TruncatedPeakAugmenter,
    EdgeArtifactsAugmenter,
    DETECTOR_MODELS,
)
from nirs4all.operators.base import SpectraTransformerMixin


# =============================================================================
# DetectorRollOffAugmenter Tests
# =============================================================================


class TestDetectorRollOffAugmenterInit:
    """Tests for DetectorRollOffAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = DetectorRollOffAugmenter()
        assert aug.detector_model == "generic_nir"
        assert aug.effect_strength == 1.0
        assert aug.noise_amplification == 0.02
        assert aug.include_baseline_distortion is True
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = DetectorRollOffAugmenter(
            detector_model="ingaas_standard",
            effect_strength=1.5,
            noise_amplification=0.05,
            include_baseline_distortion=False,
            random_state=42
        )
        assert aug.detector_model == "ingaas_standard"
        assert aug.effect_strength == 1.5
        assert aug.noise_amplification == 0.05
        assert aug.include_baseline_distortion is False
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that DetectorRollOffAugmenter inherits from SpectraTransformerMixin."""
        aug = DetectorRollOffAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = DetectorRollOffAugmenter()
        assert aug._requires_wavelengths is True


class TestDetectorRollOffAugmenterTransform:
    """Tests for DetectorRollOffAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) + 0.5
        wavelengths = np.linspace(900, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = DetectorRollOffAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = DetectorRollOffAugmenter(random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = DetectorRollOffAugmenter()

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = DetectorRollOffAugmenter(random_state=123)
        aug2 = DetectorRollOffAugmenter(random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states_different_results(self, sample_data):
        """Test that different random states produce different results."""
        X, wavelengths = sample_data
        aug1 = DetectorRollOffAugmenter(random_state=1)
        aug2 = DetectorRollOffAugmenter(random_state=2)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        assert not np.allclose(X1, X2)

    def test_invalid_detector_model_raises_error(self, sample_data):
        """Test that invalid detector model raises error."""
        X, wavelengths = sample_data
        aug = DetectorRollOffAugmenter(detector_model="invalid_model")

        with pytest.raises(ValueError, match="Unknown detector model"):
            aug.transform(X, wavelengths=wavelengths)

    def test_all_detector_models_work(self, sample_data):
        """Test that all detector models can be used."""
        X, wavelengths = sample_data

        for model_name in DETECTOR_MODELS.keys():
            aug = DetectorRollOffAugmenter(detector_model=model_name, random_state=42)
            X_transformed = aug.transform(X, wavelengths=wavelengths)
            assert X_transformed.shape == X.shape

    def test_zero_effect_strength(self, sample_data):
        """Test that zero effect_strength produces minimal change."""
        X, wavelengths = sample_data
        aug = DetectorRollOffAugmenter(
            effect_strength=0.0,
            noise_amplification=0.0,
            include_baseline_distortion=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Should be very close to original with no effects enabled
        np.testing.assert_array_almost_equal(X_transformed, X, decimal=5)


class TestDetectorRollOffAugmenterEdgeEffects:
    """Tests for edge-specific effects."""

    @pytest.fixture
    def edge_data(self):
        """Create data with wavelengths extending beyond detector optimal range."""
        X = np.ones((5, 200))
        wavelengths = np.linspace(800, 2800, 200)  # Beyond typical detector ranges
        return X, wavelengths

    def test_edge_regions_more_affected(self, edge_data):
        """Test that edge regions show larger effects than center."""
        X, wavelengths = edge_data
        aug = DetectorRollOffAugmenter(
            detector_model="generic_nir",
            effect_strength=2.0,
            noise_amplification=0.05,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Edges should have more noise/distortion than center
        center_mask = (wavelengths >= 1100) & (wavelengths <= 1500)
        edge_mask = (wavelengths < 1000) | (wavelengths > 2000)

        center_diff = np.std(X_transformed[:, center_mask] - X[:, center_mask])
        edge_diff = np.std(X_transformed[:, edge_mask] - X[:, edge_mask])

        assert edge_diff >= center_diff


# =============================================================================
# StrayLightAugmenter Tests
# =============================================================================


class TestStrayLightAugmenterInit:
    """Tests for StrayLightAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = StrayLightAugmenter()
        assert aug.stray_light_fraction == 0.001
        assert aug.edge_enhancement == 2.0
        assert aug.edge_width == 0.1
        assert aug.include_peak_truncation is True
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = StrayLightAugmenter(
            stray_light_fraction=0.005,
            edge_enhancement=3.0,
            edge_width=0.15,
            include_peak_truncation=False,
            random_state=42
        )
        assert aug.stray_light_fraction == 0.005
        assert aug.edge_enhancement == 3.0
        assert aug.edge_width == 0.15
        assert aug.include_peak_truncation is False
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that StrayLightAugmenter inherits from SpectraTransformerMixin."""
        aug = StrayLightAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = StrayLightAugmenter()
        assert aug._requires_wavelengths is True


class TestStrayLightAugmenterTransform:
    """Tests for StrayLightAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) * 0.8 + 0.2  # Absorbance range 0.2-1.0
        wavelengths = np.linspace(1000, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = StrayLightAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = StrayLightAugmenter(random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_transform_output_shape_preserved(self, sample_data):
        """Test that transform preserves input shape."""
        X, wavelengths = sample_data
        aug = StrayLightAugmenter()

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_stray_light_reduces_absorbance(self, sample_data):
        """Test that stray light reduces observed absorbance (Beer's law deviation)."""
        X, wavelengths = sample_data
        aug = StrayLightAugmenter(
            stray_light_fraction=0.01,  # 1% stray light
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Stray light should reduce observed absorbance
        # (more photons reaching detector than expected)
        assert np.mean(X_transformed) < np.mean(X)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = StrayLightAugmenter(random_state=123)
        aug2 = StrayLightAugmenter(random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_higher_stray_light_larger_effect(self, sample_data):
        """Test that higher stray light fraction produces larger effect."""
        X, wavelengths = sample_data
        aug_low = StrayLightAugmenter(stray_light_fraction=0.001, random_state=42)
        aug_high = StrayLightAugmenter(stray_light_fraction=0.01, random_state=42)

        X_low = aug_low.transform(X, wavelengths=wavelengths)
        X_high = aug_high.transform(X, wavelengths=wavelengths)

        diff_low = np.abs(X_low - X).mean()
        diff_high = np.abs(X_high - X).mean()

        assert diff_high > diff_low


class TestStrayLightAugmenterEdgeEffects:
    """Tests for edge enhancement of stray light."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra."""
        X = np.ones((5, 100)) * 0.8  # Uniform absorbance
        wavelengths = np.linspace(1000, 2500, 100)
        return X, wavelengths

    def test_edge_enhancement_effect(self, sample_data):
        """Test that edge regions are more affected with edge_enhancement > 1."""
        X, wavelengths = sample_data
        aug = StrayLightAugmenter(
            stray_light_fraction=0.005,
            edge_enhancement=3.0,
            edge_width=0.15,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Calculate effect at edges vs center
        n_edge = int(len(wavelengths) * 0.15)
        left_edge = X_transformed[:, :n_edge]
        right_edge = X_transformed[:, -n_edge:]
        center = X_transformed[:, n_edge:-n_edge]

        edge_diff = np.abs(np.concatenate([left_edge, right_edge], axis=1) - X[:, :1]).mean()
        center_diff = np.abs(center - X[:, :1]).mean()

        # Edge effect should be larger
        assert edge_diff >= center_diff


# =============================================================================
# EdgeCurvatureAugmenter Tests
# =============================================================================


class TestEdgeCurvatureAugmenterInit:
    """Tests for EdgeCurvatureAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = EdgeCurvatureAugmenter()
        assert aug.curvature_strength == 0.02
        assert aug.curvature_type == "random"
        assert aug.asymmetry == 0.0
        assert aug.edge_focus == 0.7
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = EdgeCurvatureAugmenter(
            curvature_strength=0.05,
            curvature_type="smile",
            asymmetry=0.3,
            edge_focus=0.9,
            random_state=42
        )
        assert aug.curvature_strength == 0.05
        assert aug.curvature_type == "smile"
        assert aug.asymmetry == 0.3
        assert aug.edge_focus == 0.9
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that EdgeCurvatureAugmenter inherits from SpectraTransformerMixin."""
        aug = EdgeCurvatureAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = EdgeCurvatureAugmenter()
        assert aug._requires_wavelengths is True


class TestEdgeCurvatureAugmenterTransform:
    """Tests for EdgeCurvatureAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        X = np.ones((10, 100)) * 0.5  # Flat spectra
        wavelengths = np.linspace(1000, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = EdgeCurvatureAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = EdgeCurvatureAugmenter(random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_curvature_type_smile(self, sample_data):
        """Test smile curvature (upward at edges)."""
        X, wavelengths = sample_data
        aug = EdgeCurvatureAugmenter(
            curvature_type="smile",
            curvature_strength=0.05,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Edges should be higher than center with smile
        edge_mean = (X_transformed[:, :10].mean() + X_transformed[:, -10:].mean()) / 2
        center_mean = X_transformed[:, 45:55].mean()

        assert edge_mean > center_mean

    def test_curvature_type_frown(self, sample_data):
        """Test frown curvature (downward at edges)."""
        X, wavelengths = sample_data
        aug = EdgeCurvatureAugmenter(
            curvature_type="frown",
            curvature_strength=0.05,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Edges should be lower than center with frown
        edge_mean = (X_transformed[:, :10].mean() + X_transformed[:, -10:].mean()) / 2
        center_mean = X_transformed[:, 45:55].mean()

        assert edge_mean < center_mean

    def test_curvature_type_asymmetric(self, sample_data):
        """Test asymmetric curvature."""
        X, wavelengths = sample_data
        aug = EdgeCurvatureAugmenter(
            curvature_type="asymmetric",
            asymmetry=0.5,  # Emphasize left
            curvature_strength=0.05,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Should run without error
        assert X_transformed.shape == X.shape

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = EdgeCurvatureAugmenter(curvature_type="smile", random_state=123)
        aug2 = EdgeCurvatureAugmenter(curvature_type="smile", random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_invalid_curvature_type_raises_error(self, sample_data):
        """Test that invalid curvature type raises error."""
        X, wavelengths = sample_data
        aug = EdgeCurvatureAugmenter(curvature_type="invalid")

        with pytest.raises(ValueError, match="Unknown curvature_type"):
            aug.transform(X, wavelengths=wavelengths)


# =============================================================================
# TruncatedPeakAugmenter Tests
# =============================================================================


class TestTruncatedPeakAugmenterInit:
    """Tests for TruncatedPeakAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = TruncatedPeakAugmenter()
        assert aug.peak_probability == 0.3
        assert aug.amplitude_range == (0.01, 0.1)
        assert aug.width_range == (50, 200)
        assert aug.left_edge is True
        assert aug.right_edge is True
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = TruncatedPeakAugmenter(
            peak_probability=0.5,
            amplitude_range=(0.05, 0.2),
            width_range=(100, 300),
            left_edge=False,
            right_edge=True,
            random_state=42
        )
        assert aug.peak_probability == 0.5
        assert aug.amplitude_range == (0.05, 0.2)
        assert aug.width_range == (100, 300)
        assert aug.left_edge is False
        assert aug.right_edge is True
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that TruncatedPeakAugmenter inherits from SpectraTransformerMixin."""
        aug = TruncatedPeakAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = TruncatedPeakAugmenter()
        assert aug._requires_wavelengths is True


class TestTruncatedPeakAugmenterTransform:
    """Tests for TruncatedPeakAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        X = np.zeros((20, 100))  # Zero baseline to see added peaks
        wavelengths = np.linspace(1000, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = TruncatedPeakAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = TruncatedPeakAugmenter(peak_probability=1.0, random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_high_probability_adds_peaks(self, sample_data):
        """Test that high probability adds truncated peaks."""
        X, wavelengths = sample_data
        aug = TruncatedPeakAugmenter(
            peak_probability=1.0,
            amplitude_range=(0.1, 0.2),
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Should have added some peaks at edges
        assert np.max(X_transformed) > 0

    def test_zero_probability_no_change(self, sample_data):
        """Test that zero probability produces no change."""
        X, wavelengths = sample_data
        aug = TruncatedPeakAugmenter(
            peak_probability=0.0,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X_transformed, X)

    def test_left_edge_only(self, sample_data):
        """Test that left_edge=True, right_edge=False only affects left."""
        X, wavelengths = sample_data
        aug = TruncatedPeakAugmenter(
            peak_probability=1.0,
            left_edge=True,
            right_edge=False,
            amplitude_range=(0.1, 0.2),
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Left edge should have larger values than right
        left_max = X_transformed[:, :20].max()
        right_max = X_transformed[:, -20:].max()

        assert left_max >= right_max

    def test_right_edge_only(self, sample_data):
        """Test that right_edge=True, left_edge=False only affects right."""
        X, wavelengths = sample_data
        aug = TruncatedPeakAugmenter(
            peak_probability=1.0,
            left_edge=False,
            right_edge=True,
            amplitude_range=(0.1, 0.2),
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # Right edge should have larger values than left
        left_max = X_transformed[:, :20].max()
        right_max = X_transformed[:, -20:].max()

        assert right_max >= left_max

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = TruncatedPeakAugmenter(peak_probability=1.0, random_state=123)
        aug2 = TruncatedPeakAugmenter(peak_probability=1.0, random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)


# =============================================================================
# EdgeArtifactsAugmenter (Combined) Tests
# =============================================================================


class TestEdgeArtifactsAugmenterInit:
    """Tests for EdgeArtifactsAugmenter initialization."""

    def test_default_initialization(self):
        """Test default parameter values."""
        aug = EdgeArtifactsAugmenter()
        assert aug.detector_roll_off is True
        assert aug.stray_light is True
        assert aug.edge_curvature is True
        assert aug.truncated_peaks is True
        assert aug.overall_strength == 1.0
        assert aug.detector_model == "generic_nir"
        assert aug.random_state is None

    def test_custom_initialization(self):
        """Test custom parameter values."""
        aug = EdgeArtifactsAugmenter(
            detector_roll_off=False,
            stray_light=True,
            edge_curvature=False,
            truncated_peaks=True,
            overall_strength=0.5,
            detector_model="ingaas_standard",
            random_state=42
        )
        assert aug.detector_roll_off is False
        assert aug.stray_light is True
        assert aug.edge_curvature is False
        assert aug.truncated_peaks is True
        assert aug.overall_strength == 0.5
        assert aug.detector_model == "ingaas_standard"
        assert aug.random_state == 42

    def test_inherits_from_spectra_transformer_mixin(self):
        """Test that EdgeArtifactsAugmenter inherits from SpectraTransformerMixin."""
        aug = EdgeArtifactsAugmenter()
        assert isinstance(aug, SpectraTransformerMixin)

    def test_requires_wavelengths_flag(self):
        """Test that _requires_wavelengths is True."""
        aug = EdgeArtifactsAugmenter()
        assert aug._requires_wavelengths is True


class TestEdgeArtifactsAugmenterTransform:
    """Tests for EdgeArtifactsAugmenter transform method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectra and wavelengths."""
        np.random.seed(42)
        X = np.random.rand(10, 100) * 0.5 + 0.3  # Typical absorbance range
        wavelengths = np.linspace(1000, 2500, 100)
        return X, wavelengths

    def test_transform_without_wavelengths_raises_error(self, sample_data):
        """Test that transform raises error when wavelengths not provided."""
        X, _ = sample_data
        aug = EdgeArtifactsAugmenter()

        with pytest.raises(ValueError, match="requires wavelengths"):
            aug.transform(X)

    def test_transform_with_wavelengths(self, sample_data):
        """Test that transform works when wavelengths are provided."""
        X, wavelengths = sample_data
        aug = EdgeArtifactsAugmenter(random_state=42)

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape
        assert not np.allclose(X_transformed, X)

    def test_all_effects_enabled(self, sample_data):
        """Test with all effects enabled."""
        X, wavelengths = sample_data
        aug = EdgeArtifactsAugmenter(
            detector_roll_off=True,
            stray_light=True,
            edge_curvature=True,
            truncated_peaks=True,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        assert X_transformed.shape == X.shape

    def test_all_effects_disabled(self, sample_data):
        """Test with all effects disabled."""
        X, wavelengths = sample_data
        aug = EdgeArtifactsAugmenter(
            detector_roll_off=False,
            stray_light=False,
            edge_curvature=False,
            truncated_peaks=False,
            random_state=42
        )

        X_transformed = aug.transform(X, wavelengths=wavelengths)

        # With all effects disabled, should be close to original
        np.testing.assert_array_almost_equal(X_transformed, X)

    def test_individual_effects(self, sample_data):
        """Test enabling individual effects one at a time."""
        X, wavelengths = sample_data

        effects = ["detector_roll_off", "stray_light", "edge_curvature", "truncated_peaks"]

        for effect in effects:
            kwargs = {e: False for e in effects}
            kwargs[effect] = True
            kwargs["random_state"] = 42

            aug = EdgeArtifactsAugmenter(**kwargs)
            X_transformed = aug.transform(X, wavelengths=wavelengths)

            assert X_transformed.shape == X.shape
            # At least detector roll-off and stray light should make a difference
            if effect in ["detector_roll_off", "stray_light"]:
                assert not np.allclose(X_transformed, X)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that results are reproducible with random_state."""
        X, wavelengths = sample_data
        aug1 = EdgeArtifactsAugmenter(random_state=123)
        aug2 = EdgeArtifactsAugmenter(random_state=123)

        X1 = aug1.transform(X, wavelengths=wavelengths)
        X2 = aug2.transform(X, wavelengths=wavelengths)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_strength_levels(self, sample_data):
        """Test that different strength levels produce different magnitudes."""
        X, wavelengths = sample_data
        aug_low = EdgeArtifactsAugmenter(overall_strength=0.5, random_state=42)
        aug_high = EdgeArtifactsAugmenter(overall_strength=2.0, random_state=42)

        X_low = aug_low.transform(X, wavelengths=wavelengths)
        X_high = aug_high.transform(X, wavelengths=wavelengths)

        diff_low = np.abs(X_low - X).mean()
        diff_high = np.abs(X_high - X).mean()

        assert diff_high > diff_low


# =============================================================================
# sklearn Compatibility Tests
# =============================================================================


class TestEdgeArtifactsSklearnCompatibility:
    """Tests for sklearn compatibility of all edge artifact augmenters."""

    @pytest.mark.parametrize("AugmenterClass", [
        DetectorRollOffAugmenter,
        StrayLightAugmenter,
        EdgeCurvatureAugmenter,
        TruncatedPeakAugmenter,
        EdgeArtifactsAugmenter,
    ])
    def test_fit_returns_self(self, AugmenterClass):
        """Test that fit returns self."""
        aug = AugmenterClass()
        X = np.random.rand(10, 100)
        result = aug.fit(X)
        assert result is aug

    @pytest.mark.parametrize("AugmenterClass", [
        DetectorRollOffAugmenter,
        StrayLightAugmenter,
        EdgeCurvatureAugmenter,
        TruncatedPeakAugmenter,
        EdgeArtifactsAugmenter,
    ])
    def test_get_params(self, AugmenterClass):
        """Test sklearn get_params."""
        aug = AugmenterClass(random_state=42)
        params = aug.get_params()
        assert params["random_state"] == 42

    @pytest.mark.parametrize("AugmenterClass", [
        DetectorRollOffAugmenter,
        StrayLightAugmenter,
        EdgeCurvatureAugmenter,
        TruncatedPeakAugmenter,
        EdgeArtifactsAugmenter,
    ])
    def test_clone(self, AugmenterClass):
        """Test sklearn clone."""
        aug = AugmenterClass(random_state=123)
        cloned = clone(aug)

        assert cloned.random_state == 123
        assert cloned is not aug

    @pytest.mark.parametrize("AugmenterClass", [
        DetectorRollOffAugmenter,
        StrayLightAugmenter,
        EdgeCurvatureAugmenter,
        TruncatedPeakAugmenter,
        EdgeArtifactsAugmenter,
    ])
    def test_more_tags(self, AugmenterClass):
        """Test _more_tags method."""
        aug = AugmenterClass()
        tags = aug._more_tags()
        assert tags["requires_wavelengths"] is True


# =============================================================================
# Import Tests
# =============================================================================


class TestEdgeArtifactsImports:
    """Tests for import paths."""

    def test_import_from_operators(self):
        """Test that augmenters can be imported from operators."""
        from nirs4all.operators import (
            DetectorRollOffAugmenter,
            StrayLightAugmenter,
            EdgeCurvatureAugmenter,
            TruncatedPeakAugmenter,
            EdgeArtifactsAugmenter,
            DETECTOR_MODELS,
        )

        assert DetectorRollOffAugmenter is not None
        assert StrayLightAugmenter is not None
        assert EdgeCurvatureAugmenter is not None
        assert TruncatedPeakAugmenter is not None
        assert EdgeArtifactsAugmenter is not None
        assert DETECTOR_MODELS is not None

    def test_import_from_augmentation_module(self):
        """Test that augmenters can be imported from augmentation module."""
        from nirs4all.operators.augmentation.edge_artifacts import (
            DetectorRollOffAugmenter,
            StrayLightAugmenter,
            EdgeCurvatureAugmenter,
            TruncatedPeakAugmenter,
            EdgeArtifactsAugmenter,
            DETECTOR_MODELS,
        )

        assert DetectorRollOffAugmenter is not None
        assert StrayLightAugmenter is not None
        assert EdgeCurvatureAugmenter is not None
        assert TruncatedPeakAugmenter is not None
        assert EdgeArtifactsAugmenter is not None
        assert DETECTOR_MODELS is not None

    def test_in_operators_all(self):
        """Test that augmenters are in operators.__all__."""
        from nirs4all.operators.augmentation import __all__ as aug_all

        assert "DetectorRollOffAugmenter" in aug_all
        assert "StrayLightAugmenter" in aug_all
        assert "EdgeCurvatureAugmenter" in aug_all
        assert "TruncatedPeakAugmenter" in aug_all
        assert "EdgeArtifactsAugmenter" in aug_all
        assert "DETECTOR_MODELS" in aug_all

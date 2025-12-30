"""
Unit tests for NIRBand, SpectralComponent, and ComponentLibrary classes.
"""

import pytest
import numpy as np

from nirs4all.data.synthetic import (
    NIRBand,
    SpectralComponent,
    ComponentLibrary,
    PREDEFINED_COMPONENTS,
    get_predefined_components,
)


class TestNIRBand:
    """Tests for NIRBand class."""

    def test_init_basic(self):
        """Test basic NIRBand initialization."""
        band = NIRBand(center=1450, sigma=25)
        assert band.center == 1450
        assert band.sigma == 25
        assert band.gamma == 0.0  # Default
        assert band.amplitude == 1.0  # Default
        assert band.name == ""  # Default

    def test_init_full(self, sample_band):
        """Test NIRBand with all parameters."""
        assert sample_band.center == 1450
        assert sample_band.sigma == 25
        assert sample_band.gamma == 3
        assert sample_band.amplitude == 0.8
        assert sample_band.name == "O-H 1st overtone"

    def test_compute_pure_gaussian(self, sample_wavelengths):
        """Test pure Gaussian band (gamma=0)."""
        band = NIRBand(center=1500, sigma=20, gamma=0, amplitude=1.0)
        spectrum = band.compute(sample_wavelengths)

        assert spectrum.shape == sample_wavelengths.shape
        assert np.all(np.isfinite(spectrum))

        # Peak should be at center
        peak_idx = np.argmax(spectrum)
        assert np.abs(sample_wavelengths[peak_idx] - 1500) < 5

        # Should be symmetric around center
        center_idx = np.searchsorted(sample_wavelengths, 1500)
        if center_idx > 0 and center_idx < len(spectrum) - 1:
            left = spectrum[center_idx - 1]
            right = spectrum[center_idx + 1]
            # Allow small tolerance due to grid sampling
            assert np.abs(left - right) < 0.1

    def test_compute_voigt_profile(self, sample_wavelengths, sample_band):
        """Test Voigt profile band (gamma > 0)."""
        spectrum = sample_band.compute(sample_wavelengths)

        assert spectrum.shape == sample_wavelengths.shape
        assert np.all(np.isfinite(spectrum))

        # Should have a peak near the center
        peak_idx = np.argmax(spectrum)
        assert np.abs(sample_wavelengths[peak_idx] - sample_band.center) < 10

    def test_compute_amplitude_scaling(self, sample_wavelengths):
        """Test that amplitude scales the spectrum."""
        band1 = NIRBand(center=1500, sigma=20, amplitude=1.0)
        band2 = NIRBand(center=1500, sigma=20, amplitude=2.0)

        spectrum1 = band1.compute(sample_wavelengths)
        spectrum2 = band2.compute(sample_wavelengths)

        np.testing.assert_allclose(spectrum2, spectrum1 * 2.0, rtol=1e-10)


class TestSpectralComponent:
    """Tests for SpectralComponent class."""

    def test_init_basic(self):
        """Test basic SpectralComponent initialization."""
        comp = SpectralComponent(name="test")
        assert comp.name == "test"
        assert comp.bands == []
        assert comp.correlation_group is None

    def test_init_with_bands(self, sample_component):
        """Test SpectralComponent with bands."""
        assert sample_component.name == "water"
        assert len(sample_component.bands) == 2
        assert sample_component.correlation_group == 1

    def test_compute_empty_bands(self, sample_wavelengths):
        """Test computing spectrum with no bands."""
        comp = SpectralComponent(name="empty")
        spectrum = comp.compute(sample_wavelengths)

        np.testing.assert_array_equal(spectrum, np.zeros_like(sample_wavelengths))

    def test_compute_single_band(self, sample_wavelengths):
        """Test computing spectrum with single band."""
        band = NIRBand(center=1450, sigma=25, amplitude=0.8)
        comp = SpectralComponent(name="single", bands=[band])
        spectrum = comp.compute(sample_wavelengths)

        expected = band.compute(sample_wavelengths)
        np.testing.assert_allclose(spectrum, expected)

    def test_compute_multiple_bands(self, sample_wavelengths, sample_component):
        """Test that multiple bands are summed."""
        spectrum = sample_component.compute(sample_wavelengths)

        # Manually compute expected sum
        expected = np.zeros_like(sample_wavelengths, dtype=np.float64)
        for band in sample_component.bands:
            expected += band.compute(sample_wavelengths)

        np.testing.assert_allclose(spectrum, expected)


class TestComponentLibrary:
    """Tests for ComponentLibrary class."""

    def test_init_empty(self):
        """Test empty library initialization."""
        library = ComponentLibrary(random_state=42)
        assert library.n_components == 0
        assert library.component_names == []

    def test_from_predefined_all(self):
        """Test loading all predefined components."""
        library = ComponentLibrary.from_predefined()
        assert library.n_components == len(get_predefined_components())
        assert "water" in library.component_names
        assert "protein" in library.component_names

    def test_from_predefined_subset(self, predefined_library):
        """Test loading subset of predefined components."""
        assert predefined_library.n_components == 3
        assert set(predefined_library.component_names) == {"water", "protein", "lipid"}

    def test_from_predefined_invalid_name(self):
        """Test error on invalid component name."""
        with pytest.raises(ValueError, match="Unknown predefined component"):
            ComponentLibrary.from_predefined(["water", "invalid_component"])

    def test_add_component(self, sample_component):
        """Test adding a component manually."""
        library = ComponentLibrary()
        library.add_component(sample_component)

        assert library.n_components == 1
        assert "water" in library

    def test_add_random_component(self):
        """Test generating random component."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_random_component("random_test", n_bands=4)

        assert comp.name == "random_test"
        assert len(comp.bands) == 4
        assert library.n_components == 1

    def test_generate_random_library(self, random_library):
        """Test generating random library."""
        assert random_library.n_components == 3
        # All components should have names
        for name in random_library.component_names:
            assert name.startswith("component_")

    def test_compute_all(self, predefined_library, sample_wavelengths):
        """Test computing all component spectra."""
        E = predefined_library.compute_all(sample_wavelengths)

        assert E.shape == (3, len(sample_wavelengths))
        assert np.all(np.isfinite(E))

    def test_getitem(self, predefined_library):
        """Test dictionary-style access."""
        water = predefined_library["water"]
        assert water.name == "water"

    def test_contains(self, predefined_library):
        """Test 'in' operator."""
        assert "water" in predefined_library
        assert "invalid" not in predefined_library

    def test_iter(self, predefined_library):
        """Test iteration over components."""
        components = list(predefined_library)
        assert len(components) == 3
        assert all(isinstance(c, SpectralComponent) for c in components)

    def test_len(self, predefined_library):
        """Test len() function."""
        assert len(predefined_library) == 3

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        lib1 = ComponentLibrary(random_state=42)
        lib1.generate_random_library(n_components=3)

        lib2 = ComponentLibrary(random_state=42)
        lib2.generate_random_library(n_components=3)

        # Component names should be same
        assert lib1.component_names == lib2.component_names

        # Band positions should be same
        wavelengths = np.arange(1000, 2500, 2)
        E1 = lib1.compute_all(wavelengths)
        E2 = lib2.compute_all(wavelengths)

        np.testing.assert_allclose(E1, E2)


class TestPredefinedComponents:
    """Tests for predefined components constant."""

    def test_predefined_components_exists(self):
        """Test that PREDEFINED_COMPONENTS is available."""
        assert PREDEFINED_COMPONENTS is not None
        assert len(PREDEFINED_COMPONENTS) > 0

    def test_get_predefined_components(self):
        """Test get_predefined_components function."""
        components = get_predefined_components()
        assert isinstance(components, dict)
        assert "water" in components
        assert "protein" in components
        assert "lipid" in components

    def test_predefined_water_component(self):
        """Test water component has expected properties."""
        water = get_predefined_components()["water"]
        assert water.name == "water"
        assert len(water.bands) >= 2
        # Water should have O-H bands around 1450 and 1940
        centers = [b.center for b in water.bands]
        assert any(1400 < c < 1500 for c in centers)
        assert any(1900 < c < 2000 for c in centers)

    def test_predefined_components_proxy_iteration(self):
        """Test that PREDEFINED_COMPONENTS supports iteration."""
        names = list(PREDEFINED_COMPONENTS)
        assert "water" in names

    def test_predefined_components_proxy_contains(self):
        """Test that PREDEFINED_COMPONENTS supports 'in' operator."""
        assert "water" in PREDEFINED_COMPONENTS
        assert "invalid" not in PREDEFINED_COMPONENTS

    def test_predefined_components_proxy_keys(self):
        """Test that PREDEFINED_COMPONENTS has keys() method."""
        keys = list(PREDEFINED_COMPONENTS.keys())
        assert "water" in keys

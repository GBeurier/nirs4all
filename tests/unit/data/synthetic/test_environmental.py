"""
Unit tests for the environmental module (Phase 3.1, 3.4).

Tests cover:
- Temperature effect configuration and parameters
- Temperature effect simulation
- Moisture/water activity effect simulation
- Combined environmental effects
- Convenience functions
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.synthetic.environmental import (
    # Enums
    SpectralRegion,
    # Dataclasses
    TemperatureEffectParams,
    TemperatureConfig,
    MoistureConfig,
    EnvironmentalEffectsConfig,
    # Constants
    TEMPERATURE_EFFECT_PARAMS,
    # Simulators
    TemperatureEffectSimulator,
    MoistureEffectSimulator,
    EnvironmentalEffectsSimulator,
    # Convenience functions
    apply_temperature_effects,
    apply_moisture_effects,
    simulate_temperature_series,
    get_temperature_effect_regions,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def wavelengths():
    """Standard NIR wavelength grid."""
    return np.arange(900, 2501, 2)


@pytest.fixture
def sample_spectra(wavelengths):
    """Sample synthetic spectra with water bands."""
    n_samples = 10
    n_wl = len(wavelengths)

    # Create base spectra with water-like features
    spectra = np.zeros((n_samples, n_wl))

    for i in range(n_samples):
        # Baseline
        spectra[i] = 0.3 + 0.0001 * (wavelengths - 1500)

        # Add water band at 1450 nm (O-H 1st overtone)
        water_band_1 = 0.5 * np.exp(-0.5 * ((wavelengths - 1450) / 30) ** 2)
        spectra[i] += water_band_1

        # Add water band at 1940 nm (O-H combination)
        water_band_2 = 0.8 * np.exp(-0.5 * ((wavelengths - 1940) / 25) ** 2)
        spectra[i] += water_band_2

        # Add some C-H band at 1700 nm
        ch_band = 0.3 * np.exp(-0.5 * ((wavelengths - 1700) / 40) ** 2)
        spectra[i] += ch_band

    return spectra


# ============================================================================
# Temperature Effect Tests
# ============================================================================

class TestSpectralRegion:
    """Tests for SpectralRegion enum."""

    def test_spectral_regions_defined(self):
        """Test that spectral regions are defined."""
        assert len(SpectralRegion) >= 6
        assert SpectralRegion.OH_FIRST_OVERTONE.value == "oh_1st_overtone"
        assert SpectralRegion.CH_FIRST_OVERTONE.value == "ch_1st_overtone"

    def test_spectral_regions_have_params(self):
        """Test that all regions have temperature effect parameters."""
        for region in SpectralRegion:
            assert region in TEMPERATURE_EFFECT_PARAMS


class TestTemperatureEffectParams:
    """Tests for TemperatureEffectParams dataclass."""

    def test_params_creation(self):
        """Test creating temperature effect parameters."""
        params = TemperatureEffectParams(
            wavelength_range=(1400, 1500),
            shift_per_degree=-0.3,
            intensity_change_per_degree=-0.002,
            broadening_per_degree=0.001,
            reference="Test reference"
        )

        assert params.wavelength_range == (1400, 1500)
        assert params.shift_per_degree == -0.3
        assert params.intensity_change_per_degree == -0.002

    def test_oh_region_has_negative_shift(self):
        """Test that O-H region has negative (blue) shift with temperature."""
        params = TEMPERATURE_EFFECT_PARAMS[SpectralRegion.OH_FIRST_OVERTONE]
        assert params.shift_per_degree < 0  # Blue shift

    def test_ch_region_has_smaller_shift(self):
        """Test that C-H region has smaller shift than O-H."""
        oh_params = TEMPERATURE_EFFECT_PARAMS[SpectralRegion.OH_FIRST_OVERTONE]
        ch_params = TEMPERATURE_EFFECT_PARAMS[SpectralRegion.CH_FIRST_OVERTONE]

        assert abs(ch_params.shift_per_degree) < abs(oh_params.shift_per_degree)


class TestTemperatureConfig:
    """Tests for TemperatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TemperatureConfig()
        assert config.reference_temperature == 25.0
        assert config.sample_temperature == 25.0
        assert config.enable_shift is True

    def test_delta_temperature_property(self):
        """Test delta_temperature calculation."""
        config = TemperatureConfig(
            reference_temperature=25.0,
            sample_temperature=40.0
        )
        assert config.delta_temperature == 15.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TemperatureConfig(
            sample_temperature=50.0,
            reference_temperature=25.0,
            temperature_variation=2.0,
            enable_broadening=False
        )
        assert config.sample_temperature == 50.0
        assert config.delta_temperature == 25.0
        assert config.enable_broadening is False


class TestTemperatureEffectSimulator:
    """Tests for TemperatureEffectSimulator class."""

    def test_simulator_creation(self):
        """Test creating a temperature simulator."""
        config = TemperatureConfig(sample_temperature=40.0)
        simulator = TemperatureEffectSimulator(config, random_state=42)

        assert simulator.config.sample_temperature == 40.0

    def test_apply_no_change_at_reference(self, wavelengths, sample_spectra):
        """Test that no change occurs at reference temperature."""
        config = TemperatureConfig(
            sample_temperature=25.0,
            reference_temperature=25.0
        )
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Should be identical when no temperature difference
        np.testing.assert_array_almost_equal(result, sample_spectra)

    def test_apply_changes_spectrum_at_high_temp(self, wavelengths, sample_spectra):
        """Test that spectrum changes at higher temperature."""
        config = TemperatureConfig(
            sample_temperature=50.0,
            reference_temperature=25.0
        )
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Spectra should be different
        assert not np.allclose(result, sample_spectra)

    def test_apply_with_per_sample_temperatures(self, wavelengths, sample_spectra):
        """Test applying different temperatures per sample."""
        config = TemperatureConfig(reference_temperature=25.0)
        simulator = TemperatureEffectSimulator(config, random_state=42)

        # Different temperatures for each sample
        temperatures = np.linspace(20, 50, len(sample_spectra))
        result = simulator.apply(sample_spectra, wavelengths, temperatures)

        # All samples should be affected differently
        assert result.shape == sample_spectra.shape

    def test_apply_with_temperature_variation(self, wavelengths, sample_spectra):
        """Test applying temperature with sample-to-sample variation."""
        config = TemperatureConfig(
            sample_temperature=40.0,
            reference_temperature=25.0,
            temperature_variation=5.0
        )
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Should produce varying results
        assert result.shape == sample_spectra.shape

    def test_disable_shift(self, wavelengths, sample_spectra):
        """Test disabling wavelength shift."""
        config = TemperatureConfig(
            sample_temperature=50.0,
            reference_temperature=25.0,
            enable_shift=False,
            enable_intensity=True,
            enable_broadening=False
        )
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Spectrum should still change (intensity)
        assert not np.allclose(result, sample_spectra)

    def test_reproducibility(self, wavelengths, sample_spectra):
        """Test that results are reproducible with same seed."""
        config = TemperatureConfig(
            sample_temperature=45.0,
            temperature_variation=3.0
        )

        sim1 = TemperatureEffectSimulator(config, random_state=42)
        sim2 = TemperatureEffectSimulator(config, random_state=42)

        result1 = sim1.apply(sample_spectra, wavelengths)
        result2 = sim2.apply(sample_spectra, wavelengths)

        np.testing.assert_array_equal(result1, result2)


# ============================================================================
# Moisture Effect Tests
# ============================================================================

class TestMoistureConfig:
    """Tests for MoistureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = MoistureConfig()
        assert config.water_activity == 0.5
        assert config.moisture_content == 0.10

    def test_validation_water_activity(self):
        """Test water activity validation."""
        with pytest.raises(ValueError):
            MoistureConfig(water_activity=1.5)

        with pytest.raises(ValueError):
            MoistureConfig(water_activity=-0.1)

    def test_validation_free_water_fraction(self):
        """Test free water fraction validation."""
        with pytest.raises(ValueError):
            MoistureConfig(free_water_fraction=1.5)

    def test_valid_config(self):
        """Test valid configuration."""
        config = MoistureConfig(
            water_activity=0.8,
            moisture_content=0.20,
            free_water_fraction=0.6
        )
        assert config.water_activity == 0.8
        assert config.moisture_content == 0.20


class TestMoistureEffectSimulator:
    """Tests for MoistureEffectSimulator class."""

    def test_simulator_creation(self):
        """Test creating a moisture simulator."""
        config = MoistureConfig(water_activity=0.7)
        simulator = MoistureEffectSimulator(config, random_state=42)

        assert simulator.config.water_activity == 0.7

    def test_apply_changes_spectrum(self, wavelengths, sample_spectra):
        """Test that moisture effects change spectrum."""
        config = MoistureConfig(
            water_activity=0.9,
            moisture_content=0.25
        )
        simulator = MoistureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Spectrum should be modified
        assert not np.allclose(result, sample_spectra)

    def test_high_vs_low_water_activity(self, wavelengths, sample_spectra):
        """Test difference between high and low water activity."""
        config_low = MoistureConfig(water_activity=0.2)
        config_high = MoistureConfig(water_activity=0.9)

        sim_low = MoistureEffectSimulator(config_low, random_state=42)
        sim_high = MoistureEffectSimulator(config_high, random_state=42)

        result_low = sim_low.apply(sample_spectra, wavelengths)
        result_high = sim_high.apply(sample_spectra, wavelengths)

        # Results should be different
        assert not np.allclose(result_low, result_high)

    def test_get_water_band_positions(self):
        """Test getting water band positions."""
        config = MoistureConfig()
        simulator = MoistureEffectSimulator(config, random_state=42)

        positions = simulator.get_water_band_positions(free_fraction=0.5)

        assert "1st_overtone" in positions
        assert "combination" in positions
        assert len(positions["1st_overtone"]) == 2

    def test_per_sample_water_activity(self, wavelengths, sample_spectra):
        """Test applying different water activities per sample."""
        config = MoistureConfig()
        simulator = MoistureEffectSimulator(config, random_state=42)

        water_activities = np.linspace(0.2, 0.9, len(sample_spectra))
        result = simulator.apply(sample_spectra, wavelengths, water_activities)

        assert result.shape == sample_spectra.shape


# ============================================================================
# Combined Environmental Effects Tests
# ============================================================================

class TestEnvironmentalEffectsConfig:
    """Tests for EnvironmentalEffectsConfig dataclass."""

    def test_default_config(self):
        """Test default combined configuration."""
        config = EnvironmentalEffectsConfig()
        assert config.enable_temperature is True
        assert config.enable_moisture is True
        assert config.temperature is not None
        assert config.moisture is not None

    def test_custom_config(self):
        """Test custom combined configuration."""
        config = EnvironmentalEffectsConfig(
            temperature=TemperatureConfig(sample_temperature=40.0),
            moisture=MoistureConfig(water_activity=0.8),
            enable_temperature=True,
            enable_moisture=False
        )
        assert config.temperature.sample_temperature == 40.0
        assert config.enable_moisture is False


class TestEnvironmentalEffectsSimulator:
    """Tests for EnvironmentalEffectsSimulator class."""

    def test_simulator_creation(self):
        """Test creating combined simulator."""
        config = EnvironmentalEffectsConfig()
        simulator = EnvironmentalEffectsSimulator(config, random_state=42)

        assert simulator.temperature_sim is not None
        assert simulator.moisture_sim is not None

    def test_apply_both_effects(self, wavelengths, sample_spectra):
        """Test applying both temperature and moisture effects."""
        config = EnvironmentalEffectsConfig(
            temperature=TemperatureConfig(sample_temperature=45.0),
            moisture=MoistureConfig(water_activity=0.8)
        )
        simulator = EnvironmentalEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_disable_temperature(self, wavelengths, sample_spectra):
        """Test disabling temperature effects."""
        config = EnvironmentalEffectsConfig(
            temperature=TemperatureConfig(sample_temperature=60.0),
            enable_temperature=False,
            enable_moisture=True
        )
        simulator = EnvironmentalEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Should only have moisture effects
        assert result.shape == sample_spectra.shape

    def test_disable_moisture(self, wavelengths, sample_spectra):
        """Test disabling moisture effects."""
        config = EnvironmentalEffectsConfig(
            moisture=MoistureConfig(water_activity=0.9),
            enable_temperature=True,
            enable_moisture=False
        )
        simulator = EnvironmentalEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_apply_temperature_effects(self, wavelengths, sample_spectra):
        """Test apply_temperature_effects function."""
        result = apply_temperature_effects(
            sample_spectra, wavelengths,
            temperature=40.0,
            reference_temperature=25.0,
            random_state=42
        )

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_apply_moisture_effects(self, wavelengths, sample_spectra):
        """Test apply_moisture_effects function."""
        result = apply_moisture_effects(
            sample_spectra, wavelengths,
            water_activity=0.8,
            moisture_content=0.15,
            random_state=42
        )

        assert result.shape == sample_spectra.shape

    def test_simulate_temperature_series(self, wavelengths, sample_spectra):
        """Test simulate_temperature_series function."""
        single_spectrum = sample_spectra[0]
        temperatures = [20, 25, 30, 35, 40]

        result = simulate_temperature_series(
            single_spectrum, wavelengths, temperatures,
            reference_temperature=25.0,
            random_state=42
        )

        assert result.shape == (len(temperatures), len(wavelengths))

    def test_get_temperature_effect_regions(self):
        """Test get_temperature_effect_regions function."""
        regions = get_temperature_effect_regions()

        assert len(regions) > 0
        assert "oh_1st_overtone" in regions
        assert "ch_1st_overtone" in regions

        # Check wavelength ranges are tuples
        for name, wl_range in regions.items():
            assert len(wl_range) == 2
            assert wl_range[0] < wl_range[1]


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_spectra(self, wavelengths):
        """Test with empty spectra array."""
        empty_spectra = np.zeros((0, len(wavelengths)))

        config = TemperatureConfig(sample_temperature=40.0)
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(empty_spectra, wavelengths)
        assert result.shape == (0, len(wavelengths))

    def test_single_spectrum(self, wavelengths):
        """Test with single spectrum."""
        single_spectrum = np.random.randn(1, len(wavelengths)) * 0.1 + 0.5

        config = TemperatureConfig(sample_temperature=40.0)
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(single_spectrum, wavelengths)
        assert result.shape == (1, len(wavelengths))

    def test_extreme_temperature(self, wavelengths, sample_spectra):
        """Test with extreme temperature values."""
        config = TemperatureConfig(
            sample_temperature=100.0,  # Very high
            reference_temperature=25.0
        )
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Should still produce valid output
        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))

    def test_short_wavelength_range(self):
        """Test with short wavelength range."""
        short_wl = np.arange(1400, 1500, 5)
        spectra = np.random.randn(5, len(short_wl)) * 0.1 + 0.5

        config = TemperatureConfig(sample_temperature=40.0)
        simulator = TemperatureEffectSimulator(config, random_state=42)

        result = simulator.apply(spectra, short_wl)
        assert result.shape == spectra.shape

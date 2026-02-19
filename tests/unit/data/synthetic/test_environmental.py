"""
Unit tests for environmental module configuration classes.

Tests cover:
- SpectralRegion enum
- Temperature effect parameters and configuration
- Moisture configuration with validation
- Combined environmental effects configuration
"""

from __future__ import annotations

import pytest

from nirs4all.synthesis.environmental import (
    TEMPERATURE_EFFECT_PARAMS,
    EnvironmentalEffectsConfig,
    MoistureConfig,
    SpectralRegion,
    TemperatureConfig,
    TemperatureEffectParams,
    get_temperature_effect_regions,
)


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

class TestGetTemperatureEffectRegions:
    """Tests for get_temperature_effect_regions function."""

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

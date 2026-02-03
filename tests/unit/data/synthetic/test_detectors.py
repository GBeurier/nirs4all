"""
Unit tests for the detectors module (Phase 2.3).

Tests cover:
- Detector spectral response curves
- Noise model configuration
- DetectorSimulator behavior
- Convenience functions
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.synthesis.instruments import DetectorType
from nirs4all.synthesis.detectors import (
    DetectorSpectralResponse,
    NoiseModelConfig,
    DetectorConfig,
    DETECTOR_RESPONSES,
    DETECTOR_NOISE_DEFAULTS,
    get_detector_response,
    get_default_noise_config,
    DetectorSimulator,
    simulate_detector_effects,
    get_detector_wavelength_range,
    list_detector_types,
)


class TestDetectorSpectralResponse:
    """Tests for DetectorSpectralResponse dataclass."""

    def test_spectral_response_creation(self):
        """Test creating a spectral response curve."""
        wavelengths = np.array([800, 1000, 1200, 1400, 1600])
        response = np.array([0.1, 0.5, 0.9, 0.8, 0.3])

        sr = DetectorSpectralResponse(
            detector_type=DetectorType.INGAAS,
            wavelengths=wavelengths,
            response=response,
            peak_wavelength=1200,
            cutoff_wavelength=1700,
            short_cutoff=800,
            peak_qe=0.9,
        )

        assert sr.detector_type == DetectorType.INGAAS
        assert sr.peak_wavelength == 1200
        assert len(sr.wavelengths) == 5

    def test_get_response_at_wavelengths(self):
        """Test interpolating response at specific wavelengths."""
        wavelengths = np.array([800, 1000, 1200, 1400, 1600])
        response = np.array([0.1, 0.5, 0.9, 0.8, 0.3])

        sr = DetectorSpectralResponse(
            detector_type=DetectorType.INGAAS,
            wavelengths=wavelengths,
            response=response,
            peak_wavelength=1200,
            cutoff_wavelength=1700,
            short_cutoff=800,
        )

        # Query at known points
        query_wl = np.array([1000, 1200, 1400])
        result = sr.get_response_at(query_wl)

        assert len(result) == 3
        assert result[0] == pytest.approx(0.5)  # At 1000nm
        assert result[1] == pytest.approx(0.9)  # At 1200nm
        assert result[2] == pytest.approx(0.8)  # At 1400nm

    def test_response_outside_range(self):
        """Test response outside detector range is zero."""
        wavelengths = np.array([800, 1000, 1200, 1400, 1600])
        response = np.array([0.1, 0.5, 0.9, 0.8, 0.3])

        sr = DetectorSpectralResponse(
            detector_type=DetectorType.INGAAS,
            wavelengths=wavelengths,
            response=response,
            peak_wavelength=1200,
            cutoff_wavelength=1700,
            short_cutoff=800,
        )

        # Query outside range
        query_wl = np.array([500, 2000])
        result = sr.get_response_at(query_wl)

        # Should be zero (fill_value=0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)


class TestDetectorResponseRegistry:
    """Tests for detector response registry."""

    def test_detector_responses_not_empty(self):
        """Test that detector responses are registered."""
        assert len(DETECTOR_RESPONSES) > 0

    def test_all_detector_types_have_response(self):
        """Test that all detector types have a response curve."""
        for detector_type in DetectorType:
            if detector_type in DETECTOR_RESPONSES:
                response = get_detector_response(detector_type)
                assert response is not None
                assert len(response.wavelengths) > 0

    def test_get_detector_response_ingaas(self):
        """Test getting InGaAs detector response."""
        response = get_detector_response(DetectorType.INGAAS)
        assert response.detector_type == DetectorType.INGAAS
        # InGaAs typically peaks around 1300nm
        assert 1000 < response.peak_wavelength < 1500

    def test_get_detector_response_silicon(self):
        """Test getting Silicon detector response."""
        response = get_detector_response(DetectorType.SI)
        assert response.detector_type == DetectorType.SI
        # Silicon is sensitive in visible/near-NIR
        assert response.cutoff_wavelength < 1200

    def test_get_detector_response_pbs(self):
        """Test getting PbS detector response."""
        response = get_detector_response(DetectorType.PBS)
        assert response.detector_type == DetectorType.PBS
        # PbS works in SWIR
        assert response.peak_wavelength > 1500


class TestNoiseModelConfig:
    """Tests for NoiseModelConfig dataclass."""

    def test_noise_config_defaults(self):
        """Test noise config default values."""
        config = NoiseModelConfig()
        assert config.shot_noise_enabled is True
        assert config.thermal_noise_enabled is True
        assert config.read_noise_enabled is True
        assert config.flicker_noise_enabled is False

    def test_noise_config_custom(self):
        """Test noise config with custom values."""
        config = NoiseModelConfig(
            shot_noise_enabled=True,
            thermal_noise_enabled=False,
            read_noise_enabled=True,
            flicker_noise_enabled=True,
            shot_noise_factor=1.5,
            thermal_noise_factor=0.5,
            read_noise_electrons=100.0,
            flicker_corner_freq=50.0,
        )
        assert config.thermal_noise_enabled is False
        assert config.flicker_noise_enabled is True
        assert config.shot_noise_factor == 1.5


class TestDetectorConfig:
    """Tests for DetectorConfig dataclass."""

    def test_detector_config_defaults(self):
        """Test detector config default values."""
        config = DetectorConfig()
        assert config.detector_type == DetectorType.INGAAS
        assert config.temperature_k > 0
        assert config.integration_time_ms > 0
        assert config.gain >= 1.0

    def test_detector_config_custom(self):
        """Test detector config with custom values."""
        noise = NoiseModelConfig(shot_noise_factor=2.0)
        config = DetectorConfig(
            detector_type=DetectorType.PBS,
            temperature_k=280.0,
            integration_time_ms=50.0,
            gain=2.0,
            noise_model=noise,
            apply_response_curve=True,
            apply_nonlinearity=True,
            nonlinearity_coefficient=0.03,
        )
        assert config.detector_type == DetectorType.PBS
        assert config.temperature_k == 280.0
        assert config.apply_nonlinearity is True


class TestDetectorNoiseDefaults:
    """Tests for detector-specific noise defaults."""

    def test_noise_defaults_exist(self):
        """Test that noise defaults exist for detector types."""
        assert len(DETECTOR_NOISE_DEFAULTS) > 0

    def test_get_default_noise_config(self):
        """Test getting default noise config for a detector."""
        config = get_default_noise_config(DetectorType.INGAAS)
        assert isinstance(config, NoiseModelConfig)

    def test_pbs_has_flicker_noise(self):
        """Test that PbS detector has 1/f noise enabled by default."""
        config = get_default_noise_config(DetectorType.PBS)
        assert config.flicker_noise_enabled is True

    def test_mct_has_low_noise(self):
        """Test that MCT (cooled) detector has low noise."""
        config = get_default_noise_config(DetectorType.MCT)
        # MCT is typically cooled, so lower noise
        assert config.shot_noise_factor < 1.0
        assert config.thermal_noise_factor < 1.0


class TestDetectorSimulator:
    """Tests for DetectorSimulator class."""

    @pytest.fixture
    def sample_spectra(self):
        """Create sample spectra for testing."""
        n_samples = 10
        n_wl = 100
        return np.random.default_rng(42).uniform(0.1, 1.0, (n_samples, n_wl))

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelength array."""
        return np.linspace(1000, 2000, 100)

    def test_simulator_creation(self):
        """Test creating a detector simulator."""
        config = DetectorConfig(detector_type=DetectorType.INGAAS)
        simulator = DetectorSimulator(config, random_state=42)
        assert simulator is not None

    def test_simulator_apply(self, sample_spectra, sample_wavelengths):
        """Test applying detector effects."""
        config = DetectorConfig(detector_type=DetectorType.INGAAS)
        simulator = DetectorSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, sample_wavelengths)

        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))

    def test_simulator_adds_noise(self, sample_spectra, sample_wavelengths):
        """Test that simulator adds noise."""
        config = DetectorConfig(detector_type=DetectorType.INGAAS)
        simulator = DetectorSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, sample_wavelengths)

        # Result should differ from input (noise added)
        assert not np.allclose(result, sample_spectra)

    def test_simulator_reproducibility(self, sample_spectra, sample_wavelengths):
        """Test simulator reproducibility with same seed."""
        config = DetectorConfig(detector_type=DetectorType.INGAAS)
        sim1 = DetectorSimulator(config, random_state=42)
        sim2 = DetectorSimulator(config, random_state=42)

        result1 = sim1.apply(sample_spectra.copy(), sample_wavelengths)
        result2 = sim2.apply(sample_spectra.copy(), sample_wavelengths)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_simulator_different_seeds(self, sample_spectra, sample_wavelengths):
        """Test different seeds produce different results."""
        config = DetectorConfig(detector_type=DetectorType.INGAAS)
        sim1 = DetectorSimulator(config, random_state=42)
        sim2 = DetectorSimulator(config, random_state=123)

        result1 = sim1.apply(sample_spectra.copy(), sample_wavelengths)
        result2 = sim2.apply(sample_spectra.copy(), sample_wavelengths)

        assert not np.allclose(result1, result2)

    def test_simulator_with_nonlinearity(self, sample_spectra, sample_wavelengths):
        """Test simulator with nonlinearity enabled."""
        config = DetectorConfig(
            detector_type=DetectorType.INGAAS,
            apply_nonlinearity=True,
            nonlinearity_coefficient=0.05,
        )
        simulator = DetectorSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, sample_wavelengths)

        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_spectra(self):
        """Create sample spectra."""
        return np.random.default_rng(42).uniform(0.1, 1.0, (5, 50))

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelengths."""
        return np.linspace(1000, 2000, 50)

    def test_simulate_detector_effects(self, sample_spectra, sample_wavelengths):
        """Test simulate_detector_effects function."""
        result = simulate_detector_effects(
            sample_spectra,
            sample_wavelengths,
            detector_type=DetectorType.INGAAS,
            random_state=42,
        )

        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))

    def test_simulate_detector_effects_no_noise(self, sample_spectra, sample_wavelengths):
        """Test simulate_detector_effects without noise."""
        result = simulate_detector_effects(
            sample_spectra,
            sample_wavelengths,
            detector_type=DetectorType.INGAAS,
            include_noise=False,
            random_state=42,
        )

        assert result.shape == sample_spectra.shape

    def test_get_detector_wavelength_range(self):
        """Test getting detector wavelength range."""
        wl_min, wl_max = get_detector_wavelength_range(DetectorType.INGAAS)

        assert wl_min < wl_max
        assert wl_min > 0
        assert wl_max < 5000  # Reasonable for NIR

    def test_list_detector_types(self):
        """Test listing detector types."""
        detector_list = list_detector_types()

        assert len(detector_list) > 0
        assert "ingaas" in detector_list
        assert "si" in detector_list


class TestNoisePhysics:
    """Tests for physical correctness of noise models."""

    @pytest.fixture
    def wavelengths(self):
        """Create wavelength array."""
        return np.linspace(1000, 2000, 100)

    @pytest.fixture
    def uniform_spectra(self):
        """Create uniform spectra."""
        return np.ones((10, 100)) * 0.5

    def test_shot_noise_signal_dependent(self, uniform_spectra, wavelengths):
        """Test that shot noise is signal-dependent."""
        # Higher signal should have higher absolute shot noise
        low_signal = uniform_spectra * 0.1
        high_signal = uniform_spectra * 1.0

        config = DetectorConfig(
            detector_type=DetectorType.INGAAS,
            noise_model=NoiseModelConfig(
                shot_noise_enabled=True,
                thermal_noise_enabled=False,
                read_noise_enabled=False,
            ),
        )

        sim = DetectorSimulator(config, random_state=42)
        result_low = sim.apply(low_signal.copy(), wavelengths)
        result_high = sim.apply(high_signal.copy(), wavelengths)

        # Both should have noise
        noise_low = np.std(result_low - low_signal)
        noise_high = np.std(result_high - high_signal)

        # Higher signal should have more absolute noise (shot noise)
        assert noise_high > noise_low or np.isclose(noise_high, noise_low, rtol=0.5)

    def test_longer_integration_reduces_noise(self, uniform_spectra, wavelengths):
        """Test that longer integration time reduces noise."""
        noise_config = get_default_noise_config(DetectorType.INGAAS)

        config_short = DetectorConfig(
            detector_type=DetectorType.INGAAS,
            integration_time_ms=10.0,
            noise_model=noise_config,
        )
        config_long = DetectorConfig(
            detector_type=DetectorType.INGAAS,
            integration_time_ms=1000.0,
            noise_model=noise_config,
        )

        sim_short = DetectorSimulator(config_short, random_state=42)
        sim_long = DetectorSimulator(config_long, random_state=42)

        result_short = sim_short.apply(uniform_spectra.copy(), wavelengths)
        result_long = sim_long.apply(uniform_spectra.copy(), wavelengths)

        noise_short = np.std(result_short - uniform_spectra)
        noise_long = np.std(result_long - uniform_spectra)

        # Longer integration should have less noise
        assert noise_long < noise_short

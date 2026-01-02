"""
Unit tests for the scattering module (Phase 3.2, 3.3).

Tests cover:
- Particle size distribution and configuration
- Particle size effect simulation
- EMSC-style transformation
- Scattering coefficient generation
- Combined scattering effects
- Convenience functions
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.synthetic.scattering import (
    # Enums
    ScatteringModel,
    # Dataclasses
    ParticleSizeDistribution,
    ParticleSizeConfig,
    EMSCConfig,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
    # Simulators
    ParticleSizeSimulator,
    EMSCTransformSimulator,
    ScatteringCoefficientGenerator,
    ScatteringEffectsSimulator,
    # Convenience functions
    apply_particle_size_effects,
    apply_emsc_distortion,
    generate_scattering_coefficients,
    simulate_snv_correctable_scatter,
    simulate_msc_correctable_scatter,
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
    """Sample synthetic spectra."""
    n_samples = 10
    n_wl = len(wavelengths)

    rng = np.random.default_rng(42)

    # Create realistic-looking spectra
    spectra = np.zeros((n_samples, n_wl))

    for i in range(n_samples):
        # Baseline with slight slope
        spectra[i] = 0.3 + 0.00015 * (wavelengths - 1500)

        # Add absorption bands
        for center in [1200, 1450, 1700, 1940, 2200]:
            width = rng.uniform(25, 45)
            height = rng.uniform(0.2, 0.6)
            band = height * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            spectra[i] += band

    return spectra


# ============================================================================
# Particle Size Distribution Tests
# ============================================================================

class TestParticleSizeDistribution:
    """Tests for ParticleSizeDistribution dataclass."""

    def test_default_distribution(self):
        """Test default distribution parameters."""
        dist = ParticleSizeDistribution()
        assert dist.mean_size_um == 50.0
        assert dist.std_size_um == 15.0
        assert dist.distribution == "lognormal"

    def test_sample_lognormal(self):
        """Test sampling from lognormal distribution."""
        dist = ParticleSizeDistribution(
            mean_size_um=50.0,
            std_size_um=15.0,
            distribution="lognormal"
        )
        rng = np.random.default_rng(42)

        sizes = dist.sample(100, rng)

        assert len(sizes) == 100
        assert np.all(sizes >= dist.min_size_um)
        assert np.all(sizes <= dist.max_size_um)

    def test_sample_normal(self):
        """Test sampling from normal distribution."""
        dist = ParticleSizeDistribution(
            mean_size_um=50.0,
            std_size_um=10.0,
            distribution="normal"
        )
        rng = np.random.default_rng(42)

        sizes = dist.sample(100, rng)

        assert len(sizes) == 100
        assert np.all(sizes >= dist.min_size_um)

    def test_sample_uniform(self):
        """Test sampling from uniform distribution."""
        dist = ParticleSizeDistribution(
            min_size_um=20.0,
            max_size_um=100.0,
            distribution="uniform"
        )
        rng = np.random.default_rng(42)

        sizes = dist.sample(100, rng)

        assert len(sizes) == 100
        assert np.all(sizes >= 20.0)
        assert np.all(sizes <= 100.0)

    def test_sample_clipping(self):
        """Test that sizes are clipped to valid range."""
        dist = ParticleSizeDistribution(
            mean_size_um=10.0,  # Close to min
            std_size_um=20.0,  # Large std
            min_size_um=5.0,
            max_size_um=200.0,
            distribution="normal"
        )
        rng = np.random.default_rng(42)

        sizes = dist.sample(1000, rng)

        assert np.all(sizes >= 5.0)
        assert np.all(sizes <= 200.0)


# ============================================================================
# Particle Size Config Tests
# ============================================================================

class TestParticleSizeConfig:
    """Tests for ParticleSizeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ParticleSizeConfig()
        assert config.reference_size_um == 50.0
        assert config.size_effect_strength == 1.0
        assert config.wavelength_exponent == 1.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParticleSizeConfig(
            distribution=ParticleSizeDistribution(mean_size_um=30.0),
            reference_size_um=40.0,
            wavelength_exponent=2.0
        )
        assert config.distribution.mean_size_um == 30.0
        assert config.reference_size_um == 40.0


# ============================================================================
# Particle Size Simulator Tests
# ============================================================================

class TestParticleSizeSimulator:
    """Tests for ParticleSizeSimulator class."""

    def test_simulator_creation(self):
        """Test creating particle size simulator."""
        config = ParticleSizeConfig()
        simulator = ParticleSizeSimulator(config, random_state=42)

        assert simulator.config is not None

    def test_apply_changes_spectra(self, wavelengths, sample_spectra):
        """Test that particle size effects change spectra."""
        config = ParticleSizeConfig(
            distribution=ParticleSizeDistribution(mean_size_um=30.0)
        )
        simulator = ParticleSizeSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_apply_with_custom_sizes(self, wavelengths, sample_spectra):
        """Test applying with custom particle sizes."""
        config = ParticleSizeConfig()
        simulator = ParticleSizeSimulator(config, random_state=42)

        # Custom sizes for each sample
        sizes = np.linspace(20, 100, len(sample_spectra))
        result = simulator.apply(sample_spectra, wavelengths, sizes)

        assert result.shape == sample_spectra.shape

    def test_smaller_particles_more_scattering(self, wavelengths, sample_spectra):
        """Test that smaller particles produce more scattering."""
        config = ParticleSizeConfig()
        simulator = ParticleSizeSimulator(config, random_state=42)

        # Small particles
        small_sizes = np.full(len(sample_spectra), 20.0)
        result_small = simulator.apply(sample_spectra.copy(), wavelengths, small_sizes)

        # Large particles
        large_sizes = np.full(len(sample_spectra), 100.0)
        result_large = simulator.apply(sample_spectra.copy(), wavelengths, large_sizes)

        # Results should be different
        assert not np.allclose(result_small, result_large)

    def test_generate_particle_sizes(self):
        """Test particle size generation."""
        config = ParticleSizeConfig(
            distribution=ParticleSizeDistribution(
                mean_size_um=40.0,
                std_size_um=10.0
            )
        )
        simulator = ParticleSizeSimulator(config, random_state=42)

        sizes = simulator.generate_particle_sizes(100)

        assert len(sizes) == 100
        # Should be around mean (within a few std devs)
        assert 20 < np.mean(sizes) < 80

    def test_reproducibility(self, wavelengths, sample_spectra):
        """Test reproducibility with same seed."""
        config = ParticleSizeConfig()

        sim1 = ParticleSizeSimulator(config, random_state=42)
        sim2 = ParticleSizeSimulator(config, random_state=42)

        result1 = sim1.apply(sample_spectra.copy(), wavelengths)
        result2 = sim2.apply(sample_spectra.copy(), wavelengths)

        np.testing.assert_array_equal(result1, result2)


# ============================================================================
# EMSC Transform Tests
# ============================================================================

class TestEMSCConfig:
    """Tests for EMSCConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = EMSCConfig()
        assert config.polynomial_order == 2
        assert config.multiplicative_scatter_std == 0.15
        assert config.include_wavelength_terms is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EMSCConfig(
            polynomial_order=3,
            multiplicative_scatter_std=0.2,
            additive_scatter_std=0.1
        )
        assert config.polynomial_order == 3
        assert config.multiplicative_scatter_std == 0.2


class TestEMSCTransformSimulator:
    """Tests for EMSCTransformSimulator class."""

    def test_simulator_creation(self):
        """Test creating EMSC simulator."""
        config = EMSCConfig()
        simulator = EMSCTransformSimulator(config, random_state=42)

        assert simulator.config is not None

    def test_apply_changes_spectra(self, wavelengths, sample_spectra):
        """Test that EMSC transform changes spectra."""
        config = EMSCConfig()
        simulator = EMSCTransformSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_apply_with_reference(self, wavelengths, sample_spectra):
        """Test applying with explicit reference spectrum."""
        config = EMSCConfig()
        simulator = EMSCTransformSimulator(config, random_state=42)

        reference = np.mean(sample_spectra, axis=0)
        result = simulator.apply(sample_spectra, wavelengths, reference)

        assert result.shape == sample_spectra.shape

    def test_higher_polynomial_order(self, wavelengths, sample_spectra):
        """Test with higher polynomial order."""
        config = EMSCConfig(polynomial_order=4)
        simulator = EMSCTransformSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape

    def test_get_emsc_basis(self, wavelengths):
        """Test getting EMSC polynomial basis."""
        config = EMSCConfig(polynomial_order=2)
        simulator = EMSCTransformSimulator(config, random_state=42)

        basis = simulator.get_emsc_basis(wavelengths)

        assert basis.shape == (len(wavelengths), 3)  # constant + 2 polynomial terms
        np.testing.assert_array_equal(basis[:, 0], 1.0)  # Constant term

    def test_no_wavelength_terms(self, wavelengths, sample_spectra):
        """Test without wavelength polynomial terms."""
        config = EMSCConfig(include_wavelength_terms=False)
        simulator = EMSCTransformSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        # Should only have multiplicative and additive terms
        assert result.shape == sample_spectra.shape

    def test_reproducibility(self, wavelengths, sample_spectra):
        """Test reproducibility."""
        config = EMSCConfig()

        sim1 = EMSCTransformSimulator(config, random_state=42)
        sim2 = EMSCTransformSimulator(config, random_state=42)

        result1 = sim1.apply(sample_spectra, wavelengths)
        result2 = sim2.apply(sample_spectra, wavelengths)

        np.testing.assert_array_equal(result1, result2)


# ============================================================================
# Scattering Coefficient Generator Tests
# ============================================================================

class TestScatteringCoefficientConfig:
    """Tests for ScatteringCoefficientConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScatteringCoefficientConfig()
        assert config.baseline_scattering == 1.0
        assert config.wavelength_exponent == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScatteringCoefficientConfig(
            baseline_scattering=1.5,
            wavelength_exponent=2.0,
            sample_variation=0.2
        )
        assert config.baseline_scattering == 1.5


class TestScatteringCoefficientGenerator:
    """Tests for ScatteringCoefficientGenerator class."""

    def test_generator_creation(self):
        """Test creating scattering coefficient generator."""
        config = ScatteringCoefficientConfig()
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        assert generator.config is not None

    def test_generate_coefficients(self, wavelengths):
        """Test generating scattering coefficients."""
        config = ScatteringCoefficientConfig()
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        S = generator.generate(50, wavelengths)

        assert S.shape == (50, len(wavelengths))
        assert np.all(S > 0)  # Should be positive

    def test_wavelength_dependence(self, wavelengths):
        """Test wavelength dependence of scattering."""
        config = ScatteringCoefficientConfig(wavelength_exponent=1.5)
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        S = generator.generate(10, wavelengths)

        # Average scattering should decrease with wavelength
        mean_S = np.mean(S, axis=0)
        # Check that short wavelengths have higher scattering
        short_wl_S = mean_S[:len(wavelengths)//4]
        long_wl_S = mean_S[-len(wavelengths)//4:]
        assert np.mean(short_wl_S) > np.mean(long_wl_S)

    def test_particle_size_effect(self, wavelengths):
        """Test particle size effect on scattering."""
        config = ScatteringCoefficientConfig()
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        # Smaller particles
        small_sizes = np.full(10, 20.0)
        S_small = generator.generate(10, wavelengths, small_sizes)

        # Larger particles
        large_sizes = np.full(10, 100.0)
        S_large = generator.generate(10, wavelengths, large_sizes)

        # Smaller particles should have higher scattering
        assert np.mean(S_small) > np.mean(S_large)

    def test_generate_for_particle_sizes(self, wavelengths):
        """Test generating for specific particle sizes."""
        config = ScatteringCoefficientConfig()
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        sizes = np.array([20, 30, 50, 80, 100])
        S = generator.generate_for_particle_sizes(sizes, wavelengths)

        assert S.shape == (len(sizes), len(wavelengths))

    def test_sample_variation(self, wavelengths):
        """Test sample-to-sample variation in scattering."""
        config = ScatteringCoefficientConfig(sample_variation=0.3)
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        S = generator.generate(100, wavelengths)

        # Check that there's variation between samples
        sample_means = np.mean(S, axis=1)
        assert np.std(sample_means) > 0


# ============================================================================
# Combined Scattering Effects Tests
# ============================================================================

class TestScatteringEffectsConfig:
    """Tests for ScatteringEffectsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScatteringEffectsConfig()
        assert config.model == ScatteringModel.EMSC
        assert config.enable_particle_size is True
        assert config.enable_emsc is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScatteringEffectsConfig(
            model=ScatteringModel.RAYLEIGH,
            enable_particle_size=False
        )
        assert config.model == ScatteringModel.RAYLEIGH
        assert config.enable_particle_size is False


class TestScatteringEffectsSimulator:
    """Tests for ScatteringEffectsSimulator class."""

    def test_simulator_creation(self):
        """Test creating combined simulator."""
        config = ScatteringEffectsConfig()
        simulator = ScatteringEffectsSimulator(config, random_state=42)

        assert simulator.particle_sim is not None
        assert simulator.emsc_sim is not None
        assert simulator.scatter_gen is not None

    def test_apply_all_effects(self, wavelengths, sample_spectra):
        """Test applying all scattering effects."""
        config = ScatteringEffectsConfig()
        simulator = ScatteringEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_disable_particle_size(self, wavelengths, sample_spectra):
        """Test disabling particle size effects."""
        config = ScatteringEffectsConfig(
            enable_particle_size=False,
            enable_emsc=True
        )
        simulator = ScatteringEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape

    def test_disable_emsc(self, wavelengths, sample_spectra):
        """Test disabling EMSC effects."""
        config = ScatteringEffectsConfig(
            enable_particle_size=True,
            enable_emsc=False
        )
        simulator = ScatteringEffectsSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape

    def test_generate_scattering_coefficients(self, wavelengths):
        """Test generating scattering coefficients."""
        config = ScatteringEffectsConfig()
        simulator = ScatteringEffectsSimulator(config, random_state=42)

        S = simulator.generate_scattering_coefficients(50, wavelengths)

        assert S.shape == (50, len(wavelengths))


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_apply_particle_size_effects(self, wavelengths, sample_spectra):
        """Test apply_particle_size_effects function."""
        result = apply_particle_size_effects(
            sample_spectra, wavelengths,
            mean_particle_size_um=30.0,
            size_variation=10.0,
            random_state=42
        )

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_apply_emsc_distortion(self, wavelengths, sample_spectra):
        """Test apply_emsc_distortion function."""
        result = apply_emsc_distortion(
            sample_spectra, wavelengths,
            multiplicative_std=0.2,
            additive_std=0.08,
            random_state=42
        )

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_generate_scattering_coefficients_function(self, wavelengths):
        """Test generate_scattering_coefficients function."""
        S = generate_scattering_coefficients(
            50, wavelengths,
            baseline_scattering=1.5,
            wavelength_exponent=1.2,
            random_state=42
        )

        assert S.shape == (50, len(wavelengths))
        assert np.all(S > 0)

    def test_generate_with_particle_sizes(self, wavelengths):
        """Test generating scattering coefficients with particle sizes."""
        sizes = np.linspace(20, 100, 20)
        S = generate_scattering_coefficients(
            len(sizes), wavelengths,
            particle_sizes=sizes,
            random_state=42
        )

        assert S.shape == (len(sizes), len(wavelengths))

    def test_simulate_snv_correctable_scatter(self, sample_spectra):
        """Test simulate_snv_correctable_scatter function."""
        result = simulate_snv_correctable_scatter(
            sample_spectra,
            intensity=1.0,
            random_state=42
        )

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_simulate_snv_intensity_parameter(self, sample_spectra):
        """Test SNV scatter intensity parameter."""
        low_intensity = simulate_snv_correctable_scatter(
            sample_spectra, intensity=0.5, random_state=42
        )
        high_intensity = simulate_snv_correctable_scatter(
            sample_spectra, intensity=2.0, random_state=42
        )

        # Higher intensity should cause more deviation
        low_diff = np.std(low_intensity - sample_spectra)
        high_diff = np.std(high_intensity - sample_spectra)
        assert high_diff > low_diff

    def test_simulate_msc_correctable_scatter(self, sample_spectra):
        """Test simulate_msc_correctable_scatter function."""
        result = simulate_msc_correctable_scatter(
            sample_spectra,
            intensity=1.0,
            random_state=42
        )

        assert result.shape == sample_spectra.shape
        assert not np.allclose(result, sample_spectra)

    def test_simulate_msc_with_reference(self, sample_spectra):
        """Test MSC scatter with explicit reference."""
        reference = np.mean(sample_spectra, axis=0)
        result = simulate_msc_correctable_scatter(
            sample_spectra,
            reference=reference,
            intensity=1.0,
            random_state=42
        )

        assert result.shape == sample_spectra.shape


# ============================================================================
# ScatteringModel Enum Tests
# ============================================================================

class TestScatteringModel:
    """Tests for ScatteringModel enum."""

    def test_model_values(self):
        """Test that all models are defined."""
        assert ScatteringModel.EMSC.value == "emsc"
        assert ScatteringModel.RAYLEIGH.value == "rayleigh"
        assert ScatteringModel.MIE_APPROX.value == "mie_approx"
        assert ScatteringModel.KUBELKA_MUNK.value == "kubelka_munk"
        assert ScatteringModel.POLYNOMIAL.value == "polynomial"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_spectra(self, wavelengths):
        """Test with empty spectra array."""
        empty_spectra = np.zeros((0, len(wavelengths)))

        config = ParticleSizeConfig()
        simulator = ParticleSizeSimulator(config, random_state=42)

        result = simulator.apply(empty_spectra, wavelengths)
        assert result.shape == (0, len(wavelengths))

    def test_single_spectrum(self, wavelengths):
        """Test with single spectrum."""
        single = np.random.randn(1, len(wavelengths)) * 0.1 + 0.5

        config = ParticleSizeConfig()
        simulator = ParticleSizeSimulator(config, random_state=42)

        result = simulator.apply(single, wavelengths)
        assert result.shape == (1, len(wavelengths))

    def test_very_small_particles(self, wavelengths, sample_spectra):
        """Test with very small particle sizes."""
        config = ParticleSizeConfig(
            distribution=ParticleSizeDistribution(
                mean_size_um=5.0,
                min_size_um=1.0
            )
        )
        simulator = ParticleSizeSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))

    def test_very_large_particles(self, wavelengths, sample_spectra):
        """Test with very large particle sizes."""
        config = ParticleSizeConfig(
            distribution=ParticleSizeDistribution(
                mean_size_um=500.0,
                max_size_um=1000.0
            )
        )
        simulator = ParticleSizeSimulator(config, random_state=42)

        result = simulator.apply(sample_spectra, wavelengths)

        assert result.shape == sample_spectra.shape
        assert np.all(np.isfinite(result))

    def test_zero_scatter_variation(self, wavelengths):
        """Test with zero sample variation in scattering."""
        config = ScatteringCoefficientConfig(sample_variation=0.0)
        generator = ScatteringCoefficientGenerator(config, random_state=42)

        S = generator.generate(10, wavelengths)

        # With zero variation, all samples should be identical
        # (except for random fluctuation)
        assert S.shape == (10, len(wavelengths))

    def test_short_wavelength_array(self):
        """Test with very short wavelength array."""
        short_wl = np.array([1000, 1500, 2000])
        spectra = np.random.randn(5, 3) * 0.1 + 0.5

        config = EMSCConfig(polynomial_order=1)  # Lower order for short arrays
        simulator = EMSCTransformSimulator(config, random_state=42)

        result = simulator.apply(spectra, short_wl)
        assert result.shape == spectra.shape

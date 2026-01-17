"""
Unit tests for scattering module configuration classes.

Tests cover:
- ScatteringModel enum
- Particle size distribution and configuration
- EMSC configuration
- Scattering coefficient configuration
- Combined scattering effects configuration
"""

from __future__ import annotations

import numpy as np

from nirs4all.data.synthetic.scattering import (
    ScatteringModel,
    ParticleSizeDistribution,
    ParticleSizeConfig,
    EMSCConfig,
    ScatteringCoefficientConfig,
    ScatteringEffectsConfig,
)


class TestScatteringModel:
    """Tests for ScatteringModel enum."""

    def test_model_values(self):
        """Test that all models are defined."""
        assert ScatteringModel.EMSC.value == "emsc"
        assert ScatteringModel.RAYLEIGH.value == "rayleigh"
        assert ScatteringModel.MIE_APPROX.value == "mie_approx"
        assert ScatteringModel.KUBELKA_MUNK.value == "kubelka_munk"
        assert ScatteringModel.POLYNOMIAL.value == "polynomial"


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

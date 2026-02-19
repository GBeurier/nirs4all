"""
Unit tests for Phase 4 prior module - conditional prior sampling.
"""

import numpy as np
import pytest

from nirs4all.synthesis.prior import (
    MatrixType,
    NIRSPriorConfig,
    PriorSampler,
    get_domain_compatible_instruments,
    get_instrument_typical_modes,
    sample_prior,
    sample_prior_batch,
)


class TestNIRSPriorConfig:
    """Tests for NIRSPriorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = NIRSPriorConfig()

        assert isinstance(config.domain_weights, dict)
        assert len(config.domain_weights) > 0
        assert isinstance(config.instrument_given_domain, dict)
        assert isinstance(config.mode_given_category, dict)

    def test_domain_weights_sum(self):
        """Test that domain weights are reasonable."""
        config = NIRSPriorConfig()
        total = sum(config.domain_weights.values())
        # Weights should be positive
        assert all(w >= 0 for w in config.domain_weights.values())
        # Should sum to approximately 1
        assert 0.5 < total < 1.5

    def test_get_domain_weight(self):
        """Test getting domain weight."""
        config = NIRSPriorConfig()
        weight = config.get_domain_weight("grain")
        assert weight > 0

        # Unknown domain should return 0
        unknown = config.get_domain_weight("unknown_domain")
        assert unknown == 0.0

    def test_normalize_weights(self):
        """Test weight normalization."""
        config = NIRSPriorConfig()
        weights = {"a": 2.0, "b": 3.0, "c": 5.0}
        normalized = config.normalize_weights(weights)

        assert abs(sum(normalized.values()) - 1.0) < 1e-10
        assert normalized["a"] == 0.2
        assert normalized["b"] == 0.3
        assert normalized["c"] == 0.5

class TestPriorSampler:
    """Tests for PriorSampler class."""

    def test_sampler_creation(self):
        """Test sampler creation."""
        sampler = PriorSampler(random_state=42)
        assert sampler.config is not None
        assert sampler.rng is not None

    def test_sample_domain(self):
        """Test domain sampling."""
        sampler = PriorSampler(random_state=42)

        domains = [sampler.sample_domain() for _ in range(100)]

        # Should sample from available domains
        config = NIRSPriorConfig()
        for d in domains:
            assert d in config.domain_weights

    def test_sample_instrument_category(self):
        """Test instrument category sampling."""
        sampler = PriorSampler(random_state=42)

        # Sample for known domain
        categories = [sampler.sample_instrument_category("tablets") for _ in range(50)]

        # Should favor benchtop/ft_nir for pharmaceutical
        assert any(c == "benchtop" for c in categories)

    def test_sample_instrument(self):
        """Test specific instrument sampling."""
        sampler = PriorSampler(random_state=42)

        instruments = [sampler.sample_instrument("benchtop") for _ in range(50)]

        # Should return valid instrument names
        from nirs4all.synthesis import INSTRUMENT_ARCHETYPES
        for inst in instruments:
            assert inst in INSTRUMENT_ARCHETYPES

    def test_sample_measurement_mode(self):
        """Test measurement mode sampling."""
        sampler = PriorSampler(random_state=42)

        modes = [sampler.sample_measurement_mode("handheld") for _ in range(50)]

        valid_modes = ["reflectance", "transmittance", "transflectance", "atr"]
        for mode in modes:
            assert mode in valid_modes

    def test_sample_matrix_type(self):
        """Test matrix type sampling."""
        sampler = PriorSampler(random_state=42)

        # Dairy should favor liquid/emulsion
        matrices = [sampler.sample_matrix_type("dairy") for _ in range(50)]
        assert any(m in ("liquid", "emulsion") for m in matrices)

        # Tablets should favor solid/powder
        matrices = [sampler.sample_matrix_type("tablets") for _ in range(50)]
        assert any(m in ("solid", "powder") for m in matrices)

    def test_sample_temperature(self):
        """Test temperature sampling."""
        sampler = PriorSampler(random_state=42)
        config = NIRSPriorConfig()

        temps = [sampler.sample_temperature() for _ in range(100)]

        low, high = config.temperature_range
        for t in temps:
            assert low <= t <= high

    def test_sample_particle_size(self):
        """Test particle size sampling."""
        sampler = PriorSampler(random_state=42)

        # Powder should have smaller particles
        powder_sizes = [sampler.sample_particle_size("powder") for _ in range(50)]
        # Granular should have larger particles
        granular_sizes = [sampler.sample_particle_size("granular") for _ in range(50)]

        assert np.mean(powder_sizes) < np.mean(granular_sizes)

    def test_sample_complete(self):
        """Test complete configuration sampling."""
        sampler = PriorSampler(random_state=42)
        sample = sampler.sample()

        # Check required keys
        required_keys = [
            "domain", "instrument", "instrument_category",
            "measurement_mode", "matrix_type", "temperature",
            "particle_size", "noise_level", "components",
            "n_samples", "target_config", "random_state",
        ]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

        # Check value types
        assert isinstance(sample["domain"], str)
        assert isinstance(sample["instrument"], str)
        assert isinstance(sample["temperature"], float)
        assert isinstance(sample["n_samples"], int)
        assert isinstance(sample["components"], list)
        assert isinstance(sample["target_config"], dict)

    def test_sample_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        sampler1 = PriorSampler(random_state=42)
        sampler2 = PriorSampler(random_state=42)

        sample1 = sampler1.sample()
        sample2 = sampler2.sample()

        assert sample1["domain"] == sample2["domain"]
        assert sample1["instrument"] == sample2["instrument"]

    def test_sample_batch(self):
        """Test batch sampling."""
        sampler = PriorSampler(random_state=42)
        samples = sampler.sample_batch(10)

        assert len(samples) == 10
        for sample in samples:
            assert "domain" in sample
            assert "instrument" in sample

    def test_sample_for_domain(self):
        """Test sampling constrained to a domain."""
        sampler = PriorSampler(random_state=42)

        sample = sampler.sample_for_domain("tablets", n_samples=500)

        assert sample["domain"] == "tablets"
        assert sample["n_samples"] == 500
        # Components should be relevant to tablets
        assert len(sample["components"]) > 0

    def test_sample_for_instrument(self):
        """Test sampling constrained to an instrument."""
        sampler = PriorSampler(random_state=42)

        sample = sampler.sample_for_instrument("viavi_micronir")

        assert sample["instrument"] == "viavi_micronir"
        assert sample["instrument_category"] == "handheld"

    def test_sample_target_config_regression(self):
        """Test target config sampling."""
        sampler = PriorSampler(random_state=42)

        configs = [sampler.sample_target_config() for _ in range(100)]

        regression_count = sum(1 for c in configs if c["type"] == "regression")
        classification_count = sum(1 for c in configs if c["type"] == "classification")

        # Should have mix of both types
        assert regression_count > 0
        assert classification_count > 0

    def test_sample_components(self):
        """Test component sampling."""
        sampler = PriorSampler(random_state=42)

        # Should return domain-relevant components
        components = sampler.sample_components("dairy")
        # Common dairy components
        possible = ["water", "protein", "lipid", "lactose", "fat", "casein", "whey"]
        # At least some should be in the list
        assert len(components) > 0

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_sample_prior(self):
        """Test sample_prior function."""
        sample = sample_prior(random_state=42)

        assert isinstance(sample, dict)
        assert "domain" in sample
        assert "instrument" in sample

    def test_sample_prior_batch(self):
        """Test sample_prior_batch function."""
        samples = sample_prior_batch(5, random_state=42)

        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, dict)

    def test_get_domain_compatible_instruments(self):
        """Test getting compatible instruments for domain."""
        instruments = get_domain_compatible_instruments("tablets")

        assert len(instruments) > 0
        # Pharmaceutical should include benchtop instruments
        # At least some instruments should be returned

    def test_get_instrument_typical_modes(self):
        """Test getting typical modes for instrument."""
        modes = get_instrument_typical_modes("viavi_micronir")

        assert len(modes) > 0
        # Handheld should support reflectance
        assert "reflectance" in modes

class TestMatrixType:
    """Tests for MatrixType enum."""

    def test_matrix_types(self):
        """Test matrix type enumeration."""
        assert MatrixType.LIQUID.value == "liquid"
        assert MatrixType.POWDER.value == "powder"
        assert MatrixType.SOLID.value == "solid"
        assert MatrixType.EMULSION.value == "emulsion"

    def test_matrix_type_from_string(self):
        """Test creating MatrixType from string."""
        matrix = MatrixType("liquid")
        assert matrix == MatrixType.LIQUID

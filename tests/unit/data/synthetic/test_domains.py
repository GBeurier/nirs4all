"""
Unit tests for the domains module (Phase 1.3).

Tests cover:
- Domain category enumeration
- Concentration prior class
- Domain configuration class
- Application domains registry
- Domain utility functions
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.synthesis.domains import (
    DomainCategory,
    ConcentrationPrior,
    DomainConfig,
    APPLICATION_DOMAINS,
    get_domain_config,
    list_domains,
    get_domain_components,
    create_domain_aware_library,
)


class TestDomainCategory:
    """Tests for DomainCategory enumeration."""

    def test_domain_category_values(self):
        """Test that DomainCategory contains expected values."""
        assert DomainCategory.AGRICULTURE is not None
        assert DomainCategory.FOOD is not None
        assert DomainCategory.PHARMACEUTICAL is not None
        assert DomainCategory.PETROCHEMICAL is not None

    def test_domain_category_iterable(self):
        """Test that DomainCategory is iterable."""
        categories = list(DomainCategory)
        assert len(categories) >= 6  # At least 6 categories


class TestConcentrationPrior:
    """Tests for ConcentrationPrior class."""

    def test_prior_creation_uniform(self):
        """Test uniform prior creation."""
        prior = ConcentrationPrior(
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
        )
        assert prior.distribution == "uniform"
        assert prior.params["low"] == 0.0
        assert prior.params["high"] == 1.0

    def test_prior_creation_normal(self):
        """Test normal prior creation."""
        prior = ConcentrationPrior(
            distribution="normal",
            params={"mean": 0.5, "std": 0.1},
        )
        assert prior.distribution == "normal"
        assert prior.params["mean"] == 0.5
        assert prior.params["std"] == 0.1

    def test_prior_creation_beta(self):
        """Test beta prior creation."""
        prior = ConcentrationPrior(
            distribution="beta",
            params={"a": 2.0, "b": 5.0},
        )
        assert prior.distribution == "beta"
        assert prior.params["a"] == 2.0
        assert prior.params["b"] == 5.0

    def test_prior_sample_uniform(self):
        """Test sampling from uniform prior."""
        prior = ConcentrationPrior(
            distribution="uniform",
            params={"low": 0.2, "high": 0.8},
            min_value=0.2,
            max_value=0.8,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, 100)

        assert samples.shape == (100,)
        assert np.all(samples >= 0.2)
        assert np.all(samples <= 0.8)

    def test_prior_sample_normal(self):
        """Test sampling from normal prior."""
        prior = ConcentrationPrior(
            distribution="normal",
            params={"mean": 0.5, "std": 0.1},
            min_value=0.0,
            max_value=1.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, 1000)

        assert samples.shape == (1000,)
        # Mean should be approximately 0.5
        assert 0.4 < np.mean(samples) < 0.6
        # All should be within bounds
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_prior_sample_beta(self):
        """Test sampling from beta prior."""
        prior = ConcentrationPrior(
            distribution="beta",
            params={"a": 2.0, "b": 5.0},
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, 1000)

        assert samples.shape == (1000,)
        # Beta(2,5) has mode at (2-1)/(2+5-2) = 0.2
        # Mean at 2/(2+5) ≈ 0.286
        assert 0.2 < np.mean(samples) < 0.4
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_prior_sample_reproducible(self):
        """Test that sampling is reproducible with same RNG."""
        prior = ConcentrationPrior(
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
        )
        samples1 = prior.sample(np.random.default_rng(42), 10)
        samples2 = prior.sample(np.random.default_rng(42), 10)

        np.testing.assert_array_equal(samples1, samples2)


class TestDomainConfig:
    """Tests for DomainConfig class."""

    def test_config_creation(self):
        """Test basic domain config creation."""
        config = DomainConfig(
            name="test_domain",
            category=DomainCategory.FOOD,
            description="A test domain",
            typical_components=["water", "protein"],
        )
        assert config.name == "test_domain"
        assert config.category == DomainCategory.FOOD
        assert "water" in config.typical_components

    def test_config_with_priors(self):
        """Test domain config with concentration priors."""
        priors = {
            "water": ConcentrationPrior(
                distribution="uniform",
                params={"low": 0.6, "high": 0.9},
                min_value=0.6,
                max_value=0.9,
            ),
            "protein": ConcentrationPrior(
                distribution="normal",
                params={"mean": 0.15, "std": 0.05},
            ),
        }
        config = DomainConfig(
            name="dairy",
            category=DomainCategory.FOOD,
            description="Dairy products",
            typical_components=["water", "protein", "lipid"],
            concentration_priors=priors,
        )
        assert "water" in config.concentration_priors
        assert config.concentration_priors["water"].min_value == 0.6

    def test_config_sample_concentrations(self):
        """Test sampling concentrations from domain config."""
        priors = {
            "water": ConcentrationPrior(
                distribution="uniform",
                params={"low": 0.7, "high": 0.9},
                min_value=0.7,
                max_value=0.9,
            ),
            "protein": ConcentrationPrior(
                distribution="uniform",
                params={"low": 0.05, "high": 0.15},
                min_value=0.05,
                max_value=0.15,
            ),
        }
        config = DomainConfig(
            name="test",
            category=DomainCategory.FOOD,
            description="Test",
            typical_components=["water", "protein"],
            concentration_priors=priors,
        )

        rng = np.random.default_rng(42)
        components = ["water", "protein"]
        concentrations = config.sample_concentrations(rng, components, 100)

        # Returns a matrix (n_samples, n_components)
        assert concentrations.shape == (100, 2)
        # Water is at index 0
        assert np.all(concentrations[:, 0] >= 0.7)
        assert np.all(concentrations[:, 0] <= 0.9)


class TestApplicationDomains:
    """Tests for APPLICATION_DOMAINS registry."""

    def test_application_domains_structure(self):
        """Test that APPLICATION_DOMAINS has expected structure."""
        assert isinstance(APPLICATION_DOMAINS, dict)
        assert len(APPLICATION_DOMAINS) >= 10  # At least 10 domains

    def test_all_domains_are_domain_config(self):
        """Test that all entries are DomainConfig instances."""
        for name, config in APPLICATION_DOMAINS.items():
            assert isinstance(config, DomainConfig)
            # Note: config.name is the display name, not the key
            assert config.name is not None

    def test_key_domains_present(self):
        """Test that key application domains are present."""
        expected_domains = [
            "agriculture_grain",
            "food_dairy",
            "pharma_tablets",
        ]
        for domain in expected_domains:
            assert domain in APPLICATION_DOMAINS, f"Missing domain: {domain}"
        # Check petrochem domain - may be named differently
        petrochem_domains = [k for k in APPLICATION_DOMAINS if "petrochem" in k]
        assert len(petrochem_domains) > 0, "Missing petrochemical domain"

    def test_domains_have_required_fields(self):
        """Test that all domains have required fields."""
        for name, config in APPLICATION_DOMAINS.items():
            assert config.name is not None
            assert config.category is not None
            assert config.description is not None
            assert len(config.typical_components) > 0

    def test_domains_cover_all_categories(self):
        """Test that domains cover multiple categories."""
        categories_used = set()
        for config in APPLICATION_DOMAINS.values():
            categories_used.add(config.category)

        # Should have at least 4 different categories
        assert len(categories_used) >= 4


class TestDomainUtilityFunctions:
    """Tests for domain utility functions."""

    def test_get_domain_config_existing(self):
        """Test getting an existing domain config."""
        config = get_domain_config("agriculture_grain")
        assert config is not None
        assert config.name == "Grain and Cereals"
        assert config.category == DomainCategory.AGRICULTURE

    def test_get_domain_config_nonexistent(self):
        """Test getting a non-existent domain config raises ValueError."""
        with pytest.raises(ValueError):
            get_domain_config("nonexistent_domain")

    def test_get_domain_config_case_sensitivity(self):
        """Test domain config lookup case sensitivity."""
        # Should be case-sensitive (lowercase expected)
        config_lower = get_domain_config("agriculture_grain")
        assert config_lower is not None
        # Upper case should raise
        with pytest.raises(ValueError):
            get_domain_config("AGRICULTURE_GRAIN")

    def test_list_domains(self):
        """Test listing all domain names."""
        domains = list_domains()
        assert isinstance(domains, list)
        assert len(domains) >= 10
        assert "agriculture_grain" in domains
        assert "food_dairy" in domains

    def test_list_domains_by_category_agriculture(self):
        """Test listing domains by agriculture category."""
        domains = list_domains(category=DomainCategory.AGRICULTURE)
        assert isinstance(domains, list)
        assert len(domains) >= 2
        # All should be agriculture domains
        for domain_name in domains:
            config = get_domain_config(domain_name)
            assert config.category == DomainCategory.AGRICULTURE

    def test_list_domains_by_category_food(self):
        """Test listing domains by food category."""
        domains = list_domains(category=DomainCategory.FOOD)
        assert len(domains) >= 2
        for domain_name in domains:
            config = get_domain_config(domain_name)
            assert config.category == DomainCategory.FOOD

    def test_list_domains_by_category_pharmaceutical(self):
        """Test listing domains by pharmaceutical category."""
        domains = list_domains(category=DomainCategory.PHARMACEUTICAL)
        assert len(domains) >= 1


class TestCreateDomainAwareLibrary:
    """Tests for create_domain_aware_library function."""

    def test_create_library_basic(self):
        """Test creating a domain-aware component library."""
        result = create_domain_aware_library("agriculture_grain")
        assert result is not None
        # Returns (components, concentrations)
        components, concentrations = result
        assert len(components) > 0

    def test_create_library_contains_domain_components(self):
        """Test that library contains domain-specific components."""
        config = get_domain_config("agriculture_grain")
        components, concentrations = create_domain_aware_library("agriculture_grain")

        # Library should contain at least some of the typical components
        overlap = set(config.typical_components) & set(components)
        assert len(overlap) > 0

    def test_create_library_nonexistent_domain_raises(self):
        """Test that non-existent domain raises an error."""
        with pytest.raises((ValueError, KeyError)):
            create_domain_aware_library("nonexistent_domain_xyz")

    def test_create_library_reproducible(self):
        """Test that library creation is reproducible."""
        components1, _ = create_domain_aware_library("food_dairy", random_state=42)
        components2, _ = create_domain_aware_library("food_dairy", random_state=42)

        # Should have same components
        assert set(components1) == set(components2)


class TestDomainConfigSamplingIntegration:
    """Integration tests for domain configuration sampling."""

    def test_grain_domain_realistic_sampling(self):
        """Test that grain domain produces realistic concentrations."""
        config = get_domain_config("agriculture_grain")
        rng = np.random.default_rng(42)

        # Sample components first
        components = config.sample_components(rng)
        concentrations = config.sample_concentrations(rng, components, 1000)

        # Check shape
        assert concentrations.shape == (1000, len(components))

        # All concentrations should be in valid range
        assert np.all(concentrations >= 0)
        assert np.all(concentrations <= 1)

    def test_dairy_domain_realistic_sampling(self):
        """Test that dairy domain produces realistic concentrations."""
        config = get_domain_config("food_dairy")
        rng = np.random.default_rng(42)

        # Sample components first
        components = config.sample_components(rng)
        concentrations = config.sample_concentrations(rng, components, 1000)

        # Check shape
        assert concentrations.shape == (1000, len(components))

        # All concentrations should be in valid range
        assert np.all(concentrations >= 0)
        assert np.all(concentrations <= 1)

    def test_pharmaceutical_domain_sampling(self):
        """Test pharmaceutical domain sampling."""
        config = get_domain_config("pharma_tablets")
        if config is None:
            pytest.skip("pharma_tablets domain not defined")

        rng = np.random.default_rng(42)
        components = config.sample_components(rng)
        concentrations = config.sample_concentrations(rng, components, 100)

        # All concentrations should be valid
        assert np.all(concentrations >= 0)
        assert np.all(concentrations <= 1)


class TestDomainEdgeCases:
    """Edge case tests for domain module."""

    def test_empty_concentration_priors(self):
        """Test domain with no concentration priors."""
        config = DomainConfig(
            name="minimal",
            category=DomainCategory.FOOD,
            description="Minimal domain",
            typical_components=["water"],
            concentration_priors={},
        )
        rng = np.random.default_rng(42)
        # Should return concentrations using default prior
        concentrations = config.sample_concentrations(rng, ["water"], 10)
        assert isinstance(concentrations, np.ndarray)
        assert concentrations.shape == (10, 1)

    def test_single_component_domain(self):
        """Test domain with single component."""
        priors = {
            "water": ConcentrationPrior(
                distribution="uniform",
                params={"low": 0.9, "high": 1.0},
                min_value=0.9,
                max_value=1.0,
            ),
        }
        config = DomainConfig(
            name="pure_water",
            category=DomainCategory.ENVIRONMENTAL,
            description="Pure water",
            typical_components=["water"],
            concentration_priors=priors,
        )
        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(rng, ["water"], 10)
        assert concentrations.shape == (10, 1)

    def test_all_distribution_types(self):
        """Test domain with all distribution types."""
        priors = {
            "comp1": ConcentrationPrior(
                distribution="uniform",
                params={"low": 0.1, "high": 0.3},
            ),
            "comp2": ConcentrationPrior(
                distribution="normal",
                params={"mean": 0.5, "std": 0.1},
            ),
            "comp3": ConcentrationPrior(
                distribution="beta",
                params={"a": 2, "b": 5},
            ),
        }
        config = DomainConfig(
            name="mixed",
            category=DomainCategory.FOOD,
            description="Mixed distributions",
            typical_components=["comp1", "comp2", "comp3"],
            concentration_priors=priors,
        )
        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(rng, ["comp1", "comp2", "comp3"], 100)

        assert concentrations.shape == (100, 3)
        # All values should be in valid range
        assert np.all(concentrations >= 0)
        assert np.all(concentrations <= 1)

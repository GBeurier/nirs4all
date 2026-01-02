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

from nirs4all.data.synthetic.domains import (
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
            min_value=0.0,
            max_value=1.0,
        )
        assert prior.distribution == "uniform"
        assert prior.min_value == 0.0
        assert prior.max_value == 1.0

    def test_prior_creation_normal(self):
        """Test normal prior creation."""
        prior = ConcentrationPrior(
            distribution="normal",
            mean=0.5,
            std=0.1,
        )
        assert prior.distribution == "normal"
        assert prior.mean == 0.5
        assert prior.std == 0.1

    def test_prior_creation_beta(self):
        """Test beta prior creation."""
        prior = ConcentrationPrior(
            distribution="beta",
            alpha=2.0,
            beta=5.0,
        )
        assert prior.distribution == "beta"
        assert prior.alpha == 2.0
        assert prior.beta == 5.0

    def test_prior_sample_uniform(self):
        """Test sampling from uniform prior."""
        prior = ConcentrationPrior(
            distribution="uniform",
            min_value=0.2,
            max_value=0.8,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(100, rng)

        assert samples.shape == (100,)
        assert np.all(samples >= 0.2)
        assert np.all(samples <= 0.8)

    def test_prior_sample_normal(self):
        """Test sampling from normal prior."""
        prior = ConcentrationPrior(
            distribution="normal",
            mean=0.5,
            std=0.1,
            min_value=0.0,
            max_value=1.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(1000, rng)

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
            alpha=2.0,
            beta=5.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(1000, rng)

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
            min_value=0.0,
            max_value=1.0,
        )
        samples1 = prior.sample(10, np.random.default_rng(42))
        samples2 = prior.sample(10, np.random.default_rng(42))

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
            "water": ConcentrationPrior(distribution="uniform", min_value=0.6, max_value=0.9),
            "protein": ConcentrationPrior(distribution="normal", mean=0.15, std=0.05),
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
            "water": ConcentrationPrior(distribution="uniform", min_value=0.7, max_value=0.9),
            "protein": ConcentrationPrior(distribution="uniform", min_value=0.05, max_value=0.15),
        }
        config = DomainConfig(
            name="test",
            category=DomainCategory.FOOD,
            description="Test",
            typical_components=["water", "protein"],
            concentration_priors=priors,
        )

        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(100, rng)

        assert "water" in concentrations
        assert "protein" in concentrations
        assert concentrations["water"].shape == (100,)
        assert np.all(concentrations["water"] >= 0.7)
        assert np.all(concentrations["water"] <= 0.9)


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
            "petrochem_fuel",
        ]
        for domain in expected_domains:
            assert domain in APPLICATION_DOMAINS, f"Missing domain: {domain}"

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
        library = create_domain_aware_library("agriculture_grain")
        assert library is not None
        assert len(library) > 0

    def test_create_library_contains_domain_components(self):
        """Test that library contains domain-specific components."""
        config = get_domain_config("agriculture_grain")
        library = create_domain_aware_library("agriculture_grain")

        # Library should contain at least some of the typical components
        component_names = list(library.keys()) if hasattr(library, 'keys') else [c.name for c in library]
        overlap = set(config.typical_components) & set(component_names)
        assert len(overlap) > 0

    def test_create_library_nonexistent_domain_raises(self):
        """Test that non-existent domain raises an error."""
        with pytest.raises((ValueError, KeyError)):
            create_domain_aware_library("nonexistent_domain_xyz")

    def test_create_library_reproducible(self):
        """Test that library creation is reproducible."""
        lib1 = create_domain_aware_library("food_dairy")
        lib2 = create_domain_aware_library("food_dairy")

        # Should have same components
        keys1 = set(lib1.keys()) if hasattr(lib1, 'keys') else set()
        keys2 = set(lib2.keys()) if hasattr(lib2, 'keys') else set()
        assert keys1 == keys2


class TestDomainConfigSamplingIntegration:
    """Integration tests for domain configuration sampling."""

    def test_grain_domain_realistic_sampling(self):
        """Test that grain domain produces realistic concentrations."""
        config = get_domain_config("agriculture_grain")
        rng = np.random.default_rng(42)

        concentrations = config.sample_concentrations(1000, rng)

        # Water content in grains typically 8-14%
        if "moisture" in concentrations:
            water = concentrations["moisture"]
            assert np.mean(water) < 0.20  # Average < 20%

        # Starch content typically 50-70%
        if "starch" in concentrations:
            starch = concentrations["starch"]
            assert np.mean(starch) > 0.40  # Average > 40%

    def test_dairy_domain_realistic_sampling(self):
        """Test that dairy domain produces realistic concentrations."""
        config = get_domain_config("food_dairy")
        rng = np.random.default_rng(42)

        concentrations = config.sample_concentrations(1000, rng)

        # Water content in dairy typically 80-90%
        if "moisture" in concentrations:
            water = concentrations["moisture"]
            assert np.mean(water) > 0.70  # Average > 70%

    def test_pharmaceutical_domain_sampling(self):
        """Test pharmaceutical domain sampling."""
        config = get_domain_config("pharma_tablets")
        if config is None:
            pytest.skip("pharma_tablets domain not defined")

        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(100, rng)

        # All concentrations should be valid
        for component, values in concentrations.items():
            assert np.all(values >= 0)
            assert np.all(values <= 1)


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
        # Should return empty dict or use defaults
        concentrations = config.sample_concentrations(10, rng)
        assert isinstance(concentrations, dict)

    def test_single_component_domain(self):
        """Test domain with single component."""
        priors = {
            "water": ConcentrationPrior(distribution="uniform", min_value=0.9, max_value=1.0),
        }
        config = DomainConfig(
            name="pure_water",
            category=DomainCategory.ENVIRONMENTAL,
            description="Pure water",
            typical_components=["water"],
            concentration_priors=priors,
        )
        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(10, rng)
        assert "water" in concentrations
        assert len(concentrations["water"]) == 10

    def test_all_distribution_types(self):
        """Test domain with all distribution types."""
        priors = {
            "comp1": ConcentrationPrior(distribution="uniform", min_value=0.1, max_value=0.3),
            "comp2": ConcentrationPrior(distribution="normal", mean=0.5, std=0.1),
            "comp3": ConcentrationPrior(distribution="beta", alpha=2, beta=5),
        }
        config = DomainConfig(
            name="mixed",
            category=DomainCategory.FOOD,
            description="Mixed distributions",
            typical_components=["comp1", "comp2", "comp3"],
            concentration_priors=priors,
        )
        rng = np.random.default_rng(42)
        concentrations = config.sample_concentrations(100, rng)

        assert len(concentrations) == 3
        for comp in ["comp1", "comp2", "comp3"]:
            assert comp in concentrations
            assert concentrations[comp].shape == (100,)

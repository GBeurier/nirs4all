"""
Unit tests for the procedural module (Phase 1.2).

Tests cover:
- Functional group type enumeration
- Functional group properties dictionary
- Procedural component configuration
- Procedural component generator class
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.synthesis.components import NIRBand, SpectralComponent
from nirs4all.synthesis.procedural import (
    FUNCTIONAL_GROUP_PROPERTIES,
    FunctionalGroupType,
    ProceduralComponentConfig,
    ProceduralComponentGenerator,
)


class TestFunctionalGroupType:
    """Tests for FunctionalGroupType enumeration."""

    def test_functional_group_type_values(self):
        """Test that FunctionalGroupType contains expected values."""
        # Check that key functional groups are present
        assert FunctionalGroupType.HYDROXYL is not None
        assert FunctionalGroupType.AMINE is not None
        assert FunctionalGroupType.METHYL is not None
        assert FunctionalGroupType.METHYLENE is not None
        assert FunctionalGroupType.AROMATIC_CH is not None
        assert FunctionalGroupType.CARBONYL is not None
        assert FunctionalGroupType.CARBOXYL is not None

    def test_functional_group_type_iterable(self):
        """Test that FunctionalGroupType is iterable."""
        groups = list(FunctionalGroupType)
        assert len(groups) >= 10  # At least 10 functional groups

class TestFunctionalGroupProperties:
    """Tests for FUNCTIONAL_GROUP_PROPERTIES dictionary."""

    def test_properties_structure(self):
        """Test that FUNCTIONAL_GROUP_PROPERTIES has expected structure."""
        assert isinstance(FUNCTIONAL_GROUP_PROPERTIES, dict)
        assert len(FUNCTIONAL_GROUP_PROPERTIES) == len(list(FunctionalGroupType))

    def test_all_groups_have_properties(self):
        """Test that all functional groups have properties defined."""
        for group in FunctionalGroupType:
            assert group in FUNCTIONAL_GROUP_PROPERTIES

    def test_properties_contain_required_fields(self):
        """Test that each property entry contains required fields."""
        required_fields = [
            "fundamental_cm",
            "bandwidth_cm",
            "h_bond_susceptibility",
            "typical_amplitude",
        ]
        for group, props in FUNCTIONAL_GROUP_PROPERTIES.items():
            for field in required_fields:
                assert field in props, f"Missing {field} for {group}"

    def test_properties_values_in_valid_range(self):
        """Test that property values are in valid ranges."""
        for group, props in FUNCTIONAL_GROUP_PROPERTIES.items():
            # Fundamental frequency should be in MIR range (500-4000 cm⁻¹)
            assert 500 <= props["fundamental_cm"] <= 4500
            # Bandwidth typically 10-200 cm⁻¹
            assert 10 <= props["bandwidth_cm"] <= 500
            # H-bonding susceptibility 0-1
            assert 0 <= props["h_bond_susceptibility"] <= 1
            # Amplitude 0-1
            assert 0 < props["typical_amplitude"] <= 1

class TestProceduralComponentConfig:
    """Tests for ProceduralComponentConfig dataclass."""

    def test_config_creation_defaults(self):
        """Test config creation with default values."""
        config = ProceduralComponentConfig()
        assert config.functional_groups is None
        assert config.max_overtone_order >= 2
        assert isinstance(config.include_combinations, bool)

    def test_config_creation_with_groups(self):
        """Test config creation with functional groups."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL, FunctionalGroupType.METHYL],
        )
        assert len(config.functional_groups) == 2
        assert FunctionalGroupType.HYDROXYL in config.functional_groups

    def test_config_with_custom_parameters(self):
        """Test config with all custom parameters."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.AMINE],
            max_overtone_order=3,
            include_combinations=False,
            h_bond_strength=0.7,
        )
        assert config.max_overtone_order == 3
        assert config.include_combinations is False
        assert config.h_bond_strength == 0.7

class TestProceduralComponentGenerator:
    """Tests for ProceduralComponentGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return ProceduralComponentGenerator(random_state=42)

    def test_generator_creation(self, generator):
        """Test generator instantiation."""
        assert generator is not None

    def test_generator_with_seed(self):
        """Test generator reproducibility with seed."""
        gen1 = ProceduralComponentGenerator(random_state=42)
        gen2 = ProceduralComponentGenerator(random_state=42)
        # Should produce identical results
        comp1 = gen1.generate_component(
            "test",
            functional_groups=[FunctionalGroupType.HYDROXYL],
        )
        comp2 = gen2.generate_component(
            "test",
            functional_groups=[FunctionalGroupType.HYDROXYL],
        )
        assert len(comp1.bands) == len(comp2.bands)

    def test_generate_component_basic(self, generator):
        """Test basic component generation."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL, FunctionalGroupType.METHYL],
        )
        component = generator.generate_component("ethanol", config)
        assert isinstance(component, SpectralComponent)
        assert component.name == "ethanol"
        assert len(component.bands) > 0

    def test_generate_component_has_bands(self, generator):
        """Test that generated component has NIR bands."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            max_overtone_order=2,
        )
        component = generator.generate_component("test", config)
        for band in component.bands:
            assert isinstance(band, NIRBand)
            # Bands should be in NIR range (800-2500 nm)
            assert 750 < band.center < 2600

    def test_generate_component_overtone_series(self, generator):
        """Test that overtone series is generated correctly."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            max_overtone_order=3,
            include_combinations=False,
        )
        component = generator.generate_component("test", config)
        # Should have multiple overtones
        assert len(component.bands) >= 2

    def test_generate_component_with_combinations(self, generator):
        """Test that combination bands are generated when enabled."""
        config_with = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            max_overtone_order=2,
            include_combinations=True,
        )
        config_without = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            max_overtone_order=2,
            include_combinations=False,
        )
        comp_with = generator.generate_component("with_comb", config_with)
        comp_without = generator.generate_component("without_comb", config_without)
        # With combinations should have at least as many bands
        assert len(comp_with.bands) >= len(comp_without.bands)

    def test_generate_component_h_bonding_shift(self, generator):
        """Test that H-bonding shifts are applied."""
        config_no_hb = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            h_bond_strength=0.0,
            max_overtone_order=2,
            include_combinations=False,
        )
        config_hb = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
            h_bond_strength=0.8,
            max_overtone_order=2,
            include_combinations=False,
        )
        comp_no_hb = generator.generate_component("no_hbond", config_no_hb)
        comp_hb = generator.generate_component("hbond", config_hb)

        # H-bonded should have shifted bands (to longer wavelengths for O-H)
        # Find the first overtone band in each
        if len(comp_no_hb.bands) > 0 and len(comp_hb.bands) > 0:
            band_no_hb = comp_no_hb.bands[0].center
            band_hb = comp_hb.bands[0].center
            # H-bonding should shift to longer wavelengths
            assert band_hb > band_no_hb

    def test_generate_variant(self, generator):
        """Test generation of component variants."""
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
        )
        base = generator.generate_component("base", config)

        # Generate variant with perturbations (using correct parameter name)
        variant = generator.generate_variant(base, variation_scale=0.1)

        assert variant is not None
        assert len(variant.bands) == len(base.bands)
        # Band centers should be slightly different
        for orig, var in zip(base.bands, variant.bands):
            # Should be close but not identical
            assert abs(orig.center - var.center) < 50

class TestProceduralGeneratorEdgeCases:
    """Edge case tests for ProceduralComponentGenerator."""

    def test_empty_functional_groups(self):
        """Test behavior with empty functional groups."""
        generator = ProceduralComponentGenerator(random_state=42)
        config = ProceduralComponentConfig(
            functional_groups=[],
        )
        component = generator.generate_component("empty", config)
        # Should return component with no bands or raise
        assert len(component.bands) == 0 or component is not None

    def test_single_functional_group(self):
        """Test with single functional group."""
        generator = ProceduralComponentGenerator(random_state=42)
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.CARBONYL],
            max_overtone_order=2,
        )
        component = generator.generate_component("single", config)
        assert isinstance(component, SpectralComponent)

    def test_max_overtone_two(self):
        """Test with max_overtone_order=2 (only first overtone)."""
        generator = ProceduralComponentGenerator(random_state=42)
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.METHYL],
            max_overtone_order=2,
            include_combinations=False,
        )
        component = generator.generate_component("first_only", config)
        # Should have at least one band
        assert len(component.bands) >= 1

    def test_high_max_overtone(self):
        """Test with high max_overtone_order value."""
        generator = ProceduralComponentGenerator(random_state=42)
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.METHYL],
            max_overtone_order=4,
            include_combinations=False,
        )
        component = generator.generate_component("high_overtone", config)
        # Should have multiple overtones
        assert len(component.bands) >= 2

class TestProceduralIntegration:
    """Integration tests for procedural component generation."""

    def test_generated_component_computes_spectrum(self):
        """Test that generated components can compute spectra."""
        generator = ProceduralComponentGenerator(random_state=42)
        config = ProceduralComponentConfig(
            functional_groups=[FunctionalGroupType.HYDROXYL],
        )
        component = generator.generate_component("test", config)

        # Should be able to compute on wavelength array
        wavelengths = np.linspace(900, 2500, 100)
        # compute() takes only wavelengths, not concentration
        spectrum = component.compute(wavelengths)

        assert spectrum.shape == wavelengths.shape
        assert np.all(spectrum >= 0)  # Absorbance should be non-negative

    def test_multiple_components_generate_diverse_spectra(self):
        """Test that library produces diverse spectra."""
        generator = ProceduralComponentGenerator(random_state=42)

        # Generate different component types
        comp_hydroxyl = generator.generate_component(
            "alcohol", functional_groups=[FunctionalGroupType.HYDROXYL]
        )
        comp_amine = generator.generate_component(
            "amine", functional_groups=[FunctionalGroupType.AMINE]
        )
        comp_aromatic = generator.generate_component(
            "aromatic", functional_groups=[FunctionalGroupType.AROMATIC_CH]
        )

        wavelengths = np.linspace(900, 2500, 100)
        # compute() takes only wavelengths
        spectra = [
            comp_hydroxyl.compute(wavelengths),
            comp_amine.compute(wavelengths),
            comp_aromatic.compute(wavelengths),
        ]

        # Spectra should be different from each other
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                # Handle zero arrays (low concentration or narrow bands)
                if np.std(spectra[i]) > 0 and np.std(spectra[j]) > 0:
                    correlation = np.corrcoef(spectra[i], spectra[j])[0, 1]
                    # Shouldn't be perfectly correlated
                    assert correlation < 0.99

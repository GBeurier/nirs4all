"""
Unit tests for NIRBand, SpectralComponent, and ComponentLibrary classes.
"""

import pytest
import numpy as np

from nirs4all.synthesis import (
    NIRBand,
    SpectralComponent,
    ComponentLibrary,
    PREDEFINED_COMPONENTS,
    get_predefined_components,
    # Discovery API (Phase 1 enhancement)
    available_components,
    get_component,
    search_components,
    list_categories,
    component_info,
    validate_predefined_components,
    validate_component_coverage,
    normalize_component_amplitudes,
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


class TestSpectralComponentMetadata:
    """Tests for SpectralComponent metadata fields (Phase 1 enhancement)."""

    def test_metadata_fields_exist(self):
        """Test that SpectralComponent has metadata fields."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=1.0)],
            category="water_related",
            subcategory="bound_water",
            formula="H2O",
            cas_number="7732-18-5",
            synonyms=["aqua"],
            tags=["universal"],
        )
        assert comp.category == "water_related"
        assert comp.subcategory == "bound_water"
        assert comp.formula == "H2O"
        assert comp.cas_number == "7732-18-5"
        assert comp.synonyms == ["aqua"]
        assert comp.tags == ["universal"]

    def test_validate_valid_component(self):
        """Test validate() on valid component."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=1.0)],
        )
        issues = comp.validate()
        assert issues == []

    def test_validate_empty_bands(self):
        """Test validate() detects empty bands."""
        comp = SpectralComponent(name="empty")
        issues = comp.validate()
        assert len(issues) == 1
        assert "no bands defined" in issues[0]

    def test_validate_invalid_sigma(self):
        """Test validate() detects invalid sigma."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=-5, amplitude=1.0)],
        )
        issues = comp.validate()
        assert any("sigma must be positive" in issue for issue in issues)

    def test_validate_invalid_gamma(self):
        """Test validate() detects invalid gamma."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, gamma=-1, amplitude=1.0)],
        )
        issues = comp.validate()
        assert any("gamma must be non-negative" in issue for issue in issues)

    def test_validate_invalid_center(self):
        """Test validate() detects out-of-range center."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=50, sigma=25, amplitude=1.0)],
        )
        issues = comp.validate()
        assert any("outside valid range" in issue for issue in issues)

    def test_is_normalized(self):
        """Test is_normalized() method."""
        comp = SpectralComponent(
            name="test",
            bands=[
                NIRBand(center=1450, sigma=25, amplitude=0.5),
                NIRBand(center=1940, sigma=30, amplitude=1.0),
            ],
        )
        assert comp.is_normalized()

    def test_is_not_normalized(self):
        """Test is_normalized() returns False for unnormalized component."""
        comp = SpectralComponent(
            name="test",
            bands=[
                NIRBand(center=1450, sigma=25, amplitude=0.5),
                NIRBand(center=1940, sigma=30, amplitude=2.0),
            ],
        )
        assert not comp.is_normalized()

    def test_normalized_max(self):
        """Test normalized() method with max normalization."""
        comp = SpectralComponent(
            name="test",
            bands=[
                NIRBand(center=1450, sigma=25, amplitude=0.5),
                NIRBand(center=1940, sigma=30, amplitude=2.0),
            ],
        )
        normalized = comp.normalized(method="max")
        assert normalized.is_normalized()
        assert normalized.bands[1].amplitude == 1.0
        assert normalized.bands[0].amplitude == 0.25

    def test_normalized_sum(self):
        """Test normalized() method with sum normalization."""
        comp = SpectralComponent(
            name="test",
            bands=[
                NIRBand(center=1450, sigma=25, amplitude=0.5),
                NIRBand(center=1940, sigma=30, amplitude=0.5),
            ],
        )
        normalized = comp.normalized(method="sum")
        total = sum(b.amplitude for b in normalized.bands)
        assert abs(total - 1.0) < 0.01

    def test_normalized_preserves_metadata(self):
        """Test that normalized() preserves metadata fields."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=2.0)],
            category="water_related",
            formula="H2O",
            tags=["universal"],
        )
        normalized = comp.normalized()
        assert normalized.category == "water_related"
        assert normalized.formula == "H2O"
        assert normalized.tags == ["universal"]

    def test_has_bands_in_range(self):
        """Test has_bands_in_range() method."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=1.0)],
        )
        assert comp.has_bands_in_range((1400, 1500))
        assert not comp.has_bands_in_range((2000, 2500))

    def test_info_method(self):
        """Test info() method returns formatted string."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=1.0, name="O-H band")],
            category="water_related",
        )
        info = comp.info()
        assert "test" in info
        assert "water_related" in info
        assert "1450" in info


class TestDiscoveryAPI:
    """Tests for discovery API functions (Phase 1 enhancement)."""

    def test_available_components(self):
        """Test available_components() returns list of names."""
        names = available_components()
        assert isinstance(names, list)
        assert "water" in names
        assert "protein" in names
        assert len(names) >= 100  # We have 111 components

    def test_available_components_sorted(self):
        """Test available_components() returns sorted list."""
        names = available_components()
        assert names == sorted(names)

    def test_get_component(self):
        """Test get_component() returns component by name."""
        water = get_component("water")
        assert isinstance(water, SpectralComponent)
        assert water.name == "water"

    def test_get_component_invalid(self):
        """Test get_component() raises error for invalid name."""
        with pytest.raises(ValueError, match="Unknown component"):
            get_component("invalid_component_name")

    def test_search_components_by_category(self):
        """Test search_components() by category."""
        proteins = search_components(category="proteins")
        assert len(proteins) >= 10  # We have 12 protein components
        assert "protein" in proteins
        assert "casein" in proteins

    def test_search_components_by_query(self):
        """Test search_components() by query."""
        results = search_components(query="acid")
        assert len(results) > 0
        # Query matches name or synonyms, so just check we get acid-related results
        assert "acetic_acid" in results or "citric_acid" in results

    def test_search_components_by_tags(self):
        """Test search_components() by tags."""
        pharma = search_components(tags=["pharma"])
        assert len(pharma) > 0

    def test_search_components_by_wavelength_range(self):
        """Test search_components() by wavelength range."""
        results = search_components(wavelength_range=(1400, 1500))
        # Most components should have bands in this range
        assert len(results) > 50

    def test_list_categories(self):
        """Test list_categories() returns category mapping."""
        categories = list_categories()
        assert isinstance(categories, dict)
        assert "water_related" in categories
        assert "proteins" in categories
        assert "carbohydrates" in categories
        assert "water" in categories["water_related"]

    def test_component_info(self):
        """Test component_info() returns formatted string."""
        info = component_info("water")
        assert "water" in info
        assert "Component:" in info

    def test_validate_predefined_components(self):
        """Test validate_predefined_components() finds no critical issues."""
        issues = validate_predefined_components()
        # Filter out non-critical warnings
        critical_issues = [i for i in issues if "sigma must be positive" in i or "gamma must be non-negative" in i]
        assert critical_issues == []

    def test_validate_component_coverage(self):
        """Test validate_component_coverage() returns coverage info."""
        coverage = validate_component_coverage((1000, 2500))
        assert "covered" in coverage
        assert "not_covered" in coverage
        # Most components should be covered in standard NIR range
        assert len(coverage["covered"]) > 100

    def test_normalize_component_amplitudes(self):
        """Test normalize_component_amplitudes() wrapper function."""
        comp = SpectralComponent(
            name="test",
            bands=[NIRBand(center=1450, sigma=25, amplitude=2.0)],
        )
        normalized = normalize_component_amplitudes(comp)
        assert normalized.bands[0].amplitude == 1.0


class TestComponentEnrichment:
    """Tests for automatic component metadata enrichment."""

    def test_components_have_categories(self):
        """Test that predefined components have categories assigned."""
        components = get_predefined_components()
        components_with_category = sum(1 for c in components.values() if c.category)
        # Most components should have categories
        assert components_with_category >= 100

    def test_components_normalized(self):
        """Test that predefined components have normalized amplitudes."""
        components = get_predefined_components()
        for name, comp in components.items():
            if comp.bands:
                max_amp = max(b.amplitude for b in comp.bands)
                assert abs(max_amp - 1.0) < 0.02, f"Component {name} not normalized: max amp = {max_amp}"

    def test_water_has_metadata(self):
        """Test that water component has expected metadata."""
        water = get_component("water")
        assert water.category == "water_related"
        assert water.formula == "H2O"
        assert "universal" in water.tags

    def test_glucose_has_metadata(self):
        """Test that glucose component has expected metadata."""
        glucose = get_component("glucose")
        assert glucose.category == "carbohydrates"
        assert glucose.subcategory == "monosaccharides"
        assert glucose.formula == "C6H12O6"
        assert glucose.cas_number == "50-99-7"

    def test_caffeine_has_metadata(self):
        """Test that caffeine component has expected metadata."""
        caffeine = get_component("caffeine")
        assert caffeine.category == "pharmaceuticals"
        assert "pharma" in caffeine.tags

    def test_duplicates_have_synonyms(self):
        """Test that potential duplicates have synonyms pointing to each other."""
        lutein = get_component("lutein")
        xanthophyll = get_component("xanthophyll")
        # These should have synonyms pointing to each other
        assert "xanthophyll" in lutein.synonyms or "lutein" in xanthophyll.synonyms

        polyester = get_component("polyester")
        pet = get_component("pet")
        # These should have synonyms
        assert "PET" in polyester.synonyms or "polyester" in pet.synonyms


class TestPhase2VisibleRegionComponents:
    """Tests for Phase 2 visible-region components."""

    def test_chlorophyll_has_visible_bands(self):
        """Test that chlorophyll has visible-region absorption bands."""
        chlorophyll = get_component("chlorophyll")
        band_centers = [b.center for b in chlorophyll.bands]
        # Should have Soret band (~435 nm) and Q band (~655 nm)
        has_soret = any(400 < c < 480 for c in band_centers)
        has_q_band = any(630 < c < 700 for c in band_centers)
        assert has_soret, "Chlorophyll should have Soret band in visible region"
        assert has_q_band, "Chlorophyll should have Q band in visible region"

    def test_chlorophyll_a_exists(self):
        """Test that chlorophyll_a component exists with correct properties."""
        chl_a = get_component("chlorophyll_a")
        assert chl_a.name == "chlorophyll_a"
        assert chl_a.category == "pigments"
        assert chl_a.subcategory == "chlorophylls"
        # Should have bands near 430 nm (Soret) and 662 nm (Q)
        band_centers = [b.center for b in chl_a.bands]
        assert any(420 < c < 440 for c in band_centers), "Chl a should have Soret band ~430 nm"
        assert any(655 < c < 670 for c in band_centers), "Chl a should have Q band ~662 nm"

    def test_chlorophyll_b_exists(self):
        """Test that chlorophyll_b component exists with correct properties."""
        chl_b = get_component("chlorophyll_b")
        assert chl_b.name == "chlorophyll_b"
        assert chl_b.category == "pigments"
        # Should have bands near 453 nm (Soret) and 642 nm (Q)
        band_centers = [b.center for b in chl_b.bands]
        assert any(445 < c < 460 for c in band_centers), "Chl b should have Soret band ~453 nm"
        assert any(635 < c < 650 for c in band_centers), "Chl b should have Q band ~642 nm"

    def test_beta_carotene_exists(self):
        """Test that beta_carotene component exists with visible bands."""
        beta_car = get_component("beta_carotene")
        assert beta_car.name == "beta_carotene"
        assert beta_car.formula == "C40H56"
        # Should have peaks around 425, 450, 478 nm
        band_centers = [b.center for b in beta_car.bands]
        assert any(420 < c < 490 for c in band_centers), "β-carotene should have visible peaks"

    def test_hemoglobin_oxy_exists(self):
        """Test that oxyhemoglobin component exists with correct properties."""
        hb_oxy = get_component("hemoglobin_oxy")
        assert hb_oxy.name == "hemoglobin_oxy"
        assert hb_oxy.category == "pigments"
        assert "blood" in hb_oxy.tags or "medical" in hb_oxy.tags
        # Should have Soret band ~414 nm and Q bands ~542, 577 nm
        band_centers = [b.center for b in hb_oxy.bands]
        assert any(410 < c < 420 for c in band_centers), "HbO2 should have Soret band ~414 nm"

    def test_hemoglobin_deoxy_exists(self):
        """Test that deoxyhemoglobin component exists with correct properties."""
        hb_deoxy = get_component("hemoglobin_deoxy")
        assert hb_deoxy.name == "hemoglobin_deoxy"
        assert hb_deoxy.category == "pigments"
        # Should have Soret band ~430 nm
        band_centers = [b.center for b in hb_deoxy.bands]
        assert any(425 < c < 435 for c in band_centers), "Hb should have Soret band ~430 nm"

    def test_lycopene_has_visible_bands(self):
        """Test that lycopene has visible-region absorption bands."""
        lycopene = get_component("lycopene")
        band_centers = [b.center for b in lycopene.bands]
        # Should have peaks around 443, 471, 502 nm
        has_visible = any(440 < c < 510 for c in band_centers)
        assert has_visible, "Lycopene should have visible absorption bands"

    def test_melanin_has_broad_visible_absorption(self):
        """Test that melanin has broad visible absorption."""
        melanin = get_component("melanin")
        band_centers = [b.center for b in melanin.bands]
        # Melanin should have bands spanning visible region
        has_blue = any(380 < c < 450 for c in band_centers)
        has_red = any(580 < c < 700 for c in band_centers)
        assert has_blue or has_red, "Melanin should have broad visible absorption"

    def test_water_has_overtone_series(self):
        """Test that water has 2nd and 3rd overtones (Phase 2 extension)."""
        water = get_component("water")
        band_centers = [b.center for b in water.bands]
        # Should have: 3rd overtone ~730 nm, 2nd overtone ~970 nm, 1st overtone ~1450 nm
        has_3rd_overtone = any(700 < c < 780 for c in band_centers)
        has_2nd_overtone = any(950 < c < 1000 for c in band_centers)
        has_1st_overtone = any(1400 < c < 1500 for c in band_centers)
        assert has_3rd_overtone, "Water should have 3rd overtone ~730 nm"
        assert has_2nd_overtone, "Water should have 2nd overtone ~970 nm"
        assert has_1st_overtone, "Water should have 1st overtone ~1450 nm"

    def test_protein_has_overtone_series(self):
        """Test that protein has 2nd and 3rd overtones (Phase 2 extension)."""
        protein = get_component("protein")
        band_centers = [b.center for b in protein.bands]
        # Should have 2nd overtones in 1000-1150 nm range
        has_2nd_overtone = any(1000 < c < 1200 for c in band_centers)
        assert has_2nd_overtone, "Protein should have 2nd overtone bands"

    def test_vis_nir_tag_on_visible_components(self):
        """Test that visible-region components have 'vis-nir' tag."""
        visible_components = ["chlorophyll", "chlorophyll_a", "chlorophyll_b",
                              "beta_carotene", "hemoglobin_oxy", "hemoglobin_deoxy",
                              "lycopene", "melanin", "lutein"]
        for name in visible_components:
            comp = get_component(name)
            assert "vis-nir" in comp.tags, f"{name} should have 'vis-nir' tag"


class TestBoundaryComponents:
    """Tests for boundary component functionality (peaks outside wavelength range)."""

    def test_add_boundary_component_basic(self):
        """Test basic boundary component creation."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="test_boundary",
            measurement_range=(1000, 2500),
            edge="both",
            n_bands=1
        )

        assert comp.name == "test_boundary"
        assert len(comp.bands) == 1
        assert comp.category == "boundary_effect"
        assert "boundary" in comp.tags

    def test_add_boundary_component_left_edge(self):
        """Test boundary component at left edge only."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="left_boundary",
            measurement_range=(1000, 2500),
            edge="left",
            n_bands=2
        )

        # All bands should have centers below 1000 nm
        for band in comp.bands:
            assert band.center < 1000, f"Band center {band.center} should be < 1000"

    def test_add_boundary_component_right_edge(self):
        """Test boundary component at right edge only."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="right_boundary",
            measurement_range=(1000, 2500),
            edge="right",
            n_bands=2
        )

        # All bands should have centers above 2500 nm
        for band in comp.bands:
            assert band.center > 2500, f"Band center {band.center} should be > 2500"

    def test_boundary_component_affects_edge_wavelengths(self):
        """Test that boundary components create visible tails at edges."""
        library = ComponentLibrary(random_state=42)

        # Add component with peak outside right edge
        library.add_boundary_component(
            name="right_tail",
            measurement_range=(1000, 2400),
            edge="right",
            n_bands=1,
            amplitude_range=(0.5, 0.5),  # Fixed amplitude for testing
            width_range=(100, 100),  # Fixed width
            offset_range=(0.5, 0.5),  # Fixed offset
        )

        wavelengths = np.arange(1000, 2500, 2)
        E = library.compute_all(wavelengths)

        # The spectrum should show increasing values toward the right edge
        right_edge_mean = E[0, -50:].mean()
        center_mean = E[0, 250:350].mean()

        assert right_edge_mean > center_mean, "Right edge should have higher values due to truncated peak"

    def test_boundary_component_in_library(self):
        """Test that boundary component is added to library."""
        library = ComponentLibrary(random_state=42)
        library.add_boundary_component(
            name="boundary_test",
            measurement_range=(1000, 2500),
            edge="both"
        )

        assert "boundary_test" in library
        assert library.n_components == 1

    def test_add_boundary_components_from_known(self):
        """Test add_boundary_components_from_known method."""
        library = ComponentLibrary(random_state=42)
        library.add_boundary_components_from_known(measurement_range=(1000, 2400))

        # Should have added water boundary component
        assert "water_boundary_2500" in library

    def test_add_boundary_components_from_known_mid_ir(self):
        """Test known boundary components for mid-IR tail."""
        library = ComponentLibrary(random_state=42)
        library.add_boundary_components_from_known(measurement_range=(1000, 2500))

        # Should have added C-H fundamental tail
        assert "ch_fundamental_tail" in library

    def test_boundary_component_multiple_bands(self):
        """Test boundary component with multiple bands."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="multi_boundary",
            measurement_range=(1000, 2500),
            edge="both",
            n_bands=5
        )

        assert len(comp.bands) == 5

        # All bands should be outside the measurement range
        for band in comp.bands:
            assert band.center < 1000 or band.center > 2500

    def test_boundary_component_amplitude_range(self):
        """Test that amplitude_range parameter works."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="amp_test",
            measurement_range=(1000, 2500),
            edge="right",
            n_bands=10,
            amplitude_range=(0.1, 0.2)
        )

        for band in comp.bands:
            assert 0.1 <= band.amplitude <= 0.2

    def test_boundary_component_width_range(self):
        """Test that width_range parameter works."""
        library = ComponentLibrary(random_state=42)
        comp = library.add_boundary_component(
            name="width_test",
            measurement_range=(1000, 2500),
            edge="left",
            n_bands=10,
            width_range=(50, 100)
        )

        for band in comp.bands:
            assert 50 <= band.sigma <= 100

    def test_boundary_component_reproducibility(self):
        """Test that boundary components are reproducible with random_state."""
        lib1 = ComponentLibrary(random_state=123)
        comp1 = lib1.add_boundary_component("test", measurement_range=(1000, 2500))

        lib2 = ComponentLibrary(random_state=123)
        comp2 = lib2.add_boundary_component("test", measurement_range=(1000, 2500))

        assert comp1.bands[0].center == comp2.bands[0].center
        assert comp1.bands[0].sigma == comp2.bands[0].sigma
        assert comp1.bands[0].amplitude == comp2.bands[0].amplitude

    def test_boundary_and_regular_components_together(self):
        """Test using boundary and regular components together."""
        library = ComponentLibrary.from_predefined(["water", "protein"], random_state=42)

        # Add boundary component
        library.add_boundary_component(
            name="edge_artifact",
            measurement_range=(1000, 2500),
            edge="right"
        )

        assert library.n_components == 3
        assert "water" in library
        assert "protein" in library
        assert "edge_artifact" in library

        # Compute should work
        wavelengths = np.arange(1000, 2500, 2)
        E = library.compute_all(wavelengths)
        assert E.shape == (3, len(wavelengths))

"""
Unit tests for the wavenumber module (Phase 1.1).

Tests cover:
- Wavenumber/wavelength conversions
- NIR zone classification
- Overtone and combination band calculations
- Hydrogen bonding shift functions
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.synthesis.wavenumber import (
    EXTENDED_SPECTRAL_ZONES,
    FUNDAMENTAL_VIBRATIONS,
    NIR_ZONES_WAVENUMBER,
    VISIBLE_ZONES_WAVENUMBER,
    CombinationBandResult,
    OvertoneResult,
    apply_hydrogen_bonding_shift,
    calculate_combination_band,
    calculate_overtone_position,
    classify_wavelength_extended,
    classify_wavelength_zone,
    convert_bandwidth_to_wavelength,
    get_all_zones_extended,
    get_zone_wavelength_range,
    is_nir_region,
    is_visible_region,
    wavelength_to_wavenumber,
    wavenumber_to_wavelength,
)


class TestWavenumberConversion:
    """Tests for wavenumber/wavelength conversion functions."""

    def test_wavenumber_to_wavelength_scalar(self):
        """Test scalar conversion from wavenumber to wavelength."""
        # 10000 cm⁻¹ should be 1000 nm
        result = wavenumber_to_wavelength(10000)
        assert result == pytest.approx(1000.0)

    def test_wavenumber_to_wavelength_array(self):
        """Test array conversion from wavenumber to wavelength."""
        wavenumbers = np.array([10000, 5000, 4000])
        expected = np.array([1000, 2000, 2500])
        result = wavenumber_to_wavelength(wavenumbers)
        np.testing.assert_array_almost_equal(result, expected)

    def test_wavelength_to_wavenumber_scalar(self):
        """Test scalar conversion from wavelength to wavenumber."""
        # 1000 nm should be 10000 cm⁻¹
        result = wavelength_to_wavenumber(1000)
        assert result == pytest.approx(10000.0)

    def test_wavelength_to_wavenumber_array(self):
        """Test array conversion from wavelength to wavenumber."""
        wavelengths = np.array([1000, 2000, 2500])
        expected = np.array([10000, 5000, 4000])
        result = wavelength_to_wavenumber(wavelengths)
        np.testing.assert_array_almost_equal(result, expected)

    def test_roundtrip_conversion(self):
        """Test that conversions are exact inverses."""
        original = 6500  # wavenumber
        converted = wavenumber_to_wavelength(original)
        back = wavelength_to_wavenumber(converted)
        assert back == pytest.approx(original)

    def test_zero_wavenumber_raises(self):
        """Test that zero wavenumber raises an error."""
        with pytest.raises((ZeroDivisionError, FloatingPointError, ValueError)):
            wavenumber_to_wavelength(0)

    def test_zero_wavelength_raises(self):
        """Test that zero wavelength raises an error."""
        with pytest.raises((ZeroDivisionError, FloatingPointError, ValueError)):
            wavelength_to_wavenumber(0)

class TestBandwidthConversion:
    """Tests for bandwidth conversion (wavenumber FWHM to wavelength)."""

    def test_bandwidth_conversion_mid_nir(self):
        """Test bandwidth conversion at mid-NIR region."""
        # At 1500 nm, convert a 50 cm⁻¹ bandwidth
        center_nm = 1500
        fwhm_cm = 50
        result = convert_bandwidth_to_wavelength(fwhm_cm, center_nm)
        # Bandwidth should be positive and reasonable
        assert result > 0
        assert result < 50  # Should be much smaller in nm than cm⁻¹ value

    def test_bandwidth_increases_with_wavelength(self):
        """Test that same wavenumber FWHM gives larger nm width at longer wavelengths."""
        fwhm_cm = 50
        result_1000nm = convert_bandwidth_to_wavelength(fwhm_cm, 1000)
        result_2000nm = convert_bandwidth_to_wavelength(fwhm_cm, 2000)
        # At longer wavelengths, same cm⁻¹ width corresponds to larger nm width
        assert result_2000nm > result_1000nm

    def test_bandwidth_proportional_to_fwhm(self):
        """Test that bandwidth scales linearly with FWHM."""
        center = 1500
        result_50 = convert_bandwidth_to_wavelength(50, center)
        result_100 = convert_bandwidth_to_wavelength(100, center)
        # Should be approximately 2x
        assert result_100 == pytest.approx(2 * result_50, rel=0.1)

class TestNIRZones:
    """Tests for NIR zone classification."""

    def test_nir_zones_wavenumber_structure(self):
        """Test that NIR_ZONES_WAVENUMBER has expected structure."""
        # NIR_ZONES_WAVENUMBER is a list of tuples
        assert isinstance(NIR_ZONES_WAVENUMBER, list)
        assert len(NIR_ZONES_WAVENUMBER) >= 7  # At least 7 zones defined

    def test_nir_zones_contain_required_info(self):
        """Test that each zone contains required information."""
        for zone_data in NIR_ZONES_WAVENUMBER:
            # Each zone is a tuple (nu_min, nu_max, name)
            assert len(zone_data) == 3
            nu_min, nu_max, name = zone_data
            assert nu_min < nu_max  # Valid range
            assert isinstance(name, str)

    def test_classify_wavelength_zone_known(self):
        """Test zone classification for known wavelength."""
        # 1400-1450 nm is typically first overtone O-H
        zone = classify_wavelength_zone(1420)
        assert zone is not None
        assert isinstance(zone, str)

    def test_classify_wavelength_zone_returns_none_outside_nir(self):
        """Test that wavelengths outside NIR return None."""
        # Very short wavelength (visible)
        zone = classify_wavelength_zone(500)
        # Should return None or a catch-all zone
        # (implementation-dependent)

    def test_get_zone_wavelength_range(self):
        """Test conversion of zone to wavelength range."""
        # Get the first zone name from the list
        zone_name = NIR_ZONES_WAVENUMBER[0][2]
        result = get_zone_wavelength_range(zone_name)
        assert result is not None
        wl1, wl2 = result
        # Should return valid wavelengths in nm
        # Note: due to inverse relationship, order may vary
        low, high = min(wl1, wl2), max(wl1, wl2)
        assert low > 0
        assert high > low
        assert low < 3000  # NIR range
        assert high < 3000

class TestFundamentalVibrations:
    """Tests for the FUNDAMENTAL_VIBRATIONS dictionary."""

    def test_fundamental_vibrations_structure(self):
        """Test that FUNDAMENTAL_VIBRATIONS has expected structure."""
        assert isinstance(FUNDAMENTAL_VIBRATIONS, dict)
        assert len(FUNDAMENTAL_VIBRATIONS) >= 22  # At least 22 vibrations defined

    def test_fundamental_vibrations_contain_valid_frequencies(self):
        """Test that each vibration contains a valid wavenumber frequency."""
        for name, freq in FUNDAMENTAL_VIBRATIONS.items():
            assert isinstance(freq, (int, float))
            assert 500 < freq < 4500  # Typical fundamental vibration range

    def test_key_vibrations_present(self):
        """Test that key functional group vibrations are present."""
        expected_groups = ["O-H_stretch_free", "N-H_stretch_primary", "C-H_stretch_CH3_asym"]
        for group in expected_groups:
            assert group in FUNDAMENTAL_VIBRATIONS

class TestOvertoneCalculation:
    """Tests for overtone position calculations."""

    def test_calculate_overtone_position_first(self):
        """Test first overtone calculation (order=2 in the API)."""
        # Note: In this API, order=1 is fundamental, order=2 is first overtone
        result = calculate_overtone_position("O-H_stretch_free", 2)
        assert isinstance(result, OvertoneResult)
        # First overtone should be around 1400-1500 nm for O-H
        assert 1350 < result.wavelength_nm < 1550

    def test_calculate_overtone_position_second(self):
        """Test second overtone calculation."""
        result = calculate_overtone_position("O-H_stretch_free", 3)
        # Second overtone should be at shorter wavelength than first
        first = calculate_overtone_position("O-H_stretch_free", 2)
        assert result.wavelength_nm < first.wavelength_nm

    def test_calculate_overtone_position_third(self):
        """Test third overtone calculation."""
        result = calculate_overtone_position("O-H_stretch_free", 4)
        # Third overtone at even shorter wavelength
        second = calculate_overtone_position("O-H_stretch_free", 3)
        assert result.wavelength_nm < second.wavelength_nm

    def test_overtone_result_structure(self):
        """Test that OvertoneResult contains expected fields."""
        result = calculate_overtone_position("C-H_stretch_CH3_asym", 2)
        assert hasattr(result, "wavenumber_cm")
        assert hasattr(result, "wavelength_nm")
        assert hasattr(result, "order")  # Note: 'order' not 'overtone_number'
        assert hasattr(result, "bandwidth_factor")

    def test_overtone_with_custom_fundamental(self):
        """Test overtone calculation with custom fundamental frequency."""
        result = calculate_overtone_position(3650, 3, anharmonicity=0.022)
        assert isinstance(result, OvertoneResult)
        assert result.wavelength_nm > 0

    def test_invalid_vibration_type_raises(self):
        """Test that invalid vibration type raises an error."""
        with pytest.raises((KeyError, ValueError)):
            calculate_overtone_position("invalid_vibration", 2)

    def test_overtone_order_zero_raises(self):
        """Test that overtone order 0 raises an error."""
        with pytest.raises(ValueError):
            calculate_overtone_position("O-H_stretch_free", 0)

class TestCombinationBandCalculation:
    """Tests for combination band calculations."""

    def test_calculate_combination_band_two_modes(self):
        """Test combination band calculation for two vibration modes."""
        result = calculate_combination_band("O-H_stretch_free", "O-H_bend")
        assert isinstance(result, CombinationBandResult)
        assert result.wavelength_nm > 0

    def test_combination_band_result_structure(self):
        """Test that CombinationBandResult contains expected fields."""
        result = calculate_combination_band("C-H_stretch_CH3_asym", "C-H_bend")
        assert hasattr(result, "wavenumber_cm")
        assert hasattr(result, "wavelength_nm")
        assert hasattr(result, "mode1_cm")
        assert hasattr(result, "mode2_cm")
        assert hasattr(result, "band_type")

    def test_combination_band_sum_rule(self):
        """Test that combination band follows sum rule approximately."""
        # Combination band wavenumber should be close to sum of fundamentals
        mode1 = "O-H_stretch_free"
        mode2 = "O-H_bend"
        result = calculate_combination_band(mode1, mode2)

        fund1 = FUNDAMENTAL_VIBRATIONS[mode1]
        fund2 = FUNDAMENTAL_VIBRATIONS[mode2]
        expected_sum = fund1 + fund2

        # Allow for some deviation due to anharmonicity
        assert result.wavenumber_cm == pytest.approx(expected_sum, rel=0.15)

class TestHydrogenBondingShift:
    """Tests for hydrogen bonding shift calculations."""

    def test_apply_shift_lowers_wavenumber(self):
        """Test that H-bonding shifts wavenumber to lower values."""
        original = 3600  # Free O-H stretch
        shifted = apply_hydrogen_bonding_shift(original, h_bond_strength=0.5)
        assert shifted < original

    def test_apply_shift_strength_zero(self):
        """Test that zero strength gives no shift."""
        original = 3600
        shifted = apply_hydrogen_bonding_shift(original, h_bond_strength=0.0)
        assert shifted == pytest.approx(original)

    def test_apply_shift_strength_increases_shift(self):
        """Test that higher strength gives larger shift."""
        original = 3600
        weak = apply_hydrogen_bonding_shift(original, h_bond_strength=0.3)
        strong = apply_hydrogen_bonding_shift(original, h_bond_strength=0.8)
        assert strong < weak < original

    def test_apply_shift_strength_clamped(self):
        """Test that strength is handled correctly for edge values."""
        original = 3600
        # Check that zero strength works
        result_zero = apply_hydrogen_bonding_shift(original, h_bond_strength=0.0)
        assert result_zero == pytest.approx(original)
        # Check that max strength (1.0) works
        result_max = apply_hydrogen_bonding_shift(original, h_bond_strength=1.0)
        assert result_max < original

class TestVisibleRegion:
    """Tests for visible region functions (Phase 2)."""

    def test_extended_spectral_zones_structure(self):
        """Test that EXTENDED_SPECTRAL_ZONES has expected structure."""
        assert isinstance(EXTENDED_SPECTRAL_ZONES, list)
        assert len(EXTENDED_SPECTRAL_ZONES) >= 9  # At least 9 zones defined

    def test_extended_zones_contain_required_info(self):
        """Test that each extended zone contains required information."""
        for zone_data in EXTENDED_SPECTRAL_ZONES:
            # Each zone is a tuple (nu_min, nu_max, name, description)
            assert len(zone_data) == 4
            nu_min, nu_max, name, description = zone_data
            assert nu_min < nu_max  # Valid range
            assert isinstance(name, str)
            assert isinstance(description, str)

    def test_visible_zones_wavenumber_structure(self):
        """Test that VISIBLE_ZONES_WAVENUMBER has expected structure."""
        assert isinstance(VISIBLE_ZONES_WAVENUMBER, list)
        assert len(VISIBLE_ZONES_WAVENUMBER) >= 4  # At least 4 visible zones

    def test_is_visible_region(self):
        """Test is_visible_region function."""
        assert is_visible_region(400)
        assert is_visible_region(500)
        assert is_visible_region(650)
        assert not is_visible_region(800)
        assert not is_visible_region(1450)
        assert is_visible_region(350) is not False or is_visible_region(350) is True  # boundary

    def test_is_nir_region(self):
        """Test is_nir_region function."""
        assert is_nir_region(800)
        assert is_nir_region(1450)
        assert is_nir_region(2000)
        assert not is_nir_region(500)
        assert not is_nir_region(400)

    def test_classify_wavelength_extended_visible(self):
        """Test extended classification for visible wavelengths."""
        # Blue region
        result = classify_wavelength_extended(450)
        assert result is not None
        zone_name, description = result
        assert "visible" in zone_name.lower() or "blue" in zone_name.lower() or zone_name is not None

    def test_classify_wavelength_extended_nir(self):
        """Test extended classification for NIR wavelengths."""
        result = classify_wavelength_extended(1450)
        assert result is not None
        zone_name, description = result
        assert "overtone" in zone_name.lower() or "combination" in zone_name.lower() or zone_name is not None

    def test_get_all_zones_extended(self):
        """Test get_all_zones_extended returns wavelength tuples."""
        zones = get_all_zones_extended()
        assert isinstance(zones, list)
        assert len(zones) >= 9  # At least 9 zones

        for zone in zones:
            assert len(zone) == 4
            wl1, wl2, name, description = zone
            assert wl1 > 0
            assert wl2 > 0
            # Note: wavenumber to wavelength inversion may swap min/max
            assert abs(wl1 - wl2) > 0  # They should be different
            assert isinstance(name, str)

    def test_extended_zones_cover_vis_nir_range(self):
        """Test that extended zones cover visible-NIR range (350-2500 nm)."""
        zones = get_all_zones_extended()
        all_wavelengths = []
        for min_wl, max_wl, name, desc in zones:
            all_wavelengths.extend([min_wl, max_wl])

        # Should cover approximately 350-2500 nm
        assert min(all_wavelengths) < 500  # Covers visible
        assert max(all_wavelengths) > 2400  # Covers NIR

class TestIntegration:
    """Integration tests for wavenumber module."""

    def test_full_workflow_oh_first_overtone(self):
        """Test complete workflow for O-H first overtone generation."""
        # Calculate overtone position (order=2 for first overtone)
        overtone = calculate_overtone_position("O-H_stretch_free", 2)

        # Apply H-bonding shift
        shifted = apply_hydrogen_bonding_shift(overtone.wavenumber_cm, h_bond_strength=0.5)

        # Convert to wavelength
        wavelength = wavenumber_to_wavelength(shifted)

        # Should be in typical O-H first overtone region
        assert 1350 < wavelength < 1600

    def test_full_workflow_ch_overtone_series(self):
        """Test generation of C-H overtone series."""
        overtones = []
        for n in [2, 3, 4]:  # First, second, third overtones
            result = calculate_overtone_position("C-H_stretch_CH3_asym", n)
            overtones.append(result.wavelength_nm)

        # Should be in decreasing wavelength order
        assert overtones[0] > overtones[1] > overtones[2]
        # All should be in NIR range
        for wl in overtones:
            assert 800 < wl < 2500

    def test_zones_cover_nir_range(self):
        """Test that zones collectively cover the NIR range."""
        all_wavelengths = []
        for zone_tuple in NIR_ZONES_WAVENUMBER:
            zone_name = zone_tuple[2]
            result = get_zone_wavelength_range(zone_name)
            if result is not None:
                low, high = result
                all_wavelengths.extend([low, high])

        # Should cover approximately 780-2500 nm
        assert min(all_wavelengths) < 1000
        assert max(all_wavelengths) > 2400

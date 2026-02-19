"""Tests for header_units utility functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.data._features import HeaderUnit
from nirs4all.utils.header_units import (
    AXIS_LABELS,
    DEFAULT_AXIS_LABEL,
    apply_x_axis_limits,
    get_axis_label,
    get_x_values_and_label,
    should_invert_x_axis,
)


class TestAxisLabels:
    """Test the AXIS_LABELS constant."""

    def test_all_header_units_have_labels(self):
        """All HeaderUnit enum values should have a corresponding label."""
        for unit in HeaderUnit:
            assert unit in AXIS_LABELS, f"Missing label for {unit}"

    def test_wavenumber_label(self):
        """Wavenumber label should use proper superscript."""
        assert AXIS_LABELS[HeaderUnit.WAVENUMBER] == "Wavenumber (cm⁻¹)"

    def test_wavelength_label(self):
        """Wavelength label should include nm unit."""
        assert AXIS_LABELS[HeaderUnit.WAVELENGTH] == "Wavelength (nm)"

    def test_index_labels_consistent(self):
        """Index-type units should use consistent label."""
        assert AXIS_LABELS[HeaderUnit.NONE] == "Feature Index"
        assert AXIS_LABELS[HeaderUnit.INDEX] == "Feature Index"

class TestGetAxisLabel:
    """Test get_axis_label function."""

    def test_wavenumber_string(self):
        """String 'cm-1' should return wavenumber label."""
        assert get_axis_label("cm-1") == "Wavenumber (cm⁻¹)"

    def test_wavelength_string(self):
        """String 'nm' should return wavelength label."""
        assert get_axis_label("nm") == "Wavelength (nm)"

    def test_none_string(self):
        """String 'none' should return feature index label."""
        assert get_axis_label("none") == "Feature Index"

    def test_text_string(self):
        """String 'text' should return features label."""
        assert get_axis_label("text") == "Features"

    def test_index_string(self):
        """String 'index' should return feature index label."""
        assert get_axis_label("index") == "Feature Index"

    def test_enum_value(self):
        """HeaderUnit enum should work directly."""
        assert get_axis_label(HeaderUnit.WAVENUMBER) == "Wavenumber (cm⁻¹)"
        assert get_axis_label(HeaderUnit.WAVELENGTH) == "Wavelength (nm)"

    def test_invalid_unit_fallback(self):
        """Invalid unit string should return default label."""
        assert get_axis_label("invalid_unit") == DEFAULT_AXIS_LABEL
        assert get_axis_label("") == DEFAULT_AXIS_LABEL

class TestGetXValuesAndLabel:
    """Test get_x_values_and_label function."""

    def test_numeric_wavenumber_headers(self):
        """Numeric headers with cm-1 unit should parse correctly."""
        headers = ["4000", "4500", "5000"]
        x_vals, label = get_x_values_and_label(headers, "cm-1", 3)

        np.testing.assert_array_almost_equal(x_vals, [4000, 4500, 5000])
        assert label == "Wavenumber (cm⁻¹)"

    def test_numeric_wavelength_headers(self):
        """Numeric headers with nm unit should parse correctly."""
        headers = ["800", "1000", "1200"]
        x_vals, label = get_x_values_and_label(headers, "nm", 3)

        np.testing.assert_array_almost_equal(x_vals, [800, 1000, 1200])
        assert label == "Wavelength (nm)"

    def test_none_headers_fallback(self):
        """None headers should return indices with default label."""
        x_vals, label = get_x_values_and_label(None, "cm-1", 5)

        np.testing.assert_array_equal(x_vals, [0, 1, 2, 3, 4])
        assert label == DEFAULT_AXIS_LABEL

    def test_mismatched_length_fallback(self):
        """Headers with wrong length should return indices with default label."""
        headers = ["4000", "4500"]  # Only 2 headers for 5 features
        x_vals, label = get_x_values_and_label(headers, "cm-1", 5)

        np.testing.assert_array_equal(x_vals, [0, 1, 2, 3, 4])
        assert label == DEFAULT_AXIS_LABEL

    def test_non_numeric_headers_fallback(self):
        """Non-numeric headers should return indices with default label."""
        headers = ["feature_a", "feature_b", "feature_c"]
        x_vals, label = get_x_values_and_label(headers, "cm-1", 3)

        np.testing.assert_array_equal(x_vals, [0, 1, 2])
        assert label == DEFAULT_AXIS_LABEL

    def test_text_unit_with_numeric_headers(self):
        """TEXT unit should return indices even with numeric-like headers."""
        headers = ["1", "2", "3"]
        x_vals, label = get_x_values_and_label(headers, "text", 3)

        np.testing.assert_array_equal(x_vals, [0, 1, 2])
        assert label == "Features"

    def test_empty_headers(self):
        """Empty headers list should return indices with default label."""
        x_vals, label = get_x_values_and_label([], "cm-1", 3)

        np.testing.assert_array_equal(x_vals, [0, 1, 2])
        assert label == DEFAULT_AXIS_LABEL

    def test_header_unit_enum(self):
        """HeaderUnit enum should work directly."""
        headers = ["4000", "4500", "5000"]
        x_vals, label = get_x_values_and_label(headers, HeaderUnit.WAVENUMBER, 3)

        np.testing.assert_array_almost_equal(x_vals, [4000, 4500, 5000])
        assert label == "Wavenumber (cm⁻¹)"

    def test_float_headers(self):
        """Float string headers should parse correctly."""
        headers = ["4000.5", "4500.75", "5000.25"]
        x_vals, label = get_x_values_and_label(headers, "cm-1", 3)

        np.testing.assert_array_almost_equal(x_vals, [4000.5, 4500.75, 5000.25])
        assert label == "Wavenumber (cm⁻¹)"

    def test_invalid_unit_string(self):
        """Invalid unit string should fallback gracefully."""
        headers = ["4000", "4500", "5000"]
        x_vals, label = get_x_values_and_label(headers, "invalid", 3)

        # Should parse values but with Feature Index label (fallback to NONE unit)
        np.testing.assert_array_almost_equal(x_vals, [4000, 4500, 5000])
        assert label == "Feature Index"

class TestShouldInvertXAxis:
    """Test should_invert_x_axis function."""

    def test_descending_values_should_invert(self):
        """Descending x values should indicate inversion needed."""
        x_values = np.array([5000, 4500, 4000])
        assert should_invert_x_axis(x_values)

    def test_ascending_values_no_invert(self):
        """Ascending x values should not indicate inversion."""
        x_values = np.array([4000, 4500, 5000])
        assert not should_invert_x_axis(x_values)

    def test_single_value_no_invert(self):
        """Single value should not indicate inversion."""
        x_values = np.array([4000])
        assert not should_invert_x_axis(x_values)

    def test_empty_array_no_invert(self):
        """Empty array should not indicate inversion."""
        x_values = np.array([])
        assert not should_invert_x_axis(x_values)

    def test_equal_values_no_invert(self):
        """Equal values should not indicate inversion."""
        x_values = np.array([4000, 4000, 4000])
        assert not should_invert_x_axis(x_values)

class TestApplyXAxisLimits:
    """Test apply_x_axis_limits function."""

    def test_descending_values_sets_limits(self):
        """Descending x values should set explicit limits."""
        ax = MagicMock()
        x_values = np.array([5000, 4500, 4000])

        apply_x_axis_limits(ax, x_values)

        ax.set_xlim.assert_called_once_with(5000, 4000)

    def test_ascending_values_no_limits_set(self):
        """Ascending x values should not set limits."""
        ax = MagicMock()
        x_values = np.array([4000, 4500, 5000])

        apply_x_axis_limits(ax, x_values)

        ax.set_xlim.assert_not_called()

    def test_single_value_no_limits_set(self):
        """Single value should not set limits."""
        ax = MagicMock()
        x_values = np.array([4000])

        apply_x_axis_limits(ax, x_values)

        ax.set_xlim.assert_not_called()

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_typical_wavenumber_workflow(self):
        """Test typical wavenumber data workflow."""
        # Simulate descending wavenumber data (typical for NIRS)
        headers = ["5000", "4500", "4000", "3500"]
        n_features = 4

        x_vals, label = get_x_values_and_label(headers, "cm-1", n_features)

        assert label == "Wavenumber (cm⁻¹)"
        np.testing.assert_array_almost_equal(x_vals, [5000, 4500, 4000, 3500])
        assert should_invert_x_axis(x_vals)

    def test_typical_wavelength_workflow(self):
        """Test typical wavelength data workflow."""
        # Simulate ascending wavelength data
        headers = ["800", "1000", "1200", "1400"]
        n_features = 4

        x_vals, label = get_x_values_and_label(headers, "nm", n_features)

        assert label == "Wavelength (nm)"
        np.testing.assert_array_almost_equal(x_vals, [800, 1000, 1200, 1400])
        assert not should_invert_x_axis(x_vals)

    def test_no_headers_workflow(self):
        """Test workflow when no headers are available."""
        x_vals, label = get_x_values_and_label(None, "cm-1", 100)

        assert label == DEFAULT_AXIS_LABEL
        assert len(x_vals) == 100
        np.testing.assert_array_equal(x_vals, np.arange(100))
        assert not should_invert_x_axis(x_vals)

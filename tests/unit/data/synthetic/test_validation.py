"""
Unit tests for validation utilities.
"""

import numpy as np
import pytest

from nirs4all.synthesis.validation import (
    ValidationError,
    validate_concentrations,
    validate_spectra,
    validate_synthetic_output,
    validate_wavelengths,
)


class TestValidateSpectra:
    """Tests for validate_spectra function."""

    def test_valid_spectra(self):
        """Test validation of valid spectra."""
        X = np.random.randn(100, 500)
        warnings = validate_spectra(X, expected_shape=(100, 500))
        assert warnings == []

    def test_wrong_type(self):
        """Test error on wrong type."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_spectra([[1, 2], [3, 4]])

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValidationError, match="Expected 2D array"):
            validate_spectra(np.array([1, 2, 3]))

    def test_wrong_shape(self):
        """Test error on shape mismatch."""
        X = np.random.randn(100, 500)
        with pytest.raises(ValidationError, match="Shape mismatch"):
            validate_spectra(X, expected_shape=(50, 500))

    def test_nan_values(self):
        """Test error on NaN values."""
        X = np.random.randn(100, 500)
        X[10, 20] = np.nan
        with pytest.raises(ValidationError, match="NaN values"):
            validate_spectra(X)

    def test_inf_values(self):
        """Test error on Inf values."""
        X = np.random.randn(100, 500)
        X[10, 20] = np.inf
        with pytest.raises(ValidationError, match="Inf values"):
            validate_spectra(X)

    def test_negative_values_warning(self):
        """Test warning on negative values."""
        X = np.random.randn(100, 500)  # Will have negatives
        warnings = validate_spectra(X, check_positive=True)
        assert len(warnings) > 0
        assert "negative" in warnings[0].lower()

    def test_value_range_warning(self):
        """Test warning when values out of expected range."""
        X = np.random.randn(100, 500) * 10  # Large values
        warnings = validate_spectra(X, value_range=(-1, 1))
        assert len(warnings) > 0

class TestValidateConcentrations:
    """Tests for validate_concentrations function."""

    def test_valid_concentrations(self):
        """Test validation of valid concentrations."""
        C = np.random.dirichlet(np.ones(5), size=100)
        warnings = validate_concentrations(
            C, n_samples=100, n_components=5, check_normalized=True
        )
        assert warnings == []

    def test_wrong_sample_count(self):
        """Test error on wrong sample count."""
        C = np.random.randn(100, 5)
        with pytest.raises(ValidationError, match="Expected 50 samples, got 100"):
            validate_concentrations(C, n_samples=50)

    def test_wrong_component_count(self):
        """Test error on wrong component count."""
        C = np.random.randn(100, 5)
        with pytest.raises(ValidationError, match="Expected 3 components"):
            validate_concentrations(C, n_components=3)

    def test_negative_concentration_warning(self):
        """Test warning on negative concentrations."""
        C = np.random.randn(100, 5)  # Will have negatives
        warnings = validate_concentrations(C)
        assert len(warnings) > 0
        assert "negative" in warnings[0].lower()

    def test_normalization_check(self):
        """Test normalization check."""
        C = np.random.randn(100, 5)
        C = np.abs(C)  # Make positive but not normalized
        warnings = validate_concentrations(C, check_normalized=True)
        assert len(warnings) > 0
        assert "normalized" in warnings[0].lower()

class TestValidateWavelengths:
    """Tests for validate_wavelengths function."""

    def test_valid_wavelengths(self):
        """Test validation of valid wavelengths."""
        wl = np.arange(1000, 2500, 2, dtype=float)
        warnings = validate_wavelengths(
            wl, expected_range=(900, 2600), check_monotonic=True
        )
        assert warnings == []

    def test_wrong_type(self):
        """Test error on wrong type."""
        with pytest.raises(ValidationError, match="Expected numpy array"):
            validate_wavelengths([1000, 1100, 1200])

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        wl = np.array([[1000, 1100], [1200, 1300]])
        with pytest.raises(ValidationError, match="Expected 1D"):
            validate_wavelengths(wl)

    def test_too_short(self):
        """Test error on too short array."""
        wl = np.array([1000.0])
        with pytest.raises(ValidationError, match="too short"):
            validate_wavelengths(wl)

    def test_not_monotonic(self):
        """Test error on non-monotonic wavelengths."""
        wl = np.array([1000.0, 1100.0, 1050.0, 1200.0])
        with pytest.raises(ValidationError, match="monotonically increasing"):
            validate_wavelengths(wl, check_monotonic=True)

    def test_range_warning(self):
        """Test warning when out of expected range."""
        wl = np.arange(500, 3000, 10, dtype=float)
        warnings = validate_wavelengths(wl, expected_range=(1000, 2500))
        assert len(warnings) > 0

    def test_non_uniform_warning(self):
        """Test warning on non-uniform spacing."""
        wl = np.array([1000.0, 1100.0, 1300.0, 1400.0])  # Non-uniform
        warnings = validate_wavelengths(wl, check_uniform=True)
        assert len(warnings) > 0

class TestValidateSyntheticOutput:
    """Tests for validate_synthetic_output function."""

    def test_valid_output(self):
        """Test validation of valid synthetic output."""
        n_samples, n_wavelengths, n_components = 100, 500, 5
        X = np.random.randn(n_samples, n_wavelengths)
        C = np.random.dirichlet(np.ones(n_components), size=n_samples)
        E = np.random.randn(n_components, n_wavelengths)
        wl = np.arange(1000, 1000 + n_wavelengths * 2, 2, dtype=float)

        warnings = validate_synthetic_output(X, C, E, wl)
        # May have warnings but no errors
        assert isinstance(warnings, list)

    def test_component_spectra_shape_mismatch(self):
        """Test error on E shape mismatch."""
        X = np.random.randn(100, 500)
        C = np.random.randn(100, 5)
        E = np.random.randn(3, 500)  # Wrong component count

        with pytest.raises(ValidationError, match="Component spectra shape"):
            validate_synthetic_output(X, C, E)

    def test_wavelength_length_mismatch(self):
        """Test error on wavelength array length mismatch."""
        X = np.random.randn(100, 500)
        C = np.random.randn(100, 5)
        E = np.random.randn(5, 500)
        wl = np.arange(1000, 1200, 2, dtype=float)  # Too short

        with pytest.raises(ValidationError, match="does not match"):
            validate_synthetic_output(X, C, E, wl)

class TestValidationError:
    """Tests for ValidationError exception."""

    def test_raise_validation_error(self):
        """Test raising ValidationError."""
        with pytest.raises(ValidationError, match="Test error"):
            raise ValidationError("Test error")

    def test_validation_error_message(self):
        """Test ValidationError message."""
        try:
            raise ValidationError("Custom message")
        except ValidationError as e:
            assert str(e) == "Custom message"

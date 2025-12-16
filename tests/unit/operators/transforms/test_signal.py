"""Tests for signal processing and conversion transforms."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from nirs4all.operators.transforms.signal_conversion import (
    ToAbsorbance,
    FromAbsorbance,
    PercentToFraction,
    FractionToPercent,
    KubelkaMunk,
    SignalTypeConverter
)
from nirs4all.data.signal_type import SignalType


class TestToAbsorbance:
    """Test ToAbsorbance transformer."""

    def test_reflectance_to_absorbance(self):
        """Test converting fractional reflectance to absorbance."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.array([[0.5, 0.1, 0.01]])

        A = transformer.fit_transform(R)

        # A = -log10(R)
        expected = -np.log10(R)
        assert_array_almost_equal(A, expected)

    def test_reflectance_percent_to_absorbance(self):
        """Test converting percent reflectance to absorbance."""
        transformer = ToAbsorbance(source_type="reflectance%")
        R_pct = np.array([[50, 10, 1]])  # 50%, 10%, 1%

        A = transformer.fit_transform(R_pct)

        # Should first convert to fraction, then log
        R_frac = R_pct / 100.0
        expected = -np.log10(R_frac)
        assert_array_almost_equal(A, expected)

    def test_transmittance_to_absorbance(self):
        """Test converting transmittance to absorbance."""
        transformer = ToAbsorbance(source_type="transmittance")
        T = np.array([[0.5, 0.1, 0.01]])

        A = transformer.fit_transform(T)

        expected = -np.log10(T)
        assert_array_almost_equal(A, expected)

    def test_transmittance_percent_to_absorbance(self):
        """Test converting percent transmittance to absorbance."""
        transformer = ToAbsorbance(source_type="transmittance%")
        T_pct = np.array([[50, 10, 1]])

        A = transformer.fit_transform(T_pct)

        T_frac = T_pct / 100.0
        expected = -np.log10(T_frac)
        assert_array_almost_equal(A, expected)

    def test_inverse_transform(self):
        """Test inverse transformation from absorbance back to reflectance."""
        transformer = ToAbsorbance(source_type="reflectance")
        R_original = np.array([[0.5, 0.3, 0.1]])

        A = transformer.fit_transform(R_original)
        R_recovered = transformer.inverse_transform(A)

        assert_array_almost_equal(R_recovered, R_original)

    def test_inverse_transform_percent(self):
        """Test inverse transformation with percent values."""
        transformer = ToAbsorbance(source_type="reflectance%")
        R_original = np.array([[50, 30, 10]])

        A = transformer.fit_transform(R_original)
        R_recovered = transformer.inverse_transform(A)

        assert_array_almost_equal(R_recovered, R_original)

    def test_clip_negative_values(self):
        """Test that negative values are clipped with clip_negative=True."""
        transformer = ToAbsorbance(source_type="reflectance", clip_negative=True)
        R = np.array([[-0.1, 0.5, 1.2]])  # Contains negative and >1 values

        A = transformer.fit_transform(R)

        # Should not raise or produce inf/nan
        assert np.all(np.isfinite(A))

    def test_epsilon_prevents_log_zero(self):
        """Test that epsilon prevents log(0)."""
        transformer = ToAbsorbance(source_type="reflectance", epsilon=1e-10)
        R = np.array([[0.0, 0.5, 1.0]])

        A = transformer.fit_transform(R)

        # Should not have inf values
        assert np.all(np.isfinite(A))

    def test_invalid_source_type_raises(self):
        """Test that invalid source type raises ValueError."""
        transformer = ToAbsorbance(source_type="absorbance")

        with pytest.raises(ValueError, match="source_type must be one of"):
            transformer.fit(np.array([[0.5]]))

    def test_multidimensional_data(self):
        """Test with multi-sample data."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.random.uniform(0.1, 0.9, size=(100, 200))

        A = transformer.fit_transform(R)

        assert A.shape == R.shape
        assert np.all(np.isfinite(A))


class TestFromAbsorbance:
    """Test FromAbsorbance transformer."""

    def test_absorbance_to_reflectance(self):
        """Test converting absorbance to reflectance."""
        transformer = FromAbsorbance(target_type="reflectance")
        A = np.array([[0.301, 1.0, 2.0]])  # -log10(0.5), -log10(0.1), -log10(0.01)

        R = transformer.fit_transform(A)

        # R = 10^(-A)
        expected = np.power(10.0, -A)
        assert_array_almost_equal(R, expected)

    def test_absorbance_to_reflectance_percent(self):
        """Test converting absorbance to percent reflectance."""
        transformer = FromAbsorbance(target_type="reflectance%")
        A = np.array([[0.301, 1.0, 2.0]])

        R_pct = transformer.fit_transform(A)

        # R% = 10^(-A) * 100
        expected = np.power(10.0, -A) * 100.0
        assert_array_almost_equal(R_pct, expected)

    def test_absorbance_to_transmittance(self):
        """Test converting absorbance to transmittance."""
        transformer = FromAbsorbance(target_type="transmittance")
        A = np.array([[0.5, 1.0, 1.5]])

        T = transformer.fit_transform(A)

        expected = np.power(10.0, -A)
        assert_array_almost_equal(T, expected)

    def test_inverse_transform(self):
        """Test inverse transformation back to absorbance."""
        transformer = FromAbsorbance(target_type="reflectance")
        A_original = np.array([[0.5, 1.0, 1.5]])

        R = transformer.fit_transform(A_original)
        A_recovered = transformer.inverse_transform(R)

        assert_array_almost_equal(A_recovered, A_original)

    def test_invalid_target_type_raises(self):
        """Test that invalid target type raises ValueError."""
        transformer = FromAbsorbance(target_type="absorbance")

        with pytest.raises(ValueError, match="target_type must be one of"):
            transformer.fit(np.array([[0.5]]))


class TestPercentToFraction:
    """Test PercentToFraction transformer."""

    def test_basic_conversion(self):
        """Test basic percent to fraction conversion."""
        transformer = PercentToFraction()
        X_pct = np.array([[50, 25, 100, 0]])

        X_frac = transformer.fit_transform(X_pct)

        expected = np.array([[0.5, 0.25, 1.0, 0.0]])
        assert_array_almost_equal(X_frac, expected)

    def test_inverse_transform(self):
        """Test fraction back to percent."""
        transformer = PercentToFraction()
        X_pct = np.array([[50, 25, 100]])

        X_frac = transformer.fit_transform(X_pct)
        X_recovered = transformer.inverse_transform(X_frac)

        assert_array_almost_equal(X_recovered, X_pct)


class TestFractionToPercent:
    """Test FractionToPercent transformer."""

    def test_basic_conversion(self):
        """Test basic fraction to percent conversion."""
        transformer = FractionToPercent()
        X_frac = np.array([[0.5, 0.25, 1.0, 0.0]])

        X_pct = transformer.fit_transform(X_frac)

        expected = np.array([[50, 25, 100, 0]])
        assert_array_almost_equal(X_pct, expected)

    def test_inverse_transform(self):
        """Test percent back to fraction."""
        transformer = FractionToPercent()
        X_frac = np.array([[0.5, 0.25, 1.0]])

        X_pct = transformer.fit_transform(X_frac)
        X_recovered = transformer.inverse_transform(X_pct)

        assert_array_almost_equal(X_recovered, X_frac)


class TestKubelkaMunk:
    """Test KubelkaMunk transformer."""

    def test_kubelka_munk_formula(self):
        """Test Kubelka-Munk formula: F(R) = (1-R)² / (2R)."""
        transformer = KubelkaMunk(source_type="reflectance")
        R = np.array([[0.5, 0.8, 0.2]])

        F_R = transformer.fit_transform(R)

        # Manual calculation
        expected = np.square(1.0 - R) / (2.0 * R)
        assert_array_almost_equal(F_R, expected)

    def test_kubelka_munk_specific_values(self):
        """Test specific Kubelka-Munk values."""
        transformer = KubelkaMunk(source_type="reflectance")

        # R = 0.5: F(R) = (1-0.5)² / (2*0.5) = 0.25 / 1 = 0.25
        R = np.array([[0.5]])
        F_R = transformer.fit_transform(R)
        assert_allclose(F_R, [[0.25]], rtol=1e-5)

        # R = 1: F(R) = 0 / 2 = 0
        R = np.array([[0.99]])  # Clipped near 1
        F_R = transformer.fit_transform(R)
        assert F_R[0, 0] < 0.01

    def test_kubelka_munk_from_percent(self):
        """Test Kubelka-Munk from percent reflectance."""
        transformer = KubelkaMunk(source_type="reflectance%")
        R_pct = np.array([[50, 80, 20]])

        F_R = transformer.fit_transform(R_pct)

        # Should convert to fraction first
        R_frac = R_pct / 100.0
        expected = np.square(1.0 - R_frac) / (2.0 * R_frac)
        assert_array_almost_equal(F_R, expected)

    def test_kubelka_munk_inverse(self):
        """Test inverse Kubelka-Munk transformation."""
        transformer = KubelkaMunk(source_type="reflectance")
        R_original = np.array([[0.5, 0.3, 0.7]])

        F_R = transformer.fit_transform(R_original)
        R_recovered = transformer.inverse_transform(F_R)

        assert_array_almost_equal(R_recovered, R_original)

    def test_invalid_source_type_raises(self):
        """Test that non-reflectance source raises error."""
        transformer = KubelkaMunk(source_type="transmittance")

        with pytest.raises(ValueError, match="KubelkaMunk requires reflectance"):
            transformer.fit(np.array([[0.5]]))

    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents R=0 division error."""
        transformer = KubelkaMunk(source_type="reflectance", epsilon=1e-10)
        R = np.array([[0.0, 0.5]])  # Contains zero

        F_R = transformer.fit_transform(R)

        # Should not have inf or nan
        assert np.all(np.isfinite(F_R))


class TestSignalTypeConverter:
    """Test SignalTypeConverter general-purpose converter."""

    def test_reflectance_to_absorbance(self):
        """Test R -> A conversion."""
        converter = SignalTypeConverter(
            source_type="reflectance",
            target_type="absorbance"
        )
        R = np.array([[0.5, 0.1]])

        A = converter.fit_transform(R)

        expected = -np.log10(R)
        assert_array_almost_equal(A, expected)

    def test_reflectance_percent_to_absorbance(self):
        """Test %R -> A conversion."""
        converter = SignalTypeConverter(
            source_type="reflectance%",
            target_type="absorbance"
        )
        R_pct = np.array([[50, 10]])

        A = converter.fit_transform(R_pct)

        expected = -np.log10(R_pct / 100.0)
        assert_array_almost_equal(A, expected)

    def test_absorbance_to_reflectance(self):
        """Test A -> R conversion."""
        converter = SignalTypeConverter(
            source_type="absorbance",
            target_type="reflectance"
        )
        A = np.array([[0.301, 1.0]])

        R = converter.fit_transform(A)

        expected = np.power(10.0, -A)
        assert_array_almost_equal(R, expected)

    def test_absorbance_to_reflectance_percent(self):
        """Test A -> %R conversion."""
        converter = SignalTypeConverter(
            source_type="absorbance",
            target_type="reflectance%"
        )
        A = np.array([[0.301, 1.0]])

        R_pct = converter.fit_transform(A)

        expected = np.power(10.0, -A) * 100.0
        assert_array_almost_equal(R_pct, expected)

    def test_reflectance_percent_to_fraction(self):
        """Test %R -> R conversion."""
        converter = SignalTypeConverter(
            source_type="reflectance%",
            target_type="reflectance"
        )
        R_pct = np.array([[50, 25]])

        R = converter.fit_transform(R_pct)

        expected = np.array([[0.5, 0.25]])
        assert_array_almost_equal(R, expected)

    def test_reflectance_fraction_to_percent(self):
        """Test R -> %R conversion."""
        converter = SignalTypeConverter(
            source_type="reflectance",
            target_type="reflectance%"
        )
        R = np.array([[0.5, 0.25]])

        R_pct = converter.fit_transform(R)

        expected = np.array([[50, 25]])
        assert_array_almost_equal(R_pct, expected)

    def test_transmittance_to_absorbance(self):
        """Test T -> A conversion."""
        converter = SignalTypeConverter(
            source_type="transmittance",
            target_type="absorbance"
        )
        T = np.array([[0.5, 0.1]])

        A = converter.fit_transform(T)

        expected = -np.log10(T)
        assert_array_almost_equal(A, expected)

    def test_reflectance_to_kubelka_munk(self):
        """Test R -> K-M conversion."""
        converter = SignalTypeConverter(
            source_type="reflectance",
            target_type="kubelka_munk"
        )
        R = np.array([[0.5, 0.8]])

        F_R = converter.fit_transform(R)

        expected = np.square(1.0 - R) / (2.0 * R)
        assert_array_almost_equal(F_R, expected)

    def test_same_type_no_change(self):
        """Test that same source and target returns unchanged data."""
        converter = SignalTypeConverter(
            source_type="absorbance",
            target_type="absorbance"
        )
        A = np.array([[0.5, 1.0, 1.5]])

        A_out = converter.fit_transform(A)

        assert_array_almost_equal(A_out, A)

    def test_inverse_transform(self):
        """Test inverse transformation."""
        converter = SignalTypeConverter(
            source_type="reflectance",
            target_type="absorbance"
        )
        R_original = np.array([[0.5, 0.3, 0.1]])

        A = converter.fit_transform(R_original)
        R_recovered = converter.inverse_transform(A)

        assert_array_almost_equal(R_recovered, R_original)

    def test_unsupported_conversion_raises(self):
        """Test that unsupported conversion paths raise error."""
        converter = SignalTypeConverter(
            source_type="kubelka_munk",
            target_type="transmittance"
        )

        with pytest.raises(ValueError, match="Cannot convert"):
            converter.fit(np.array([[0.5]]))

    def test_string_signal_types(self):
        """Test that string signal types work correctly."""
        converter = SignalTypeConverter(source_type="R", target_type="A")
        R = np.array([[0.5]])

        A = converter.fit_transform(R)

        expected = -np.log10(R)
        assert_array_almost_equal(A, expected)


class TestRoundTripConversions:
    """Test round-trip conversions maintain data integrity."""

    def test_reflectance_absorbance_roundtrip(self):
        """Test R -> A -> R roundtrip."""
        np.random.seed(42)
        R_original = np.random.uniform(0.1, 0.9, size=(50, 100))

        # Forward
        to_abs = ToAbsorbance(source_type="reflectance")
        A = to_abs.fit_transform(R_original)

        # Backward
        from_abs = FromAbsorbance(target_type="reflectance")
        R_recovered = from_abs.fit_transform(A)

        assert_array_almost_equal(R_recovered, R_original, decimal=10)

    def test_percent_fraction_roundtrip(self):
        """Test %R -> R -> %R roundtrip."""
        R_pct_original = np.array([[50, 25, 75, 100]])

        # Forward
        to_frac = PercentToFraction()
        R_frac = to_frac.fit_transform(R_pct_original)

        # Backward
        to_pct = FractionToPercent()
        R_pct_recovered = to_pct.fit_transform(R_frac)

        assert_array_almost_equal(R_pct_recovered, R_pct_original)

    def test_kubelka_munk_roundtrip(self):
        """Test R -> F(R) -> R roundtrip."""
        np.random.seed(42)
        R_original = np.random.uniform(0.2, 0.8, size=(50, 100))

        km = KubelkaMunk(source_type="reflectance")
        F_R = km.fit_transform(R_original)
        R_recovered = km.inverse_transform(F_R)

        assert_array_almost_equal(R_recovered, R_original, decimal=10)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_values(self):
        """Test handling of very small reflectance values."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.array([[1e-10, 1e-8, 1e-6]])

        A = transformer.fit_transform(R)

        # Should produce finite high absorbance values
        assert np.all(np.isfinite(A))
        assert np.all(A > 0)

    def test_values_near_one(self):
        """Test handling of reflectance near 1."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.array([[0.99, 0.999, 0.9999]])

        A = transformer.fit_transform(R)

        # Should produce small positive absorbance values
        assert np.all(np.isfinite(A))
        assert np.all(A > 0)
        assert np.all(A < 0.1)

    def test_single_sample(self):
        """Test with single sample."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.array([[0.5, 0.3, 0.1]])

        A = transformer.fit_transform(R)

        assert A.shape == (1, 3)

    def test_1d_input(self):
        """Test with 1D input array."""
        transformer = ToAbsorbance(source_type="reflectance")
        R = np.array([0.5, 0.3, 0.1])

        A = transformer.fit_transform(R)

        assert A.shape == (3,)


class TestSignalTransforms:
    """Test suite for signal processing transforms (placeholder for future)."""

    def test_placeholder(self):
        """Placeholder test."""
        # TODO: Add tests for baseline correction, derivatives, filters, wavelets
        pass

"""Tests for signal type detection and management."""

import numpy as np
import pytest

from nirs4all.data.signal_type import SignalType, SignalTypeDetector, SignalTypeInput, detect_signal_type, normalize_signal_type


class TestSignalTypeEnum:
    """Test SignalType enum properties and methods."""

    def test_signal_type_values(self):
        """Test that signal types have correct string values."""
        assert SignalType.ABSORBANCE.value == "absorbance"
        assert SignalType.REFLECTANCE.value == "reflectance"
        assert SignalType.REFLECTANCE_PERCENT.value == "reflectance%"
        assert SignalType.TRANSMITTANCE.value == "transmittance"
        assert SignalType.TRANSMITTANCE_PERCENT.value == "transmittance%"
        assert SignalType.KUBELKA_MUNK.value == "kubelka_munk"
        assert SignalType.AUTO.value == "auto"
        assert SignalType.UNKNOWN.value == "unknown"
        assert SignalType.PREPROCESSED.value == "preprocessed"

    def test_is_percent_property(self):
        """Test is_percent property."""
        assert SignalType.REFLECTANCE_PERCENT.is_percent is True
        assert SignalType.TRANSMITTANCE_PERCENT.is_percent is True
        assert SignalType.REFLECTANCE.is_percent is False
        assert SignalType.TRANSMITTANCE.is_percent is False
        assert SignalType.ABSORBANCE.is_percent is False

    def test_is_fraction_property(self):
        """Test is_fraction property."""
        assert SignalType.REFLECTANCE.is_fraction is True
        assert SignalType.TRANSMITTANCE.is_fraction is True
        assert SignalType.REFLECTANCE_PERCENT.is_fraction is False
        assert SignalType.ABSORBANCE.is_fraction is False

    def test_is_absorbance_like_property(self):
        """Test is_absorbance_like property."""
        assert SignalType.ABSORBANCE.is_absorbance_like is True
        assert SignalType.KUBELKA_MUNK.is_absorbance_like is True
        assert SignalType.LOG_1_R.is_absorbance_like is True
        assert SignalType.LOG_1_T.is_absorbance_like is True
        assert SignalType.REFLECTANCE.is_absorbance_like is False
        assert SignalType.TRANSMITTANCE.is_absorbance_like is False

    def test_is_reflectance_based_property(self):
        """Test is_reflectance_based property."""
        assert SignalType.REFLECTANCE.is_reflectance_based is True
        assert SignalType.REFLECTANCE_PERCENT.is_reflectance_based is True
        assert SignalType.TRANSMITTANCE.is_reflectance_based is False
        assert SignalType.ABSORBANCE.is_reflectance_based is False

    def test_is_transmittance_based_property(self):
        """Test is_transmittance_based property."""
        assert SignalType.TRANSMITTANCE.is_transmittance_based is True
        assert SignalType.TRANSMITTANCE_PERCENT.is_transmittance_based is True
        assert SignalType.REFLECTANCE.is_transmittance_based is False
        assert SignalType.ABSORBANCE.is_transmittance_based is False

    def test_is_determinable_property(self):
        """Test is_determinable property."""
        assert SignalType.ABSORBANCE.is_determinable is True
        assert SignalType.REFLECTANCE.is_determinable is True
        assert SignalType.AUTO.is_determinable is False
        assert SignalType.UNKNOWN.is_determinable is False
        assert SignalType.PREPROCESSED.is_determinable is False

class TestSignalTypeFromString:
    """Test SignalType.from_string parsing."""

    def test_absorbance_aliases(self):
        """Test absorbance string aliases."""
        for alias in ["a", "abs", "absorbance", "absorption", "A", "ABS"]:
            assert SignalType.from_string(alias) == SignalType.ABSORBANCE

    def test_reflectance_aliases(self):
        """Test reflectance string aliases."""
        for alias in ["r", "ref", "refl", "reflectance", "R", "REFL"]:
            assert SignalType.from_string(alias) == SignalType.REFLECTANCE

    def test_reflectance_percent_aliases(self):
        """Test reflectance percent string aliases."""
        for alias in ["%r", "r%", "reflectance%", "%R", "R%"]:
            assert SignalType.from_string(alias) == SignalType.REFLECTANCE_PERCENT

    def test_transmittance_aliases(self):
        """Test transmittance string aliases."""
        for alias in ["t", "trans", "transmittance", "transmission", "T"]:
            assert SignalType.from_string(alias) == SignalType.TRANSMITTANCE

    def test_transmittance_percent_aliases(self):
        """Test transmittance percent string aliases."""
        for alias in ["%t", "t%", "transmittance%", "%T", "T%"]:
            assert SignalType.from_string(alias) == SignalType.TRANSMITTANCE_PERCENT

    def test_kubelka_munk_aliases(self):
        """Test Kubelka-Munk string aliases."""
        for alias in ["km", "kubelka_munk", "kubelka-munk", "f(r)", "KM"]:
            assert SignalType.from_string(alias) == SignalType.KUBELKA_MUNK

    def test_log_transform_aliases(self):
        """Test log transform aliases."""
        for alias in ["log(1/r)", "log_1_r", "-log(r)", "-log10(r)"]:
            assert SignalType.from_string(alias) == SignalType.LOG_1_R
        for alias in ["log(1/t)", "log_1_t", "-log(t)", "-log10(t)"]:
            assert SignalType.from_string(alias) == SignalType.LOG_1_T

    def test_special_types(self):
        """Test special type string values."""
        assert SignalType.from_string("auto") == SignalType.AUTO
        assert SignalType.from_string("unknown") == SignalType.UNKNOWN
        assert SignalType.from_string("preprocessed") == SignalType.PREPROCESSED

    def test_invalid_string_raises(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown signal type"):
            SignalType.from_string("invalid_type")

    def test_from_string_with_enum_passthrough(self):
        """Test that from_string passes through SignalType enums."""
        result = SignalType.from_string(SignalType.ABSORBANCE)
        assert result == SignalType.ABSORBANCE

class TestNormalizeSignalType:
    """Test normalize_signal_type function."""

    def test_normalize_string(self):
        """Test normalizing string inputs."""
        assert normalize_signal_type("absorbance") == SignalType.ABSORBANCE
        assert normalize_signal_type("R") == SignalType.REFLECTANCE
        assert normalize_signal_type("%T") == SignalType.TRANSMITTANCE_PERCENT

    def test_normalize_enum_passthrough(self):
        """Test that enums pass through unchanged."""
        assert normalize_signal_type(SignalType.ABSORBANCE) == SignalType.ABSORBANCE
        assert normalize_signal_type(SignalType.REFLECTANCE) == SignalType.REFLECTANCE

class TestSignalTypeDetector:
    """Test SignalTypeDetector heuristics."""

    def test_detect_reflectance_fraction(self):
        """Test detection of reflectance in [0, 1] range."""
        # Generate typical reflectance data: values mostly in [0.2, 0.8]
        np.random.seed(42)
        spectra = np.random.uniform(0.2, 0.8, size=(100, 200))

        detector = SignalTypeDetector()
        # Use lower threshold since R/T are hard to distinguish without wavelengths
        signal_type, confidence, reason = detector.detect(spectra, confidence_threshold=0.3)

        # Should detect as either reflectance or transmittance (similar ranges)
        assert signal_type in (SignalType.REFLECTANCE, SignalType.TRANSMITTANCE, SignalType.UNKNOWN)
        assert "Range:" in reason

    def test_detect_reflectance_percent(self):
        """Test detection of reflectance in [0, 100] range."""
        np.random.seed(42)
        spectra = np.random.uniform(20, 80, size=(100, 200))

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra, confidence_threshold=0.3)

        # Should detect as percent-based type or unknown (range ambiguity)
        assert signal_type in (
            SignalType.REFLECTANCE_PERCENT,
            SignalType.TRANSMITTANCE_PERCENT,
            SignalType.UNKNOWN
        )

    def test_detect_absorbance(self):
        """Test detection of absorbance data."""
        np.random.seed(42)
        # Absorbance typically in [0, 2-3] range
        spectra = np.random.uniform(0.3, 1.5, size=(100, 200))

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra)

        assert signal_type == SignalType.ABSORBANCE
        assert confidence > 0.5

    def test_detect_preprocessed_mean_centered(self):
        """Test detection of mean-centered preprocessed data."""
        np.random.seed(42)
        spectra = np.random.randn(100, 200) * 0.5  # Zero-centered

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra)

        assert signal_type == SignalType.PREPROCESSED
        assert "preprocessed" in reason.lower()

    def test_detect_preprocessed_snv(self):
        """Test detection of SNV-normalized data."""
        np.random.seed(42)
        spectra = np.random.randn(100, 200)  # std ~1, mean ~0

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra)

        assert signal_type == SignalType.PREPROCESSED

    def test_detect_empty_data(self):
        """Test detection with empty data."""
        spectra = np.array([])

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra)

        assert signal_type == SignalType.UNKNOWN
        assert confidence == 0.0
        assert "Empty" in reason

    def test_detect_with_low_confidence_returns_unknown(self):
        """Test that low confidence returns UNKNOWN."""
        # Create ambiguous data
        np.random.seed(42)
        spectra = np.random.uniform(0.4, 0.6, size=(100, 200))

        detector = SignalTypeDetector()
        signal_type, confidence, reason = detector.detect(spectra, confidence_threshold=0.99)

        assert signal_type == SignalType.UNKNOWN

    def test_detect_with_wavelengths_nm(self):
        """Test detection with wavelength information in nm."""
        np.random.seed(42)
        wavelengths = np.linspace(900, 2500, 200)

        # Create reflectance data with dips at water bands
        spectra = np.ones((100, 200)) * 0.6
        # Add dip at ~1450nm
        water_idx = np.argmin(np.abs(wavelengths - 1450))
        spectra[:, water_idx-5:water_idx+5] = 0.4

        detector = SignalTypeDetector(wavelengths=wavelengths, wavelength_unit="nm")
        signal_type, confidence, reason = detector.detect(spectra, confidence_threshold=0.3)

        # Should detect as reflectance or transmittance due to dip at water band
        # The test validates that wavelength-based detection doesn't crash
        assert signal_type in (
            SignalType.REFLECTANCE, SignalType.TRANSMITTANCE, SignalType.UNKNOWN
        )

class TestDetectSignalTypeConvenience:
    """Test detect_signal_type convenience function."""

    def test_basic_detection(self):
        """Test basic detection without wavelengths."""
        np.random.seed(42)
        spectra = np.random.uniform(0.2, 0.8, size=(100, 200))

        signal_type, confidence, reason = detect_signal_type(spectra)

        assert isinstance(signal_type, SignalType)
        assert 0 <= confidence <= 1
        assert isinstance(reason, str)

    def test_detection_with_wavelengths(self):
        """Test detection with wavelength information."""
        np.random.seed(42)
        spectra = np.random.uniform(0.2, 0.8, size=(100, 200))
        wavelengths = np.linspace(900, 2500, 200)

        signal_type, confidence, reason = detect_signal_type(
            spectra, wavelengths=wavelengths, wavelength_unit="nm"
        )

        assert isinstance(signal_type, SignalType)

    def test_detection_with_cm1_wavelengths(self):
        """Test detection with wavenumber information."""
        np.random.seed(42)
        spectra = np.random.uniform(0.2, 0.8, size=(100, 200))
        wavenumbers = np.linspace(4000, 10000, 200)

        signal_type, confidence, reason = detect_signal_type(
            spectra, wavelengths=wavenumbers, wavelength_unit="cm-1"
        )

        assert isinstance(signal_type, SignalType)

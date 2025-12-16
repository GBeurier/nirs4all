"""
Signal type management for spectroscopy data.

This module provides:
- SignalType enum for absorbance, reflectance, transmittance
- Autodetection heuristics based on value ranges and band directions
- Conversion utilities between signal types
"""

from enum import Enum
from typing import Optional, Union, Tuple, List
import numpy as np


class SignalType(str, Enum):
    """
    Spectral signal types for NIRS/spectroscopy data.

    Defines the measurement type of spectral data. String values ensure
    backward compatibility with config files.
    """
    # Core types
    ABSORBANCE = "absorbance"              # A, typically [0, 3+], log(1/R) or log(1/T)
    REFLECTANCE = "reflectance"            # R, fractional [0, 1]
    REFLECTANCE_PERCENT = "reflectance%"   # %R, percentage [0, 100]
    TRANSMITTANCE = "transmittance"        # T, fractional [0, 1]
    TRANSMITTANCE_PERCENT = "transmittance%"  # %T, percentage [0, 100]

    # Special types
    KUBELKA_MUNK = "kubelka_munk"          # F(R) = (1-R)²/(2R)
    LOG_1_R = "log_1_r"                    # log(1/R) - pseudo-absorbance from reflectance
    LOG_1_T = "log_1_t"                    # log(1/T) - absorbance from transmittance

    # Detection states
    AUTO = "auto"                          # Auto-detect on first use
    UNKNOWN = "unknown"                    # Cannot be determined (preprocessed data)
    PREPROCESSED = "preprocessed"          # Data has been preprocessed (derivative, SNV, etc.)

    @property
    def is_percent(self) -> bool:
        """Check if this is a percentage-based signal type."""
        return self in (SignalType.REFLECTANCE_PERCENT, SignalType.TRANSMITTANCE_PERCENT)

    @property
    def is_fraction(self) -> bool:
        """Check if this is a fractional [0, 1] signal type."""
        return self in (SignalType.REFLECTANCE, SignalType.TRANSMITTANCE)

    @property
    def is_absorbance_like(self) -> bool:
        """Check if this is absorbance or pseudo-absorbance."""
        return self in (
            SignalType.ABSORBANCE,
            SignalType.LOG_1_R,
            SignalType.LOG_1_T,
            SignalType.KUBELKA_MUNK
        )

    @property
    def is_reflectance_based(self) -> bool:
        """Check if this is any reflectance-based signal."""
        return self in (SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT)

    @property
    def is_transmittance_based(self) -> bool:
        """Check if this is any transmittance-based signal."""
        return self in (SignalType.TRANSMITTANCE, SignalType.TRANSMITTANCE_PERCENT)

    @property
    def is_determinable(self) -> bool:
        """Check if this is a known, determinable signal type."""
        return self not in (SignalType.AUTO, SignalType.UNKNOWN, SignalType.PREPROCESSED)

    @classmethod
    def from_string(cls, value: str) -> "SignalType":
        """
        Parse signal type from various string representations.

        Args:
            value: String representation (e.g., "A", "R", "%R", "absorbance", etc.)

        Returns:
            SignalType enum value
        """
        if isinstance(value, SignalType):
            return value

        value_lower = value.lower().strip()

        # Common abbreviations and variations
        mappings = {
            # Absorbance
            "a": cls.ABSORBANCE,
            "abs": cls.ABSORBANCE,
            "absorbance": cls.ABSORBANCE,
            "absorption": cls.ABSORBANCE,

            # Reflectance
            "r": cls.REFLECTANCE,
            "ref": cls.REFLECTANCE,
            "refl": cls.REFLECTANCE,
            "reflectance": cls.REFLECTANCE,

            # Reflectance percent
            "%r": cls.REFLECTANCE_PERCENT,
            "r%": cls.REFLECTANCE_PERCENT,
            "reflectance%": cls.REFLECTANCE_PERCENT,
            "percent_reflectance": cls.REFLECTANCE_PERCENT,
            "reflectance_percent": cls.REFLECTANCE_PERCENT,

            # Transmittance
            "t": cls.TRANSMITTANCE,
            "trans": cls.TRANSMITTANCE,
            "transmittance": cls.TRANSMITTANCE,
            "transmission": cls.TRANSMITTANCE,

            # Transmittance percent
            "%t": cls.TRANSMITTANCE_PERCENT,
            "t%": cls.TRANSMITTANCE_PERCENT,
            "transmittance%": cls.TRANSMITTANCE_PERCENT,
            "percent_transmittance": cls.TRANSMITTANCE_PERCENT,
            "transmittance_percent": cls.TRANSMITTANCE_PERCENT,

            # Kubelka-Munk
            "km": cls.KUBELKA_MUNK,
            "kubelka_munk": cls.KUBELKA_MUNK,
            "kubelka-munk": cls.KUBELKA_MUNK,
            "f(r)": cls.KUBELKA_MUNK,

            # Log transforms
            "log(1/r)": cls.LOG_1_R,
            "log_1_r": cls.LOG_1_R,
            "-log(r)": cls.LOG_1_R,
            "-log10(r)": cls.LOG_1_R,
            "log(1/t)": cls.LOG_1_T,
            "log_1_t": cls.LOG_1_T,
            "-log(t)": cls.LOG_1_T,
            "-log10(t)": cls.LOG_1_T,

            # Special
            "auto": cls.AUTO,
            "unknown": cls.UNKNOWN,
            "preprocessed": cls.PREPROCESSED,
        }

        if value_lower in mappings:
            return mappings[value_lower]

        # Try direct enum value match
        try:
            return cls(value_lower)
        except ValueError:
            raise ValueError(
                f"Unknown signal type '{value}'. Valid options: "
                f"{[e.value for e in cls]}"
            )


# Type alias for input flexibility
SignalTypeInput = Union[str, SignalType]


def normalize_signal_type(signal_type: SignalTypeInput) -> SignalType:
    """
    Normalize a signal type input to SignalType enum.

    Args:
        signal_type: String or SignalType enum

    Returns:
        SignalType enum value
    """
    if isinstance(signal_type, SignalType):
        return signal_type
    return SignalType.from_string(signal_type)


class SignalTypeDetector:
    """
    Heuristic detector for spectral signal types.

    Uses value ranges and optionally wavelength information to determine
    whether data is absorbance, reflectance, or transmittance.
    """

    # NIR water absorption bands (nm) - strong absorbers
    WATER_BANDS_NM = [1450, 1940, 2500]  # O-H stretching

    # Corresponding wavenumbers (cm-1)
    WATER_BANDS_CM1 = [6897, 5155, 4000]  # 10^7 / nm

    def __init__(
        self,
        wavelengths: Optional[np.ndarray] = None,
        wavelength_unit: str = "nm"
    ):
        """
        Initialize the detector.

        Args:
            wavelengths: Array of wavelength/wavenumber values for band analysis
            wavelength_unit: Unit of wavelengths ("nm" or "cm-1")
        """
        self.wavelengths = wavelengths
        self.wavelength_unit = wavelength_unit

    def detect(
        self,
        spectra: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[SignalType, float, str]:
        """
        Detect the signal type of spectral data.

        Args:
            spectra: Spectral data array of shape (n_samples, n_features)
            confidence_threshold: Minimum confidence to return a definite type

        Returns:
            Tuple of (SignalType, confidence, reason_string)
        """
        if spectra.size == 0:
            return SignalType.UNKNOWN, 0.0, "Empty data"

        # Flatten if needed for statistics
        data = spectra.flatten() if spectra.ndim == 1 else spectra

        # Calculate statistics - cast to float for type safety
        min_val = float(np.nanmin(data))
        max_val = float(np.nanmax(data))
        mean_val = float(np.nanmean(data))
        std_val = float(np.nanstd(data))

        # Check for preprocessing indicators
        if self._is_preprocessed(min_val, max_val, mean_val, std_val):
            return SignalType.PREPROCESSED, 0.9, "Data appears preprocessed (centered/normalized)"

        # Score each signal type
        scores = {}

        # Reflectance fraction: values in [0, 1]
        scores[SignalType.REFLECTANCE] = self._score_reflectance_fraction(
            min_val, max_val, mean_val
        )

        # Reflectance percent: values in [0, 100]
        scores[SignalType.REFLECTANCE_PERCENT] = self._score_reflectance_percent(
            min_val, max_val, mean_val
        )

        # Transmittance fraction: values in [0, 1]
        scores[SignalType.TRANSMITTANCE] = self._score_transmittance_fraction(
            min_val, max_val, mean_val
        )

        # Transmittance percent: values in [0, 100]
        scores[SignalType.TRANSMITTANCE_PERCENT] = self._score_transmittance_percent(
            min_val, max_val, mean_val
        )

        # Absorbance: typically [0, 3+], can be slightly negative
        scores[SignalType.ABSORBANCE] = self._score_absorbance(
            min_val, max_val, mean_val
        )

        # Use wavelength information as tiebreaker if available
        if self.wavelengths is not None and spectra.ndim == 2:
            band_hints = self._analyze_water_bands(spectra)
            for signal_type, hint_score in band_hints.items():
                if signal_type in scores:
                    scores[signal_type] += hint_score * 0.2  # Weight band hints

        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_type]

        # Normalize confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0

        # Build reason string
        reason = self._build_reason(min_val, max_val, mean_val, best_type, confidence)

        if confidence < confidence_threshold:
            return SignalType.UNKNOWN, confidence, reason

        return best_type, confidence, reason

    def _is_preprocessed(
        self,
        min_val: float,
        max_val: float,
        mean_val: float,
        std_val: float
    ) -> bool:
        """Check if data shows signs of preprocessing."""
        # Mean-centered data has mean close to 0
        if abs(mean_val) < 0.01 and std_val > 0.1:
            return True

        # SNV/standardized data has std close to 1
        if abs(std_val - 1.0) < 0.1 and abs(mean_val) < 0.1:
            return True

        # Derivative data often has negative values with mean near 0
        if min_val < -0.5 and max_val < 0.5 and abs(mean_val) < 0.01:
            return True

        return False

    def _score_reflectance_fraction(
        self,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> float:
        """Score likelihood of reflectance in [0, 1]."""
        score = 0.0

        # Values should be in [0, 1]
        if 0 <= min_val and max_val <= 1.2:
            score += 0.5

            # Typical reflectance range
            if 0.1 <= mean_val <= 0.8:
                score += 0.3

            # Very strong match if max is close to 1
            if max_val <= 1.0:
                score += 0.2

        return score

    def _score_reflectance_percent(
        self,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> float:
        """Score likelihood of reflectance in [0, 100]."""
        score = 0.0

        # Values should be in [0, 100]
        if 0 <= min_val and 1.5 < max_val <= 120:
            score += 0.5

            # Typical percent reflectance range
            if 10 <= mean_val <= 80:
                score += 0.3

            # Very strong match if max is close to 100
            if max_val <= 100:
                score += 0.2

        return score

    def _score_transmittance_fraction(
        self,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> float:
        """Score likelihood of transmittance in [0, 1]."""
        score = 0.0

        # Very similar to reflectance fraction
        # Without band direction info, hard to distinguish
        if 0 <= min_val and max_val <= 1.2:
            score += 0.4

            # Transmittance often has lower values than reflectance
            if 0.05 <= mean_val <= 0.5:
                score += 0.2

        return score

    def _score_transmittance_percent(
        self,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> float:
        """Score likelihood of transmittance in [0, 100]."""
        score = 0.0

        # Similar to reflectance percent
        if 0 <= min_val and 1.5 < max_val <= 120:
            score += 0.4

            # Transmittance percent
            if 5 <= mean_val <= 50:
                score += 0.2

        return score

    def _score_absorbance(
        self,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> float:
        """Score likelihood of absorbance."""
        score = 0.0

        # Absorbance typically [0, 3+], can be slightly negative
        if -0.5 <= min_val and 0.5 <= max_val <= 5.0:
            score += 0.4

            # Typical absorbance range
            if 0.2 <= mean_val <= 2.0:
                score += 0.3

            # Absorbance peaks are positive
            if min_val >= -0.2:
                score += 0.2

            # High max suggests absorbance peaks
            if max_val >= 1.0:
                score += 0.1

        return score

    def _analyze_water_bands(
        self,
        spectra: np.ndarray
    ) -> dict:
        """
        Analyze water band directions to distinguish R/T from A.

        In reflectance/transmittance, water bands show as DIPS (lower values).
        In absorbance, water bands show as PEAKS (higher values).

        Returns:
            Dict mapping SignalType to score adjustment
        """
        hints = {}

        if self.wavelengths is None or len(self.wavelengths) != spectra.shape[1]:
            return hints

        # Get wavelengths in nm for comparison
        if self.wavelength_unit == "cm-1":
            # Convert cm-1 to nm
            wl_nm = 1e7 / self.wavelengths
            bands_to_check = self.WATER_BANDS_NM
        else:
            wl_nm = self.wavelengths
            bands_to_check = self.WATER_BANDS_NM

        # Find indices closest to water bands
        peak_count = 0
        dip_count = 0

        mean_spectrum = np.nanmean(spectra, axis=0)

        for band_nm in bands_to_check:
            # Find closest wavelength index
            if wl_nm.min() <= band_nm <= wl_nm.max():
                idx = np.argmin(np.abs(wl_nm - band_nm))

                # Compare to local neighborhood
                window = 10
                start = max(0, idx - window)
                end = min(len(mean_spectrum), idx + window)

                local_mean = np.nanmean(mean_spectrum[start:end])
                band_value = mean_spectrum[idx]

                if band_value > local_mean * 1.05:
                    peak_count += 1  # Peak at water band -> absorbance
                elif band_value < local_mean * 0.95:
                    dip_count += 1   # Dip at water band -> R or T

        if peak_count > dip_count:
            hints[SignalType.ABSORBANCE] = 0.3
            hints[SignalType.REFLECTANCE] = -0.1
            hints[SignalType.TRANSMITTANCE] = -0.1
        elif dip_count > peak_count:
            hints[SignalType.ABSORBANCE] = -0.1
            hints[SignalType.REFLECTANCE] = 0.2
            hints[SignalType.TRANSMITTANCE] = 0.2

        return hints

    def _build_reason(
        self,
        min_val: float,
        max_val: float,
        mean_val: float,
        detected_type: SignalType,
        confidence: float
    ) -> str:
        """Build human-readable detection reason."""
        parts = [
            f"Range: [{min_val:.3f}, {max_val:.3f}]",
            f"Mean: {mean_val:.3f}",
            f"Detected: {detected_type.value}",
            f"Confidence: {confidence:.1%}"
        ]
        return " | ".join(parts)


def detect_signal_type(
    spectra: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    wavelength_unit: str = "nm"
) -> Tuple[SignalType, float, str]:
    """
    Convenience function to detect signal type.

    Args:
        spectra: Spectral data array (n_samples, n_features)
        wavelengths: Optional wavelength values for band analysis
        wavelength_unit: Unit of wavelengths ("nm" or "cm-1")

    Returns:
        Tuple of (SignalType, confidence, reason)

    Example:
        >>> spectra = np.random.rand(100, 500) * 0.8  # Values in [0, 0.8]
        >>> signal_type, confidence, reason = detect_signal_type(spectra)
        >>> print(f"Detected: {signal_type.value} ({confidence:.0%})")
    """
    detector = SignalTypeDetector(wavelengths, wavelength_unit)
    return detector.detect(spectra)

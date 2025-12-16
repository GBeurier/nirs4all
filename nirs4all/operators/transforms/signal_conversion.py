"""
Signal type conversion transformers.

This module provides sklearn-compatible transformers for converting between
spectral signal types (absorbance, reflectance, transmittance).

Mathematical conversions:
- Reflectance to Absorbance: A = -log10(R) or A = log10(1/R)
- Transmittance to Absorbance: A = -log10(T)
- Percent to Fraction: X_frac = X_pct / 100
- Kubelka-Munk: F(R) = (1-R)² / (2R)

All transformers follow the sklearn TransformerMixin pattern.
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Optional, Literal

from nirs4all.data.signal_type import SignalType, SignalTypeInput, normalize_signal_type


class ToAbsorbance(TransformerMixin, BaseEstimator):
    """
    Convert reflectance or transmittance to absorbance.

    Applies the log transform: A = -log10(X)

    For reflectance, this gives "pseudo-absorbance" which is widely used
    in NIR but is not identical to true absorbance in transmission.

    Parameters
    ----------
    source_type : str or SignalType
        Input signal type. If "auto", attempts to detect.
        Valid: "reflectance", "reflectance%", "transmittance", "transmittance%"
    epsilon : float, default=1e-10
        Small value to add to avoid log(0)
    clip_negative : bool, default=True
        If True, clips negative values to epsilon before log transform

    Attributes
    ----------
    source_type_ : SignalType
        Detected or specified source signal type
    is_percent_ : bool
        Whether source was in percent (requires /100)

    Examples
    --------
    >>> from nirs4all.operators.transforms.signal_conversion import ToAbsorbance
    >>> transformer = ToAbsorbance(source_type="reflectance")
    >>> R = np.array([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])
    >>> A = transformer.fit_transform(R)
    >>> # A ≈ [[0.301, 0.398, 0.523], [0.222, 0.301, 0.398]]
    """

    def __init__(
        self,
        source_type: SignalTypeInput = "reflectance",
        epsilon: float = 1e-10,
        clip_negative: bool = True
    ):
        self.source_type = source_type
        self.epsilon = epsilon
        self.clip_negative = clip_negative

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectral data
        y : None
            Ignored

        Returns
        -------
        self
        """
        self.source_type_ = normalize_signal_type(self.source_type)

        # Validate source type
        valid_sources = [
            SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT,
            SignalType.TRANSMITTANCE, SignalType.TRANSMITTANCE_PERCENT
        ]
        if self.source_type_ not in valid_sources:
            raise ValueError(
                f"source_type must be one of {[s.value for s in valid_sources]}, "
                f"got '{self.source_type_.value}'"
            )

        self.is_percent_ = self.source_type_.is_percent
        return self

    def transform(self, X, y=None):
        """
        Transform reflectance/transmittance to absorbance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input spectral data

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Absorbance values
        """
        X = np.asarray(X, dtype=np.float64)

        # Convert percent to fraction if needed
        if self.is_percent_:
            X = X / 100.0

        # Handle edge cases
        if self.clip_negative:
            X = np.clip(X, self.epsilon, None)
        else:
            X = np.maximum(X, self.epsilon)

        # Apply log transform: A = -log10(X) = log10(1/X)
        A = -np.log10(X)

        return A

    def inverse_transform(self, X, y=None):
        """
        Convert absorbance back to reflectance/transmittance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Absorbance values

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reflectance or transmittance values
        """
        X = np.asarray(X, dtype=np.float64)

        # Inverse of A = -log10(R) is R = 10^(-A)
        R_or_T = np.power(10.0, -X)

        # Convert back to percent if original was percent
        if self.is_percent_:
            R_or_T = R_or_T * 100.0

        return R_or_T


class FromAbsorbance(TransformerMixin, BaseEstimator):
    """
    Convert absorbance to reflectance or transmittance.

    Applies the inverse log transform: R/T = 10^(-A)

    Parameters
    ----------
    target_type : str or SignalType
        Output signal type.
        Valid: "reflectance", "reflectance%", "transmittance", "transmittance%"

    Examples
    --------
    >>> from nirs4all.operators.transforms.signal_conversion import FromAbsorbance
    >>> transformer = FromAbsorbance(target_type="reflectance")
    >>> A = np.array([[0.301, 0.398], [0.222, 0.301]])
    >>> R = transformer.fit_transform(A)
    >>> # R ≈ [[0.5, 0.4], [0.6, 0.5]]
    """

    def __init__(self, target_type: SignalTypeInput = "reflectance"):
        self.target_type = target_type

    def fit(self, X, y=None):
        """Fit the transformer."""
        self.target_type_ = normalize_signal_type(self.target_type)

        valid_targets = [
            SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT,
            SignalType.TRANSMITTANCE, SignalType.TRANSMITTANCE_PERCENT
        ]
        if self.target_type_ not in valid_targets:
            raise ValueError(
                f"target_type must be one of {[s.value for s in valid_targets]}, "
                f"got '{self.target_type_.value}'"
            )

        self.is_percent_ = self.target_type_.is_percent
        return self

    def transform(self, X, y=None):
        """Transform absorbance to reflectance/transmittance."""
        X = np.asarray(X, dtype=np.float64)

        # R/T = 10^(-A)
        result = np.power(10.0, -X)

        if self.is_percent_:
            result = result * 100.0

        return result

    def inverse_transform(self, X, y=None):
        """Convert back to absorbance."""
        X = np.asarray(X, dtype=np.float64)

        if self.is_percent_:
            X = X / 100.0

        X = np.maximum(X, 1e-10)
        return -np.log10(X)


class PercentToFraction(TransformerMixin, BaseEstimator):
    """
    Convert percentage values to fractional [0, 1] range.

    Simply divides by 100.

    Examples
    --------
    >>> transformer = PercentToFraction()
    >>> X_pct = np.array([[50, 60], [70, 80]])
    >>> X_frac = transformer.fit_transform(X_pct)
    >>> # X_frac = [[0.5, 0.6], [0.7, 0.8]]
    """

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X, y=None):
        """Transform percent to fraction."""
        return np.asarray(X, dtype=np.float64) / 100.0

    def inverse_transform(self, X, y=None):
        """Transform fraction to percent."""
        return np.asarray(X, dtype=np.float64) * 100.0


class FractionToPercent(TransformerMixin, BaseEstimator):
    """
    Convert fractional [0, 1] values to percentage [0, 100] range.

    Simply multiplies by 100.

    Examples
    --------
    >>> transformer = FractionToPercent()
    >>> X_frac = np.array([[0.5, 0.6], [0.7, 0.8]])
    >>> X_pct = transformer.fit_transform(X_frac)
    >>> # X_pct = [[50, 60], [70, 80]]
    """

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X, y=None):
        """Transform fraction to percent."""
        return np.asarray(X, dtype=np.float64) * 100.0

    def inverse_transform(self, X, y=None):
        """Transform percent to fraction."""
        return np.asarray(X, dtype=np.float64) / 100.0


class KubelkaMunk(TransformerMixin, BaseEstimator):
    """
    Apply Kubelka-Munk transformation for diffuse reflectance.

    The Kubelka-Munk function: F(R) = (1-R)² / (2R)

    This is theoretically more appropriate for scattering media (powders)
    than simple log(1/R), though in NIR the benefit is dataset-dependent.

    Parameters
    ----------
    source_type : str or SignalType
        Input signal type.
        Valid: "reflectance", "reflectance%"
    epsilon : float, default=1e-10
        Small value to avoid division by zero

    Examples
    --------
    >>> from nirs4all.operators.transforms.signal_conversion import KubelkaMunk
    >>> transformer = KubelkaMunk(source_type="reflectance")
    >>> R = np.array([[0.5, 0.4], [0.6, 0.5]])
    >>> F_R = transformer.fit_transform(R)
    >>> # F_R[0,0] = (1-0.5)² / (2*0.5) = 0.25 / 1 = 0.25
    """

    def __init__(
        self,
        source_type: SignalTypeInput = "reflectance",
        epsilon: float = 1e-10
    ):
        self.source_type = source_type
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Fit the transformer."""
        self.source_type_ = normalize_signal_type(self.source_type)

        valid_sources = [SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT]
        if self.source_type_ not in valid_sources:
            raise ValueError(
                f"KubelkaMunk requires reflectance input, got '{self.source_type_.value}'"
            )

        self.is_percent_ = self.source_type_.is_percent
        return self

    def transform(self, X, y=None):
        """Apply Kubelka-Munk transformation."""
        X = np.asarray(X, dtype=np.float64)

        # Convert percent to fraction if needed
        if self.is_percent_:
            X = X / 100.0

        # Clip to avoid division by zero
        R = np.clip(X, self.epsilon, 1.0 - self.epsilon)

        # F(R) = (1-R)² / (2R)
        F_R = np.square(1.0 - R) / (2.0 * R)

        return F_R

    def inverse_transform(self, X, y=None):
        """
        Inverse Kubelka-Munk to recover reflectance.

        From F(R) = (1-R)² / (2R), solving for R:
        R = 1 + F - sqrt(F² + 2F)
        """
        F = np.asarray(X, dtype=np.float64)
        F = np.maximum(F, self.epsilon)

        # Solve quadratic: R = 1 + F - sqrt(F² + 2F)
        R = 1.0 + F - np.sqrt(np.square(F) + 2.0 * F)

        if self.is_percent_:
            R = R * 100.0

        return R


class SignalTypeConverter(TransformerMixin, BaseEstimator):
    """
    General-purpose signal type converter.

    Automatically determines the conversion path between source and target
    signal types and applies the appropriate transformation.

    Parameters
    ----------
    source_type : str or SignalType
        Input signal type
    target_type : str or SignalType
        Output signal type
    epsilon : float, default=1e-10
        Small value to avoid numerical issues

    Examples
    --------
    >>> from nirs4all.operators.transforms.signal_conversion import SignalTypeConverter
    >>> converter = SignalTypeConverter(
    ...     source_type="reflectance%",
    ...     target_type="absorbance"
    ... )
    >>> R_pct = np.array([[50, 40], [60, 50]])
    >>> A = converter.fit_transform(R_pct)
    """

    def __init__(
        self,
        source_type: SignalTypeInput = "reflectance",
        target_type: SignalTypeInput = "absorbance",
        epsilon: float = 1e-10
    ):
        self.source_type = source_type
        self.target_type = target_type
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Fit the converter by determining the conversion path."""
        self.source_type_ = normalize_signal_type(self.source_type)
        self.target_type_ = normalize_signal_type(self.target_type)

        # Build conversion pipeline
        self._build_conversion_pipeline()

        return self

    def _build_conversion_pipeline(self):
        """Build the sequence of transformations needed."""
        self.transformers_ = []

        src = self.source_type_
        tgt = self.target_type_

        # No conversion needed
        if src == tgt:
            return

        # Direct conversions
        if src in (SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT,
                   SignalType.TRANSMITTANCE, SignalType.TRANSMITTANCE_PERCENT):

            if tgt == SignalType.ABSORBANCE:
                self.transformers_.append(
                    ToAbsorbance(source_type=src, epsilon=self.epsilon)
                )
                return

            if tgt == SignalType.KUBELKA_MUNK and src.is_reflectance_based:
                self.transformers_.append(
                    KubelkaMunk(source_type=src, epsilon=self.epsilon)
                )
                return

        # Percent <-> Fraction conversions
        if src == SignalType.REFLECTANCE_PERCENT and tgt == SignalType.REFLECTANCE:
            self.transformers_.append(PercentToFraction())
            return

        if src == SignalType.REFLECTANCE and tgt == SignalType.REFLECTANCE_PERCENT:
            self.transformers_.append(FractionToPercent())
            return

        if src == SignalType.TRANSMITTANCE_PERCENT and tgt == SignalType.TRANSMITTANCE:
            self.transformers_.append(PercentToFraction())
            return

        if src == SignalType.TRANSMITTANCE and tgt == SignalType.TRANSMITTANCE_PERCENT:
            self.transformers_.append(FractionToPercent())
            return

        # Absorbance to R/T
        if src == SignalType.ABSORBANCE:
            if tgt in (SignalType.REFLECTANCE, SignalType.REFLECTANCE_PERCENT,
                       SignalType.TRANSMITTANCE, SignalType.TRANSMITTANCE_PERCENT):
                self.transformers_.append(FromAbsorbance(target_type=tgt))
                return

        # Multi-step conversions (e.g., %R -> A -> %T - though this is unusual)
        # For now, raise an error for unsupported conversions
        raise ValueError(
            f"Cannot convert from '{src.value}' to '{tgt.value}'. "
            f"Consider converting to absorbance as an intermediate step."
        )

    def transform(self, X, y=None):
        """Apply the conversion transformation."""
        X = np.asarray(X, dtype=np.float64)

        for transformer in self.transformers_:
            transformer.fit(X)
            X = transformer.transform(X)

        return X

    def inverse_transform(self, X, y=None):
        """Apply inverse transformation."""
        X = np.asarray(X, dtype=np.float64)

        # Apply inverse transformations in reverse order
        for transformer in reversed(self.transformers_):
            X = transformer.inverse_transform(X)

        return X

"""
Validation utilities for synthetic data generation.

This module provides functions to validate generated synthetic data
for correctness and expected properties.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ValidationError(Exception):
    """Exception raised when synthetic data validation fails."""

    pass


def validate_spectra(
    X: np.ndarray,
    expected_shape: Optional[Tuple[int, int]] = None,
    check_finite: bool = True,
    check_positive: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
) -> List[str]:
    """
    Validate generated spectra matrix.

    Args:
        X: Spectra matrix to validate.
        expected_shape: Expected (n_samples, n_wavelengths) shape.
        check_finite: Whether to check for NaN/Inf values.
        check_positive: Whether to require all positive values.
        value_range: Optional (min, max) expected range.

    Returns:
        List of validation warning messages (empty if all OK).

    Raises:
        ValidationError: If critical validation fails.

    Example:
        >>> X = np.random.randn(100, 500)
        >>> warnings = validate_spectra(X, expected_shape=(100, 500))
        >>> if warnings:
        ...     print("Warnings:", warnings)
    """
    warnings: List[str] = []

    # Check type
    if not isinstance(X, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(X).__name__}")

    # Check dimensions
    if X.ndim != 2:
        raise ValidationError(f"Expected 2D array, got {X.ndim}D")

    # Check shape
    if expected_shape is not None:
        if X.shape != expected_shape:
            raise ValidationError(
                f"Shape mismatch: expected {expected_shape}, got {X.shape}"
            )

    # Check finite values
    if check_finite:
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        if n_nan > 0:
            raise ValidationError(f"Found {n_nan} NaN values in spectra")
        if n_inf > 0:
            raise ValidationError(f"Found {n_inf} Inf values in spectra")

    # Check positive values
    if check_positive:
        n_negative = (X < 0).sum()
        if n_negative > 0:
            warnings.append(
                f"Found {n_negative} negative values ({100*n_negative/X.size:.2f}%)"
            )

    # Check value range
    if value_range is not None:
        min_val, max_val = value_range
        if X.min() < min_val:
            warnings.append(
                f"Minimum value {X.min():.4f} below expected {min_val}"
            )
        if X.max() > max_val:
            warnings.append(
                f"Maximum value {X.max():.4f} above expected {max_val}"
            )

    return warnings


def validate_concentrations(
    C: np.ndarray,
    n_samples: Optional[int] = None,
    n_components: Optional[int] = None,
    check_normalized: bool = False,
    tolerance: float = 0.01,
) -> List[str]:
    """
    Validate concentration matrix.

    Args:
        C: Concentration matrix to validate.
        n_samples: Expected number of samples.
        n_components: Expected number of components.
        check_normalized: Whether concentrations should sum to 1.
        tolerance: Tolerance for normalization check.

    Returns:
        List of validation warning messages.

    Raises:
        ValidationError: If critical validation fails.
    """
    warnings: List[str] = []

    if not isinstance(C, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(C).__name__}")

    if C.ndim != 2:
        raise ValidationError(f"Expected 2D concentration matrix, got {C.ndim}D")

    if n_samples is not None and C.shape[0] != n_samples:
        raise ValidationError(
            f"Expected {n_samples} samples, got {C.shape[0]}"
        )

    if n_components is not None and C.shape[1] != n_components:
        raise ValidationError(
            f"Expected {n_components} components, got {C.shape[1]}"
        )

    # Check for negative concentrations
    n_negative = (C < 0).sum()
    if n_negative > 0:
        warnings.append(f"Found {n_negative} negative concentration values")

    # Check normalization
    if check_normalized:
        row_sums = C.sum(axis=1)
        deviations = np.abs(row_sums - 1.0)
        if deviations.max() > tolerance:
            warnings.append(
                f"Concentrations not normalized: max deviation = {deviations.max():.4f}"
            )

    return warnings


def validate_wavelengths(
    wavelengths: np.ndarray,
    expected_range: Optional[Tuple[float, float]] = None,
    check_monotonic: bool = True,
    check_uniform: bool = True,
) -> List[str]:
    """
    Validate wavelength array.

    Args:
        wavelengths: Wavelength array to validate.
        expected_range: Optional (min, max) expected range in nm.
        check_monotonic: Whether to check for monotonically increasing values.
        check_uniform: Whether to check for uniform spacing.

    Returns:
        List of validation warning messages.

    Raises:
        ValidationError: If critical validation fails.
    """
    warnings: List[str] = []

    if not isinstance(wavelengths, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(wavelengths).__name__}")

    if wavelengths.ndim != 1:
        raise ValidationError(f"Expected 1D wavelength array, got {wavelengths.ndim}D")

    if len(wavelengths) < 2:
        raise ValidationError(
            f"Wavelength array too short: {len(wavelengths)} points"
        )

    # Check range
    if expected_range is not None:
        min_wl, max_wl = expected_range
        if wavelengths.min() < min_wl or wavelengths.max() > max_wl:
            warnings.append(
                f"Wavelength range [{wavelengths.min():.1f}, {wavelengths.max():.1f}] "
                f"outside expected [{min_wl}, {max_wl}]"
            )

    # Check monotonic
    if check_monotonic:
        diffs = np.diff(wavelengths)
        if not np.all(diffs > 0):
            raise ValidationError("Wavelengths must be monotonically increasing")

    # Check uniform spacing
    if check_uniform:
        diffs = np.diff(wavelengths)
        if diffs.std() / diffs.mean() > 0.01:  # 1% tolerance
            warnings.append("Wavelength spacing is not uniform")

    return warnings


def validate_synthetic_output(
    X: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Validate complete synthetic generation output.

    Args:
        X: Generated spectra (n_samples, n_wavelengths).
        C: Concentration matrix (n_samples, n_components).
        E: Component spectra (n_components, n_wavelengths).
        wavelengths: Optional wavelength array.

    Returns:
        List of all validation warnings.

    Raises:
        ValidationError: If critical validation fails.

    Example:
        >>> from nirs4all.data.synthetic import SyntheticNIRSGenerator
        >>> gen = SyntheticNIRSGenerator(random_state=42)
        >>> X, C, E = gen.generate(100)
        >>> warnings = validate_synthetic_output(X, C, E, gen.wavelengths)
    """
    all_warnings: List[str] = []

    n_samples, n_wavelengths = X.shape
    n_components = C.shape[1]

    # Validate spectra
    all_warnings.extend(
        validate_spectra(X, expected_shape=(n_samples, n_wavelengths))
    )

    # Validate concentrations
    all_warnings.extend(
        validate_concentrations(C, n_samples=n_samples, n_components=n_components)
    )

    # Validate component spectra shape
    if E.shape != (n_components, n_wavelengths):
        raise ValidationError(
            f"Component spectra shape mismatch: expected "
            f"({n_components}, {n_wavelengths}), got {E.shape}"
        )

    # Validate wavelengths if provided
    if wavelengths is not None:
        all_warnings.extend(validate_wavelengths(wavelengths))
        if len(wavelengths) != n_wavelengths:
            raise ValidationError(
                f"Wavelength array length {len(wavelengths)} does not match "
                f"spectra width {n_wavelengths}"
            )

    return all_warnings

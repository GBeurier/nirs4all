"""
Centralized header unit utilities for consistent handling across the codebase.

This module provides a single source of truth for:
- Axis labels based on header unit type
- X-values extraction from headers
- Axis orientation (for wavenumber inversion)

All visualization code should use these utilities instead of inline logic.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from nirs4all.data._features import HeaderUnit, normalize_header_unit


# Canonical axis labels - single source of truth
AXIS_LABELS = {
    HeaderUnit.WAVENUMBER: "Wavenumber (cm⁻¹)",
    HeaderUnit.WAVELENGTH: "Wavelength (nm)",
    HeaderUnit.NONE: "Feature Index",
    HeaderUnit.TEXT: "Features",
    HeaderUnit.INDEX: "Feature Index",
}

# Default label when unit unknown or invalid
DEFAULT_AXIS_LABEL = "Features"


def get_axis_label(unit: Union[str, HeaderUnit]) -> str:
    """Get the appropriate axis label for a given unit type.

    Args:
        unit: Header unit type (string like "cm-1", "nm" or HeaderUnit enum)

    Returns:
        Human-readable axis label string

    Examples:
        >>> get_axis_label("cm-1")
        'Wavenumber (cm⁻¹)'
        >>> get_axis_label(HeaderUnit.WAVELENGTH)
        'Wavelength (nm)'
        >>> get_axis_label("unknown")  # Falls back gracefully
        'Features'
    """
    try:
        normalized = normalize_header_unit(unit) if isinstance(unit, str) else unit
        return AXIS_LABELS.get(normalized, DEFAULT_AXIS_LABEL)
    except ValueError:
        return DEFAULT_AXIS_LABEL


def get_x_values_and_label(
    headers: Optional[List[str]],
    header_unit: Union[str, HeaderUnit],
    n_features: int
) -> Tuple[np.ndarray, str]:
    """Get x-axis values and label from headers and unit.

    This is the main utility function for chart x-axis setup. It handles:
    - Numeric headers (wavelengths/wavenumbers) → parsed float array
    - Non-numeric headers → fallback to indices
    - Missing or mismatched headers → fallback to indices

    Args:
        headers: List of header strings (wavelengths, feature names, etc.)
        header_unit: Header unit type ("cm-1", "nm", "none", "text", "index")
        n_features: Number of features (for fallback and validation)

    Returns:
        Tuple of (x_values array, axis_label string)

    Examples:
        >>> x_vals, label = get_x_values_and_label(["4000", "4500", "5000"], "cm-1", 3)
        >>> x_vals
        array([4000., 4500., 5000.])
        >>> label
        'Wavenumber (cm⁻¹)'

        >>> x_vals, label = get_x_values_and_label(None, "cm-1", 5)
        >>> x_vals
        array([0, 1, 2, 3, 4])
        >>> label
        'Features'
    """
    # No headers or mismatched length - use indices with generic label
    if headers is None or len(headers) != n_features:
        return np.arange(n_features), DEFAULT_AXIS_LABEL

    # Try to normalize the unit
    try:
        normalized = normalize_header_unit(header_unit) if isinstance(header_unit, str) else header_unit
    except ValueError:
        normalized = HeaderUnit.NONE

    # For numeric unit types, try to parse headers as floats
    if normalized in (HeaderUnit.WAVENUMBER, HeaderUnit.WAVELENGTH, HeaderUnit.NONE, HeaderUnit.INDEX):
        try:
            x_values = np.array([float(h) for h in headers])
            return x_values, AXIS_LABELS.get(normalized, DEFAULT_AXIS_LABEL)
        except (ValueError, TypeError):
            # Headers not numeric - fall back to indices
            return np.arange(n_features), DEFAULT_AXIS_LABEL

    # For TEXT unit - use indices but with TEXT label
    return np.arange(n_features), AXIS_LABELS.get(normalized, DEFAULT_AXIS_LABEL)


def should_invert_x_axis(x_values: np.ndarray) -> bool:
    """Check if x-axis should be inverted (for wavenumber convention).

    In spectroscopy, wavenumber (cm⁻¹) axes are often displayed in
    descending order (high to low) if the data is ordered that way.

    Args:
        x_values: Array of x-axis values

    Returns:
        True if x_values are in descending order and should be displayed as such
    """
    if len(x_values) < 2:
        return False
    return x_values[0] > x_values[-1]


def apply_x_axis_limits(ax, x_values: np.ndarray) -> None:
    """Apply appropriate x-axis limits to preserve data ordering.

    Matplotlib may auto-sort axis values. This function sets explicit
    limits to preserve the original ordering (ascending or descending).

    Args:
        ax: Matplotlib Axes object
        x_values: Array of x-axis values
    """
    if len(x_values) > 1 and x_values[0] > x_values[-1]:
        ax.set_xlim(x_values[0], x_values[-1])

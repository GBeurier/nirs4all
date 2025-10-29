"""Constants and enums for feature management."""

from enum import Enum
from typing import Union

# Default processing identifier
DEFAULT_PROCESSING = "raw"


class FeatureLayout(str, Enum):
    """Feature data layout formats.

    String values ensure backward compatibility with existing pipelines
    that use layout="3d_transpose" as strings.
    """
    FLAT_2D = "2d"                           # (samples, processings * features)
    FLAT_2D_INTERLEAVED = "2d_interleaved"   # (samples, features * processings)
    VOLUME_3D = "3d"                         # (samples, processings, features)
    VOLUME_3D_TRANSPOSE = "3d_transpose"     # (samples, features, processings)


class HeaderUnit(str, Enum):
    """Feature header unit types.

    Defines the type of measurement units used in feature headers.
    """
    WAVENUMBER = "cm-1"    # Wavenumber in cm⁻¹
    WAVELENGTH = "nm"      # Wavelength in nanometers
    NONE = "none"          # No units
    TEXT = "text"          # Text labels
    INDEX = "index"        # Numeric indices


# Type aliases for backward compatibility
LayoutType = Union[str, FeatureLayout]
HeaderUnitType = Union[str, HeaderUnit]


def normalize_layout(layout: LayoutType) -> FeatureLayout:
    """Convert string layout to enum for backward compatibility.

    Args:
        layout: Layout as string or enum

    Returns:
        FeatureLayout enum value

    Raises:
        ValueError: If layout string is invalid
    """
    if isinstance(layout, FeatureLayout):
        return layout

    try:
        return FeatureLayout(layout)
    except ValueError:
        valid = [e.value for e in FeatureLayout]
        raise ValueError(f"Invalid layout '{layout}'. Valid options: {valid}")


def normalize_header_unit(unit: HeaderUnitType) -> HeaderUnit:
    """Convert string header unit to enum.

    Args:
        unit: Unit as string or enum

    Returns:
        HeaderUnit enum value

    Raises:
        ValueError: If unit string is invalid
    """
    if isinstance(unit, HeaderUnit):
        return unit

    try:
        return HeaderUnit(unit)
    except ValueError:
        valid = [e.value for e in HeaderUnit]
        raise ValueError(f"Invalid header unit '{unit}'. Valid options: {valid}")

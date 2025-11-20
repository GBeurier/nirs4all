"""Feature header and unit management."""

from typing import Optional, List
from nirs4all.data._features.feature_constants import HeaderUnit, normalize_header_unit


class HeaderManager:
    """Manages feature headers and their units.

    Handles storage and retrieval of feature names/wavelengths and their
    measurement units.

    Attributes:
        headers: List of feature header strings.
        header_unit: Unit type for the headers.
    """

    def __init__(self):
        """Initialize with no headers."""
        self._headers: Optional[List[str]] = None
        self._header_unit: HeaderUnit = HeaderUnit.WAVENUMBER

    @property
    def headers(self) -> Optional[List[str]]:
        """Get the feature headers.

        Returns:
            List of header strings, or None if not set.
        """
        return self._headers

    @property
    def header_unit(self) -> str:
        """Get the unit type of the headers.

        Returns:
            Unit type string for backward compatibility.
        """
        return self._header_unit.value

    def set_headers(self, headers: Optional[List[str]], unit: str = "cm-1") -> None:
        """Set feature headers with unit metadata.

        Args:
            headers: List of header strings (wavelengths, feature names, etc.).
            unit: Unit type - "cm-1" (wavenumber), "nm" (wavelength),
                  "none", "text", "index".
        """
        self._headers = headers
        self._header_unit = normalize_header_unit(unit)

    def clear_headers(self) -> None:
        """Clear the headers."""
        self._headers = None

    def __repr__(self) -> str:
        num_headers = len(self._headers) if self._headers else 0
        return f"HeaderManager(num_headers={num_headers}, unit={self._header_unit.value})"

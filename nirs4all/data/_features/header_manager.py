"""Feature header and unit management."""

from typing import Optional

from nirs4all.data._features.feature_constants import HeaderUnit, normalize_header_unit


class HeaderManager:
    """Manages feature headers and their units.

    Handles storage and retrieval of feature names/wavelengths and their
    measurement units.

    Attributes:
        headers: List of feature header strings.
        header_unit: Unit type for the headers.
        header_unit_recorded: Whether the unit was supplied explicitly rather
            than taken from the implicit ``cm-1`` default.
    """

    # Class-level default keeps instances restored from older pickles readable.
    _header_unit_recorded: bool = False

    def __init__(self) -> None:
        """Initialize with no headers."""
        self._headers: list[str] | None = None
        self._header_unit: HeaderUnit = HeaderUnit.WAVENUMBER

    @property
    def headers(self) -> list[str] | None:
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

    @property
    def header_unit_recorded(self) -> bool:
        """Whether the header unit was supplied explicitly.

        Returns:
            False while the unit is the implicit ``cm-1`` default.
        """
        return self._header_unit_recorded

    def set_headers(self, headers: list[str] | None, unit: str | None = None) -> None:
        """Set feature headers with unit metadata.

        Args:
            headers: List of header strings (wavelengths, feature names, etc.).
            unit: Unit type - "cm-1" (wavenumber), "nm" (wavelength),
                  "none", "text", "index". ``None`` keeps the implicit
                  ``cm-1`` default without recording it as a declared unit.
        """
        self._headers = headers
        self._header_unit = normalize_header_unit(unit if unit is not None else HeaderUnit.WAVENUMBER)
        self._header_unit_recorded = unit is not None

    def clear_headers(self) -> None:
        """Clear the headers."""
        self._headers = None

    def __repr__(self) -> str:
        num_headers = len(self._headers) if self._headers else 0
        return f"HeaderManager(num_headers={num_headers}, unit={self._header_unit.value})"

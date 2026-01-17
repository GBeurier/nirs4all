"""
Base classes for nirs4all operators.

This module provides foundational mixin classes and base classes for building
custom operators that integrate with the nirs4all pipeline system.

Classes:
    SpectraTransformerMixin: Base class for wavelength-aware spectral transformations
"""

from .spectra_mixin import SpectraTransformerMixin

__all__ = ["SpectraTransformerMixin"]

# -*- coding: utf-8 -*-
# nirs4all/spectra/__init__.py

"""This module initializes the spectra package for NIRS4ALL.
It imports the main classes and functions used for handling spectral data,
features, and targets in the NIRS4ALL project.
"""

from .SpectraDataset import SpectraDataset
from .SpectraFeatures import SpectraFeatures
from .SpectraTargets import SpectraTargets
# from .SpectraResults import SpectraResults

__all__ = [
    "SpectraDataset",
    "SpectraFeatures",
    "SpectraTargets",
    # "SpectraResults",
]
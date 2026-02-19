"""
Splitters module for presets.

This module contains data splitting presets and utilities.
"""
from .grouped_wrapper import GroupedSplitterWrapper
from .splitters import (
    BinnedStratifiedGroupKFold,
    KBinsStratifiedSplitter,
    KennardStoneSplitter,
    KMeansSplitter,
    SPlitSplitter,
    SPXYFold,
    SPXYGFold,
    SPXYSplitter,
    SystematicCircularSplitter,
)

__all__ = [
    "KennardStoneSplitter",
    "SPXYSplitter",
    "SPXYFold",
    "SPXYGFold",
    "KMeansSplitter",
    "SPlitSplitter",
    "SystematicCircularSplitter",
    "KBinsStratifiedSplitter",
    "BinnedStratifiedGroupKFold",
    "GroupedSplitterWrapper",
]

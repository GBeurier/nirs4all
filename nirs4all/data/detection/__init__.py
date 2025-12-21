"""
Auto-detection module for dataset configuration.

This module provides enhanced auto-detection capabilities for file formats,
delimiters, headers, signal types, and other file parameters.
"""

from .detector import (
    AutoDetector,
    DetectionResult,
    detect_file_parameters,
    detect_signal_type,
)

__all__ = [
    "AutoDetector",
    "DetectionResult",
    "detect_file_parameters",
    "detect_signal_type",
]

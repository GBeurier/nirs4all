"""
Pipeline module for nirs4all package.

This module contains pipeline classes for processing workflows.
"""

from .config import PipelineConfigs
from .runner import PipelineRunner

__all__ = [
    'PipelineConfigs',
    'PipelineRunner',
]

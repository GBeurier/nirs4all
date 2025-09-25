"""
Command-line interface for the nirs4all package.
"""

from .main import nirs4all_cli
from .presets import load_preset, apply_preset_to_config

__all__ = [
    'nirs4all_cli',
    'load_preset',
    'apply_preset_to_config'
]
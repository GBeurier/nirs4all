"""
NIRS4All - A comprehensive package for Near-Infrared Spectroscopy data processing and analysis.

This package provides tools for spectroscopy data handling, preprocessing, model building,
and pipeline management with support for multiple ML backends.
"""
__version__ = "0.2.1"

import os
if os.environ.get('DISABLE_EMOJIS') == '1':  # Set to True to always disable
    import re
    original_print = __builtins__['print']

    def strip_emojis(text):
        # Force ASCII-only output by encoding to ASCII and ignoring errors
        try:
            # Convert to ASCII, ignore non-ASCII characters
            ascii_text = text.encode('ascii', 'ignore').decode('ascii')
            # Also remove ANSI escape codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub('', ascii_text)
        except (UnicodeError, AttributeError):
            return str(text)

    def emoji_free_print(*args, **kwargs):
        new_args = []
        for arg in args:
            # Convert everything to string first, then strip emojis
            try:
                str_arg = str(arg)
                cleaned_arg = strip_emojis(str_arg)
                new_args.append(cleaned_arg)
            except (UnicodeError, AttributeError):
                new_args.append("[UNICODE ERROR]")
        original_print(*new_args, **kwargs)
    __builtins__['print'] = emoji_free_print


# Core pipeline components - most commonly used
from .pipeline import PipelineRunner, PipelineConfigs, PipelineHistory
from .controllers import register_controller, CONTROLLER_REGISTRY

# Utility functions for backend detection
from .utils import (
    is_tensorflow_available,
    # is_torch_available,
    is_gpu_available,
    framework
)

# Make commonly used classes available at package level
__all__ = [
    # Pipeline components
    "PipelineRunner",
    "PipelineConfigs",
    "PipelineHistory",

    # Controller system
    "register_controller",
    "CONTROLLER_REGISTRY",

    # Utilities
    "is_tensorflow_available",
    # "is_torch_available",
    "is_gpu_available",
    "framework"
]

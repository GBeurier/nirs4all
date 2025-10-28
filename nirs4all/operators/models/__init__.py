"""
Models module for presets.

This module contains model definitions and references organized by framework.
"""

from .base import BaseModelOperator

# Import TensorFlow models
from .tensorflow.nicon import *
from .tensorflow.generic import *

# Import PyTorch models (currently commented out)
# from .pytorch.nicon import *
# from .pytorch.generic import *

__all__ = [
    "BaseModelOperator",
]

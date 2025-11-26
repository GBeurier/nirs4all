"""
Models module for presets.

This module contains model definitions and references organized by framework.
"""

from .base import BaseModelOperator

# Import sklearn models
from .sklearn.pls import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS
from .sklearn.lwpls import LWPLS

# Import TensorFlow models
from .tensorflow.nicon import *
from .tensorflow.generic import *

# Import PyTorch models (currently commented out)
# from .pytorch.nicon import *
# from .pytorch.generic import *

__all__ = [
    "BaseModelOperator",
    "PLSDA",
    "IKPLS",
    "OPLS",
    "OPLSDA",
    "MBPLS",
    "DiPLS",
    "SparsePLS",
    "LWPLS",
]

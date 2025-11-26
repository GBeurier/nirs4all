"""
Models module for presets.

This module contains model definitions and references organized by framework.
"""

from .base import BaseModelOperator

# Import sklearn models
from .sklearn.pls import PLSDA, IKPLS, OPLS, OPLSDA, MBPLS, DiPLS, SparsePLS, SIMPLS
from .sklearn.lwpls import LWPLS
from .sklearn.ipls import IntervalPLS
from .sklearn.robust_pls import RobustPLS
from .sklearn.recursive_pls import RecursivePLS
from .sklearn.kopls import KOPLS
from .sklearn.nlpls import KernelPLS, NLPLS, KPLS

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
    "SIMPLS",
    "IntervalPLS",
    "RobustPLS",
    "RecursivePLS",
    "KOPLS",
    "KernelPLS",
    "NLPLS",
    "KPLS",
]

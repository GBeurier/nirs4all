"""aompls — AOM-PLS compact (PLS1) implemented in C++ with pybind11 bindings."""

from .sklearn import AOMPLSCompact  # noqa: F401
from .tune import tune  # noqa: F401

__all__ = ["AOMPLSCompact", "tune"]
__version__ = "0.1.0"

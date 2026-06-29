"""Opt-in operators backed by the portable ``nirs4all-methods`` engine.

These operators are **opt-in**: they are not registered in the default
operator namespace and existing operators / dag-ml behaviour are untouched.
They dispatch to the ``n4m`` Python binding (the ctypes wrapper around the
portable C ABI ``libn4m``), so the same numerical core that powers every
language binding also powers these nirs4all pipeline steps.

Each operator is a plain scikit-learn-contract object (``TransformerMixin``
for :class:`MethodsSNV`, ``RegressorMixin`` for :class:`MethodsPLS`), so it is
matched and executed by the existing nirs4all controllers under *both* the
legacy engine and the dag-ml engine without any special handling: the dag-ml
path imports the operator by its fully-qualified name and calls
``fit`` / ``transform`` / ``predict`` / ``get_params`` exactly like any other
sklearn estimator.

Import this module only when ``nirs4all-methods`` (the ``n4m`` distribution)
is installed; importing it without the binding raises a clear ImportError.
"""

from .n4m_ops import METHODS_AVAILABLE, MethodsPLS, MethodsSNV

__all__ = ["METHODS_AVAILABLE", "MethodsPLS", "MethodsSNV"]

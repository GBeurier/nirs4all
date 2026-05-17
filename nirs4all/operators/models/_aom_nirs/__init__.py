"""aom-nirs: Adaptive Operator-Mixture PLS and Ridge for NIR spectroscopy.

Companion code for the Talanta paper *Operator-adaptive PLS and Ridge
calibration for NIR spectroscopy*.

Subpackages
-----------
- :mod:`aom_nirs.pls` — AOM-PLS, POP-PLS, AOM-PLS-DA, POP-PLS-DA. The
  operator-adaptive PLS family with NIPALS and SIMPLS engines, five
  selection policies (global / per-component / soft / superblock /
  none), and the strict-linear operator bank (identity, Savitzky-Golay,
  finite difference, detrend, Norris-Williams, Whittaker, FCK).
- :mod:`aom_nirs.ridge` — AOM-Ridge family: ``AOMRidgeRegressor``,
  ``AOMRidgeClassifier``, ``AOMRidgeBlender``, ``AOMRidgeAutoSelector``,
  ``AOMRidgePLS``, ``AOMMultiKernelRidge``, ``AOMMultiBranchMKL``,
  ``AOMLocalRidge``.
- :mod:`aom_nirs.fast` — FastAOM chain-screening framework with
  ``SingleChainPLSRidge``, ``HardAOMChainPLSRidge``,
  ``SoftAOMChainPLSRidge``, ``SparseMultiKernelRidge``, and the
  ``FastAOMPLSRidge`` orchestrator.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]

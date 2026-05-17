"""AOM-PLS public entry point for `nirs4all`.

Re-exports the canonical implementation from
``nirs4all.operators.models._aom_nirs.pls`` (vendored copy of the
``aom-nirs`` package; once ``aom-nirs`` is on PyPI this file will
switch to ``from aom_nirs.pls import ...``).

History
-------
The pre-migration pure-Python implementation (with ``gate='sparsemax'``,
``FFTBandpass`` / ``WaveletProjection`` operators, and in-fit torch
dispatch) lived directly in this file and is preserved at
``aom_nirs/_archive/deprecated_nirs4all/aom_pls.py`` for reference.
Those features are not part of the Talanta paper variants and were
dropped in favour of the canonical AOM-PLS / POP-PLS family.
"""

from __future__ import annotations

from nirs4all.operators.models._aom_nirs.pls.estimators import (
    AOMPLSRegressor,
    POPPLSRegressor,
)
from nirs4all.operators.models._aom_nirs.pls.operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    ExplicitMatrixOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WhittakerOperator,
)
from nirs4all.operators.models._aom_nirs.pls.banks import (
    bank_by_name,
    compact_bank,
    default_bank,
    extended_bank,
)


def default_operator_bank(p: int):
    """Return the default 100-operator AOM bank for spectra of length ``p``."""
    return default_bank(p=p)


__all__ = [
    "AOMPLSRegressor",
    "POPPLSRegressor",
    "LinearSpectralOperator",
    "IdentityOperator",
    "SavitzkyGolayOperator",
    "FiniteDifferenceOperator",
    "DetrendProjectionOperator",
    "NorrisWilliamsOperator",
    "WhittakerOperator",
    "ComposedOperator",
    "ExplicitMatrixOperator",
    "compact_bank",
    "default_bank",
    "extended_bank",
    "bank_by_name",
    "default_operator_bank",
]

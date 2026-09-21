"""AOM/POP-PLS public entry point backed by the native n4m engine."""

from __future__ import annotations

from n4m.model_selection.aom_search import AOMPLSRegressor, POPPLSRegressor

from nirs4all.operators.models._aom_nirs.pls.banks import (
    bank_by_name,
    compact_bank,
    default_bank,
    extended_bank,
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

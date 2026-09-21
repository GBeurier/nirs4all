"""Shared AOM operator helpers and Python classification implementations.

The AOM/POP regressors are native n4m estimators. The remaining helpers are
used by the classification and AOM-Ridge surfaces that have not moved to n4m.
"""

from contextlib import suppress

from n4m.model_selection.aom_search import AOMPLSRegressor, POPPLSRegressor

from .banks import (
    bank_by_name,
    compact_bank,
    default_bank,
    extended_bank,
)
from .operators import (
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

with suppress(ImportError):  # pragma: no cover - import-time guard
    from .classification import AOMPLSDAClassifier, POPPLSDAClassifier  # noqa: F401

"""AOM_v0: Operator-Adaptive Partial Least Squares.

Public package for the Operator-Adaptive PLS framework. The package implements
standard PLS, AOM (global) selection, POP (per-component) selection, soft
mixtures, and superblock baselines, all in a unified estimator family. Both
NIPALS and SIMPLS engines are provided, with materialized references and fast
covariance/adjoint variants.

The mathematical convention is `(X A^T)^T Y = A X^T Y`, which lets covariance
SIMPLS evaluate operator candidates in the cross-covariance space.

The package targets `bench/AOM_v0` and is independent from the production
`nirs4all` library.
"""

from .operators import (
    LinearSpectralOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
    FiniteDifferenceOperator,
    DetrendProjectionOperator,
    NorrisWilliamsOperator,
    WhittakerOperator,
    ComposedOperator,
    ExplicitMatrixOperator,
)
from .banks import (
    compact_bank,
    default_bank,
    extended_bank,
    bank_by_name,
)

# Lazy imports for higher-level components so that lower-level modules can be
# imported in isolation during incremental development and tests.
try:  # pragma: no cover - import-time guard
    from .estimators import AOMPLSRegressor, POPPLSRegressor  # noqa: F401
    from .classification import AOMPLSDAClassifier, POPPLSDAClassifier  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"

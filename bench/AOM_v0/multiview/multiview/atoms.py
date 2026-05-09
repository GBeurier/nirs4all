"""Canonical Phase-11 atoms — importable Python classes.

Surfaces the four atoms used in Phase-11 super-learner benchmarks
(`run_smoke4_phase11.py` ``_atom_bases``) as proper public classes so
they can be referenced from registry YAML or imported outside the
benchmark scripts.

Atoms
-----
- :class:`LazyV2AOM` — ``AOMPLSRegressor`` on the V2-style combined view
  bank (compact, K=3, equal_width blocks + global). Was previously the
  inline ``_build_lazy_v2_aom`` factory in ``run_smoke4_phase11.py``.
- :class:`AOMMoEMultiK` — re-exported from :mod:`.moe_advanced`. Use
  ``K_list=(3, 5, 7)`` for the canonical ``multiK-3-5-7`` variant.
- :class:`AOMMoERegressor` — re-exported from :mod:`.moe`. Use
  ``expert_layout="per_preproc"``, ``routing="soft"`` for the canonical
  ``moe-preproc-soft`` variant.
- :class:`AOMPLSRegressor` — re-exported from :mod:`aompls.estimators`.
  Use ``operator_bank="compact"``, ``criterion="holdout"`` for the
  canonical ``aom-pls-compact`` variant.
"""

from __future__ import annotations

from aompls.estimators import AOMPLSRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from .moe import AOMMoERegressor
from .moe_advanced import AOMMoEMultiK
from .views import ViewBuilder

__all__ = [
    "LazyV2AOM",
    "AOMMoEMultiK",
    "AOMMoERegressor",
    "AOMPLSRegressor",
]


class LazyV2AOM(RegressorMixin, BaseEstimator):
    """AOMPLSRegressor on a V2-style combined view bank, built lazily.

    The view bank depends on the input feature count ``p`` and is therefore
    constructed in :meth:`fit` rather than ``__init__``. This is the
    canonical Phase-11 ``lazy-V2-AOM`` atom.

    Parameters
    ----------
    max_components : int, default=10
        Maximum number of latent components for the inner AOMPLSRegressor.
    K : int, default=3
        Number of equal-width wavelength blocks for the view bank.
    bank_name : str, default="compact"
        Per-view operator bank name (forwarded to ``ViewBuilder.combined``).
    strategy : str, default="equal_width"
        Block partition strategy (forwarded to ``ViewBuilder.combined``).
    include_global : bool, default=True
        Include the global (no-mask) view in the combined bank.
    engine : str, default="simpls_covariance"
        Inner SIMPLS engine for AOMPLSRegressor.
    selection : str, default="global"
        AOM selection scope.
    criterion : str, default="holdout"
        Component selection criterion.
    random_state : int, default=0
        Seeding for the inner AOMPLSRegressor.

    Attributes
    ----------
    estimator_ : AOMPLSRegressor
        The fitted inner estimator (set by :meth:`fit`).
    """

    def __init__(
        self,
        max_components: int = 10,
        K: int = 3,
        bank_name: str = "compact",
        strategy: str = "equal_width",
        include_global: bool = True,
        engine: str = "simpls_covariance",
        selection: str = "global",
        criterion: str = "holdout",
        random_state: int = 0,
    ) -> None:
        self.max_components = max_components
        self.K = K
        self.bank_name = bank_name
        self.strategy = strategy
        self.include_global = include_global
        self.engine = engine
        self.selection = selection
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        bank = ViewBuilder.combined(
            bank_name=self.bank_name,
            K=self.K,
            strategy=self.strategy,
            include_global=self.include_global,
        ).build(p=X.shape[1])
        self.estimator_ = AOMPLSRegressor(
            n_components="auto",
            max_components=self.max_components,
            engine=self.engine,
            selection=self.selection,
            criterion=self.criterion,
            operator_bank=bank,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

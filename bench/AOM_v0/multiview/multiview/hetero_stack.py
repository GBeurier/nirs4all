"""Heterogeneous stacking ensemble of multi-view + parallel-session bases.

Combines:
- moe-preproc-soft / moe-view-soft (multi-view MoE)
- AOM-Ridge (parallel session: bench/AOM_v0/Ridge)
- TabPFN (transformer-based ICL)
- Optionally: NICON-V2 stack

Trains via standard OOF stacking (StackingHybrid pattern) — Ridge meta on
out-of-fold predictions, refit each base on full training data for the
final predict path.

Use `make_hetero_stack(seed, max_components, p)` to construct the
StackingHybrid with the right base estimators wired up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Ensure parallel-session packages are importable (if installed locally).
# __file__ = bench/AOM_v0/multiview/multiview/hetero_stack.py → parents[2] = AOM_v0
_AOM_ROOT = Path(__file__).resolve().parents[2]
_RIDGE_ROOT = _AOM_ROOT / "Ridge"
_NICON_ROOT = _AOM_ROOT.parent / "nicon_v2"

if _RIDGE_ROOT.exists() and str(_RIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RIDGE_ROOT))
if _NICON_ROOT.exists() and str(_NICON_ROOT) not in sys.path:
    sys.path.insert(0, str(_NICON_ROOT))


class TabPFNAdapter(BaseEstimator, RegressorMixin):
    """sklearn-compatible TabPFN regressor with a soft sample cap.

    TabPFN is designed for n <= 10000; we cap inputs to 5000 by random
    subsample (with the seed) to avoid OOM on huge cohorts. This is the
    standard tabpfn_paper recipe for big datasets.

    Parameters
    ----------
    n_max : int
        Cap on training samples passed to TabPFN.
    seed : int
        RNG seed for subsampling.
    device : str
        "cpu", "cuda", or "auto".
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_max: int = 5000,
        seed: int = 0,
        device: str = "cpu",
        ignore_pretraining_limits: bool = True,
    ) -> None:
        self.n_max = n_max
        self.seed = seed
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNAdapter":
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if n > self.n_max:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.n_max, replace=False)
            X = X[idx]
            y = y[idx]
        # TabPFN signature varies across versions — pass the supported subset.
        kwargs = {"device": self.device, "random_state": self.seed}
        try:
            self._model = TabPFNRegressor(
                ignore_pretraining_limits=self.ignore_pretraining_limits,
                **kwargs,
            )
        except TypeError:
            self._model = TabPFNRegressor(**kwargs)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_model"):
            raise RuntimeError("TabPFNAdapter not fitted")
        X = np.asarray(X, dtype=float)
        return np.asarray(self._model.predict(X)).ravel()


def make_aom_ridge(
    seed: int = 0,
    selection: str = "superblock",
    fast: bool = False,
) -> Optional[BaseEstimator]:
    """Build AOM-Ridge from bench/AOM_v0/Ridge (parallel session).

    ``fast=True`` uses ``alpha=1.0`` (no alpha CV grid search) for ~10x
    speed-up at the cost of suboptimal regularisation. Use for full-cohort
    sweeps where end-to-end runtime dominates.

    Returns `None` if the package can't be imported.
    """
    try:
        from aomridge.estimators import AOMRidgeRegressor
    except ImportError:
        return None
    if fast:
        return AOMRidgeRegressor(
            selection=selection,
            operator_bank="compact",
            alpha=1.0,  # bypass alphas CV
            center=True,
            random_state=seed,
        )
    return AOMRidgeRegressor(
        selection=selection,
        operator_bank="compact",
        alphas="auto",
        cv=3,
        center=True,
        random_state=seed,
    )


def make_nicon_stack(seed: int = 0) -> Optional[BaseEstimator]:
    """Build NICON-V2 Stack-Ridge-PLS-V1c (parallel session). Returns
    `None` if the package can't be imported. CNN training in `v1c` is slow
    (~200 epochs) — use sparingly.
    """
    try:
        from nicon_v2.models.stacking import StackedRegressor, StackingConfig
    except ImportError:
        return None
    return StackedRegressor(
        cfg=StackingConfig(
            base_learners=("ridge", "pls", "v1c"),
            n_folds=3,
            seed=seed,
            cnn_train_epochs=80,  # cap to keep stack-of-stacks tractable
            cnn_train_patience=10,
        )
    )


def make_tabpfn(seed: int = 0, n_max: int = 5000) -> Optional[BaseEstimator]:
    """Build TabPFNAdapter. Returns None if tabpfn isn't installed."""
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return None
    return TabPFNAdapter(n_max=n_max, seed=seed, device="cpu")


def collect_hetero_bases(
    seed: int,
    max_components: int,
    p: int,
    *,
    include_aom_ridge: bool = True,
    include_tabpfn: bool = True,
    include_nicon: bool = False,
) -> List[Tuple[str, BaseEstimator]]:
    """Assemble the heterogeneous base-estimator list for stacking."""
    from aompls.estimators import AOMPLSRegressor
    from .moe import AOMMoERegressor
    from .views import ViewBuilder
    from aompls.estimators import POPPLSRegressor

    bases: List[Tuple[str, BaseEstimator]] = [
        ("aom_pls", AOMPLSRegressor(
            n_components="auto", max_components=max_components,
            engine="simpls_covariance", selection="global",
            criterion="holdout", operator_bank="compact", random_state=seed,
        )),
        ("moe_preproc_soft", AOMMoERegressor(
            expert_layout="per_preproc", routing="soft",
            bank_name="compact", per_expert_components=min(10, max_components),
            n_oof_folds=3, random_state=seed,
        )),
        ("moe_view_soft", AOMMoERegressor(
            expert_layout="per_view", routing="soft",
            K=3, per_expert_components=min(10, max_components),
            n_oof_folds=3, random_state=seed,
        )),
    ]
    # lazy-V2-AOM-combined (uses ViewBuilder bank)
    bank = ViewBuilder.combined(
        bank_name="compact", K=3, strategy="equal_width", include_global=True,
    ).build(p=p)
    bases.append(("lazy_v2_aom", AOMPLSRegressor(
        n_components="auto", max_components=max_components,
        engine="simpls_covariance", selection="global",
        criterion="holdout", operator_bank=bank, random_state=seed,
    )))

    if include_aom_ridge:
        ar = make_aom_ridge(seed=seed)
        if ar is not None:
            bases.append(("aom_ridge", ar))
    if include_tabpfn:
        tp = make_tabpfn(seed=seed)
        if tp is not None:
            bases.append(("tabpfn", tp))
    if include_nicon:
        nic = make_nicon_stack(seed=seed)
        if nic is not None:
            bases.append(("nicon_stack", nic))
    return bases

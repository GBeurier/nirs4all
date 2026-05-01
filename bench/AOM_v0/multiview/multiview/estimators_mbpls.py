"""sklearn-compatible regressor for block-sparse AOM-MBPLS.

The estimator wraps `fit_block_sparse_aom` from `selection_mbpls.py`. The
public API mirrors AOMPLSRegressor / POPPLSRegressor from
`bench/AOM_v0/aompls/estimators.py` so existing benchmark scaffolding can
treat it identically.

Block-sparse AOM-MBPLS = per-LV winning-block selection with block-sparse
deflation. AOM-style locality is achieved by composing operators with block
masks via `ViewBuilder` upfront.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from aompls.banks import bank_by_name, fit_bank
from aompls.metrics import r2
from aompls.operators import ComposedOperator, IdentityOperator, LinearSpectralOperator
from aompls.scorers import CriterionConfig

from .selection_mbpls import (
    MBPLSResult,
    derive_block_metadata,
    fit_block_sparse_aom,
)
from .views import BlockMaskOperator, ViewBuilder


class BlockSparseAOMMBPLSRegressor(BaseEstimator, RegressorMixin):
    """Block-sparse AOM-MBPLS regressor.

    Parameters mirror AOMPLSRegressor where possible. Block geometry is
    set up via `K`, `strategy`, and an optional preproc bank that gets
    composed with each block mask.

    Parameters
    ----------
    n_components : int or "auto"
        Maximum number of components. With "auto", selection runs to
        `max_components` and the estimator returns the full sequence; the
        criterion's auto-prefix logic is not yet wired (pragmatic shrinkage
        is applied via the inner pinv regularisation).
    max_components : int
        Upper bound on `n_components` when `n_components="auto"`.
    K : int
        Number of equal-width blocks (>= 2).
    strategy : str
        Wavelength-block strategy. `"equal_width"` for Phase 2.
    preproc_bank_name : str or None
        Optional preproc bank to compose with each block mask. `None` means
        block-only V1 (only block masks). `"compact"` etc. enables V2.
    include_global : bool
        Unused for block-sparse: identity / bare preproc are skipped at
        bank construction time. Kept for API parity with ViewBuilder.
    criterion : str
        `"holdout"` (default Phase 2) or `"cv"`. SPXY-CV requires
        `cv_splitter`.
    cv : int
        Number of CV folds (when `criterion='cv'`).
    cv_splitter : object or None
        Optional sklearn-compatible splitter (e.g. SPXYFold).
    random_state : int
        Seed for holdout / KFold randomness.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: Union[int, str] = "auto",
        max_components: int = 15,
        K: int = 3,
        strategy: str = "equal_width",
        preproc_bank_name: Optional[str] = None,
        include_global: bool = False,
        criterion: str = "holdout",
        cv: int = 3,
        cv_splitter=None,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_components = max_components
        self.K = K
        self.strategy = strategy
        self.preproc_bank_name = preproc_bank_name
        self.include_global = include_global
        self.criterion = criterion
        self.cv = cv
        self.cv_splitter = cv_splitter
        self.random_state = random_state

    def _build_bank_and_metadata(self, p: int):
        if self.preproc_bank_name is None:
            builder = ViewBuilder.blocks_only(K=self.K, strategy=self.strategy)
            bank = builder.build(p=p)
        else:
            builder = ViewBuilder.combined(
                bank_name=self.preproc_bank_name,
                K=self.K,
                strategy=self.strategy,
                include_global=False,
            )
            bank = builder.build(p=p)
        # Derive blocks from the bank's BlockMaskOperator entries (deterministic
        # from the ViewBuilder strategy).
        block_masks_in_bank = [
            op for op in bank if isinstance(op, BlockMaskOperator)
        ]
        # Sort by start to keep block indices stable.
        block_masks_in_bank.sort(key=lambda m: m.start)
        blocks = [(m.start, m.end) for m in block_masks_in_bank]
        operators_kept, op_to_block, block_masks = derive_block_metadata(
            bank=bank, blocks=blocks, p=p
        )
        return operators_kept, op_to_block, block_masks, blocks

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlockSparseAOMMBPLSRegressor":
        start = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n, p = X.shape
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean(axis=0)
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        operators, op_to_block, block_masks, blocks = self._build_bank_and_metadata(p)
        if not operators:
            raise ValueError(
                "block-sparse bank is empty after filtering identity/bare-preproc; "
                "check K, strategy, preproc_bank_name parameters."
            )
        # Fit operators on training X (most are no-ops; only sets `p`).
        for op in operators:
            op.fit(Xc)

        max_components = min(
            self.max_components,
            max(1, n - 1),
            p,
        )
        n_request = max_components if self.n_components == "auto" else min(
            int(self.n_components), max_components
        )

        criterion = CriterionConfig(
            kind=self.criterion,
            cv=self.cv,
            random_state=self.random_state,
            task="regression",
            cv_splitter=self.cv_splitter,
        )

        result = fit_block_sparse_aom(
            Xc=Xc, yc=yc,
            operators=operators,
            op_to_block=op_to_block,
            block_masks=block_masks,
            n_components_max=n_request,
            criterion=criterion,
        )

        coef = result.coef()
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        self.coef_ = coef
        self.intercept_ = self.y_mean_ - self.x_mean_ @ coef
        self.n_components_ = result.n_components
        self.x_weights_ = result.Z.copy()
        self.x_loadings_ = result.P.copy()
        self.y_loadings_ = result.Q.copy()
        self.x_scores_ = result.T.copy()
        self.block_winners_ = list(result.block_winners)
        self.blocks_ = list(blocks)
        self.bank_ = list(operators)
        self.op_to_block_ = list(op_to_block)
        self.fit_time_s_ = float(time.perf_counter() - start)
        self._result = result
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=float)
        pred = (X - self.x_mean_) @ self.coef_ + self.y_mean_
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return r2(np.asarray(y).ravel(), np.asarray(pred).ravel())

    def get_block_winners(self) -> list:
        if not hasattr(self, "block_winners_"):
            raise RuntimeError("Estimator not fitted")
        return list(self.block_winners_)

    def get_selected_operators(self) -> list:
        if not hasattr(self, "bank_"):
            raise RuntimeError("Estimator not fitted")
        return [self.bank_[idx].name for _, idx in self.block_winners_]

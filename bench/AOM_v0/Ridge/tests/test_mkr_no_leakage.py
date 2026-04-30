"""No-leakage tests for AOMMultiKernelRidge.

Use the same SpyOperator pattern as test_ridge_cv_no_leakage.py: wrap an
identity operator that records the row signatures of every fit / apply_cov
call. The kernelizer is fitted on outer training data only, so test rows
must never appear in the spy's records.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator

from aomridge.mkr_estimator import AOMMultiKernelRidge


class SpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that records the inputs it observes."""

    def __init__(self, p=None) -> None:
        super().__init__(name="spy_identity", p=p)
        self.fit_row_signatures: list[float] = []
        self.apply_cov_col_signatures: list[float] = []

    def fit(self, X=None, y=None):
        if X is not None:
            X = np.asarray(X)
            self.fit_row_signatures.extend(np.sum(X, axis=1).tolist())
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        # S has shape (p, n_signatures); column sums encode the rows that
        # produced them. Record them.
        self.apply_cov_col_signatures.extend(np.sum(S, axis=0).tolist())
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def _row_signatures(X: np.ndarray) -> set[float]:
    return {float(s) for s in np.sum(X, axis=1)}


@pytest.mark.parametrize("strategy", ["uniform", "kta"])
def test_kernelizer_fitted_only_on_train(strategy):
    """The mkR kernelizer is fitted on outer training data only. Test rows
    must never appear in the spy's fit_row_signatures.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, p = 30, 12, 16
    X_train = rng.normal(size=(n_train, p))
    X_test = rng.normal(size=(n_test, p))
    y_train = rng.normal(size=n_train)
    spy = SpyOperator(p=p)

    # Construct a custom bank: identity + spy.
    custom_bank = [IdentityOperator(p=p), spy]
    est = AOMMultiKernelRidge(
        operator_bank=custom_bank,
        weight_strategy=strategy,
        alpha_grid_size=5,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    train_rows = _row_signatures(X_train)
    test_rows = _row_signatures(X_test)

    # Spy.fit must have observed only training rows.
    fit_seen = set(spy.fit_row_signatures)
    assert fit_seen.issubset(train_rows), (
        f"spy fit observed rows not in training: {fit_seen - train_rows}"
    )

    # apply_cov is called on Xc.T (column sums encode original X rows).
    # We can verify that no test-row signature appeared yet.
    apply_cov_observed_post_fit = set(spy.apply_cov_col_signatures)
    assert not (apply_cov_observed_post_fit & test_rows), (
        "spy apply_cov observed test rows during fit; suspected leakage"
    )

    # Now predict on test data; this triggers transform on the spy via the
    # cross kernel. Verify nothing changes about the FIT records.
    fit_seen_after_predict = set(spy.fit_row_signatures)
    _ = est.predict(X_test)
    fit_seen_after_predict_2 = set(spy.fit_row_signatures)
    assert fit_seen_after_predict == fit_seen_after_predict_2, (
        "fit records must not change during predict"
    )


def test_train_kernel_uses_only_train_data():
    """Training kernels stored at fit time depend only on training data."""
    rng = np.random.default_rng(1)
    n_train, p = 40, 20
    X_train = rng.normal(size=(n_train, p))
    y_train = rng.normal(size=n_train)
    spy = SpyOperator(p=p)
    custom_bank = [IdentityOperator(p=p), spy]
    est = AOMMultiKernelRidge(
        operator_bank=custom_bank,
        weight_strategy="uniform",
        alpha_grid_size=5,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    train_rows = _row_signatures(X_train)
    fit_seen = set(spy.fit_row_signatures)
    assert fit_seen.issubset(train_rows)

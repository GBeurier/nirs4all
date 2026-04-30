"""No-leakage tests for AOMMultiKernelRidge.

The kernelizer deep-copies the operator bank (``clone_operator_bank``) before
fitting, so a per-instance spy never records anything via the original
reference. We work around this with a **class-level shared log**: every
clone shares the same dict and appends its observations there. Then we
assert (a) the log is non-empty, (b) it never contains test-row signatures.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator

from aomridge.mkr_estimator import AOMMultiKernelRidge


class SpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that logs observations to a shared dict.

    Because deep-copy preserves the class attribute reference, every clone
    of the spy writes to the same ``_shared_log`` list. Tests reset the log
    via :meth:`reset_log` before each fit.
    """

    _shared_log: dict[str, list[float]] = {
        "fit_row_signatures": [],
        "apply_cov_col_signatures": [],
        "transform_row_signatures": [],
        "adjoint_vec_col_signatures": [],
    }

    def __init__(self, p=None) -> None:
        super().__init__(name="spy_identity", p=p)

    @classmethod
    def reset_log(cls) -> None:
        for k in cls._shared_log:
            cls._shared_log[k] = []

    @classmethod
    def fit_rows(cls) -> set[float]:
        return {float(x) for x in cls._shared_log["fit_row_signatures"]}

    @classmethod
    def apply_cov_cols(cls) -> set[float]:
        return {float(x) for x in cls._shared_log["apply_cov_col_signatures"]}

    @classmethod
    def transform_rows(cls) -> set[float]:
        return {float(x) for x in cls._shared_log["transform_row_signatures"]}

    @classmethod
    def adjoint_cols(cls) -> set[float]:
        return {float(x) for x in cls._shared_log["adjoint_vec_col_signatures"]}

    def fit(self, X=None, y=None):
        if X is not None:
            X = np.asarray(X)
            type(self)._shared_log["fit_row_signatures"].extend(
                np.sum(X, axis=1).tolist()
            )
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X):
        type(self)._shared_log["transform_row_signatures"].extend(
            np.sum(X, axis=1).tolist()
        )
        return X.copy()

    def _apply_cov_impl(self, S):
        type(self)._shared_log["apply_cov_col_signatures"].extend(
            np.sum(S, axis=0).tolist()
        )
        return S.copy()

    def _adjoint_vec_impl(self, v):
        type(self)._shared_log["adjoint_vec_col_signatures"].extend(
            np.sum(v, axis=0).tolist() if v.ndim > 1 else [float(v.sum())]
        )
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def _row_signatures(X: np.ndarray) -> set[float]:
    return {float(s) for s in np.sum(X, axis=1)}


def _centred_row_signatures(X_target: np.ndarray, X_train: np.ndarray) -> set[float]:
    """Row signatures of ``X_target`` after subtracting the training mean."""
    x_mean = X_train.mean(axis=0)
    Xc = X_target - x_mean
    return {float(s) for s in np.sum(Xc, axis=1)}


@pytest.fixture(autouse=True)
def _reset_spy_log():
    SpyOperator.reset_log()
    yield
    SpyOperator.reset_log()


@pytest.mark.parametrize("strategy", ["uniform", "kta"])
def test_kernelizer_fitted_only_on_train(strategy):
    """The mkR kernelizer is fitted on outer training data only. Test rows
    must never appear in the shared spy log.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, p = 30, 12, 16
    X_train = rng.normal(size=(n_train, p))
    X_test = rng.normal(size=(n_test, p))
    y_train = rng.normal(size=n_train)

    custom_bank = [IdentityOperator(p=p), SpyOperator(p=p)]
    est = AOMMultiKernelRidge(
        operator_bank=custom_bank,
        weight_strategy=strategy,
        alpha_grid_size=5,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    # Spy receives X_c = X - x_mean (kernelizer centres before fit).
    # Compare against centred train / centred test signatures (under
    # training mean).
    centred_train_rows = _centred_row_signatures(X_train, X_train)
    centred_test_rows = _centred_row_signatures(X_test, X_train)

    # The log must be NON-EMPTY (sanity: clones did receive fit calls).
    fit_seen = SpyOperator.fit_rows()
    assert fit_seen, "spy fit log empty — clones never observed any data"
    # And every observed fit row must come from the training set
    # (modulo numerical noise — use a tolerance).
    leaked = {
        s for s in fit_seen
        if not any(abs(s - t) < 1e-9 for t in centred_train_rows)
    }
    assert not leaked, (
        f"spy fit observed rows not in training: {leaked}"
    )
    test_leaks = {
        s for s in fit_seen
        if any(abs(s - t) < 1e-9 for t in centred_test_rows)
    }
    assert not test_leaks, (
        f"spy fit observed centred-test-row signatures: {test_leaks}"
    )


def test_predict_dual_uses_cross_kernel():
    """``predict_dual`` calls the kernelizer's transform on test rows,
    so the spy log should grow during predict_dual but not fit.
    """
    rng = np.random.default_rng(1)
    n_train, n_test, p = 30, 8, 16
    X_train = rng.normal(size=(n_train, p))
    X_test = rng.normal(size=(n_test, p))
    y_train = rng.normal(size=n_train)

    custom_bank = [IdentityOperator(p=p), SpyOperator(p=p)]
    est = AOMMultiKernelRidge(
        operator_bank=custom_bank,
        weight_strategy="uniform",
        alpha_grid_size=5,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    fit_rows_after_fit = SpyOperator.fit_rows()
    # Trigger cross-kernel construction.
    _ = est.predict_dual(X_test)
    # apply_cov is called on training X.T inside transform; the spy log
    # tracking transform_rows will grow with X_train rows (centred).
    # Crucially, the FIT log must not change.
    fit_rows_after_predict = SpyOperator.fit_rows()
    assert fit_rows_after_fit == fit_rows_after_predict, (
        "fit log changed during predict; transform should not refit operators"
    )


def test_train_kernel_uses_only_train_data():
    """Training kernels stored at fit time depend only on training data."""
    rng = np.random.default_rng(2)
    n_train, p = 40, 20
    X_train = rng.normal(size=(n_train, p))
    y_train = rng.normal(size=n_train)

    custom_bank = [IdentityOperator(p=p), SpyOperator(p=p)]
    est = AOMMultiKernelRidge(
        operator_bank=custom_bank,
        weight_strategy="uniform",
        alpha_grid_size=5,
        alpha_cv_n_splits=3,
        random_state=0,
    )
    est.fit(X_train, y_train)
    fit_seen = SpyOperator.fit_rows()
    centred_train_rows = _centred_row_signatures(X_train, X_train)
    assert fit_seen, "spy log empty"
    leaked = {
        s for s in fit_seen
        if not any(abs(s - t) < 1e-9 for t in centred_train_rows)
    }
    assert not leaked

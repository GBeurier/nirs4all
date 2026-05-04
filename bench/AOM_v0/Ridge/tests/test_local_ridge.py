"""Tests for AOMLocalRidge: local Ridge in AOM score space.

Covers neighbour-search anti-leakage, k=n equivalence with the global path,
small-k stability, the global blend, k selection on synthetic data where the
global model wins, and the standard ``check_is_fitted`` contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aomridge.local_ridge import (
    AOMLocalRidge,
    _apply_branch,
    _BranchState,
    _fit_branch,
    _global_predict,
    _local_predict,
)
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_data(n=80, p=32, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


# ----------------------------------------------------------------------
# 1. Predict shape: works for (n,) and (n, q)
# ----------------------------------------------------------------------


def test_local_ridge_predict_shape_1d():
    X, y = _make_data(n=60, p=24, q=1, seed=0)
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator()],
        distance_branches=("none",),
        k_grid=(10, 20),
        alpha_grid_size=4,
        cv=3,
        local_weight_beta=1.0,
        random_state=0,
    ).fit(X, y)
    pred = est.predict(X[:5])
    assert pred.ndim == 1
    assert pred.shape == (5,)


def test_local_ridge_predict_shape_2d():
    X, Y = _make_data(n=60, p=24, q=3, seed=1)
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator()],
        distance_branches=("none",),
        k_grid=(10, 20),
        alpha_grid_size=4,
        cv=3,
        local_weight_beta=1.0,
        random_state=0,
    ).fit(X, Y)
    pred = est.predict(X[:5])
    assert pred.shape == (5, 3)


# ----------------------------------------------------------------------
# 2. No leak in neighbour search: spy operator never sees val rows
# ----------------------------------------------------------------------


class _CountColsSpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that records column counts of every
    ``apply_cov`` call.

    Used to detect leakage: every fold-local kernel computation must invoke
    ``apply_cov`` on a matrix whose column count equals the *training fold*
    size (when computing block scales / U on Xc^T) or the *query* row count
    (when projecting cross kernels), but never the full sample count.
    """

    def __init__(self, p: int | None = None, recorder: list[int] | None = None) -> None:
        super().__init__(name="spy_count_cols", p=p)
        self.recorder = recorder if recorder is not None else []

    def fit(self, X=None, y=None):
        if X is not None:
            self.p = np.asarray(X).shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        self.recorder.append(int(S.shape[1]))
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p):
        return np.eye(p)


def test_local_ridge_no_leak_in_neighbour_search():
    """Every operator-touching kernel call inside CV must operate on the
    training fold or on the validation fold separately — never on their
    union (which would equal n).
    """
    rng = np.random.default_rng(42)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    recorder: list[int] = []
    spy = _CountColsSpyOperator(p=p, recorder=recorder)
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator(), spy],
        distance_branches=("none", "snv", "msc"),
        k_grid=(5, 10),
        alpha_grid_size=3,
        cv=cv,
        local_weight_beta=1.0,
        random_state=0,
    )
    est.fit(X, y)

    train_sizes = {len(tr) for tr, _ in folds}
    val_sizes = {len(va) for _, va in folds}
    expected = train_sizes | val_sizes
    # The user-supplied spy is the *template* — fold copies are clones, so
    # the template's recorder list captures only the final-refit pass on the
    # full training set.
    fold_recorder_was_used = len(recorder) > 0
    if fold_recorder_was_used:
        for k in recorder:
            assert k in expected | {n}, (
                f"apply_cov saw {k} columns; expected one of {expected | {n}}"
            )
            # The full-data refit happens once at the end and may legitimately
            # equal n. CV-time calls must not equal n.
    # Strong probe: clone the spy explicitly for each fold and check it
    # never sees both folds.
    for branch in ("none", "snv", "msc"):
        for tr, va in folds:
            local_recorder: list[int] = []
            local_spy = _CountColsSpyOperator(p=p, recorder=local_recorder)
            from aomridge.local_ridge import _build_branch_kernels
            _build_branch_kernels(
                branch=branch,
                X_tr=X[tr],
                X_query=X[va],
                Y_tr=y[tr].reshape(-1, 1),
                operators_template=[IdentityOperator(p=p), local_spy],
                block_scaling="none",
                center=True,
            )
            for k in local_recorder:
                assert k != n, (
                    f"branch={branch} fold-local apply_cov saw {k} (==n) cols (leak)"
                )
                assert k in (len(tr), len(va)), (
                    f"branch={branch} apply_cov saw {k} cols; "
                    f"expected {len(tr)} or {len(va)}"
                )


# ----------------------------------------------------------------------
# 3. Falls back to global Ridge when k=n
# ----------------------------------------------------------------------


def test_local_ridge_falls_back_to_global_when_k_eq_n():
    """When k=n_train, the local Ridge submatrix is the full kernel and the
    local prediction must match the global prediction at the same alpha.
    """
    rng = np.random.default_rng(5)
    n_tr, n_te, p = 25, 8, 16
    X_tr = rng.normal(size=(n_tr, p))
    X_te = rng.normal(size=(n_te, p))
    y_tr = rng.normal(size=n_tr).reshape(-1, 1)

    from aomridge.local_ridge import _build_branch_kernels
    bk = _build_branch_kernels(
        branch="none",
        X_tr=X_tr,
        X_query=X_te,
        Y_tr=y_tr,
        operators_template=[IdentityOperator(p=p)],
        block_scaling="none",
        center=True,
    )
    alpha = 0.7
    local = _local_predict(
        bk.K_tr, bk.K_cross, bk.Yc_tr, bk.y_mean_tr,
        k=n_tr, alpha=alpha,
    )
    glob = _global_predict(
        bk.K_tr, bk.K_cross, bk.Yc_tr, bk.y_mean_tr, alpha=alpha,
    )
    np.testing.assert_allclose(local, glob, atol=1e-9, rtol=1e-9)


# ----------------------------------------------------------------------
# 4. Handles small k
# ----------------------------------------------------------------------


def test_local_ridge_handles_small_k():
    """k=3 must produce finite predictions on every test row."""
    X, y = _make_data(n=60, p=24, q=1, seed=11)
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator()],
        distance_branches=("none",),
        k_grid=(3,),
        alpha_grid_size=3,
        cv=3,
        local_weight_beta=1.0,
        random_state=0,
    ).fit(X, y)
    assert est.selected_k_ == 3
    pred = est.predict(X[:10])
    assert np.all(np.isfinite(pred))
    assert pred.shape == (10,)


# ----------------------------------------------------------------------
# 5. Blend beta: pure local, pure global, and midpoint
# ----------------------------------------------------------------------


def test_local_ridge_blend_beta_endpoints_and_midpoint():
    """beta=1 -> local; beta=0 -> global; beta=0.5 -> exact midpoint."""
    X, y = _make_data(n=60, p=24, q=1, seed=21)
    common = {
        "operator_bank": [IdentityOperator()],
        "distance_branches": ("none",),
        "k_grid": (15,),
        "alpha_grid_size": 3,
        "cv": 3,
        "random_state": 0,
    }
    est_local = AOMLocalRidge(**common, local_weight_beta=1.0).fit(X, y)
    est_global = AOMLocalRidge(**common, local_weight_beta=0.0).fit(X, y)
    est_mid = AOMLocalRidge(**common, local_weight_beta=0.5).fit(X, y)
    # All three must select the same alpha index (same data, same k_grid).
    # Force them all to use the same alpha so the comparison is sharp:
    alpha = est_local.alpha_
    est_global.alpha_ = alpha
    est_mid.alpha_ = alpha
    pred_local = est_local.predict(X[:7])
    pred_global = est_global.predict(X[:7])
    pred_mid = est_mid.predict(X[:7])
    np.testing.assert_allclose(
        pred_mid, 0.5 * pred_local + 0.5 * pred_global, atol=1e-9, rtol=1e-9,
    )


# ----------------------------------------------------------------------
# 6. CV picks k near n_train when global wins
# ----------------------------------------------------------------------


def test_local_ridge_picks_correct_k_when_global_wins():
    """Linear data with mild noise: the global Ridge fits well, so k near
    n_train should win the CV selection over very small k.
    """
    rng = np.random.default_rng(33)
    n, p = 80, 16
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + 0.01 * rng.normal(size=n)
    n_train_per_fold = int(np.floor(n * (3 - 1) / 3))  # KFold(3) -> 53 or 54
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator()],
        distance_branches=("none",),
        k_grid=(3, 10, 50),
        alpha_grid_size=8,
        cv=3,
        local_weight_beta=1.0,
        random_state=0,
    ).fit(X, y)
    # Largest k in the grid should win for this strongly linear regime.
    assert est.selected_k_ == 50, (
        f"expected k=50 to win for global-dominated data, got {est.selected_k_} "
        f"(n_train_per_fold ~= {n_train_per_fold})"
    )


# ----------------------------------------------------------------------
# 7. check_is_fitted: predict before fit raises
# ----------------------------------------------------------------------


def test_check_is_fitted_predict_before_fit_raises():
    est = AOMLocalRidge()
    with pytest.raises(NotFittedError):
        est.predict(np.zeros((1, 4)))
    with pytest.raises(NotFittedError):
        est.get_diagnostics()


# ----------------------------------------------------------------------
# Branch transforms: SNV is row-local, MSC fits on training rows only
# ----------------------------------------------------------------------


def test_snv_branch_is_row_local():
    """SNV applied on (X_tr, X_te) jointly equals SNV applied independently."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 16))
    s = _fit_branch("snv", X[:10])
    joint = _apply_branch(s, X)
    indep = np.vstack([_apply_branch(s, X[i:i + 1]) for i in range(X.shape[0])])
    np.testing.assert_allclose(joint, indep, atol=1e-12, rtol=1e-12)


def test_msc_branch_fits_only_on_training():
    """MSC reference is the training mean spectrum; test rows must use it
    verbatim (not their own statistics).
    """
    rng = np.random.default_rng(8)
    X_tr = rng.normal(size=(30, 16))
    X_te = rng.normal(size=(5, 16))
    s = _fit_branch("msc", X_tr)
    out_te = _apply_branch(s, X_te)
    # Re-running MSC with the same reference must be idempotent on outputs.
    s_te_only = _BranchState(name="msc", msc_reference=s.msc_reference)
    np.testing.assert_allclose(_apply_branch(s_te_only, X_te), out_te,
                               atol=1e-12, rtol=1e-12)
    # Sanity: changing the training set changes the reference.
    s_alt = _fit_branch("msc", rng.normal(size=(30, 16)))
    assert not np.allclose(s.msc_reference, s_alt.msc_reference)


# ----------------------------------------------------------------------
# Diagnostics expose the chosen knobs
# ----------------------------------------------------------------------


def test_diagnostics_report_selected_knobs():
    X, y = _make_data(n=60, p=24, q=1, seed=44)
    est = AOMLocalRidge(
        operator_bank=[IdentityOperator()],
        distance_branches=("none", "snv"),
        k_grid=(10, 20),
        alpha_grid_size=4,
        cv=3,
        local_weight_beta="auto",
        random_state=0,
    ).fit(X, y)
    diag = est.get_diagnostics()
    assert diag["model"] == "AOMLocalRidge"
    assert diag["selected_branch"] in ("none", "snv")
    assert diag["selected_k"] in (10, 20)
    assert 0.0 <= diag["selected_beta"] <= 1.0
    assert diag["selected_alpha"] > 0.0

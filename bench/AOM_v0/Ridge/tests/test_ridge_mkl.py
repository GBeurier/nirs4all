"""Phase A2 tests: MKL-light supervised block weights for AOM-Ridge."""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import (
    ExplicitMatrixOperator,
    IdentityOperator,
    LinearSpectralOperator,
    SavitzkyGolayOperator,
)
from aomridge.estimators import AOMRidgeRegressor
from aomridge.kernels import fit_operator_bank
from aomridge.mkl import (
    kta_score,
    learn_block_weights,
    mkl_kernel_train,
)
from aomridge.selection import cv_score_alphas_mkl
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _make_data(n=80, p=48, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


# ----------------------------------------------------------------------
# kta_score sanity
# ----------------------------------------------------------------------


def test_kta_score_zero_norm_returns_zero():
    K = np.zeros((5, 5))
    YYt = np.eye(5)
    assert kta_score(K, YYt) == 0.0
    assert kta_score(np.eye(5), np.zeros((5, 5))) == 0.0


def test_kta_score_self_alignment_is_one():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 3))
    K = A @ A.T
    K = 0.5 * (K + K.T)
    val = kta_score(K, K)
    assert abs(val - 1.0) < 1e-10


# ----------------------------------------------------------------------
# learn_block_weights: simplex constraint
# ----------------------------------------------------------------------


def test_mkl_weights_on_simplex():
    rng = np.random.default_rng(1)
    n, p = 30, 20
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, 1))
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=p),
    ]
    fit_operator_bank(bank, Xc)
    scales = np.ones(len(bank))
    w = learn_block_weights(bank, Xc, Yc, scales, top_k=3, mode="alignment")
    assert w.shape == (len(bank),)
    assert np.all(w >= 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_mkl_weights_top_k_caps_active_count():
    rng = np.random.default_rng(2)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    Y = X[:, :3].sum(axis=1, keepdims=True) + 0.05 * rng.normal(size=(n, 1))
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=2, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=3, deriv=0, p=p),
        SavitzkyGolayOperator(window_length=15, polyorder=3, deriv=1, p=p),
    ]
    fit_operator_bank(bank, Xc)
    scales = np.ones(len(bank))
    w = learn_block_weights(bank, Xc, Yc, scales, top_k=2, mode="alignment")
    assert int(np.sum(w > 0)) <= 2
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_mkl_weights_unknown_mode_raises():
    bank = [IdentityOperator(p=4)]
    fit_operator_bank(bank, np.zeros((3, 4)))
    with pytest.raises(ValueError):
        learn_block_weights(bank, np.zeros((3, 4)), np.zeros((3, 1)),
                            np.array([1.0]), top_k=1, mode="bogus")


# ----------------------------------------------------------------------
# Combined kernel equivalence: linear-in-weight sum of explicit block kernels
# ----------------------------------------------------------------------


def test_mkl_kernel_equals_explicit():
    rng = np.random.default_rng(3)
    n, p = 25, 18
    X = rng.normal(size=(n, p))
    Xc = X - X.mean(axis=0)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=2, p=p),
    ]
    fit_operator_bank(bank, Xc)
    scales = np.array([1.0, 0.5, 2.0])
    weights = np.array([0.5, 0.3, 0.2])
    K_mkl, _ = mkl_kernel_train(Xc, bank, weights, scales=scales)
    # Reference: sum_b w_b * (s_b X A_b^T)(s_b X A_b^T)^T explicitly.
    K_ref = np.zeros((n, n), dtype=float)
    for op, w, s in zip(bank, weights, scales, strict=False):
        Z_b = float(s) * op.transform(Xc)            # (n, p)
        K_b = Z_b @ Z_b.T
        K_ref += float(w) * K_b
    K_ref = 0.5 * (K_ref + K_ref.T)
    np.testing.assert_allclose(K_mkl, K_ref, atol=1e-9, rtol=1e-9)


# ----------------------------------------------------------------------
# Identity-only collapse: MKL must reproduce vanilla Ridge
# ----------------------------------------------------------------------


def test_mkl_identity_only_matches_ridge():
    """With a single identity operator and weight 1, the dual MKL Ridge
    must match sklearn Ridge to floating-point precision."""
    X, y = _make_data()
    alpha = 0.42
    est = AOMRidgeRegressor(
        selection="mkl",
        operator_bank=[IdentityOperator()],
        alpha=alpha,
        block_scaling="none",
        cv=3,
        random_state=0,
        mkl_top_k=1,
    )
    est.fit(X, y)
    sk = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(est.coef_, sk.coef_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.intercept_, sk.intercept_, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(est.predict(X), sk.predict(X), atol=1e-8, rtol=1e-8)
    # Weight on identity must equal 1 (only block).
    assert abs(float(est.mkl_weights_[0]) - 1.0) < 1e-12


# ----------------------------------------------------------------------
# No-leak spy: validation rows must never enter weight learning
# ----------------------------------------------------------------------


class CountColsSpy(LinearSpectralOperator):
    """Identity-equivalent operator that records the column count of every
    ``apply_cov`` call.

    During MKL CV ``apply_cov`` is called twice per block per fold:
      (a) ``Xc^T`` (n_train columns) inside ``compute_block_scales_from_xt``,
      (b) ``Xc^T`` again inside ``learn_block_weights`` when the per-block
          kernel is built, and a third time inside ``mkl_kernel_train``.

    None of these calls may use the *full* sample count (which would imply
    that validation rows leaked into the weight learning).
    """

    def __init__(self, p=None) -> None:
        super().__init__(name="count_cols_spy", p=p)
        self.seen_col_counts: list[int] = []

    def fit(self, X=None, y=None):
        if X is not None:
            self.p = np.asarray(X).shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        self.seen_col_counts.append(int(S.shape[1]))
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def test_mkl_no_leak_in_cv():
    rng = np.random.default_rng(4)
    n, p = 30, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    spy = CountColsSpy(p=p)
    bank = [IdentityOperator(), spy]
    cv_score_alphas_mkl(
        X, y.reshape(-1, 1), bank, np.array([1.0]), cv,
        block_scaling="none", center=True, mkl_top_k=2,
    )
    # The user-supplied template must NEVER be fitted/applied directly:
    # every fold uses a fresh clone.
    assert spy.seen_col_counts == []


def test_mkl_no_leak_clones_observe_only_training_rows():
    """Patch the spy clones (via class-level list) to assert column counts
    inside every fold equal the training-fold size, never the full n.
    """
    rng = np.random.default_rng(5)
    n, p = 30, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))
    seen_col_counts: list[int] = []

    class SharedSpy(CountColsSpy):
        def _apply_cov_impl(self, S):
            seen_col_counts.append(int(S.shape[1]))
            return S.copy()

    bank = [IdentityOperator(), SharedSpy(p=p)]
    cv_score_alphas_mkl(
        X, y.reshape(-1, 1), bank, np.array([0.5]), cv,
        block_scaling="rms", center=True, mkl_top_k=2,
    )
    train_sizes = {len(tr) for tr, _ in folds}
    assert seen_col_counts, "spy clone was never invoked"
    for k in seen_col_counts:
        assert k != n, (
            f"MKL CV apply_cov saw {k} columns (full dataset = {n}); leak"
        )
        # Each call uses Xc^T (n_train cols) — no other shape exists in the
        # CV path. (In contrast Y has q=1 columns; but the spy only logs
        # apply_cov on Xc^T because identity / spy never receive Yc here.)
        assert k in train_sizes


# ----------------------------------------------------------------------
# Synthetic signal-vs-noise: an informative operator must get nonzero weight
# ----------------------------------------------------------------------


def test_mkl_picks_relevant_block_on_synthetic():
    """Build a synthetic problem where one operator is *clearly* informative
    (it returns the data unchanged) and a competing operator is decoupled
    from the target (it projects onto an orthogonal subspace). After weight
    learning, the informative operator must have a nonzero weight.

    The orthogonal projection is implemented as a constant matrix
    ``A = I - v v^T / ||v||^2`` where ``v`` is the response direction;
    its block kernel ``X A^T A X^T`` is a noise channel that has near-zero
    Frobenius alignment with ``Y Y^T``.
    """
    rng = np.random.default_rng(6)
    n, p = 60, 16
    # y depends on a sparse linear combination of features.
    coef = np.zeros(p)
    coef[0] = 1.5
    coef[3] = -0.8
    coef[7] = 0.5
    X = rng.normal(size=(n, p))
    y = X @ coef + 0.05 * rng.normal(size=n)
    Xc = X - X.mean(axis=0)
    Yc = (y - y.mean()).reshape(-1, 1)

    # Operator A: identity (informative).
    op_id = IdentityOperator(p=p)
    op_id.fit(Xc)

    # Operator B: orthogonal projection onto the *complement* of coef,
    # implemented as an explicit linear matrix.
    v = coef / (np.linalg.norm(coef) + 1e-12)
    P_orth = np.eye(p) - np.outer(v, v)
    op_orth = ExplicitMatrixOperator(P_orth, name="orth_to_target")
    op_orth.fit(Xc)

    bank = [op_id, op_orth]
    scales = np.ones(2)
    w = learn_block_weights(bank, Xc, Yc, scales, top_k=2, mode="alignment")
    # Identity must be nonzero. Ideally w[0] > w[1].
    assert w[0] > 0.0, f"identity got zero weight ({w})"
    assert w[0] > w[1], f"identity ({w[0]}) should beat orth ({w[1]}) on aligned target"


# ----------------------------------------------------------------------
# Estimator-level smoke test
# ----------------------------------------------------------------------


def test_mkl_estimator_runs_end_to_end():
    """The MKL estimator must fit, predict, and expose mkl_weights_ on the
    simplex with nonzero entries on at least one block.
    """
    X, y = _make_data(n=50, p=24, seed=10)
    est = AOMRidgeRegressor(
        selection="mkl",
        operator_bank="compact",
        block_scaling="none",
        cv=3,
        random_state=0,
        mkl_top_k=6,
    ).fit(X, y)
    pred = est.predict(X[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))
    w = est.mkl_weights_
    assert w is not None
    assert abs(float(np.sum(w)) - 1.0) < 1e-10
    assert np.all(w >= 0.0)
    # Diagnostics expose the MKL fields.
    diag = est.get_diagnostics()
    assert diag["selection"] == "mkl"
    assert diag["mkl_top_k"] == 6
    assert "mkl_weights" in diag
    assert "mkl_operator_weights" in diag


def test_mkl_estimator_rejects_bad_top_k():
    X, y = _make_data(n=20, p=8)
    with pytest.raises(ValueError):
        AOMRidgeRegressor(
            selection="mkl", operator_bank="compact", mkl_top_k=0,
        ).fit(X, y)


def test_mkl_estimator_rejects_unknown_mode():
    X, y = _make_data(n=20, p=8)
    with pytest.raises(ValueError):
        AOMRidgeRegressor(
            selection="mkl", operator_bank="compact", mkl_mode="bogus",
        ).fit(X, y)

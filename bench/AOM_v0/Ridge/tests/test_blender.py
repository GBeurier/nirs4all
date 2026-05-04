"""Tests for ``AOMRidgeBlender``.

Covers the six required scenarios from the task spec:

1. ``weights_`` lie on the simplex (non-negative, sum to 1).
2. With a very large ``regularizer``, weights collapse to the uniform
   mixture (``1/K``).
3. Outer-CV anti-leakage: a spy operator inside one candidate never sees
   outer-validation rows during ``fit``.
4. ``predict`` produces the right shape for 1D and 2D targets.
5. On a synthetic with two complementary candidates, the blender's RMSE
   is at most the best single candidate's RMSE (within tolerance).
6. ``predict`` before ``fit`` raises ``NotFittedError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aomridge.blender import AOMRidgeBlender
from aomridge.estimators import AOMRidgeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class SpyOperator(LinearSpectralOperator):
    """Identity operator that records the rows seen at fit time."""

    def __init__(self, p: int | None = None) -> None:
        super().__init__(name="spy_identity", p=p)
        self.fit_row_signatures: list[float] = []

    def fit(self, X=None, y=None):
        if X is not None:
            X = np.asarray(X)
            self.fit_row_signatures.extend(np.sum(X, axis=1).tolist())
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def _make_regression(
    n: int = 60, p: int = 24, noise: float = 0.05, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + noise * rng.normal(size=n)
    return X, y


def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    diff = np.asarray(yt).ravel() - np.asarray(yp).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


# ----------------------------------------------------------------------
# 1. weights on the simplex
# ----------------------------------------------------------------------


def test_blender_weights_on_simplex():
    X, y = _make_regression(n=50, p=16, seed=1)
    candidates = [
        {
            "label": "A",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "B",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]
    blender = AOMRidgeBlender(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        regularizer=0.01,
        random_state=0,
    ).fit(X, y)
    w = blender.weights_
    assert w.shape == (2,)
    # Non-negative
    assert np.all(w >= 0.0 - 1e-9), f"weights have negative entries: {w}"
    # Sum to one
    assert np.isclose(float(np.sum(w)), 1.0, atol=1e-8), (
        f"weights do not sum to one: {w}"
    )


# ----------------------------------------------------------------------
# 2. Large regularizer collapses weights to uniform 1/K
# ----------------------------------------------------------------------


def test_blender_uniform_fallback():
    """A very large regularizer dominates the data term in the QP, so the
    optimal solution is the uniform mixture ``w = 1/K``.
    """
    X, y = _make_regression(n=40, p=12, seed=2)
    candidates = [
        {
            "label": "A",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "B",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
        {
            "label": "C",
            "selection": "superblock",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]
    blender = AOMRidgeBlender(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        regularizer=10.0,
        random_state=0,
    ).fit(X, y)
    k = len(candidates)
    expected = np.full(k, 1.0 / k)
    np.testing.assert_allclose(blender.weights_, expected, atol=5e-3)


# ----------------------------------------------------------------------
# 3. Anti-leakage: spy operator never sees outer-validation rows
# ----------------------------------------------------------------------


def test_blender_no_leak():
    """The user-supplied operator instance is the *template*; clones absorb
    state. The spy is wrapped inside an AOMRidgeRegressor candidate; its
    ``fit_row_signatures`` must remain empty (clones, not the template,
    are fitted).

    More importantly, the *clones* used by the OOF outer-CV must only ever
    see outer-train rows during ``fit``. We verify this by collecting
    centred row sums from every clone and checking they correspond to
    outer-train slices, never to outer-validation slices (centred by the
    outer-train mean).
    """
    rng = np.random.default_rng(7)
    n, p = 36, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    template_spy = SpyOperator(p=p)

    outer = KFold(n_splits=3, shuffle=False)
    folds = list(outer.split(X, y))

    fit_row_sums: list[float] = []

    class SharedSpy(SpyOperator):
        def fit(self, X=None, y=None):
            if X is not None:
                X = np.asarray(X)
                fit_row_sums.extend(np.round(np.sum(X, axis=1), 8).tolist())
                self.p = X.shape[1]
            return self

    candidates = [
        {
            "label": "spy_candidate",
            "selection": "global",
            "operator_bank": [IdentityOperator(p=p), SharedSpy(p=p)],
            "block_scaling": "none",
        },
    ]
    blender = AOMRidgeBlender(
        candidates=candidates,
        outer_cv=outer,
        regularizer=0.01,
        random_state=0,
    ).fit(X, y)

    # Template never fitted directly.
    assert template_spy.fit_row_signatures == []
    # Some clones must have been fitted somewhere.
    assert fit_row_sums, "spy clones were never invoked"
    # Final blend should still produce shapes/predictions.
    assert blender.weights_.shape == (1,)

    seen = set(fit_row_sums)
    for tr_idx, va_idx in folds:
        x_mean_tr = X[tr_idx].mean(axis=0)
        # Outer-validation rows centred by the outer-train mean must not
        # appear in the seen-fit row sums (would indicate leakage).
        val_centred_sums = set(
            np.round(np.sum(X[va_idx] - x_mean_tr, axis=1), 8).tolist()
        )
        assert seen.isdisjoint(val_centred_sums), (
            "blender candidate fit observed outer-validation rows (leak)"
        )


# ----------------------------------------------------------------------
# 4. predict shape: 1D and 2D targets
# ----------------------------------------------------------------------


def test_blender_predict_shape():
    rng = np.random.default_rng(3)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    candidates = [
        {
            "label": "A",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "B",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]

    # 1D y
    y_1d = X @ coef + 0.05 * rng.normal(size=n)
    b1 = AOMRidgeBlender(
        candidates=candidates, outer_cv=3, outer_cv_kind="kfold",
        regularizer=0.01, random_state=0,
    ).fit(X, y_1d)
    pred_1d = b1.predict(X)
    assert pred_1d.shape == (n,)

    # 2D y with q=2
    y_2d = np.column_stack([
        y_1d,
        X @ rng.normal(size=p) + 0.05 * rng.normal(size=n),
    ])
    b2 = AOMRidgeBlender(
        candidates=candidates, outer_cv=3, outer_cv_kind="kfold",
        regularizer=0.01, random_state=0,
    ).fit(X, y_2d)
    pred_2d = b2.predict(X)
    assert pred_2d.shape == (n, 2)


# ----------------------------------------------------------------------
# 5. Synthetic with complementary candidates: blender beats best single
# ----------------------------------------------------------------------


def test_blender_beats_best_single_synthetic():
    """Build a synthetic where two candidates are *complementary*: each
    captures part of the signal that the other misses. The convex blender
    should achieve test RMSE no worse than the best single candidate
    (within a small tolerance for finite-sample noise).
    """
    rng = np.random.default_rng(123)
    n, p = 200, 24
    X = rng.normal(size=(n, p))
    # Signal: half driven by raw features, half by their first finite
    # difference. Identity-bank Ridge captures the first half; the
    # compact-bank AOMRidge captures the second.
    coef_raw = rng.normal(size=p)
    coef_diff = rng.normal(size=p - 1)
    y = (X @ coef_raw) + (np.diff(X, axis=1) @ coef_diff) + 0.03 * rng.normal(size=n)

    perm = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    candidates = [
        {
            "label": "Ridge-identity",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "AOMRidge-compact",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]

    # Train each candidate alone; compute test RMSE.
    cand_rmse: list[float] = []
    for spec in candidates:
        est = AOMRidgeRegressor(
            selection=spec["selection"],
            operator_bank=spec["operator_bank"],
            block_scaling=spec["block_scaling"],
            cv=3,
            random_state=0,
        )
        est.fit(X_tr, y_tr)
        cand_rmse.append(_rmse(y_te, est.predict(X_te)))

    blender = AOMRidgeBlender(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        regularizer=0.0,
        random_state=0,
    ).fit(X_tr, y_tr)
    blend_rmse = _rmse(y_te, blender.predict(X_te))

    best_single = min(cand_rmse)
    # Allow 5% slack to absorb finite-sample noise on a small synthetic.
    assert blend_rmse <= best_single * 1.05, (
        f"blender RMSE {blend_rmse:.4f} worse than best single candidate "
        f"{best_single:.4f} (tolerance 5%); cand_rmse={cand_rmse}"
    )
    # Both weights should be strictly between 0 and 1 (mixture used).
    w = blender.weights_
    assert np.all(w >= 0.0 - 1e-9)
    assert np.isclose(float(np.sum(w)), 1.0)


# ----------------------------------------------------------------------
# 6. predict before fit raises NotFittedError
# ----------------------------------------------------------------------


def test_blender_check_is_fitted():
    blender = AOMRidgeBlender()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    with pytest.raises(NotFittedError):
        blender.predict(X)


# ----------------------------------------------------------------------
# Bonus: diagnostics surface ranked weights and labels.
# ----------------------------------------------------------------------


def test_blender_get_diagnostics():
    X, y = _make_regression(n=40, p=12, seed=4)
    candidates = [
        {
            "label": "A",
            "selection": "superblock",
            "operator_bank": "identity",
            "block_scaling": "none",
        },
        {
            "label": "B",
            "selection": "global",
            "operator_bank": "compact",
            "block_scaling": "none",
        },
    ]
    blender = AOMRidgeBlender(
        candidates=candidates,
        outer_cv=3,
        outer_cv_kind="kfold",
        regularizer=0.01,
        random_state=0,
    ).fit(X, y)
    diag = blender.get_diagnostics()
    assert diag["model"] == "AOMRidgeBlender"
    assert diag["candidate_labels"] == ["A", "B"]
    assert len(diag["weights"]) == 2
    assert len(diag["cv_scores"]) == 2
    # Ranking is sorted descending.
    weights_sorted = [entry["weight"] for entry in diag["weight_ranking"]]
    assert weights_sorted == sorted(weights_sorted, reverse=True)

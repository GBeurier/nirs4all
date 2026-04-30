"""Phase A3 tests: 1-SE selection rule + RepeatedSPXYFold helper.

The 1-SE rule and per-fold tracking are exercised on synthetic CV curves
where the expected behaviour is unambiguous. Leakage assertions reuse the
same ``SpyOperator`` machinery as ``test_ridge_cv_no_leakage.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aomridge.cv import RepeatedSPXYFold
from aomridge.selection import (
    cv_score_alphas,
    select_alpha_with_rule,
)
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# 1-SE rule: synthetic CV curves
# ----------------------------------------------------------------------


def test_one_se_rule_picks_more_regularised():
    """Build a synthetic ``(n_folds, n_alphas)`` matrix whose minimum is at
    index 5 but where index 9 (a higher / more-regularised alpha) is within
    one standard error of that minimum. The 1-SE rule must prefer index 9.
    """
    n_folds, n_alphas = 5, 12
    alphas = np.logspace(-3, 3, n_alphas)
    # Mean curve: clear min at idx 5, gentle rise afterwards.
    base = np.array([1.5, 1.2, 1.0, 0.85, 0.70, 0.60, 0.62, 0.65, 0.68, 0.72, 0.85, 1.05])
    # Construct per-fold deviations at idx 5 with std 1.0 (SE = 1/sqrt(5) ≈ 0.447)
    # so that *every* alpha up to idx 9 (rmse 0.72) sits inside the band
    # 0.60 + 0.447 ≈ 1.05.
    rmse_per_fold = np.tile(base, (n_folds, 1)).astype(float)
    # Inject deviations only at the argmin column to drive a known SE.
    deviations = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rmse_per_fold[:, 5] = base[5] + deviations
    # Sanity: argmin of mean is index 5
    assert int(np.argmin(rmse_per_fold.mean(axis=0))) == 5

    idx_min = select_alpha_with_rule(rmse_per_fold, alphas, rule="min")
    assert idx_min == 5, "min rule should pick the global argmin"

    idx_1se = select_alpha_with_rule(rmse_per_fold, alphas, rule="1se")
    # 1-SE picks a *more regularised* alpha than the argmin.
    assert idx_1se > idx_min, (
        f"1-SE rule should pick a higher-alpha index; got {idx_1se} vs argmin {idx_min}"
    )
    # In this construction idx 9 (rmse 0.72) is within the SE band — the rule should reach it.
    assert idx_1se >= 9


def test_one_se_with_flat_curve_picks_min_or_higher():
    """A perfectly flat CV curve (all alphas identical) gives a tied
    minimum; the 1-SE rule must break the tie in favour of the most
    regularised alpha.
    """
    n_folds, n_alphas = 4, 8
    alphas = np.logspace(-2, 2, n_alphas)
    rmse_per_fold = np.full((n_folds, n_alphas), 0.5)
    idx_1se = select_alpha_with_rule(rmse_per_fold, alphas, rule="1se")
    assert idx_1se == n_alphas - 1, (
        "with a flat curve the 1-SE rule must pick the most regularised alpha"
    )
    # And the min rule still returns *some* argmin (here, the first index).
    idx_min = select_alpha_with_rule(rmse_per_fold, alphas, rule="min")
    assert rmse_per_fold.mean(axis=0)[idx_min] == rmse_per_fold.mean(axis=0).min()


def test_one_se_single_fold_returns_argmin():
    """With one fold the standard error is undefined; the rule falls back to argmin."""
    rmse_per_fold = np.array([[2.0, 1.0, 1.5, 0.8, 1.2]])
    alphas = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
    idx_1se = select_alpha_with_rule(rmse_per_fold, alphas, rule="1se")
    assert idx_1se == int(np.argmin(rmse_per_fold[0]))


def test_select_alpha_with_rule_validates_inputs():
    rng = np.random.default_rng(0)
    rmse = rng.uniform(size=(3, 5))
    alphas = np.linspace(1.0, 5.0, 5)
    with pytest.raises(ValueError):
        select_alpha_with_rule(rmse, np.linspace(1.0, 5.0, 4), rule="min")
    with pytest.raises(ValueError):
        select_alpha_with_rule(rmse, alphas, rule="bogus")
    with pytest.raises(ValueError):
        select_alpha_with_rule(rmse[0], alphas, rule="min")


def test_cv_score_alphas_returns_per_fold_matrix_when_requested():
    """``return_per_fold=True`` must yield ``(n_folds, n_alphas)`` whose row
    means equal the summary RMSE (when scoring is the default mean-of-RMSE).
    """
    rng = np.random.default_rng(7)
    n, p = 30, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    alphas = np.logspace(-2, 2, 5)
    summary, per_fold = cv_score_alphas(
        X, y.reshape(-1, 1), [IdentityOperator()], alphas, cv,
        block_scaling="none", scoring="rmse_mean", return_per_fold=True,
    )
    assert per_fold.shape == (3, alphas.size)
    np.testing.assert_allclose(per_fold.mean(axis=0), summary, rtol=1e-12)


# ----------------------------------------------------------------------
# Threading 1-SE through the estimator: behaves and produces diagnostics
# ----------------------------------------------------------------------


def test_estimator_selection_rule_one_se_recorded_in_diagnostics():
    """The estimator must accept ``selection_rule="1se"`` and surface it via
    diagnostics, leaving ``"min"`` as the default.
    """
    from aomridge.estimators import AOMRidgeRegressor

    rng = np.random.default_rng(12)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.05 * rng.normal(size=n)
    est_min = AOMRidgeRegressor(
        operator_bank="compact", cv=3, random_state=0,
    ).fit(X, y)
    est_1se = AOMRidgeRegressor(
        operator_bank="compact", cv=3, random_state=0, selection_rule="1se",
    ).fit(X, y)
    assert est_min.diagnostics_["selection_rule"] == "min"
    assert est_1se.diagnostics_["selection_rule"] == "1se"
    # 1-SE never picks an alpha *smaller* than the min rule.
    assert est_1se.alpha_ >= est_min.alpha_ - 1e-12


# ----------------------------------------------------------------------
# RepeatedSPXYFold
# ----------------------------------------------------------------------


def test_repeated_cv_yields_n_splits_x_n_repeats():
    """``get_n_splits`` and the iterator length must equal ``n_splits * n_repeats``."""
    rng = np.random.default_rng(0)
    n, p = 30, 8
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    rcv = RepeatedSPXYFold(n_splits=3, n_repeats=4, random_state=0)
    assert rcv.get_n_splits(X, y) == 12
    folds = list(rcv.split(X, y))
    assert len(folds) == 12
    # Each fold must be a (train, valid) pair of disjoint integer arrays.
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))
        assert len(tr) + len(va) == n


def test_repeated_cv_kfold_fallback_uses_distinct_seeds(monkeypatch):
    """When the SPXY backend is unavailable the fallback ``KFold(shuffle)``
    path must produce distinct fold assignments across repeats; otherwise the
    wrapper is not stabilising anything.
    """
    import aomridge.cv as cv_mod
    monkeypatch.setattr(cv_mod, "_try_import_spxyfold", lambda: None)

    rng = np.random.default_rng(1)
    n = 40
    X = rng.normal(size=(n, 6))
    y = rng.normal(size=n)
    rcv = cv_mod.RepeatedSPXYFold(n_splits=3, n_repeats=2, random_state=123)
    folds = list(rcv.split(X, y))
    first_repeat = folds[:3]
    second_repeat = folds[3:]
    set_first = tuple(tuple(sorted(va.tolist())) for _, va in first_repeat)
    set_second = tuple(tuple(sorted(va.tolist())) for _, va in second_repeat)
    assert set_first != set_second


# ----------------------------------------------------------------------
# Leakage probe re-using the SpyOperator pattern from test_ridge_cv_no_leakage
# ----------------------------------------------------------------------


class _SpyOperator(LinearSpectralOperator):
    """Identity-equivalent spy that records ``apply_cov`` column counts."""

    def __init__(self, p=None) -> None:
        super().__init__(name="spy_identity_a3", p=p)
        self.col_counts: list[int] = []

    def fit(self, X=None, y=None):
        if X is not None:
            self.p = np.asarray(X).shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        self.col_counts.append(int(S.shape[1]))
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def test_repeated_cv_no_leak():
    """Validation rows must never enter operator fits or kernel construction
    when ``RepeatedSPXYFold`` drives ``cv_score_alphas`` — the column count
    seen by ``apply_cov`` must equal a training-fold size, never ``n``.
    """
    rng = np.random.default_rng(99)
    n, p = 30, 10
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    rcv = RepeatedSPXYFold(n_splits=3, n_repeats=2, random_state=0)
    folds = list(rcv.split(X, y))
    train_sizes = {len(tr) for tr, _ in folds}

    spy = _SpyOperator(p=p)
    cv_score_alphas(
        X, y.reshape(-1, 1), [spy],
        np.array([0.5]), rcv,
        block_scaling="rms", center=True,
    )
    # The user-supplied template is never fitted directly: clones absorb the
    # state. The fold clones do, however, share the same class so their
    # accumulated col counts from the template instance stay empty.
    assert spy.col_counts == []

    # Now wire a CountColsSpy directly to inspect the per-fold clones.
    seen: list[int] = []

    class _Counter(_SpyOperator):
        def _apply_cov_impl(self, S):
            seen.append(int(S.shape[1]))
            return S.copy()

    cv_score_alphas(
        X, y.reshape(-1, 1), [_Counter(p=p)],
        np.array([0.5]), rcv,
        block_scaling="rms", center=True,
    )
    assert seen, "spy operator was never invoked"
    for k in seen:
        assert k in train_sizes, (
            f"apply_cov saw {k} columns; expected a training-fold size in {train_sizes}"
        )
        assert k != n, "operator was applied to the full dataset (leak)"

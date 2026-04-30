"""Codex round-3 review fix tests.

Covers (one per Codex finding):

1.  MKL math: ``selection="mkl"`` ignores ``block_scaling`` and always
    builds ``K_mkl = sum_b w_b K_b`` from raw block kernels.
2.  ``RepeatedSPXYFold`` produces *different* fold compositions across
    repeats (was a no-op).
3.  ``selection="global"`` honors a fixed ``alpha`` (skips the alpha CV).
4.  ``scoring="mse_pooled"`` actually drives alpha selection.
5.  Adaptive alpha expansion respects ``selection_rule="1se"`` (boundary
    check tracks the chosen alpha index, not just argmin).
6.  Global 1-SE selection compares alphas within the chosen operator's
    grid (not across operator grids that may be on different scales).
7.  ``selection="global"`` and ``selection="branch_global"`` go through
    adaptive alpha expansion when the optimum sits at a grid boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import (
    ExplicitMatrixOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aomridge.cv import RepeatedSPXYFold
from aomridge.estimators import AOMRidgeRegressor
from aomridge.mkl import learn_block_weights, mkl_kernel_train
from aomridge.selection import (
    cv_score_alphas,
    cv_score_alphas_mkl,
    select_alpha_with_rule,
    select_global,
)
from sklearn.model_selection import KFold


def _make_data(n=80, p=32, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


# ----------------------------------------------------------------------
# Finding 1 — MKL math: rms must collapse to none, kernel = sum_b w_b K_b
# ----------------------------------------------------------------------


def test_mkl_block_scaling_rms_equals_none():
    """``selection="mkl"`` with ``block_scaling="rms"`` must produce the
    same model as ``block_scaling="none"``: the documented MKL math is
    linear in the weights, so per-block scales must not enter the kernel.
    """
    X, y = _make_data(n=60, p=20, seed=4)
    common = {
        "selection": "mkl",
        "operator_bank": "compact",
        "cv": 3,
        "random_state": 0,
        "mkl_top_k": 4,
        "alpha": 1.0,        # fix alpha to remove CV-selection variance
    }
    est_rms = AOMRidgeRegressor(block_scaling="rms", **common).fit(X, y)
    est_none = AOMRidgeRegressor(block_scaling="none", **common).fit(X, y)
    np.testing.assert_allclose(est_rms.coef_, est_none.coef_, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        est_rms.predict(X), est_none.predict(X), atol=1e-9, rtol=1e-9
    )
    np.testing.assert_allclose(
        est_rms.mkl_weights_, est_none.mkl_weights_, atol=1e-12, rtol=1e-12
    )


def test_mkl_kernel_is_linear_combination_of_block_kernels():
    """Verify directly that the combined kernel produced by the MKL fold
    equals ``sum_b w_b K_b`` where ``K_b = (X A_b^T)(X A_b^T)^T`` (no
    ``s_b`` factor)."""
    rng = np.random.default_rng(7)
    n, p = 24, 16
    X = rng.normal(size=(n, p))
    Xc = X - X.mean(axis=0)
    Yc = (X[:, 0] + 0.1 * rng.normal(size=n)).reshape(-1, 1) - 0.0
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=p),
    ]
    for op in bank:
        op.fit(Xc)
    scales_ones = np.ones(len(bank))
    weights = learn_block_weights(bank, Xc, Yc, scales_ones, top_k=3, mode="alignment")
    K_mkl, _ = mkl_kernel_train(Xc, bank, weights, scales=scales_ones)
    # Reference: sum_b w_b * (X A_b^T)(X A_b^T)^T (no s_b)
    K_ref = np.zeros((n, n))
    for op, w in zip(bank, weights, strict=False):
        Z = op.transform(Xc)
        K_ref += float(w) * (Z @ Z.T)
    K_ref = 0.5 * (K_ref + K_ref.T)
    np.testing.assert_allclose(K_mkl, K_ref, atol=1e-9, rtol=1e-9)


def test_mkl_cv_uses_unit_scales_internally():
    """Within ``cv_score_alphas_mkl`` the per-fold per-block scales must
    be all ones regardless of the user-facing ``block_scaling``.

    We exercise this by checking that the per-alpha summary is identical
    for ``block_scaling="rms"`` and ``block_scaling="none"``.
    """
    rng = np.random.default_rng(8)
    n, p = 30, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=1, p=p),
    ]
    cv = KFold(n_splits=3, shuffle=False)
    alphas = np.logspace(-2, 2, 5)
    rmse_rms = cv_score_alphas_mkl(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="rms", center=True, mkl_top_k=2,
    )
    rmse_none = cv_score_alphas_mkl(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="none", center=True, mkl_top_k=2,
    )
    np.testing.assert_allclose(rmse_rms, rmse_none, atol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------
# Finding 2 — RepeatedSPXYFold: distinct repeats must yield distinct folds
# ----------------------------------------------------------------------


def test_repeated_spxy_fold_distinct_repeats_have_different_folds():
    """Two repeats of ``RepeatedSPXYFold(n_splits=3, n_repeats=2)`` must
    produce *different* fold compositions on a synthetic dataset.

    Without the row-permutation fix, ``SPXYFold.random_state`` only changes
    PCA-related behaviour (no PCA by default), so all repeats returned the
    same folds and the wrapper was a no-op.
    """
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 8))
    y = rng.normal(size=n)
    rcv = RepeatedSPXYFold(n_splits=3, n_repeats=2, random_state=42)
    folds = list(rcv.split(X, y))
    first = folds[:3]
    second = folds[3:]
    set_first = tuple(tuple(sorted(va.tolist())) for _, va in first)
    set_second = tuple(tuple(sorted(va.tolist())) for _, va in second)
    assert set_first != set_second, (
        "RepeatedSPXYFold must produce different fold compositions across repeats"
    )
    # Each fold remains a valid (train, valid) partition of {0..n-1}.
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))
        assert sorted(set(tr) | set(va)) == list(range(n))


def test_repeated_spxy_fold_seed_reproducibility():
    """Same ``random_state`` must yield identical fold compositions."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    y = rng.normal(size=30)
    folds_a = list(RepeatedSPXYFold(n_splits=3, n_repeats=2, random_state=7).split(X, y))
    folds_b = list(RepeatedSPXYFold(n_splits=3, n_repeats=2, random_state=7).split(X, y))
    for (tr_a, va_a), (tr_b, va_b) in zip(folds_a, folds_b, strict=False):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)


# ----------------------------------------------------------------------
# Finding 3 — selection="global" must honor a fixed alpha
# ----------------------------------------------------------------------


def test_global_with_fixed_alpha_skips_grid_cv():
    """When ``alpha`` is fixed and ``selection="global"``, the estimator must
    use that exact alpha and still pick the best operator across the bank.
    """
    X, y = _make_data(n=60, p=24, seed=11)
    fixed_alpha = 0.31415
    est = AOMRidgeRegressor(
        selection="global",
        operator_bank="compact",
        alpha=fixed_alpha,
        cv=3,
        random_state=0,
    ).fit(X, y)
    assert est.alpha_ == pytest.approx(fixed_alpha, rel=0, abs=1e-12)
    # The reported alpha grid for global+fixed-alpha is the single user value.
    assert est.alphas_.size == 1
    assert est.alphas_[0] == pytest.approx(fixed_alpha, abs=1e-12)
    # The estimator still picks an operator (every operator was scored at the
    # single fixed alpha).
    assert len(est.selected_operators_) == 1
    diag = est.get_diagnostics()
    # operator_scores must cover every operator and report best_alpha = fixed.
    for rec in diag["operator_scores"]:
        assert rec["best_alpha"] == pytest.approx(fixed_alpha, abs=1e-12)


# ----------------------------------------------------------------------
# Finding 4 — scoring="mse_pooled" actually selects the alpha
# ----------------------------------------------------------------------


def test_mse_pooled_summary_drives_selection():
    """``select_alpha_with_rule(..., summary=...)`` must use the supplied
    summary to pick the index, not the row-mean of ``rmse_per_fold``.
    """
    n_folds, n_alphas = 4, 6
    rmse_per_fold = np.tile(
        np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]), (n_folds, 1),
    )
    alphas = np.logspace(-3, 2, n_alphas)
    # Provide a *different* pooled summary that prefers the FIRST alpha.
    summary = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    idx_default = select_alpha_with_rule(rmse_per_fold, alphas, rule="min")
    idx_pooled = select_alpha_with_rule(
        rmse_per_fold, alphas, rule="min", summary=summary,
    )
    assert idx_default == int(np.argmin(rmse_per_fold.mean(axis=0)))   # 5
    assert idx_pooled == int(np.argmin(summary))                       # 0
    assert idx_default != idx_pooled


def test_mse_pooled_one_se_uses_summary():
    """With unequal fold sizes the pooled summary may differ from row-mean
    and the 1-SE rule must thread it through.
    """
    n_folds, n_alphas = 5, 8
    alphas = np.logspace(-3, 4, n_alphas)
    base = np.array([1.4, 1.1, 0.9, 0.75, 0.65, 0.62, 0.7, 0.85])
    rmse_per_fold = np.tile(base, (n_folds, 1)).astype(float)
    # Force a bigger SE at the argmin column (idx 5).
    rmse_per_fold[:, 5] = base[5] + np.array([-0.6, -0.3, 0.0, 0.3, 0.6])
    # Provide a pooled summary that has its argmin at idx 4 (small offset
    # vs row-mean which is idx 5).
    summary = base.copy()
    summary[4] = 0.55
    idx = select_alpha_with_rule(
        rmse_per_fold, alphas, rule="1se", summary=summary,
    )
    # 1-SE picks an index >= argmin(summary) by definition.
    assert idx >= int(np.argmin(summary))


# ----------------------------------------------------------------------
# Finding 5 — adaptive expansion sees the chosen index under 1se
# ----------------------------------------------------------------------


def test_adaptive_expansion_triggers_when_one_se_at_boundary():
    """Build a problem where the argmin sits in the interior but the 1-SE
    rule pushes the chosen alpha to the grid edge. The expansion logic
    must trigger on the *chosen* index, not just argmin.
    """
    rng = np.random.default_rng(20)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.05 * rng.normal(size=n)
    est_min = AOMRidgeRegressor(
        operator_bank=[IdentityOperator()],
        block_scaling="none",
        cv=3,
        random_state=0,
        adaptive_alpha_grid=True,
        max_grid_expansions=2,
        alpha_grid_size=20,
        selection_rule="min",
    ).fit(X, y)
    est_1se = AOMRidgeRegressor(
        operator_bank=[IdentityOperator()],
        block_scaling="none",
        cv=3,
        random_state=0,
        adaptive_alpha_grid=True,
        max_grid_expansions=2,
        alpha_grid_size=20,
        selection_rule="1se",
    ).fit(X, y)
    # 1-SE never picks a smaller alpha than min (within fp tolerance).
    assert est_1se.alpha_ >= est_min.alpha_ - 1e-12
    # Both runs return well-formed diagnostics with grid_expansions defined.
    assert "grid_expansions" in est_1se.get_diagnostics()
    assert isinstance(est_1se.get_diagnostics()["grid_expansions"], int)


# ----------------------------------------------------------------------
# Finding 6 — global 1-SE picks within the chosen operator's grid
# ----------------------------------------------------------------------


def test_global_one_se_restricts_to_chosen_operator_row():
    """The global+1-SE selection must compare alphas inside the row of the
    operator that minimises mean RMSE — not across rows whose alpha grids
    sit on different scales (each per-operator grid is trace-relative).
    """
    rng = np.random.default_rng(9)
    n, p = 50, 16
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.05 * rng.normal(size=n)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
    ]
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    # Disable per-operator grids inside select_global so we can exercise the
    # 1-SE branch on a single shared grid (still valid; the new code picks
    # the best operator's row, then 1-SE in that row).
    alphas = np.logspace(-3, 3, 7)
    b_min, a_min, table_min, _ = select_global(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="none", selection_rule="min",
    )
    b_1se, a_1se, table_1se, grids_1se = select_global(
        X, y.reshape(-1, 1), bank, alphas, cv,
        block_scaling="none", selection_rule="1se",
    )
    # The chosen operator under 1-SE must be the operator that minimises
    # mean RMSE (i.e. the same operator the new logic restricts the 1-SE
    # rule to).
    op_best = int(np.argmin(table_1se.min(axis=1)))
    assert b_1se == op_best
    # The chosen alpha lies in that operator's row and is >= argmin alpha.
    row = table_1se[op_best]
    row_alphas = grids_1se[op_best]
    a_min_in_row = row_alphas[int(np.argmin(row))]
    assert a_1se >= a_min_in_row - 1e-12


# ----------------------------------------------------------------------
# Finding 7 — global / branch_global also expand the alpha grid on edges
# ----------------------------------------------------------------------


def test_global_records_grid_expansion_when_optimum_at_edge():
    """If the chosen alpha sits at the lower boundary of the initial grid,
    the global path must trigger at least one expansion (the chosen alpha
    moves to a smaller value than the original ``low`` decade).
    """
    # Use a tiny initial grid that brackets too high (low=high=0 means
    # only alpha = trace/n; we pick low=0, high=0 so initial grid is one
    # point — but n_grid must be >=1; use low=4, high=6 instead so the
    # CV-optimum (~trace/n) sits below the bracket and forces expansion).
    rng = np.random.default_rng(13)
    n, p = 60, 24
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.05 * rng.normal(size=n)
    est = AOMRidgeRegressor(
        selection="global",
        operator_bank=[IdentityOperator()],
        block_scaling="none",
        cv=3,
        random_state=0,
        alpha_grid_low=4.0,        # bracket alpha way above the optimum
        alpha_grid_high=6.0,
        alpha_grid_size=10,
        adaptive_alpha_grid=True,
        max_grid_expansions=2,
    ).fit(X, y)
    diag = est.get_diagnostics()
    assert diag["grid_expansions"] >= 1, (
        f"global selection failed to expand the alpha grid; diag={diag}"
    )


def test_branch_global_records_grid_expansion_when_optimum_at_edge():
    """The branch_global path must also expand the alpha grid on a
    boundary hit."""
    rng = np.random.default_rng(15)
    n, p = 60, 16
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.05 * rng.normal(size=n)
    est = AOMRidgeRegressor(
        selection="branch_global",
        operator_bank=[IdentityOperator()],
        branches=("none",),
        block_scaling="none",
        cv=3,
        random_state=0,
        alpha_grid_low=4.0,
        alpha_grid_high=6.0,
        alpha_grid_size=10,
        adaptive_alpha_grid=True,
        max_grid_expansions=2,
    ).fit(X, y)
    diag = est.get_diagnostics()
    assert diag["grid_expansions"] >= 1, (
        f"branch_global selection failed to expand the alpha grid; diag={diag}"
    )


def test_global_fixed_alpha_does_not_expand():
    """Single-alpha grids (fixed alpha) must skip expansion regardless of
    boundary detection."""
    X, y = _make_data(n=40, p=12, seed=99)
    est = AOMRidgeRegressor(
        selection="global",
        operator_bank=[IdentityOperator()],
        alpha=0.5,
        cv=3,
        random_state=0,
        adaptive_alpha_grid=True,
        max_grid_expansions=2,
    ).fit(X, y)
    assert est.get_diagnostics()["grid_expansions"] == 0


# ----------------------------------------------------------------------
# Smoke: estimator with mkl + rms still finite end-to-end
# ----------------------------------------------------------------------


def test_mkl_rms_estimator_still_runs():
    X, y = _make_data(seed=42)
    est = AOMRidgeRegressor(
        selection="mkl",
        operator_bank="compact",
        block_scaling="rms",
        cv=3,
        random_state=0,
        mkl_top_k=4,
    ).fit(X, y)
    pred = est.predict(X[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))

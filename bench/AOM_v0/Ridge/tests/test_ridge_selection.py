"""Phase 5/6 tests: global hard selection and active superblock."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from aompls.banks import compact_bank
from aompls.operators import (
    ExplicitMatrixOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aomridge.estimators import AOMRidgeRegressor
from aomridge.selection import screen_active_operators, select_global
from sklearn.model_selection import KFold


def _make_data(n=80, p=64, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


# ----------------------------------------------------------------------
# Global hard selection
# ----------------------------------------------------------------------


def test_global_identity_only_picks_identity():
    X, y = _make_data()
    est = AOMRidgeRegressor(
        selection="global",
        operator_bank=[IdentityOperator()],
        cv=3,
        random_state=0,
    ).fit(X, y)
    assert est.selected_operators_ == ["identity"]
    assert est.selected_operator_indices_ == [0]
    assert est.alpha_ in est.alphas_
    pred = est.predict(X[:5])
    assert np.all(np.isfinite(pred))


def test_global_selects_valid_pair_with_compact_bank():
    X, y = _make_data(seed=1)
    est = AOMRidgeRegressor(
        selection="global",
        operator_bank="compact",
        cv=3,
        random_state=0,
    ).fit(X, y)
    assert len(est.selected_operators_) == 1
    assert est.selected_operator_indices_[0] >= 0
    assert est.alpha_ in est.alphas_
    diag = est.get_diagnostics()
    assert "operator_scores" in diag
    # Operator scores cover every candidate (identity always present) and
    # are a JSON-serializable list of records keyed by index.
    bank_size = len(compact_bank(p=X.shape[1]))
    assert len(diag["operator_scores"]) == bank_size
    indices = {rec["index"] for rec in diag["operator_scores"]}
    assert indices == set(range(bank_size))
    np.testing.assert_array_equal(np.isfinite(est.predict(X[:5])), True)


def test_global_returns_alpha_in_grid_for_function_api():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 16))
    y = X[:, 0] + 0.1 * rng.normal(size=40)
    bank = [IdentityOperator(),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=16)]
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    alphas = np.logspace(-3, 3, 6)
    b, a, table, grids = select_global(
        X, y.reshape(-1, 1), bank, alphas, cv, block_scaling="rms"
    )
    assert 0 <= b < len(bank)
    assert a in alphas
    assert table.shape == (len(bank), len(alphas))
    assert np.all(np.isfinite(table))


# ----------------------------------------------------------------------
# Active superblock
# ----------------------------------------------------------------------


def test_active_superblock_keeps_identity_and_caps_size():
    X, y = _make_data(seed=3)
    est = AOMRidgeRegressor(
        selection="active_superblock",
        operator_bank="default",
        active_top_m=10,
        active_diversity_threshold=0.98,
        cv=3,
        random_state=0,
    ).fit(X, y)
    assert "identity" in est.selected_operators_
    assert len(est.selected_operators_) <= 10
    diag = est.get_diagnostics()
    assert "active_operator_names" in diag
    assert diag["active_top_m"] == 10
    assert isinstance(diag["active_pruned_count"], int)
    assert np.all(np.isfinite(est.predict(X[:5])))


def test_active_superblock_prunes_duplicate_operators():
    rng = np.random.default_rng(4)
    n, p = 40, 32
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    # Bank with identity plus 5 *different gain* explicit identity copies.
    # All 5 produce a perfectly-correlated response signature, so they must
    # be pruned in favour of the first identity.
    duplicates = [ExplicitMatrixOperator(g * np.eye(p), name=f"gain_{g}")
                  for g in (1.0, 2.0, 3.0, 4.0, 5.0)]
    bank = [IdentityOperator()] + duplicates
    active, scores, pruned = screen_active_operators(
        X, y.reshape(-1, 1), bank,
        block_scaling="rms",
        center=True,
        top_m=20,
        diversity_threshold=0.95,
        keep_identity=True,
    )
    # Identity is present, and at least 4 of 5 duplicates were pruned
    assert 0 in active
    assert pruned >= 4


def test_active_max_per_family_quota_enforced():
    """``max_per_family`` caps how many operators of each family enter the
    active set, preventing one family (e.g. SG smoothers) from monopolising
    the slots when multiple SG variants score high.
    """
    from aompls.banks import compact_bank
    rng = np.random.default_rng(20)
    n, p = 60, 64
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = compact_bank(p=p)            # has identity, 2 SG-smooth, 2 SG-d1, 1 SG-d2, 2 detrend, 1 fd
    active, _, _ = screen_active_operators(
        X, y.reshape(-1, 1), bank,
        block_scaling="none",
        top_m=20,
        diversity_threshold=0.999,      # disable diversity pruning
        max_per_family=1,
        keep_identity=True,
    )
    # Each family seen at most once (identity is its own family)
    from aomridge.selection import _operator_family
    families = [_operator_family(bank[i].name) for i in active]
    assert len(families) == len(set(families)), f"family quota violated: {families}"


def test_active_score_method_kta_and_blend_run():
    """Both KTA and blend score methods must produce a finite ranking."""
    rng = np.random.default_rng(21)
    n, p = 30, 20
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = [IdentityOperator(),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=p)]
    for method in ("kta", "blend"):
        active, scores, _ = screen_active_operators(
            X, y.reshape(-1, 1), bank,
            block_scaling="none",
            top_m=2, diversity_threshold=0.99,
            score_method=method,
        )
        assert len(active) == 2
        assert all(np.isfinite(scores))


def test_active_superblock_top_m_one_returns_only_identity():
    rng = np.random.default_rng(8)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=p),
    ]
    active, _, _ = screen_active_operators(
        X, y.reshape(-1, 1), bank,
        block_scaling="rms",
        top_m=1,
        diversity_threshold=0.98,
        keep_identity=True,
    )
    assert active == [0]


def test_active_superblock_strict_threshold_keeps_only_identity_for_dup_set():
    rng = np.random.default_rng(5)
    n, p = 40, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = [IdentityOperator(), ExplicitMatrixOperator(np.eye(p), name="dup")]
    active, scores, pruned = screen_active_operators(
        X, y.reshape(-1, 1), bank,
        block_scaling="rms",
        diversity_threshold=0.95,
        keep_identity=True,
    )
    assert active == [0]
    assert pruned == 1


# ----------------------------------------------------------------------
# Operator cloning safety: estimator must not mutate user instances
# ----------------------------------------------------------------------


def test_estimator_does_not_mutate_user_bank_state():
    X, y = _make_data(seed=6)
    bank = [IdentityOperator(),
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=X.shape[1])]
    snapshot = [deepcopy(op) for op in bank]
    AOMRidgeRegressor(operator_bank=bank, alpha=0.5, cv=3, random_state=0).fit(X, y)
    # User-supplied operators are unchanged in name/parameters
    for original, after in zip(snapshot, bank, strict=False):
        assert original.name == after.name
        if hasattr(original, "window_length"):
            assert original.window_length == after.window_length
            assert original.polyorder == after.polyorder
            assert original.deriv == after.deriv


# ----------------------------------------------------------------------
# Diagnostics serialisability
# ----------------------------------------------------------------------


def test_diagnostics_are_json_serialisable():
    import json

    X, y = _make_data(seed=7)
    est = AOMRidgeRegressor(
        operator_bank="compact", cv=3, random_state=0
    ).fit(X, y)
    diag = est.get_diagnostics()
    json.dumps(diag)  # raises TypeError if any value is not JSON-friendly

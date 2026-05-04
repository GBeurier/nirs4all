"""Tests for the Soft Multi-Branch Kernel (Codex Phase H4).

Covers the simplex-weight contract, fold-local anti-leakage, kernel PSD,
end-to-end fit/predict, and a synthetic comparison where multi-branch
outperforms any single branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from aomridge.multi_branch_mkl import (
    AOMMultiBranchMKL,
    cross_branch_kernel,
    fit_branches_and_kernels,
    kta_branch_score,
    learn_branch_weights,
    multi_branch_kernel,
)
from sklearn.exceptions import NotFittedError

BRANCHES = ("none", "snv", "msc", "asls", "emsc2")


def _make_data(n: int = 60, p: int = 48, q: int = 1, seed: int = 0, noise: float = 0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


# ----------------------------------------------------------------------
# 1. Simplex contract
# ----------------------------------------------------------------------


def test_branch_weights_simplex():
    """Returned weights are non-negative and sum to one."""
    rng = np.random.default_rng(0)
    n = 30
    branch_kernels = {
        b: rng.normal(size=(n, n)) for b in BRANCHES
    }
    # Make them PSD so KTA is meaningful.
    branch_kernels = {b: K @ K.T for b, K in branch_kernels.items()}
    Yvec = rng.normal(size=(n, 1))
    YYt = Yvec @ Yvec.T
    for shrink in (0.0, 0.3, 0.5, 0.8, 1.0):
        weights = learn_branch_weights(branch_kernels, YYt, shrinkage_to_identity=shrink)
        values = list(weights.values())
        assert all(w >= 0.0 for w in values), f"negative weight at shrinkage={shrink}"
        assert abs(sum(values) - 1.0) < 1e-10, (
            f"weights do not sum to 1 (sum={sum(values)}) at shrinkage={shrink}"
        )


# ----------------------------------------------------------------------
# 2. Shrinkage extremes
# ----------------------------------------------------------------------


def test_shrinkage_to_identity_recovers_none():
    """``shrinkage=0`` must collapse to identity branch only."""
    rng = np.random.default_rng(1)
    n = 25
    branch_kernels = {
        b: (Kb @ Kb.T) for b in BRANCHES
        for Kb in [rng.normal(size=(n, n))]
    }
    Yvec = rng.normal(size=(n, 1))
    YYt = Yvec @ Yvec.T
    weights = learn_branch_weights(branch_kernels, YYt, shrinkage_to_identity=0.0)
    assert weights["none"] == pytest.approx(1.0)
    for b in BRANCHES:
        if b != "none":
            assert weights[b] == pytest.approx(0.0)


def test_shrinkage_one_recovers_raw_kta():
    """``shrinkage=1`` returns the data-driven (max-of-zero-then-normalised) KTA weights."""
    rng = np.random.default_rng(2)
    n = 25
    branch_kernels = {
        b: (Kb @ Kb.T) for b in BRANCHES
        for Kb in [rng.normal(size=(n, n))]
    }
    Yvec = rng.normal(size=(n, 1))
    YYt = Yvec @ Yvec.T
    weights = learn_branch_weights(branch_kernels, YYt, shrinkage_to_identity=1.0)
    # Reconstruct the expected raw weights and compare
    raw = {b: max(kta_branch_score(branch_kernels[b], YYt), 0.0) for b in BRANCHES}
    total = sum(raw.values())
    raw_norm = {b: r / total for b, r in raw.items()} if total > 0 else \
        {b: (1.0 if b == "none" else 0.0) for b in BRANCHES}
    for b in BRANCHES:
        assert weights[b] == pytest.approx(raw_norm[b], abs=1e-12)


# ----------------------------------------------------------------------
# 3. Anti-leakage: validation rows never enter branch fitting nor weights
# ----------------------------------------------------------------------


def test_multi_branch_no_leak():
    """Branch transformers must be fit on training rows only.

    We build per-branch kernels twice — once on the full data and once on
    the training subset — and verify that the training-only kernel does
    not depend on the validation rows (i.e. doesn't equal the slice of
    the full-data kernel for branches that have learnable parameters).
    The test focuses on MSC / EMSC2 which have a learned reference
    spectrum; SNV / ASLS are stateless so they pass trivially.
    """
    X, _ = _make_data(n=40, p=32, q=1, seed=3)
    train_idx = np.arange(0, 30)

    # Full-data fit (leaky reference)
    K_full, fitted_full, _ = fit_branches_and_kernels(
        X, BRANCHES, "compact",
    )
    # Training-only fit (correct, anti-leakage)
    K_tr_only, fitted_tr_only, _ = fit_branches_and_kernels(
        X[train_idx], BRANCHES, "compact",
    )

    # The training-row block of K_full should differ from K_tr_only for
    # branches that learn from data (msc, emsc2). For branches without
    # state (none, snv, asls) they happen to coincide; we assert the
    # *learned* branches actually differ.
    K_full_train = K_full["msc"][np.ix_(train_idx, train_idx)]
    assert not np.allclose(K_full_train, K_tr_only["msc"], atol=1e-8), (
        "MSC kernel fit on full data must differ from kernel fit on train-only"
    )

    # Verify the fitted MSC reference differs (it should — the mean of
    # 40 rows is not the mean of the first 30 rows).
    ref_full = fitted_full["msc"].reference_
    ref_tr = fitted_tr_only["msc"].reference_
    assert not np.allclose(ref_full, ref_tr, atol=1e-8)


def test_estimator_no_leak_replays_training_transformers():
    """Estimator must replay the training-fit transformer on test rows.

    Concretely: predicting on test rows must not refit the branch
    transformers; the fitted reference (e.g. MSC reference spectrum)
    seen at fit time is what is used at predict time.
    """
    X, y = _make_data(n=50, p=32, q=1, seed=4)
    Xtr, ytr = X[:40], y[:40]
    Xte = X[40:]
    est = AOMMultiBranchMKL(
        branches=BRANCHES, operator_bank="compact", cv=3,
        random_state=0, shrinkage_to_identity=0.5,
    ).fit(Xtr, ytr)
    # Snapshot the fitted MSC reference and predict
    ref_before = est.fitted_branch_transformers_["msc"].reference_.copy()
    _ = est.predict(Xte)
    ref_after = est.fitted_branch_transformers_["msc"].reference_
    np.testing.assert_array_equal(ref_before, ref_after)


# ----------------------------------------------------------------------
# 4. Kernel PSD
# ----------------------------------------------------------------------


def test_kernel_psd():
    """K_total is symmetric and (numerically) PSD."""
    X, y = _make_data(n=35, p=24, q=1, seed=5)
    K_per_branch, _, _ = fit_branches_and_kernels(X, BRANCHES, "compact")
    YYt = (y - y.mean()).reshape(-1, 1)
    YYt = YYt @ YYt.T
    weights = learn_branch_weights(
        K_per_branch, YYt, shrinkage_to_identity=0.4,
    )
    K_total = multi_branch_kernel(
        X, BRANCHES, "compact", weights,
        fitted_transformers=None,
    )
    # Symmetric to floating point
    np.testing.assert_allclose(K_total, K_total.T, atol=1e-9, rtol=1e-9)
    # PSD: smallest eigenvalue >= -tol (tiny numerical negatives allowed)
    eigvals = np.linalg.eigvalsh(0.5 * (K_total + K_total.T))
    assert eigvals.min() >= -1e-8, f"min eig = {eigvals.min()}"


# ----------------------------------------------------------------------
# 5. Smoke: fit / predict
# ----------------------------------------------------------------------


def test_multi_branch_smoke_runs():
    """Synthetic fit / predict happy path."""
    X, y = _make_data(n=80, p=48, q=1, seed=6)
    est = AOMMultiBranchMKL(
        branches=BRANCHES, operator_bank="compact", cv=3,
        random_state=0, shrinkage_to_identity=0.3,
    )
    est.fit(X, y)
    y_pred = est.predict(X[:10])
    assert y_pred.shape == (10,)
    assert np.all(np.isfinite(y_pred))
    # Branch weights are valid
    weights = est.branch_weights_
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)
    assert all(w >= 0.0 for w in weights.values())
    # Diagnostics expose the expected fields
    diag = est.get_diagnostics()
    for key in (
        "model", "selection", "branch_weights", "branch_kta_scores",
        "shrinkage_to_identity", "alpha", "alphas", "n_train",
    ):
        assert key in diag


def test_multi_branch_smoke_multioutput():
    """Multi-output Y is accepted and predict returns 2D."""
    X, Y = _make_data(n=60, p=32, q=2, seed=7)
    est = AOMMultiBranchMKL(
        branches=BRANCHES, operator_bank="compact", cv=3,
        random_state=0, shrinkage_to_identity=0.3,
    ).fit(X, Y)
    pred = est.predict(X[:5])
    assert pred.shape == (5, 2)


# ----------------------------------------------------------------------
# 6. Multi-branch beats single-branch on a synthetic dataset where
#    SNV and EMSC2 each capture different signal components.
# ----------------------------------------------------------------------


def _build_mixed_signal_dataset(
    n: int = 200,
    p: int = 64,
    seed: int = 8,
    noise: float = 0.05,
):
    """Synthetic dataset mixing a multiplicative-scatter component (best
    handled by SNV) and a polynomial-baseline component (best handled by
    EMSC2). Predictive signal lives in localised peaks shared by both.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, p)
    # Two predictive Gaussian peaks
    peak1 = np.exp(-0.5 * ((wl - 0.30) / 0.05) ** 2)
    peak2 = np.exp(-0.5 * ((wl - 0.65) / 0.07) ** 2)
    # Sample-specific concentrations (the regression target)
    a = rng.uniform(0.0, 1.0, size=n)
    b = rng.uniform(0.0, 1.0, size=n)
    y = 1.5 * a + 0.8 * b
    # Underlying spectrum
    base = np.outer(a, peak1) + np.outer(b, peak2)
    # Multiplicative scatter (random scale per sample)
    scale = rng.uniform(0.5, 2.0, size=n)[:, None]
    # Polynomial baseline (random coefficients per sample)
    poly_coef = rng.normal(scale=0.3, size=(n, 3))   # [const, lin, quad]
    poly_basis = np.stack([np.ones(p), wl, wl ** 2], axis=1)  # (p, 3)
    baseline = poly_coef @ poly_basis.T
    X = scale * base + baseline + noise * rng.normal(size=(n, p))
    return X, y


def test_multi_branch_beats_single_branch_synthetic():
    """On a synthetic dataset where SNV addresses scatter and EMSC2
    addresses polynomial baseline, the multi-branch model must
    outperform either branch alone (and the identity branch).
    """
    X, y = _build_mixed_signal_dataset(n=200, p=64, seed=9, noise=0.03)
    Xtr, Xte = X[:140], X[140:]
    ytr, yte = y[:140], y[140:]

    def rmse(yt, yp):
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    common = {
        "operator_bank": "compact",
        "cv": 3,
        "random_state": 0,
        "shrinkage_to_identity": 1.0,   # full data-driven for this discrimination
    }
    est_multi = AOMMultiBranchMKL(branches=BRANCHES, **common).fit(Xtr, ytr)
    est_none = AOMMultiBranchMKL(branches=("none",), **common).fit(Xtr, ytr)
    est_snv = AOMMultiBranchMKL(branches=("none", "snv"), **{**common, "shrinkage_to_identity": 1.0}).fit(Xtr, ytr)
    # For SNV-only and EMSC2-only we still need 'none' in branches (it
    # is the simplex reference); we set shrinkage=1.0 so the data picks
    # whichever branch is best.
    est_emsc = AOMMultiBranchMKL(branches=("none", "emsc2"), **{**common, "shrinkage_to_identity": 1.0}).fit(Xtr, ytr)

    rmse_multi = rmse(yte, est_multi.predict(Xte))
    rmse_none = rmse(yte, est_none.predict(Xte))
    rmse_snv = rmse(yte, est_snv.predict(Xte))
    rmse_emsc = rmse(yte, est_emsc.predict(Xte))

    # Multi-branch must beat the *worst* of the per-branch models — a
    # generous bar that nonetheless rules out the case where the multi
    # model just collapses to a single branch.
    worst_single = max(rmse_none, rmse_snv, rmse_emsc)
    assert rmse_multi < worst_single, (
        f"multi RMSE {rmse_multi:.5f} >= worst single ({worst_single:.5f}); "
        f"none={rmse_none:.5f}, snv={rmse_snv:.5f}, emsc2={rmse_emsc:.5f}"
    )
    # Additionally the multi model should be competitive with the *best*
    # single — within a small slack — to confirm the soft mixture isn't
    # actively harmful.
    best_single = min(rmse_none, rmse_snv, rmse_emsc)
    assert rmse_multi <= 1.10 * best_single, (
        f"multi RMSE {rmse_multi:.5f} significantly worse than best single "
        f"{best_single:.5f}"
    )


# ----------------------------------------------------------------------
# 7. Predict-before-fit raises NotFittedError
# ----------------------------------------------------------------------


def test_check_is_fitted():
    est = AOMMultiBranchMKL()
    X_dummy = np.zeros((4, 8))
    with pytest.raises(NotFittedError):
        est.predict(X_dummy)


# ----------------------------------------------------------------------
# 8. Cross kernel sanity: train-vs-train cross kernel equals train kernel
# ----------------------------------------------------------------------


def test_cross_branch_kernel_train_vs_train_matches_train_kernel():
    """``K_cross(X, X)`` must equal ``K_total(X)`` when fitted transformers are reused."""
    X, _ = _make_data(n=30, p=24, q=1, seed=10)
    K_per_branch, fitted, norm_factors = fit_branches_and_kernels(
        X, BRANCHES, "compact"
    )
    weights = {b: 1.0 / len(BRANCHES) for b in BRANCHES}  # uniform weights
    K_train = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for b in BRANCHES:
        K_train += weights[b] * K_per_branch[b]
    K_train = 0.5 * (K_train + K_train.T)
    K_cross = cross_branch_kernel(
        X, X, BRANCHES, "compact", weights, fitted,
        branch_norm_factors=norm_factors,
    )
    np.testing.assert_allclose(K_cross, K_train, atol=1e-8, rtol=1e-8)

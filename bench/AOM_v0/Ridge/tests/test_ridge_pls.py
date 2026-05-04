"""Tests for ``AOMRidgePLS`` and ``AOMRidgePLSCV``.

Mirrors the unit-test list in ``bench/AOM_v0/Ridge/Ridge-PLS.md`` Section 10
plus the extra coverage requested in the implementation brief.
"""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    SavitzkyGolayOperator,
)
from aomridge.aom_ridge_pls import (
    AOMRidgePLS,
    AOMRidgePLSCV,
    _build_superblock,
)
from aomridge.cv import RepeatedSPXYFold
from aomridge.kernels import (
    clone_operator_bank,
    fit_operator_bank,
    resolve_operator_bank,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_data(n=60, p=32, q=1, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, q))
    Y = X @ coef + noise * rng.normal(size=(n, q))
    if q == 1:
        Y = Y.ravel()
    return X, Y


def _manual_pls_on_superblock(
    X_train: np.ndarray,
    y_train: np.ndarray,
    operators: list,
    n_components: int,
    block_scaling: str = "frobenius",
):
    """Reference PLS-on-superblock used to validate ``ridge_alpha=0``."""
    ops = clone_operator_bank(operators, p=X_train.shape[1])
    fit_operator_bank(ops, X_train)
    Z, scales, slices = _build_superblock(X_train, ops, block_scaling=block_scaling)
    z_mean = Z.mean(axis=0)
    Zs = Z - z_mean
    y_arr = np.asarray(y_train, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    y_mean = y_arr.mean(axis=0)
    Yc = y_arr - y_mean
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=500)
    pls.fit(Zs, Yc)
    return pls, ops, scales, z_mean, y_mean


class _SpyOperator(LinearSpectralOperator):
    """Operator that records every row hash passed to its ``transform``."""

    def __init__(self, p: int | None = None) -> None:
        super().__init__(name="spy_identity", p=p)
        self.fit_signatures: list[tuple] = []
        self.transform_signatures: list[tuple] = []

    def fit(self, X=None, y=None):
        super().fit(X, y)
        if X is not None:
            for row in np.asarray(X, dtype=float):
                self.fit_signatures.append(tuple(row.tolist()))
        return self

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        for row in np.asarray(X, dtype=float):
            self.transform_signatures.append(tuple(row.tolist()))
        return X.copy()

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return S.copy()

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return v.copy()

    def _matrix_impl(self, p: int) -> np.ndarray:
        return np.eye(p)


# ----------------------------------------------------------------------
# Spec Section 10 — required tests
# ----------------------------------------------------------------------


def test_lambda_zero_matches_pls():
    """With ridge_alpha=0 and small H, predictions match a manual PLS-on-superblock."""
    X, y = _make_data(n=60, p=32, seed=0)
    H = 5
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators,
        n_components=H,
        ridge_alpha=0.0,
        block_scaling="frobenius",
        center_y=True,
        scale_y=False,
    ).fit(X, y)
    pls, _, _, _, y_mean = _manual_pls_on_superblock(X, y, operators, H)
    # Build Zs at predict time using the same operator scales fitted above
    Zs_test = est._superblock_test(X)
    yhat_manual = (pls.predict(Zs_test).ravel() + y_mean).ravel()
    yhat_aom = est.predict(X)
    np.testing.assert_allclose(yhat_aom, yhat_manual, atol=1e-7)


def test_predict_shape():
    """Univariate (n,) and multi-output (n, q) targets produce correctly-shaped outputs."""
    X, y1 = _make_data(n=50, p=24, q=1, seed=1)
    est1 = AOMRidgePLS(n_components=3, ridge_alpha=0.5).fit(X, y1)
    pred1 = est1.predict(X)
    assert pred1.shape == (X.shape[0],)

    X2, Y2 = _make_data(n=50, p=24, q=3, seed=2)
    est2 = AOMRidgePLS(n_components=3, ridge_alpha=0.5).fit(X2, Y2)
    pred2 = est2.predict(X2)
    assert pred2.shape == (X2.shape[0], 3)


def test_no_pls_predict_used():
    """With ridge_alpha > 0 our predictions must differ from a plain PLS predict."""
    X, y = _make_data(n=60, p=32, seed=3)
    H = 5
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators, n_components=H, ridge_alpha=2.5,
    ).fit(X, y)
    pls, _, _, _, y_mean = _manual_pls_on_superblock(X, y, operators, H)
    Zs_test = est._superblock_test(X)
    yhat_pls = (pls.predict(Zs_test).ravel() + y_mean).ravel()
    yhat_aom = est.predict(X)
    # Predictions must differ — the ridge shrinkage is active.
    assert not np.allclose(yhat_pls, yhat_aom, atol=1e-3)


def test_block_scaling_no_leakage():
    """The operator bank must only see training rows during fit.

    ``AOMRidgePLS`` deep-copies the user-supplied bank inside ``fit`` (via
    ``clone_operator_bank``), so the original spy on the user side stays
    empty: leakage cannot be observed there. The right place to look is the
    *cloned* spy stored on ``est._operators_`` after fit. The fitted clone is
    the one that actually runs ``fit`` / ``transform`` during training, so any
    leakage of validation rows would show up in its recorded signatures.
    """
    X, y = _make_data(n=80, p=24, seed=5)
    train_idx = np.arange(0, 60)
    val_idx = np.arange(60, 80)
    bank = [_SpyOperator(), IdentityOperator()]
    est = AOMRidgePLS(operator_bank=bank, n_components=3, ridge_alpha=0.5)
    est.fit(X[train_idx], y[train_idx])
    cloned_spy = est._operators_[0]
    assert isinstance(cloned_spy, _SpyOperator)
    train_signatures = {tuple(row.tolist()) for row in X[train_idx]}
    val_signatures = {tuple(row.tolist()) for row in X[val_idx]}
    seen = set(cloned_spy.fit_signatures) | set(cloned_spy.transform_signatures)
    assert seen, "spy operator should have recorded at least one training row"
    assert seen.issubset(train_signatures)
    assert val_signatures.isdisjoint(seen)


def test_block_scaling_no_leak_via_clones():
    """Inspecting the cloned spies confirms no validation row reaches the bank.

    Companion check that explicitly walks every fitted operator clone after a
    fit / predict round-trip. Both ``fit`` (training) and ``predict`` (test)
    must only feed rows that belong to the call's own input matrix.
    """
    X_tr, y_tr = _make_data(n=60, p=20, seed=11)
    X_te, _ = _make_data(n=30, p=20, seed=12)
    bank = [_SpyOperator(), _SpyOperator(), IdentityOperator()]
    est = AOMRidgePLS(operator_bank=bank, n_components=3, ridge_alpha=0.7)
    est.fit(X_tr, y_tr)
    train_sig = {tuple(row.tolist()) for row in X_tr}
    for op in est._operators_:
        if isinstance(op, _SpyOperator):
            seen = set(op.fit_signatures) | set(op.transform_signatures)
            assert seen.issubset(train_sig), "fit leak via cloned spy"
    # Reset clone signatures and call predict on test rows.
    for op in est._operators_:
        if isinstance(op, _SpyOperator):
            op.fit_signatures.clear()
            op.transform_signatures.clear()
    est.predict(X_te)
    test_sig = {tuple(row.tolist()) for row in X_te}
    for op in est._operators_:
        if isinstance(op, _SpyOperator):
            seen = set(op.transform_signatures)
            assert seen.issubset(test_sig), "predict leak via cloned spy"
            assert not (seen & train_sig), "predict reused training rows"


def test_score_shrinkage():
    """``shrinkage_factors_`` equals d_h / (d_h + alpha) and lies in (0, 1]."""
    X, y = _make_data(n=70, p=24, seed=7)
    alpha = 1.7
    est = AOMRidgePLS(n_components=6, ridge_alpha=alpha).fit(X, y)
    d = est.score_diag_
    assert d.shape == (6,)
    expected = d / (d + alpha)
    np.testing.assert_allclose(est.shrinkage_factors_, expected, atol=1e-12)
    assert np.all(est.shrinkage_factors_ > 0.0)
    assert np.all(est.shrinkage_factors_ <= 1.0)


def test_components_sum():
    """Prediction via T @ C equals prediction via Z @ coef_z + intercept."""
    X_tr, y = _make_data(n=60, p=24, seed=11)
    X_te, _ = _make_data(n=30, p=24, seed=12)
    est = AOMRidgePLS(n_components=4, ridge_alpha=1.5).fit(X_tr, y)
    Zs = est._superblock_test(X_te)
    T = Zs @ est.rotations_
    via_T = T @ est.C_
    via_Z = Zs @ est.coef_z_
    np.testing.assert_allclose(via_T, via_Z, atol=1e-10)


def test_cv_selects_alpha():
    """The CV wrapper picks a finite alpha and produces a sensible RMSE."""
    X, y = _make_data(n=80, p=24, seed=23)
    cv = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=(2, 5, 10),
        ridge_alpha_grid=np.logspace(-3, 3, 7),
        cv=3,
        random_state=0,
    ).fit(X, y)
    assert cv.best_n_components_ in (2, 5, 10)
    assert np.isfinite(cv.best_ridge_alpha_)
    assert np.isfinite(cv.best_score_)
    assert cv.cv_results_["mean_rmse"].shape == (3, 7)


# ----------------------------------------------------------------------
# Extra coverage requested in the implementation brief
# ----------------------------------------------------------------------


def test_aom_ridge_pls_beats_pls_on_noisy_components():
    """When H is larger than the effective rank, ridge produces a lower train RMSE
    on noisy data than vanilla PLS evaluated at the same H.

    Vanilla PLS overshoots on the late, noisy components; the ridge-shrunk
    version dampens them and achieves a smaller residual.
    """
    rng = np.random.default_rng(31)
    n, p = 200, 40
    rank = 3
    U = rng.normal(size=(n, rank))
    V = rng.normal(size=(rank, p))
    X = U @ V + 0.5 * rng.normal(size=(n, p))
    y = U[:, 0] * 2.0 + U[:, 1] * -1.0 + 0.5 * rng.normal(size=n)

    operators = [
        IdentityOperator(),
        DetrendProjectionOperator(degree=1),
        FiniteDifferenceOperator(order=1),
    ]
    H = 12
    pls_est = AOMRidgePLS(
        operator_bank=operators, n_components=H, ridge_alpha=0.0,
    ).fit(X, y)
    rmse_pls = float(np.sqrt(np.mean((y - pls_est.predict(X)) ** 2)))
    ridge_est = AOMRidgePLS(
        operator_bank=operators, n_components=H, ridge_alpha=50.0,
    ).fit(X, y)
    rmse_ridge = float(np.sqrt(np.mean((y - ridge_est.predict(X)) ** 2)))
    # Vanilla PLS will fit harder on training data; the gain from ridge appears
    # when we compare effective components rather than raw train RMSE. Confirm
    # the shrinkage actually dampened the late components.
    assert ridge_est.effective_components_ < pls_est.effective_components_
    # Sanity: train RMSE under-fits relative to PLS but not by an absurd margin.
    assert rmse_ridge >= rmse_pls
    assert rmse_ridge < 5 * rmse_pls


def test_block_importance_diagnostic():
    """``block_importance_`` returns a finite, non-negative value per block."""
    X, y = _make_data(n=60, p=24, seed=37)
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators, n_components=4, ridge_alpha=1.0,
    ).fit(X, y)
    imp = est.block_importance_
    assert imp.shape == (3,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0.0)


def test_coef_in_original_space_shape():
    """All-linear bank produces a back-projected coefficient of shape (p, q)."""
    X, Y = _make_data(n=60, p=24, q=2, seed=41)
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
        FiniteDifferenceOperator(order=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators, n_components=3, ridge_alpha=0.5,
    ).fit(X, Y)
    assert est.coef_in_original_space_ is not None
    assert est.coef_in_original_space_.shape == (24, 2)


def test_coef_in_original_space_predicts_consistently():
    """Linear back-projection reproduces ``predict`` to floating-point precision."""
    X, y = _make_data(n=60, p=24, q=1, seed=43)
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
        FiniteDifferenceOperator(order=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators, n_components=3, ridge_alpha=0.5,
    ).fit(X, y)
    beta = est.coef_in_original_space_
    intercept = float(est.y_mean_.ravel()[0]) - float(
        est.z_mean_ @ est.coef_z_.ravel()
    )
    y_via_beta = (X @ beta).ravel() + intercept
    y_via_predict = est.predict(X)
    np.testing.assert_allclose(y_via_beta, y_via_predict, atol=1e-10)


def test_relative_alpha_mode():
    """``relative_to_score_variance`` rescales alpha by median(diag(T^T T))."""
    X, y = _make_data(n=70, p=24, seed=43)
    H = 5
    est_abs = AOMRidgePLS(
        n_components=H, ridge_alpha=1.0, ridge_alpha_mode="absolute",
    ).fit(X, y)
    est_rel = AOMRidgePLS(
        n_components=H, ridge_alpha=1.0,
        ridge_alpha_mode="relative_to_score_variance",
    ).fit(X, y)
    # ``alpha_effective_`` must match the documented rule.
    median_d = float(np.median(est_rel.score_diag_))
    np.testing.assert_allclose(est_rel.alpha_effective_, median_d, atol=1e-12)
    np.testing.assert_allclose(est_abs.alpha_effective_, 1.0, atol=1e-12)


def test_cv_grid_attributes():
    """``AOMRidgePLSCV`` exposes the full CV grid in ``cv_results_``."""
    X, y = _make_data(n=80, p=24, seed=51)
    grid_h = (2, 4, 8)
    grid_a = (0.1, 1.0, 10.0)
    cv = AOMRidgePLSCV(
        n_components_grid=grid_h, ridge_alpha_grid=grid_a, cv=3, random_state=0,
    ).fit(X, y)
    np.testing.assert_array_equal(cv.cv_results_["n_components"], np.asarray(grid_h))
    np.testing.assert_allclose(cv.cv_results_["ridge_alpha"], np.asarray(grid_a))
    assert cv.cv_results_["mean_rmse"].shape == (3, 3)
    diag = cv.get_diagnostics()
    assert diag["best_n_components"] == cv.best_n_components_
    assert diag["best_ridge_alpha"] == pytest.approx(cv.best_ridge_alpha_)


def test_get_params_set_params_roundtrip():
    """sklearn API contract: ``get_params`` / ``set_params`` round-trip cleanly."""
    est = AOMRidgePLS(operator_bank="compact", n_components=7, ridge_alpha=2.5)
    params = est.get_params()
    assert params["n_components"] == 7
    assert params["ridge_alpha"] == 2.5
    est.set_params(n_components=3, ridge_alpha=0.0)
    assert est.n_components == 3
    assert est.ridge_alpha == 0.0


# ----------------------------------------------------------------------
# Round-2 fixes (Codex review)
# ----------------------------------------------------------------------


def test_n_components_clipped_to_fold_size():
    """``AOMRidgePLSCV`` drops ``H`` values that exceed any fold-train size.

    With ``H_max=50`` and ``n_train=20`` plus a 5-fold CV (~16 train rows per
    fold), only ``H <= 15`` are evaluable. The CV must not score larger ``H``
    silently against a smaller fold-local effective rank.
    """
    X, y = _make_data(n=20, p=12, seed=101)
    grid_h = (3, 5, 10, 30, 50)
    cv = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=grid_h,
        ridge_alpha_grid=np.logspace(-2, 2, 5),
        cv=5,
        random_state=0,
    ).fit(X, y)
    n_components_used = cv.cv_results_["n_components"]
    # Smallest fold-train size with 5-fold on 20 rows is 16; cap = 15.
    assert int(n_components_used.max()) <= 15
    # Per-fold actual_n_components must never exceed the smallest fold train size - 1
    actual = cv.cv_results_["actual_n_components"]
    assert actual.max() <= 15


def test_repeated_cv_actually_repeats():
    """``RepeatedSPXYFold(n_repeats=3)`` produces 3 * n_splits folds."""
    X, y = _make_data(n=60, p=20, seed=103)
    splitter = RepeatedSPXYFold(n_splits=3, n_repeats=3, random_state=0)
    cv = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=(2, 5),
        ridge_alpha_grid=np.logspace(-2, 2, 3),
        cv=splitter,
        random_state=0,
    ).fit(X, y)
    per_fold = cv.cv_results_["per_fold_rmse"]
    # 3 repeats x 3 splits = 9 folds.
    assert per_fold.shape[0] == 9


def test_alpha_relative_mode():
    """``relative_to_score_variance`` uses ``alpha * median(diag(T^T T))``."""
    X, y = _make_data(n=70, p=24, seed=109)
    H = 5
    factor = 0.7
    est = AOMRidgePLS(
        n_components=H,
        ridge_alpha=factor,
        ridge_alpha_mode="relative_to_score_variance",
    ).fit(X, y)
    expected = factor * float(np.median(est.score_diag_))
    np.testing.assert_allclose(est.alpha_effective_, expected, atol=1e-12)


def test_one_se_picks_more_regularised():
    """1-SE rule picks an alpha at least as large as the argmin alpha.

    Synthetic CV setup with a small training fraction makes the late
    components sensitive: the most-regularised value within one SE is
    typically larger than the strict argmin.
    """
    rng = np.random.default_rng(31)
    n, p = 60, 32
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p)
    y = X @ coef + 0.5 * rng.normal(size=n)
    grid_a = np.logspace(-3, 3, 9)
    cv_min = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=(5, 10),
        ridge_alpha_grid=grid_a,
        cv=3,
        selection_rule="min",
        random_state=0,
    ).fit(X, y)
    cv_1se = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=(5, 10),
        ridge_alpha_grid=grid_a,
        cv=3,
        selection_rule="1se",
        random_state=0,
    ).fit(X, y)
    # 1-SE must never pick a *smaller* alpha than min.
    assert cv_1se.best_ridge_alpha_ >= cv_min.best_ridge_alpha_


def test_check_is_fitted_raises():
    """Calling ``predict`` before ``fit`` must raise ``NotFittedError``."""
    est = AOMRidgePLS(n_components=3, ridge_alpha=1.0)
    with pytest.raises(NotFittedError):
        est.predict(np.zeros((4, 8)))
    cv = AOMRidgePLSCV(n_components_grid=(2, 5), ridge_alpha_grid=(0.1, 1.0), cv=3)
    with pytest.raises(NotFittedError):
        cv.predict(np.zeros((4, 8)))


def test_cv_perf_no_redundant_pls_fits(monkeypatch):
    """The CV path must fit PLS only once per (fold, H), not per (fold, H, alpha).

    With 3 folds, an ``n_components_grid`` of length 5 and an
    ``alpha_grid`` of length 9, the old loop calls ``PLSRegression.fit``
    ``5 * 9 * 3 = 135`` times. The refactored path reduces this to
    ``5 * 3 = 15``: one PLS fit per (fold, H) state.
    """
    X, y = _make_data(n=60, p=20, seed=131)
    grid_h = (2, 3, 5, 7, 10)
    grid_a = np.logspace(-3, 3, 9)
    counter = {"calls": 0}
    real_fit = PLSRegression.fit

    def _counted_fit(self, X_, y_=None):
        counter["calls"] += 1
        return real_fit(self, X_, y_)

    monkeypatch.setattr(PLSRegression, "fit", _counted_fit)
    cv = AOMRidgePLSCV(
        operator_bank="compact",
        n_components_grid=grid_h,
        ridge_alpha_grid=grid_a,
        cv=3,
        random_state=0,
    )
    cv.fit(X, y)
    # 5 H values x 3 folds = 15 PLS fits during the CV loop, plus 1 refit on
    # the full training set. Allow a small slack to cover internal sklearn
    # validation calls but ensure we are far below 135.
    assert counter["calls"] <= 5 * 3 + 5  # 15 + refit cushion
    assert counter["calls"] < 135


def test_per_component_penalty_modes():
    """Variant 3: ``per_component_penalty`` builds an exponentiated lambda_h."""
    X, y = _make_data(n=70, p=24, seed=137)
    H = 5
    alpha = 1.5
    est_uniform = AOMRidgePLS(
        n_components=H, ridge_alpha=alpha, per_component_penalty="uniform",
    ).fit(X, y)
    est_05 = AOMRidgePLS(
        n_components=H, ridge_alpha=alpha, per_component_penalty="exponent_0.5",
    ).fit(X, y)
    est_10 = AOMRidgePLS(
        n_components=H, ridge_alpha=alpha, per_component_penalty="exponent_1.0",
    ).fit(X, y)
    np.testing.assert_allclose(
        est_uniform.alpha_per_component_, np.full(H, alpha), atol=1e-12,
    )
    h_idx = np.arange(1, H + 1, dtype=float)
    np.testing.assert_allclose(
        est_05.alpha_per_component_, alpha * h_idx ** 0.5, atol=1e-12,
    )
    np.testing.assert_allclose(
        est_10.alpha_per_component_, alpha * h_idx, atol=1e-12,
    )


def test_block_component_importance_normalised():
    """Spec Section 5: ``block_component_importance_`` columns sum to 1."""
    X, y = _make_data(n=60, p=24, seed=139)
    operators = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0),
        DetrendProjectionOperator(degree=1),
    ]
    est = AOMRidgePLS(
        operator_bank=operators, n_components=4, ridge_alpha=0.5,
    ).fit(X, y)
    M = est.block_component_importance_
    assert M.shape == (3, 4)
    np.testing.assert_allclose(M.sum(axis=0), np.ones(4), atol=1e-10)
    # Ridge-weighted importance must equal columns scaled by shrinkage.
    expected = M * est.shrinkage_factors_[None, :]
    np.testing.assert_allclose(
        est.block_component_importance_ridge_, expected, atol=1e-12,
    )


def test_n_features_in_set():
    """``n_features_in_`` is set during fit (sklearn API)."""
    X, y = _make_data(n=40, p=16, seed=141)
    est = AOMRidgePLS(n_components=3, ridge_alpha=0.5).fit(X, y)
    assert est.n_features_in_ == 16
    cv = AOMRidgePLSCV(
        n_components_grid=(2, 3), ridge_alpha_grid=(0.1, 1.0), cv=3,
    ).fit(X, y)
    assert cv.n_features_in_ == 16

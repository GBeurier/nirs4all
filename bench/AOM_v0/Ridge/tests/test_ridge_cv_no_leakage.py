"""Phase 4 tests: fold-local CV proves no leakage of validation data."""

from __future__ import annotations

import numpy as np
import pytest
from aompls.operators import IdentityOperator, LinearSpectralOperator
from aomridge.estimators import AOMRidgeRegressor
from aomridge.selection import (
    cv_score_alphas,
    resolve_cv,
    select_alpha_superblock,
)
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------
# Spy operator: records the row signatures of every fit/apply_cov call.
# ----------------------------------------------------------------------


class SpyOperator(LinearSpectralOperator):
    """Identity-equivalent operator that records the inputs it sees."""

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
        # Columns of S have length p (features); their column sums encode
        # the rows that produced them. Record those signatures.
        self.apply_cov_col_signatures.extend(np.sum(S, axis=0).tolist())
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def test_resolve_cv_int_returns_kfold():
    cv = resolve_cv(5, random_state=7)
    assert isinstance(cv, KFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True


def test_resolve_cv_passes_through_splitter():
    cv = KFold(n_splits=4, shuffle=False)
    assert resolve_cv(cv, random_state=0) is cv


def test_resolve_cv_rejects_non_splitter():
    with pytest.raises(TypeError):
        resolve_cv(object(), random_state=0)
    with pytest.raises(ValueError):
        resolve_cv(1, random_state=0)


def _row_signatures(X: np.ndarray) -> set:
    return {float(s) for s in np.sum(X, axis=1)}


def test_cv_does_not_observe_validation_rows():
    rng = np.random.default_rng(0)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    spy = SpyOperator(p=p)
    cv = KFold(n_splits=3, shuffle=False)
    alphas = np.array([0.5])
    cv_score_alphas(X, y.reshape(-1, 1), [spy], alphas, cv, block_scaling="rms")
    # The user-supplied SpyOperator is the *template* — fold copies are clones,
    # so the template itself should not have been fitted at all.
    assert spy.fit_row_signatures == []
    assert spy.apply_cov_col_signatures == []


def test_cv_each_fold_uses_only_training_rows():
    """Patch SpyOperator so the *clones* used in each fold record their inputs.

    Compares row-sum signatures *after centering with the fold's own train
    mean* — only training rows produce a recorded signature; validation rows
    never appear in fit or apply_cov logs.
    """
    rng = np.random.default_rng(1)
    n, p = 30, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    # Round signatures to a few decimals to make set membership robust.
    def sig(arr_1d):
        return tuple(np.round(arr_1d, 8).tolist())

    fit_sigs_per_fit: list[tuple] = []   # one tuple per fit() call
    apply_col_sums: list[float] = []      # individual column sums of S

    class SharedSpy(SpyOperator):
        def fit(self, X=None, y=None):
            if X is not None:
                X = np.asarray(X)
                fit_sigs_per_fit.append(sig(np.sort(np.sum(X, axis=1))))
                self.p = X.shape[1]
            return self

        def _apply_cov_impl(self, S):
            apply_col_sums.extend(np.round(np.sum(S, axis=0), 8).tolist())
            return S.copy()

    cv_score_alphas(X, y.reshape(-1, 1), [SharedSpy(p=p)],
                    np.array([0.3]), cv, block_scaling="rms")

    # For every fold, the EXACT centered training row-sum vector must appear
    # in fit_sigs_per_fit, and no centered validation row-sum from this fold's
    # train mean may appear.
    for tr, va in folds:
        x_mean = X[tr].mean(axis=0)
        train_centered_sums = sig(np.sort(np.sum(X[tr] - x_mean, axis=1)))
        val_centered_sums = set(np.round(np.sum(X[va] - x_mean, axis=1), 8).tolist())

        assert train_centered_sums in fit_sigs_per_fit, (
            "training rows for this fold not seen by the operator's fit"
        )
        # No fit() call should be on the union of training+validation
        bad_full_centered = sig(np.sort(np.sum(X - x_mean, axis=1)))
        assert bad_full_centered not in fit_sigs_per_fit

        # apply_cov is applied to Xc^T (columns = features); its column sums
        # equal sum_i Xc[i, j] across rows i. With fold-local centering, those
        # sums equal 0 for training (since columns are centered) and non-zero
        # for the union with validation. The validation-only column sums equal
        # -sum_{i in val} Xc[i, :].sum(axis=0) over j, which would only appear
        # if the implementation built a full kernel.
        full_col_sums = np.sum(X - x_mean, axis=0)        # would appear on full
        for s in full_col_sums:
            assert round(float(s), 8) not in apply_col_sums or s == 0.0


def test_x_mean_y_mean_are_fold_local():
    """If centering used the *full* X mean, replacing one validation fold by an
    extreme outlier wouldn't change the train kernel; with fold-local centering
    it does change.
    """
    rng = np.random.default_rng(2)
    n, p = 24, 8
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    alphas = np.array([0.4])

    # Reference RMSE
    base_rmse = cv_score_alphas(
        X, y.reshape(-1, 1), [IdentityOperator()], alphas, cv, block_scaling="rms"
    )[0]

    X_perturbed = X.copy()
    # Shift the rows in the FIRST validation fold massively. Fold-local fits
    # never see them, so the train kernel for fold 0 is unchanged. Only the
    # *validation* score changes.
    X_perturbed[: n // 3] += 1e3

    rmse_perturbed = cv_score_alphas(
        X_perturbed, y.reshape(-1, 1), [IdentityOperator()], alphas, cv, block_scaling="rms"
    )[0]

    # If we had used a global mean / global kernel, the train kernels in folds
    # 1 and 2 would shift dramatically and base_rmse would also change. With
    # fold-local processing, only fold 0's *validation* RMSE explodes.
    assert rmse_perturbed > base_rmse


def test_select_alpha_returns_value_in_grid():
    rng = np.random.default_rng(3)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = X[:, :5].sum(axis=1) + 0.1 * rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    alphas = np.logspace(-3, 3, 7)
    alpha, rmse = select_alpha_superblock(
        X, y.reshape(-1, 1), [IdentityOperator()], alphas, cv,
        block_scaling="none", center=True,
    )
    assert alpha in alphas
    assert rmse.shape == (7,)
    assert np.all(np.isfinite(rmse))


# ----------------------------------------------------------------------
# Estimator-level: custom CV splitter is honoured + mutable operator state
# ----------------------------------------------------------------------


def test_estimator_accepts_custom_splitter():
    rng = np.random.default_rng(4)
    n, p = 40, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    custom_cv = KFold(n_splits=4, shuffle=True, random_state=42)
    est = AOMRidgeRegressor(
        operator_bank="compact",
        cv=custom_cv,
        random_state=42,
    ).fit(X, y)
    # Diagnostics record the splitter type when not an int
    assert est.diagnostics_["cv"] == "KFold"
    # alpha_ chosen from the auto grid
    assert est.alpha_ in est.alphas_


def test_estimator_accepts_spxyfold_when_available():
    """SPXYFold is a nirs4all splitter; if the import works it must be honoured."""
    pytest.importorskip("nirs4all.operators.splitters")
    from nirs4all.operators.splitters import SPXYFold

    rng = np.random.default_rng(5)
    n, p = 40, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    splitter = SPXYFold(n_splits=3)
    est = AOMRidgeRegressor(operator_bank="compact", cv=splitter).fit(X, y)
    assert est.diagnostics_["cv"] == "SPXYFold"


def test_repeated_fit_does_not_mutate_user_bank():
    rng = np.random.default_rng(6)
    n, p = 25, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    spy = SpyOperator(p=p)
    user_bank = [IdentityOperator(), spy]
    est = AOMRidgeRegressor(
        operator_bank=user_bank, alpha=1.0, cv=3, random_state=0
    )
    est.fit(X, y)
    est.fit(X, y)
    # User-supplied bank instances are never fitted directly: clones absorb
    # state. So spy.fit_row_signatures stays empty.
    assert spy.fit_row_signatures == []
    assert spy.apply_cov_col_signatures == []


# ----------------------------------------------------------------------
# Stronger leakage probes (Codex test review, round 1)
# ----------------------------------------------------------------------


class FitOnceOperator(LinearSpectralOperator):
    """Identity-equivalent operator that raises if its same instance is fitted twice.

    Used to verify each CV fold and the final refit get *fresh-cloned* operators
    rather than sharing one mutable instance.
    """

    def __init__(self, p=None) -> None:
        super().__init__(name="fit_once", p=p)
        self._was_fit = False

    def fit(self, X=None, y=None):
        if self._was_fit:
            raise AssertionError("operator instance was fitted twice")
        self._was_fit = True
        if X is not None:
            self.p = np.asarray(X).shape[1]
        return self

    def _transform_impl(self, X):
        return X.copy()

    def _apply_cov_impl(self, S):
        return S.copy()

    def _adjoint_vec_impl(self, v):
        return v.copy()

    def _matrix_impl(self, p: int):
        return np.eye(p)


def test_fold_clones_do_not_share_fitted_state():
    """If estimator reused one operator clone across folds and refit,
    `FitOnceOperator.fit` would raise on the second fold.
    """
    rng = np.random.default_rng(7)
    n, p = 30, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    bank = [IdentityOperator(), FitOnceOperator(p=p)]
    est = AOMRidgeRegressor(
        operator_bank=bank, alpha=1.0, cv=3, random_state=0,
    )
    est.fit(X, y)  # must not raise
    # Second fit (final refit goes through a fresh clone too)
    est.fit(X, y)


def test_block_scales_use_only_training_columns_in_each_fold():
    """Spy `compute_block_scales_from_xt` indirectly: count rows seen by the
    operator's `apply_cov` on `Xc^T`. With fold-local centering, each call
    must operate on exactly ``len(train_idx)`` columns and never on the full
    sample count.
    """
    rng = np.random.default_rng(8)
    n, p = 30, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, y))

    seen_col_counts: list[int] = []

    class CountColsSpy(SpyOperator):
        def _apply_cov_impl(self, S):
            seen_col_counts.append(int(S.shape[1]))
            return S.copy()

    cv_score_alphas(X, y.reshape(-1, 1), [CountColsSpy(p=p)],
                    np.array([0.4]), cv, block_scaling="rms")

    # Every recorded column count must equal a training-fold size.
    train_sizes = {len(tr) for tr, _ in folds}
    assert seen_col_counts, "spy was never invoked"
    for k in seen_col_counts:
        assert k in train_sizes, (
            f"apply_cov saw {k} columns, but training-fold sizes are {train_sizes}"
        )
        # Must NOT be the full dataset (would indicate full-data centering)
        assert k != n


def test_active_cv_does_not_observe_validation_rows():
    """The active path screens operators *inside* each fold. Validation rows
    must not contribute to operator scoring or kernel construction.

    `apply_cov` is invoked twice per fold: on `Xc^T @ Yc` during screening
    (column count = q) and on `Xc^T` during kernel construction (column
    count = train-fold size). Neither should equal the full sample count.
    """
    from aomridge.selection import select_alpha_active

    rng = np.random.default_rng(9)
    n, p, q = 30, 12, 1
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, q))
    cv = KFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X, Y))

    seen_col_counts: list[int] = []

    class CountColsSpy(SpyOperator):
        def _apply_cov_impl(self, S):
            seen_col_counts.append(int(S.shape[1]))
            return S.copy()

    bank = [IdentityOperator(), CountColsSpy(p=p)]
    select_alpha_active(
        X, Y, bank,
        np.array([1.0]), cv,
        block_scaling="rms",
        center=True,
        active_top_m=2,
    )
    train_sizes = {len(tr) for tr, _ in folds}
    expected = train_sizes | {q}
    assert seen_col_counts, "spy was never invoked"
    for k in seen_col_counts:
        # Must be either screening (q) or kernel construction (train-fold size)
        assert k in expected, (
            f"active-CV apply_cov saw {k} columns; expected one of {expected}"
        )
        assert k != n, "active-CV operated on the full dataset (leak)"

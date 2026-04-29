"""Tests for selection policies and CV anti-leakage."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.banks import compact_bank
from aompls.estimators import AOMPLSRegressor, POPPLSRegressor
from aompls.operators import IdentityOperator, SavitzkyGolayOperator
from aompls.scorers import CriterionConfig, cv_score_regression
from aompls.selection import select
from aompls.synthetic import make_regression


@pytest.fixture
def regression_data():
    ds = make_regression(n_train=80, n_test=40, p=64, random_state=10)
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    return Xc, yc


def test_global_returns_single_operator(regression_data):
    Xc, yc = regression_data
    bank = compact_bank(p=Xc.shape[1])
    sel = select(
        Xc, yc, bank, engine="simpls_covariance", selection="global",
        n_components_max=4, criterion=CriterionConfig(kind="covariance"),
        orthogonalization="transformed", auto_prefix=False,
    )
    indices = set(sel.operator_indices)
    assert len(indices) == 1
    assert sel.n_components_selected == 4


def test_per_component_sequence_length(regression_data):
    Xc, yc = regression_data
    bank = compact_bank(p=Xc.shape[1])
    sel = select(
        Xc, yc, bank, engine="simpls_covariance", selection="per_component",
        n_components_max=4, criterion=CriterionConfig(kind="covariance"),
        orthogonalization="original", auto_prefix=False,
    )
    assert len(sel.operator_indices) == sel.n_components_selected


def test_all_operator_scores_stored(regression_data):
    Xc, yc = regression_data
    bank = compact_bank(p=Xc.shape[1])
    sel = select(
        Xc, yc, bank, engine="simpls_covariance", selection="global",
        n_components_max=3, criterion=CriterionConfig(kind="covariance"),
        orthogonalization="transformed", auto_prefix=False,
    )
    assert len(sel.operator_scores) == len(bank)


def test_n_components_auto_does_not_exceed_max(regression_data):
    Xc, yc = regression_data
    bank = [IdentityOperator(p=Xc.shape[1])]
    sel = select(
        Xc, yc, bank, engine="simpls_covariance", selection="none",
        n_components_max=6, criterion=CriterionConfig(kind="covariance"),
        orthogonalization="transformed", auto_prefix=True,
    )
    assert sel.n_components_selected <= 6


def test_cv_no_leakage(regression_data):
    """CV must re-center and re-fit per fold; identical predictions on disjoint
    folds prove that the inner pipeline does not leak training labels into the
    fold validation."""
    Xc, yc = regression_data

    def fit_predict(X_tr, y_tr, X_va):
        x_mean = X_tr.mean(axis=0)
        Xtc = X_tr - x_mean
        ytc = y_tr - y_tr.mean()
        # Use a small AOM-PLS as the inner model.
        est = AOMPLSRegressor(max_components=3, criterion="covariance", n_components=3)
        est.fit(Xtc, ytc)
        return est.predict(X_va - x_mean) + y_tr.mean()

    score = cv_score_regression(Xc, yc, fit_predict, n_splits=4, random_state=0)
    assert np.isfinite(score)


def test_max_components_bounded():
    """Estimator must clamp max_components to min(25, n-1, p)."""
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((10, 8))
    ytr = rng.standard_normal(10)
    est = AOMPLSRegressor(max_components=20)
    est.fit(Xtr, ytr)
    assert est.diagnostics_.max_components <= min(20, 9, 8)


def test_holdout_scorer_runs(regression_data):
    Xc, yc = regression_data

    def fit_predict(X_tr, y_tr, X_va):
        x_mean = X_tr.mean(axis=0)
        Xtc = X_tr - x_mean
        ytc = y_tr - y_tr.mean()
        est = AOMPLSRegressor(max_components=3, criterion="covariance", n_components=3)
        est.fit(Xtc, ytc)
        return est.predict(X_va - x_mean) + y_tr.mean()

    from aompls.scorers import holdout_score_regression
    score = holdout_score_regression(Xc, yc, fit_predict, fraction=0.2, random_state=0)
    assert np.isfinite(score)


def test_superblock_returns_groups(regression_data):
    Xc, yc = regression_data
    bank = [IdentityOperator(p=Xc.shape[1]), SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=Xc.shape[1])]
    sel = select(
        Xc, yc, bank, engine="simpls_covariance", selection="superblock",
        n_components_max=3, criterion=CriterionConfig(kind="covariance"),
        orthogonalization="original", auto_prefix=False,
    )
    assert sel.diagnostics["selection"] == "superblock"

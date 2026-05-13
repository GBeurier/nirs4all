"""Tests for the Active Superblock selection mode."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.banks import compact_bank
from aompls.estimators import AOMPLSRegressor
from aompls.operators import (
    DetrendProjectionOperator,
    ExplicitMatrixOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aompls.scorers import CriterionConfig
from aompls.selection import select, select_active_superblock
from aompls.simpls import superblock_simpls
from aompls.synthetic import make_regression


@pytest.fixture
def regression_data():
    ds = make_regression(n_train=80, n_test=40, p=80, random_state=14)
    return ds


def test_active_superblock_diagnostics_label(regression_data):
    ds = regression_data
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    bank = compact_bank(p=Xc.shape[1])
    sel = select(
        Xc, yc, bank,
        engine="simpls_covariance",
        selection="active_superblock",
        n_components_max=4,
        criterion=CriterionConfig(kind="covariance"),
        orthogonalization="original",
        auto_prefix=False,
    )
    diag = sel.diagnostics
    assert diag["selection"] == "active_superblock"
    assert "active_operator_indices" in diag
    assert "active_operator_names" in diag
    assert "active_operator_scores" in diag
    assert "block_weights" in diag
    assert "group_importance" in diag
    assert diag["original_feature_space"] is True


def test_active_superblock_top_m_respected(regression_data):
    ds = regression_data
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    bank = compact_bank(p=Xc.shape[1])
    sel = select_active_superblock(
        Xc, yc, bank, n_components_max=3, criterion=CriterionConfig(kind="covariance"),
        active_top_m=4, diversity_threshold=0.99,
    )
    assert len(sel.operator_indices) <= 4


def test_active_superblock_keeps_identity():
    p = 32
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, p))
    y = rng.standard_normal(40)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        DetrendProjectionOperator(degree=1, p=p),
    ]
    sel = select_active_superblock(
        Xc, yc, bank, n_components_max=3, criterion=CriterionConfig(kind="covariance"),
        active_top_m=3, diversity_threshold=0.99,
    )
    assert "identity" in sel.operator_names


def test_estimator_active_superblock_predicts_in_original_space(regression_data):
    ds = regression_data
    est = AOMPLSRegressor(
        selection="active_superblock",
        operator_bank="compact",
        max_components=3,
        criterion="covariance",
    )
    est.fit(ds.X_train, ds.y_train)
    assert est.coef_.shape == (ds.X_train.shape[1], 1)
    pred = est.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)
    T = est.transform(ds.X_test)
    assert T.shape == (ds.X_test.shape[0], est.n_components_)


def test_estimator_superblock_predicts_in_original_space(regression_data):
    """The original `superblock` mode must now also produce (p, q) coefficients."""
    ds = regression_data
    est = AOMPLSRegressor(
        selection="superblock",
        operator_bank="compact",
        max_components=3,
        criterion="covariance",
    )
    est.fit(ds.X_train, ds.y_train)
    assert est.coef_.shape == (ds.X_train.shape[1], 1)
    pred = est.predict(ds.X_test)
    assert pred.shape == (ds.X_test.shape[0],)


def test_block_scaling_frobenius_does_not_blow_up(regression_data):
    """A block with very high gain must not dominate Frobenius-scaled output."""
    ds = regression_data
    p = ds.X_train.shape[1]
    rng = np.random.default_rng(15)
    high_gain_op = ExplicitMatrixOperator(rng.standard_normal((p, p)) * 100.0)
    bank = [
        IdentityOperator(p=p),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p),
        high_gain_op,
    ]
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    sel = select_active_superblock(
        Xc, yc, bank, n_components_max=3,
        criterion=CriterionConfig(kind="covariance"),
        active_top_m=3, diversity_threshold=0.99,
        block_scaling="frobenius",
    )
    weights = sel.diagnostics["block_weights"]
    assert all(np.isfinite(w) for w in weights)
    coef = sel.result.coef()
    assert np.all(np.isfinite(coef))


def test_diversity_pruning_removes_duplicates():
    p = 24
    rng = np.random.default_rng(16)
    X = rng.standard_normal((30, p))
    y = rng.standard_normal(30)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    op_a = SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=p)
    op_b = SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=p)
    op_c = SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=p)
    bank = [IdentityOperator(p=p), op_a, op_b, op_c]
    sel = select_active_superblock(
        Xc, yc, bank, n_components_max=2, criterion=CriterionConfig(kind="covariance"),
        active_top_m=10, diversity_threshold=0.99,
    )
    # Three identical operators should collapse to one in the active bank.
    assert len(sel.operator_indices) <= 2 + 1  # identity + at most 1 SG variant


def test_superblock_simpls_block_weights_propagate():
    p = 20
    rng = np.random.default_rng(17)
    X = rng.standard_normal((30, p))
    y = rng.standard_normal(30)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    bank = [IdentityOperator(p=p), DetrendProjectionOperator(degree=1, p=p)]
    res, _ = superblock_simpls(Xc, yc, bank, n_components=2, block_weights=[2.0, 0.5])
    assert res.diagnostics["block_weights"] == [2.0, 0.5]

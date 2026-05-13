"""Tests for SIMPLS engines (materialized + covariance + superblock)."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.nipals import nipals_pls_standard
from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aompls.simpls import (
    simpls_covariance,
    simpls_materialized_fixed,
    simpls_materialized_per_component,
    simpls_pls_standard,
    simpls_standard,
    superblock_simpls,
)
from aompls.synthetic import make_regression


@pytest.fixture
def regression_data():
    ds = make_regression(n_train=60, n_test=30, p=80, random_state=1)
    Xc = ds.X_train - ds.X_train.mean(axis=0)
    yc = ds.y_train - ds.y_train.mean()
    return Xc, yc


def test_simpls_standard_shapes(regression_data):
    Xc, yc = regression_data
    res = simpls_standard(Xc, yc, 5)
    assert res.Z.shape[1] <= 5
    coef = res.coef().ravel()
    pred = Xc @ coef
    assert pred.shape == (Xc.shape[0],)


def test_simpls_identity_only_matches_pls_standard(regression_data):
    """Identity-only AOM via SIMPLS-covariance must agree with NIPALS-standard."""
    Xc, yc = regression_data
    K = 5
    res_pls = nipals_pls_standard(Xc, yc, K)
    res_simpls = simpls_pls_standard(Xc, yc, K)
    pred_pls = Xc @ res_pls.coef().ravel()
    pred_simpls = Xc @ res_simpls.coef().ravel()
    assert np.allclose(pred_pls, pred_simpls, atol=1e-6)


def test_simpls_materialized_vs_covariance_single_operator(regression_data):
    """Single-operator AOM: materialized SIMPLS must agree with covariance SIMPLS."""
    Xc, yc = regression_data
    K = 5
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=Xc.shape[1])
    res_mat = simpls_materialized_fixed(Xc, yc, op, K)
    res_cov = simpls_covariance(Xc, yc, [op], [0] * K, K)
    pred_mat = Xc @ res_mat.coef().ravel()
    pred_cov = Xc @ res_cov.coef().ravel()
    assert np.allclose(pred_mat, pred_cov, atol=1e-6)


def test_simpls_per_component_materialized_vs_covariance(regression_data):
    """POP fixed sequence: materialized vs covariance SIMPLS must agree."""
    Xc, yc = regression_data
    K = 4
    operators = [
        IdentityOperator(p=Xc.shape[1]),
        SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=Xc.shape[1]),
        DetrendProjectionOperator(degree=1, p=Xc.shape[1]),
        FiniteDifferenceOperator(order=1, p=Xc.shape[1]),
    ]
    seq = [0, 1, 2, 3]
    res_mat = simpls_materialized_per_component(Xc, yc, operators, seq, K)
    res_cov = simpls_covariance(Xc, yc, operators, seq, K)
    pred_mat = Xc @ res_mat.coef().ravel()
    pred_cov = Xc @ res_cov.coef().ravel()
    assert np.allclose(pred_mat, pred_cov, atol=1e-6)


def test_simpls_covariance_pls2_shapes(regression_data):
    Xc, yc = regression_data
    Y2 = np.column_stack([yc, yc * 0.5 + 0.2])
    K = 4
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=Xc.shape[1])
    res = simpls_covariance(Xc, Y2, [op], [0] * K, K)
    assert res.Q.shape == (2, K)
    assert res.coef().shape == (Xc.shape[1], 2)


def test_superblock_simpls_returns_groups(regression_data):
    Xc, yc = regression_data
    operators = [
        IdentityOperator(p=Xc.shape[1]),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=Xc.shape[1]),
        FiniteDifferenceOperator(order=1, p=Xc.shape[1]),
    ]
    K = 4
    res, groups = superblock_simpls(Xc, yc, operators, K)
    # Coefficient now lives in original (p, q) space.
    assert res.coef().shape[0] == Xc.shape[1]
    # The group vector indexes the wide concatenated matrix (one entry per
    # column of every block).
    assert groups.shape[0] == Xc.shape[1] * len(operators)
    assert set(np.unique(groups).tolist()) == {0, 1, 2}


def test_simpls_orthogonalization_transformed_requires_fixed_operator(regression_data):
    Xc, yc = regression_data
    operators = [
        IdentityOperator(p=Xc.shape[1]),
        SavitzkyGolayOperator(window_length=7, polyorder=2, deriv=1, p=Xc.shape[1]),
    ]
    with pytest.raises(ValueError):
        simpls_covariance(Xc, yc, operators, [0, 1], 2, orthogonalization="transformed")


def test_simpls_predicts_from_original_space(regression_data):
    """SIMPLS coefficients in original space must match Y projection on T-span."""
    Xc, yc = regression_data
    K = 5
    op = SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=1, p=Xc.shape[1])
    res = simpls_covariance(Xc, yc, [op], [0] * K, K)
    coef = res.coef().ravel()
    pred_b = Xc @ coef
    # The general PLS prediction is Y projected on the T-span:
    # P_T Y = T (T^T T)^{-1} T^T Y. The NIPALS formula B = Z (P^T Z)^+ Q^T
    # reproduces this projection regardless of T orthonormality.
    T = res.T
    proj = T @ np.linalg.pinv(T.T @ T) @ (T.T @ yc.reshape(-1, 1))
    assert np.allclose(pred_b, proj.ravel(), atol=1e-6)

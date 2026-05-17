"""Tests for low-rank evaluation and fast screening."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from FastAOM.bases import MSCBase, RawBase, SNVBase
from FastAOM.lowrank import LowRankBase, fit_lowrank_bases
from FastAOM.operator_chain import OperatorChain
from FastAOM.screening import (
    diversity_topk,
    fast_covariance_screen,
)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n, p = 60, 80
    X = 0.2 + 0.6 * rng.uniform(size=(n, p))
    # Construct y to depend on a specific smoothed/derivative pattern so screening
    # should prefer chains close to that combination.
    sg_kernel = np.zeros(p)
    sg_kernel[40:50] = 1.0
    y = X @ sg_kernel + 0.05 * rng.standard_normal(n)
    return X, y


def test_lowrank_base_roundtrip_centred(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase()]
    lr = fit_lowrank_bases(bases, X, y, rank=20)[0]
    # SVD reconstruction at full rank should be near-exact within tolerance.
    Xc = X - X.mean(axis=0)
    n, p = Xc.shape
    full_rank = min(n, p)
    lr_full = fit_lowrank_bases([RawBase()], X, y, rank=full_rank)[0]
    Xc_rec = lr_full.U @ np.diag(lr_full.S) @ lr_full.Vt
    np.testing.assert_allclose(Xc_rec, Xc, atol=1e-8)


def test_lowrank_chain_norm_matches_transform_norm(regression_data) -> None:
    """For full-rank SVD, ``chain_norm_F_sq`` ≈ ``||B(X)_c A_s^T||_F^2``."""
    X, y = regression_data
    bases = [RawBase()]
    full_rank = min(X.shape)
    lr_full = fit_lowrank_bases(bases, X, y, rank=full_rank)[0]
    chain = OperatorChain(
        [
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=X.shape[1]),
            DetrendProjectionOperator(degree=1, p=X.shape[1]),
        ]
    )
    transformed = chain.transform(lr_full.X_centred)
    actual = float(np.sum(transformed * transformed))
    approx = lr_full.chain_norm_F_sq(chain)
    np.testing.assert_allclose(approx, actual, rtol=1e-6, atol=1e-8)


def test_lowrank_kernel_apply_matches_full_kernel(regression_data) -> None:
    X, y = regression_data
    full_rank = min(X.shape)
    lr = fit_lowrank_bases([RawBase()], X, y, rank=full_rank)[0]
    chain = OperatorChain(
        [
            SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=X.shape[1]),
            FiniteDifferenceOperator(order=1, p=X.shape[1]),
        ]
    )
    # Compare K_s @ y_c (low-rank vs. dense)
    transformed = chain.transform(lr.X_centred)
    K_full = transformed @ transformed.T
    expected = K_full @ lr.y_centred
    got = lr.kernel_apply(chain, lr.y_centred)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)


def test_lowrank_kernel_matrix_matches_dense(regression_data) -> None:
    X, y = regression_data
    full_rank = min(X.shape)
    lr = fit_lowrank_bases([RawBase()], X, y, rank=full_rank)[0]
    chain = OperatorChain([IdentityOperator(p=X.shape[1])])
    Xc = lr.X_centred
    K_full = Xc @ Xc.T
    K_lr = lr.kernel_matrix_lowrank(chain)
    np.testing.assert_allclose(K_lr, K_full, rtol=1e-6, atol=1e-8)


def test_screening_returns_all_candidates(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase(), SNVBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=30)
    chains = [
        OperatorChain([IdentityOperator(p=X.shape[1])]),
        OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X.shape[1])]),
        OperatorChain([SavitzkyGolayOperator(21, 2, 1, p=X.shape[1])]),
    ]
    candidates = fast_covariance_screen(lowrank, chains)
    assert len(candidates) == len(bases) * len(chains)
    # Scores should be sorted descending and in [0, 1]
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    for c in candidates:
        assert 0.0 <= c.score <= 1.0 + 1e-6


def test_screening_identity_matches_naive_pls_alignment(regression_data) -> None:
    """For the identity chain on the raw base, the score should match the
    classical Cauchy-Schwarz alignment ``(X^T y)^T (X^T y) / (||X||_F^2 ||y||^2)``."""
    X, y = regression_data
    bases = [RawBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=min(X.shape))
    chain = OperatorChain([IdentityOperator(p=X.shape[1])])
    candidates = fast_covariance_screen(lowrank, [chain])
    assert len(candidates) == 1
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    expected = float(np.dot(Xc.T @ yc, Xc.T @ yc)) / (float(np.sum(Xc * Xc)) * float(np.dot(yc, yc)))
    np.testing.assert_allclose(candidates[0].score, expected, rtol=1e-6, atol=1e-8)


def test_diversity_topk_caps_global(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase(), MSCBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=30)
    chains = [
        OperatorChain([SavitzkyGolayOperator(w, 2, 1, p=X.shape[1])])
        for w in (5, 7, 9, 11, 15, 21, 31)
    ] + [OperatorChain([IdentityOperator(p=X.shape[1])])]
    candidates = fast_covariance_screen(lowrank, chains)
    topk = diversity_topk(candidates, top_k_global=4)
    assert len(topk) == 4


def test_diversity_topk_caps_per_family(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=30)
    # All chains share the same family tag (sg_d1)
    chains = [
        OperatorChain([SavitzkyGolayOperator(w, 2, 1, p=X.shape[1])])
        for w in (5, 7, 9, 11, 15)
    ]
    candidates = fast_covariance_screen(lowrank, chains)
    topk = diversity_topk(candidates, top_k_global=10, top_k_per_family=2)
    assert len(topk) == 2  # diversity cap kicks in


def test_diversity_topk_caps_per_base(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase(), SNVBase(), MSCBase()]
    lowrank = fit_lowrank_bases(bases, X, y, rank=30)
    chains = [OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X.shape[1])])]
    candidates = fast_covariance_screen(lowrank, chains)
    topk = diversity_topk(candidates, top_k_global=10, top_k_per_base=1)
    assert len(topk) == 3  # one per base


def test_screening_score_is_invariant_to_y_scaling(regression_data) -> None:
    X, y = regression_data
    bases = [RawBase()]
    lr_a = fit_lowrank_bases(bases, X, y, rank=30)
    lr_b = fit_lowrank_bases(bases, X, 5.0 * y, rank=30)
    chain = OperatorChain([SavitzkyGolayOperator(11, 2, 1, p=X.shape[1])])
    s_a = fast_covariance_screen(lr_a, [chain])[0].score
    s_b = fast_covariance_screen(lr_b, [chain])[0].score
    np.testing.assert_allclose(s_b, s_a, rtol=1e-6, atol=1e-8)

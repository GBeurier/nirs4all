"""Tests for :class:`OperatorChain`: adjoint correctness, simplification, signatures."""

from __future__ import annotations

import numpy as np
import pytest

from aompls.operators import (
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from FastAOM.operator_chain import OperatorChain


@pytest.fixture
def small_chain() -> OperatorChain:
    return OperatorChain([
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=64),
        SavitzkyGolayOperator(window_length=9, polyorder=2, deriv=1, p=64),
        DetrendProjectionOperator(degree=1, p=64),
    ])


def test_chain_basic_attributes(small_chain: OperatorChain) -> None:
    assert small_chain.depth() == 3
    assert len(small_chain) == 3
    assert small_chain.signature
    assert ">" in small_chain.signature
    names = small_chain.names()
    assert names[0].startswith("sg_smooth")
    assert names[1].startswith("sg_d1")
    assert names[2].startswith("detrend")


def test_chain_transform_matches_matrix(small_chain: OperatorChain) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((25, 64))
    Y_transform = small_chain.transform(X)
    A = small_chain.matrix(64)
    Y_matrix = X @ A.T
    np.testing.assert_allclose(Y_transform, Y_matrix, atol=1e-8)


def test_chain_adjoint_dot_test(small_chain: OperatorChain) -> None:
    """Dot test: <A x, y> == <x, A^T y> for random x, y."""
    rng = np.random.default_rng(1)
    p = 64
    X = rng.standard_normal((1, p))
    y = rng.standard_normal(p)
    Ax = small_chain.transform(X).ravel()  # shape (p,)
    ATy = small_chain.adjoint_vec(y)  # shape (p,)
    lhs = float(np.dot(Ax, y))
    rhs = float(np.dot(X.ravel(), ATy))
    assert abs(lhs - rhs) < 1e-8, f"dot test failed: {lhs} vs {rhs}"


def test_chain_apply_cov_matches_matrix(small_chain: OperatorChain) -> None:
    rng = np.random.default_rng(2)
    p = 64
    S = rng.standard_normal((p, 5))
    out = small_chain.apply_cov(S)
    A = small_chain.matrix(p)
    np.testing.assert_allclose(out, A @ S, atol=1e-8)


def test_chain_dot_test_vector(small_chain: OperatorChain) -> None:
    """Dot test on a single vector: <A x, y> == <x, A^T y>."""
    rng = np.random.default_rng(3)
    p = 64
    x = rng.standard_normal(p)
    y = rng.standard_normal(p)
    Ax = small_chain.apply_cov(x)
    ATy = small_chain.adjoint_vec(y)
    np.testing.assert_allclose(float(np.dot(Ax, y)), float(np.dot(x, ATy)), atol=1e-8)


def test_chain_simplify_drops_identity() -> None:
    chain = OperatorChain([
        IdentityOperator(p=32),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=32),
        IdentityOperator(p=32),
    ])
    simplified = chain.simplify()
    assert simplified.depth() == 1
    assert simplified.signature.startswith("sg_d1")


def test_chain_simplify_collapses_double_detrend() -> None:
    chain = OperatorChain([
        DetrendProjectionOperator(degree=1, p=32),
        DetrendProjectionOperator(degree=2, p=32),
    ])
    simplified = chain.simplify()
    assert simplified.depth() == 1
    assert getattr(simplified.operators[0], "degree", -1) == 2


def test_chain_simplify_keeps_trailing_detrend_after_derivative() -> None:
    """Zero-padded SG / FD derivatives leave boundary trends; the trailing
    detrend remains a meaningful candidate."""
    chain = OperatorChain([
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=2, p=32),
        DetrendProjectionOperator(degree=1, p=32),
    ])
    simplified = chain.simplify()
    assert simplified.depth() == 2


def test_chain_simplify_keeps_detrend_when_higher_degree() -> None:
    chain = OperatorChain([
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=32),
        DetrendProjectionOperator(degree=3, p=32),
    ])
    simplified = chain.simplify()
    assert simplified.depth() == 2


def test_chain_signature_uniqueness() -> None:
    a = OperatorChain([
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=64),
        DetrendProjectionOperator(degree=1, p=64),
    ])
    b = OperatorChain([
        DetrendProjectionOperator(degree=1, p=64),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=64),
    ])
    assert a.signature != b.signature


def test_chain_compose_with_explicit_matrix() -> None:
    """Compare composed transform against an explicitly built matrix product."""
    rng = np.random.default_rng(42)
    p = 48
    fd = FiniteDifferenceOperator(order=1, p=p)
    sg = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=p)
    chain = OperatorChain([fd, sg])
    A_fd = fd.matrix(p)
    A_sg = sg.matrix(p)
    A_chain = A_sg @ A_fd
    np.testing.assert_allclose(chain.matrix(p), A_chain, atol=1e-8)
    X = rng.standard_normal((10, p))
    np.testing.assert_allclose(chain.transform(X), X @ A_chain.T, atol=1e-8)


def test_chain_empty_raises() -> None:
    with pytest.raises(ValueError):
        OperatorChain([])

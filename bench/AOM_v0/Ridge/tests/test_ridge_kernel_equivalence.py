"""Phase 1 equivalence tests for AOM-Ridge kernels."""

from __future__ import annotations

import numpy as np
import pytest
from aompls.banks import compact_bank
from aompls.operators import (
    DetrendProjectionOperator,
    ExplicitMatrixOperator,
    IdentityOperator,
    SavitzkyGolayOperator,
)
from aomridge.kernels import (
    as_2d_y,
    clone_operator_bank,
    compute_block_scales_from_xt,
    explicit_superblock,
    fit_operator_bank,
    linear_operator_kernel_cross,
    linear_operator_kernel_train,
    metric_times_xt,
    resolve_operator_bank,
)


def _make_data(n=40, p=64, q=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, q))
    return X, Y


def test_resolve_bank_identity_present_and_unique():
    ops = resolve_operator_bank("compact", p=32)
    identity_count = sum(1 for op in ops if isinstance(op, IdentityOperator))
    assert identity_count == 1
    # Bank without identity must get one prepended
    custom = [SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=32)]
    ops2 = resolve_operator_bank(custom, p=32)
    assert isinstance(ops2[0], IdentityOperator)
    assert len(ops2) == 2


def test_resolve_bank_clones_input():
    sg = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=None)
    ops = resolve_operator_bank([IdentityOperator(), sg], p=32)
    # Returned operator is not the same instance as the user supplied
    sg_returned = next(op for op in ops if op.name == sg.name)
    assert sg_returned is not sg


def test_resolve_bank_dedupes_duplicate_identity():
    bank = [
        IdentityOperator(),
        SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=32),
        IdentityOperator(),
    ]
    ops = resolve_operator_bank(bank, p=32)
    identity_count = sum(1 for op in ops if isinstance(op, IdentityOperator))
    assert identity_count == 1
    assert len(ops) == 2


def test_clone_operator_bank_drops_cache_and_resets_p():
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=0, p=16)
    op.matrix(16)  # populate cache
    assert op._matrix_cache is not None
    cloned = clone_operator_bank([op], p=None)
    assert cloned[0]._matrix_cache is None
    assert cloned[0].p is None
    cloned2 = clone_operator_bank([op], p=24)
    assert cloned2[0].p == 24


def test_identity_only_kernel_equals_xx():
    X, _ = _make_data()
    Xc = X - X.mean(axis=0, keepdims=True)
    ops = resolve_operator_bank([IdentityOperator()], p=Xc.shape[1])
    fit_operator_bank(ops, Xc)
    scales = np.array([1.0])
    K, U = linear_operator_kernel_train(Xc, ops, scales)
    np.testing.assert_allclose(K, Xc @ Xc.T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(U, Xc.T, atol=1e-10, rtol=1e-10)


def test_kernel_equals_explicit_superblock_strict_linear():
    X, _ = _make_data()
    Xc = X - X.mean(axis=0, keepdims=True)
    ops = resolve_operator_bank("compact", p=Xc.shape[1])
    fit_operator_bank(ops, Xc)
    Xt = Xc.T
    scales = compute_block_scales_from_xt(Xt, ops, block_scaling="rms")
    K, _ = linear_operator_kernel_train(Xc, ops, scales)
    Phi = explicit_superblock(Xc, ops, scales)
    np.testing.assert_allclose(K, Phi @ Phi.T, atol=1e-8, rtol=1e-8)


def test_metric_times_xt_matches_dense_M():
    X, _ = _make_data(n=20, p=32)
    Xc = X - X.mean(axis=0, keepdims=True)
    ops = resolve_operator_bank("compact", p=Xc.shape[1])
    fit_operator_bank(ops, Xc)
    Xt = Xc.T
    scales = compute_block_scales_from_xt(Xt, ops, block_scaling="rms")
    U_dual = metric_times_xt(Xt, ops, scales)
    # Dense reference: build M = sum_b s_b^2 A_b^T A_b explicitly
    p = Xc.shape[1]
    M = np.zeros((p, p))
    for op, s in zip(ops, scales, strict=False):
        A = op.matrix(p)
        M += float(s) ** 2 * A.T @ A
    np.testing.assert_allclose(U_dual, M @ Xt, atol=1e-8, rtol=1e-8)


def test_cross_kernel_consistent_with_features():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 24))
    X_left = rng.normal(size=(7, 24))
    Xc = X - X.mean(axis=0, keepdims=True)
    X_left_c = X_left - X.mean(axis=0, keepdims=True)
    ops = resolve_operator_bank("compact", p=24)
    fit_operator_bank(ops, Xc)
    Xt = Xc.T
    scales = compute_block_scales_from_xt(Xt, ops, block_scaling="rms")
    _, U = linear_operator_kernel_train(Xc, ops, scales)
    Kx = linear_operator_kernel_cross(X_left_c, U)
    Phi = explicit_superblock(Xc, ops, scales)
    Phi_left = explicit_superblock(X_left_c, ops, scales)
    np.testing.assert_allclose(Kx, Phi_left @ Phi.T, atol=1e-8, rtol=1e-8)


def test_block_scales_rms_normalize_gain():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 20))
    Xc = X - X.mean(axis=0, keepdims=True)
    # Identity and 10*I: pure gain duplicate
    high_gain = ExplicitMatrixOperator(10.0 * np.eye(20), name="ten_identity")
    ops = resolve_operator_bank([IdentityOperator(), high_gain], p=20)
    fit_operator_bank(ops, Xc)
    scales = compute_block_scales_from_xt(Xc.T, ops, block_scaling="rms")
    # The two scaled blocks should have the same Frobenius norm
    Z0 = scales[0] * ops[0].transform(Xc)
    Z1 = scales[1] * ops[1].transform(Xc)
    assert abs(np.linalg.norm(Z0) - np.linalg.norm(Z1)) < 1e-6


def test_block_scaling_none_returns_unit_scales():
    X, _ = _make_data()
    Xc = X - X.mean(axis=0, keepdims=True)
    ops = resolve_operator_bank("compact", p=Xc.shape[1])
    fit_operator_bank(ops, Xc)
    scales = compute_block_scales_from_xt(Xc.T, ops, block_scaling="none")
    np.testing.assert_array_equal(scales, np.ones(len(ops)))


def test_as_2d_y_handles_both_shapes():
    Y1 = np.array([1.0, 2.0, 3.0])
    Y2, was_1d = as_2d_y(Y1)
    assert was_1d is True
    assert Y2.shape == (3, 1)
    Y3 = np.zeros((4, 2))
    Y4, was_1d = as_2d_y(Y3)
    assert was_1d is False
    assert Y4.shape == (4, 2)
    with pytest.raises(ValueError):
        as_2d_y(np.zeros((2, 3, 4)))

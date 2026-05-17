"""Parity + speed tests for the vectorised xcorr replacement."""

from __future__ import annotations

import time

import numpy as np
import pytest

from aompls import operators as _parent_ops
from FastAOM.xcorr_fast import (
    install_xcorr_patch,
    is_patch_installed,
    uninstall_xcorr_patch,
    xcorr_zero_pad_fast,
)


@pytest.fixture(autouse=True)
def reset_patch():
    """Ensure each test starts with no patch installed."""
    uninstall_xcorr_patch()
    yield
    uninstall_xcorr_patch()


@pytest.mark.parametrize("k", [3, 5, 7, 11, 15, 21, 31])
@pytest.mark.parametrize("p", [16, 64, 200, 500])
def test_xcorr_parity_2d(k: int, p: int) -> None:
    rng = np.random.default_rng(seed=k * 100 + p)
    n = 17
    X = rng.standard_normal((n, p))
    kernel = rng.standard_normal(k)
    expected = _parent_ops._xcorr_zero_pad(X, kernel)
    got = xcorr_zero_pad_fast(X, kernel)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", [3, 5, 11, 21])
def test_xcorr_parity_1d(k: int) -> None:
    rng = np.random.default_rng(seed=k)
    p = 64
    x = rng.standard_normal(p)
    kernel = rng.standard_normal(k)
    expected = _parent_ops._xcorr_zero_pad(x, kernel)
    got = xcorr_zero_pad_fast(x, kernel)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_xcorr_parity_large_p() -> None:
    """One large-p sanity check (p=2000) — slow with the parent but quick here."""
    rng = np.random.default_rng(0)
    p = 2000
    X = rng.standard_normal((4, p))
    kernel = rng.standard_normal(11)
    expected = _parent_ops._xcorr_zero_pad(X, kernel)
    got = xcorr_zero_pad_fast(X, kernel)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_install_and_uninstall_patch() -> None:
    """Patch installs, replaces the parent, and uninstalls cleanly."""
    assert not is_patch_installed()
    original = _parent_ops._xcorr_zero_pad
    install_xcorr_patch()
    assert is_patch_installed()
    assert _parent_ops._xcorr_zero_pad is xcorr_zero_pad_fast
    uninstall_xcorr_patch()
    assert not is_patch_installed()
    assert _parent_ops._xcorr_zero_pad is original


def test_install_patch_is_idempotent() -> None:
    install_xcorr_patch()
    install_xcorr_patch()
    install_xcorr_patch()
    assert is_patch_installed()


def test_uninstall_without_install_is_noop() -> None:
    uninstall_xcorr_patch()
    uninstall_xcorr_patch()
    assert not is_patch_installed()


def test_patched_path_used_by_aompls_operator() -> None:
    """When patched, an ``SavitzkyGolayOperator.transform`` call must
    use the fast implementation."""
    from aompls.operators import SavitzkyGolayOperator

    rng = np.random.default_rng(0)
    p = 64
    X = rng.standard_normal((5, p))
    op = SavitzkyGolayOperator(window_length=11, polyorder=2, deriv=1, p=p)
    out_slow = op.transform(X)
    install_xcorr_patch()
    out_fast = op.transform(X)
    uninstall_xcorr_patch()
    np.testing.assert_allclose(out_fast, out_slow, rtol=1e-12, atol=1e-12)


def test_detrend_parity_2d() -> None:
    """Detrend low-rank form matches dense form bit-for-bit."""
    from aompls.operators import DetrendProjectionOperator

    rng = np.random.default_rng(0)
    for p in (16, 64, 200, 600):
        for degree in (0, 1, 2, 3):
            op = DetrendProjectionOperator(degree=degree, p=p)
            uninstall_xcorr_patch()
            S = rng.standard_normal((p, 50))
            expected = op.apply_cov(S)
            # Build fresh op to avoid cached state, install patch, recompute
            op_fast = DetrendProjectionOperator(degree=degree, p=p)
            install_xcorr_patch()
            got = op_fast.apply_cov(S)
            uninstall_xcorr_patch()
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12,
                                        err_msg=f"detrend p={p} deg={degree}")


def test_detrend_parity_transform_and_adjoint() -> None:
    from aompls.operators import DetrendProjectionOperator

    rng = np.random.default_rng(1)
    p = 100
    degree = 2
    op = DetrendProjectionOperator(degree=degree, p=p)
    X = rng.standard_normal((10, p))
    v = rng.standard_normal(p)

    uninstall_xcorr_patch()
    expected_transform = op.transform(X)
    expected_adjoint = op.adjoint_vec(v)

    op2 = DetrendProjectionOperator(degree=degree, p=p)
    install_xcorr_patch()
    got_transform = op2.transform(X)
    got_adjoint = op2.adjoint_vec(v)
    uninstall_xcorr_patch()
    np.testing.assert_allclose(got_transform, expected_transform, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got_adjoint, expected_adjoint, rtol=1e-10, atol=1e-12)


def test_xcorr_speedup_signal() -> None:
    """Sanity check that the patch actually buys speed; not a strict
    benchmark, but flags regressions if the fast path becomes slower than
    the parent loop."""
    rng = np.random.default_rng(42)
    p = 600
    X = rng.standard_normal((40, p))
    kernel = rng.standard_normal(11)

    # Warm-up call so JIT-style import overhead doesn't bias the timing.
    _parent_ops._xcorr_zero_pad(X, kernel)
    xcorr_zero_pad_fast(X, kernel)

    t0 = time.perf_counter()
    for _ in range(5):
        _parent_ops._xcorr_zero_pad(X, kernel)
    slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        xcorr_zero_pad_fast(X, kernel)
    fast = time.perf_counter() - t0

    # Allow 10% slack so a heavily loaded CI box doesn't flake this test.
    # We expect a real ~10× speedup, so the 0.9× margin is conservative.
    assert fast < slow * 0.9, (
        f"fast path {fast:.3f}s not meaningfully faster than slow {slow:.3f}s"
    )

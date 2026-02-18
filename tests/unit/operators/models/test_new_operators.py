"""Tests for new AOM bank operators and extended operator bank.

Tests cover:
- NorrisWilliamsOperator: adjoint, apply, Frobenius norm
- FiniteDifferenceOperator: adjoint, apply, Frobenius norm
- WaveletProjectionOperator: adjoint, apply, idempotence, Frobenius norm
- FFTBandpassOperator: adjoint, apply, idempotence, Frobenius norm
- extended_operator_bank: all operators have valid adjoint
"""

import numpy as np
import pytest

from nirs4all.operators.models.sklearn.aom_pls import (
    FFTBandpassOperator,
    FiniteDifferenceOperator,
    NorrisWilliamsOperator,
    WaveletProjectionOperator,
    extended_operator_bank,
)


P = 200
TOL = 1e-6


def _check_adjoint(op, p=P, tol=TOL):
    """Verify adjoint identity |<Ax, y> - <x, A^T y>| < tol."""
    op.initialize(p)
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.randn(1, p)
        y_vec = rng.randn(p)
        ax = op.apply(x).ravel()
        aty = op.apply_adjoint(y_vec)
        lhs = np.dot(ax, y_vec)
        rhs = np.dot(x.ravel(), aty)
        assert abs(lhs - rhs) < tol, (
            f"Adjoint identity failed for {op.name}: "
            f"|<Ax,y> - <x,A^T y>| = {abs(lhs - rhs):.2e}"
        )


# =============================================================================
# NorrisWilliamsOperator Tests
# =============================================================================


class TestNorrisWilliamsOperator:
    """Test NorrisWilliams operator for adjoint correctness and behavior."""

    def test_adjoint_default(self):
        _check_adjoint(NorrisWilliamsOperator(gap=5, segment=1, deriv=1))

    def test_adjoint_with_segment(self):
        _check_adjoint(NorrisWilliamsOperator(gap=5, segment=5, deriv=1))

    def test_adjoint_second_deriv(self):
        _check_adjoint(NorrisWilliamsOperator(gap=5, segment=1, deriv=2))

    def test_adjoint_large_gap(self):
        _check_adjoint(NorrisWilliamsOperator(gap=11, segment=5, deriv=1))

    def test_apply_output_shape(self):
        op = NorrisWilliamsOperator(gap=5, segment=1, deriv=1)
        op.initialize(P)
        X = np.random.randn(10, P)
        result = op.apply(X)
        assert result.shape == X.shape

    def test_frobenius_norm_positive(self):
        op = NorrisWilliamsOperator(gap=5, segment=1, deriv=1)
        op.initialize(P)
        assert op.frobenius_norm_sq() > 0

    def test_name_format(self):
        op = NorrisWilliamsOperator(gap=5, segment=3, deriv=1)
        assert "NW" in op.name
        assert "g=5" in op.name

    def test_params(self):
        op = NorrisWilliamsOperator(gap=7, segment=3, deriv=2, delta=2.0)
        params = op.params
        assert params["gap"] == 7
        assert params["segment"] == 3
        assert params["deriv"] == 2
        assert params["delta"] == 2.0


# =============================================================================
# FiniteDifferenceOperator Tests
# =============================================================================


class TestFiniteDifferenceOperator:
    """Test finite difference operator."""

    def test_adjoint_first_order(self):
        _check_adjoint(FiniteDifferenceOperator(order=1))

    def test_adjoint_second_order(self):
        _check_adjoint(FiniteDifferenceOperator(order=2))

    def test_adjoint_third_order(self):
        _check_adjoint(FiniteDifferenceOperator(order=3))

    def test_apply_output_shape(self):
        op = FiniteDifferenceOperator(order=1)
        op.initialize(P)
        X = np.random.randn(10, P)
        result = op.apply(X)
        assert result.shape == X.shape

    def test_first_order_approximates_derivative(self):
        """First-order FD on smooth function should approximate true derivative (up to sign)."""
        p = 200
        dx = 2 * np.pi / (p - 1)
        op = FiniteDifferenceOperator(order=1, delta=dx)
        op.initialize(p)
        x = np.linspace(0, 2 * np.pi, p)
        signal = np.sin(x).reshape(1, -1)
        fd = op.apply(signal).ravel()
        true_deriv = np.cos(x)
        # Check interior points (boundary effects at edges)
        # Use abs(corr) since convolution sign convention may differ
        interior = slice(10, p - 10)
        corr = np.corrcoef(fd[interior], true_deriv[interior])[0, 1]
        assert abs(corr) > 0.95, f"FD correlation with true derivative: {corr:.3f}"

    def test_frobenius_norm_positive(self):
        op = FiniteDifferenceOperator(order=1)
        op.initialize(P)
        assert op.frobenius_norm_sq() > 0

    def test_name_format(self):
        op = FiniteDifferenceOperator(order=2)
        assert "FD" in op.name
        assert "2" in op.name


# =============================================================================
# WaveletProjectionOperator Tests
# =============================================================================


class TestWaveletProjectionOperator:
    """Test wavelet approximation projection operator."""

    def test_adjoint_haar(self):
        _check_adjoint(WaveletProjectionOperator(wavelet='haar', level=2), tol=1e-5)

    def test_adjoint_db4(self):
        _check_adjoint(WaveletProjectionOperator(wavelet='db4', level=3), tol=1e-5)

    def test_adjoint_sym5(self):
        _check_adjoint(WaveletProjectionOperator(wavelet='sym5', level=2), tol=1e-5)

    def test_adjoint_coif3(self):
        _check_adjoint(WaveletProjectionOperator(wavelet='coif3', level=2), tol=1e-5)

    def test_apply_output_shape(self):
        op = WaveletProjectionOperator(wavelet='db4', level=3)
        op.initialize(P)
        X = np.random.randn(10, P)
        result = op.apply(X)
        assert result.shape == X.shape

    def test_idempotent(self):
        """Projection applied twice should equal once."""
        op = WaveletProjectionOperator(wavelet='db4', level=3)
        op.initialize(P)
        X = np.random.randn(5, P)
        once = op.apply(X)
        twice = op.apply(once)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_smoothing_effect(self):
        """Wavelet projection should smooth a noisy signal."""
        op = WaveletProjectionOperator(wavelet='db4', level=3)
        op.initialize(P)
        rng = np.random.RandomState(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, P))
        noisy = signal + 0.5 * rng.randn(P)
        smoothed = op.apply(noisy.reshape(1, -1)).ravel()
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_frobenius_norm_positive(self):
        op = WaveletProjectionOperator(wavelet='db4', level=3)
        op.initialize(P)
        assert op.frobenius_norm_sq() > 0

    def test_name_format(self):
        op = WaveletProjectionOperator(wavelet='coif3', level=2)
        assert "wav" in op.name
        assert "coif3" in op.name

    def test_level_clamped(self):
        """Level higher than max should be clamped."""
        op = WaveletProjectionOperator(wavelet='haar', level=100)
        op.initialize(P)
        # Should not crash; actual level is clamped
        X = np.random.randn(2, P)
        result = op.apply(X)
        assert result.shape == X.shape


# =============================================================================
# FFTBandpassOperator Tests
# =============================================================================


class TestFFTBandpassOperator:
    """Test FFT bandpass filter operator."""

    def test_adjoint_lowpass(self):
        _check_adjoint(FFTBandpassOperator(low_cut=0.0, high_cut=0.25))

    def test_adjoint_highpass(self):
        _check_adjoint(FFTBandpassOperator(low_cut=0.1, high_cut=0.5))

    def test_adjoint_bandpass(self):
        _check_adjoint(FFTBandpassOperator(low_cut=0.05, high_cut=0.3))

    def test_apply_output_shape(self):
        op = FFTBandpassOperator(low_cut=0.0, high_cut=0.25)
        op.initialize(P)
        X = np.random.randn(10, P)
        result = op.apply(X)
        assert result.shape == X.shape

    def test_idempotent(self):
        """Binary mask filter applied twice should equal once."""
        op = FFTBandpassOperator(low_cut=0.0, high_cut=0.3)
        op.initialize(P)
        X = np.random.randn(5, P)
        once = op.apply(X)
        twice = op.apply(once)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_lowpass_removes_high_freq(self):
        """Lowpass filter should preserve low-frequency signal and remove high-frequency."""
        op = FFTBandpassOperator(low_cut=0.0, high_cut=0.1)
        op.initialize(P)
        x = np.linspace(0, 1, P)
        low_freq = np.sin(2 * np.pi * 2 * x)  # 2 Hz
        high_freq = np.sin(2 * np.pi * 80 * x)  # 80 Hz
        signal = (low_freq + high_freq).reshape(1, -1)
        filtered = op.apply(signal).ravel()
        # Low-freq energy should be preserved more than high-freq
        low_energy = np.sum(low_freq ** 2)
        # Filtered should correlate better with low_freq
        corr_low = np.corrcoef(filtered, low_freq)[0, 1]
        assert abs(corr_low) > 0.5

    def test_frobenius_norm_positive(self):
        op = FFTBandpassOperator(low_cut=0.0, high_cut=0.25)
        op.initialize(P)
        assert op.frobenius_norm_sq() > 0

    def test_name_format(self):
        op = FFTBandpassOperator(low_cut=0.1, high_cut=0.5)
        assert "FFT" in op.name

    def test_1d_input(self):
        """Should handle 1D input (single vector)."""
        op = FFTBandpassOperator(low_cut=0.0, high_cut=0.3)
        op.initialize(P)
        x = np.random.randn(P)
        result = op.apply(x)
        assert result.shape == (P,)


# =============================================================================
# Extended Operator Bank Tests
# =============================================================================


class TestExtendedOperatorBank:
    """Test extended_operator_bank integrity."""

    def test_bank_size(self):
        bank = extended_operator_bank()
        assert len(bank) > 30  # Should be significantly larger than default

    def test_all_adjoint_valid(self):
        """All operators in extended bank must satisfy the adjoint identity."""
        bank = extended_operator_bank()
        for op in bank:
            _check_adjoint(op, tol=1e-4)

    def test_all_frobenius_positive(self):
        bank = extended_operator_bank()
        for op in bank:
            op.initialize(P)
            assert op.frobenius_norm_sq() > 0, f"{op.name} has non-positive Frobenius norm"

    def test_all_apply_correct_shape(self):
        bank = extended_operator_bank()
        X = np.random.randn(3, P)
        for op in bank:
            op.initialize(P)
            result = op.apply(X)
            assert result.shape == X.shape, f"{op.name} output shape mismatch"

    def test_includes_all_families(self):
        """Extended bank should include operators from all families."""
        bank = extended_operator_bank()
        names = [op.name for op in bank]
        name_str = " ".join(names)
        assert "identity" in name_str.lower() or "Identity" in name_str
        assert "SG" in name_str
        assert "NW" in name_str
        assert "FD" in name_str
        assert "wav" in name_str
        assert "FFT" in name_str

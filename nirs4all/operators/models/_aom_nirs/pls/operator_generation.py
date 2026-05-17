"""Operator generation: primitive grids and a small composition grammar.

Generates a structured catalogue of strict-linear primitive operators for
exploration:

- Savitzky-Golay scale-space (windows × polyorder × derivatives).
- Whittaker smoothers across a logarithmic `lambda` grid.
- Gaussian-derivative filters (sigma × order).
- Detrend projections (polynomial degree).
- Finite differences (order 1, 2).
- Norris-Williams gap derivatives.
- Fixed-shift wavelength operators (small integer shifts).

A canonicalisation function simplifies common redundancies when chains are
composed (identity removal, detrend collapse, dual-smooth fusion).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .operators import (
    ComposedOperator,
    DetrendProjectionOperator,
    FiniteDifferenceOperator,
    IdentityOperator,
    LinearSpectralOperator,
    NorrisWilliamsOperator,
    SavitzkyGolayOperator,
    WhittakerOperator,
    _xcorr_zero_pad,
)


# ---------------------------------------------------------------------------
# Gaussian derivative operator
# ---------------------------------------------------------------------------


class GaussianDerivativeOperator(LinearSpectralOperator):
    """Derivative-of-Gaussian convolution operator.

    Acts as cross-correlation of a row spectrum with a Hermite-weighted
    Gaussian kernel. `order=0` is pure Gaussian smoothing; `order in {1,2,3}`
    are derivative-of-Gaussian (DoG) kernels.
    """

    def __init__(self, sigma: float = 5.0, order: int = 1, half_width: int = None, p: int = None) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if order not in (0, 1, 2, 3):
            raise ValueError("order must be 0, 1, 2, or 3")
        if half_width is None:
            half_width = max(3, int(round(4 * sigma)))
        super().__init__(name=f"gauss_d{order}_s{sigma:g}", p=p)
        self.sigma = float(sigma)
        self.order = order
        self.half_width = int(half_width)
        x = np.arange(-self.half_width, self.half_width + 1, dtype=float)
        s2 = sigma * sigma
        gauss = np.exp(-(x * x) / (2.0 * s2)) / (sigma * math.sqrt(2.0 * math.pi))
        if order == 0:
            kernel = gauss
        elif order == 1:
            kernel = -(x / s2) * gauss
        elif order == 2:
            kernel = ((x * x - s2) / (s2 * s2)) * gauss
        else:  # order == 3
            kernel = -((x**3 - 3.0 * s2 * x) / (s2 * s2 * s2)) * gauss
        self._kernel = kernel

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(X, self._kernel)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(S.T, self._kernel).T

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(v, self._kernel[::-1])

    def _matrix_impl(self, p: int) -> np.ndarray:
        return self._apply_cov_impl(np.eye(p))


class FixedShiftOperator(LinearSpectralOperator):
    """Fixed integer-pixel cyclic-padded shift along the spectrum.

    Shifting is linear: `(S(shift) x)[i] = x[i - shift]` with zero-padding
    at the boundaries. Used to absorb small wavelength misalignment.
    """

    def __init__(self, shift: int = 1, p: int = None) -> None:
        super().__init__(name=f"shift_{shift:+d}", p=p)
        self.shift = int(shift)

    def _shift(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=float)
        s = self.shift
        if s == 0:
            return X.copy()
        if X.ndim == 1:
            if s > 0:
                out[s:] = X[:-s]
            else:
                out[:s] = X[-s:]
            return out
        if s > 0:
            out[:, s:] = X[:, :-s]
        else:
            out[:, :s] = X[:, -s:]
        return out

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return self._shift(X)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        # apply A @ S where A is the shift matrix.
        s = self.shift
        out = np.zeros_like(S, dtype=float)
        if s == 0:
            return S.copy()
        if S.ndim == 1:
            if s > 0:
                out[s:] = S[:-s]
            else:
                out[:s] = S[-s:]
            return out
        if s > 0:
            out[s:, :] = S[:-s, :]
        else:
            out[:s, :] = S[-s:, :]
        return out

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        # adjoint of shift = shift in the opposite direction
        s = -self.shift
        out = np.zeros_like(v, dtype=float)
        if s == 0:
            return v.copy()
        if s > 0:
            out[s:] = v[:-s]
        else:
            out[:s] = v[-s:]
        return out

    def _matrix_impl(self, p: int) -> np.ndarray:
        return self._apply_cov_impl(np.eye(p))


# ---------------------------------------------------------------------------
# Primitive grid builders
# ---------------------------------------------------------------------------


def primitive_savitzky_golay_grid(p: int) -> List[LinearSpectralOperator]:
    out: List[LinearSpectralOperator] = []
    max_window = max(7, min(45, p // 3))
    windows = [w for w in (5, 7, 9, 11, 15, 21, 31, 41) if 3 <= w <= max_window]
    for w in windows:
        for d in (0, 1, 2):
            for poly in (max(2, d + 1), max(2, d + 2)):
                if poly < w:
                    out.append(SavitzkyGolayOperator(window_length=w, polyorder=poly, deriv=d, p=p))
    return out


def primitive_whittaker_grid(p: int) -> List[LinearSpectralOperator]:
    out: List[LinearSpectralOperator] = []
    for lam in (1.0, 10.0, 1e2, 1e4, 1e6):
        out.append(WhittakerOperator(lam=lam, p=p))
    return out


def primitive_gaussian_derivative_grid(p: int) -> List[LinearSpectralOperator]:
    out: List[LinearSpectralOperator] = []
    for sigma in (1.5, 3.0, 6.0, 12.0):
        for order in (0, 1, 2):
            out.append(GaussianDerivativeOperator(sigma=sigma, order=order, p=p))
    return out


def primitive_finite_difference_grid(p: int) -> List[LinearSpectralOperator]:
    return [FiniteDifferenceOperator(order=1, p=p), FiniteDifferenceOperator(order=2, p=p)]


def primitive_detrend_grid(p: int) -> List[LinearSpectralOperator]:
    return [DetrendProjectionOperator(degree=d, p=p) for d in (0, 1, 2, 3)]


def primitive_norris_williams_grid(p: int) -> List[LinearSpectralOperator]:
    out: List[LinearSpectralOperator] = []
    for gap in (3, 5, 11):
        for smoothing in (1, 5):
            for order in (1, 2):
                out.append(NorrisWilliamsOperator(gap=gap, smoothing=smoothing, order=order, p=p))
    return out


def primitive_shift_grid(p: int) -> List[LinearSpectralOperator]:
    return [FixedShiftOperator(shift=s, p=p) for s in (-2, -1, 1, 2)]


def primitive_bank(p: int) -> List[LinearSpectralOperator]:
    """Default primitive bank used by the explorer (~80 operators)."""
    bank: List[LinearSpectralOperator] = [IdentityOperator(p=p)]
    bank.extend(primitive_savitzky_golay_grid(p))
    bank.extend(primitive_whittaker_grid(p))
    bank.extend(primitive_gaussian_derivative_grid(p))
    bank.extend(primitive_finite_difference_grid(p))
    bank.extend(primitive_detrend_grid(p))
    bank.extend(primitive_norris_williams_grid(p))
    bank.extend(primitive_shift_grid(p))
    return bank


# ---------------------------------------------------------------------------
# Family signatures and grammar
# ---------------------------------------------------------------------------


def family_signature(op: LinearSpectralOperator) -> str:
    """Return a coarse family tag for an operator."""
    name = op.name.lower()
    if name.startswith("identity"):
        return "identity"
    if name.startswith("sg_smooth") or name.startswith("sg_d0"):
        return "sg_smooth"
    if name.startswith("sg_d1"):
        return "sg_d1"
    if name.startswith("sg_d2"):
        return "sg_d2"
    if name.startswith("fd_d"):
        return "finite_difference"
    if name.startswith("nw_"):
        return "norris_williams"
    if name.startswith("detrend_"):
        return "detrend"
    if name.startswith("whittaker"):
        return "whittaker"
    if name.startswith("gauss_d0"):
        return "gauss_smooth"
    if name.startswith("gauss_d"):
        return "gauss_deriv"
    if name.startswith("shift_"):
        return "shift"
    if name.startswith("compose("):
        return "composed"
    return "other"


def grammar_allows(chain: Tuple[LinearSpectralOperator, ...], op: LinearSpectralOperator) -> bool:
    """Reject obviously redundant chain extensions.

    Rules implemented:

    - Never append identity to a non-empty chain (already neutral).
    - No two consecutive smoothers (sg_smooth/whittaker/gauss_smooth) of the
      same family.
    - No two consecutive detrends.
    - No two consecutive derivatives of the same order.
    - Allow up to one shift per chain (drift correction).
    """
    if not chain:
        return op.name != "identity"
    fam_op = family_signature(op)
    if fam_op == "identity":
        return False
    last = chain[-1]
    fam_last = family_signature(last)
    if fam_last == fam_op and fam_op in ("sg_smooth", "gauss_smooth", "whittaker", "detrend", "shift"):
        return False
    # Avoid stacking many derivatives
    if fam_op in ("sg_d1", "fd_d1", "gauss_deriv", "norris_williams") and any(
        family_signature(c) in ("sg_d1", "fd_d1", "gauss_deriv", "norris_williams") for c in chain
    ):
        return False
    if fam_op == "sg_d2" and any(family_signature(c) == "sg_d2" for c in chain):
        return False
    return True


def canonicalize(chain: Tuple[LinearSpectralOperator, ...]) -> Tuple[LinearSpectralOperator, ...]:
    """Apply simple chain simplifications.

    - Drop identity stages.
    - Collapse two consecutive detrends to the higher-degree one.
    """
    if not chain:
        return chain
    out: List[LinearSpectralOperator] = []
    for op in chain:
        if family_signature(op) == "identity":
            continue
        if out and family_signature(out[-1]) == "detrend" and family_signature(op) == "detrend":
            d_prev = getattr(out[-1], "degree", 0)
            d_op = getattr(op, "degree", 0)
            if d_op > d_prev:
                out[-1] = op
            continue
        out.append(op)
    return tuple(out)


def chain_signature(chain: Tuple[LinearSpectralOperator, ...]) -> str:
    """Stable string signature for a chain (used for deduplication)."""
    return ">".join(op.name for op in chain) if chain else "identity"

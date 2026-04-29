"""Linear spectral operators.

A spectral operator `A in R^{p x p}` acts on row spectra as `X_b = X A^T`.
For strict linear operators this implies the cross-covariance identity
`X_b^T Y = A X^T Y`, which lets fast engines (covariance SIMPLS, adjoint
NIPALS) evaluate operator candidates without materializing the transformed
spectra.

Every concrete operator implements the protocol below.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import solve_banded


class LinearSpectralOperator:
    """Protocol for strict linear spectral operators.

    Subclasses implement at minimum `_matrix_impl(p)`. Subclasses with cheap
    convolution implementations override `_transform_impl`, `_apply_cov_impl`,
    and `_adjoint_vec_impl` to avoid materializing the explicit `p x p`
    matrix.

    Attributes:
        name: A short string identifying the operator instance.
        p: The number of features the operator was initialised for. Set on
            first use of `transform` / `apply_cov` / `adjoint_vec` if not yet
            initialised.
    """

    name: str
    p: Optional[int]
    is_strict_linear: bool = True

    def __init__(self, name: str, p: Optional[int] = None) -> None:
        self.name = name
        self.p = p
        self._matrix_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "LinearSpectralOperator":
        """Bind the operator to the feature dimensionality.

        Strict-linear operators do not learn parameters from data; the only
        information they extract from `X` is its number of columns. Supervised
        operators may override `fit` to learn parameters; their prediction-time
        apply must remain linear in the input.
        """
        if X is not None:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X must be 2-dimensional")
            self.p = X.shape[1]
        return self

    # -------------------------------------------------------- public API

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the operator to row spectra: `X_b = X A^T`."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("transform expects a 2D array")
        if self.p is None:
            self.p = X.shape[1]
        if X.shape[1] != self.p:
            raise ValueError(f"X has {X.shape[1]} features; operator {self.name} expects {self.p}")
        return self._transform_impl(X)

    def apply_cov(self, S: np.ndarray) -> np.ndarray:
        """Apply the operator to a cross-covariance: `A S`."""
        S = np.asarray(S)
        if S.ndim == 1:
            if self.p is None:
                self.p = S.shape[0]
            if S.shape[0] != self.p:
                raise ValueError(f"S has shape {S.shape}; operator {self.name} expects p={self.p}")
            return self._apply_cov_impl(S.reshape(-1, 1)).ravel()
        if S.ndim != 2:
            raise ValueError("apply_cov expects a 1D or 2D array")
        if self.p is None:
            self.p = S.shape[0]
        if S.shape[0] != self.p:
            raise ValueError(f"S has {S.shape[0]} rows; operator {self.name} expects {self.p}")
        return self._apply_cov_impl(S)

    def adjoint_vec(self, v: np.ndarray) -> np.ndarray:
        """Apply the adjoint to a vector or matrix: `A^T v`."""
        v = np.asarray(v)
        if v.ndim == 1:
            if self.p is None:
                self.p = v.shape[0]
            if v.shape[0] != self.p:
                raise ValueError(f"v has shape {v.shape}; operator {self.name} expects p={self.p}")
            return self._adjoint_vec_impl(v)
        if v.ndim != 2:
            raise ValueError("adjoint_vec expects a 1D or 2D array")
        if self.p is None:
            self.p = v.shape[0]
        if v.shape[0] != self.p:
            raise ValueError(f"v has shape {v.shape}; operator {self.name} expects p={self.p}")
        cols = [self._adjoint_vec_impl(v[:, j]) for j in range(v.shape[1])]
        return np.column_stack(cols)

    def matrix(self, p: Optional[int] = None) -> np.ndarray:
        """Return the explicit `p x p` matrix of the operator."""
        if p is None:
            p = self.p
        if p is None:
            raise ValueError("matrix requires a feature dimensionality")
        if self._matrix_cache is not None and self._matrix_cache.shape == (p, p):
            return self._matrix_cache
        prev_p = self.p
        self.p = p
        try:
            mat = self._matrix_impl(p)
        finally:
            self.p = prev_p
        self._matrix_cache = np.asarray(mat, dtype=float)
        return self._matrix_cache

    def is_linear_at_apply(self) -> bool:
        return bool(self.is_strict_linear)

    def fitted_parameters(self) -> dict:
        return {}

    # ------------------------------------------------- subclass interface

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return X @ self.matrix(X.shape[1]).T

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return self.matrix(S.shape[0]) @ S

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return self.matrix(v.shape[0]).T @ v

    def _matrix_impl(self, p: int) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Cross-correlation helper
# ---------------------------------------------------------------------------


def _xcorr_zero_pad(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 'same' cross-correlation with zero-padded boundaries to each row.

    Defined so that `out[:, i] = sum_j kernel[j] * X[:, i + j - half]` with
    zero-padding outside `[0, p)`. This corresponds to applying the symmetric
    Toeplitz convolution matrix `M` with `M[i, t] = kernel[t - i + half]`.
    """
    X = np.asarray(X, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    k = kernel.shape[0]
    half_left = (k - 1) // 2
    half_right = k - 1 - half_left
    if X.ndim == 1:
        X2 = X.reshape(1, -1)
        squeeze = True
    else:
        X2 = X
        squeeze = False
    n, p = X2.shape
    pad = np.zeros((n, p + k - 1), dtype=float)
    pad[:, half_left : half_left + p] = X2
    out = np.empty((n, p), dtype=float)
    for i in range(p):
        out[:, i] = pad[:, i : i + k] @ kernel
    return out.ravel() if squeeze else out


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityOperator(LinearSpectralOperator):
    """The identity operator. Always present in every default bank."""

    def __init__(self, p: Optional[int] = None) -> None:
        super().__init__(name="identity", p=p)

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return X.copy()

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return S.copy()

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return v.copy()

    def _matrix_impl(self, p: int) -> np.ndarray:
        return np.eye(p)


# ---------------------------------------------------------------------------
# Savitzky-Golay (smoothing and derivatives)
# ---------------------------------------------------------------------------


def _sg_coefficients(window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    """Compute Savitzky-Golay coefficients for a centered window.

    The returned 1D kernel `k` has shape `(window_length,)` and is intended
    for cross-correlation: `out[i] = sum_j k[j] * x[i - half + j]`. This
    matches the SG fit convention where `out[i]` is the `deriv`-th derivative
    of the polynomial of order `polyorder` fitted by least squares to the
    window `x[i - half : i + half + 1]`, evaluated at the center of the
    window.
    """
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer >= 3")
    if not (0 <= polyorder < window_length):
        raise ValueError("polyorder must be in [0, window_length)")
    if not (0 <= deriv <= polyorder):
        raise ValueError("deriv must be in [0, polyorder]")
    half = (window_length - 1) // 2
    j = np.arange(-half, half + 1, dtype=float)
    A = np.vander(j, polyorder + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coeffs = pinv[deriv] * float(math.factorial(deriv))
    return coeffs


class SavitzkyGolayOperator(LinearSpectralOperator):
    """Savitzky-Golay smoothing or derivative as a strict linear operator.

    Boundaries are zero-padded so that the operator is strictly linear in the
    input. This avoids data-dependent boundary modes which would otherwise
    couple samples through the operator.
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 0,
        p: Optional[int] = None,
    ) -> None:
        if deriv == 0:
            tag = f"sg_smooth_w{window_length}_p{polyorder}"
        else:
            tag = f"sg_d{deriv}_w{window_length}_p{polyorder}"
        super().__init__(name=tag, p=p)
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self._kernel = _sg_coefficients(window_length, polyorder, deriv)

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(X, self._kernel)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        # apply_cov(S) = A @ S applies A column-wise. For each column v of S,
        # (A v)[t] = sum_l k[l - t + half] * v[l] = xcorr(v, k)[t]. We treat
        # the columns of S as rows by transposing.
        return _xcorr_zero_pad(S.T, self._kernel).T

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        # The adjoint A^T satisfies (A^T v)[t] = sum_l k[t - l + half] * v[l]
        # which equals xcorr(v, k_rev)[t] with k_rev[j] = k[k_len - 1 - j].
        return _xcorr_zero_pad(v, self._kernel[::-1])

    def _matrix_impl(self, p: int) -> np.ndarray:
        # A is the column-action matrix; apply_cov(eye) returns A directly.
        return self._apply_cov_impl(np.eye(p))


# ---------------------------------------------------------------------------
# Finite difference (centered)
# ---------------------------------------------------------------------------


class FiniteDifferenceOperator(LinearSpectralOperator):
    """First or second order centered finite difference."""

    def __init__(self, order: int = 1, p: Optional[int] = None) -> None:
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        super().__init__(name=f"fd_d{order}", p=p)
        self.order = order
        if order == 1:
            self._kernel = np.array([-0.5, 0.0, 0.5], dtype=float)
        else:
            self._kernel = np.array([1.0, -2.0, 1.0], dtype=float)

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(X, self._kernel)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(S.T, self._kernel).T

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(v, self._kernel[::-1])

    def _matrix_impl(self, p: int) -> np.ndarray:
        return self._apply_cov_impl(np.eye(p))


# ---------------------------------------------------------------------------
# Detrend projection (symmetric)
# ---------------------------------------------------------------------------


class DetrendProjectionOperator(LinearSpectralOperator):
    """Polynomial detrend as the orthogonal projector onto the polynomial complement.

    For a polynomial degree `d`, the basis matrix `P in R^{p x (d+1)}` spans
    `{1, t, t^2, ..., t^d}` evaluated at evenly spaced points. The operator is
    the orthogonal projector onto the complement of `range(P)`:
    `A = I - Q Q^T` where `Q` are the QR-orthonormal columns of `P`. Both the
    transform and the matrix are symmetric (`A = A^T`).
    """

    def __init__(self, degree: int = 1, p: Optional[int] = None) -> None:
        if degree < 0:
            raise ValueError("degree must be >= 0")
        super().__init__(name=f"detrend_d{degree}", p=p)
        self.degree = degree
        self._cached_p: Optional[int] = None
        self._cached_complement: Optional[np.ndarray] = None

    def _basis_complement(self, p: int) -> np.ndarray:
        if self._cached_p == p and self._cached_complement is not None:
            return self._cached_complement
        if p < self.degree + 1:
            raise ValueError(f"DetrendProjection degree={self.degree} requires p >= {self.degree + 1}")
        t = np.linspace(-1.0, 1.0, p)
        cols = [t**k for k in range(self.degree + 1)]
        P = np.column_stack(cols)
        Q, _ = np.linalg.qr(P)
        complement = np.eye(p) - Q @ Q.T
        self._cached_p = p
        self._cached_complement = complement
        return complement

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        complement = self._basis_complement(X.shape[1])
        return X @ complement.T

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        complement = self._basis_complement(S.shape[0])
        return complement @ S

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        complement = self._basis_complement(v.shape[0])
        return complement.T @ v

    def _matrix_impl(self, p: int) -> np.ndarray:
        return self._basis_complement(p)


# ---------------------------------------------------------------------------
# Norris-Williams gap derivative
# ---------------------------------------------------------------------------


class NorrisWilliamsOperator(LinearSpectralOperator):
    """Norris-Williams gap derivative with optional moving-average smoothing.

    The operator first applies a centered moving-average smoothing of length
    `smoothing` and then computes a gap derivative
    `D x[i] = (x[i + g] - x[i - g]) / (2 g)` (or the second-difference
    equivalent for `order=2`). The two stages compose into a single Toeplitz
    convolution; the result is strictly linear in the input.
    """

    def __init__(self, gap: int = 5, smoothing: int = 5, order: int = 1, p: Optional[int] = None) -> None:
        if gap < 1:
            raise ValueError("gap must be >= 1")
        if smoothing < 1 or smoothing % 2 == 0:
            raise ValueError("smoothing must be an odd integer >= 1")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        super().__init__(name=f"nw_g{gap}_s{smoothing}_d{order}", p=p)
        self.gap = gap
        self.smoothing = smoothing
        self.order = order
        # Match the production nirs4all `NorrisWilliamsOperator` convention:
        # build the gap derivative kernel `[-1/(2g), 0, ..., 0, +1/(2g)]` of
        # length 2g+1, then for order=2 convolve it with itself, and finally
        # convolve with the segment smoothing kernel.
        seg_kernel = np.ones(smoothing) / float(smoothing) if smoothing > 1 else np.array([1.0])
        gap_kernel = np.zeros(2 * gap + 1, dtype=float)
        gap_kernel[0] = -1.0 / (2.0 * gap)
        gap_kernel[-1] = 1.0 / (2.0 * gap)
        composed = np.convolve(seg_kernel, gap_kernel)
        if order == 2:
            composed = np.convolve(composed, gap_kernel)
        self._kernel = composed

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(X, self._kernel)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(S.T, self._kernel).T

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return _xcorr_zero_pad(v, self._kernel[::-1])

    def _matrix_impl(self, p: int) -> np.ndarray:
        return self._apply_cov_impl(np.eye(p))


# ---------------------------------------------------------------------------
# Whittaker smoother
# ---------------------------------------------------------------------------


class WhittakerOperator(LinearSpectralOperator):
    """Whittaker smoother as a strict linear operator.

    The smoother solves `(I + lam D^T D) z = x` with `D` the (p-2) x p
    second-difference matrix. The matrix `A = (I + lam D^T D)^{-1}` is
    symmetric positive definite, so `A^T = A`. We pre-factorise the banded
    matrix once per feature dimensionality and apply the factor when needed.
    """

    def __init__(self, lam: float = 1e3, p: Optional[int] = None) -> None:
        if lam <= 0:
            raise ValueError("lam must be positive")
        super().__init__(name=f"whittaker_l{lam:g}", p=p)
        self.lam = float(lam)
        self._cached_p: Optional[int] = None
        self._banded: Optional[np.ndarray] = None

    def _ensure_factor(self, p: int) -> None:
        if self._cached_p == p:
            return
        if p < 4:
            # Build the explicit dense matrix for tiny p; the banded form is
            # not well-defined when p < l + u + 1.
            self._cached_p = p
            self._banded = None
            return
        # Diagonals of D^T D where D is the (p-2) x p second-difference matrix.
        diag0 = np.full(p, 6.0)
        diag0[0] = 1.0
        diag0[1] = 5.0
        diag0[-2] = 5.0
        diag0[-1] = 1.0
        diag1 = np.full(p - 1, -4.0)
        diag1[0] = -2.0
        diag1[-1] = -2.0
        diag2 = np.ones(p - 2)
        ab = np.zeros((5, p))
        ab[2] = 1.0 + self.lam * diag0
        ab[1, 1:] = self.lam * diag1
        ab[0, 2:] = self.lam * diag2
        ab[3, :-1] = self.lam * diag1
        ab[4, :-2] = self.lam * diag2
        self._banded = ab
        self._cached_p = p

    def _solve(self, B: np.ndarray) -> np.ndarray:
        # B has shape (p,) or (p, k). Returns the same shape.
        p = B.shape[0]
        self._ensure_factor(p)
        if self._banded is None:
            # Fall back to dense for very small p
            mat = self._matrix_impl(p)
            return mat @ B
        return solve_banded((2, 2), self._banded, B)

    def _solve_rows(self, X: np.ndarray) -> np.ndarray:
        # X has shape (n, p). Apply the operator along axis 1.
        return self._solve(X.T).T

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return self._solve_rows(X)

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return self._solve(S)

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return self._solve(v)

    def _matrix_impl(self, p: int) -> np.ndarray:
        # Build the dense matrix by solving against the identity. For the
        # tiny-p fall-through, this is the only available representation.
        if p < 4:
            # Build I + lam D^T D explicitly and invert.
            D = np.zeros((max(0, p - 2), p))
            for i in range(p - 2):
                D[i, i] = 1.0
                D[i, i + 1] = -2.0
                D[i, i + 2] = 1.0
            B = np.eye(p) + self.lam * D.T @ D
            return np.linalg.inv(B)
        self._ensure_factor(p)
        return solve_banded((2, 2), self._banded, np.eye(p))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class ComposedOperator(LinearSpectralOperator):
    """Composition of strict linear operators.

    Given operators `A_1, A_2, ..., A_K`, the composed operator is
    `A = A_K @ ... @ A_2 @ A_1`. It is strictly linear in the input.

    The convention is "apply leftmost first": `transform` calls
    `ops[0].transform`, then `ops[1].transform`, etc., which corresponds to
    `(((X A_1^T) A_2^T) ... A_K^T) = X A_K^T ... A_1^T = X (A_K ... A_1)^T`.
    """

    def __init__(self, operators: Sequence[LinearSpectralOperator], name: Optional[str] = None) -> None:
        if not operators:
            raise ValueError("operators must be non-empty")
        if name is None:
            name = "compose(" + " | ".join(op.name for op in operators) + ")"
        super().__init__(name=name)
        self.operators = list(operators)

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "ComposedOperator":
        for op in self.operators:
            op.fit(X, y)
        if X is not None:
            self.p = X.shape[1]
        return self

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        out = X
        for op in self.operators:
            out = op.transform(out)
        return out

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        out = S
        for op in self.operators:
            out = op.apply_cov(out)
        return out

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        out = v
        for op in reversed(self.operators):
            out = op.adjoint_vec(out)
        return out

    def _matrix_impl(self, p: int) -> np.ndarray:
        mat = np.eye(p)
        for op in self.operators:
            mat = op.matrix(p) @ mat
        return mat


# ---------------------------------------------------------------------------
# Explicit-matrix operator (for tests)
# ---------------------------------------------------------------------------


class ExplicitMatrixOperator(LinearSpectralOperator):
    """Operator backed by a fixed, externally provided matrix.

    Used in tests to verify that the protocol holds for arbitrary linear maps.
    """

    def __init__(self, matrix: np.ndarray, name: str = "explicit") -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square 2D array")
        super().__init__(name=name, p=matrix.shape[0])
        self._matrix = np.asarray(matrix, dtype=float)
        self._matrix_cache = self._matrix

    def _transform_impl(self, X: np.ndarray) -> np.ndarray:
        return X @ self._matrix.T

    def _apply_cov_impl(self, S: np.ndarray) -> np.ndarray:
        return self._matrix @ S

    def _adjoint_vec_impl(self, v: np.ndarray) -> np.ndarray:
        return self._matrix.T @ v

    def _matrix_impl(self, p: int) -> np.ndarray:
        if p != self._matrix.shape[0]:
            raise ValueError(f"explicit operator built for p={self._matrix.shape[0]}, got {p}")
        return self._matrix

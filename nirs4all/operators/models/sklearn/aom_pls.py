"""AOM-PLS: Adaptive Operator-Mixture PLS regressor for nirs4all.

A sklearn-compatible implementation of AOM-PLS that learns preprocessing
selection within a single PLS training run over a linear operator bank.
Each PLS component selects its own preprocessing via sparse gating
(sparsemax), avoiding exhaustive grid search over preprocessing configs.

Uses NIPALS deflation for correct interaction between operator-transformed
components and residual data. Each component evaluates all operators in the
bank on the current residual X, selects the best via sparsemax gating, and
extracts a PLS component through that operator's "lens".

Mathematical formulation
------------------------
Let X ∈ ℝ^{n×p} be the input matrix and y ∈ ℝ^n the response vector.
Given an operator bank {A_b}_{b=1..B} of p×p linear operators (SG filters,
detrend projections, identity), AOM-PLS extracts K predictive components:

Hard gating (default): For each operator b, run full K-component NIPALS:
1. For k = 1..K: w_k via adjoint trick A_b, t_k = X_res w_k, NIPALS deflation
2. Compute prefix regression coefficients B_k for k = 1..K
3. If validation data: select operator with lowest validation RMSE at best k
4. If no validation: select operator with highest first-component R²

Sparsemax gating (experimental): Soft operator mixing via R² scoring:
1. R² scoring per operator on first component
2. γ = sparsemax(s / τ) for sparse weight vector
3. NIPALS with mixed weights: w_k = Σ_b γ_b A_b ŵ_b

References
----------
- de Jong, S. (1993). SIMPLS: An alternative approach to partial least
  squares regression. Chemometrics and Intelligent Laboratory Systems.
- Martens, M. & Martens, H. (2001). Multivariate Analysis of Quality:
  An Introduction. Wiley.
- Peters, B. et al. (2019). Sparse Sequence-to-Sequence Models. ACL.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import convolve1d as _convolve1d
from scipy.signal import savgol_coeffs
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# =============================================================================
# Linear Operator Bank
# =============================================================================

class LinearOperator:
    """Base class for linear operators in the AOM-PLS bank.

    Each operator represents a p×p linear transformation A_b that can be
    applied to spectral data. The key requirement is that both the forward
    and adjoint operations are efficient (O(p) per sample, not O(p²)).
    """

    @property
    def name(self) -> str:
        """Human-readable name for reporting."""
        return self.__class__.__name__

    @property
    def params(self) -> dict:
        """Operator parameters for metadata."""
        return {}

    def initialize(self, p: int) -> None:
        """Initialize operator for signals of length p.

        Called once during AOMPLSRegressor.fit() before any apply/adjoint calls.
        """
        self.p_ = p

    def apply(self, X: NDArray) -> NDArray:
        """Apply operator: X_b = X @ A_b.

        Parameters
        ----------
        X : ndarray of shape (n, p) or (p,)
            Input data.

        Returns
        -------
        X_b : ndarray, same shape as X
            Transformed data.
        """
        raise NotImplementedError

    def apply_adjoint(self, c: NDArray) -> NDArray:
        """Apply adjoint: g = A_b^T @ c.

        Parameters
        ----------
        c : ndarray of shape (p,)
            Input vector (typically cross-covariance X^T y).

        Returns
        -------
        g : ndarray of shape (p,)
            Adjoint-transformed vector.
        """
        raise NotImplementedError

    def frobenius_norm_sq(self) -> float:
        """Compute ||A_b||_F^2 for normalized block scoring.

        Must be called after initialize().
        """
        raise NotImplementedError


class IdentityOperator(LinearOperator):
    """Identity operator (no preprocessing).

    Always included in the bank to guarantee recovery of standard PLS
    and provide a baseline that other operators must beat.
    """

    @property
    def name(self) -> str:
        return "identity"

    def initialize(self, p: int) -> None:
        super().initialize(p)
        self._nu = float(p)

    def apply(self, X: NDArray) -> NDArray:
        return X.copy()

    def apply_adjoint(self, c: NDArray) -> NDArray:
        return c.copy()

    def frobenius_norm_sq(self) -> float:
        return self._nu


class SavitzkyGolayOperator(LinearOperator):
    """Savitzky-Golay filter operator with explicit zero-padding.

    Uses zero-padded 'same' convolution to maintain strict linearity,
    ensuring the adjoint identity <A x, y> = <x, A^T y> holds exactly.

    Parameters
    ----------
    window : int
        Window length (must be odd, > polyorder).
    polyorder : int
        Polynomial order for the SG filter.
    deriv : int, default=0
        Derivative order (0=smoothing, 1=first derivative, etc.).
    delta : float, default=1.0
        Sampling interval.
    """

    def __init__(self, window: int = 11, polyorder: int = 2, deriv: int = 0, delta: float = 1.0):
        self.window = window
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    @property
    def name(self) -> str:
        return f"SG(w={self.window},p={self.polyorder},d={self.deriv})"

    @property
    def params(self) -> dict:
        return {"window": self.window, "polyorder": self.polyorder, "deriv": self.deriv, "delta": self.delta}

    def initialize(self, p: int) -> None:
        super().initialize(p)
        # SG coefficients in "dot product" form
        coeffs = savgol_coeffs(self.window, self.polyorder, deriv=self.deriv, delta=self.delta)
        # Convolution kernel = reversed coefficients (matching savgol_filter internals)
        self._conv_kernel = coeffs[::-1].astype(np.float64, copy=True)
        # Adjoint kernel = original coefficients
        self._adj_kernel = coeffs.astype(np.float64, copy=True)
        # Precompute Frobenius norm squared
        self._nu = self._compute_frobenius_norm_sq(p)

    def _compute_frobenius_norm_sq(self, p: int) -> float:
        """Compute ||A||_F^2 analytically from the kernel.

        For same-mode zero-padded convolution, each row of the operator matrix
        contains a (possibly truncated) copy of the kernel. Interior rows have
        the full kernel, boundary rows have partial overlap.
        """
        hw = (self.window - 1) // 2
        h2 = self._conv_kernel ** 2
        total = 0.0
        for i in range(p):
            # For position i, the kernel centered at i overlaps indices [i-hw, i+hw]
            # Clipped to [0, p-1]. The kernel index offset: k_start to k_end
            k_start = max(0, hw - i)
            k_end = min(self.window, p - i + hw)
            total += np.sum(h2[k_start:k_end])
        return total

    def apply(self, X: NDArray) -> NDArray:
        return _convolve1d(X, self._conv_kernel, axis=-1, mode='constant', cval=0.0)

    def apply_adjoint(self, c: NDArray) -> NDArray:
        return _convolve1d(c, self._adj_kernel, axis=-1, mode='constant', cval=0.0)

    def frobenius_norm_sq(self) -> float:
        return self._nu


class DetrendProjectionOperator(LinearOperator):
    """Detrend projection operator.

    Removes polynomial trend of given degree by projecting onto the
    orthogonal complement of the polynomial basis. The resulting operator
    A = I - Q Q^T is symmetric (A^T = A).

    Parameters
    ----------
    degree : int, default=1
        Polynomial degree to remove (1=linear, 2=quadratic).
    """

    def __init__(self, degree: int = 1):
        self.degree = degree

    @property
    def name(self) -> str:
        return f"detrend(deg={self.degree})"

    @property
    def params(self) -> dict:
        return {"degree": self.degree}

    def initialize(self, p: int) -> None:
        super().initialize(p)
        # Build orthonormal polynomial basis via QR
        t = np.linspace(-1, 1, p)
        V = np.column_stack([t ** d for d in range(self.degree + 1)])
        Q, _ = np.linalg.qr(V)
        self._Q = Q  # (p, degree+1), orthonormal columns
        # ||A||_F^2 = trace(A) = p - (degree+1) for an orthogonal projection complement
        self._nu = float(p - self.degree - 1)

    def apply(self, X: NDArray) -> NDArray:
        if X.ndim == 1:
            return X - self._Q @ (self._Q.T @ X)
        # X (n, p): XA = X - (X Q) Q^T
        return X - (X @ self._Q) @ self._Q.T

    def apply_adjoint(self, c: NDArray) -> NDArray:
        # Symmetric operator: A^T = A
        return self.apply(c)

    def frobenius_norm_sq(self) -> float:
        return self._nu


class ComposedOperator(LinearOperator):
    """Composition of two linear operators: A = A_second @ A_first.

    Applies A_first then A_second. The adjoint is A^T = A_first^T @ A_second^T.

    Parameters
    ----------
    first : LinearOperator
        First operator to apply.
    second : LinearOperator
        Second operator to apply.
    """

    def __init__(self, first: LinearOperator, second: LinearOperator):
        self.first = first
        self.second = second

    @property
    def name(self) -> str:
        return f"{self.second.name}∘{self.first.name}"

    @property
    def params(self) -> dict:
        return {"first": self.first.params, "second": self.second.params}

    def initialize(self, p: int) -> None:
        super().initialize(p)
        self.first.initialize(p)
        self.second.initialize(p)
        # Frobenius norm of composition: compute empirically
        self._nu = self._compute_nu_empirical(p)

    def _compute_nu_empirical(self, p: int, n_probes: int = 50) -> float:
        """Estimate ||A||_F^2 via random probing.

        E[||Ax||^2] = ||A||_F^2 when x ~ N(0, I).
        """
        rng = np.random.RandomState(42)
        total = 0.0
        for _ in range(n_probes):
            x = rng.randn(1, p)
            ax = self.apply(x)
            total += np.sum(ax ** 2)
        return total / n_probes

    def apply(self, X: NDArray) -> NDArray:
        return self.second.apply(self.first.apply(X))

    def apply_adjoint(self, c: NDArray) -> NDArray:
        # (A_second @ A_first)^T = A_first^T @ A_second^T
        return self.first.apply_adjoint(self.second.apply_adjoint(c))

    def frobenius_norm_sq(self) -> float:
        return self._nu


def default_operator_bank() -> list[LinearOperator]:
    """Build the default operator bank for AOM-PLS.

    Includes identity, SG filters at key configurations, and detrend
    projections. Kept lean (~11 operators) to avoid diluting selection
    signal across near-duplicate operators.

    Note: SNV (Standard Normal Variate) is NOT included because it is
    a non-linear operator (per-sample std division). The linear part of
    SNV (mean centering) is captured by DetrendProjectionOperator(degree=0).

    Returns
    -------
    operators : list of LinearOperator
        Default operator bank with ~11 operators.
    """
    savgols = [
        # SG smoothing
        SavitzkyGolayOperator(window=11, polyorder=2, deriv=0),
        SavitzkyGolayOperator(window=21, polyorder=2, deriv=0),
        SavitzkyGolayOperator(window=31, polyorder=2, deriv=0),
        # SG 1st derivative (the workhorse of NIRS preprocessing)
        SavitzkyGolayOperator(window=11, polyorder=2, deriv=1),
        SavitzkyGolayOperator(window=21, polyorder=2, deriv=1),
        SavitzkyGolayOperator(window=31, polyorder=2, deriv=1),
        SavitzkyGolayOperator(window=11, polyorder=3, deriv=1),
        # SG 2nd derivative
        SavitzkyGolayOperator(window=11, polyorder=2, deriv=2),
        SavitzkyGolayOperator(window=21, polyorder=2, deriv=2),
        SavitzkyGolayOperator(window=31, polyorder=2, deriv=2),
        # SG 2nd derivative
        SavitzkyGolayOperator(window=11, polyorder=3, deriv=2),
        SavitzkyGolayOperator(window=21, polyorder=3, deriv=2),
        SavitzkyGolayOperator(window=31, polyorder=3, deriv=2),
    ]

    DetrendOps = [
        DetrendProjectionOperator(degree=0),  # mean centering
    ]

    ComposedOps = []
    for sg in savgols:
        for dt in DetrendOps:
            ComposedOps.append(ComposedOperator(first=sg, second=dt))

    return [
        # Identity (always included as baseline — recovers standard PLS)
        IdentityOperator(),
        # Savitzky-Golay filters (various smoothing and derivative configs)
        *savgols,
        # Detrend projections
        DetrendProjectionOperator(degree=0),  # mean centering
        DetrendProjectionOperator(degree=1),  # linear detrend
        DetrendProjectionOperator(degree=2),  # quadratic detrend
        # Composed operators (e.g., SG smoothing + detrend)
        *ComposedOps,
    ]


# =============================================================================
# Sparsemax
# =============================================================================

def _sparsemax(z: NDArray) -> NDArray:
    """Sparsemax activation function (Martins & Astudillo, 2016).

    Projects z onto the probability simplex, producing a sparse output
    where weak entries are exactly zero.

    Parameters
    ----------
    z : ndarray of shape (d,)
        Input logits.

    Returns
    -------
    p : ndarray of shape (d,)
        Sparse probability vector summing to 1.
    """
    d = len(z)
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    # Find threshold: largest k such that z_sorted[k] > (cumsum[k] - 1) / (k + 1)
    k_range = np.arange(1, d + 1, dtype=np.float64)
    thresholds = (cumsum - 1.0) / k_range
    support = z_sorted > thresholds
    k_star = np.max(np.where(support)[0]) + 1 if np.any(support) else 1
    tau = (cumsum[k_star - 1] - 1.0) / k_star
    return np.maximum(z - tau, 0.0)


# =============================================================================
# OPLS Pre-filter (optional)
# =============================================================================

def _opls_prefilter(X: NDArray, y: NDArray, n_orth: int) -> tuple[NDArray, NDArray, NDArray]:
    """Extract and remove orthogonal components from X.

    Simple OPLS-style pre-filter: extracts components that have maximum
    variance in X but are orthogonal to y. These represent systematic
    variation not related to the response.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Centered input data.
    y : ndarray of shape (n,) or (n, 1)
        Centered response.
    n_orth : int
        Number of orthogonal components to remove.

    Returns
    -------
    X_filtered : ndarray of shape (n, p)
        X with orthogonal variation removed.
    P_orth : ndarray of shape (p, n_orth)
        Orthogonal loadings (for prediction).
    T_orth : ndarray of shape (n, n_orth)
        Orthogonal scores.
    """
    y_flat = y.ravel()
    n, p = X.shape
    P_orth = np.zeros((p, n_orth), dtype=np.float64)
    T_orth = np.zeros((n, n_orth), dtype=np.float64)

    X_filt = X.copy()
    for i in range(n_orth):
        # PLS weight: direction of maximum covariance
        w = X_filt.T @ y_flat
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # PLS score
        t = X_filt @ w
        tt = t @ t
        if tt < 1e-14:
            break

        # PLS loading
        p_vec = X_filt.T @ t / tt

        # Orthogonal loading: residual of p_vec after removing y-predictive part
        p_orth = p_vec - w * (w @ p_vec)
        p_orth_norm = np.linalg.norm(p_orth)
        if p_orth_norm < 1e-14:
            break
        p_orth = p_orth / p_orth_norm

        # Orthogonal score
        t_orth = X_filt @ p_orth
        tt_orth = t_orth @ t_orth
        if tt_orth < 1e-14:
            break

        # Remove orthogonal component
        X_filt = X_filt - np.outer(t_orth, t_orth @ X_filt / tt_orth)

        P_orth[:, i] = p_orth
        T_orth[:, i] = t_orth

    return X_filt, P_orth, T_orth


# =============================================================================
# NIPALS Single-Operator Extraction
# =============================================================================

def _nipals_extract(
    X: NDArray,
    Y: NDArray,
    operator: LinearOperator,
    n_components: int,
    eps: float = 1e-12,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, int]:
    """Run K-component NIPALS PLS with a single linear operator.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Centered X matrix.
    Y : ndarray of shape (n, q)
        Centered Y matrix.
    operator : LinearOperator
        Operator for weight computation via adjoint trick.
    n_components : int
        Maximum components.
    eps : float
        Numerical tolerance.

    Returns
    -------
    W : ndarray of shape (p, n_extracted)
    T : ndarray of shape (n, n_extracted)
    P : ndarray of shape (p, n_extracted)
    Q : ndarray of shape (q, n_extracted)
    B_coefs : ndarray of shape (n_extracted, p, q)
        Prefix regression coefficients.
    n_extracted : int
    """
    n, p = X.shape
    q = Y.shape[1]

    W = np.zeros((p, n_components), dtype=np.float64)
    T = np.zeros((n, n_components), dtype=np.float64)
    P = np.zeros((p, n_components), dtype=np.float64)
    Q = np.zeros((q, n_components), dtype=np.float64)

    X_res = X.copy()
    Y_res = Y.copy()
    n_extracted = 0

    for k in range(n_components):
        c_k = X_res.T @ Y_res
        if q == 1:
            c_k = c_k[:, 0]
        else:
            u, s, _ = np.linalg.svd(c_k, full_matrices=False)
            c_k = u[:, 0] * s[0]

        c_norm = np.linalg.norm(c_k)
        if c_norm < eps:
            break

        g = operator.apply_adjoint(c_k)
        g_norm = np.linalg.norm(g)
        if g_norm < eps:
            break
        w_hat = g / g_norm
        a_w = operator.apply(w_hat.reshape(1, -1)).ravel()
        a_w_norm = np.linalg.norm(a_w)
        if a_w_norm < eps:
            break
        w_k = a_w / a_w_norm

        t_k = X_res @ w_k
        tt = t_k @ t_k
        if tt < eps:
            break

        p_k = (X_res.T @ t_k) / tt
        q_k = (Y_res.T @ t_k) / tt

        W[:, k] = w_k
        T[:, k] = t_k
        P[:, k] = p_k
        Q[:, k] = q_k
        n_extracted = k + 1

        X_res -= np.outer(t_k, p_k)
        Y_res -= np.outer(t_k, q_k)

    W = W[:, :n_extracted]
    T = T[:, :n_extracted]
    P = P[:, :n_extracted]
    Q = Q[:, :n_extracted]

    # Compute prefix regression coefficients
    B_coefs = np.zeros((n_extracted, p, q), dtype=np.float64)
    for k in range(n_extracted):
        PtW = P[:, :k + 1].T @ W[:, :k + 1]
        try:
            R_k = W[:, :k + 1] @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            R_k = W[:, :k + 1] @ np.linalg.pinv(PtW)
        B_coefs[k] = R_k @ Q[:, :k + 1].T

    return W, T, P, Q, B_coefs, n_extracted


# =============================================================================
# NumPy Backend Implementation
# =============================================================================

def _aompls_fit_numpy(
    X: NDArray,
    Y: NDArray,
    operators: list[LinearOperator],
    n_components: int,
    tau: float,
    n_orth: int,
    gate: str,
    X_val: NDArray | None = None,
    Y_val: NDArray | None = None,
    operator_index: int | None = None,
) -> dict:
    """Fit AOM-PLS model using NumPy with NIPALS deflation.

    For hard gating: tries each operator with full K-component NIPALS and
    selects the best by validation RMSE (if validation data provided) or
    first-component R² (fallback). This directly evaluates multi-component
    prediction quality rather than relying on proxy scoring criteria.

    For sparsemax gating: uses first-component R² scoring for soft mixing,
    then runs NIPALS with the weighted operator combination.

    Guarantees AOM-PLS >= PLS: identity always competes, and if no operator
    genuinely helps, identity wins and AOM-PLS reduces to standard NIPALS PLS.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Centered X matrix (NOT per-column scaled — preserves spectral shape).
    Y : ndarray of shape (n, q)
        Centered Y matrix.
    operators : list of LinearOperator
        Initialized operator bank.
    n_components : int
        Maximum number of components to extract.
    tau : float
        Sparsemax temperature (only used when gate='sparsemax').
    n_orth : int
        Number of OPLS orthogonal components to pre-filter.
    gate : str
        'hard' for argmax operator selection, 'sparsemax' for soft mixing.
    X_val : ndarray of shape (n_val, p) or None
        Centered/scaled validation X (same space as X).
    Y_val : ndarray of shape (n_val, q) or None
        Centered/scaled validation Y (same space as Y).

    Returns
    -------
    artifacts : dict
        Dictionary containing all fitted artifacts.
    """
    n, p = X.shape
    q = Y.shape[1]
    B = len(operators)
    eps = 1e-12

    # OPLS pre-filter
    P_orth = None
    if n_orth > 0:
        X, P_orth, T_orth = _opls_prefilter(X, Y[:, 0] if q == 1 else Y, n_orth)
        # Apply OPLS to validation data
        if X_val is not None:
            X_val = X_val.copy()
            for j in range(P_orth.shape[1]):
                p_o = P_orth[:, j]
                t_o = X_val @ p_o
                X_val = X_val - np.outer(t_o, p_o)

    if gate == "hard":
        has_val = X_val is not None and Y_val is not None and X_val.shape[0] > 0
        best_k = None  # None = use all components

        if operator_index is not None:
            # Operator pre-selected (e.g. by Optuna) — skip selection, just fit
            best_b = min(operator_index, B - 1)
        elif has_val:
            # External validation data — use it for operator + prefix selection
            best_val_rmse = np.inf
            best_b = 0
            for b, op in enumerate(operators):
                W_b, T_b, P_b, Q_b, B_coefs_b, n_ext = _nipals_extract(X, Y, op, n_components, eps)
                if n_ext == 0:
                    continue
                for k in range(1, n_ext + 1):
                    y_pred = X_val @ B_coefs_b[k - 1]
                    rmse = np.sqrt(np.mean((Y_val - y_pred) ** 2))
                    if rmse < best_val_rmse:
                        best_val_rmse = rmse
                        best_b = b
                        best_k = k
        else:
            # No external help — internal holdout for operator + prefix selection
            rng = np.random.RandomState(42)
            n_ho = max(3, n // 5)
            idx = rng.permutation(n)
            X_ho_train, X_ho_val = X[idx[n_ho:]], X[idx[:n_ho]]
            Y_ho_train, Y_ho_val = Y[idx[n_ho:]], Y[idx[:n_ho]]
            best_val_rmse = np.inf
            best_b = 0
            for b, op in enumerate(operators):
                W_b, T_b, P_b, Q_b, B_coefs_b, n_ext = _nipals_extract(X_ho_train, Y_ho_train, op, n_components, eps)
                if n_ext == 0:
                    continue
                for k in range(1, n_ext + 1):
                    y_pred = X_ho_val @ B_coefs_b[k - 1]
                    rmse = np.sqrt(np.mean((Y_ho_val - y_pred) ** 2))
                    if rmse < best_val_rmse:
                        best_val_rmse = rmse
                        best_b = b
                        best_k = k

        # Fit with selected operator on ALL data
        W, T, P, Q, B_coefs, n_extracted = _nipals_extract(X, Y, operators[best_b], n_components, eps)

        # Safety: if selected operator couldn't extract, fall back to identity
        if n_extracted == 0:
            identity_idx = next((b for b, op in enumerate(operators) if isinstance(op, IdentityOperator)), 0)
            W, T, P, Q, B_coefs, n_extracted = _nipals_extract(X, Y, operators[identity_idx], n_components, eps)
            best_b = identity_idx
            best_k = None

        # Limit to selected prefix count
        if best_k is not None and best_k < n_extracted:
            n_extracted = best_k
            W = W[:, :n_extracted]
            T = T[:, :n_extracted]
            P = P[:, :n_extracted]
            Q = Q[:, :n_extracted]
            B_coefs = B_coefs[:n_extracted]

        Gamma = np.zeros((n_extracted, B), dtype=np.float64)
        Gamma[:, best_b] = 1.0

    else:
        # ---- Sparsemax: R² scoring for soft mixing ----
        c_0 = X.T @ Y
        if q == 1:
            c_0 = c_0[:, 0]
            y_score = Y[:, 0]
        else:
            u0, s0, vt0 = np.linalg.svd(c_0, full_matrices=False)
            c_0 = u0[:, 0] * s0[0]
            y_score = Y @ vt0[0]

        y_norm_sq = np.dot(y_score, y_score)
        scores = np.zeros(B, dtype=np.float64)
        for b, op in enumerate(operators):
            g_b = op.apply_adjoint(c_0)
            g_norm = np.linalg.norm(g_b)
            if g_norm < eps:
                continue
            w_hat_b = g_b / g_norm
            a_w = op.apply(w_hat_b.reshape(1, -1)).ravel()
            a_w_norm = np.linalg.norm(a_w)
            if a_w_norm < eps:
                continue
            w_b = a_w / a_w_norm
            t_b = X @ w_b
            cov_yt = np.dot(y_score, t_b)
            scores[b] = cov_yt ** 2 / (y_norm_sq * np.dot(t_b, t_b) + eps)

        gamma_row = _sparsemax(scores / (tau * np.max(scores) + eps))
        selected_ops = [(b, gamma_row[b]) for b in range(B) if gamma_row[b] > eps]

        # NIPALS with mixed operators
        W = np.zeros((p, n_components), dtype=np.float64)
        T = np.zeros((n, n_components), dtype=np.float64)
        P = np.zeros((p, n_components), dtype=np.float64)
        Q = np.zeros((q, n_components), dtype=np.float64)
        Gamma = np.zeros((n_components, B), dtype=np.float64)

        X_res = X.copy()
        Y_res = Y.copy()
        n_extracted = 0

        for k in range(n_components):
            c_k = X_res.T @ Y_res
            if q == 1:
                c_k = c_k[:, 0]
            else:
                u, s, _ = np.linalg.svd(c_k, full_matrices=False)
                c_k = u[:, 0] * s[0]

            c_norm = np.linalg.norm(c_k)
            if c_norm < eps:
                break

            w_k = np.zeros(p, dtype=np.float64)
            for b_idx, weight in selected_ops:
                g_b = operators[b_idx].apply_adjoint(c_k)
                g_norm = np.linalg.norm(g_b)
                if g_norm < eps:
                    continue
                w_hat_b = g_b / g_norm
                a_w = operators[b_idx].apply(w_hat_b.reshape(1, -1)).ravel()
                a_w_norm = np.linalg.norm(a_w)
                if a_w_norm < eps:
                    continue
                w_k += weight * (a_w / a_w_norm)

            w_norm = np.linalg.norm(w_k)
            if w_norm < eps:
                break
            w_k = w_k / w_norm

            Gamma[k] = gamma_row

            t_k = X_res @ w_k
            tt = t_k @ t_k
            if tt < eps:
                break
            p_k = (X_res.T @ t_k) / tt
            q_k = (Y_res.T @ t_k) / tt

            W[:, k] = w_k
            T[:, k] = t_k
            P[:, k] = p_k
            Q[:, k] = q_k
            n_extracted = k + 1

            X_res -= np.outer(t_k, p_k)
            Y_res -= np.outer(t_k, q_k)

        W = W[:, :n_extracted]
        T = T[:, :n_extracted]
        P = P[:, :n_extracted]
        Q = Q[:, :n_extracted]
        Gamma = Gamma[:n_extracted]

        # Compute prefix regression coefficients
        B_coefs = np.zeros((n_extracted, p, q), dtype=np.float64)
        for k in range(n_extracted):
            W_a = W[:, :k + 1]
            P_a = P[:, :k + 1]
            Q_a = Q[:, :k + 1]
            PtW = P_a.T @ W_a
            try:
                R_a = W_a @ np.linalg.inv(PtW)
            except np.linalg.LinAlgError:
                R_a = W_a @ np.linalg.pinv(PtW)
            B_coefs[k] = R_a @ Q_a.T

    return {
        "n_extracted": n_extracted,
        "W": W,
        "T": T,
        "P": P,
        "Q": Q,
        "Gamma": Gamma,
        "B_coefs": B_coefs,
        "P_orth": P_orth,
    }


# =============================================================================
# Torch Backend Availability
# =============================================================================

def _check_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


# =============================================================================
# AOMPLSRegressor
# =============================================================================

class AOMPLSRegressor(BaseEstimator, RegressorMixin):
    """Adaptive Operator-Mixture PLS regressor.

    Automatically selects the best preprocessing operator from a bank of
    linear operators (SG filters, detrend projections, identity) by running
    full NIPALS PLS for each operator and selecting the one with the best
    validation RMSE. All PLS components use the selected operator, combining
    automatic preprocessing selection with standard NIPALS PLS.

    Guarantees AOM-PLS >= standard PLS: identity always competes in the bank,
    so if no operator genuinely helps, AOM-PLS reduces to NIPALS PLS.

    Uses NIPALS deflation and centering-only X standardization (no per-column
    scaling) to preserve spectral shape for the operator bank.

    Parameters
    ----------
    n_components : int, default=15
        Maximum number of PLS components to extract.
    operator_bank : list of LinearOperator or None, default=None
        Explicit list of operators. If None, uses default_operator_bank().
    gate : str, default='hard'
        Gating function for block selection per component.
        'hard': argmax — selects the single best operator per component.
            Guarantees AOM-PLS ≥ standard PLS (identity always competes).
        'sparsemax': sparse soft mixing of operators (experimental).
    tau : float, default=0.5
        Temperature for sparsemax gating (ignored when gate='hard').
        Lower values → sparser selection. Range: 0.1–2.0.
    n_orth : int, default=0
        Number of OPLS orthogonal components to pre-filter.
    operator_index : int or None, default=None
        If set, skip operator selection and use this operator from the bank.
        Intended to be tuned by Optuna alongside n_components and n_orth,
        so the refit inherits the best operator without needing validation data.
    center : bool, default=True
        Whether to center X and Y (subtract mean).
    scale : bool, default=False
        Whether to scale X and Y to unit variance per column.
        WARNING: per-column scaling destroys spectral shape and cripples
        SG/detrend operators. Only enable if your data is not spectral.
    selection : str, default='validation'
        Component count selection strategy. 'validation' uses held-out data
        if provided, otherwise uses all components.
    random_state : int or None, default=None
        Random state for reproducibility.
    backend : str, default='numpy'
        Computational backend ('numpy' or 'torch').

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components extracted.
    k_selected_ : int
        Selected number of components (after validation).
    gamma_ : ndarray of shape (n_components_, n_blocks)
        Per-component block gating weights.
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients using selected components.
    block_names_ : list of str
        Names of operators in the bank.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 200)
    >>> y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(100)
    >>> model = AOMPLSRegressor(n_components=10)
    >>> model.fit(X, y)
    AOMPLSRegressor(n_components=10)
    >>> preds = model.predict(X)

    See Also
    --------
    SIMPLS : Standard SIMPLS regressor.
    MBPLS : Multiblock PLS regressor.
    FCKPLS : Fractional Convolutional Kernel PLS.
    """

    _webapp_meta = {
        "category": "pls",
        "tier": "advanced",
        "tags": ["pls", "aom-pls", "preprocessing", "multiblock", "regression", "sparse-gating"],
    }

    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 15,
        operator_bank: list[LinearOperator] | None = None,
        gate: str = "hard",
        tau: float = 0.5,
        n_orth: int = 0,
        operator_index: int | None = None,
        center: bool = True,
        scale: bool = False,
        selection: str = "validation",
        random_state: int | None = None,
        backend: str = "numpy",
    ):
        self.n_components = n_components
        self.operator_bank = operator_bank
        self.gate = gate
        self.tau = tau
        self.n_orth = n_orth
        self.operator_index = operator_index
        self.center = center
        self.scale = scale
        self.selection = selection
        self.random_state = random_state
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
    ) -> "AOMPLSRegressor":
        """Fit the AOM-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        X_val : array-like of shape (n_val, n_features), optional
            Validation data for prefix selection.
        y_val : array-like of shape (n_val,) or (n_val, n_targets), optional
            Validation targets.

        Returns
        -------
        self : AOMPLSRegressor
            Fitted estimator.
        """
        if self.backend not in ("numpy", "torch"):
            raise ValueError(f"backend must be 'numpy' or 'torch', got '{self.backend}'")

        if self.backend == "torch" and not _check_torch_available():
            raise ImportError(
                "PyTorch is required for AOMPLSRegressor with backend='torch'. "
                "Install it with: pip install torch"
            )

        if self.gate not in ("hard", "sparsemax"):
            raise ValueError(f"gate must be 'hard' or 'sparsemax', got '{self.gate}'")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Limit components by data dimensions
        max_components = min(n_samples - 1, n_features)
        n_comp = min(self.n_components, max_components)

        # Center and optionally scale
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = np.zeros(y.shape[1], dtype=np.float64)

        if self.scale:
            self.x_std_ = X.std(axis=0, ddof=1)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.x_std_ = np.where(self.x_std_ < 1e-10, 1.0, self.x_std_)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.x_std_ = np.ones(n_features, dtype=np.float64)
            self.y_std_ = np.ones(y.shape[1], dtype=np.float64)

        X_centered = (X - self.x_mean_) / self.x_std_
        Y_centered = (y - self.y_mean_) / self.y_std_

        # Initialize operator bank
        operators = self.operator_bank if self.operator_bank is not None else default_operator_bank()
        # Ensure identity is present
        if not any(isinstance(op, IdentityOperator) for op in operators):
            operators = [IdentityOperator()] + list(operators)
        self.operators_ = list(operators)
        for op in self.operators_:
            op.initialize(n_features)
        self.block_names_ = [op.name for op in self.operators_]

        # Center/scale validation data for operator selection
        X_val_c = None
        Y_val_c = None
        if X_val is not None and y_val is not None:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)
            if self._y_1d and y_v.ndim == 1:
                y_v = y_v.reshape(-1, 1)
            X_val_c = (X_v - self.x_mean_) / self.x_std_
            Y_val_c = (y_v - self.y_mean_) / self.y_std_

        # Fit using appropriate backend
        if self.backend == "torch":
            from nirs4all.operators.models.pytorch.aom_pls import aompls_fit_torch
            artifacts = aompls_fit_torch(X_centered, Y_centered, self.operators_, n_comp, self.tau, self.n_orth, self.gate)
        else:
            artifacts = _aompls_fit_numpy(X_centered, Y_centered, self.operators_, n_comp, self.tau, self.n_orth, self.gate, X_val_c, Y_val_c, self.operator_index)

        # Unpack artifacts
        self.n_components_ = artifacts["n_extracted"]
        self._W = artifacts["W"]
        self._T = artifacts["T"]
        self._P = artifacts["P"]
        self._Q = artifacts["Q"]
        self.gamma_ = artifacts["Gamma"]
        self._B_coefs = artifacts["B_coefs"]
        self._P_orth = artifacts["P_orth"]

        # Prefix selection: hard gate already handles operator + prefix internally
        # (via validation or internal holdout). Only use external _select_prefix
        # for sparsemax gate or when operator_index is set with external val data.
        if self.gate == "hard":
            self.k_selected_ = self.n_components_
        else:
            self.k_selected_ = self._select_prefix(X_val, y_val)

        # Store regression coefficients for selected prefix
        if self.n_components_ > 0:
            B_selected = self._B_coefs[self.k_selected_ - 1]
            self.coef_ = B_selected * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]
        else:
            self.coef_ = np.zeros((n_features, y.shape[1]), dtype=np.float64)

        return self

    def _select_prefix(self, X_val: ArrayLike | None, y_val: ArrayLike | None) -> int:
        """Select the best number of components via validation."""
        if self.n_components_ == 0:
            return 0

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            if self._y_1d and y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

            X_val_c = (X_val - self.x_mean_) / self.x_std_

            # Apply OPLS filter if used
            if self._P_orth is not None:
                for j in range(self._P_orth.shape[1]):
                    p_o = self._P_orth[:, j]
                    t_o = X_val_c @ p_o
                    X_val_c = X_val_c - np.outer(t_o, p_o)

            best_k = 1
            best_rmse = np.inf
            for k in range(1, self.n_components_ + 1):
                B_k = self._B_coefs[k - 1]
                y_pred_std = X_val_c @ B_k
                y_pred = y_pred_std * self.y_std_ + self.y_mean_
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k
            return best_k

        return self.n_components_

    def predict(
        self,
        X: ArrayLike,
        n_components: int | None = None,
    ) -> NDArray[np.floating]:
        """Predict using the AOM-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use. If None, uses k_selected_.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ["x_mean_", "x_std_", "y_mean_", "y_std_", "_B_coefs"])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        # Apply OPLS filter
        if self._P_orth is not None:
            for j in range(self._P_orth.shape[1]):
                p_o = self._P_orth[:, j]
                t_o = X_centered @ p_o
                X_centered = X_centered - np.outer(t_o, p_o)

        if n_components is None:
            n_components = self.k_selected_
        n_components = min(n_components, self.n_components_)

        if n_components == 0:
            y_pred = np.full((X.shape[0], len(self.y_mean_)), self.y_mean_, dtype=np.float64)
        else:
            B_k = self._B_coefs[n_components - 1]
            y_pred_std = X_centered @ B_k
            y_pred = y_pred_std * self.y_std_ + self.y_mean_

        if self._y_1d:
            y_pred = y_pred.ravel()
        return y_pred

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Transform X to score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, k_selected_)
            X scores.
        """
        check_is_fitted(self, ["x_mean_", "x_std_", "_W"])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        if self._P_orth is not None:
            for j in range(self._P_orth.shape[1]):
                p_o = self._P_orth[:, j]
                t_o = X_centered @ p_o
                X_centered = X_centered - np.outer(t_o, p_o)

        return X_centered @ self._W[:, :self.k_selected_]

    def get_block_weights(self) -> NDArray[np.floating]:
        """Get per-component block gating weights.

        Returns
        -------
        gamma : ndarray of shape (n_components_, n_blocks)
            Gating weights γ_{k,b}. Each row sums to 1 (approximately)
            and contains zeros for blocks not selected for that component.
        """
        check_is_fitted(self, ["gamma_"])
        return self.gamma_.copy()

    def get_preprocessing_report(self) -> list[dict]:
        """Get a human-readable report of preprocessing selections.

        Returns
        -------
        report : list of dict
            One entry per component with fields: 'component', 'blocks'
            (list of {name, weight} dicts for non-zero blocks).
        """
        check_is_fitted(self, ["gamma_", "block_names_"])
        report = []
        for k in range(self.n_components_):
            blocks = []
            for b, name in enumerate(self.block_names_):
                if self.gamma_[k, b] > 1e-6:
                    blocks.append({"name": name, "weight": float(self.gamma_[k, b])})
            blocks.sort(key=lambda x: x["weight"], reverse=True)
            report.append({"component": k + 1, "blocks": blocks})
        return report

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "n_components": self.n_components,
            "operator_bank": self.operator_bank,
            "gate": self.gate,
            "tau": self.tau,
            "n_orth": self.n_orth,
            "operator_index": self.operator_index,
            "center": self.center,
            "scale": self.scale,
            "selection": self.selection,
            "random_state": self.random_state,
            "backend": self.backend,
        }

    def set_params(self, **params) -> "AOMPLSRegressor":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"AOMPLSRegressor(n_components={self.n_components}, "
            f"tau={self.tau}, n_orth={self.n_orth}, "
            f"backend='{self.backend}')"
        )

"""Robust PLS (RSIMPLS) regressor for nirs4all.

A sklearn-compatible implementation of Robust PLS using iteratively
reweighted SIMPLS. This algorithm down-weights outliers using robust
weighting schemes (Huber or Tukey) to provide resistance against outliers
in both X and Y space.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
- Hubert, M., & Vanden Branden, K. (2003). Robust procedures for
  partial least squares regression. Chemometrics and Intelligent
  Laboratory Systems, 65(2), 101-121.
- Gil, J. A., & Romera, R. (1998). On robust partial least squares
  (PLS) methods. Journal of Chemometrics, 12(6), 365-378.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax
        return True
    except ImportError:
        return False


# =============================================================================
# Robust Weight Functions
# =============================================================================

def _huber_weights(
    residuals: NDArray[np.floating],
    c: float = 1.345,
) -> NDArray[np.floating]:
    """Compute Huber weights for robust regression.

    Huber's psi function provides a smooth transition between
    L2 (least squares) for small residuals and L1 for large residuals.

    Parameters
    ----------
    residuals : ndarray
        Standardized residuals (divided by robust scale estimate).
    c : float, default=1.345
        Tuning constant. The default gives 95% efficiency for normal data.

    Returns
    -------
    weights : ndarray
        Weights in [0, 1] for each observation.
    """
    abs_r = np.abs(residuals)
    # Avoid division by zero
    abs_r_safe = np.maximum(abs_r, 1e-10)
    weights = np.where(abs_r <= c, 1.0, c / abs_r_safe)
    return weights


def _tukey_weights(
    residuals: NDArray[np.floating],
    c: float = 4.685,
) -> NDArray[np.floating]:
    """Compute Tukey's bisquare weights for robust regression.

    Tukey's bisquare (biweight) function provides redescending weights
    that completely down-weight extreme outliers (zero weight beyond c).

    Parameters
    ----------
    residuals : ndarray
        Standardized residuals (divided by robust scale estimate).
    c : float, default=4.685
        Tuning constant. The default gives 95% efficiency for normal data.

    Returns
    -------
    weights : ndarray
        Weights in [0, 1] for each observation. Zero for |r| > c.
    """
    abs_r = np.abs(residuals)
    weights = np.where(abs_r <= c, (1 - (residuals / c) ** 2) ** 2, 0.0)
    return weights


def _mad_scale(x: NDArray[np.floating]) -> float:
    """Compute robust scale estimate using Median Absolute Deviation.

    Parameters
    ----------
    x : ndarray
        Data array.

    Returns
    -------
    scale : float
        Robust scale estimate (MAD * 1.4826 to be consistent with std for normal).
    """
    mad = median_abs_deviation(x, scale='normal')
    return float(max(mad, 1e-10))  # Avoid division by zero


# =============================================================================
# Weighted SIMPLS (shared NumPy implementation)
# =============================================================================

def _weighted_simpls_fit(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
    sample_weights: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],  # T (X scores)
    NDArray[np.floating],  # U (Y scores)
    NDArray[np.floating],  # W (X weights)
    NDArray[np.floating],  # P (X loadings)
    NDArray[np.floating],  # Q (Y loadings)
    NDArray[np.floating],  # R (rotation matrix)
    NDArray[np.floating],  # B (regression coefficients for each component)
]:
    """Fit weighted SIMPLS model using NumPy.

    This is the core SIMPLS algorithm with sample weights incorporated
    into the covariance computation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered/scaled X matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered/scaled Y matrix.
    n_components : int
        Number of PLS components.
    sample_weights : ndarray of shape (n_samples,)
        Sample weights for robust fitting.

    Returns
    -------
    T, U, W, P, Q, R, B : tuple of ndarrays
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # Storage for component outputs
    T = np.zeros((n_samples, n_components), dtype=np.float64)
    U = np.zeros((n_samples, n_components), dtype=np.float64)
    W = np.zeros((n_features, n_components), dtype=np.float64)
    P = np.zeros((n_features, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    B = np.zeros((n_components, n_features, n_targets), dtype=np.float64)

    # V for orthogonalization in SIMPLS-style deflation
    V = np.zeros((n_features, n_components), dtype=np.float64)

    # Weighted covariance matrix
    W_diag = np.diag(sample_weights)
    S = X.T @ W_diag @ Y  # (n_features, n_targets)

    for a in range(n_components):
        # Step 1: Get weight as dominant left singular vector of S
        if n_targets == 1:
            w = S[:, 0].copy()
        else:
            u_svd, s_svd, vh = np.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

        # Normalize weight
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # Step 2: Compute X score
        t = X @ w

        # Step 3: Weighted loadings
        ttW = t.T @ W_diag @ t
        if ttW < 1e-14:
            break

        p = (X.T @ W_diag @ t) / ttW  # X loading
        q = (Y.T @ W_diag @ t) / ttW  # Y loading

        # Y score
        u = Y @ q
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-14:
            u = u / u_norm

        # Store
        T[:, a] = t.ravel()
        U[:, a] = u.ravel()
        W[:, a] = w
        P[:, a] = p.ravel()
        Q[:, a] = q.ravel()

        # Orthogonalize v against previous v's
        v = p.ravel().copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p.ravel())

        v_norm = np.linalg.norm(v)
        if v_norm > 1e-14:
            v = v / v_norm
        V[:, a] = v

        # Deflate S
        S = S - v[:, np.newaxis] @ (v[np.newaxis, :] @ S)

        # Compute B for this component count
        W_a = W[:, :a+1]
        P_a = P[:, :a+1]
        Q_a = Q[:, :a+1]

        PtW_a = P_a.T @ W_a
        try:
            R_a = W_a @ np.linalg.inv(PtW_a)
        except np.linalg.LinAlgError:
            R_a = W_a @ np.linalg.pinv(PtW_a)

        B[a] = R_a @ Q_a.T

    # Compute final R
    PtW_final = P.T @ W
    try:
        R = W @ np.linalg.inv(PtW_final)
    except np.linalg.LinAlgError:
        R = W @ np.linalg.pinv(PtW_final)

    return T, U, W, P, Q, R, B


def _compute_irls_weights(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
    weighting: str = 'huber',
    max_iter: int = 100,
    tol: float = 1e-6,
    c: float | None = None,
) -> NDArray[np.floating]:
    """Compute IRLS sample weights for Robust PLS.

    This is always computed with NumPy for consistency between backends.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered/scaled X matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered/scaled Y matrix.
    n_components : int
        Number of PLS components.
    weighting : str
        Weighting scheme ('huber' or 'tukey').
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
    c : float or None
        Tuning constant for weight function.

    Returns
    -------
    sample_weights : ndarray of shape (n_samples,)
        Final sample weights from IRLS.
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # Set default tuning constant
    if c is None:
        c = 1.345 if weighting == 'huber' else 4.685

    # Weight function
    weight_func = _huber_weights if weighting == 'huber' else _tukey_weights

    # Initialize uniform weights
    sample_weights = np.ones(n_samples, dtype=np.float64)
    prev_weights = sample_weights.copy()

    for iteration in range(max_iter):
        # Fit weighted SIMPLS
        _, _, W_mat, P_mat, Q_mat, _, _ = _weighted_simpls_fit(
            X, Y, n_components, sample_weights
        )

        # Compute regression coefficients
        PtW = P_mat.T @ W_mat
        try:
            R = W_mat @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            R = W_mat @ np.linalg.pinv(PtW)

        B_current = R @ Q_mat.T  # (n_features, n_targets)

        # Compute residuals
        Y_pred = X @ B_current
        residuals = Y - Y_pred  # (n_samples, n_targets)

        # Compute combined residuals (handle multivariate Y)
        if n_targets > 1:
            res_combined = np.sqrt(np.sum(residuals ** 2, axis=1))
        else:
            res_combined = residuals.ravel()

        # Robust scale estimate
        scale = _mad_scale(res_combined)

        # Standardize residuals
        std_residuals = res_combined / scale

        # Compute new weights
        sample_weights = weight_func(std_residuals, c)

        # Ensure minimum weight to avoid zero weights everywhere
        min_weight = 0.01
        sample_weights = np.maximum(sample_weights, min_weight)

        # Normalize weights to sum to n_samples
        sample_weights = sample_weights * n_samples / sample_weights.sum()

        # Check convergence
        weight_change = np.linalg.norm(sample_weights - prev_weights)
        if weight_change < tol:
            break

        prev_weights = sample_weights.copy()

    return sample_weights


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_robust_pls_functions():
    """Lazy import and create JAX Robust PLS functions.

    Returns the JAX-accelerated fit and predict functions.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def weighted_simpls_fit_jax(
        X: jax.Array,
        Y: jax.Array,
        n_components: int,
        sample_weights: jax.Array,
    ) -> tuple:
        """JIT-compiled weighted SIMPLS fit using JAX.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Centered/scaled X matrix.
        Y : jax.Array of shape (n_samples, n_targets)
            Centered/scaled Y matrix.
        n_components : int
            Number of components.
        sample_weights : jax.Array of shape (n_samples,)
            Sample weights for robust fitting.

        Returns
        -------
        Tuple of T, U, W, P, Q, R, B.
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Weighted covariance matrix
        S = (X.T * sample_weights) @ Y

        def component_step(a, carry):
            S, T, U, W_mat, P, Q, V = carry

            # Get weight as dominant left singular vector of S
            u_svd, s_svd, vh = jnp.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

            # Normalize weight
            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-14, w / w_norm, w)

            # X score
            t = X @ w

            # Weighted inner product
            ttW = jnp.sum(t ** 2 * sample_weights)
            ttW_safe = jnp.where(ttW > 1e-14, ttW, 1.0)

            # Loadings
            p = (X.T * sample_weights) @ t / ttW_safe
            q = (Y.T * sample_weights) @ t / ttW_safe

            # Y score
            u = Y @ q
            u_norm = jnp.linalg.norm(u)
            u = jnp.where(u_norm > 1e-14, u / u_norm, u)

            # Store
            T = T.at[:, a].set(t)
            U = U.at[:, a].set(u)
            W_mat = W_mat.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q)

            # Orthogonalize v
            v = p.copy()
            prev_V = V * jnp.where(jnp.arange(n_components) < a, 1.0, 0.0)
            v = v - prev_V @ (prev_V.T @ p)

            v_norm = jnp.linalg.norm(v)
            v = jnp.where(v_norm > 1e-14, v / v_norm, v)
            V = V.at[:, a].set(v)

            # Deflate S
            S = S - jnp.outer(v, v @ S)

            return S, T, U, W_mat, P, Q, V

        # Initialize arrays
        T = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        U = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        W_mat = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        V = jnp.zeros((n_features, n_components), dtype=jnp.float64)

        init_carry = (S, T, U, W_mat, P, Q, V)
        S, T, U, W_mat, P, Q, V = lax.fori_loop(
            0, n_components, component_step, init_carry
        )

        # Compute B coefficients for each component count
        def compute_B(a, B):
            mask = jnp.where(jnp.arange(n_components) <= a, 1.0, 0.0)
            W_a = W_mat * mask
            P_a = P * mask
            Q_a = Q * mask

            PtW_a = P_a.T @ W_a
            R_a = W_a @ jnp.linalg.pinv(PtW_a)
            B_a = R_a @ Q_a.T
            B = B.at[a].set(B_a)
            return B

        B = jnp.zeros((n_components, n_features, n_targets), dtype=jnp.float64)
        B = lax.fori_loop(0, n_components, compute_B, B)

        # Final R
        PtW = P.T @ W_mat
        R = W_mat @ jnp.linalg.pinv(PtW)

        return T, U, W_mat, P, Q, R, B

    @jax.jit
    def robust_pls_predict_jax(
        X: jax.Array,
        X_mean: jax.Array,
        X_std: jax.Array,
        Y_mean: jax.Array,
        Y_std: jax.Array,
        B: jax.Array,
        n_components: int,
    ) -> jax.Array:
        """JIT-compiled Robust PLS predict using JAX."""
        X_centered = (X - X_mean) / X_std
        B_coef = B[n_components - 1]
        Y_pred_std = X_centered @ B_coef
        Y_pred = Y_pred_std * Y_std + Y_mean
        return Y_pred

    return weighted_simpls_fit_jax, robust_pls_predict_jax


# Cache for JAX functions
_JAX_ROBUST_PLS_FUNCS = None


def _get_cached_jax_robust_pls():
    """Get cached JAX Robust PLS functions."""
    global _JAX_ROBUST_PLS_FUNCS
    if _JAX_ROBUST_PLS_FUNCS is None:
        _JAX_ROBUST_PLS_FUNCS = _get_jax_robust_pls_functions()
    return _JAX_ROBUST_PLS_FUNCS


# =============================================================================
# RobustPLS Estimator Class
# =============================================================================

class RobustPLS(BaseEstimator, RegressorMixin):
    """Robust Partial Least Squares (Robust PLS) regressor.

    Robust PLS uses iteratively reweighted least squares (IRLS) to down-weight
    outliers during model fitting. This makes the model more resistant to
    outliers in both X (leverage points) and Y (vertical outliers).

    The algorithm iterates between:
    1. Fitting PLS with weighted covariance matrix
    2. Computing residuals and updating weights using robust M-estimation

    Two weighting schemes are available:
    - 'huber': Huber's psi function - smooth transition from L2 to L1
    - 'tukey': Tukey's bisquare - completely down-weights extreme outliers

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    weighting : {'huber', 'tukey'}, default='huber'
        Robust weighting scheme:
        - 'huber': Huber's psi function with smooth redescending.
        - 'tukey': Tukey's bisquare with hard rejection of outliers.
    c : float or None, default=None
        Tuning constant for the weight function. Controls the threshold
        beyond which observations are down-weighted.
        - For 'huber': default is 1.345 (95% efficiency)
        - For 'tukey': default is 4.685 (95% efficiency)
    max_iter : int, default=100
        Maximum number of IRLS iterations.
    tol : float, default=1e-6
        Convergence tolerance for weight changes.
    scale : bool, default=True
        Whether to scale X and Y to unit variance.
    center : bool, default=True
        Whether to center X and Y (subtract mean).
    backend : str, default='numpy'
        Computational backend to use:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU acceleration).
        Note: IRLS weight computation is always done in NumPy for consistency.
        The backend affects only the final PLS fit and prediction.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.
    x_mean_ : ndarray of shape (n_features,)
        Mean of X.
    x_std_ : ndarray of shape (n_features,)
        Standard deviation of X.
    y_mean_ : ndarray of shape (n_targets,)
        Mean of Y.
    y_std_ : ndarray of shape (n_targets,)
        Standard deviation of Y.
    x_scores_ : ndarray of shape (n_samples, n_components_)
        X scores (T).
    y_scores_ : ndarray of shape (n_samples, n_components_)
        Y scores (U).
    x_weights_ : ndarray of shape (n_features, n_components_)
        X weights (W).
    x_loadings_ : ndarray of shape (n_features, n_components_)
        X loadings (P).
    y_loadings_ : ndarray of shape (n_targets, n_components_)
        Y loadings (Q).
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.
    sample_weights_ : ndarray of shape (n_samples,)
        Final sample weights from IRLS. Low values indicate potential outliers.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.robust_pls import RobustPLS
    >>> import numpy as np
    >>> # Generate data with outliers
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(100)
    >>> # Add outliers
    >>> y[0:5] = y[0:5] + 10  # Vertical outliers
    >>> # Fit Robust PLS
    >>> model = RobustPLS(n_components=10, weighting='huber')
    >>> model.fit(X, y)
    RobustPLS(n_components=10, weighting='huber')
    >>> predictions = model.predict(X)
    >>> # Check which samples were down-weighted (potential outliers)
    >>> outlier_mask = model.sample_weights_ < 0.5
    >>> print(f"Potential outliers: {np.where(outlier_mask)[0]}")

    Notes
    -----
    Robust PLS is particularly useful when:
    - Data contains outliers in X or Y
    - Standard PLS gives poor predictions due to leverage points
    - You want to identify potential outliers via sample weights

    The sample_weights_ attribute can be used to identify outliers after fitting.
    Samples with low weights (e.g., < 0.5) may be outliers worth investigating.

    See Also
    --------
    SIMPLS : Standard SIMPLS algorithm without robust weighting.
    sklearn.cross_decomposition.PLSRegression : sklearn's PLS implementation.

    References
    ----------
    - Hubert, M., & Vanden Branden, K. (2003). Robust procedures for
      partial least squares regression. Chemometrics and Intelligent
      Laboratory Systems, 65(2), 101-121.
    - Gil, J. A., & Romera, R. (1998). On robust partial least squares
      (PLS) methods. Journal of Chemometrics, 12(6), 365-378.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        weighting: Literal['huber', 'tukey'] = 'huber',
        c: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        scale: bool = True,
        center: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize Robust PLS regressor."""
        self.n_components = n_components
        self.weighting = weighting
        self.c = c
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.center = center
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "RobustPLS":
        """Fit the Robust PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : RobustPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is not 'numpy' or 'jax'.
            If weighting is not 'huber' or 'tukey'.
        ImportError
            If backend is 'jax' and JAX is not installed.
        """
        # Validate parameters
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.weighting not in ('huber', 'tukey'):
            raise ValueError(
                f"weighting must be 'huber' or 'tukey', got '{self.weighting}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for RobustPLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Handle 1D y
        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features

        # Limit components by data dimensions
        max_components = min(n_samples - 1, n_features, n_samples)
        self.n_components_ = min(self.n_components, max_components)

        # Center and scale
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = np.zeros(n_targets, dtype=np.float64)

        if self.scale:
            self.x_std_ = X.std(axis=0, ddof=1)
            self.y_std_ = y.std(axis=0, ddof=1)
            # Avoid division by zero
            self.x_std_ = np.where(self.x_std_ < 1e-10, 1.0, self.x_std_)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.x_std_ = np.ones(n_features, dtype=np.float64)
            self.y_std_ = np.ones(n_targets, dtype=np.float64)

        X_centered = (X - self.x_mean_) / self.x_std_
        Y_centered = (y - self.y_mean_) / self.y_std_

        # Set tuning constant
        c = self.c
        if c is None:
            c = 1.345 if self.weighting == 'huber' else 4.685

        # Step 1: Compute IRLS weights using NumPy (always)
        # This ensures identical weights regardless of backend
        self.sample_weights_ = _compute_irls_weights(
            X_centered, Y_centered, self.n_components_,
            self.weighting, self.max_iter, self.tol, c
        )

        # Step 2: Final fit with converged weights using selected backend
        if self.backend == 'jax':
            import jax.numpy as jnp

            weighted_simpls_fit_jax, _ = _get_cached_jax_robust_pls()

            X_jax = jnp.asarray(X_centered)
            Y_jax = jnp.asarray(Y_centered)
            weights_jax = jnp.asarray(self.sample_weights_)

            T, U, W, P, Q, R, B = weighted_simpls_fit_jax(
                X_jax, Y_jax, self.n_components_, weights_jax
            )

            # Convert back to NumPy
            self.x_scores_ = np.asarray(T)
            self.y_scores_ = np.asarray(U)
            self.x_weights_ = np.asarray(W)
            self.x_loadings_ = np.asarray(P)
            self.y_loadings_ = np.asarray(Q)
            self._R = np.asarray(R)
            self._B = np.asarray(B)
        else:
            # NumPy backend
            T, U, W, P, Q, R, B = _weighted_simpls_fit(
                X_centered, Y_centered, self.n_components_, self.sample_weights_
            )

            self.x_scores_ = T
            self.y_scores_ = U
            self.x_weights_ = W
            self.x_loadings_ = P
            self.y_loadings_ = Q
            self._R = R
            self._B = B

        # Store final regression coefficients
        B_final = self._B[self.n_components_ - 1]
        self.coef_ = B_final * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: Union[int, None] = None,
    ) -> NDArray[np.floating]:
        """Predict using the Robust PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use for prediction.
            If None, uses all fitted components.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['x_mean_', 'x_std_', 'y_mean_', 'y_std_', '_B'])

        X = np.asarray(X, dtype=np.float64)

        if n_components is None:
            n_components = self.n_components_
        else:
            n_components = min(n_components, self.n_components_)

        if self.backend == 'jax':
            import jax.numpy as jnp

            _, robust_pls_predict_jax = _get_cached_jax_robust_pls()

            X_jax = jnp.asarray(X)
            x_mean_jax = jnp.asarray(self.x_mean_)
            x_std_jax = jnp.asarray(self.x_std_)
            y_mean_jax = jnp.asarray(self.y_mean_)
            y_std_jax = jnp.asarray(self.y_std_)
            B_jax = jnp.asarray(self._B)

            y_pred = robust_pls_predict_jax(
                X_jax, x_mean_jax, x_std_jax,
                y_mean_jax, y_std_jax, B_jax, n_components
            )
            y_pred = np.asarray(y_pred)
        else:
            # NumPy prediction
            X_centered = (X - self.x_mean_) / self.x_std_
            B_coef = self._B[n_components - 1]
            y_pred_std = X_centered @ B_coef
            y_pred = y_pred_std * self.y_std_ + self.y_mean_

        # Flatten if single target
        if self._y_1d:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Transform X to score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components_)
            X scores.
        """
        check_is_fitted(self, ['x_mean_', 'x_std_', '_R'])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        # Compute scores: T = X @ R
        T = X_centered @ self._R

        return T

    def get_outlier_mask(
        self,
        threshold: float = 0.5,
    ) -> NDArray[np.bool_]:
        """Get mask of potential outliers based on sample weights.

        Parameters
        ----------
        threshold : float, default=0.5
            Weight threshold below which samples are considered outliers.

        Returns
        -------
        outlier_mask : ndarray of shape (n_samples,)
            Boolean mask where True indicates potential outlier.
        """
        check_is_fitted(self, ['sample_weights_'])
        return self.sample_weights_ < threshold

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_components': self.n_components,
            'weighting': self.weighting,
            'c': self.c,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'scale': self.scale,
            'center': self.center,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "RobustPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : RobustPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RobustPLS(n_components={self.n_components}, "
            f"weighting='{self.weighting}', c={self.c}, "
            f"max_iter={self.max_iter}, tol={self.tol}, "
            f"scale={self.scale}, center={self.center}, "
            f"backend='{self.backend}')"
        )

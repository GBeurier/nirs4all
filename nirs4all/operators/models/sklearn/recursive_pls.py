"""Recursive PLS (RPLS) regressor for nirs4all.

A sklearn-compatible implementation of Recursive Partial Least Squares.
RPLS enables online model updates for drifting processes through incremental
updates using a forgetting factor.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
- Qin, S. J. (1998). Recursive PLS algorithms for adaptive data modeling.
  Computers & Chemical Engineering, 22(4-5), 503-514.
- Helland, K., Berntsen, H. E., Borgen, O. S., & Martens, H. (1992).
  Recursive algorithm for partial least squares regression.
  Chemometrics and Intelligent Laboratory Systems, 14(1-3), 129-137.
- Dayal, B. S., & MacGregor, J. F. (1997). Recursive exponentially
  weighted PLS and its applications to adaptive control and prediction.
  Journal of Process Control, 7(3), 169-179.
"""

from __future__ import annotations

from functools import partial
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
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
# NumPy Backend Implementation
# =============================================================================

def _initial_pls_fit_numpy(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
) -> tuple[
    NDArray[np.floating],  # W (X weights)
    NDArray[np.floating],  # P (X loadings)
    NDArray[np.floating],  # Q (Y loadings)
    NDArray[np.floating],  # R (W * inv(P'W) for direct regression)
    NDArray[np.floating],  # B (regression coefficients)
    NDArray[np.floating],  # Cov_X (running covariance of X)
    NDArray[np.floating],  # Cov_XY (running cross-covariance of X and Y)
]:
    """Initial batch PLS fit using SIMPLS-like algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered X matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered Y matrix.
    n_components : int
        Number of PLS components.

    Returns
    -------
    W, P, Q, R, B, Cov_X, Cov_XY : tuple of ndarrays
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # Initialize arrays
    W = np.zeros((n_features, n_components), dtype=np.float64)
    P = np.zeros((n_features, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    V = np.zeros((n_features, n_components), dtype=np.float64)

    # Initial covariance matrices
    Cov_X = X.T @ X  # (n_features, n_features)
    Cov_XY = X.T @ Y  # (n_features, n_targets)
    S = Cov_XY.copy()

    for a in range(n_components):
        # Get weight as dominant left singular vector of S
        if n_targets == 1:
            w = S[:, 0].copy()
        else:
            u_svd, s_svd, vh = np.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

        # Normalize
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # Score and loadings
        t = X @ w
        tt = t.T @ t
        if tt < 1e-14:
            break

        p = (X.T @ t) / tt
        q = (Y.T @ t) / tt

        # Store
        W[:, a] = w
        P[:, a] = p.ravel()
        Q[:, a] = q.ravel()

        # Orthogonalization for deflation
        v = p.ravel().copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p.ravel())
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-14:
            v = v / v_norm
        V[:, a] = v

        # Deflate S
        S = S - v[:, np.newaxis] @ (v[np.newaxis, :] @ S)

    # Compute R = W @ inv(P'W)
    PtW = P.T @ W
    try:
        R = W @ np.linalg.inv(PtW)
    except np.linalg.LinAlgError:
        R = W @ np.linalg.pinv(PtW)

    # Compute regression coefficients
    B = R @ Q.T

    return W, P, Q, R, B, Cov_X, Cov_XY


def _recursive_update_numpy(
    X_new: NDArray[np.floating],
    Y_new: NDArray[np.floating],
    n_components: int,
    W: NDArray[np.floating],
    P: NDArray[np.floating],
    Q: NDArray[np.floating],
    Cov_X: NDArray[np.floating],
    Cov_XY: NDArray[np.floating],
    forgetting_factor: float,
    n_samples_seen: int,
) -> tuple[
    NDArray[np.floating],  # W
    NDArray[np.floating],  # P
    NDArray[np.floating],  # Q
    NDArray[np.floating],  # R
    NDArray[np.floating],  # B
    NDArray[np.floating],  # Cov_X
    NDArray[np.floating],  # Cov_XY
    int,  # n_samples_seen
]:
    """Recursive update of PLS model.

    Uses exponentially weighted recursive least squares to update
    the covariance matrices and recompute PLS loadings.

    Parameters
    ----------
    X_new : ndarray of shape (n_new_samples, n_features)
        New centered X samples.
    Y_new : ndarray of shape (n_new_samples, n_targets)
        New centered Y samples.
    n_components : int
        Number of PLS components.
    W, P, Q : ndarrays
        Current weights and loadings.
    Cov_X, Cov_XY : ndarrays
        Current covariance matrices.
    forgetting_factor : float
        Forgetting factor in (0, 1]. Lower values forget faster.
    n_samples_seen : int
        Number of samples seen so far.

    Returns
    -------
    Updated W, P, Q, R, B, Cov_X, Cov_XY, n_samples_seen
    """
    n_new = X_new.shape[0]
    n_features = X_new.shape[1]
    n_targets = Y_new.shape[1]

    ff = forgetting_factor

    # Update covariance matrices with exponential weighting
    for i in range(n_new):
        x_i = X_new[i:i+1].T  # (n_features, 1)
        y_i = Y_new[i:i+1].T  # (n_targets, 1)

        # Exponentially weighted update
        Cov_X = ff * Cov_X + x_i @ x_i.T
        Cov_XY = ff * Cov_XY + x_i @ y_i.T
        n_samples_seen += 1

    # Recompute PLS components from updated covariances
    # This is the key step: use updated covariances to get new loadings

    V = np.zeros((n_features, n_components), dtype=np.float64)
    S = Cov_XY.copy()

    for a in range(n_components):
        # Get weight from covariance
        if n_targets == 1:
            w = S[:, 0].copy()
        else:
            u_svd, s_svd, vh = np.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # Loading from covariance: p = Cov_X @ w / (w' @ Cov_X @ w)
        Cov_X_w = Cov_X @ w
        denominator = w.T @ Cov_X_w
        if np.abs(denominator) < 1e-14:
            break

        p = Cov_X_w / denominator
        q = Cov_XY.T @ w / denominator

        W[:, a] = w
        P[:, a] = p
        Q[:, a] = q

        # Orthogonalization
        v = p.copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-14:
            v = v / v_norm
        V[:, a] = v

        # Deflate S
        S = S - v[:, np.newaxis] @ (v[np.newaxis, :] @ S)

    # Compute R and B
    PtW = P.T @ W
    try:
        R = W @ np.linalg.inv(PtW)
    except np.linalg.LinAlgError:
        R = W @ np.linalg.pinv(PtW)

    B = R @ Q.T

    return W, P, Q, R, B, Cov_X, Cov_XY, n_samples_seen


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_recursive_pls_functions():
    """Lazy import and create JAX Recursive PLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def initial_pls_fit_jax(
        X: jax.Array,
        Y: jax.Array,
        n_components: int,
    ) -> tuple:
        """JIT-compiled initial PLS fit using JAX."""
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Initial covariance matrices
        Cov_X = X.T @ X
        Cov_XY = X.T @ Y
        S = Cov_XY.copy()

        def component_step(a, carry):
            S, W, P, Q, V = carry

            # Get weight from SVD
            u_svd, s_svd, vh = jnp.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

            # Normalize
            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-14, w / w_norm, w)

            # Score and loadings
            t = X @ w
            tt = t.T @ t
            tt_safe = jnp.where(tt > 1e-14, tt, 1.0)

            p = (X.T @ t) / tt_safe
            q = (Y.T @ t) / tt_safe

            # Store
            W = W.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q)

            # Orthogonalization
            v = p.copy()
            prev_V = V * jnp.where(jnp.arange(n_components) < a, 1.0, 0.0)
            v = v - prev_V @ (prev_V.T @ p)
            v_norm = jnp.linalg.norm(v)
            v = jnp.where(v_norm > 1e-14, v / v_norm, v)
            V = V.at[:, a].set(v)

            # Deflate
            S = S - jnp.outer(v, v @ S)

            return S, W, P, Q, V

        # Initialize
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        V = jnp.zeros((n_features, n_components), dtype=jnp.float64)

        init_carry = (S, W, P, Q, V)
        S, W, P, Q, V = lax.fori_loop(0, n_components, component_step, init_carry)

        # Compute R and B
        PtW = P.T @ W
        R = W @ jnp.linalg.pinv(PtW)
        B = R @ Q.T

        return W, P, Q, R, B, Cov_X, Cov_XY

    @partial(jax.jit, static_argnums=(2,))
    def recursive_update_jax(
        X_new: jax.Array,
        Y_new: jax.Array,
        n_components: int,
        W: jax.Array,
        P: jax.Array,
        Q: jax.Array,
        Cov_X: jax.Array,
        Cov_XY: jax.Array,
        forgetting_factor: float,
    ) -> tuple:
        """JIT-compiled recursive PLS update using JAX."""
        n_new = X_new.shape[0]
        n_features = X_new.shape[1]
        n_targets = Y_new.shape[1]

        ff = forgetting_factor

        # Update covariances in a loop using lax.dynamic_slice for JAX compatibility
        def update_cov_step(i, carry):
            Cov_X, Cov_XY = carry
            # Use lax.dynamic_slice instead of Python slicing
            x_i = lax.dynamic_slice(X_new, (i, 0), (1, n_features)).T
            y_i = lax.dynamic_slice(Y_new, (i, 0), (1, n_targets)).T
            Cov_X = ff * Cov_X + x_i @ x_i.T
            Cov_XY = ff * Cov_XY + x_i @ y_i.T
            return Cov_X, Cov_XY

        Cov_X, Cov_XY = lax.fori_loop(0, n_new, update_cov_step, (Cov_X, Cov_XY))

        # Recompute PLS from updated covariances
        S = Cov_XY.copy()

        def component_step(a, carry):
            S, W, P, Q, V = carry

            # Get weight
            u_svd, s_svd, vh = jnp.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-14, w / w_norm, w)

            # Loading from covariance
            Cov_X_w = Cov_X @ w
            denom = w.T @ Cov_X_w
            denom_safe = jnp.where(jnp.abs(denom) > 1e-14, denom, 1.0)

            p = Cov_X_w / denom_safe
            q = Cov_XY.T @ w / denom_safe

            W = W.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q)

            # Orthogonalization
            v = p.copy()
            prev_V = V * jnp.where(jnp.arange(n_components) < a, 1.0, 0.0)
            v = v - prev_V @ (prev_V.T @ p)
            v_norm = jnp.linalg.norm(v)
            v = jnp.where(v_norm > 1e-14, v / v_norm, v)
            V = V.at[:, a].set(v)

            # Deflate
            S = S - jnp.outer(v, v @ S)

            return S, W, P, Q, V

        V = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        init_carry = (S, W, P, Q, V)
        S, W, P, Q, V = lax.fori_loop(0, n_components, component_step, init_carry)

        # Compute R and B
        PtW = P.T @ W
        R = W @ jnp.linalg.pinv(PtW)
        B = R @ Q.T

        return W, P, Q, R, B, Cov_X, Cov_XY

    @jax.jit
    def predict_jax(
        X: jax.Array,
        X_mean: jax.Array,
        X_std: jax.Array,
        Y_mean: jax.Array,
        Y_std: jax.Array,
        B: jax.Array,
    ) -> jax.Array:
        """JIT-compiled prediction using JAX."""
        X_centered = (X - X_mean) / X_std
        Y_pred_std = X_centered @ B
        Y_pred = Y_pred_std * Y_std + Y_mean
        return Y_pred

    return initial_pls_fit_jax, recursive_update_jax, predict_jax


# Cache for JAX functions
_JAX_RECURSIVE_PLS_FUNCS = None


def _get_cached_jax_recursive_pls():
    """Get cached JAX Recursive PLS functions."""
    global _JAX_RECURSIVE_PLS_FUNCS
    if _JAX_RECURSIVE_PLS_FUNCS is None:
        _JAX_RECURSIVE_PLS_FUNCS = _get_jax_recursive_pls_functions()
    return _JAX_RECURSIVE_PLS_FUNCS


# =============================================================================
# RecursivePLS Estimator Class
# =============================================================================

class RecursivePLS(BaseEstimator, RegressorMixin):
    """Recursive Partial Least Squares (Recursive PLS) regressor.

    Recursive PLS enables online model updates for drifting processes.
    It uses a forgetting factor to exponentially weight old samples,
    allowing the model to adapt to non-stationary data streams.

    The algorithm maintains running covariance matrices that are updated
    incrementally with each new batch of samples. The PLS loadings are
    then recomputed from these updated covariances.

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    forgetting_factor : float, default=0.99
        Forgetting factor in (0, 1]. Controls the rate of adaptation:
        - 1.0: No forgetting, standard batch PLS
        - <1.0: Exponential forgetting of old samples
        - Typical values: 0.95-0.999 depending on drift speed
    scale : bool, default=True
        Whether to scale X and Y to unit variance.
    center : bool, default=True
        Whether to center X and Y (subtract mean).
    backend : str, default='numpy'
        Computational backend to use:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU acceleration).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.
    n_samples_seen_ : int
        Total number of samples seen (including partial_fit calls).
    x_mean_ : ndarray of shape (n_features,)
        Mean of X (updated with exponential moving average).
    x_std_ : ndarray of shape (n_features,)
        Standard deviation of X.
    y_mean_ : ndarray of shape (n_targets,)
        Mean of Y (updated with exponential moving average).
    y_std_ : ndarray of shape (n_targets,)
        Standard deviation of Y.
    x_weights_ : ndarray of shape (n_features, n_components_)
        X weights (W).
    x_loadings_ : ndarray of shape (n_features, n_components_)
        X loadings (P).
    y_loadings_ : ndarray of shape (n_targets, n_components_)
        Y loadings (Q).
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.recursive_pls import RecursivePLS
    >>> import numpy as np
    >>> # Initial batch fit
    >>> np.random.seed(42)
    >>> X_init = np.random.randn(100, 50)
    >>> y_init = X_init[:, :5].sum(axis=1) + 0.1 * np.random.randn(100)
    >>> model = RecursivePLS(n_components=10, forgetting_factor=0.99)
    >>> model.fit(X_init, y_init)
    RecursivePLS(n_components=10)
    >>> # Online update with new samples
    >>> X_new = np.random.randn(10, 50)
    >>> y_new = X_new[:, :5].sum(axis=1) + 0.1 * np.random.randn(10)
    >>> model.partial_fit(X_new, y_new)
    >>> # Predict
    >>> predictions = model.predict(X_new)
    >>> print(f"Samples seen: {model.n_samples_seen_}")

    Notes
    -----
    Recursive PLS is particularly useful when:
    - Data arrives in streams and batch retraining is too expensive
    - Process conditions drift over time (sensor aging, raw material changes)
    - You need to adapt a calibration model to local conditions

    The forgetting factor controls the adaptation speed:
    - Higher values (0.99-0.999): Slow adaptation, stable model
    - Lower values (0.9-0.95): Fast adaptation, may be unstable

    See Also
    --------
    SIMPLS : Batch SIMPLS algorithm.
    sklearn.cross_decomposition.PLSRegression : sklearn's batch PLS.

    References
    ----------
    - Qin, S. J. (1998). Recursive PLS algorithms for adaptive data
      modeling. Computers & Chemical Engineering, 22(4-5), 503-514.
    - Dayal, B. S., & MacGregor, J. F. (1997). Recursive exponentially
      weighted PLS and its applications to adaptive control and prediction.
      Journal of Process Control, 7(3), 169-179.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        forgetting_factor: float = 0.99,
        scale: bool = True,
        center: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize Recursive PLS regressor."""
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor
        self.scale = scale
        self.center = center
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "RecursivePLS":
        """Fit the Recursive PLS model with initial batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : RecursivePLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is not 'numpy' or 'jax'.
            If forgetting_factor is not in (0, 1].
        ImportError
            If backend is 'jax' and JAX is not installed.
        """
        # Validate backend
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for RecursivePLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        # Validate forgetting factor
        if not (0 < self.forgetting_factor <= 1):
            raise ValueError(
                f"forgetting_factor must be in (0, 1], got {self.forgetting_factor}"
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
        self.n_samples_seen_ = n_samples

        # Limit components
        max_components = min(n_samples - 1, n_features)
        self.n_components_ = min(self.n_components, max_components)

        # Compute and store scaling parameters
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(n_features, dtype=np.float64)
            self.y_mean_ = np.zeros(n_targets, dtype=np.float64)

        if self.scale:
            self.x_std_ = X.std(axis=0, ddof=1)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.x_std_ = np.where(self.x_std_ < 1e-10, 1.0, self.x_std_)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.x_std_ = np.ones(n_features, dtype=np.float64)
            self.y_std_ = np.ones(n_targets, dtype=np.float64)

        X_centered = (X - self.x_mean_) / self.x_std_
        Y_centered = (y - self.y_mean_) / self.y_std_

        # Initial fit
        if self.backend == 'jax':
            import jax.numpy as jnp

            fit_jax, _, _ = _get_cached_jax_recursive_pls()

            X_jax = jnp.asarray(X_centered)
            Y_jax = jnp.asarray(Y_centered)

            W, P, Q, R, B, Cov_X, Cov_XY = fit_jax(
                X_jax, Y_jax, self.n_components_
            )

            self.x_weights_ = np.asarray(W)
            self.x_loadings_ = np.asarray(P)
            self.y_loadings_ = np.asarray(Q)
            self._R = np.asarray(R)
            self._B = np.asarray(B)
            self._Cov_X = np.asarray(Cov_X)
            self._Cov_XY = np.asarray(Cov_XY)
        else:
            W, P, Q, R, B, Cov_X, Cov_XY = _initial_pls_fit_numpy(
                X_centered, Y_centered, self.n_components_
            )

            self.x_weights_ = W
            self.x_loadings_ = P
            self.y_loadings_ = Q
            self._R = R
            self._B = B
            self._Cov_X = Cov_X
            self._Cov_XY = Cov_XY

        # Store final coefficients in original space
        self.coef_ = self._B * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]

        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "RecursivePLS":
        """Update the Recursive PLS model with new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            New target values.

        Returns
        -------
        self : RecursivePLS
            Updated estimator.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, ['x_mean_', 'x_std_', '_B', '_Cov_X', '_Cov_XY'])

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self._y_1d:
            y = y.reshape(-1, 1)

        n_new = X.shape[0]

        # Update running mean with exponential moving average
        ff = self.forgetting_factor
        for i in range(n_new):
            # EMA update for mean (optional: can also keep fixed from initial fit)
            # self.x_mean_ = ff * self.x_mean_ + (1 - ff) * X[i]
            # self.y_mean_ = ff * self.y_mean_ + (1 - ff) * y[i]
            pass  # Keep mean fixed for stability

        # Center new data using stored parameters
        X_centered = (X - self.x_mean_) / self.x_std_
        Y_centered = (y - self.y_mean_) / self.y_std_

        # Recursive update
        if self.backend == 'jax':
            import jax.numpy as jnp

            _, update_jax, _ = _get_cached_jax_recursive_pls()

            X_jax = jnp.asarray(X_centered)
            Y_jax = jnp.asarray(Y_centered)
            W_jax = jnp.asarray(self.x_weights_)
            P_jax = jnp.asarray(self.x_loadings_)
            Q_jax = jnp.asarray(self.y_loadings_)
            Cov_X_jax = jnp.asarray(self._Cov_X)
            Cov_XY_jax = jnp.asarray(self._Cov_XY)

            W, P, Q, R, B, Cov_X, Cov_XY = update_jax(
                X_jax, Y_jax, self.n_components_,
                W_jax, P_jax, Q_jax, Cov_X_jax, Cov_XY_jax,
                self.forgetting_factor
            )

            self.x_weights_ = np.asarray(W)
            self.x_loadings_ = np.asarray(P)
            self.y_loadings_ = np.asarray(Q)
            self._R = np.asarray(R)
            self._B = np.asarray(B)
            self._Cov_X = np.asarray(Cov_X)
            self._Cov_XY = np.asarray(Cov_XY)
        else:
            W, P, Q, R, B, Cov_X, Cov_XY, n_seen = _recursive_update_numpy(
                X_centered, Y_centered, self.n_components_,
                self.x_weights_, self.x_loadings_, self.y_loadings_,
                self._Cov_X, self._Cov_XY,
                self.forgetting_factor, self.n_samples_seen_
            )

            self.x_weights_ = W
            self.x_loadings_ = P
            self.y_loadings_ = Q
            self._R = R
            self._B = B
            self._Cov_X = Cov_X
            self._Cov_XY = Cov_XY

        self.n_samples_seen_ += n_new

        # Update coefficients
        self.coef_ = self._B * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Predict using the Recursive PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['x_mean_', 'x_std_', 'y_mean_', 'y_std_', '_B'])

        X = np.asarray(X, dtype=np.float64)

        if self.backend == 'jax':
            import jax.numpy as jnp

            _, _, predict_jax = _get_cached_jax_recursive_pls()

            X_jax = jnp.asarray(X)
            x_mean_jax = jnp.asarray(self.x_mean_)
            x_std_jax = jnp.asarray(self.x_std_)
            y_mean_jax = jnp.asarray(self.y_mean_)
            y_std_jax = jnp.asarray(self.y_std_)
            B_jax = jnp.asarray(self._B)

            y_pred = predict_jax(
                X_jax, x_mean_jax, x_std_jax, y_mean_jax, y_std_jax, B_jax
            )
            y_pred = np.asarray(y_pred)
        else:
            X_centered = (X - self.x_mean_) / self.x_std_
            y_pred_std = X_centered @ self._B
            y_pred = y_pred_std * self.y_std_ + self.y_mean_

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

        T = X_centered @ self._R

        return T

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
            'forgetting_factor': self.forgetting_factor,
            'scale': self.scale,
            'center': self.center,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "RecursivePLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : RecursivePLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RecursivePLS(n_components={self.n_components}, "
            f"forgetting_factor={self.forgetting_factor}, "
            f"scale={self.scale}, center={self.center}, "
            f"backend='{self.backend}')"
        )

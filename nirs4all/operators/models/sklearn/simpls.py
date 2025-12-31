"""SIMPLS (Simple PLS) regressor for nirs4all.

A sklearn-compatible implementation of the SIMPLS algorithm by de Jong (1993).
SIMPLS is an alternative to NIPALS that computes PLS components via projections
of the covariance matrix, avoiding the iterative deflation of X.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
- de Jong, S. (1993). SIMPLS: An alternative approach to partial least
  squares regression. Chemometrics and Intelligent Laboratory Systems,
  18(3), 251-263. https://doi.org/10.1016/0169-7439(93)85002-X
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

def _simpls_fit_numpy(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
) -> tuple[
    NDArray[np.floating],  # T (X scores)
    NDArray[np.floating],  # U (Y scores)
    NDArray[np.floating],  # W (X weights)
    NDArray[np.floating],  # P (X loadings)
    NDArray[np.floating],  # Q (Y loadings)
    NDArray[np.floating],  # R (rotation matrix for regression)
    NDArray[np.floating],  # B (regression coefficients for each component)
]:
    """Fit SIMPLS model using NumPy.

    SIMPLS algorithm (de Jong, 1993):
    1. Compute weight w as dominant left singular vector of X'Y.
    2. Compute X score t = Xw (not normalized for loadings computation).
    3. Compute loadings p = X't / (t't), q = Y't / (t't).
    4. Orthogonalize and deflate the covariance matrix S.
    5. Repeat for each component.
    6. Compute regression coefficients B = R @ Q.T where R = W @ inv(P.T @ W).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered/scaled X matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Centered/scaled Y matrix.
    n_components : int
        Number of PLS components to extract.

    Returns
    -------
    T : ndarray of shape (n_samples, n_components)
        X scores.
    U : ndarray of shape (n_samples, n_components)
        Y scores.
    W : ndarray of shape (n_features, n_components)
        X weights.
    P : ndarray of shape (n_features, n_components)
        X loadings.
    Q : ndarray of shape (n_targets, n_components)
        Y loadings.
    R : ndarray of shape (n_features, n_components)
        Rotation matrix (R = W @ (P'W)^-1).
    B : ndarray of shape (n_components, n_features, n_targets)
        Regression coefficients for each component count.
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # Initialize output arrays
    T = np.zeros((n_samples, n_components), dtype=np.float64)
    U = np.zeros((n_samples, n_components), dtype=np.float64)
    W = np.zeros((n_features, n_components), dtype=np.float64)
    P = np.zeros((n_features, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    B = np.zeros((n_components, n_features, n_targets), dtype=np.float64)

    # V is used to keep the orthonormal basis for deflating S
    V = np.zeros((n_features, n_components), dtype=np.float64)

    # Initial covariance matrix
    S = X.T @ Y  # (n_features, n_targets)

    n_components_actual = 0

    for a in range(n_components):
        # Step 1: Get weight as dominant left singular vector of S
        if n_targets == 1:
            w = S[:, 0].copy()
        else:
            # SVD of S to get dominant left singular vector
            u_svd, s_svd, vh = np.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

        # Normalize weight
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-14:
            break
        w = w / w_norm

        # Step 2: Compute X score (NOT normalized for loadings)
        t = X @ w

        # Step 3: Compute loadings
        tt = t.T @ t
        if tt < 1e-14:
            break

        p = (X.T @ t) / tt  # X loading
        q = (Y.T @ t) / tt  # Y loading

        # Step 4: Compute u (Y score) for completeness
        u = Y @ q
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-14:
            u = u / u_norm

        # Step 5: Store results
        T[:, a] = t.ravel()
        U[:, a] = u.ravel()
        W[:, a] = w
        P[:, a] = p.ravel()
        Q[:, a] = q.ravel()

        n_components_actual = a + 1

        # Step 6: Orthogonalize v against previous v's (for S deflation)
        v = p.ravel().copy()
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ p.ravel())

        v_norm = np.linalg.norm(v)
        if v_norm > 1e-14:
            v = v / v_norm
        V[:, a] = v

        # Step 7: Deflate S
        S = S - v[:, np.newaxis] @ (v[np.newaxis, :] @ S)

        # Step 8: Compute regression coefficients using a+1 components
        # B = R @ Q.T where R = W @ inv(P.T @ W)
        W_a = W[:, :a+1]
        P_a = P[:, :a+1]
        Q_a = Q[:, :a+1]

        PtW = P_a.T @ W_a
        try:
            R_a = W_a @ np.linalg.inv(PtW)
        except np.linalg.LinAlgError:
            R_a = W_a @ np.linalg.pinv(PtW)

        B[a] = R_a @ Q_a.T

    # Compute R (rotation matrix) for transform
    if n_components_actual > 0:
        W_final = W[:, :n_components_actual]
        P_final = P[:, :n_components_actual]
        PtW_final = P_final.T @ W_final
        try:
            R = W_final @ np.linalg.inv(PtW_final)
        except np.linalg.LinAlgError:
            R = W_final @ np.linalg.pinv(PtW_final)
        # Pad R to full n_components size
        R_full = np.zeros((n_features, n_components), dtype=np.float64)
        R_full[:, :n_components_actual] = R
        R = R_full
    else:
        R = np.zeros((n_features, n_components), dtype=np.float64)

    return T, U, W, P, Q, R, B


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_simpls_functions():
    """Lazy import and create JAX SIMPLS functions.

    Returns the JAX-accelerated fit and predict functions.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def simpls_fit_jax(
        X: jax.Array,
        Y: jax.Array,
        n_components: int,
    ) -> tuple:
        """JIT-compiled SIMPLS fit using JAX.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Centered/scaled X matrix.
        Y : jax.Array of shape (n_samples, n_targets)
            Centered/scaled Y matrix.
        n_components : int
            Number of components to extract.

        Returns
        -------
        Tuple containing T, U, W, P, Q, R, B arrays.
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Initial covariance matrix
        S = X.T @ Y

        def component_step(a, carry):
            S, T, U, W, P, Q, V = carry

            # Get weight as dominant left singular vector of S
            u_svd, s_svd, vh = jnp.linalg.svd(S, full_matrices=False)
            w = u_svd[:, 0]

            # Normalize weight
            w_norm = jnp.linalg.norm(w)
            w = jnp.where(w_norm > 1e-14, w / w_norm, w)

            # Compute X score (NOT normalized for loadings)
            t = X @ w

            # Compute loadings: p = X'*t / (t'*t), q = Y'*t / (t'*t)
            tt = t.T @ t
            tt_safe = jnp.where(tt > 1e-14, tt, 1.0)
            p = (X.T @ t) / tt_safe
            q = (Y.T @ t) / tt_safe

            # Compute u (Y score)
            u = Y @ q
            u_norm = jnp.linalg.norm(u)
            u = jnp.where(u_norm > 1e-14, u / u_norm, u)

            # Store results
            T = T.at[:, a].set(t)
            U = U.at[:, a].set(u)
            W = W.at[:, a].set(w)
            P = P.at[:, a].set(p)
            Q = Q.at[:, a].set(q)

            # Orthogonalize v against previous v's
            v = p.copy()
            # Use masking for previous components
            prev_V = V * jnp.where(jnp.arange(n_components) < a, 1.0, 0.0)
            v = v - prev_V @ (prev_V.T @ p)

            v_norm = jnp.linalg.norm(v)
            v = jnp.where(v_norm > 1e-14, v / v_norm, v)
            V = V.at[:, a].set(v)

            # Deflate S
            S = S - jnp.outer(v, v @ S)

            return S, T, U, W, P, Q, V

        # Initialize arrays
        T = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        U = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        V = jnp.zeros((n_features, n_components), dtype=jnp.float64)

        # Run component loop
        init_carry = (S, T, U, W, P, Q, V)
        S, T, U, W, P, Q, V = lax.fori_loop(
            0, n_components, component_step, init_carry
        )

        # Compute B coefficients for each component count
        # B[a] = R @ Q.T where R = W @ inv(P.T @ W)
        def compute_B(a, B):
            # Create masks for components up to and including a
            mask = jnp.where(jnp.arange(n_components) <= a, 1.0, 0.0)
            W_a = W * mask
            P_a = P * mask
            Q_a = Q * mask

            # Compute PtW (using masked matrices)
            PtW = P_a.T @ W_a

            # Compute R = W @ inv(PtW) using pseudoinverse for stability
            R_a = W_a @ jnp.linalg.pinv(PtW)

            # B = R @ Q.T
            B_a = R_a @ Q_a.T
            B = B.at[a].set(B_a)
            return B

        B = jnp.zeros((n_components, n_features, n_targets), dtype=jnp.float64)
        B = lax.fori_loop(0, n_components, compute_B, B)

        # Compute final R for transform
        PtW = P.T @ W
        R = W @ jnp.linalg.pinv(PtW)

        return T, U, W, P, Q, R, B

    @jax.jit
    def simpls_predict_jax(
        X: jax.Array,
        X_mean: jax.Array,
        X_std: jax.Array,
        Y_mean: jax.Array,
        Y_std: jax.Array,
        B: jax.Array,
        n_components: int,
    ) -> jax.Array:
        """JIT-compiled SIMPLS predict using JAX.

        Parameters
        ----------
        X : jax.Array of shape (n_samples, n_features)
            Input data.
        X_mean : jax.Array of shape (n_features,)
            Mean of training X.
        X_std : jax.Array of shape (n_features,)
            Std of training X.
        Y_mean : jax.Array of shape (n_targets,)
            Mean of training Y.
        Y_std : jax.Array of shape (n_targets,)
            Std of training Y.
        B : jax.Array of shape (n_components, n_features, n_targets)
            Regression coefficients.
        n_components : int
            Number of components to use.

        Returns
        -------
        Y_pred : jax.Array of shape (n_samples, n_targets)
            Predicted values.
        """
        # Standardize X
        X_centered = (X - X_mean) / X_std

        # Apply regression coefficients
        # B has shape (n_components, n_features, n_targets)
        # We use B[n_components-1] for the coefficients with n_components
        B_coef = B[n_components - 1]
        Y_pred_std = X_centered @ B_coef

        # Inverse transform
        Y_pred = Y_pred_std * Y_std + Y_mean

        return Y_pred

    return simpls_fit_jax, simpls_predict_jax


# Cache for JAX functions
_JAX_SIMPLS_FUNCS = None


def _get_cached_jax_simpls():
    """Get cached JAX SIMPLS functions."""
    global _JAX_SIMPLS_FUNCS
    if _JAX_SIMPLS_FUNCS is None:
        _JAX_SIMPLS_FUNCS = _get_jax_simpls_functions()
    return _JAX_SIMPLS_FUNCS


# =============================================================================
# SIMPLS Estimator Class
# =============================================================================

class SIMPLS(BaseEstimator, RegressorMixin):
    """SIMPLS (Simple PLS) regressor.

    SIMPLS is an alternative to NIPALS-based PLS that computes components
    via projections of the covariance matrix X'Y. It produces the same
    predictions as PLSRegression for univariate Y, and slightly different
    (but equivalent in terms of prediction accuracy) results for multivariate Y.

    SIMPLS is often faster than NIPALS for high-dimensional data because
    it avoids the iterative deflation of X.

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    scale : bool, default=True
        Whether to scale X and Y to unit variance.
    center : bool, default=True
        Whether to center X and Y (subtract mean).
    backend : str, default='numpy'
        Computational backend to use:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU acceleration).
        JAX backend requires JAX to be installed: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used (may be less than n_components
        if limited by data dimensions).
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
        Regression coefficients (using all components).

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.simpls import SIMPLS
    >>> import numpy as np
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(100)
    >>> # Fit SIMPLS model
    >>> model = SIMPLS(n_components=10)
    >>> model.fit(X, y)
    SIMPLS(n_components=10)
    >>> predictions = model.predict(X)
    >>> # Use JAX backend for GPU acceleration
    >>> model_jax = SIMPLS(n_components=10, backend='jax')

    Notes
    -----
    SIMPLS differs from NIPALS in how the deflation is performed:
    - NIPALS deflates X after each component (X := X - t*p')
    - SIMPLS deflates the covariance matrix S = X'Y

    For univariate Y, both methods produce identical predictions.
    For multivariate Y, SIMPLS produces Y loadings that span the same
    space as NIPALS but with slightly different orientations.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : sklearn's NIPALS-based PLS.
    IKPLS : Fast PLS implementation from the ikpls package.

    References
    ----------
    - de Jong, S. (1993). SIMPLS: An alternative approach to partial
      least squares regression. Chemometrics and Intelligent Laboratory
      Systems, 18(3), 251-263.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        scale: bool = True,
        center: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize SIMPLS regressor.

        Parameters
        ----------
        n_components : int, default=10
            Number of PLS components to extract.
        scale : bool, default=True
            Whether to scale X and Y to unit variance.
        center : bool, default=True
            Whether to center X and Y.
        backend : str, default='numpy'
            Computational backend ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "SIMPLS":
        """Fit the SIMPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : SIMPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is not 'numpy' or 'jax'.
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
                "JAX is required for SIMPLS with backend='jax'. "
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

        # Fit using appropriate backend
        if self.backend == 'jax':
            import jax.numpy as jnp

            simpls_fit_jax, _ = _get_cached_jax_simpls()

            X_jax = jnp.asarray(X_centered)
            Y_jax = jnp.asarray(Y_centered)

            T, U, W, P, Q, R, B = simpls_fit_jax(
                X_jax, Y_jax, self.n_components_
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
            T, U, W, P, Q, R, B = _simpls_fit_numpy(
                X_centered, Y_centered, self.n_components_
            )

            self.x_scores_ = T
            self.y_scores_ = U
            self.x_weights_ = W
            self.x_loadings_ = P
            self.y_loadings_ = Q
            self._R = R
            self._B = B

        # Store final regression coefficients
        # coef_ = B[n_components_-1] in standardized space, need to transform
        # Y_pred_std = X_std @ B
        # Y_pred = Y_pred_std * y_std + y_mean
        # Y_pred = ((X - x_mean) / x_std) @ B * y_std + y_mean
        # Y_pred = X @ (B * y_std / x_std) - x_mean @ (B * y_std / x_std) + y_mean
        # coef_ in original space: B * y_std / x_std (for centered prediction)
        B_final = self._B[self.n_components_ - 1]  # (n_features, n_targets)
        self.coef_ = B_final * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: Union[int, None] = None,
    ) -> NDArray[np.floating]:
        """Predict using the SIMPLS model.

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

            _, simpls_predict_jax = _get_cached_jax_simpls()

            X_jax = jnp.asarray(X)
            x_mean_jax = jnp.asarray(self.x_mean_)
            x_std_jax = jnp.asarray(self.x_std_)
            y_mean_jax = jnp.asarray(self.y_mean_)
            y_std_jax = jnp.asarray(self.y_std_)
            B_jax = jnp.asarray(self._B)

            y_pred = simpls_predict_jax(
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
        check_is_fitted(self, ['x_mean_', 'x_std_', 'x_weights_'])

        X = np.asarray(X, dtype=np.float64)
        X_centered = (X - self.x_mean_) / self.x_std_

        # Compute scores: T = X @ W
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
            'scale': self.scale,
            'center': self.center,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "SIMPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : SIMPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SIMPLS(n_components={self.n_components}, "
            f"scale={self.scale}, center={self.center}, "
            f"backend='{self.backend}')"
        )

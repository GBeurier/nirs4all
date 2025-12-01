"""Sparse PLS (sPLS) regressor with L1 regularization for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import logging

def _check_sparse_pls_available():
    """Check if sparse-pls package is available."""
    try:
        import sparse_pls
        return True
    except ImportError:
        return False

def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax
        return True
    except ImportError:
        return False

def _get_jax_sparse_pls_functions():
    """Get JAX-accelerated Sparse PLS functions.

    Implements the same algorithm as the sparse-pls package for identical results.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    def soft_threshold(z, alpha):
        """Soft thresholding operator for L1 regularization."""
        return jnp.sign(z) * jnp.maximum(jnp.abs(z) - alpha, 0.0)

    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def sparse_pls_fit_jax(X, y, n_components, alpha, max_iter, tol):
        """Fit Sparse PLS using JAX - matches sparse-pls package algorithm.

        Uses iterative alternating optimization with soft thresholding on both
        X and Y weight vectors, matching the sparse-pls package behavior.
        """
        # Ensure float64 for numerical precision
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        n_samples, n_features = X.shape

        # Center and scale (matching sparse-pls StandardScaler behavior)
        X_mean = jnp.mean(X, axis=0, keepdims=True)
        X_std = jnp.std(X, axis=0, keepdims=True, ddof=0)  # ddof=0 for sklearn StandardScaler
        X_std = jnp.where(X_std < 1e-10, 1.0, X_std)
        X_scaled = (X - X_mean) / X_std

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        n_targets = y.shape[1]
        y_mean = jnp.mean(y, axis=0, keepdims=True)
        y_std = jnp.std(y, axis=0, keepdims=True, ddof=0)
        y_std = jnp.where(y_std < 1e-10, 1.0, y_std)
        y_scaled = (y - y_mean) / y_std

        # Initialize weight matrices
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        C = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)

        def compute_sparse_component(X_res, Y_res, key):
            """Compute one sparse PLS component using alternating optimization."""
            # Initialize c randomly (use deterministic init for reproducibility)
            c = jnp.ones((n_targets, 1), dtype=jnp.float64)
            c = c / jnp.linalg.norm(c)

            def sparse_iter_body(carry):
                c, _ = carry

                # Compute w = soft_threshold(X.T @ Y @ c, alpha)
                z_w = X_res.T @ Y_res @ c
                w = soft_threshold(z_w, alpha)
                w_norm = jnp.linalg.norm(w)
                w = lax.cond(w_norm > 1e-10, lambda: w / w_norm, lambda: w)

                # Compute t = X @ w
                t = X_res @ w

                # Compute c = soft_threshold(Y.T @ t, alpha)
                z_c = Y_res.T @ t
                c_new = soft_threshold(z_c, alpha)
                c_norm = jnp.linalg.norm(c_new)
                c_new = lax.cond(c_norm > 1e-10, lambda: c_new / c_norm, lambda: c_new)

                # Compute change for convergence check
                change = jnp.linalg.norm(c_new - c)

                return (c_new, change)

            def sparse_iter_cond(carry):
                _, change = carry
                return change >= tol

            # Run iterations with while_loop for convergence
            def run_iterations(c_init):
                # First iteration
                c, change = sparse_iter_body((c_init, jnp.float64(1.0)))

                # Continue iterations
                def iteration_step(i, carry):
                    c, change, converged = carry
                    # Only iterate if not converged
                    new_c, new_change = lax.cond(
                        converged,
                        lambda: (c, change),
                        lambda: sparse_iter_body((c, change))
                    )
                    new_converged = converged | (new_change < tol)
                    return (new_c, new_change, new_converged)

                c, change, _ = lax.fori_loop(
                    0, max_iter - 1, iteration_step,
                    (c, change, change < tol)
                )
                return c

            c_final = run_iterations(c)

            # Final w computation
            z_w = X_res.T @ Y_res @ c_final
            w = soft_threshold(z_w, alpha)
            w_norm = jnp.linalg.norm(w)
            w = lax.cond(w_norm > 1e-10, lambda: w / w_norm, lambda: w)

            return w.ravel(), c_final.ravel()

        def component_step(comp_i, carry):
            X_res, Y_res, W, C, P, Q, key = carry

            # Split key for random initialization (though we use deterministic init above)
            key, subkey = jax.random.split(key)

            # Compute component
            w, c = compute_sparse_component(X_res, Y_res, subkey)

            # Calculate scores
            t = X_res @ w
            u = Y_res @ c

            # Normalize scores
            t_norm = jnp.linalg.norm(t)
            t_safe = lax.cond(t_norm > 1e-10, lambda: t / t_norm, lambda: t)
            u_safe = lax.cond(t_norm > 1e-10, lambda: u / t_norm, lambda: u)

            # Calculate loadings
            p = X_res.T @ t_safe
            q = Y_res.T @ t_safe

            # Store
            W = W.at[:, comp_i].set(w)
            C = C.at[:, comp_i].set(c)
            P = P.at[:, comp_i].set(p)
            Q = Q.at[:, comp_i].set(q)

            # Deflate
            X_res = X_res - jnp.outer(t_safe, p)
            Y_res = Y_res - jnp.outer(t_safe, q)

            return X_res, Y_res, W, C, P, Q, key

        # Run loop over components
        key = jax.random.PRNGKey(42)
        _, _, W, C, P, Q, _ = lax.fori_loop(
            0, n_components, component_step,
            (X_scaled, y_scaled, W, C, P, Q, key)
        )

        # Compute final regression coefficients: B = W @ inv(P.T @ W) @ Q.T
        PtW = P.T @ W
        PtW_reg = PtW + 1e-5 * jnp.eye(n_components)
        PtW_inv = jnp.linalg.pinv(PtW_reg)
        B_final = W @ PtW_inv @ Q.T

        return B_final, W, P, Q, X_mean, X_std, y_mean, y_std

    @jax.jit
    def sparse_pls_predict_jax(X, B, X_mean, X_std, y_mean, y_std):
        """Predict using Sparse PLS coefficients."""
        X = jnp.asarray(X, dtype=jnp.float64)
        X_scaled = (X - X_mean) / X_std
        y_pred_scaled = X_scaled @ B
        return y_pred_scaled * y_std + y_mean

    return sparse_pls_fit_jax, sparse_pls_predict_jax

# Cache for JAX SparsePLS functions
_JAX_SPARSE_PLS_FUNCS = None

def _get_cached_jax_sparse_pls():
    """Get cached JAX SparsePLS functions to avoid recompilation."""
    global _JAX_SPARSE_PLS_FUNCS
    if _JAX_SPARSE_PLS_FUNCS is None:
        _JAX_SPARSE_PLS_FUNCS = _get_jax_sparse_pls_functions()
    return _JAX_SPARSE_PLS_FUNCS

class SparsePLS(BaseEstimator, RegressorMixin):
    """Sparse PLS (sPLS) regressor with L1 regularization.

    Sparse PLS performs joint prediction and variable selection by applying
    L1 (Lasso) regularization to the PLS loadings. This produces sparse
    loadings where many wavelengths/features have zero weights, effectively
    selecting the most relevant variables.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent variables to extract.
    alpha : float, default=1.0
        Regularization strength. Higher values produce more sparsity.
    max_iter : int, default=500
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    scale : bool, default=True
        Whether to scale X and y before fitting.
    backend : str, default='numpy'
        Backend to use for computation. Options are:
        - 'numpy': Use NumPy backend via sparse-pls package (CPU only).
        - 'jax': Use JAX backend (supports GPU/TPU acceleration).
        JAX backend requires JAX: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.
    coef_ : ndarray of shape (n_features,) or (n_features, n_targets)
        Regression coefficients (sparse).

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import SparsePLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = SparsePLS(n_components=5, alpha=0.5)
    >>> model.fit(X, y)
    SparsePLS(n_components=5, alpha=0.5)
    >>> predictions = model.predict(X)
    >>> # Check sparsity
    >>> n_selected = np.sum(model.coef_ != 0)
    >>> # JAX backend for GPU acceleration
    >>> model_jax = SparsePLS(n_components=5, alpha=0.5, backend='jax')

    Notes
    -----
    NumPy backend requires the `sparse-pls` package: ``pip install sparse-pls``

    JAX backend uses a custom implementation and does not require sparse-pls.
    For JAX with GPU support: ``pip install jax[cuda12]``

    The alpha parameter controls the trade-off between prediction accuracy
    and sparsity. Use cross-validation to find the optimal value.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard non-sparse PLS.

    References
    ----------
    .. [1] Lê Cao, K.-A., et al. (2008). Sparse PLS discriminant analysis:
           biologically relevant feature selection and graphical displays
           for multiclass problems. BMC Bioinformatics, 9(1), 1-18.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 5,
        alpha: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        scale: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize SparsePLS regressor.

        Parameters
        ----------
        n_components : int, default=5
            Number of latent variables to extract.
        alpha : float, default=1.0
            Regularization strength.
        max_iter : int, default=500
            Maximum number of iterations.
        tol : float, default=1e-6
            Convergence tolerance.
        scale : bool, default=True
            Whether to scale X and y before fitting.
        backend : str, default='numpy'
            Backend to use ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.backend = backend

    def fit(self, X, y):
        """Fit the Sparse PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : SparsePLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If sparse-pls package is not installed (NumPy backend),
            or JAX is not available (JAX backend).
        ValueError
            If backend is not 'numpy' or 'jax'.
        """
        # Validate backend
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Limit components
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components_ = min(self.n_components, max_components)

        if self.backend == 'jax':
            if not _check_jax_available():
                raise ImportError(
                    "JAX is required for SparsePLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

            import jax.numpy as jnp

            # Get JAX functions
            sparse_pls_fit_jax, _ = _get_cached_jax_sparse_pls()

            # Fit using JAX
            X_jax = jnp.asarray(X)
            y_jax = jnp.asarray(y)

            result = sparse_pls_fit_jax(
                X_jax, y_jax,
                self.n_components_,
                self.alpha,
                self.max_iter,
                self.tol
            )
            (self._B, self._W, self._P, self._Q,
             self._X_mean, self._X_std,
             self._y_mean, self._y_std) = result

            # Store coefficients
            self.coef_ = np.asarray(self._B)

            self._model = None  # Not using sparse-pls package
        else:
            # NumPy backend using sparse-pls
            if not _check_sparse_pls_available():
                raise ImportError(
                    "sparse-pls package is required for SparsePLS with backend='numpy'. "
                    "Install it with: pip install sparse-pls"
                )

            # Suppress verbose INFO logging from sparse_pls
            logging.getLogger('sparse_pls.model').setLevel(logging.WARNING)

            from sparse_pls import SparsePLS as SPLSModel

            # Create and fit sparse-pls model
            self._model = SPLSModel(
                n_components=self.n_components_,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                scale=self.scale,
            )
            self._model.fit(X, y)

            # Store coefficients for compatibility
            if hasattr(self._model, 'coef_'):
                self.coef_ = self._model.coef_
            else:
                self.coef_ = None

        return self

    def predict(self, X):
        """Predict using the Sparse PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        X = np.asarray(X)

        if self.backend == 'jax':
            import jax.numpy as jnp

            _, sparse_pls_predict_jax = _get_cached_jax_sparse_pls()

            X_jax = jnp.asarray(X)
            y_pred = sparse_pls_predict_jax(
                X_jax, self._B,
                self._X_mean, self._X_std,
                self._y_mean, self._y_std
            )
            y_pred = np.asarray(y_pred)
        else:
            y_pred = self._model.predict(X)

        # Flatten if single target and 2D
        if hasattr(y_pred, 'ndim') and y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X):
        """Transform X to latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components)
            Latent variables (scores).
        """
        if self.backend == 'jax':
            raise NotImplementedError(
                "transform() is not implemented for JAX backend. "
                "Use backend='numpy' for transform functionality."
            )

        X = np.asarray(X)
        return self._model.transform(X)

    def get_selected_features(self):
        """Get indices of selected (non-zero) features.

        Returns
        -------
        indices : ndarray
            Indices of features with non-zero coefficients.
        """
        if self.backend == 'jax':
            if self.coef_ is not None:
                # Handle multi-target case
                if self.coef_.ndim > 1:
                    return np.where(np.any(self.coef_ != 0, axis=-1))[0]
                else:
                    return np.where(self.coef_ != 0)[0]
            else:
                return np.arange(self.n_features_in_)
        else:
            if hasattr(self._model, 'get_selected_feature_names'):
                return self._model.get_selected_feature_names()
            elif self.coef_ is not None:
                return np.where(np.any(self.coef_ != 0, axis=-1))[0]
            else:
                return np.arange(self.n_features_in_)

    def get_params(self, deep=True):
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
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'scale': self.scale,
            'backend': self.backend,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : SparsePLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        """Return string representation."""
        params = [
            f"n_components={self.n_components}",
            f"alpha={self.alpha}",
        ]
        if self.max_iter != 500:
            params.append(f"max_iter={self.max_iter}")
        if not self.scale:
            params.append("scale=False")
        if self.backend != 'numpy':
            params.append(f"backend='{self.backend}'")
        return f"SparsePLS({', '.join(params)})"

"""PLS (Partial Least Squares) model operators for nirs4all.

This module provides PLS-based model operators that are sklearn-compatible
for use in nirs4all pipelines. Supports both NumPy and JAX backends.

Classes
-------
PLSDA
    PLS Discriminant Analysis for classification tasks.
IKPLS
    Improved Kernel PLS with NumPy/JAX backend (fast implementation).
OPLS
    Orthogonal PLS for regression (removes Y-orthogonal variation).
OPLSDA
    Orthogonal PLS Discriminant Analysis for classification.
MBPLS
    Multiblock PLS for fusing multiple X blocks.
DiPLS
    Dynamic PLS for time-lagged process data.
SparsePLS
    Sparse PLS with L1 regularization for variable selection.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def _check_ikpls_available():
    """Check if ikpls package is available."""
    try:
        import ikpls
        return True
    except ImportError:
        return False


def _check_pyopls_available():
    """Check if pyopls package is available."""
    try:
        import pyopls
        return True
    except ImportError:
        return False


def _check_mbpls_available():
    """Check if mbpls package is available."""
    try:
        import mbpls
        return True
    except ImportError:
        return False


def _check_trendfitter_available():
    """Check if trendfitter package is available."""
    try:
        from trendfitter.models import DiPLS
        return True
    except ImportError:
        return False


def _check_sparse_pls_available():
    """Check if sparse-pls package is available."""
    try:
        from sparse_pls import SparsePLS
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


# =============================================================================
# JAX Backend Implementations
# =============================================================================

def _get_jax_opls_functions():
    """Get JAX-accelerated OPLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def opls_fit_jax(X, y, n_components):
        """Fit OPLS model using JAX.

        OPLS algorithm based on Trygg & Wold (2002).
        Removes Y-orthogonal variation from X.

        Returns
        -------
        W_ortho : Orthogonal weight matrix
        P_ortho : Orthogonal loading matrix
        X_mean : Mean of X
        X_std : Std of X
        y_mean : Mean of y
        y_std : Std of y
        """
        # Ensure float64 for numerical precision
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        n_samples, n_features = X.shape

        # Center and scale
        X_mean = jnp.mean(X, axis=0, keepdims=True)
        X_std = jnp.std(X, axis=0, keepdims=True, ddof=1)
        X_std = jnp.where(X_std < 1e-10, 1.0, X_std)
        X_centered = (X - X_mean) / X_std

        y_mean = jnp.mean(y)
        y_std = jnp.std(y, ddof=1)
        y_std = jnp.where(y_std < 1e-10, 1.0, y_std)
        y_centered = (y - y_mean) / y_std

        # Initialize storage for orthogonal components
        W_ortho = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P_ortho = jnp.zeros((n_features, n_components), dtype=jnp.float64)

        def component_step(i, carry):
            X_res, W_ortho, P_ortho = carry

            # Calculate weight vector w (predictive direction)
            w = X_res.T @ y_centered
            w = w / (jnp.linalg.norm(w) + 1e-10)

            # Calculate scores
            t = X_res @ w

            # Calculate orthogonal weight
            p = X_res.T @ t / (t.T @ t + 1e-10)

            # Orthogonal weight (component orthogonal to w)
            w_ortho = p - (w.T @ p) * w
            w_ortho = w_ortho / (jnp.linalg.norm(w_ortho) + 1e-10)

            # Orthogonal scores and loadings
            t_ortho = X_res @ w_ortho
            p_ortho = X_res.T @ t_ortho / (t_ortho.T @ t_ortho + 1e-10)

            # Deflate X
            X_res = X_res - jnp.outer(t_ortho, p_ortho)

            # Store
            W_ortho = W_ortho.at[:, i].set(w_ortho)
            P_ortho = P_ortho.at[:, i].set(p_ortho)

            return X_res, W_ortho, P_ortho

        X_filtered, W_ortho, P_ortho = lax.fori_loop(
            0, n_components, component_step, (X_centered, W_ortho, P_ortho)
        )

        return W_ortho, P_ortho, X_mean, X_std, y_mean, y_std

    @jax.jit
    def opls_transform_jax(X, W_ortho, P_ortho, X_mean, X_std):
        """Transform X by removing orthogonal variation."""
        # Ensure float64 for consistent dtypes in fori_loop
        X = jnp.asarray(X, dtype=jnp.float64)
        X_centered = (X - X_mean) / X_std
        n_components = W_ortho.shape[1]

        def remove_component(i, X_res):
            t_ortho = X_res @ W_ortho[:, i]
            X_res = X_res - jnp.outer(t_ortho, P_ortho[:, i])
            return X_res

        X_filtered = lax.fori_loop(0, n_components, remove_component, X_centered)
        return X_filtered * X_std + X_mean

    return opls_fit_jax, opls_transform_jax


def _get_jax_mbpls_functions():
    """Get JAX-accelerated MBPLS functions (single block)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def mbpls_fit_jax(X, y, n_components):
        """Fit single-block MBPLS using JAX (equivalent to NIPALS PLS).

        Returns
        -------
        B : Regression coefficients for each component
        W : Weight matrix
        P : Loading matrix
        Q : Y loading matrix
        T : Score matrix
        X_mean, X_std, y_mean, y_std : Preprocessing parameters
        """
        # Ensure float64 for numerical precision
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        n_samples, n_features = X.shape

        # Center and scale
        X_mean = jnp.mean(X, axis=0, keepdims=True)
        X_std = jnp.std(X, axis=0, keepdims=True, ddof=1)
        X_std = jnp.where(X_std < 1e-10, 1.0, X_std)
        X_centered = (X - X_mean) / X_std

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        n_targets = y.shape[1]
        y_mean = jnp.mean(y, axis=0, keepdims=True)
        y_std = jnp.std(y, axis=0, keepdims=True, ddof=1)
        y_std = jnp.where(y_std < 1e-10, 1.0, y_std)
        y_centered = (y - y_mean) / y_std

        # Initialize matrices with explicit dtype
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        T = jnp.zeros((n_samples, n_components), dtype=jnp.float64)

        def component_step(i, carry):
            X_res, y_res, W, P, Q, T = carry

            # Weight vector
            w = X_res.T @ y_res
            if n_targets == 1:
                w = w.ravel()
            else:
                # For multi-target, use first singular vector
                w = w[:, 0]
            w = w / (jnp.linalg.norm(w) + 1e-10)

            # Scores
            t = X_res @ w
            t_norm = t.T @ t + 1e-10

            # Loadings
            p = X_res.T @ t / t_norm
            q = y_res.T @ t / t_norm

            # Store
            W = W.at[:, i].set(w)
            P = P.at[:, i].set(p)
            Q = Q.at[:, i].set(q.ravel())
            T = T.at[:, i].set(t)

            # Deflate
            X_res = X_res - jnp.outer(t, p)
            y_res = y_res - jnp.outer(t, q)

            return X_res, y_res, W, P, Q, T

        _, _, W, P, Q, T = lax.fori_loop(
            0, n_components, component_step,
            (X_centered, y_centered, W, P, Q, T)
        )

        # Compute final regression coefficients: B = W @ inv(P.T @ W) @ Q.T
        PtW = P.T @ W
        PtW_reg = PtW + 1e-10 * jnp.eye(n_components)
        PtW_inv = jnp.linalg.pinv(PtW_reg)
        B_final = W @ PtW_inv @ Q.T

        return B_final, W, P, Q, T, X_mean, X_std, y_mean, y_std

    @jax.jit
    def mbpls_predict_jax(X, B, X_mean, X_std, y_mean, y_std):
        """Predict using MBPLS coefficients."""
        X = jnp.asarray(X, dtype=jnp.float64)
        X_centered = (X - X_mean) / X_std
        y_pred_centered = X_centered @ B
        return y_pred_centered * y_std + y_mean

    return mbpls_fit_jax, mbpls_predict_jax



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

            # Split key for this component
            key, subkey = jax.random.split(key)

            w, c = compute_sparse_component(X_res, Y_res, subkey)

            # Compute scores
            t = X_res @ w
            u = Y_res @ c

            # Normalize scores
            t_norm = jnp.linalg.norm(t)
            t_safe = lax.cond(t_norm > 1e-10, lambda: t / t_norm, lambda: t)
            u_safe = lax.cond(t_norm > 1e-10, lambda: u / t_norm, lambda: u)

            # Compute loadings
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

        # Initialize random key for reproducibility
        key = jax.random.PRNGKey(42)

        _, _, W, C, P, Q, _ = lax.fori_loop(
            0, n_components, component_step,
            (X_scaled, y_scaled, W, C, P, Q, key)
        )

        # Compute regression coefficients: coef = W @ pinv(P.T @ W) @ Q.T
        PtW = P.T @ W
        PtW_reg = PtW + 1e-5 * jnp.eye(n_components)  # Match sparse-pls reg_param
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


# Cache JAX functions
_JAX_OPLS_FUNCS = None
_JAX_MBPLS_FUNCS = None
_JAX_SPARSE_PLS_FUNCS = None


def _get_cached_jax_opls():
    """Get cached JAX OPLS functions."""
    global _JAX_OPLS_FUNCS
    if _JAX_OPLS_FUNCS is None:
        _JAX_OPLS_FUNCS = _get_jax_opls_functions()
    return _JAX_OPLS_FUNCS


def _get_cached_jax_mbpls():
    """Get cached JAX MBPLS functions."""
    global _JAX_MBPLS_FUNCS
    if _JAX_MBPLS_FUNCS is None:
        _JAX_MBPLS_FUNCS = _get_jax_mbpls_functions()
    return _JAX_MBPLS_FUNCS


def _get_cached_jax_sparse_pls():
    """Get cached JAX Sparse PLS functions."""
    global _JAX_SPARSE_PLS_FUNCS
    if _JAX_SPARSE_PLS_FUNCS is None:
        _JAX_SPARSE_PLS_FUNCS = _get_jax_sparse_pls_functions()
    return _JAX_SPARSE_PLS_FUNCS


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS Discriminant Analysis (PLS-DA) classifier.

    PLS-DA uses PLSRegression with encoded targets for classification.
    For binary classification, targets are label-encoded to 0/1.
    For multiclass, targets are one-hot encoded.
    Predictions are made by finding the class with highest response.

    Parameters
    ----------
    n_components : int, default=5
        Number of PLS components to extract.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    pls_ : PLSRegression
        Fitted PLS model.
    encoder_ : LabelEncoder or OneHotEncoder
        Encoder used for target transformation.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import PLSDA
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=50, n_classes=3,
    ...                            n_informative=10, random_state=42)
    >>> model = PLSDA(n_components=5)
    >>> model.fit(X, y)
    PLSDA(n_components=5)
    >>> predictions = model.predict(X)

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Underlying PLS model.

    References
    ----------
    .. [1] Barker, M., & Rayens, W. (2003). Partial least squares for discrimination.
           Journal of Chemometrics, 17(3), 166-173.
    """

    def __init__(self, n_components: int = 5):
        """Initialize PLSDA classifier.

        Parameters
        ----------
        n_components : int, default=5
            Number of PLS components to extract.
        """
        self.n_components = n_components

    def fit(self, X, y):
        """Fit the PLS-DA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : PLSDA
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # Binary classification: use single column (0/1)
            self.encoder_ = LabelEncoder()
            y_encoded = self.encoder_.fit_transform(y).reshape(-1, 1).astype(float)
        else:
            # Multiclass: one-hot encode
            self.encoder_ = OneHotEncoder(sparse_output=False)
            y_encoded = self.encoder_.fit_transform(y.reshape(-1, 1))

        # Fit PLS with number of components limited by features
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=n_comp)
        self.pls_.fit(X, y_encoded)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        y_pred_raw = self.pls_.predict(X)

        if len(self.classes_) == 2:
            # Binary: threshold at 0.5
            y_pred_idx = (y_pred_raw.ravel() > 0.5).astype(int)
            return self.classes_[y_pred_idx]
        else:
            # Multiclass: argmax
            y_pred_idx = np.argmax(y_pred_raw, axis=1)
            return self.classes_[y_pred_idx]

    def predict_proba(self, X):
        """Return pseudo-probabilities (PLS responses).

        Note: These are not true probabilities but raw PLS predictions.
        For binary classification, returns a 2-column array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Pseudo-probability estimates.
        """
        X = np.asarray(X)
        y_pred_raw = self.pls_.predict(X)

        if len(self.classes_) == 2:
            # Convert single column to 2-column format
            proba_pos = y_pred_raw.ravel()
            proba_neg = 1 - proba_pos
            return np.column_stack([proba_neg, proba_pos])

        return y_pred_raw

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
        return {'n_components': self.n_components}

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : PLSDA
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class IKPLS(BaseEstimator, RegressorMixin):
    """Improved Kernel PLS (IKPLS) regressor.

    A sklearn-compatible wrapper for the ikpls package, which provides
    fast PLS implementations using NumPy or JAX (for GPU/TPU acceleration).
    IKPLS is significantly faster than sklearn's PLSRegression, especially
    for cross-validation.

    Parameters
    ----------
    n_components : int, default=10
        Number of PLS components to extract.
    algorithm : int, default=1
        IKPLS algorithm variant (1 or 2). Algorithm 1 is generally faster.
    center : bool, default=True
        Whether to center X and Y before fitting.
    scale : bool, default=True
        Whether to scale X and Y before fitting.
    backend : str, default='numpy'
        Backend to use for computation. Options are:
        - 'numpy': Use NumPy backend (CPU only).
        - 'jax': Use JAX backend (supports GPU/TPU acceleration).
        JAX backend requires JAX to be installed: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used (may be less than n_components
        if limited by data dimensions).
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import IKPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> # NumPy backend (default)
    >>> model = IKPLS(n_components=10)
    >>> model.fit(X, y)
    IKPLS(n_components=10)
    >>> predictions = model.predict(X)
    >>> # JAX backend for GPU acceleration
    >>> model_jax = IKPLS(n_components=10, backend='jax')

    Notes
    -----
    Requires the `ikpls` package: ``pip install ikpls``

    For JAX backend with GPU support, install JAX with CUDA:
    ``pip install jax[cuda12]``

    The JAX backend is end-to-end differentiable, allowing gradient
    propagation when using PLS as a layer in a deep learning model.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard sklearn PLS.

    References
    ----------
    .. [1] Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms.
           Journal of Chemometrics, 11(1), 73-85.
    """

    def __init__(
        self,
        n_components: int = 10,
        algorithm: int = 1,
        center: bool = True,
        scale: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize IKPLS regressor.

        Parameters
        ----------
        n_components : int, default=10
            Number of PLS components to extract.
        algorithm : int, default=1
            IKPLS algorithm variant (1 or 2).
        center : bool, default=True
            Whether to center X and Y before fitting.
        scale : bool, default=True
            Whether to scale X and Y before fitting.
        backend : str, default='numpy'
            Backend to use ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.algorithm = algorithm
        self.center = center
        self.scale = scale
        self.backend = backend

    def fit(self, X, y):
        """Fit the IKPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : IKPLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If ikpls package is not installed, or JAX is not available
            when using 'jax' backend.
        ValueError
            If backend is not 'numpy' or 'jax'.
        """
        if not _check_ikpls_available():
            raise ImportError(
                "ikpls package is required for IKPLS. "
                "Install it with: pip install ikpls"
            )

        # Validate backend
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        X = np.asarray(X)
        y = np.asarray(y)

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Limit components by data dimensions
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components_ = min(self.n_components, max_components)

        # Import and create the appropriate backend
        if self.backend == 'jax':
            if not _check_jax_available():
                raise ImportError(
                    "JAX is required for IKPLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )
            # Enable float64 for JAX (important for numerical precision)
            import jax
            jax.config.update("jax_enable_x64", True)

            # Import JAX backend based on algorithm
            if self.algorithm == 1:
                from ikpls.jax_ikpls_alg_1 import PLS as JaxPLS
            else:
                from ikpls.jax_ikpls_alg_2 import PLS as JaxPLS

            # Convert to JAX arrays
            import jax.numpy as jnp
            X_jax = jnp.asarray(X)
            y_jax = jnp.asarray(y)

            # Create and fit JAX model
            self._model = JaxPLS()
            self._model.fit(X_jax, y_jax, A=self.n_components_)

            # Store coefficient for compatibility - convert back to numpy
            self.coef_ = np.asarray(self._model.B[-1])
        else:
            # NumPy backend
            from ikpls.numpy_ikpls import PLS as NumpyPLS

            # Create and fit ikpls model
            self._model = NumpyPLS(
                algorithm=self.algorithm,
                center_X=self.center,
                center_Y=self.center,
                scale_X=self.scale,
                scale_Y=self.scale,
            )
            self._model.fit(X, y, A=self.n_components_)

            # Store coefficient for compatibility (last component's coefficients)
            # B shape is (n_components, n_features, n_targets), take last component
            self.coef_ = self._model.B[-1]  # shape: (n_features, n_targets)

        return self

    def predict(self, X, n_components=None):
        """Predict using the IKPLS model.

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
            Predicted values (always returns NumPy arrays).
        """
        if n_components is None:
            n_components = self.n_components_

        if self.backend == 'jax':
            import jax.numpy as jnp
            X_jax = jnp.asarray(X)
            y_pred = self._model.predict(X_jax, n_components=n_components)
            # Convert back to numpy
            y_pred = np.asarray(y_pred)
        else:
            X = np.asarray(X)
            y_pred = self._model.predict(X, n_components=n_components)

        # Flatten if single target
        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

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
            'algorithm': self.algorithm,
            'center': self.center,
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
        self : IKPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OPLS(BaseEstimator, RegressorMixin):
    """Orthogonal PLS (OPLS) regressor.

    OPLS removes Y-orthogonal variation from X before applying PLS regression.
    This improves model interpretability by separating predictive from
    non-predictive (orthogonal) variation.

    The model first fits an OPLS filter to remove orthogonal components,
    then applies standard PLS regression on the filtered data.

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove.
    pls_components : int, default=1
        Number of PLS components for the predictive model.
        Typically 1 for OPLS (predictive variation is in one direction).
    scale : bool, default=True
        Whether to scale X before fitting.
    backend : str, default='numpy'
        Backend to use for computation. Options are:
        - 'numpy': Use NumPy backend via pyopls (CPU only).
        - 'jax': Use JAX backend (supports GPU/TPU acceleration).
        JAX backend requires JAX to be installed: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    opls_ : pyopls.OPLS or None
        Fitted OPLS transformer (NumPy backend only).
    pls_ : PLSRegression
        Fitted PLS model on filtered data.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import OPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = OPLS(n_components=2, pls_components=1)
    >>> model.fit(X, y)
    OPLS(n_components=2, pls_components=1)
    >>> predictions = model.predict(X)
    >>> # JAX backend for GPU acceleration
    >>> model_jax = OPLS(n_components=2, backend='jax')

    Notes
    -----
    NumPy backend requires the `pyopls` package: ``pip install pyopls``

    JAX backend uses a custom implementation and does not require pyopls.
    For JAX with GPU support: ``pip install jax[cuda12]``

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard PLS regression.
    OPLSDA : OPLS for classification tasks.

    References
    ----------
    .. [1] Trygg, J., & Wold, S. (2002). Orthogonal projections to latent
           structures (O-PLS). Journal of Chemometrics, 16(3), 119-128.
    """

    def __init__(
        self,
        n_components: int = 1,
        pls_components: int = 1,
        scale: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize OPLS regressor.

        Parameters
        ----------
        n_components : int, default=1
            Number of orthogonal components to remove.
        pls_components : int, default=1
            Number of PLS components for the predictive model.
        scale : bool, default=True
            Whether to scale X before fitting.
        backend : str, default='numpy'
            Backend to use ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.pls_components = pls_components
        self.scale = scale
        self.backend = backend

    def fit(self, X, y):
        """Fit the OPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : OPLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If pyopls package is not installed (NumPy backend),
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

        if y.ndim == 1:
            y_flat = y
            y = y.reshape(-1, 1)
        else:
            y_flat = y.ravel()

        self.n_features_in_ = X.shape[1]

        # Limit components
        max_ortho = min(self.n_components, X.shape[1] - 1, X.shape[0] - 2)
        n_ortho = max(1, max_ortho)
        self.n_components_ = n_ortho

        if self.backend == 'jax':
            if not _check_jax_available():
                raise ImportError(
                    "JAX is required for OPLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

            import jax.numpy as jnp

            # Get JAX functions
            opls_fit_jax, opls_transform_jax = _get_cached_jax_opls()

            # Fit OPLS filter using JAX
            X_jax = jnp.asarray(X)
            y_jax = jnp.asarray(y_flat)

            result = opls_fit_jax(X_jax, y_jax, n_ortho)
            (self._W_ortho, self._P_ortho,
             self._X_mean, self._X_std,
             self._y_mean, self._y_std) = result

            # Transform training data
            X_filtered = opls_transform_jax(
                X_jax, self._W_ortho, self._P_ortho,
                self._X_mean, self._X_std
            )
            X_filtered = np.asarray(X_filtered)

            self.opls_ = None  # Not using pyopls in JAX mode
        else:
            # NumPy backend using pyopls
            if not _check_pyopls_available():
                raise ImportError(
                    "pyopls package is required for OPLS with backend='numpy'. "
                    "Install it with: pip install pyopls"
                )

            from pyopls import OPLS as PyOPLS

            # Fit OPLS transformer
            self.opls_ = PyOPLS(n_components=n_ortho, scale=self.scale)
            X_filtered = self.opls_.fit_transform(X, y_flat)

        # Fit PLS on filtered data (both backends)
        max_pls = min(self.pls_components, X_filtered.shape[1], X_filtered.shape[0] - 1)
        n_pls = max(1, max_pls)

        self.pls_ = PLSRegression(n_components=n_pls)
        self.pls_.fit(X_filtered, y)

        return self

    def predict(self, X):
        """Predict using the OPLS model.

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

        # Transform X to remove orthogonal variation
        X_filtered = self.transform(X)

        # Predict with PLS
        y_pred = self.pls_.predict(X_filtered)

        # Flatten if single target
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X):
        """Transform X by removing orthogonal variation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            Transformed samples with orthogonal variation removed.
        """
        X = np.asarray(X)

        if self.backend == 'jax':
            import jax.numpy as jnp

            _, opls_transform_jax = _get_cached_jax_opls()

            X_jax = jnp.asarray(X)
            X_filtered = opls_transform_jax(
                X_jax, self._W_ortho, self._P_ortho,
                self._X_mean, self._X_std
            )
            return np.asarray(X_filtered)
        else:
            return self.opls_.transform(X)

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
            'pls_components': self.pls_components,
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
        self : OPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        """Return string representation."""
        params = [
            f"n_components={self.n_components}",
            f"pls_components={self.pls_components}",
        ]
        if not self.scale:
            params.append("scale=False")
        if self.backend != 'numpy':
            params.append(f"backend='{self.backend}'")
        return f"OPLS({', '.join(params)})"


class OPLSDA(BaseEstimator, ClassifierMixin):
    """Orthogonal PLS Discriminant Analysis (OPLS-DA) classifier.

    OPLS-DA combines OPLS filtering with PLS-DA classification.
    It removes Y-orthogonal variation from X before applying PLS-DA,
    improving class separation and model interpretability.

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove.
    pls_components : int, default=5
        Number of PLS components for the discriminant model.
    scale : bool, default=True
        Whether to scale X before fitting.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    opls_ : pyopls.OPLS
        Fitted OPLS transformer.
    plsda_ : PLSDA
        Fitted PLS-DA model on filtered data.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import OPLSDA
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=50, n_classes=2,
    ...                            n_informative=10, random_state=42)
    >>> model = OPLSDA(n_components=1, pls_components=5)
    >>> model.fit(X, y)
    OPLSDA(n_components=1, pls_components=5)
    >>> predictions = model.predict(X)

    Notes
    -----
    Requires the `pyopls` package: ``pip install pyopls``

    See Also
    --------
    PLSDA : Standard PLS-DA without orthogonal filtering.
    OPLS : OPLS for regression tasks.

    References
    ----------
    .. [1] Bylesjö, M., et al. (2006). OPLS discriminant analysis: combining
           the strengths of PLS-DA and SIMCA classification. Journal of
           Chemometrics, 20(8-10), 341-351.
    """

    def __init__(
        self,
        n_components: int = 1,
        pls_components: int = 5,
        scale: bool = True,
    ):
        """Initialize OPLSDA classifier.

        Parameters
        ----------
        n_components : int, default=1
            Number of orthogonal components to remove.
        pls_components : int, default=5
            Number of PLS components for the discriminant model.
        scale : bool, default=True
            Whether to scale X before fitting.
        """
        self.n_components = n_components
        self.pls_components = pls_components
        self.scale = scale

    def fit(self, X, y):
        """Fit the OPLS-DA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : OPLSDA
            Fitted estimator.

        Raises
        ------
        ImportError
            If pyopls package is not installed.
        """
        if not _check_pyopls_available():
            raise ImportError(
                "pyopls package is required for OPLSDA. "
                "Install it with: pip install pyopls"
            )

        from pyopls import OPLS as PyOPLS

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Encode y for OPLS fitting (use numeric encoding)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        # Limit components
        max_ortho = min(self.n_components, X.shape[1] - 1, X.shape[0] - 2)
        n_ortho = max(1, max_ortho)

        # Fit OPLS transformer
        self.opls_ = PyOPLS(n_components=n_ortho, scale=self.scale)
        X_filtered = self.opls_.fit_transform(X, y_encoded)

        # Fit PLS-DA on filtered data
        self.plsda_ = PLSDA(n_components=self.pls_components)
        self.plsda_.fit(X_filtered, y)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)

        # Transform X to remove orthogonal variation
        X_filtered = self.opls_.transform(X)

        # Predict with PLS-DA
        return self.plsda_.predict(X_filtered)

    def predict_proba(self, X):
        """Return pseudo-probabilities (PLS responses).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Pseudo-probability estimates.
        """
        X = np.asarray(X)

        # Transform X to remove orthogonal variation
        X_filtered = self.opls_.transform(X)

        # Get probabilities from PLS-DA
        return self.plsda_.predict_proba(X_filtered)

    def transform(self, X):
        """Transform X by removing orthogonal variation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            Transformed samples with orthogonal variation removed.
        """
        X = np.asarray(X)
        return self.opls_.transform(X)

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
            'pls_components': self.pls_components,
            'scale': self.scale,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : OPLSDA
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class MBPLS(BaseEstimator, RegressorMixin):
    """Multiblock PLS (MB-PLS) regressor.

    MB-PLS fuses multiple X blocks (e.g., different preprocessing variants,
    multiple sensors) into a single predictive model. Each block contributes
    to the latent variables according to its relevance to Y.

    This wrapper adapts the mbpls package for single-block usage in nirs4all
    pipelines. For true multiblock usage, access the underlying model.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent variables to extract.
    method : str, default='NIPALS'
        Decomposition method. Options: 'NIPALS', 'SVD', 'SIMPLS'.
        Note: Only used with NumPy backend.
    standardize : bool, default=True
        Whether to standardize blocks before fitting.
    max_tol : float, default=1e-14
        Convergence tolerance for NIPALS.
    backend : str, default='numpy'
        Backend to use for computation. Options are:
        - 'numpy': Use NumPy backend via mbpls package (CPU only).
        - 'jax': Use JAX backend (supports GPU/TPU acceleration).
          Note: JAX backend only supports single-block mode.
        JAX backend requires JAX: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import MBPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = MBPLS(n_components=5)
    >>> model.fit(X, y)
    MBPLS(n_components=5)
    >>> predictions = model.predict(X)
    >>> # JAX backend for GPU acceleration
    >>> model_jax = MBPLS(n_components=5, backend='jax')

    Notes
    -----
    NumPy backend requires the `mbpls` package: ``pip install mbpls``

    JAX backend uses a custom implementation and does not require mbpls.
    For JAX with GPU support: ``pip install jax[cuda12]``

    For true multiblock usage with multiple X blocks, use the underlying
    mbpls.mbpls.MBPLS class directly with a list of X matrices.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard single-block PLS.

    References
    ----------
    .. [1] Westerhuis, J. A., et al. (1998). Analysis of multiblock and
           hierarchical PCA and PLS models. Journal of Chemometrics, 12(5),
           301-321.
    """

    def __init__(
        self,
        n_components: int = 5,
        method: str = 'NIPALS',
        standardize: bool = True,
        max_tol: float = 1e-14,
        backend: str = 'numpy',
    ):
        """Initialize MBPLS regressor.

        Parameters
        ----------
        n_components : int, default=5
            Number of latent variables to extract.
        method : str, default='NIPALS'
            Decomposition method (NumPy backend only).
        standardize : bool, default=True
            Whether to standardize blocks before fitting.
        max_tol : float, default=1e-14
            Convergence tolerance for NIPALS.
        backend : str, default='numpy'
            Backend to use ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.method = method
        self.standardize = standardize
        self.max_tol = max_tol
        self.backend = backend

    def fit(self, X, y):
        """Fit the MB-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            Training data. Can be a single matrix or a list of X blocks
            for true multiblock analysis (NumPy backend only).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : MBPLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If mbpls package is not installed (NumPy backend),
            or JAX is not available (JAX backend).
        ValueError
            If backend is not 'numpy' or 'jax', or if multiblock
            input is used with JAX backend.
        """
        # Validate backend
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        # Handle single array or list of blocks
        if isinstance(X, list):
            if self.backend == 'jax':
                raise ValueError(
                    "JAX backend only supports single-block mode. "
                    "Use backend='numpy' for multiblock analysis."
                )
            X_blocks = [np.asarray(x) for x in X]
            self.n_features_in_ = sum(x.shape[1] for x in X_blocks)
            self._is_multiblock = True
        else:
            X = np.asarray(X)
            X_blocks = [X]
            self.n_features_in_ = X.shape[1]
            self._is_multiblock = False

        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Limit components
        n_samples = X_blocks[0].shape[0]
        max_components = min(n_samples - 1, self.n_features_in_)
        self.n_components_ = min(self.n_components, max_components)

        if self.backend == 'jax':
            if not _check_jax_available():
                raise ImportError(
                    "JAX is required for MBPLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

            import jax.numpy as jnp

            # Get JAX functions
            mbpls_fit_jax, _ = _get_cached_jax_mbpls()

            # Fit using JAX
            X_jax = jnp.asarray(X_blocks[0])
            y_jax = jnp.asarray(y)

            result = mbpls_fit_jax(X_jax, y_jax, self.n_components_)
            (self._B, self._W, self._P, self._Q, self._T,
             self._X_mean, self._X_std,
             self._y_mean, self._y_std) = result

            # Store coefficients (final component)
            self.coef_ = np.asarray(self._B[self.n_components_ - 1])

            self._model = None  # Not using mbpls package
        else:
            # NumPy backend using mbpls
            if not _check_mbpls_available():
                raise ImportError(
                    "mbpls package is required for MBPLS with backend='numpy'. "
                    "Install it with: pip install mbpls"
                )

            from mbpls.mbpls import MBPLS as MBPLSModel

            # Fit MB-PLS model
            self._model = MBPLSModel(
                n_components=self.n_components_,
                method=self.method,
                standardize=self.standardize,
                max_tol=self.max_tol,
            )
            self._model.fit(X_blocks, y)

            # Store coefficients for compatibility
            self.coef_ = self._model.beta_

        return self

    def predict(self, X):
        """Predict using the MB-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
            Samples to predict. Must match the format used in fit().

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        # Handle single array or list of blocks
        if isinstance(X, list):
            X_blocks = [np.asarray(x) for x in X]
        else:
            X = np.asarray(X)
            X_blocks = [X]

        if self.backend == 'jax':
            import jax.numpy as jnp

            _, mbpls_predict_jax = _get_cached_jax_mbpls()

            X_jax = jnp.asarray(X_blocks[0])
            y_pred = mbpls_predict_jax(
                X_jax, self._B,
                self._X_mean, self._X_std,
                self._y_mean, self._y_std
            )
            y_pred = np.asarray(y_pred)
        else:
            y_pred = self._model.predict(X_blocks)

        # Flatten if single target
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X):
        """Transform X to latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of arrays
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

        if isinstance(X, list):
            X_blocks = [np.asarray(x) for x in X]
        else:
            X = np.asarray(X)
            X_blocks = [X]

        return self._model.transform(X_blocks)

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
            'method': self.method,
            'standardize': self.standardize,
            'max_tol': self.max_tol,
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
        self : MBPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        """Return string representation."""
        params = [f"n_components={self.n_components}"]
        if self.method != 'NIPALS':
            params.append(f"method='{self.method}'")
        if not self.standardize:
            params.append("standardize=False")
        if self.backend != 'numpy':
            params.append(f"backend='{self.backend}'")
        return f"MBPLS({', '.join(params)})"


class DiPLS(BaseEstimator, RegressorMixin):
    """Dynamic PLS (DiPLS) regressor.

    DiPLS handles time-lagged process/NIR streams via Hankelization.
    It's useful for process analytics where there are temporal dependencies
    between measurements and responses.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent variables to extract.
    lags : int, default=1
        Number of time lags to consider (s parameter in DiPLS).
    cv_splits : int, default=7
        Number of cross-validation splits for automatic component selection.
    tol : float, default=1e-8
        Convergence tolerance.
    max_iter : int, default=1000
        Maximum number of iterations.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.pls import DiPLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> model = DiPLS(n_components=5, lags=2)
    >>> model.fit(X, y)
    DiPLS(n_components=5, lags=2)
    >>> predictions = model.predict(X)

    Notes
    -----
    Requires the `trendfitter` package: ``pip install trendfitter``

    DiPLS is particularly useful for:
    - Process monitoring with temporal dependencies
    - NIR data collected over time
    - Batch process analytics

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard PLS without dynamics.

    References
    ----------
    .. [1] Dong, Y., & Qin, S. J. (2018). A novel dynamic PLS soft sensor
           based on moving-window modeling. Chemical Engineering Research
           and Design, 131, 509-519.
    """

    def __init__(
        self,
        n_components: int = 5,
        lags: int = 1,
        cv_splits: int = 7,
        tol: float = 1e-8,
        max_iter: int = 1000,
    ):
        """Initialize DiPLS regressor.

        Parameters
        ----------
        n_components : int, default=5
            Number of latent variables to extract.
        lags : int, default=1
            Number of time lags to consider.
        cv_splits : int, default=7
            Number of cross-validation splits.
        tol : float, default=1e-8
            Convergence tolerance.
        max_iter : int, default=1000
            Maximum number of iterations.
        """
        self.n_components = n_components
        self.lags = lags
        self.cv_splits = cv_splits
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit the DiPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (time-ordered measurements).
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : DiPLS
            Fitted estimator.

        Raises
        ------
        ImportError
            If trendfitter package is not installed.
        """
        if not _check_trendfitter_available():
            raise ImportError(
                "trendfitter package is required for DiPLS. "
                "Install it with: pip install trendfitter"
            )

        from trendfitter.models import DiPLS as TFDiPLS

        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Create and fit trendfitter DiPLS
        self._model = TFDiPLS(
            cv_splits_number=self.cv_splits,
            tol=self.tol,
            loop_limit=self.max_iter,
        )

        # Fit with specified components and lags
        self._model.fit(
            X, y,
            latent_variables=self.n_components,
            s=self.lags,
        )

        self.n_components_ = self.n_components

        return self

    def predict(self, X):
        """Predict using the DiPLS model.

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

        y_pred = self._model.predict(X)

        # Flatten if single target
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

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
            'lags': self.lags,
            'cv_splits': self.cv_splits,
            'tol': self.tol,
            'max_iter': self.max_iter,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DiPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


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

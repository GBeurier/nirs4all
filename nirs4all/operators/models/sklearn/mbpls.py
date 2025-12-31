"""Multiblock PLS (MB-PLS) regressor for nirs4all.

MB-PLS fuses multiple X blocks (e.g., different preprocessing variants,
multiple sensors) into a single predictive model. Each block contributes
to the latent variables according to its relevance to Y.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


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

def _mbpls_fit_numpy(X, y, n_components, standardize=True):
    """Fit single-block MBPLS using pure NumPy (NIPALS algorithm).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_components : int
        Number of components to extract.
    standardize : bool, default=True
        Whether to standardize X and y.

    Returns
    -------
    B : ndarray
        Regression coefficients.
    W : ndarray
        Weight matrix.
    P : ndarray
        X loading matrix.
    Q : ndarray
        Y loading matrix.
    T : ndarray
        Score matrix.
    X_mean, X_std, y_mean, y_std : ndarray
        Preprocessing parameters.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_samples, n_features = X.shape

    # Center and scale
    X_mean = np.mean(X, axis=0, keepdims=True)
    if standardize:
        X_std = np.std(X, axis=0, keepdims=True, ddof=1)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
    else:
        X_std = np.ones((1, n_features), dtype=np.float64)
    X_centered = (X - X_mean) / X_std

    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n_targets = y.shape[1]
    y_mean = np.mean(y, axis=0, keepdims=True)
    if standardize:
        y_std = np.std(y, axis=0, keepdims=True, ddof=1)
        y_std = np.where(y_std < 1e-10, 1.0, y_std)
    else:
        y_std = np.ones((1, n_targets), dtype=np.float64)
    y_centered = (y - y_mean) / y_std

    # Initialize matrices
    W = np.zeros((n_features, n_components), dtype=np.float64)
    P = np.zeros((n_features, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    T = np.zeros((n_samples, n_components), dtype=np.float64)

    X_res = X_centered.copy()
    y_res = y_centered.copy()

    for i in range(n_components):
        # Weight vector
        w = X_res.T @ y_res
        if n_targets == 1:
            w = w.ravel()
        else:
            # For multi-target, use first column
            w = w[:, 0]
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            w = w / w_norm

        # Scores
        t = X_res @ w
        t_norm = t.T @ t + 1e-10

        # Loadings
        p = X_res.T @ t / t_norm
        q = y_res.T @ t / t_norm

        # Store
        W[:, i] = w
        P[:, i] = p
        Q[:, i] = q.ravel()
        T[:, i] = t

        # Deflate
        X_res = X_res - np.outer(t, p)
        y_res = y_res - np.outer(t, q)

    # Compute final regression coefficients: B = W @ inv(P.T @ W) @ Q.T
    PtW = P.T @ W
    PtW_reg = PtW + 1e-10 * np.eye(n_components)
    PtW_inv = np.linalg.pinv(PtW_reg)
    B_final = W @ PtW_inv @ Q.T

    return B_final, W, P, Q, T, X_mean, X_std, y_mean, y_std


def _mbpls_fit_multiblock_numpy(X_blocks, y, n_components, standardize=True):
    """Fit multiblock MBPLS using pure NumPy.

    Parameters
    ----------
    X_blocks : list of ndarray
        List of X blocks, each of shape (n_samples, n_features_block).
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    n_components : int
        Number of components to extract.
    standardize : bool, default=True
        Whether to standardize blocks.

    Returns
    -------
    B : ndarray
        Regression coefficients for concatenated features.
    W : ndarray
        Weight matrix.
    P : ndarray
        X loading matrix.
    Q : ndarray
        Y loading matrix.
    T : ndarray
        Super score matrix.
    block_weights : list of ndarray
        Block-level weight matrices.
    X_means, X_stds : list of ndarray
        Preprocessing parameters for each block.
    y_mean, y_std : ndarray
        Preprocessing parameters for y.
    """
    n_blocks = len(X_blocks)
    n_samples = X_blocks[0].shape[0]

    y = np.asarray(y, dtype=np.float64)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n_targets = y.shape[1]

    # Preprocess y
    y_mean = np.mean(y, axis=0, keepdims=True)
    if standardize:
        y_std = np.std(y, axis=0, keepdims=True, ddof=1)
        y_std = np.where(y_std < 1e-10, 1.0, y_std)
    else:
        y_std = np.ones((1, n_targets), dtype=np.float64)
    y_centered = (y - y_mean) / y_std

    # Preprocess each block
    X_centered_blocks = []
    X_means = []
    X_stds = []
    block_sizes = []

    for X_block in X_blocks:
        X_block = np.asarray(X_block, dtype=np.float64)
        n_features_block = X_block.shape[1]
        block_sizes.append(n_features_block)

        X_mean = np.mean(X_block, axis=0, keepdims=True)
        if standardize:
            X_std = np.std(X_block, axis=0, keepdims=True, ddof=1)
            X_std = np.where(X_std < 1e-10, 1.0, X_std)
        else:
            X_std = np.ones((1, n_features_block), dtype=np.float64)

        X_means.append(X_mean)
        X_stds.append(X_std)
        X_centered_blocks.append((X_block - X_mean) / X_std)

    total_features = sum(block_sizes)

    # Initialize matrices
    W = np.zeros((total_features, n_components), dtype=np.float64)
    P = np.zeros((total_features, n_components), dtype=np.float64)
    Q = np.zeros((n_targets, n_components), dtype=np.float64)
    T = np.zeros((n_samples, n_components), dtype=np.float64)
    block_weights = [np.zeros((bs, n_components), dtype=np.float64) for bs in block_sizes]

    X_res_blocks = [X.copy() for X in X_centered_blocks]
    y_res = y_centered.copy()

    for comp in range(n_components):
        # Compute block scores and weights
        block_scores = []
        for b, X_res in enumerate(X_res_blocks):
            # Block weight
            w_b = X_res.T @ y_res
            if n_targets == 1:
                w_b = w_b.ravel()
            else:
                w_b = w_b[:, 0]
            w_norm = np.linalg.norm(w_b)
            if w_norm > 1e-10:
                w_b = w_b / w_norm

            # Block score
            t_b = X_res @ w_b
            block_scores.append(t_b)
            block_weights[b][:, comp] = w_b

        # Super score (average of block scores)
        t = np.mean(block_scores, axis=0)
        t_norm = np.linalg.norm(t)
        if t_norm > 1e-10:
            t = t / t_norm
        t_dot = t.T @ t + 1e-10

        T[:, comp] = t

        # Compute loadings for each block and concatenate
        p_concat = []
        for b, X_res in enumerate(X_res_blocks):
            p_b = X_res.T @ t / t_dot
            p_concat.append(p_b)

        p_full = np.concatenate(p_concat)
        P[:, comp] = p_full

        # Y loading
        q = y_res.T @ t / t_dot
        Q[:, comp] = q.ravel()

        # Concatenated weight
        w_full = np.concatenate([block_weights[b][:, comp] for b in range(n_blocks)])
        W[:, comp] = w_full

        # Deflate each block
        for b, X_res in enumerate(X_res_blocks):
            p_b = p_concat[b]
            X_res_blocks[b] = X_res - np.outer(t, p_b)

        y_res = y_res - np.outer(t, q)

    # Compute final regression coefficients
    PtW = P.T @ W
    PtW_reg = PtW + 1e-10 * np.eye(n_components)
    PtW_inv = np.linalg.pinv(PtW_reg)
    B_final = W @ PtW_inv @ Q.T

    return B_final, W, P, Q, T, block_weights, X_means, X_stds, y_mean, y_std


def _mbpls_predict_numpy(X, B, X_mean, X_std, y_mean, y_std):
    """Predict using MBPLS coefficients (NumPy)."""
    X = np.asarray(X, dtype=np.float64)
    X_centered = (X - X_mean) / X_std
    y_pred_centered = X_centered @ B
    return y_pred_centered * y_std + y_mean


def _mbpls_predict_multiblock_numpy(X_blocks, B, X_means, X_stds, y_mean, y_std):
    """Predict using multiblock MBPLS coefficients (NumPy)."""
    # Preprocess and concatenate blocks
    X_centered_list = []
    for b, X_block in enumerate(X_blocks):
        X_block = np.asarray(X_block, dtype=np.float64)
        X_centered = (X_block - X_means[b]) / X_stds[b]
        X_centered_list.append(X_centered)

    X_concat = np.hstack(X_centered_list)
    y_pred_centered = X_concat @ B
    return y_pred_centered * y_std + y_mean


def _mbpls_transform_numpy(X, W, X_mean, X_std):
    """Transform X to latent space (NumPy)."""
    X = np.asarray(X, dtype=np.float64)
    X_centered = (X - X_mean) / X_std
    return X_centered @ W


def _mbpls_transform_multiblock_numpy(X_blocks, W, X_means, X_stds, block_sizes):
    """Transform multiblock X to latent space (NumPy)."""
    X_centered_list = []
    for b, X_block in enumerate(X_blocks):
        X_block = np.asarray(X_block, dtype=np.float64)
        X_centered = (X_block - X_means[b]) / X_stds[b]
        X_centered_list.append(X_centered)

    X_concat = np.hstack(X_centered_list)
    return X_concat @ W

# =============================================================================
# JAX Backend Implementations
# =============================================================================

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

# Cache JAX functions
_JAX_MBPLS_FUNCS = None

def _get_cached_jax_mbpls():
    """Get cached JAX MBPLS functions."""
    global _JAX_MBPLS_FUNCS
    if _JAX_MBPLS_FUNCS is None:
        _JAX_MBPLS_FUNCS = _get_jax_mbpls_functions()
    return _JAX_MBPLS_FUNCS

class MBPLS(BaseEstimator, RegressorMixin):
    """Multiblock PLS (MB-PLS) regressor.

    MB-PLS fuses multiple X blocks (e.g., different preprocessing variants,
    multiple sensors) into a single predictive model. Each block contributes
    to the latent variables according to its relevance to Y.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent variables to extract.
    method : str, default='NIPALS'
        Decomposition method. Currently only 'NIPALS' is supported.
    standardize : bool, default=True
        Whether to standardize blocks before fitting.
    max_tol : float, default=1e-14
        Convergence tolerance for NIPALS.
    backend : str, default='numpy'
        Backend to use for computation. Options are:
        - 'numpy': Use NumPy backend (CPU only).
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
    >>> # Multiblock usage
    >>> X1 = np.random.randn(100, 30)
    >>> X2 = np.random.randn(100, 20)
    >>> model.fit([X1, X2], y)
    >>> # JAX backend for GPU acceleration
    >>> model_jax = MBPLS(n_components=5, backend='jax')

    Notes
    -----
    For JAX with GPU support: ``pip install jax[cuda12]``

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard single-block PLS.

    References
    ----------
    .. [1] Westerhuis, J. A., et al. (1998). Analysis of multiblock and
           hierarchical PCA and PLS models. Journal of Chemometrics, 12(5),
           301-321.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

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
            self._block_sizes = [x.shape[1] for x in X_blocks]
        else:
            X = np.asarray(X)
            X_blocks = [X]
            self.n_features_in_ = X.shape[1]
            self._is_multiblock = False
            self._block_sizes = [X.shape[1]]

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

            # Store coefficients
            self.coef_ = np.asarray(self._B)
        else:
            # NumPy backend - native implementation
            if self._is_multiblock:
                result = _mbpls_fit_multiblock_numpy(
                    X_blocks, y,
                    self.n_components_,
                    standardize=self.standardize
                )
                (self._B, self._W, self._P, self._Q, self._T,
                 self._block_weights, self._X_means, self._X_stds,
                 self._y_mean, self._y_std) = result
            else:
                result = _mbpls_fit_numpy(
                    X_blocks[0], y,
                    self.n_components_,
                    standardize=self.standardize
                )
                (self._B, self._W, self._P, self._Q, self._T,
                 self._X_mean, self._X_std,
                 self._y_mean, self._y_std) = result

            # Store coefficients
            self.coef_ = self._B

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
            if self._is_multiblock:
                y_pred = _mbpls_predict_multiblock_numpy(
                    X_blocks, self._B,
                    self._X_means, self._X_stds,
                    self._y_mean, self._y_std
                )
            else:
                y_pred = _mbpls_predict_numpy(
                    X_blocks[0], self._B,
                    self._X_mean, self._X_std,
                    self._y_mean, self._y_std
                )

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
        if isinstance(X, list):
            X_blocks = [np.asarray(x) for x in X]
        else:
            X = np.asarray(X)
            X_blocks = [X]

        if self.backend == 'jax':
            import jax.numpy as jnp

            X_jax = jnp.asarray(X_blocks[0])
            X_centered = (X_jax - self._X_mean) / self._X_std
            return np.asarray(X_centered @ self._W)
        else:
            if self._is_multiblock:
                return _mbpls_transform_multiblock_numpy(
                    X_blocks, self._W,
                    self._X_means, self._X_stds,
                    self._block_sizes
                )
            else:
                return _mbpls_transform_numpy(
                    X_blocks[0], self._W,
                    self._X_mean, self._X_std
                )

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

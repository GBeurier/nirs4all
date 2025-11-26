"""Locally-Weighted Partial Least Squares (LWPLS) model operator.

This module provides a sklearn-compatible LWPLS implementation for nirs4all.
The core algorithm is adapted from the original implementation by Hiromasa Kaneko
(https://github.com/hkaneko1985/lwpls), licensed under MIT License.

LWPLS builds just-in-time local PLS models near each query sample, which is
useful when dealing with drift, local nonlinearity, or heterogeneous data.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
.. [1] Kim, S., Kano, M., Nakagawa, H., & Hasebe, S. (2011).
       Estimation of active pharmaceutical ingredient content using
       locally weighted partial least squares and statistical wavelength
       selection. International Journal of Pharmaceutics, 421(2), 269-274.

.. [2] https://datachemeng.com/locallyweightedpartialleastsquares/

License
-------
Original lwpls.py by Hiromasa Kaneko is MIT licensed.
"""

from __future__ import annotations

from functools import partial
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


def _lwpls_predict(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    max_component_number: int,
    lambda_in_similarity: float,
) -> NDArray[np.floating]:
    """Core LWPLS prediction algorithm.

    Builds a locally-weighted PLS model for each test sample using
    Gaussian kernel weights based on Euclidean distance.

    Parameters
    ----------
    x_train : ndarray of shape (n_train, n_features)
        Autoscaled training X data.
    y_train : ndarray of shape (n_train,) or (n_train, 1)
        Autoscaled training y data.
    x_test : ndarray of shape (n_test, n_features)
        Autoscaled test X data.
    max_component_number : int
        Maximum number of PLS components to extract.
    lambda_in_similarity : float
        Parameter controlling the kernel width. Smaller values give
        more localized models; larger values approach global PLS.

    Returns
    -------
    estimated_y_test : ndarray of shape (n_test, max_component_number)
        Predictions for each number of components (column i contains
        predictions using i+1 components).

    Notes
    -----
    The algorithm:
    1. For each test sample, compute distances to all training samples
    2. Convert distances to similarities using Gaussian kernel
    3. Compute weighted mean of X and Y
    4. Build weighted PLS components iteratively
    5. Predict Y by accumulating component contributions
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_train = np.reshape(y_train, (len(y_train), 1))
    x_test = np.asarray(x_test, dtype=np.float64)

    n_test = x_test.shape[0]
    n_train = x_train.shape[0]
    n_features = x_train.shape[1]

    estimated_y_test = np.zeros((n_test, max_component_number))

    # Precompute distance matrix for efficiency
    distance_matrix = cdist(x_train, x_test, metric='euclidean')

    for test_idx in range(n_test):
        query_x_test = x_test[test_idx:test_idx + 1, :]

        # Get distances and compute similarities
        distance = distance_matrix[:, test_idx]
        distance_std = distance.std(ddof=1) if distance.std(ddof=1) > 0 else 1.0

        # Gaussian kernel weights
        similarity_weights = np.exp(-distance / distance_std / lambda_in_similarity)
        similarity = np.diag(similarity_weights)
        similarity_sum = similarity_weights.sum()

        if similarity_sum < 1e-10:
            # All samples too far away; use uniform weights
            similarity_weights = np.ones(n_train) / n_train
            similarity = np.diag(similarity_weights)
            similarity_sum = 1.0

        # Weighted means - use diagonal of similarity matrix
        similarity_diag = np.diag(similarity)  # Extract diagonal as 1D array
        y_w = (y_train.T @ similarity_diag / similarity_sum).reshape(1, 1)
        x_w = (x_train.T @ similarity_diag / similarity_sum).reshape(1, n_features)

        # Center data
        centered_y = y_train - y_w
        centered_x = x_train - np.ones((n_train, 1)) @ x_w
        centered_query_x_test = query_x_test - x_w

        # Initialize prediction with weighted mean
        estimated_y_test[test_idx, :] += y_w.ravel()[0]

        # Build PLS components
        for comp_num in range(max_component_number):
            # Weighted loading direction
            numerator = centered_x.T @ similarity @ centered_y
            norm_val = np.linalg.norm(numerator)

            if norm_val < 1e-10:
                # Degenerate case - no more variance to explain
                break

            w_a = numerator / norm_val

            # Scores
            t_a = centered_x @ w_a

            # Loadings
            denom = t_a.T @ similarity @ t_a
            if denom < 1e-10:
                break

            p_a = (centered_x.T @ similarity @ t_a) / denom
            q_a = (centered_y.T @ similarity @ t_a) / denom

            # Query score
            t_q_a = centered_query_x_test @ w_a

            # Accumulate prediction for this and all subsequent components
            estimated_y_test[test_idx, comp_num:] += (t_q_a * q_a).ravel()[0]

            # Deflate for next component
            if comp_num < max_component_number - 1:
                centered_x = centered_x - t_a @ p_a.T
                centered_y = centered_y - t_a * q_a
                centered_query_x_test = centered_query_x_test - t_q_a @ p_a.T

    return estimated_y_test


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_lwpls_functions():
    """Lazy import and create JAX LWPLS functions.

    Returns the JAX-accelerated prediction function. This is done lazily
    to avoid importing JAX unless needed.

    Returns
    -------
    lwpls_predict_jax : callable
        JAX-accelerated LWPLS prediction function.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    # Enable float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    def _lwpls_single_query(
        x_train: jax.Array,
        y_train: jax.Array,
        query_x: jax.Array,
        max_components: int,
        lambda_sim: float,
    ) -> jax.Array:
        """LWPLS prediction for a single query sample.

        Parameters
        ----------
        x_train : jax.Array of shape (n_train, n_features)
            Training X data.
        y_train : jax.Array of shape (n_train, 1)
            Training y data.
        query_x : jax.Array of shape (n_features,)
            Single query sample.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : jax.Array of shape (max_components,)
            Predictions for each number of components.
        """
        n_train, n_features = x_train.shape

        # Compute Euclidean distances from query to all training samples
        diff = x_train - query_x[jnp.newaxis, :]
        distances = jnp.sqrt(jnp.sum(diff ** 2, axis=1))

        # Compute distance std (with Bessel correction, matching NumPy)
        dist_mean = jnp.mean(distances)
        dist_std = jnp.sqrt(jnp.sum((distances - dist_mean) ** 2) / (n_train - 1))
        dist_std = jnp.maximum(dist_std, 1e-10)  # Avoid division by zero

        # Gaussian kernel weights
        weights = jnp.exp(-distances / dist_std / lambda_sim)
        weight_sum = jnp.sum(weights)

        # Handle degenerate case
        weights = lax.cond(
            weight_sum < 1e-10,
            lambda w: jnp.ones(n_train) / n_train,
            lambda w: w,
            weights,
        )
        weight_sum = lax.cond(
            weight_sum < 1e-10,
            lambda _: 1.0,
            lambda ws: ws,
            weight_sum,
        )

        # Weighted means
        y_w = jnp.sum(y_train[:, 0] * weights) / weight_sum
        x_w = jnp.sum(x_train * weights[:, jnp.newaxis], axis=0) / weight_sum

        # Center data
        centered_x = x_train - x_w[jnp.newaxis, :]
        centered_y = y_train - y_w
        centered_query = query_x - x_w

        # Initialize predictions with weighted mean
        predictions = jnp.full(max_components, y_w)

        # Build PLS components using lax.fori_loop for JIT compatibility
        def component_step(comp_idx, carry):
            centered_x, centered_y, centered_query, predictions, weights = carry

            # Weighted loading direction: X^T @ W @ y
            # W is diagonal, so X^T @ W @ y = sum(x_i * w_i * y_i)
            numerator = jnp.sum(
                centered_x * (weights * centered_y[:, 0])[:, jnp.newaxis],
                axis=0,
            )
            norm_val = jnp.linalg.norm(numerator)

            # Safe normalization
            w_a = lax.cond(
                norm_val < 1e-10,
                lambda n: jnp.zeros(n_features),
                lambda n: numerator / norm_val,
                numerator,
            )

            # Scores: t = X @ w
            t_a = centered_x @ w_a  # shape: (n_train,)

            # Weighted denominator: t^T @ W @ t
            denom = jnp.sum(t_a ** 2 * weights)
            denom = jnp.maximum(denom, 1e-10)

            # Loadings
            # p = (X^T @ W @ t) / denom
            p_a = jnp.sum(centered_x * (weights * t_a)[:, jnp.newaxis], axis=0) / denom
            # q = (y^T @ W @ t) / denom
            q_a = jnp.sum(centered_y[:, 0] * weights * t_a) / denom

            # Query score
            t_q = jnp.dot(centered_query, w_a)

            # Update predictions for this and all subsequent components
            contribution = t_q * q_a
            # Add contribution to predictions[comp_idx:]
            mask = jnp.arange(max_components) >= comp_idx
            predictions = predictions + contribution * mask

            # Deflate for next component
            centered_x = centered_x - jnp.outer(t_a, p_a)
            centered_y = centered_y - (t_a * q_a)[:, jnp.newaxis]
            centered_query = centered_query - t_q * p_a

            return (centered_x, centered_y, centered_query, predictions, weights)

        # Run the component loop
        init_carry = (centered_x, centered_y, centered_query, predictions, weights)
        _, _, _, predictions, _ = lax.fori_loop(
            0, max_components, component_step, init_carry
        )

        return predictions

    # Vectorize over test samples using vmap
    _lwpls_batch = jax.vmap(
        _lwpls_single_query,
        in_axes=(None, None, 0, None, None),  # Vectorize over query samples
    )

    @partial(jax.jit, static_argnums=(3,))
    def lwpls_predict_jax(
        x_train: jax.Array,
        y_train: jax.Array,
        x_test: jax.Array,
        max_components: int,
        lambda_sim: float,
    ) -> jax.Array:
        """JIT-compiled LWPLS prediction for batch of test samples.

        Parameters
        ----------
        x_train : jax.Array of shape (n_train, n_features)
            Training X data.
        y_train : jax.Array of shape (n_train, 1)
            Training y data.
        x_test : jax.Array of shape (n_test, n_features)
            Test X data.
        max_components : int
            Maximum number of PLS components.
        lambda_sim : float
            Kernel width parameter.

        Returns
        -------
        predictions : jax.Array of shape (n_test, max_components)
            Predictions for each test sample and number of components.
        """
        return _lwpls_batch(x_train, y_train, x_test, max_components, lambda_sim)

    return lwpls_predict_jax


# Cache the JAX function to avoid re-creating it
_JAX_LWPLS_FUNC = None


def _lwpls_predict_jax(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    x_test: NDArray[np.floating],
    max_component_number: int,
    lambda_in_similarity: float,
) -> NDArray[np.floating]:
    """JAX-accelerated LWPLS prediction.

    Same interface as _lwpls_predict but uses JAX for GPU/TPU acceleration.

    Parameters
    ----------
    x_train : ndarray of shape (n_train, n_features)
        Autoscaled training X data.
    y_train : ndarray of shape (n_train,) or (n_train, 1)
        Autoscaled training y data.
    x_test : ndarray of shape (n_test, n_features)
        Autoscaled test X data.
    max_component_number : int
        Maximum number of PLS components to extract.
    lambda_in_similarity : float
        Parameter controlling the kernel width.

    Returns
    -------
    estimated_y_test : ndarray of shape (n_test, max_component_number)
        Predictions for each number of components.
    """
    global _JAX_LWPLS_FUNC

    if _JAX_LWPLS_FUNC is None:
        _JAX_LWPLS_FUNC = _get_jax_lwpls_functions()

    import jax.numpy as jnp

    # Convert to JAX arrays
    x_train_jax = jnp.asarray(x_train, dtype=jnp.float64)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float64)
    if y_train_jax.ndim == 1:
        y_train_jax = y_train_jax.reshape(-1, 1)
    x_test_jax = jnp.asarray(x_test, dtype=jnp.float64)

    # Run JAX prediction
    predictions_jax = _JAX_LWPLS_FUNC(
        x_train_jax,
        y_train_jax,
        x_test_jax,
        max_component_number,
        lambda_in_similarity,
    )

    # Convert back to NumPy
    return np.asarray(predictions_jax)


class LWPLS(BaseEstimator, RegressorMixin):
    """Locally-Weighted Partial Least Squares (LWPLS) regressor.

    LWPLS builds a local PLS model for each query sample, weighting
    training samples by their similarity (proximity) to the query.
    This approach is useful for:

    - Data with local nonlinearity
    - Drifting processes where the relationship changes over time
    - Heterogeneous data where a single global model is inadequate

    The similarity is computed using a Gaussian kernel based on
    Euclidean distance, controlled by the `lambda_in_similarity` parameter.

    Parameters
    ----------
    n_components : int, default=10
        Maximum number of PLS components to extract for each local model.
    lambda_in_similarity : float, default=1.0
        Kernel width parameter. Smaller values create more localized models
        (more weight on nearby samples), larger values approach global PLS.
        Typical values range from 2^-9 to 2^5 depending on the data.
    scale : bool, default=True
        Whether to standardize X and y before fitting. Strongly recommended
        as LWPLS uses Euclidean distances.
    backend : str, default='numpy'
        Computational backend to use. Options are:
        - 'numpy': NumPy backend (CPU only, default).
        - 'jax': JAX backend (supports GPU/TPU acceleration).
        JAX backend requires JAX to be installed: ``pip install jax``
        For GPU support: ``pip install jax[cuda12]``

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_components_ : int
        Actual number of components used (limited by data dimensions).
    X_train_ : ndarray of shape (n_samples, n_features)
        Stored training X data (standardized if scale=True).
    y_train_ : ndarray of shape (n_samples,)
        Stored training y data (standardized if scale=True).
    x_scaler_ : StandardScaler or None
        Fitted scaler for X (if scale=True).
    y_scaler_ : StandardScaler or None
        Fitted scaler for y (if scale=True).

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.lwpls import LWPLS
    >>> import numpy as np
    >>> # Nonlinear data
    >>> np.random.seed(42)
    >>> X = 5 * np.random.rand(100, 2)
    >>> y = 3 * X[:, 0]**2 + 10 * np.log(X[:, 1] + 0.1) + np.random.randn(100)
    >>> # Split data
    >>> X_train, X_test = X[:70], X[70:]
    >>> y_train, y_test = y[:70], y[70:]
    >>> # Fit LWPLS with NumPy backend (default)
    >>> model = LWPLS(n_components=5, lambda_in_similarity=0.25)
    >>> model.fit(X_train, y_train)
    LWPLS(n_components=5, lambda_in_similarity=0.25)
    >>> y_pred = model.predict(X_test)
    >>> # Use JAX backend for GPU acceleration
    >>> model_jax = LWPLS(n_components=5, lambda_in_similarity=0.25, backend='jax')
    >>> model_jax.fit(X_train, y_train)
    >>> y_pred_jax = model_jax.predict(X_test)

    Notes
    -----
    LWPLS is computationally more expensive than standard PLS because
    it builds a separate weighted model for each prediction. The training
    data must be stored for prediction.

    The JAX backend provides significant speedups on GPU by:
    - Vectorizing the per-sample loop using ``jax.vmap``
    - JIT-compiling the prediction function
    - Running on GPU/TPU when available

    The optimal `lambda_in_similarity` should be tuned via cross-validation.
    Typical search range is 2^k for k in [-9, 6].

    This implementation is adapted from the original code by Hiromasa Kaneko
    (https://github.com/hkaneko1985/lwpls), licensed under MIT License.

    See Also
    --------
    sklearn.cross_decomposition.PLSRegression : Standard global PLS.
    IKPLS : Fast PLS implementation.

    References
    ----------
    .. [1] Kim, S., et al. (2011). Estimation of active pharmaceutical
           ingredient content using locally weighted partial least squares.
           International Journal of Pharmaceutics, 421(2), 269-274.
    """

    def __init__(
        self,
        n_components: int = 10,
        lambda_in_similarity: float = 1.0,
        scale: bool = True,
        backend: str = 'numpy',
    ):
        """Initialize LWPLS regressor.

        Parameters
        ----------
        n_components : int, default=10
            Maximum number of PLS components.
        lambda_in_similarity : float, default=1.0
            Kernel width parameter for similarity computation.
        scale : bool, default=True
            Whether to standardize X and y.
        backend : str, default='numpy'
            Computational backend ('numpy' or 'jax').
        """
        self.n_components = n_components
        self.lambda_in_similarity = lambda_in_similarity
        self.scale = scale
        self.backend = backend

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "LWPLS":
        """Fit the LWPLS model.

        This stores the training data and fits scalers if requested.
        Actual model building happens lazily at prediction time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : LWPLS
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
                "JAX is required for LWPLS with backend='jax'. "
                "Install it with: pip install jax\n"
                "For GPU support: pip install jax[cuda12]"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.n_features_in_ = X.shape[1]

        # Limit components by data dimensions
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components_ = min(self.n_components, max_components)

        if self.scale:
            self.x_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()

            self.X_train_ = self.x_scaler_.fit_transform(X)
            self.y_train_ = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            self.x_scaler_ = None
            self.y_scaler_ = None
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()

        # Store original data for reference
        self._n_train_samples = X.shape[0]

        return self

    def predict(
        self,
        X: ArrayLike,
        n_components: Union[int, None] = None,
    ) -> NDArray[np.floating]:
        """Predict using the LWPLS model.

        Builds a local weighted PLS model for each test sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use for prediction.
            If None, uses n_components_ (all fitted components).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'n_components_'])

        X = np.asarray(X, dtype=np.float64)

        if n_components is None:
            n_components = self.n_components_
        else:
            n_components = min(n_components, self.n_components_)

        # Scale input if needed
        if self.scale and self.x_scaler_ is not None:
            X_scaled = self.x_scaler_.transform(X)
        else:
            X_scaled = X

        # Get predictions for all component numbers using appropriate backend
        if self.backend == 'jax':
            all_predictions = _lwpls_predict_jax(
                self.X_train_,
                self.y_train_,
                X_scaled,
                n_components,
                self.lambda_in_similarity,
            )
        else:
            all_predictions = _lwpls_predict(
                self.X_train_,
                self.y_train_,
                X_scaled,
                n_components,
                self.lambda_in_similarity,
            )

        # Take prediction from the requested number of components
        y_pred_scaled = all_predictions[:, n_components - 1]

        # Inverse transform if needed
        if self.scale and self.y_scaler_ is not None:
            y_pred = self.y_scaler_.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).ravel()
        else:
            y_pred = y_pred_scaled

        return y_pred

    def predict_all_components(
        self,
        X: ArrayLike,
    ) -> NDArray[np.floating]:
        """Predict with all component numbers (for component selection).

        Returns predictions for each number of components, which can be
        used for cross-validation to select the optimal n_components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred_all : ndarray of shape (n_samples, n_components_)
            Predictions where column i contains predictions using i+1 components.
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'n_components_'])

        X = np.asarray(X, dtype=np.float64)

        # Scale input if needed
        if self.scale and self.x_scaler_ is not None:
            X_scaled = self.x_scaler_.transform(X)
        else:
            X_scaled = X

        # Get predictions for all component numbers using appropriate backend
        if self.backend == 'jax':
            all_predictions = _lwpls_predict_jax(
                self.X_train_,
                self.y_train_,
                X_scaled,
                self.n_components_,
                self.lambda_in_similarity,
            )
        else:
            all_predictions = _lwpls_predict(
                self.X_train_,
                self.y_train_,
                X_scaled,
                self.n_components_,
                self.lambda_in_similarity,
            )

        # Inverse transform if needed
        if self.scale and self.y_scaler_ is not None:
            # Need to inverse transform each column
            y_pred_all = np.zeros_like(all_predictions)
            for i in range(all_predictions.shape[1]):
                y_pred_all[:, i] = self.y_scaler_.inverse_transform(
                    all_predictions[:, i : i + 1]
                ).ravel()
        else:
            y_pred_all = all_predictions

        return y_pred_all

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
            'lambda_in_similarity': self.lambda_in_similarity,
            'scale': self.scale,
            'backend': self.backend,
        }

    def set_params(self, **params) -> "LWPLS":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : LWPLS
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LWPLS(n_components={self.n_components}, "
            f"lambda_in_similarity={self.lambda_in_similarity}, "
            f"scale={self.scale}, backend='{self.backend}')"
        )

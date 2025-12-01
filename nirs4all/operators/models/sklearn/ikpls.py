"""Improved Kernel PLS (IKPLS) regressor for nirs4all.

A sklearn-compatible wrapper for the ikpls package, which provides
fast PLS implementations using NumPy or JAX (for GPU/TPU acceleration).
IKPLS is significantly faster than sklearn's PLSRegression, especially
for cross-validation.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_ikpls_available():
    """Check if ikpls package is available."""
    try:
        import ikpls
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

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

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
            "n_components": self.n_components,
            "algorithm": self.algorithm,
            "center": self.center,
            "scale": self.scale,
            "backend": self.backend
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

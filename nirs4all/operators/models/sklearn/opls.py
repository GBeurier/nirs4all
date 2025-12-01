# =============================================================================
# JAX Backend Implementations for OPLS
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
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        n_samples, n_features = X.shape
        X_mean = jnp.mean(X, axis=0, keepdims=True)
        X_std = jnp.std(X, axis=0, keepdims=True, ddof=1)
        X_std = jnp.where(X_std < 1e-10, 1.0, X_std)
        X_centered = (X - X_mean) / X_std
        y_mean = jnp.mean(y)
        y_std = jnp.std(y, ddof=1)
        y_std = jnp.where(y_std < 1e-10, 1.0, y_std)
        y_centered = (y - y_mean) / y_std
        W_ortho = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P_ortho = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        def component_step(i, carry):
            X_res, W_ortho, P_ortho = carry
            w = X_res.T @ y_centered
            w = w / (jnp.linalg.norm(w) + 1e-10)
            t = X_res @ w
            p = X_res.T @ t / (t.T @ t + 1e-10)
            w_ortho = p - (w.T @ p) * w
            w_ortho = w_ortho / (jnp.linalg.norm(w_ortho) + 1e-10)
            t_ortho = X_res @ w_ortho
            p_ortho = X_res.T @ t_ortho / (t_ortho.T @ t_ortho + 1e-10)
            X_res = X_res - jnp.outer(t_ortho, p_ortho)
            W_ortho = W_ortho.at[:, i].set(w_ortho)
            P_ortho = P_ortho.at[:, i].set(p_ortho)
            return X_res, W_ortho, P_ortho
        X_filtered, W_ortho, P_ortho = lax.fori_loop(
            0, n_components, component_step, (X_centered, W_ortho, P_ortho)
        )
        return W_ortho, P_ortho, X_mean, X_std, y_mean, y_std

    @jax.jit
    def opls_transform_jax(X, W_ortho, P_ortho, X_mean, X_std):
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

# Cache for JAX OPLS functions
_JAX_OPLS_FUNCS = None
def _get_cached_jax_opls():
    global _JAX_OPLS_FUNCS
    if _JAX_OPLS_FUNCS is None:
        _JAX_OPLS_FUNCS = _get_jax_opls_functions()
    return _JAX_OPLS_FUNCS
"""Orthogonal PLS (OPLS) regressor for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression

def _check_pyopls_available():
    try:
        import pyopls
        return True
    except ImportError:
        return False

class OPLS(BaseEstimator, RegressorMixin):
    """Orthogonal PLS (OPLS) regressor.
    (See pls.py for full docstring)
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(self, n_components: int = 1, pls_components: int = 1, scale: bool = True, backend: str = 'numpy'):
        self.n_components = n_components
        self.pls_components = pls_components
        self.scale = scale
        self.backend = backend

    def fit(self, X, y):
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
            try:
                import jax
                import jax.numpy as jnp
            except ImportError:
                raise ImportError(
                    "JAX is required for OPLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

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

            import pyopls

            # Fit OPLS transformer
            self.opls_ = pyopls.OPLS(n_components=n_ortho, scale=self.scale)
            X_filtered = self.opls_.fit_transform(X, y_flat)

        # Fit PLS on filtered data (both backends)
        max_pls = min(self.pls_components, X_filtered.shape[1], X_filtered.shape[0] - 1)
        n_pls = max(1, max_pls)

        self.pls_ = PLSRegression(n_components=n_pls)
        self.pls_.fit(X_filtered, y)

        return self

    def predict(self, X):
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
        return {"n_components": self.n_components, "pls_components": self.pls_components, "scale": self.scale, "backend": self.backend}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

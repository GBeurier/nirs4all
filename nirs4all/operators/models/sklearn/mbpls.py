# =============================================================================
# JAX Backend Implementations for MBPLS
# =============================================================================

def _get_jax_mbpls_functions():
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    @partial(jax.jit, static_argnums=(2,))
    def mbpls_fit_jax(X, y, n_components):
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        n_samples, n_features = X.shape
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
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        T = jnp.zeros((n_samples, n_components), dtype=jnp.float64)
        def component_step(i, carry):
            X_res, y_res, W, P, Q, T = carry
            w = X_res.T @ y_res
            if n_targets == 1:
                w = w.ravel()
            else:
                w = w[:, 0]
            w = w / (jnp.linalg.norm(w) + 1e-10)
            t = X_res @ w
            t_norm = t.T @ t + 1e-10
            p = X_res.T @ t / t_norm
            q = y_res.T @ t / t_norm
            W = W.at[:, i].set(w)
            P = P.at[:, i].set(p)
            Q = Q.at[:, i].set(q.ravel())
            T = T.at[:, i].set(t)
            X_res = X_res - jnp.outer(t, p)
            y_res = y_res - jnp.outer(t, q)
            return X_res, y_res, W, P, Q, T
        _, _, W, P, Q, T = lax.fori_loop(
            0, n_components, component_step,
            (X_centered, y_centered, W, P, Q, T)
        )
        PtW = P.T @ W
        PtW_reg = PtW + 1e-10 * jnp.eye(n_components)
        PtW_inv = jnp.linalg.pinv(PtW_reg)
        B_final = W @ PtW_inv @ Q.T
        return B_final, W, P, Q, T, X_mean, X_std, y_mean, y_std

    @jax.jit
    def mbpls_predict_jax(X, B, X_mean, X_std, y_mean, y_std):
        X = jnp.asarray(X, dtype=jnp.float64)
        X_centered = (X - X_mean) / X_std
        y_pred_centered = X_centered @ B
        return y_pred_centered * y_std + y_mean

    return mbpls_fit_jax, mbpls_predict_jax

# Cache for JAX MBPLS functions
_JAX_MBPLS_FUNCS = None
def _get_cached_jax_mbpls():
    global _JAX_MBPLS_FUNCS
    if _JAX_MBPLS_FUNCS is None:
        _JAX_MBPLS_FUNCS = _get_jax_mbpls_functions()
    return _JAX_MBPLS_FUNCS
"""Multiblock PLS (MBPLS) regressor for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_mbpls_available():
    try:
        import mbpls
        return True
    except ImportError:
        return False

class MBPLS(BaseEstimator, RegressorMixin):
    """Multiblock PLS (MBPLS) regressor.
    (See pls.py for full docstring)
    """
    def __init__(self, n_components: int = 5, method: str = 'NIPALS', standardize: bool = True, max_tol: float = 1e-14, backend: str = 'numpy'):
        self.n_components = n_components
        self.method = method
        self.standardize = standardize
        self.max_tol = max_tol
        self.backend = backend

    def fit(self, X, y):
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
            try:
                import jax
                import jax.numpy as jnp
            except ImportError:
                raise ImportError(
                    "JAX is required for MBPLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

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

            self.model_ = None  # Not using mbpls package
        else:
            # NumPy backend using mbpls
            if not _check_mbpls_available():
                raise ImportError(
                    "mbpls package is required for MBPLS with backend='numpy'. "
                    "Install it with: pip install mbpls"
                )

            import mbpls

            # Fit MB-PLS model
            self.model_ = mbpls.MBPLS(
                n_components=self.n_components_,
                method=self.method,
                standardize=self.standardize,
                max_tol=self.max_tol,
            )
            self.model_.fit(X_blocks, y)

            # Store coefficients for compatibility
            self.coef_ = self.model_.beta_

        return self

    def predict(self, X):
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
            y_pred = self.model_.predict(X_blocks)

        # Flatten if single target
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X):
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

        return self.model_.transform(X_blocks)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "method": self.method, "standardize": self.standardize, "max_tol": self.max_tol, "backend": self.backend}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

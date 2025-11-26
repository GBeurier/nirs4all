# =============================================================================
# JAX Backend Implementations for SparsePLS
# =============================================================================

def _get_jax_sparse_pls_functions():
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    def soft_threshold(z, alpha):
        return jnp.sign(z) * jnp.maximum(jnp.abs(z) - alpha, 0.0)

    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def sparse_pls_fit_jax(X, y, n_components, alpha, max_iter, tol):
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        n_samples, n_features = X.shape
        X_mean = jnp.mean(X, axis=0, keepdims=True)
        X_std = jnp.std(X, axis=0, keepdims=True, ddof=0)
        X_std = jnp.where(X_std < 1e-10, 1.0, X_std)
        X_scaled = (X - X_mean) / X_std
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        n_targets = y.shape[1]
        y_mean = jnp.mean(y, axis=0, keepdims=True)
        y_std = jnp.std(y, axis=0, keepdims=True, ddof=0)
        y_std = jnp.where(y_std < 1e-10, 1.0, y_std)
        y_scaled = (y - y_mean) / y_std
        W = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        C = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        P = jnp.zeros((n_features, n_components), dtype=jnp.float64)
        Q = jnp.zeros((n_targets, n_components), dtype=jnp.float64)
        def compute_sparse_component(X_res, Y_res, key):
            c = jnp.ones((n_targets, 1), dtype=jnp.float64)
            c = c / jnp.linalg.norm(c)
            def sparse_iter_body(carry):
                c, _ = carry
                z_w = X_res.T @ Y_res @ c
                w = soft_threshold(z_w, alpha)
                w_norm = jnp.linalg.norm(w)
                w = lax.cond(w_norm > 1e-10, lambda: w / w_norm, lambda: w)
                t = X_res @ w
                z_c = Y_res.T @ t
                c_new = soft_threshold(z_c, alpha)
                c_norm = jnp.linalg.norm(c_new)
                c_new = lax.cond(c_norm > 1e-10, lambda: c_new / c_norm, lambda: c_new)
                change = jnp.linalg.norm(c_new - c)
                return (c_new, change)
            def sparse_iter_cond(carry):
                _, change = carry
                return change >= tol
            def run_iterations(c_init):
                c, change = sparse_iter_body((c_init, jnp.float64(1.0)))
                def iteration_step(i, carry):
                    c, change, converged = carry
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
            z_w = X_res.T @ Y_res @ c_final
            w = soft_threshold(z_w, alpha)
            w_norm = jnp.linalg.norm(w)
            w = lax.cond(w_norm > 1e-10, lambda: w / w_norm, lambda: w)
            return w.ravel(), c_final.ravel()
        def component_step(comp_i, carry):
            X_res, Y_res, W, C, P, Q, key = carry
            key, subkey = jax.random.split(key)
            w, c = compute_sparse_component(X_res, Y_res, subkey)
            t = X_res @ w
            u = Y_res @ c
            t_norm = jnp.linalg.norm(t)
            t_safe = lax.cond(t_norm > 1e-10, lambda: t / t_norm, lambda: t)
            u_safe = lax.cond(t_norm > 1e-10, lambda: u / t_norm, lambda: u)
            p = X_res.T @ t_safe
            q = Y_res.T @ t_safe
            W = W.at[:, comp_i].set(w)
            C = C.at[:, comp_i].set(c)
            P = P.at[:, comp_i].set(p)
            Q = Q.at[:, comp_i].set(q)
            X_res = X_res - jnp.outer(t_safe, p)
            Y_res = Y_res - jnp.outer(t_safe, q)
            return X_res, Y_res, W, C, P, Q, key
        key = jax.random.PRNGKey(42)
        _, _, W, C, P, Q, _ = lax.fori_loop(
            0, n_components, component_step,
            (X_scaled, y_scaled, W, C, P, Q, key)
        )
        PtW = P.T @ W
        PtW_reg = PtW + 1e-5 * jnp.eye(n_components)
        PtW_inv = jnp.linalg.pinv(PtW_reg)
        B_final = W @ PtW_inv @ Q.T
        return B_final, W, P, Q, X_mean, X_std, y_mean, y_std

    @jax.jit
    def sparse_pls_predict_jax(X, B, X_mean, X_std, y_mean, y_std):
        X = jnp.asarray(X, dtype=jnp.float64)
        X_scaled = (X - X_mean) / X_std
        y_pred_scaled = X_scaled @ B
        return y_pred_scaled * y_std + y_mean

    return sparse_pls_fit_jax, sparse_pls_predict_jax

# Cache for JAX SparsePLS functions
_JAX_SPARSE_PLS_FUNCS = None
def _get_cached_jax_sparse_pls():
    global _JAX_SPARSE_PLS_FUNCS
    if _JAX_SPARSE_PLS_FUNCS is None:
        _JAX_SPARSE_PLS_FUNCS = _get_jax_sparse_pls_functions()
    return _JAX_SPARSE_PLS_FUNCS
"""Sparse PLS (sPLS) regressor with L1 regularization for nirs4all.

See pls.py for full documentation and usage examples.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

def _check_sparse_pls_available():
    try:
        from sparse_pls import SparsePLS
        return True
    except ImportError:
        return False

class SparsePLS(BaseEstimator, RegressorMixin):
    """Sparse PLS (sPLS) regressor with L1 regularization.
    (See pls.py for full docstring)
    """
    def __init__(self, n_components: int = 5, alpha: float = 1.0, max_iter: int = 500, tol: float = 1e-6, scale: bool = True, backend: str = 'numpy'):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
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

        self.n_features_in_ = X.shape[1]

        # Limit components
        max_components = min(X.shape[0] - 1, X.shape[1])
        self.n_components_ = min(self.n_components, max_components)

        if self.backend == 'jax':
            try:
                import jax
                import jax.numpy as jnp
            except ImportError:
                raise ImportError(
                    "JAX is required for SparsePLS with backend='jax'. "
                    "Install it with: pip install jax\n"
                    "For GPU support: pip install jax[cuda12]"
                )

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

            self.model_ = None  # Not using sparse-pls package
        else:
            # NumPy backend using sparse-pls
            if not _check_sparse_pls_available():
                raise ImportError(
                    "sparse-pls package is required for SparsePLS with backend='numpy'. "
                    "Install it with: pip install sparse-pls"
                )

            # Suppress verbose INFO logging from sparse_pls
            import logging
            logging.getLogger('sparse_pls.model').setLevel(logging.WARNING)

            from sparse_pls import SparsePLS as _SparsePLS

            # Create and fit sparse-pls model
            self.model_ = _SparsePLS(
                n_components=self.n_components_,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                scale=self.scale,
            )
            self.model_.fit(X, y)

            # Store coefficients for compatibility
            if hasattr(self.model_, 'coef_'):
                self.coef_ = self.model_.coef_
            else:
                self.coef_ = None

        return self

    def predict(self, X):
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
            return np.asarray(y_pred)
        else:
            return self.model_.predict(X)

    def transform(self, X):
        if self.backend == 'jax':
             raise NotImplementedError(
                "transform() is not implemented for JAX backend. "
                "Use backend='numpy' for transform functionality."
            )
        return self.model_.transform(X)

    def get_selected_features(self):
        if self.backend == 'jax':
             raise NotImplementedError(
                "get_selected_features() is not implemented for JAX backend. "
                "Use backend='numpy' for transform functionality."
            )
        return self.model_.get_selected_features()

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "alpha": self.alpha, "max_iter": self.max_iter, "tol": self.tol, "scale": self.scale, "backend": self.backend}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

"""Online Koopman Latent-Mode PLS (OKLM-PLS) regressor for nirs4all.

A sklearn-compatible implementation of OKLM-PLS that combines Koopman operator
theory with PLS for time-series regression. The model learns latent dynamics
T_{t+1} ≈ F @ T_t while simultaneously fitting Y_t ≈ T_t @ B.

This is useful for spectral data collected over time where temporal coherence
provides additional information for prediction.

Supports both NumPy (CPU) and JAX (GPU/TPU) backends.

References
----------
.. [1] Brunton, S. L., Budisic, M., Kaiser, E., & Kutz, J. N. (2021).
       Modern Koopman theory for dynamical systems. arXiv:2102.12086.

.. [2] Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015).
       A data-driven approximation of the Koopman operator: Extending
       dynamic mode decomposition. Journal of Nonlinear Science, 25, 1307-1346.

Mathematical formulation
------------------------
Let

- X ∈ ℝ^{n×p} be the input matrix of n samples and p features
  (e.g. NIRS spectra), ordered in time t = 1,…,n.
- Y ∈ ℝ^{n×q} be the corresponding response matrix.

An optional feature map ψ : ℝ^p → ℝ^d is applied to each row x_t,
giving Z ∈ ℝ^{n×d} with rows

    z_t = ψ(x_t),       t = 1,…,n.

OKLM-PLS seeks:

- a loading matrix W ∈ ℝ^{d×r} (r latent components),
- a latent dynamic matrix F ∈ ℝ^{r×r},
- a regression matrix B ∈ ℝ^{r×q},

such that the latent scores

    T = Z W  ∈ ℝ^{n×r},       t_t = row t of T,

(1) capture covariance between X (via Z) and Y in the PLS sense, (2) follow
a simple linear dynamics in latent space, and (3) predict Y linearly.

The latent dynamics is modeled as a Koopman-like linear evolution:

    t_{t+1} ≈ F t_t,       for t = 1,…,n−1.

The regression in latent space is

    y_t ≈ Bᵀ t_t,          for t = 1,…,n,

or in matrix form

    Y ≈ T B.

A simple joint objective is

    L(W, F, B)
      = λ_dyn ∑_{t=1}^{n−1} ‖ t_{t+1} − F t_t ‖₂²
        + λ_reg ∑_{t=1}^{n}   ‖ y_t − Bᵀ t_t ‖₂²,

subject to PLS-style constraints on W and T (e.g. orthonormal scores and
components ordered by decreasing covariance with Y):

    (1/n) Tᵀ T = I_r,
    components sorted by cov(T_j, Y).

In practice, the objective is optimized by alternating updates:

1) Latent scores
   Given W, we form Z and T = Z W.

2) Dynamics update
   Given T, we estimate F as the least-squares solution of

       F = argmin_F ∑_{t=1}^{n−1} ‖ t_{t+1} − F t_t ‖₂²,

   which has the closed form

       F = (∑_{t} t_{t+1} t_tᵀ) (∑_{t} t_t t_tᵀ)^{-1}.

3) Regression update
   Given T, we estimate B as the least-squares solution of

       B = argmin_B ‖ Y − T B ‖_F²,

   giving

       B = (Tᵀ T)^{-1} Tᵀ Y.

4) PLS-like loading update
   W is initialized from a standard PLS on (Z, Y), and can optionally be
   refined by (approximate) gradient-based updates on L(W, F, B), while
   enforcing column normalization and PLS-style orthogonality constraints.

The fitted model predicts Y for new inputs X* by:

1) applying the same preprocessing and feature map ψ to obtain Z*,
2) computing scores T* = Z* W,
3) returning Ŷ* = T* B (with inverse scaling if standardization is used).

The term “online” refers to the fact that, for streaming data, F and B
can be updated recursively as new scores t_t arrive, while W is kept
fixed or updated more infrequently.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _check_jax_available():
    """Check if JAX is available for GPU acceleration."""
    try:
        import jax
        return True
    except ImportError:
        return False


# =============================================================================
# Featurizers
# =============================================================================

class IdentityFeaturizer(BaseEstimator, TransformerMixin):
    """Identity featurizer: ψ(x) = x.

    This is the default featurizer for OKLMPLS when no nonlinear
    transformation is needed.
    """

    def fit(self, X, y=None):
        """Fit the featurizer (no-op for identity).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : IdentityFeaturizer
        """
        return self

    def transform(self, X):
        """Transform X (identity).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Same as input.
        """
        return np.asarray(X)


class PolynomialFeaturizer(BaseEstimator, TransformerMixin):
    """Polynomial featurizer for OKLM-PLS.

    Creates polynomial features up to specified degree without interaction
    terms (for efficiency with high-dimensional spectral data).

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features.
    include_original : bool, default=True
        Whether to include the original features (degree 1).
    """

    def __init__(self, degree: int = 2, include_original: bool = True):
        self.degree = degree
        self.include_original = include_original

    def fit(self, X, y=None):
        """Fit the featurizer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : PolynomialFeaturizer
        """
        return self

    def transform(self, X):
        """Transform X to polynomial features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_features * degree)
            Polynomial features.
        """
        X = np.asarray(X)
        features = []

        if self.include_original:
            features.append(X)

        for d in range(2, self.degree + 1):
            features.append(X ** d)

        return np.hstack(features)


class RBFFeaturizer(BaseEstimator, TransformerMixin):
    """Random Fourier Features (RBF approximation) featurizer for OKLM-PLS.

    Approximates the RBF kernel using random Fourier features, which is
    useful for adding nonlinearity to the Koopman embedding.

    Parameters
    ----------
    n_components : int, default=100
        Number of random Fourier features.
    gamma : float, optional
        Kernel coefficient. If None, uses 1/n_features.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 100,
        gamma: float | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the featurizer by sampling random frequencies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self : RBFFeaturizer
        """
        X = np.asarray(X)
        n_features = X.shape[1]

        gamma = self.gamma if self.gamma is not None else 1.0 / n_features

        rng = np.random.default_rng(self.random_state)

        # Sample from Gaussian with std = sqrt(2 * gamma)
        self.random_weights_ = rng.normal(
            0, np.sqrt(2 * gamma), (n_features, self.n_components)
        )
        self.random_offset_ = rng.uniform(
            0, 2 * np.pi, self.n_components
        )

        return self

    def transform(self, X):
        """Transform X to random Fourier features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_rff : ndarray of shape (n_samples, n_components)
            Random Fourier features.
        """
        check_is_fitted(self, ['random_weights_', 'random_offset_'])

        X = np.asarray(X)
        projection = X @ self.random_weights_ + self.random_offset_

        return np.sqrt(2 / self.n_components) * np.cos(projection)


# =============================================================================
# NumPy Backend Implementation
# =============================================================================

def _oklmpls_fit_numpy(
    Z: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
    lambda_dyn: float,
    lambda_reg_y: float,
    max_iter: int,
    tol: float,
    W_init: NDArray[np.floating] | None = None,
    B_init: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.floating],  # W (projection weights)
    NDArray[np.floating],  # F (dynamics matrix)
    NDArray[np.floating],  # B (regression coefficients)
    int,                   # n_iter (actual iterations)
]:
    """Fit OKLM-PLS model using alternating optimization.

    Parameters
    ----------
    Z : ndarray of shape (n_samples, n_features_z)
        Featurized and centered X data.
    Y : ndarray of shape (n_samples, n_targets)
        Centered Y data.
    n_components : int
        Number of latent components.
    lambda_dyn : float
        Weight for dynamic consistency loss.
    lambda_reg_y : float
        Weight for regression loss.
    max_iter : int
        Maximum alternating optimization iterations.
    tol : float
        Convergence tolerance.
    W_init : ndarray, optional
        Initial W weights.
    B_init : ndarray, optional
        Initial B coefficients.

    Returns
    -------
    W : ndarray of shape (n_features_z, n_components)
        Learned projection weights.
    F : ndarray of shape (n_components, n_components)
        Learned dynamics matrix.
    B : ndarray of shape (n_components, n_targets)
        Learned regression coefficients.
    n_iter : int
        Number of iterations until convergence.
    """
    n_samples, d = Z.shape
    n_targets = Y.shape[1]
    r = n_components

    # Initialize W and B
    if W_init is not None:
        W = W_init.copy()
    else:
        # Random initialization with orthogonalization
        rng = np.random.default_rng(42)
        W = rng.normal(size=(d, r))
        W, _ = np.linalg.qr(W)

    if B_init is not None:
        B = B_init.copy()
    else:
        B = np.zeros((r, n_targets), dtype=np.float64)

    # Initialize F as identity
    F = np.eye(r, dtype=np.float64)

    prev_loss = np.inf

    for iteration in range(max_iter):
        # Step 1: Compute latent scores T = Z @ W
        T = Z @ W  # (n_samples, r)

        # Step 2: Update dynamics matrix F
        # Minimize ||T_{t+1} - F @ T_t||^2
        # Solution: F = (T_{t+1}^T @ T_t) @ (T_t^T @ T_t)^{-1}
        if lambda_dyn > 0 and n_samples > 1:
            T_t = T[:-1]  # (n_samples-1, r)
            T_next = T[1:]  # (n_samples-1, r)

            # F^T = (T_t^T @ T_t)^{-1} @ T_t^T @ T_next
            TtT = T_t.T @ T_t + 1e-8 * np.eye(r)
            TtN = T_t.T @ T_next
            F_T = np.linalg.solve(TtT, TtN)
            F = F_T.T

        # Step 3: Update regression coefficients B
        # Minimize ||Y - T @ B||^2
        # Solution: B = (T^T @ T)^{-1} @ T^T @ Y
        if lambda_reg_y > 0:
            TtT = T.T @ T + 1e-8 * np.eye(r)
            TtY = T.T @ Y
            B = np.linalg.solve(TtT, TtY)

        # Step 4: Update W using gradient descent
        # This is the key step that makes OKLM-PLS different from standard PLS
        # We minimize: lambda_dyn * ||T_{t+1} - F @ T_t||^2 + lambda_reg_y * ||Y - T @ B||^2
        # Gradient w.r.t. W is complex, so we use an approximate update

        # Compute gradient of combined objective
        T = Z @ W  # Recompute T with current W

        # Gradient from regression loss: -2 * Z^T @ (Y - T @ B) @ B^T
        residual_y = Y - T @ B
        grad_reg = -2 * Z.T @ residual_y @ B.T

        # Gradient from dynamics loss
        grad_dyn = np.zeros_like(W)
        if lambda_dyn > 0 and n_samples > 1:
            T_t = T[:-1]
            T_next = T[1:]
            residual_dyn = T_next - T_t @ F.T

            # Gradient contributions
            Z_t = Z[:-1]
            Z_next = Z[1:]

            # d/dW ||T_next - F @ T_t||^2
            # = d/dW ||Z_next @ W - F @ Z_t @ W||^2
            grad_dyn = (
                -2 * Z_next.T @ residual_dyn
                + 2 * Z_t.T @ (T_t @ F.T - T_next) @ F.T
            )

        # Combined gradient
        grad = lambda_reg_y * grad_reg + lambda_dyn * grad_dyn

        # Adaptive step size with line search
        step_size = 0.01 / (1 + 0.1 * iteration)

        # Update W with gradient step
        W_new = W - step_size * grad

        # Orthogonalize W to maintain well-conditioned projection
        W_new, _ = np.linalg.qr(W_new)
        W = W_new

        # Compute loss for convergence check
        T = Z @ W
        dyn_loss = 0.0
        if lambda_dyn > 0 and n_samples > 1:
            T_t = T[:-1]
            T_next = T[1:]
            dyn_loss = np.sum((T_next - T_t @ F.T) ** 2)

        reg_loss = 0.0
        if lambda_reg_y > 0:
            Y_hat = T @ B
            reg_loss = np.sum((Y - Y_hat) ** 2)

        loss = lambda_dyn * dyn_loss + lambda_reg_y * reg_loss

        if np.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return W, F, B, iteration + 1


def _oklmpls_predict_numpy(
    Z: NDArray[np.floating],
    W: NDArray[np.floating],
    B: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Predict using OKLM-PLS model.

    Parameters
    ----------
    Z : ndarray of shape (n_samples, n_features_z)
        Featurized and centered X data.
    W : ndarray of shape (n_features_z, n_components)
        Projection weights.
    B : ndarray of shape (n_components, n_targets)
        Regression coefficients.

    Returns
    -------
    Y_pred : ndarray of shape (n_samples, n_targets)
        Predicted values.
    """
    T = Z @ W
    return T @ B


# =============================================================================
# JAX Backend Implementation
# =============================================================================

def _get_jax_oklmpls_functions():
    """Lazy import and create JAX OKLM-PLS functions."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def compute_scores_jax(Z, W):
        """Compute latent scores T = Z @ W."""
        return Z @ W

    def update_dynamics_jax(T, r):
        """Update dynamics matrix F."""
        T_t = T[:-1]
        T_next = T[1:]

        TtT = T_t.T @ T_t + 1e-8 * jnp.eye(r)
        TtN = T_t.T @ T_next
        F_T = jnp.linalg.solve(TtT, TtN)
        return F_T.T

    def update_regression_jax(T, Y, r):
        """Update regression coefficients B."""
        TtT = T.T @ T + 1e-8 * jnp.eye(r)
        TtY = T.T @ Y
        return jnp.linalg.solve(TtT, TtY)

    @jax.jit
    def compute_loss_jax(Z, Y, W, F, B, lambda_dyn, lambda_reg_y):
        """Compute total loss."""
        T = Z @ W
        n_samples = T.shape[0]

        # Dynamic loss
        T_t = T[:-1]
        T_next = T[1:]
        dyn_loss = jnp.sum((T_next - T_t @ F.T) ** 2)

        # Regression loss
        reg_loss = jnp.sum((Y - T @ B) ** 2)

        return lambda_dyn * dyn_loss + lambda_reg_y * reg_loss

    @jax.jit
    def compute_gradient_jax(Z, Y, W, F, B, lambda_dyn, lambda_reg_y):
        """Compute gradient of loss w.r.t. W."""
        T = Z @ W

        # Gradient from regression
        residual_y = Y - T @ B
        grad_reg = -2 * Z.T @ residual_y @ B.T

        # Gradient from dynamics
        T_t = T[:-1]
        T_next = T[1:]
        residual_dyn = T_next - T_t @ F.T

        Z_t = Z[:-1]
        Z_next = Z[1:]

        grad_dyn = (
            -2 * Z_next.T @ residual_dyn
            + 2 * Z_t.T @ (T_t @ F.T - T_next) @ F.T
        )

        return lambda_reg_y * grad_reg + lambda_dyn * grad_dyn

    @jax.jit
    def predict_jax(Z, W, B):
        """Predict using fitted model."""
        T = Z @ W
        return T @ B

    return {
        'compute_scores': compute_scores_jax,
        'update_dynamics': update_dynamics_jax,
        'update_regression': update_regression_jax,
        'compute_loss': compute_loss_jax,
        'compute_gradient': compute_gradient_jax,
        'predict': predict_jax,
    }


_JAX_OKLMPLS_FUNCS = None


def _get_cached_jax_oklmpls():
    """Get cached JAX OKLM-PLS functions."""
    global _JAX_OKLMPLS_FUNCS
    if _JAX_OKLMPLS_FUNCS is None:
        _JAX_OKLMPLS_FUNCS = _get_jax_oklmpls_functions()
    return _JAX_OKLMPLS_FUNCS


def _oklmpls_fit_jax(
    Z: NDArray[np.floating],
    Y: NDArray[np.floating],
    n_components: int,
    lambda_dyn: float,
    lambda_reg_y: float,
    max_iter: int,
    tol: float,
    W_init: NDArray[np.floating] | None = None,
    B_init: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    int,
]:
    """Fit OKLM-PLS model using JAX backend."""
    import jax.numpy as jnp

    jax_funcs = _get_cached_jax_oklmpls()

    n_samples, d = Z.shape
    n_targets = Y.shape[1]
    r = n_components

    # Convert to JAX arrays
    Z_jax = jnp.asarray(Z)
    Y_jax = jnp.asarray(Y)

    # Initialize W and B
    if W_init is not None:
        W = jnp.asarray(W_init)
    else:
        rng = np.random.default_rng(42)
        W_np = rng.normal(size=(d, r))
        W_np, _ = np.linalg.qr(W_np)
        W = jnp.asarray(W_np)

    if B_init is not None:
        B = jnp.asarray(B_init)
    else:
        B = jnp.zeros((r, n_targets), dtype=jnp.float64)

    F = jnp.eye(r, dtype=jnp.float64)

    prev_loss = np.inf

    for iteration in range(max_iter):
        # Update F
        T = jax_funcs['compute_scores'](Z_jax, W)
        if lambda_dyn > 0 and n_samples > 1:
            F = jax_funcs['update_dynamics'](T, r)

        # Update B
        if lambda_reg_y > 0:
            B = jax_funcs['update_regression'](T, Y_jax, r)

        # Update W
        grad = jax_funcs['compute_gradient'](
            Z_jax, Y_jax, W, F, B, lambda_dyn, lambda_reg_y
        )

        step_size = 0.01 / (1 + 0.1 * iteration)
        W_new = W - step_size * grad

        # Orthogonalize
        W_np, _ = np.linalg.qr(np.asarray(W_new))
        W = jnp.asarray(W_np)

        # Check convergence
        loss = float(jax_funcs['compute_loss'](
            Z_jax, Y_jax, W, F, B, lambda_dyn, lambda_reg_y
        ))

        if np.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return np.asarray(W), np.asarray(F), np.asarray(B), iteration + 1


# =============================================================================
# OKLMPLS Estimator Class
# =============================================================================

class OKLMPLS(BaseEstimator, RegressorMixin):
    """Online Koopman Latent-Mode Partial Least Squares (OKLM-PLS).

    OKLM-PLS combines Koopman operator theory with PLS for time-series
    regression. It learns latent scores T = ψ(X) @ W and simultaneously:
    - Enforces dynamic coherence: T_{t+1} ≈ F @ T_t
    - Learns regression: Y_t ≈ T_t @ B

    This is useful for spectral data collected over time where temporal
    coherence provides additional predictive information.

    Parameters
    ----------
    n_components : int, default=5
        Number of latent components.
    featurizer : TransformerMixin, optional
        Feature map ψ: X -> Z. If None, identity is used.
        Options include PolynomialFeaturizer and RBFFeaturizer.
    lambda_dyn : float, default=1.0
        Weight for dynamic consistency loss ||T_{t+1} - F @ T_t||².
        Higher values enforce stronger temporal coherence.
    lambda_reg_y : float, default=1.0
        Weight for regression loss ||Y - T @ B||².
    max_iter : int, default=50
        Maximum alternating optimization iterations.
    tol : float, default=1e-4
        Convergence tolerance on the objective function.
    warm_start_pls : bool, default=True
        If True, initialize W/B from a standard PLSRegression fit.
    standardize : bool, default=True
        Whether to standardize X and Y before fitting.
    backend : str, default='numpy'
        Computational backend:
        - 'numpy': NumPy backend (CPU only).
        - 'jax': JAX backend (supports GPU/TPU).
    random_state : int, optional
        Random seed for initialization.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in input X.
    n_components_ : int
        Actual number of components.
    W_ : ndarray of shape (n_features_z, n_components_)
        Projection weights (in featurized space).
    F_ : ndarray of shape (n_components_, n_components_)
        Dynamics matrix for latent scores.
    B_ : ndarray of shape (n_components_, n_targets)
        Regression coefficients.
    n_iter_ : int
        Number of iterations until convergence.

    Examples
    --------
    >>> from nirs4all.operators.models.sklearn.oklmpls import OKLMPLS
    >>> import numpy as np
    >>> # Generate time-series data
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(100)
    >>> # Fit OKLM-PLS
    >>> model = OKLMPLS(n_components=10, lambda_dyn=1.0, lambda_reg_y=1.0)
    >>> model.fit(X, y)
    OKLMPLS(...)
    >>> predictions = model.predict(X)
    >>> # Use with polynomial featurizer for nonlinearity
    >>> from nirs4all.operators.models.sklearn.oklmpls import PolynomialFeaturizer
    >>> model_poly = OKLMPLS(n_components=10, featurizer=PolynomialFeaturizer(degree=2))
    >>> model_poly.fit(X, y)

    Notes
    -----
    OKLM-PLS is designed for temporally-ordered data where samples are
    sequential in time. The dynamics constraint helps capture temporal
    patterns and can improve prediction when the underlying process
    has smooth temporal evolution.

    For non-temporal data, set lambda_dyn=0 to disable the dynamics
    constraint (equivalent to standard PLS with optional featurization).

    See Also
    --------
    SIMPLS : Standard PLS without dynamics.
    RecursivePLS : Online PLS with forgetting factor.
    """

    # Explicitly declare estimator type for sklearn compatibility (e.g., StackingRegressor)
    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 5,
        featurizer: TransformerMixin | None = None,
        lambda_dyn: float = 1.0,
        lambda_reg_y: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        warm_start_pls: bool = True,
        standardize: bool = True,
        backend: str = 'numpy',
        random_state: int | None = None,
    ):
        """Initialize OKLM-PLS regressor."""
        self.n_components = n_components
        self.featurizer = featurizer
        self.lambda_dyn = lambda_dyn
        self.lambda_reg_y = lambda_reg_y
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start_pls = warm_start_pls
        self.standardize = standardize
        self.backend = backend
        self.random_state = random_state

    def _init_state(
        self,
        X: NDArray[np.floating],
        Y: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Initialize preprocessing and featurization.

        Returns
        -------
        Z : ndarray
            Featurized and processed X.
        X_proc : ndarray
            Processed X (before featurization).
        Y_proc : ndarray
            Processed Y.
        """
        # Standardization
        if self.standardize:
            self.x_scaler_ = StandardScaler(with_mean=True, with_std=True)
            self.y_scaler_ = StandardScaler(with_mean=True, with_std=True)
        else:
            self.x_scaler_ = None
            self.y_scaler_ = None

        X_proc = X.copy()
        Y_proc = Y.copy()

        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.fit_transform(X_proc)
        if self.y_scaler_ is not None:
            Y_proc = self.y_scaler_.fit_transform(Y_proc)

        # Featurizer
        if self.featurizer is None:
            self.featurizer_ = IdentityFeaturizer()
        else:
            self.featurizer_ = self.featurizer

        Z = self.featurizer_.fit_transform(X_proc)

        return Z, X_proc, Y_proc

    def _warm_start(
        self,
        Z: NDArray[np.floating],
        Y_proc: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Initialize W and B from PLSRegression.

        Returns
        -------
        W_init : ndarray of shape (n_features_z, n_components)
            Initial projection weights.
        B_init : ndarray of shape (n_components, n_targets)
            Initial regression coefficients.
        """
        r = min(self.n_components, Z.shape[0] - 1, Z.shape[1])

        pls = PLSRegression(n_components=r)
        pls.fit(Z, Y_proc)

        # W is the x_weights_ orthogonalized
        W_init = pls.x_weights_  # (d, r)

        # B in latent space: T @ B ≈ Y, where T = Z @ W
        T = pls.x_scores_
        TtT = T.T @ T + 1e-8 * np.eye(r)
        B_init = np.linalg.solve(TtT, T.T @ Y_proc)

        return W_init, B_init

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> "OKLMPLS":
        """Fit the OKLM-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Samples should be temporally ordered.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : OKLMPLS
            Fitted estimator.

        Raises
        ------
        ValueError
            If backend is invalid.
        ImportError
            If JAX backend requested but not installed.
        """
        if self.backend not in ('numpy', 'jax'):
            raise ValueError(
                f"backend must be 'numpy' or 'jax', got '{self.backend}'"
            )

        if self.backend == 'jax' and not _check_jax_available():
            raise ImportError(
                "JAX is required for OKLMPLS with backend='jax'. "
                "Install with: pip install jax"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features

        # Initialize preprocessing
        Z, X_proc, Y_proc = self._init_state(X, y)

        # Limit components
        n_features_z = Z.shape[1]
        max_components = min(n_samples - 1, n_features_z)
        self.n_components_ = min(self.n_components, max_components)

        # Warm start initialization
        W_init = None
        B_init = None
        if self.warm_start_pls:
            W_init, B_init = self._warm_start(Z, Y_proc)

        # Fit using appropriate backend
        if self.backend == 'jax':
            W, F, B, n_iter = _oklmpls_fit_jax(
                Z, Y_proc, self.n_components_,
                self.lambda_dyn, self.lambda_reg_y,
                self.max_iter, self.tol,
                W_init, B_init,
            )
        else:
            W, F, B, n_iter = _oklmpls_fit_numpy(
                Z, Y_proc, self.n_components_,
                self.lambda_dyn, self.lambda_reg_y,
                self.max_iter, self.tol,
                W_init, B_init,
            )

        self.W_ = W
        self.F_ = F
        self.B_ = B
        self.n_iter_ = n_iter

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict using the OKLM-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ['W_', 'B_', 'featurizer_'])

        X = np.asarray(X, dtype=np.float64)

        # Preprocess
        X_proc = X.copy()
        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.transform(X_proc)

        Z = self.featurizer_.transform(X_proc)

        # Predict
        if self.backend == 'jax':
            import jax.numpy as jnp
            jax_funcs = _get_cached_jax_oklmpls()
            Y_proc = np.asarray(jax_funcs['predict'](
                jnp.asarray(Z), jnp.asarray(self.W_), jnp.asarray(self.B_)
            ))
        else:
            Y_proc = _oklmpls_predict_numpy(Z, self.W_, self.B_)

        # Inverse transform
        if self.y_scaler_ is not None:
            Y = self.y_scaler_.inverse_transform(Y_proc)
        else:
            Y = Y_proc

        if self._y_1d:
            Y = Y.ravel()

        return Y

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        """Transform X to latent score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components_)
            Latent scores.
        """
        check_is_fitted(self, ['W_', 'featurizer_'])

        X = np.asarray(X, dtype=np.float64)

        X_proc = X.copy()
        if self.x_scaler_ is not None:
            X_proc = self.x_scaler_.transform(X_proc)

        Z = self.featurizer_.transform(X_proc)
        T = Z @ self.W_

        return T

    def predict_dynamic(
        self,
        X: ArrayLike,
        n_steps: int = 1,
    ) -> NDArray[np.floating]:
        """Predict using dynamics model for future timesteps.

        Given the last sample's latent scores, predict future values
        using the learned dynamics T_{t+1} = F @ T_t.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Current data. Uses last sample for propagation.
        n_steps : int, default=1
            Number of future timesteps to predict.

        Returns
        -------
        y_future : ndarray of shape (n_steps, n_targets)
            Predicted future values.
        """
        check_is_fitted(self, ['W_', 'F_', 'B_'])

        T = self.transform(X)
        T_current = T[-1:]  # Last timestep

        predictions = []
        for _ in range(n_steps):
            T_next = T_current @ self.F_.T
            y_next = T_next @ self.B_

            if self.y_scaler_ is not None:
                y_next = self.y_scaler_.inverse_transform(y_next)

            predictions.append(y_next.flatten())
            T_current = T_next

        result = np.array(predictions)
        # If univariate y and output is (n_steps, 1), squeeze to (n_steps,)
        if self._y_1d and result.ndim == 2 and result.shape[1] == 1:
            result = result.ravel()
        return result

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'featurizer': self.featurizer,
            'lambda_dyn': self.lambda_dyn,
            'lambda_reg_y': self.lambda_reg_y,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'warm_start_pls': self.warm_start_pls,
            'standardize': self.standardize,
            'backend': self.backend,
            'random_state': self.random_state,
        }

    def set_params(self, **params) -> "OKLMPLS":
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OKLMPLS(n_components={self.n_components}, "
            f"lambda_dyn={self.lambda_dyn}, "
            f"lambda_reg_y={self.lambda_reg_y}, "
            f"max_iter={self.max_iter}, "
            f"backend='{self.backend}')"
        )

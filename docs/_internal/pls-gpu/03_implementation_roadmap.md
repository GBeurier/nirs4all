# pls-gpu: Implementation Roadmap

**Version:** 1.0.0-draft
**Date:** 2024-12-31
**Status:** Specification Draft

## 1. Project Timeline Overview

The project is divided into 5 phases over approximately 6 months:

> **Note on timeline**: This roadmap assumes a small team (2-3 developers) working part-time,
> or 1 full-time developer with prior experience in PLS implementations and multi-backend
> development. For a single part-time contributor, multiply time estimates by 2-3x.
> Phase 1 and 2 are critical and should not be compressed.
>
> **Descoping Strategy**: If deadlines slip, prioritize in this order:
> 1. **Must-have (v1.0)**: PLSRegression (NIPALS, SIMPLS, IKPLS), OPLS, NumPy + JAX backends
> 2. **Should-have (v1.0)**: Kernel PLS, PLS-DA, OPLS-DA, PyTorch backend
> 3. **Nice-to-have (v1.1)**: Sparse PLS, LW-PLS, MB-PLS, TensorFlow backend
> 4. **Future (v1.2+)**: Recursive PLS, Robust PLS, DiPLS, O2PLS, variable selection wrappers
>
> TensorFlow backend is explicitly lower priority—most GPU users prefer JAX or PyTorch.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Foundation (Weeks 1-4)                                            │
│ - Project setup, CI/CD, core architecture                                  │
│ - Backend abstraction layer                                                │
│ - NIPALS and SIMPLS core algorithms                                        │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Core Methods (Weeks 5-10)                                         │
│ - PLSRegression (all algorithms)                                           │
│ - OPLS, Kernel PLS                                                         │
│ - Numerical validation vs sklearn/R                                        │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Extended Methods (Weeks 11-16)                                    │
│ - MB-PLS, LW-PLS, Sparse PLS                                              │
│ - K-OPLS, Recursive PLS, Robust PLS                                       │
│ - Variable selection wrappers                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Benchmarks & Validation (Weeks 17-20)                            │
│ - Full benchmark suite                                                     │
│ - GPU performance optimization                                             │
│ - Cross-platform testing                                                   │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Documentation & Publication (Weeks 21-24)                        │
│ - Complete documentation                                                   │
│ - RTD deployment                                                           │
│ - Paper writing and submission                                             │
│ - PyPI release                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## 2. Phase 1: Foundation (Weeks 1-4)

### 2.1 Week 1: Project Setup

#### 2.1.1 Repository Structure

```
pls-gpu/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Main CI pipeline
│   │   ├── benchmarks.yml      # Benchmark automation
│   │   └── release.yml         # PyPI release
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── getting_started/
│   │   ├── user_guide/
│   │   ├── api/
│   │   └── mathematical_reference/
│   └── requirements.txt
├── pls_gpu/
│   ├── __init__.py
│   ├── _version.py
│   ├── core/
│   ├── backends/
│   └── ...
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── examples/                   # User-facing examples (separate from publication)
│   ├── getting_started/        # Basic usage examples
│   ├── sklearn_integration/    # sklearn Pipeline, GridSearchCV
│   ├── nirs4all_integration/   # nirs4all pipeline examples
│   ├── backend_specific/       # JAX, PyTorch, TensorFlow examples
│   └── notebooks/              # Jupyter tutorials
├── publication/                # Publication-specific (reproducible benchmarks)
│   ├── paper/                  # Manuscript, figures, tables
│   ├── benchmarks/             # All benchmark scripts
│   ├── datasets/               # Dataset download/prep scripts
│   └── results/                # Generated results (git-ignored)
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

#### 2.1.2 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1.1 | Create GitHub repository | Repository with branch protection |
| 1.1.2 | Configure pyproject.toml | Build configuration, dependencies |
| 1.1.3 | Setup pre-commit hooks | Linting (ruff), formatting (black) |
| 1.1.4 | Configure pytest | Test infrastructure |
| 1.1.5 | Setup GitHub Actions CI | Basic CI pipeline |
| 1.1.6 | Configure Sphinx docs | Documentation skeleton |
| 1.1.7 | Setup ReadTheDocs | RTD integration |

#### 2.1.3 pyproject.toml Configuration

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pls-gpu"
dynamic = ["version"]
description = "GPU-accelerated Partial Least Squares implementations"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "pls-gpu developers"}
]
keywords = ["PLS", "partial least squares", "chemometrics", "GPU", "machine learning"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
]

dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
jax = ["jax>=0.4.20", "jaxlib>=0.4.20"]
jax-cuda = ["jax[cuda12]>=0.4.20"]
torch = ["torch>=2.0.0"]
tensorflow = ["tensorflow>=2.15.0"]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "ruff>=0.1.0",
    "black>=23.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "numpydoc>=1.5.0",
    "sphinx-copybutton>=0.5.0",
    "myst-parser>=2.0.0",
]
all = ["pls-gpu[jax,torch,tensorflow,dev,docs]"]

[project.urls]
Homepage = "https://github.com/yourusername/pls-gpu"
Documentation = "https://pls-gpu.readthedocs.io"
Repository = "https://github.com/yourusername/pls-gpu"

[tool.setuptools_scm]
write_to = "pls_gpu/_version.py"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "gpu: marks tests requiring GPU",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]
```

### 2.2 Week 2: Backend Abstraction Layer

#### 2.2.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.2.1 | Design backend interface | Abstract backend protocol |
| 1.2.2 | Implement NumPy backend | `backends/numpy/` module |
| 1.2.3 | Implement JAX backend | `backends/jax/` module |
| 1.2.4 | Implement PyTorch backend | `backends/torch/` module |
| 1.2.5 | Implement TensorFlow backend | `backends/tensorflow/` module |
| 1.2.6 | Backend registry and selection | `backends/__init__.py` |
| 1.2.7 | Unit tests for backends | Backend test suite |

#### 2.2.2 Backend Interface Design

```python
# pls_gpu/backends/protocol.py

from typing import Protocol, Tuple, Any
import numpy as np

ArrayType = Any  # Backend-specific array type

class BackendProtocol(Protocol):
    """Protocol defining the backend interface."""

    # Array creation
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> ArrayType: ...
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> ArrayType: ...
    def eye(self, n: int, dtype: Any = None) -> ArrayType: ...
    def asarray(self, x: Any, dtype: Any = None) -> ArrayType: ...

    # Linear algebra
    def dot(self, a: ArrayType, b: ArrayType) -> ArrayType: ...
    def matmul(self, a: ArrayType, b: ArrayType) -> ArrayType: ...
    def svd(self, a: ArrayType, full_matrices: bool = True) -> Tuple[ArrayType, ArrayType, ArrayType]: ...
    def pinv(self, a: ArrayType) -> ArrayType: ...
    def norm(self, x: ArrayType, axis: int = None) -> ArrayType: ...
    def eigh(self, a: ArrayType) -> Tuple[ArrayType, ArrayType]: ...

    # Reductions
    def sum(self, x: ArrayType, axis: int = None) -> ArrayType: ...
    def mean(self, x: ArrayType, axis: int = None) -> ArrayType: ...
    def std(self, x: ArrayType, axis: int = None, ddof: int = 0) -> ArrayType: ...

    # Element-wise operations
    def sqrt(self, x: ArrayType) -> ArrayType: ...
    def abs(self, x: ArrayType) -> ArrayType: ...
    def sign(self, x: ArrayType) -> ArrayType: ...
    def maximum(self, x: ArrayType, y: ArrayType) -> ArrayType: ...
    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType: ...

    # Shapes and indexing
    def reshape(self, x: ArrayType, shape: Tuple[int, ...]) -> ArrayType: ...
    def transpose(self, x: ArrayType) -> ArrayType: ...
    def concatenate(self, arrays: list, axis: int = 0) -> ArrayType: ...
    def outer(self, a: ArrayType, b: ArrayType) -> ArrayType: ...

    # Conversion
    def to_numpy(self, x: ArrayType) -> np.ndarray: ...
    def from_numpy(self, x: np.ndarray) -> ArrayType: ...

    # Properties
    @property
    def float64(self) -> Any: ...
    @property
    def name(self) -> str: ...
```

### 2.3 Week 3: Core Algorithms - NIPALS

#### 2.3.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.3.1 | Implement NIPALS algorithm | `core/nipals.py` |
| 1.3.2 | NumPy implementation | Validated NumPy NIPALS |
| 1.3.3 | JAX JIT implementation | Optimized JAX NIPALS |
| 1.3.4 | PyTorch implementation | PyTorch NIPALS |
| 1.3.5 | TensorFlow implementation | TensorFlow NIPALS |
| 1.3.6 | Unit tests | NIPALS test suite |
| 1.3.7 | Validation vs sklearn | Equivalence tests |

#### 2.3.2 NIPALS Algorithm Implementation

```python
# pls_gpu/core/nipals.py

"""
NIPALS (Nonlinear Iterative Partial Least Squares) algorithm.

The core iterative algorithm for extracting PLS components.

References
----------
- Wold, H. (1966). Estimation of principal components and related models
  by iterative least squares.
- Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic
  tool of chemometrics.
"""

from typing import Tuple, Any
from ..backends import get_backend

def nipals_component(
    X: Any,
    Y: Any,
    max_iter: int = 500,
    tol: float = 1e-6,
    backend: str = "numpy"
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Extract one PLS component using NIPALS.

    Algorithm:
    1. Initialize u = first column of Y
    2. Repeat until convergence:
       a. w = X'u / (u'u)
       b. w = w / ||w||
       c. t = Xw
       d. q = Y't / (t't)
       e. u_new = Yq
       f. Check convergence: ||u_new - u|| < tol
    3. p = X't / (t't)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Centered/scaled X matrix.
    Y : array-like of shape (n_samples, n_targets)
        Centered/scaled Y matrix.
    max_iter : int, default=500
        Maximum iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    backend : str, default="numpy"
        Computational backend.

    Returns
    -------
    w : array of shape (n_features,)
        X weight vector.
    t : array of shape (n_samples,)
        X score vector.
    p : array of shape (n_features,)
        X loading vector.
    q : array of shape (n_targets,)
        Y loading vector.
    u : array of shape (n_samples,)
        Y score vector.
    """
    be = get_backend(backend)

    # Initialize u with first column of Y
    u = Y[:, 0].copy()

    for iteration in range(max_iter):
        u_old = u.copy()

        # w = X'u / (u'u), then normalize
        uu = be.dot(u, u)
        w = be.dot(X.T, u) / (uu + 1e-10)
        w_norm = be.norm(w)
        w = w / (w_norm + 1e-10)

        # t = Xw
        t = be.dot(X, w)

        # q = Y't / (t't)
        tt = be.dot(t, t)
        q = be.dot(Y.T, t) / (tt + 1e-10)

        # u = Yq (for next iteration or final u)
        u = be.dot(Y, q)

        # Check convergence
        diff = be.norm(u - u_old)
        if diff < tol:
            break

    # p = X't / (t't)
    p = be.dot(X.T, t) / (tt + 1e-10)

    return w, t, p, q, u


def nipals_fit(
    X: Any,
    Y: Any,
    n_components: int,
    max_iter: int = 500,
    tol: float = 1e-6,
    backend: str = "numpy"
) -> dict:
    """
    Fit PLS model using NIPALS algorithm.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Centered/scaled X matrix.
    Y : array-like of shape (n_samples, n_targets)
        Centered/scaled Y matrix.
    n_components : int
        Number of components to extract.
    max_iter : int, default=500
        Maximum iterations per component.
    tol : float, default=1e-6
        Convergence tolerance.
    backend : str, default="numpy"
        Computational backend.

    Returns
    -------
    result : dict
        Dictionary containing W, T, P, Q, U matrices and regression coefficients.
    """
    be = get_backend(backend)

    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # Initialize storage
    W = be.zeros((n_features, n_components))
    T = be.zeros((n_samples, n_components))
    P = be.zeros((n_features, n_components))
    Q = be.zeros((n_targets, n_components))
    U = be.zeros((n_samples, n_components))

    # Working copies for deflation
    X_res = X.copy()
    Y_res = Y.copy()

    for i in range(n_components):
        # Extract component
        w, t, p, q, u = nipals_component(
            X_res, Y_res, max_iter, tol, backend
        )

        # Store
        W[:, i] = w
        T[:, i] = t
        P[:, i] = p
        Q[:, i] = q
        U[:, i] = u

        # Deflate X and Y
        X_res = X_res - be.outer(t, p)
        Y_res = Y_res - be.outer(t, q)

    # Compute regression coefficients: B = W @ inv(P'W) @ Q'
    PtW = be.dot(P.T, W)
    PtW_inv = be.pinv(PtW)
    B = be.dot(be.dot(W, PtW_inv), Q.T)

    return {
        'W': W,
        'T': T,
        'P': P,
        'Q': Q,
        'U': U,
        'B': B,
        'n_components': n_components
    }
```

### 2.4 Week 4: Core Algorithms - SIMPLS

#### 2.4.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.4.1 | Implement SIMPLS algorithm | `core/simpls.py` |
| 1.4.2 | NumPy implementation | Validated NumPy SIMPLS |
| 1.4.3 | JAX JIT implementation | Optimized JAX SIMPLS |
| 1.4.4 | PyTorch/TF implementations | Backend parity |
| 1.4.5 | Unit tests | SIMPLS test suite |
| 1.4.6 | Validation vs R pls | Equivalence tests |
| 1.4.7 | Phase 1 integration tests | End-to-end tests |

## 3. Phase 2: Core Methods (Weeks 5-10)

### 3.1 Week 5-6: PLSRegression Class

#### 3.1.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1.1 | Implement PLSRegression | Main estimator class |
| 2.1.2 | Add IKPLS algorithms | ikpls1, ikpls2 algorithms |
| 2.1.3 | sklearn compatibility | Pass sklearn tests |
| 2.1.4 | Complete API | fit, predict, transform, score |
| 2.1.5 | Documentation | Docstrings, examples |
| 2.1.6 | Unit tests | Comprehensive test suite |

#### 3.1.2 PLSRegression Implementation

```python
# pls_gpu/regression/pls.py

"""
Partial Least Squares Regression.
"""

from typing import Literal, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..core import nipals_fit, simpls_fit, ikpls_fit
from ..backends import get_backend

BackendType = Literal["numpy", "jax", "torch", "tensorflow"]
AlgorithmType = Literal["nipals", "simpls", "ikpls1", "ikpls2"]

class PLSRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    """Partial Least Squares Regression.

    Multiple algorithm implementations with GPU acceleration.

    Parameters
    ----------
    n_components : int, default=10
        Number of components.
    algorithm : {'nipals', 'simpls', 'ikpls1', 'ikpls2'}, default='simpls'
        Algorithm to use.
    scale : bool, default=True
        Scale to unit variance.
    center : bool, default=True
        Center the data.
    max_iter : int, default=500
        Max iterations (NIPALS).
    tol : float, default=1e-6
        Convergence tolerance (NIPALS).
    backend : {'numpy', 'jax', 'torch', 'tensorflow'}, default='numpy'
        Computational backend.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    n_components_ : int
        Actual number of components.
    x_weights_ : ndarray of shape (n_features, n_components)
        X weights (W).
    x_loadings_ : ndarray of shape (n_features, n_components)
        X loadings (P).
    y_loadings_ : ndarray of shape (n_targets, n_components)
        Y loadings (Q).
    x_scores_ : ndarray of shape (n_samples, n_components)
        X scores (T).
    y_scores_ : ndarray of shape (n_samples, n_components)
        Y scores (U).
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.
    x_mean_, y_mean_ : ndarray
        Means.
    x_std_, y_std_ : ndarray
        Standard deviations.

    Examples
    --------
    >>> from pls_gpu import PLSRegression
    >>> import numpy as np
    >>> X = np.random.randn(100, 50)
    >>> y = np.random.randn(100)
    >>> pls = PLSRegression(n_components=10, backend='jax')
    >>> pls.fit(X, y)
    >>> y_pred = pls.predict(X)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        algorithm: AlgorithmType = "simpls",
        scale: bool = True,
        center: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        backend: BackendType = "numpy",
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.scale = scale
        self.center = center
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend

    def fit(self, X, y):
        """Fit the PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : PLSRegression
            Fitted estimator.
        """
        be = get_backend(self.backend)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Handle 1D y
        self._y_1d = y.ndim == 1
        if self._y_1d:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        self.n_features_in_ = n_features

        # Limit components
        max_components = min(n_samples - 1, n_features)
        self.n_components_ = min(self.n_components, max_components)

        # Center and scale
        if self.center:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = y.mean(axis=0)
        else:
            self.x_mean_ = np.zeros(n_features)
            self.y_mean_ = np.zeros(n_targets)

        if self.scale:
            self.x_std_ = X.std(axis=0, ddof=1)
            self.y_std_ = y.std(axis=0, ddof=1)
            self.x_std_ = np.where(self.x_std_ < 1e-10, 1.0, self.x_std_)
            self.y_std_ = np.where(self.y_std_ < 1e-10, 1.0, self.y_std_)
        else:
            self.x_std_ = np.ones(n_features)
            self.y_std_ = np.ones(n_targets)

        X_scaled = (X - self.x_mean_) / self.x_std_
        y_scaled = (y - self.y_mean_) / self.y_std_

        # Select algorithm
        if self.algorithm == "nipals":
            result = nipals_fit(
                X_scaled, y_scaled, self.n_components_,
                self.max_iter, self.tol, self.backend
            )
        elif self.algorithm == "simpls":
            result = simpls_fit(
                X_scaled, y_scaled, self.n_components_,
                self.backend
            )
        elif self.algorithm in ("ikpls1", "ikpls2"):
            alg_num = 1 if self.algorithm == "ikpls1" else 2
            result = ikpls_fit(
                X_scaled, y_scaled, self.n_components_,
                algorithm=alg_num, backend=self.backend
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Store results
        self.x_weights_ = np.asarray(result['W'])
        self.x_loadings_ = np.asarray(result['P'])
        self.y_loadings_ = np.asarray(result['Q'])
        self.x_scores_ = np.asarray(result['T'])
        self.y_scores_ = np.asarray(result.get('U', result['T']))

        # Compute coefficients in original space
        B_scaled = np.asarray(result['B'])
        self.coef_ = B_scaled * self.y_std_[np.newaxis, :] / self.x_std_[:, np.newaxis]

        return self

    def predict(self, X, n_components: Optional[int] = None):
        """Predict using the PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        n_components : int, optional
            Number of components to use. Default: all.

        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        check_is_fitted(self)

        X = np.asarray(X, dtype=np.float64)

        if n_components is None:
            n_components = self.n_components_
        else:
            n_components = min(n_components, self.n_components_)

        # Scale and predict
        X_scaled = (X - self.x_mean_) / self.x_std_

        # Use appropriate coefficients for n_components
        # (For simplicity, using full coefficients here)
        y_pred = X_scaled @ self.coef_ * self.y_std_ + self.y_mean_

        if self._y_1d:
            y_pred = y_pred.ravel()

        return y_pred

    def transform(self, X):
        """Transform X to score space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        T : ndarray of shape (n_samples, n_components_)
            Scores.
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.x_mean_) / self.x_std_

        # T = X @ W @ inv(P'W)
        PtW = self.x_loadings_.T @ self.x_weights_
        R = self.x_weights_ @ np.linalg.pinv(PtW)
        return X_scaled @ R

    def fit_transform(self, X, y):
        """Fit and transform X."""
        return self.fit(X, y).transform(X)

    def score(self, X, y):
        """Return R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    def get_params(self, deep=True):
        """Get parameters."""
        return {
            'n_components': self.n_components,
            'algorithm': self.algorithm,
            'scale': self.scale,
            'center': self.center,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'backend': self.backend,
        }

    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

### 3.2 Week 7-8: OPLS and Kernel PLS

#### 3.2.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.2.1 | Implement OPLS | OPLS class |
| 2.2.2 | Implement Kernel PLS | KernelPLS class |
| 2.2.3 | Kernel functions | RBF, polynomial, sigmoid |
| 2.2.4 | GPU kernel optimization | JIT-compiled kernels |
| 2.2.5 | Validation vs pyopls | Equivalence tests |
| 2.2.6 | Documentation | Complete docstrings |

### 3.3 Week 9-10: Classification and Validation

#### 3.3.1 Tasks

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.3.1 | Implement PLS-DA | PLSDA class |
| 2.3.2 | Implement OPLS-DA | OPLSDA class |
| 2.3.3 | Full sklearn equivalence | Pass all sklearn tests |
| 2.3.4 | R package equivalence | Validated vs R pls |
| 2.3.5 | Phase 2 integration | End-to-end tests |
| 2.3.6 | Documentation update | User guide sections |

## 4. Phase 3: Extended Methods (Weeks 11-16)

### 4.1 Weeks 11-12: Multiblock and Local Methods

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1.1 | Implement MB-PLS | MBPLS class |
| 3.1.2 | Implement LW-PLS | LWPLS class |
| 3.1.3 | Implement SO-PLS | SOPLS class |
| 3.1.4 | Implement O2PLS | O2PLS class (two-way orthogonal PLS) |
| 3.1.5 | GPU optimization | Batched predictions |
| 3.1.6 | Validation | Equivalence tests |

### 4.2 Weeks 13-14: Regularized Methods

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.2.1 | Implement Sparse PLS | SparsePLS class |
| 3.2.2 | Implement Robust PLS | RobustPLS class |
| 3.2.3 | Implement Recursive PLS | RecursivePLS class |
| 3.2.4 | Implement Dynamic PLS | DiPLS class |
| 3.2.5 | Cross-validation support | Fast CV utilities |

### 4.3 Weeks 15-16: Variable Selection and K-OPLS

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.3.1 | Implement K-OPLS | KOPLS class |
| 3.3.2 | Implement iPLS | IntervalPLS class |
| 3.3.3 | Implement VIP | VIP selector |
| 3.3.4 | Implement MC-UVE | MCUVE selector |
| 3.3.5 | Phase 3 integration | Full test suite |

## 5. Phase 4: Benchmarks & Validation (Weeks 17-20)

### 5.1 Week 17: Benchmark Infrastructure

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1.1 | Setup benchmark suite | bench/ structure |
| 4.1.2 | Reference data generation | Test datasets |
| 4.1.3 | Comparison scripts | sklearn/R/MATLAB comparison |
| 4.1.4 | Performance benchmarks | Backend comparison |

### 5.2 Week 18: GPU Optimization

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.2.1 | JAX optimization | Memory-efficient JIT |
| 4.2.2 | PyTorch optimization | CUDA optimization |
| 4.2.3 | Batch processing | Large dataset support |
| 4.2.4 | Memory profiling | Memory optimization |

### 5.3 Week 19-20: Comprehensive Testing

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.3.1 | Cross-platform testing | Linux/macOS/Windows |
| 4.3.2 | GPU testing | CUDA/MPS testing |
| 4.3.3 | Real dataset benchmarks | Method comparison |
| 4.3.4 | Performance reports | Benchmark reports |

## 6. Phase 5: Documentation & Publication (Weeks 21-24)

### 6.1 Week 21-22: Documentation

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1.1 | Complete API docs | All docstrings |
| 5.1.2 | User guide | Getting started, tutorials |
| 5.1.3 | Mathematical reference | Algorithm descriptions |
| 5.1.4 | Examples | Jupyter notebooks |
| 5.1.5 | RTD deployment | Live documentation |

### 6.2 Week 23-24: Publication and Release

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.2.1 | Paper writing | Draft manuscript |
| 5.2.2 | Figure generation | Publication figures |
| 5.2.3 | PyPI release | pip install pls-gpu |
| 5.2.4 | GitHub release | v1.0.0 release |
| 5.2.5 | nirs4all integration | Compatibility layer |
| 5.2.6 | Paper submission | Journal submission |

## 7. Verification Checkpoints

### 7.1 Phase 1 Checkpoint

- [ ] Repository setup complete
- [ ] CI/CD pipeline working
- [ ] All 4 backends implemented
- [ ] NIPALS validated vs sklearn (rtol < 1e-5)
- [ ] SIMPLS validated vs R pls (rtol < 1e-5)
- [ ] Test coverage > 80%

### 7.2 Phase 2 Checkpoint

- [ ] PLSRegression fully functional
- [ ] OPLS validated vs pyopls
- [ ] KernelPLS validated
- [ ] PLS-DA and OPLS-DA working
- [ ] Full sklearn compatibility
- [ ] Documentation started

### 7.3 Phase 3 Checkpoint

- [ ] All extended methods implemented
- [ ] GPU optimization complete
- [ ] Variable selection wrappers working
- [ ] Validation complete for all methods
- [ ] Test coverage > 90%

### 7.4 Phase 4 Checkpoint

- [ ] Benchmark suite complete
- [ ] All equivalence tests passing
- [ ] GPU speedups documented
- [ ] Real dataset benchmarks complete (with optimal component selection)
- [ ] Cross-platform testing complete
- [ ] Classification validation (PLS-DA, OPLS-DA) with proper metrics
- [ ] All results report Q², RPD, RMSE at minimum

### 7.5 Phase 5 Checkpoint

- [ ] Documentation complete
- [ ] RTD live and working
- [ ] PyPI release successful
- [ ] Paper draft complete
- [ ] nirs4all integration tested
- [ ] Release v1.0.0

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Mitigation |
|------|-----------|
| Numerical instability | Extensive validation, use float64 |
| GPU memory limits | Chunked processing, configurable batch sizes |
| Backend incompatibility | Abstract interface, thorough testing |
| Performance regression | Continuous benchmarking in CI |

### 8.2 Schedule Risks

| Risk | Mitigation |
|------|-----------|
| Scope creep | Prioritized features, MVP first |
| Validation failures | Early validation, incremental testing |
| Documentation delays | Documentation-first approach |
| Publication delays | Start writing early, parallel work |

## 9. Dependencies

### 9.1 External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24 | Core array operations |
| scipy | >=1.10 | Linear algebra fallbacks |
| scikit-learn | >=1.3 | API compatibility, validation |
| jax | >=0.4.20 | GPU backend (optional) |
| torch | >=2.0 | GPU backend (optional) |
| tensorflow | >=2.15 | GPU backend (optional) |

### 9.2 Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Testing |
| ruff | Linting |
| black | Formatting |
| sphinx | Documentation |
| pytest-benchmark | Performance testing |

## 10. Success Metrics

### 10.1 Technical Metrics

- Numerical equivalence: max_diff < 1e-5 vs reference implementations
- GPU speedup: >= 5x for n > 5000 samples
- Test coverage: >= 90%
- Documentation coverage: 100% public API

### 10.2 Adoption Metrics

- PyPI downloads: Target 1000+ in first month
- GitHub stars: Target 50+ in first quarter
- nirs4all integration: Seamless replacement

### 10.3 Publication Metrics

- Benchmark coverage: 30+ real datasets
- Method coverage: 15+ PLS variants
- Backend coverage: 4 backends

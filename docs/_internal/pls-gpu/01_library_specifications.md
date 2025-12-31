# pls-gpu: Library Specifications

**Version:** 1.0.0-draft
**Date:** 2024-12-31
**Status:** Specification Draft

## 1. Executive Summary

**pls-gpu** is a Python library providing GPU-accelerated implementations of Partial Least Squares (PLS) methods and related dimensionality reduction techniques. The library offers:

- **Unified backend API**: Switch between NumPy, JAX, PyTorch, and TensorFlow with a single `backend="jax"` parameter
- **Complete PLS coverage**: All major PLS variants published in chemometrics literature
- **sklearn compatibility**: Full compatibility with scikit-learn's estimator API
- **No external dependencies**: Pure implementations requiring only backend libraries (NumPy/JAX/PyTorch/TensorFlow)
- **Hierarchical architecture**: Core algorithms (NIPALS, SIMPLS, etc.) separated from extensions
- **nirs4all integration**: Drop-in replacement for nirs4all's current operators

## 2. Rationale

### 2.1 Why a Dedicated PLS Library?

1. **Fragmented ecosystem**: PLS implementations are scattered across multiple packages with inconsistent APIs (`sklearn`, `ikpls`, `pyopls`, `mbpls`, `sparse-pls`, etc.)
2. **Missing GPU support**: Most implementations are CPU-only; `ikpls` provides JAX but limited to IKPLS algorithm
3. **Incomplete coverage**: No single library covers all PLS variants used in chemometrics
4. **External dependencies**: Existing wrappers depend on third-party packages that may have different licenses, maintenance states, or APIs
5. **Research reproducibility**: Need for consistent implementations across R/MATLAB/Python with verified numerical equivalence

### 2.2 Target Use Cases

| Use Case | Requirements |
|----------|-------------|
| NIRS calibration | PLS1/PLS2, OPLS, iPLS, variable selection |
| High-throughput screening | GPU acceleration, batch processing |
| Online/adaptive models | Recursive PLS, LW-PLS |
| Multi-sensor fusion | MB-PLS, SO-PLS |
| Interpretable models | Sparse PLS, VIP, OPLS-DA |
| Research/Publication | Reproducible, verified against reference implementations |

## 3. Architecture

### 3.1 Package Structure

```
pls_gpu/
├── __init__.py                 # Main exports (PLSRegressor, PLSClassifier)
├── _version.py                 # Version info
├── unified.py                  # PLSRegressor, PLSClassifier unified API
├── core/                       # Core algorithms (backend-agnostic)
│   ├── __init__.py
│   ├── base.py                 # BasePLS abstract class
│   ├── nipals.py               # NIPALS algorithm (core)
│   ├── simpls.py               # SIMPLS algorithm (core)
│   ├── kernel_pls.py           # Kernel PLS algorithm (core)
│   └── utils.py                # Shared utilities (centering, SVD, etc.)
├── backends/                   # Backend implementations
│   ├── __init__.py
│   ├── numpy/                  # NumPy implementations
│   │   ├── __init__.py
│   │   ├── linalg.py           # Linear algebra operations
│   │   └── kernels.py          # Kernel functions
│   ├── jax/                    # JAX implementations
│   │   ├── __init__.py
│   │   ├── linalg.py
│   │   └── kernels.py
│   ├── torch/                  # PyTorch implementations
│   │   ├── __init__.py
│   │   ├── linalg.py
│   │   └── kernels.py
│   └── tensorflow/             # TensorFlow implementations
│       ├── __init__.py
│       ├── linalg.py
│       └── kernels.py
├── regression/                 # Regression models
│   ├── __init__.py
│   ├── pls.py                  # PLSRegression (NIPALS/SIMPLS/IKPLS)
│   ├── opls.py                 # OPLS
│   ├── kopls.py                # Kernel OPLS
│   ├── mbpls.py                # Multiblock PLS
│   ├── sopls.py                # Sequential/Parallel Orthogonalized PLS
│   ├── lwpls.py                # Locally-Weighted PLS
│   ├── sparse_pls.py           # Sparse PLS
│   ├── recursive_pls.py        # Recursive/Online PLS
│   ├── robust_pls.py           # Robust PLS (RSIMPLS)
│   └── dynamic_pls.py          # Dynamic PLS (DiPLS)
├── classification/             # Classification models
│   ├── __init__.py
│   ├── plsda.py                # PLS-DA
│   └── oplsda.py               # OPLS-DA
├── selection/                  # Variable selection wrappers
│   ├── __init__.py
│   ├── ipls.py                 # Interval PLS
│   ├── vip.py                  # Variable Importance in Projection
│   ├── mcuve.py                # MC-UVE
│   ├── cars.py                 # CARS
│   └── spa.py                  # SPA (Successive Projections Algorithm)
├── kernel/                     # Kernel PLS variants
│   ├── __init__.py
│   ├── kpls.py                 # Kernel PLS
│   └── nlpls.py                # Nonlinear PLS
├── multiway/                   # Tensor/multiway methods
│   ├── __init__.py
│   ├── npls.py                 # N-way PLS
│   └── hopls.py                # Higher-Order PLS (future)
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── validation.py           # Input validation
│   ├── metrics.py              # PLS-specific metrics (VIP, Q2, etc.)
│   └── cross_validation.py     # Fast CV utilities
├── compat/                     # Compatibility layers
│   ├── __init__.py
│   ├── sklearn.py              # sklearn wrappers
│   └── nirs4all.py             # nirs4all integration
├── _testing/                   # Testing utilities
│   ├── __init__.py
│   ├── fixtures.py             # Test data generators
│   └── reference.py            # Reference implementation results
└── _licenses/                  # Third-party license files
    ├── SKLEARN_BSD3.txt
    ├── IKPLS_MIT.txt
    ├── PYOPLS_MIT.txt
    └── MBPLS_MIT.txt
```

### 3.2 Hierarchical Algorithm Design

The library separates **core algorithms** from **method extensions**:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER-FACING API                         │
│  PLSRegression, OPLS, SparsePLS, LWPLS, PLSDA, etc.        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   METHOD EXTENSIONS                          │
│  - OPLS: orthogonal filter + PLS core                       │
│  - LWPLS: weighted PLS core per query                       │
│  - SparsePLS: L1 penalty in PLS core                        │
│  - MBPLS: block-weighted PLS core                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE ALGORITHMS                         │
│  - NIPALS: iterative weight/score/loading extraction        │
│  - SIMPLS: covariance matrix deflation                      │
│  - IKPLS: improved kernel PLS algorithms (#1, #2)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND OPERATIONS                        │
│  Linear algebra, kernel functions, centering/scaling        │
│  NumPy │ JAX │ PyTorch │ TensorFlow                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Backend Switching Mechanism

```python
# Single parameter controls backend selection
from pls_gpu import PLSRegression

# NumPy (default, CPU)
pls_cpu = PLSRegression(n_components=10, backend="numpy")

# JAX (GPU/TPU)
pls_jax = PLSRegression(n_components=10, backend="jax")

# PyTorch (GPU)
pls_torch = PLSRegression(n_components=10, backend="torch")

# TensorFlow (GPU)
pls_tf = PLSRegression(n_components=10, backend="tensorflow")
```

**Backend registry implementation:**

```python
# pls_gpu/backends/__init__.py
from typing import Dict, Type, Callable
import importlib

_BACKENDS: Dict[str, str] = {
    "numpy": "pls_gpu.backends.numpy",
    "jax": "pls_gpu.backends.jax",
    "torch": "pls_gpu.backends.torch",
    "tensorflow": "pls_gpu.backends.tensorflow",
}

def get_backend(name: str = "numpy"):
    """Get backend module by name."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")
    return importlib.import_module(_BACKENDS[name])

def check_backend_available(name: str) -> bool:
    """Check if a backend is available."""
    try:
        get_backend(name)
        return True
    except ImportError:
        return False
```

## 4. Complete Method Coverage

### 4.1 Core Algorithms (Priority 1)

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **NIPALS** | Original iterative PLS algorithm | Wold 1966 |
| **SIMPLS** | Covariance deflation, no X deflation | de Jong 1993 |
| **IKPLS #1** | Improved Kernel PLS, Algorithm 1 | Dayal & MacGregor 1997 |
| **IKPLS #2** | Improved Kernel PLS, Algorithm 2 | Dayal & MacGregor 1997 |

### 4.2 Regression Methods (Priority 1-2)

| Method | Description | Reference | Priority |
|--------|-------------|-----------|----------|
| **PLSRegression** | Standard PLS1/PLS2 | Wold 1966 | 1 |
| **OPLS** | Orthogonal PLS | Trygg & Wold 2002 | 1 |
| **Kernel PLS (KPLS)** | Nonlinear PLS in RKHS | Rosipal & Trejo 2001 | 1 |
| **K-OPLS** | Kernel OPLS | Rantalainen et al. 2007 | 2 |
| **MB-PLS** | Multiblock PLS | Westerhuis et al. 1998 | 2 |
| **SO-PLS** | Sequential Orthogonalized PLS | Menichelli et al. 2014 | 3 |
| **LW-PLS** | Locally-Weighted PLS | Kim et al. 2011 | 2 |
| **Sparse PLS** | L1-regularized PLS | Lê Cao et al. 2008 | 2 |
| **Recursive PLS** | Online/adaptive PLS (supports `partial_fit`) | Helland et al. 1992 | 3 |
| **Robust PLS** | Outlier-resistant PLS (RSIMPLS) | Hubert & Vanden Branden 2003 | 3 |
| **DiPLS** | Dynamic PLS with time lags | Kaspar & Ray 1993 | 3 |
| **O2PLS** | Two-way Orthogonal PLS | Trygg 2002, Bouhaddani et al. 2016 | 2 |
| **ENPLS** | Ensemble PLS | Li et al. 2002 | 3 |

> **Priority 4 Methods (Experimental)**: The following methods are marked as experimental:
>
> | Method | Description | Reference |
> |--------|-------------|----------|
> | FC-KPLS | Feature-Constrained Kernel PLS | Lahat et al. 2019 |
> | OKLM-PLS | Orthogonal Kernel Latent Mapping PLS | - |
> | TBPLS | Target-Balanced PLS | - |
> | HOPLS | Higher-Order PLS | Zhao et al. 2012 |

### 4.3 Classification Methods (Priority 2)

| Method | Description | Reference | Priority |
|--------|-------------|-----------|----------|
| **PLS-DA** | PLS Discriminant Analysis | Barker & Rayens 2003 | 2 |
| **OPLS-DA** | Orthogonal PLS-DA | Bylesjö et al. 2006 | 2 |

> **PLS-DA Interface Design**: Classification methods accept categorical labels directly in
> `fit(X, y)` and handle one-hot encoding internally. The `predict()` method returns class
> labels (not one-hot), while `predict_proba()` returns class probabilities via softmax-scaled
> scores. This design simplifies user workflow compared to manual encoding.

### 4.4 Variable Selection Methods (Priority 2-3)

| Method | Description | Reference | Priority |
|--------|-------------|-----------|----------|
| **iPLS** | Interval PLS | Norgaard et al. 2000 | 2 |
| **VIP** | Variable Importance in Projection | Wold et al. 1993 | 2 |
| **MC-UVE** | Monte Carlo UVE | Centner et al. 1996 | 3 |
| **CARS** | Competitive Adaptive Reweighted Sampling | Li et al. 2009 | 3 |
| **SPA** | Successive Projections Algorithm | Araújo et al. 2001 | 3 |

### 4.5 Multiway Methods (Priority 3)

| Method | Description | Reference | Priority |
|--------|-------------|-----------|----------|
| **N-PLS** | N-way PLS for 3D+ data | Bro 1996 | 3 |

> **HOPLS Deferred**: Higher-Order PLS (HOPLS) is deferred to v1.2+ as it requires
> significant tensor operations and has limited reference implementations for validation.

### 4.6 Dimensionality Reduction (Bonus)

| Method | Description | Reference | Priority |
|--------|-------------|-----------|----------|
| **PCA** | Principal Component Analysis | - | 2 |
| **Kernel PCA** | Nonlinear PCA | Schölkopf et al. 1998 | 3 |
| **CCA** | Canonical Correlation Analysis | Hotelling 1936 | 3 |

## 5. API Design

### 5.1 Unified Top-Level API

The library provides two unified top-level estimators that serve as the **primary user-facing API**:
- `PLSRegressor`: Unified interface for all PLS regression methods
- `PLSClassifier`: Unified interface for all PLS classification methods

These classes accept either a configuration dictionary or individual method-specific parameters,
ensuring a consistent signature while delegating to the appropriate underlying implementation.

```python
from pls_gpu import PLSRegressor, PLSClassifier

# Option 1: Simple usage with method selection
model = PLSRegressor(method='simpls', n_components=10, backend='jax')
model.fit(X, y)

# Option 2: Configuration dict for advanced methods
config = {
    'method': 'opls',
    'n_components': 1,
    'pls_components': 5,
    'scale': True
}
model = PLSRegressor(config=config, backend='jax')

# Option 3: Method-specific parameters passed through
model = PLSRegressor(
    method='lwpls',
    n_components=10,
    lambda_in_similarity=0.5,  # LWPLS-specific parameter
    backend='numpy'
)

# Classification
clf = PLSClassifier(method='plsda', n_components=5)
clf.fit(X, y_labels)
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)
```

#### 5.1.1 PLSRegressor Implementation

```python
from typing import Literal, Optional, Union, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

MethodType = Literal[
    'pls', 'nipals', 'simpls', 'ikpls1', 'ikpls2',  # Core PLS
    'opls', 'kopls',                                  # Orthogonal
    'kpls',                                           # Kernel
    'mbpls', 'sopls',                                 # Multiblock
    'lwpls',                                          # Local
    'spls',                                           # Sparse
    'rpls',                                           # Recursive
    'robpls',                                         # Robust
    'dipls',                                          # Dynamic
    'o2pls',                                          # Two-way orthogonal
]

class PLSRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """Unified PLS Regressor supporting all PLS methods.

    This is the recommended top-level API for PLS regression. It provides
    a consistent interface across all PLS variants while allowing method-specific
    parameters.

    Parameters
    ----------
    method : str, default='simpls'
        The PLS method to use. Options:
        - Core: 'pls', 'nipals', 'simpls', 'ikpls1', 'ikpls2'
        - Orthogonal: 'opls', 'kopls'
        - Kernel: 'kpls'
        - Multiblock: 'mbpls', 'sopls'
        - Local: 'lwpls'
        - Sparse: 'spls'
        - Recursive: 'rpls'
        - Robust: 'robpls'
        - Dynamic: 'dipls'
        - Two-way: 'o2pls'

    config : dict, optional
        Configuration dictionary for advanced method setup.
        If provided, overrides individual parameters.
        Keys depend on selected method.

    n_components : int, default=10
        Number of components to extract.

    scale : bool, default=True
        Whether to scale X and Y to unit variance.

    center : bool, default=True
        Whether to center X and Y.

    backend : str, default='numpy'
        Computational backend: 'numpy', 'jax', 'torch', 'tensorflow'.

    **method_params : dict
        Additional method-specific parameters passed to the underlying
        estimator (e.g., lambda_in_similarity for LWPLS, alpha for SparsePLS).

    Attributes
    ----------
    estimator_ : BasePLS
        The underlying fitted PLS estimator.
    method : str
        The actual method used.

    Examples
    --------
    >>> from pls_gpu import PLSRegressor
    >>> model = PLSRegressor(method='opls', n_components=1, pls_components=5)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X_test)
    """

    _METHOD_MAP = {
        'pls': 'PLSRegression',
        'nipals': 'PLSRegression',
        'simpls': 'PLSRegression',
        'ikpls1': 'PLSRegression',
        'ikpls2': 'PLSRegression',
        'opls': 'OPLS',
        'kopls': 'KOPLS',
        'kpls': 'KernelPLS',
        'mbpls': 'MBPLS',
        'sopls': 'SOPLS',
        'lwpls': 'LWPLS',
        'spls': 'SparsePLS',
        'rpls': 'RecursivePLS',
        'robpls': 'RobustPLS',
        'dipls': 'DiPLS',
        'o2pls': 'O2PLS',
    }

    def __init__(
        self,
        method: MethodType = 'simpls',
        config: Optional[Dict[str, Any]] = None,
        n_components: int = 10,
        scale: bool = True,
        center: bool = True,
        backend: str = 'numpy',
        **method_params
    ):
        self.method = method
        self.config = config
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.backend = backend
        self.method_params = method_params

    def _get_estimator_class(self):
        """Get the underlying estimator class for the method."""
        from . import regression, kernel
        class_name = self._METHOD_MAP.get(self.method)
        if class_name is None:
            raise ValueError(f"Unknown method: {self.method}")
        # Import from appropriate submodule
        # ... (implementation details)
        return estimator_class

    def _build_estimator(self):
        """Build the underlying estimator with appropriate parameters."""
        cls = self._get_estimator_class()

        # Start with base params
        params = {
            'n_components': self.n_components,
            'scale': self.scale,
            'center': self.center,
            'backend': self.backend,
        }

        # Handle algorithm selection for PLSRegression
        if self.method in ('nipals', 'simpls', 'ikpls1', 'ikpls2'):
            params['algorithm'] = self.method

        # Override with config if provided
        if self.config:
            params.update(self.config)

        # Add method-specific params
        params.update(self.method_params)

        return cls(**params)

    def fit(self, X, y) -> 'PLSRegressor':
        """Fit the PLS model."""
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict using the fitted model."""
        return self.estimator_.predict(X)

    def transform(self, X) -> np.ndarray:
        """Transform X to score space."""
        return self.estimator_.transform(X)

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and transform X."""
        return self.fit(X, y).transform(X)

    def score(self, X, y) -> float:
        """Return R² score."""
        return self.estimator_.score(X, y)

    # Delegate attribute access to underlying estimator
    def __getattr__(self, name):
        if name.startswith('_') or name in ('estimator_', 'method', 'config'):
            raise AttributeError(name)
        if hasattr(self, 'estimator_'):
            return getattr(self.estimator_, name)
        raise AttributeError(name)
```

#### 5.1.2 PLSClassifier Implementation

```python
class PLSClassifier(BaseEstimator):
    """Unified PLS Classifier supporting all PLS-DA methods.

    Parameters
    ----------
    method : str, default='plsda'
        Classification method: 'plsda', 'oplsda'.

    config : dict, optional
        Configuration dictionary for advanced setup.

    n_components : int, default=5
        Number of components.

    backend : str, default='numpy'
        Computational backend.

    **method_params : dict
        Additional method-specific parameters.

    Examples
    --------
    >>> clf = PLSClassifier(method='oplsda', n_components=3)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> proba = clf.predict_proba(X_test)
    """

    _METHOD_MAP = {
        'plsda': 'PLSDA',
        'oplsda': 'OPLSDA',
    }

    def __init__(
        self,
        method: Literal['plsda', 'oplsda'] = 'plsda',
        config: Optional[Dict[str, Any]] = None,
        n_components: int = 5,
        scale: bool = True,
        center: bool = True,
        backend: str = 'numpy',
        **method_params
    ):
        self.method = method
        self.config = config
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.backend = backend
        self.method_params = method_params

    def fit(self, X, y) -> 'PLSClassifier':
        """Fit the classifier. y should be class labels."""
        # ... (similar to PLSRegressor)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return self.estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        return self.estimator_.predict_proba(X)

    def score(self, X, y) -> float:
        """Return classification accuracy."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
```

### 5.2 Base Estimator Interface

```python
from abc import ABC, abstractmethod
from typing import Union, Literal, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

BackendType = Literal["numpy", "jax", "torch", "tensorflow", "cupy"]

class BasePLS(BaseEstimator, RegressorMixin, TransformerMixin, ABC):
    """Abstract base class for all PLS estimators."""

    _estimator_type = "regressor"

    def __init__(
        self,
        n_components: int = 10,
        scale: bool = True,
        center: bool = True,
        backend: BackendType = "numpy",
        **kwargs
    ):
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.backend = backend

    @abstractmethod
    def fit(self, X, y) -> "BasePLS":
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict using the fitted model."""
        pass

    def transform(self, X) -> np.ndarray:
        """Transform X to score space."""
        pass

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and transform X."""
        return self.fit(X, y).transform(X)

    def score(self, X, y) -> float:
        """Return R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    # Standard sklearn attributes after fit:
    # n_features_in_: int
    # n_components_: int
    # x_mean_: ndarray
    # x_std_: ndarray
    # y_mean_: ndarray
    # y_std_: ndarray
    # x_weights_: ndarray (W)
    # x_loadings_: ndarray (P)
    # y_loadings_: ndarray (Q)
    # x_scores_: ndarray (T) - training scores
    # y_scores_: ndarray (U) - training scores
    # coef_: ndarray - regression coefficients
    # r2x_per_component_: ndarray - R² of X explained per component
    # r2y_per_component_: ndarray - R² of Y explained per component

    def explained_variance_ratio(self) -> dict:
        """Return explained variance ratio for X and Y per component."""
        pass
```

### 5.3 PLSRegression API

```python
class PLSRegression(BasePLS):
    """Partial Least Squares Regression.

    Parameters
    ----------
    n_components : int, default=10
        Number of components to extract.
    algorithm : {'nipals', 'simpls', 'ikpls1', 'ikpls2'}, default='simpls'
        Algorithm to use for fitting:
        - 'nipals': Original NIPALS algorithm
        - 'simpls': SIMPLS algorithm (de Jong 1993)
        - 'ikpls1': Improved Kernel PLS Algorithm #1
        - 'ikpls2': Improved Kernel PLS Algorithm #2
    scale : bool, default=True
        Whether to scale X and Y to unit variance.
    center : bool, default=True
        Whether to center X and Y.
    max_iter : int, default=500
        Maximum iterations for NIPALS convergence.
    tol : float, default=1e-6
        Convergence tolerance for NIPALS.
    backend : {'numpy', 'jax', 'torch', 'tensorflow'}, default='numpy'
        Computational backend.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in training data.
    n_components_ : int
        Actual number of components (may be less than n_components).
    x_weights_ : ndarray of shape (n_features, n_components)
        X weights (W matrix).
    x_loadings_ : ndarray of shape (n_features, n_components)
        X loadings (P matrix).
    y_loadings_ : ndarray of shape (n_targets, n_components)
        Y loadings (Q matrix).
    x_scores_ : ndarray of shape (n_samples, n_components)
        X scores (T matrix) from training.
    y_scores_ : ndarray of shape (n_samples, n_components)
        Y scores (U matrix) from training.
    coef_ : ndarray of shape (n_features, n_targets)
        Regression coefficients.
    x_mean_, y_mean_ : ndarray
        Mean values for centering.
    x_std_, y_std_ : ndarray
        Standard deviations for scaling.

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

    def __init__(
        self,
        n_components: int = 10,
        algorithm: Literal["nipals", "simpls", "ikpls1", "ikpls2"] = "simpls",
        scale: bool = True,
        center: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        backend: BackendType = "numpy",
    ):
        super().__init__(n_components=n_components, scale=scale,
                        center=center, backend=backend)
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
```

### 5.4 OPLS API

```python
class OPLS(BasePLS):
    """Orthogonal Partial Least Squares.

    OPLS separates systematic variation in X into two parts:
    - Variation correlated with Y (predictive)
    - Variation orthogonal to Y (non-predictive)

    Parameters
    ----------
    n_components : int, default=1
        Number of orthogonal components to remove.
    pls_components : int, default=1
        Number of predictive PLS components.
    scale : bool, default=True
        Whether to scale X and Y.
    backend : {'numpy', 'jax', 'torch', 'tensorflow'}, default='numpy'
        Computational backend.

    Attributes
    ----------
    x_weights_ortho_ : ndarray
        Orthogonal X weights.
    x_loadings_ortho_ : ndarray
        Orthogonal X loadings.
    x_scores_ortho_ : ndarray
        Orthogonal X scores.
    """
```

### 5.5 Kernel PLS API

```python
class KernelPLS(BasePLS):
    """Kernel PLS for nonlinear regression.

    Parameters
    ----------
    n_components : int, default=10
        Number of components.
    kernel : {'rbf', 'linear', 'poly', 'sigmoid'}, default='rbf'
        Kernel function.
    gamma : float, optional
        Kernel coefficient. Default: 1/n_features.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term for poly/sigmoid kernels.
    center_kernel : bool, default=True
        Whether to center the kernel matrix.
    backend : {'numpy', 'jax', 'torch', 'tensorflow'}, default='numpy'
        Computational backend.
    """
```

## 6. Implementation Requirements

### 6.1 Numerical Precision

- All computations in `float64` by default
- JAX: `jax.config.update("jax_enable_x64", True)`
- PyTorch: `torch.set_default_dtype(torch.float64)`
- Tolerance for numerical equivalence: `rtol=1e-5, atol=1e-8`

### 6.2 Backend-Specific Optimizations

| Backend | Optimizations |
|---------|--------------|
| **NumPy** | BLAS/LAPACK, vectorization |
| **JAX** | JIT compilation, vmap, GPU/TPU, XLA |
| **PyTorch** | CUDA kernels, autograd (optional) |
| **TensorFlow** | XLA compilation, mixed precision |

> **GPU Optimization Note**: Iterative algorithms like NIPALS may underutilize GPU parallelism
> if implemented with Python loops. For GPU backends, the library **prioritizes fully-vectorized
> algorithms** (SIMPLS, IKPLS, batched SVD) that leverage GPU parallelism effectively. NIPALS
> is included primarily for sklearn compatibility and reference validation, with an internal
> recommendation to use `algorithm='simpls'` or `algorithm='ikpls1'` for GPU backends.
>
> The default algorithm selection is backend-aware:
> - `backend='numpy'`: defaults to `algorithm='simpls'`
> - `backend='jax'|'torch'|'tensorflow'`: defaults to `algorithm='ikpls1'` (fully vectorized)

### 6.3 Memory Efficiency

- Lazy loading of backends (only import when used)
- Standard in-memory operation by default (optimized for typical chemometrics datasets)
- **Future/Optional**: Chunked processing for very large datasets (>100k samples or >GB-scale)
- **Future/Optional**: Memory-mapped array support for out-of-core processing

> **Design Philosophy**: Focus on efficient in-memory operation first. Chunking and memory-mapping
> are deferred to v1.1+ unless real use cases (e.g., large spectral imaging libraries, LUCAS soil
> database) demonstrate clear need. Avoid over-engineering for typical NIR/chemometrics datasets.

### 6.4 Thread Safety

- No global state modification
- Backend-local caches
- Thread-safe JIT compilation caching

## 7. Testing Strategy

### 7.1 Unit Tests

- Each method has dedicated test module
- Test all backends (parametrized)
- Test edge cases:
  - Single sample, single feature inputs
  - Zero-variance columns in X or Y (should be handled gracefully)
  - Requesting more components than `min(n_samples, n_features)` (should cap and warn)
  - Extremely high collinearity in X (should still function)
  - Multi-target Y (PLS2) with correlated targets
  - Classification datasets for PLS-DA (test label encoding, predict, predict_proba)

### 7.2 Numerical Equivalence Tests

- Compare against reference implementations:
  - sklearn PLSRegression (NIPALS)
  - R pls package (SIMPLS, NIPALS)
  - MATLAB implementations
  - ikpls package
- Maximum acceptable deviation: `rtol=1e-5`
- Test sklearn Pipeline integration (GridSearchCV, cross_val_score)
- Ensure outputs are always NumPy arrays for sklearn compatibility

### 7.3 Performance Tests

- Benchmark each backend
- Scaling tests (samples, features, components)
- Memory profiling

## 8. Documentation Requirements

### 8.1 API Documentation

- Numpy-style docstrings
- Examples for each class
- Mathematical notation and formulas

### 8.2 User Guide

- Getting started
- Backend selection guide
- Performance tuning
- Migration from sklearn/other packages

### 8.3 Mathematical Reference

- Detailed algorithm descriptions
- Notation conventions
- Derivations for each method
- Literature references

## 9. nirs4all Integration

### 9.1 Drop-in Replacement

```python
# Current nirs4all usage:
from nirs4all.operators.models.sklearn import SIMPLS, OPLS

# With pls-gpu:
from pls_gpu import PLSRegression, OPLS
# or
from pls_gpu.compat.nirs4all import SIMPLS, OPLS  # Aliased imports
```

### 9.2 Compatibility Layer

```python
# pls_gpu/compat/nirs4all.py
from pls_gpu import PLSRegression, OPLS as _OPLS

class SIMPLS(PLSRegression):
    """nirs4all-compatible SIMPLS wrapper."""
    def __init__(self, n_components=10, **kwargs):
        super().__init__(n_components=n_components,
                        algorithm='simpls', **kwargs)

# Export all aliases
__all__ = ['SIMPLS', 'OPLS', 'IKPLS', 'LWPLS', 'SparsePLS', ...]
```

## 10. License and Distribution

- **License**: MIT or BSD-3-Clause (permissive, compatible with sklearn)
- **Distribution**: PyPI (`pip install pls-gpu`)
- **Python versions**: 3.11+
- **Optional dependencies**: jax, torch, tensorflow (extras)

### 10.1 Third-Party Code Attribution

Some algorithm implementations in pls-gpu are derived from or inspired by existing open-source libraries. All such code is properly attributed and used in compliance with their respective licenses.

| Source Library | License | Methods Derived | Attribution |
|---------------|---------|-----------------|-------------|
| scikit-learn | BSD-3-Clause | NIPALS base algorithm, estimator patterns | sklearn/cross_decomposition/_pls.py |
| ikpls | MIT | IKPLS Algorithm #1 and #2 | Engstrøm 2024 |
| pyopls | MIT | OPLS orthogonalization approach | pyopls/opls.py |
| mixOmics (R) | GPL-3.0 | Sparse PLS L1 penalty formulation | Reference only (reimplemented) |
| R pls | GPL-2.0+ | SIMPLS algorithm | Reference only (reimplemented) |
| mbpls | MIT | Multiblock PLS structure | mbpls/mbpls.py |

> **Important License Notes**:
> - All GPL-licensed code (mixOmics, R pls) is used as **reference only** for algorithm correctness.
>   The pls-gpu implementations are clean-room reimplementations based on published papers.
> - BSD/MIT-licensed code may be adapted directly with proper attribution in source files.
> - Each source file with derived code MUST include a header comment with:
>   1. Original source library and file
>   2. Original license
>   3. Original authors/copyright holders
>   4. Modifications made

**Example attribution header:**

```python
# pls_gpu/core/nipals.py
#
# NIPALS algorithm implementation.
#
# Portions derived from scikit-learn (BSD-3-Clause License)
# Original source: sklearn/cross_decomposition/_pls.py
# Copyright (c) 2007-2024 The scikit-learn developers.
# See THIRD_PARTY_NOTICES.md for full license text.
#
# Modifications:
# - Multi-backend support (NumPy, JAX, PyTorch, TensorFlow)
# - Configurable precision (float32/float64)
# - GPU-optimized vectorization
```

### 10.2 THIRD_PARTY_NOTICES.md

The repository MUST include a `THIRD_PARTY_NOTICES.md` file at the root containing:
1. Full text of all third-party licenses
2. Complete list of derived code with file locations
3. Acknowledgments to original authors

---

## Appendix A: Reference Implementations

| Method | Reference Implementation | Language | Source |
|--------|-------------------------|----------|--------|
| NIPALS | sklearn PLSRegression | Python | sklearn |
| SIMPLS | pls::plsr | R | CRAN pls |
| IKPLS | ikpls | Python | PyPI |
| OPLS | pyopls | Python | GitHub |
| OPLS | ropls | R | Bioconductor |
| MB-PLS | mbpls | Python | PyPI |
| LW-PLS | lwpls | Python | GitHub |
| Sparse PLS | mixOmics::spls | R | Bioconductor |
| K-OPLS | ConsensusOPLS | R | GitHub |

## Appendix B: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| X | Input matrix (n × p) |
| Y | Target matrix (n × q) |
| T | X scores (n × k) |
| U | Y scores (n × k) |
| W | X weights (p × k) |
| P | X loadings (p × k) |
| Q | Y loadings (q × k) |
| B | Regression coefficients (p × q) |
| k | Number of components |
| n | Number of samples |
| p | Number of features |
| q | Number of targets |

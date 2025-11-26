# PLS Methods Integration Roadmap for nirs4all

This roadmap describes the integration plan for PLS (Partial Least Squares) methods into nirs4all.
Methods are sorted from easiest to hardest to integrate based on:
1. Existing Python implementation availability
2. sklearn API compatibility
3. Required dependencies
4. Implementation complexity

---
First create the module for pls in operators/models
When implementing a story:
- add the dependancy to import and install the package
- create the operator (for now you nearly created all of them in Q18_pls_method.py - that's what not intended) and put them with other models (for now 1 file per model) and update __init__.
- add the new operator in a simple example (it should have been Q18 but far too complex so use Q19)
- add unitary tests
- launch example et tests.

---


## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Already integrated or trivial |
| 🟢 | Easy - Direct sklearn usage or well-maintained package |
| 🟡 | Medium - Requires wrapper or minor adaptation |
| 🟠 | Hard - Requires significant wrapper or custom implementation |
| 🔴 | Very Hard - No Python implementation, needs full custom code |

---

## Tier 1: Direct sklearn Integration (Easiest) 🟢

These methods use sklearn directly or are fully sklearn-compatible.

### Story 1.1: PLSRegression (NIPALS PLS1/PLS2)

**Priority:** P0 - Already Available
**Status:** ✅ Already integrated
**Difficulty:** Trivial

**Description:**
Standard PLS regression using NIPALS algorithm. This is the baseline for NIRS calibration.

**Dependency:**
```toml
# Already in dependencies
"scikit-learn>=0.24.0"
```

**Usage in nirs4all:**
```python
from sklearn.cross_decomposition import PLSRegression
pipeline = [
    PLSRegression(n_components=10),
]
```

**Test:** `examples/Q1_regression.py` already uses PLSRegression.

---

### Story 1.2: PLS-DA (Discriminant Analysis)

**Priority:** P0 - Already Available
**Status:** ✅ Already integrated via PLSRegression + OneHotEncoder
**Difficulty:** Trivial

**Description:**
PLS-DA for classification uses PLSRegression with one-hot encoded Y targets.

**Dependency:**
```toml
# Already in dependencies
"scikit-learn>=0.24.0"
```

**Implementation:**
Use `PLSRegression` with `OneHotEncoder` for classification tasks. The `SklearnModelController` already handles this.

**Usage in nirs4all:**
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# For classification, wrap PLSRegression with one-hot encoding
# or use the PLSDA operator (see Story 1.2b)
```

**Test:** Add to `examples/Q19_pls_tests.py

---

## Tier 2: External Well-Maintained Packages 🟢

These methods require additional packages that are well-maintained and sklearn-compatible.

### Story 2.1: IKPLS - Improved Kernel PLS with Fast CV

**Priority:** P1 - High Value
**Status:** ✅ Complete
**Difficulty:** Low
**Package:** `ikpls`

**Description:**
Fast PLS with NumPy and JAX backends. Supports GPU acceleration and orders-of-magnitude faster cross-validation. Both algorithm variants (1 and 2) are supported.

**Dependency:**
```toml
[project.dependencies]
# Add to main dependencies
"ikpls>=1.0.0"

[project.optional-dependencies]
jax = ["jax>=0.4.10", "jaxlib>=0.4.10", "ikpls>=1.0.0"]
```

```python
from nirs4all.operators.models.sklearn.pls import IKPLS

# NumPy backend (default) - CPU
model_numpy = IKPLS(n_components=10, backend='numpy')
model_numpy.fit(X, y)
y_pred = model_numpy.predict(X_val)

# JAX backend - GPU/TPU acceleration
model_jax = IKPLS(n_components=10, backend='jax', algorithm=1)
model_jax.fit(X, y)
y_pred = model_jax.predict(X_val)
```

**Tasks:**
- [x] Add `ikpls` to dependencies
- [x] Create `IKPLS` wrapper for consistent API
- [x] Add tests for IKPLS with NumPy backend
- [x] Add tests for IKPLS with JAX backend (optional dependency)
- [x] Add `backend` parameter to switch between 'numpy' and 'jax'
- [x] Add algorithm parameter support (1 or 2)

**Test:** `examples/Q19_pls_test.py`, `tests/unit/operators/models/test_sklearn_pls.py`

---

### Story 2.2: OPLS / OPLS-DA (Orthogonal PLS)

**Priority:** P1 - High Value
**Status:** 🟢 Easy
**Difficulty:** Low
**Package:** `pyopls`

**Description:**
Removes Y-orthogonal X-variation to sharpen interpretability. Standard in metabolomics, applicable to NIRS.

**Dependency:**
```toml
[project.dependencies]
"pyopls>=0.1.0"
```

```python
from pyopls import OPLS
from sklearn.cross_decomposition import PLSRegression

# OPLS as transformer + 1-comp PLS
Z = OPLS(n_components=1).fit_transform(X, y)
model = PLSRegression(n_components=1).fit(Z, y)
```

**Tasks:**
- [x] Add `pyopls` to dependencies
- [x] Create `OPLSTransformer` wrapper
- [x] Create `OPLSDA` classifier wrapper
- [x] Add tests

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 2.3: MB-PLS (Multiblock PLS)

**Priority:** P2 - Medium Value
**Status:** 🟢 Easy
**Difficulty:** Low
**Package:** `mbpls`

**Description:**
Fuse multiple X blocks (e.g., different preprocessing variants, multiple sensors). sklearn-style API.

**Dependency:**
```toml
[project.dependencies]
"mbpls>=1.0.0"
```


```python
from mbpls.mbpls import MBPLS

# Multiblock usage
model = MBPLS(n_components=6).fit([X_raw, X_snv], y)
y_pred = model.predict([X_raw_val, X_snv_val])
```

**Tasks:**
- [x] Add `mbpls` to dependencies
- [x] Create special handling in pipeline for multi-block inputs
- [x] Add tests

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 2.4: DiPLS (Dynamic PLS)

**Priority:** P3 - Specialized
**Status:** 🟢 Easy
**Difficulty:** Low
**Package:** `trendfitter`

**Description:**
Handle time-lagged process/NIR streams via Hankelization. Useful for process analytics.

**Dependency:**
```toml
[project.dependencies]
"trendfitter>=0.1.0"
```


```python
from trendfitter import DiPLS

model = DiPLS(n_components=8, lags=5).fit(X_time, y_time)
y_pred = model.predict(X_time_val)
```

**Tasks:**
- [x] Add `trendfitter` to dependencies
- [x] Create `DiPLSOperator` wrapper
- [x] Add tests

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 2.5: Sparse PLS (sPLS)

**Priority:** P2 - Medium Value
**Status:** 🟢 Easy
**Difficulty:** Low
**Package:** `py-ddspls`

**Description:**
Joint prediction and variable selection via penalized loadings. Useful with many wavelengths.

**Dependency:**
```toml
[project.dependencies]
"py-ddspls>=0.1.0"
```


```python
from ddspls import ddsPLS

model = ddsPLS(n_components=5).fit(X, y)
y_pred = model.predict(X_val)
```

**Tasks:**
- [x] Add `py-ddspls` to dependencies
- [x] Create `SparsePLSOperator` wrapper
- [x] Add tests

**Test:** Add to `examples/Q19_pls_test.py`

---

## Tier 3: Packages Requiring Wrappers 🟡

These methods have Python implementations but need custom wrappers for sklearn compatibility.

### Story 3.1: LW-PLS (Locally-Weighted PLS)

**Priority:** P2 - Medium Value
**Status:** ✅ Implemented
**Difficulty:** Medium
**Package:** Vendored from `hkaneko1985/lwpls` (MIT License)

**Description:**
Just-in-time local models near each query sample. Useful with drift or local nonlinearity.

**Implementation:**
Core algorithm vendored from https://github.com/hkaneko1985/lwpls (MIT License).
Wrapped as sklearn-compatible `LWPLS` class with `fit()`, `predict()`, and full sklearn API.

**Operator:** `nirs4all/operators/models/sklearn/lwpls.py`

```python
from nirs4all.operators.models import LWPLS

# Build a local PLS model for each prediction
model = LWPLS(n_components=10, lambda_in_similarity=0.5, scale=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get predictions for all component numbers (for component selection)
all_preds = model.predict_all_components(X_test)
```

**Tasks:**
- [x] Vendor lwpls algorithm (MIT licensed)
- [x] Create sklearn-compatible wrapper with proper docstrings
- [x] Add `predict_all_components()` for component selection
- [x] Add comprehensive unit tests (22 tests)
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestLWPLS`

---

## Tier 4: Variable Selection Methods 🟡

These methods wrap PLS for wavelength/variable selection.

### Story 4.1: VIP (Variable Importance in Projection)

**Priority:** P2 - High Value for NIRS
**Status:** 🟡 Medium
**Difficulty:** Medium
**Package:** `auswahl`

**Description:**
Rank wavelengths from a fitted PLS model. Quick filter for variable selection.

**Dependency:**
```toml
[project.dependencies]
"auswahl>=0.9.0"
```

**Operator:** `nirs4all/operators/transformations/feature_selection.py`
```python
from auswahl import VIP

selector = VIP(pls_kwargs=dict(n_components=8)).fit(X, y)
X_selected = X[:, selector.get_support(indices=True)]
```

**Tasks:**
- [ ] Add `auswahl` to dependencies
- [ ] Create `VIPSelector` transformer
- [ ] Add tests

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 4.2: MC-UVE (Monte-Carlo Uninformative Variable Elimination)

**Priority:** P2 - High Value for NIRS
**Status:** 🟡 Medium
**Difficulty:** Medium
**Package:** `auswahl`

**Description:**
Stability of regression weights under resampling for robust wavelength culling.

**Dependency:**
```toml
[project.dependencies]
"auswahl>=0.9.0"
```

**Operator:** `nirs4all/operators/transformations/feature_selection.py`
```python
from auswahl import MCUVE

selector = MCUVE(pls_kwargs=dict(n_components=8), n_iter=200).fit(X, y)
X_selected = X[:, selector.get_support()]
```

**Tasks:**
- [ ] Add tests for MCUVE
- [ ] Document usage

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 4.3: CARS (Competitive Adaptive Reweighted Sampling)

**Priority:** P2 - High Value for NIRS
**Status:** 🟡 Medium
**Difficulty:** Medium
**Package:** `auswahl`

**Description:**
Strong NIR track record for wavelength selection.

**Dependency:**
```toml
[project.dependencies]
"auswahl>=0.9.0"
```

**Operator:** `nirs4all/operators/transformations/feature_selection.py`
```python
from auswahl import CARS

selector = CARS(pls_kwargs=dict(n_components=8), n_runs=50, n_select=80).fit(X, y)
X_selected = X[:, selector.get_support(indices=True)]
```

**Tasks:**
- [ ] Add tests for CARS
- [ ] Document usage

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 4.4: SPA (Successive Projections Algorithm)

**Priority:** P2 - High Value for NIRS
**Status:** 🟡 Medium
**Difficulty:** Medium
**Package:** `auswahl`

**Description:**
Successive projections for wavelength selection.

**Dependency:**
```toml
[project.dependencies]
"auswahl>=0.9.0"
```

**Operator:** `nirs4all/operators/transformations/feature_selection.py`
```python
from auswahl import SPA

selector = SPA(n_features_to_select=50).fit(X, y)
X_selected = X[:, selector.get_support(indices=True)]
```

**Tasks:**
- [ ] Add tests for SPA
- [ ] Document usage

**Test:** Add to `examples/Q19_pls_test.py`

---

## Tier 5: Custom Implementations Required 🟠

These methods have no standard Python package and require custom implementation.

### Story 5.1: SIMPLS

**Priority:** P3 - Nice to Have
**Status:** 🟠 Hard
**Difficulty:** Medium-High

**Description:**
Same predictions as PLSRegression with different deflation algorithm. Common in chemometrics literature.

**Dependency:** None (pure Python/NumPy implementation)

**Implementation Plan:**
1. Implement de Jong 1993 SIMPLS algorithm
2. Iterative weight via dominant eigenvector of (X^T Y Y^T X)
3. Deflate covariance, accumulate loadings
4. Mirror sklearn API for fit/predict
5. Add CV support

**Operator:** `nirs4all/operators/models/sklearn/simpls.py`
```python
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SIMPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        # Implement SIMPLS algorithm
        ...
        return self

    def predict(self, X):
        ...
```

**Tasks:**
- [x] Implement SIMPLS algorithm from de Jong 1993 paper
- [x] Create sklearn-compatible class
- [x] Add unit tests comparing to PLSRegression
- [x] Benchmark performance

**Test:** Add to `examples/Q19_pls_test.py`

---

### Story 5.2: iPLS (Interval PLS)

**Priority:** P2 - High Value for NIRS
**Status:** ✅ Implemented
**Difficulty:** Medium

**Description:**
Grid or heuristic search over contiguous wavelength windows with local PLS models.
Identifies optimal spectral regions for prediction in NIRS data.

**Dependency:** None (uses sklearn internally)

**Implementation:**
Full implementation with NumPy and JAX backends. Supports three selection modes
(single, forward, backward) and two combination methods (best, union).

**Operator:** `nirs4all/operators/models/sklearn/ipls.py`

```python
from nirs4all.operators.models import IntervalPLS

# Single interval mode - select best single interval
model_single = IntervalPLS(n_components=5, n_intervals=10, mode='single')
model_single.fit(X, y)
y_pred = model_single.predict(X_val)

# Forward selection mode - iteratively add intervals
model_forward = IntervalPLS(n_components=5, n_intervals=10, mode='forward')
model_forward.fit(X, y)
print(f"Selected intervals: {model_forward.selected_intervals_}")
print(f"Selected regions: {model_forward.selected_regions_}")

# Get interval evaluation info
info = model_forward.get_interval_info()
print(f"Interval scores: {info['interval_scores']}")

# JAX backend for GPU acceleration
model_jax = IntervalPLS(n_components=5, n_intervals=10, backend='jax')
model_jax.fit(X, y)
```

**Tasks:**
- [x] Implement interval grid search logic
- [x] Support forward/backward interval selection
- [x] Support single interval mode
- [x] Support union/best combination methods
- [x] Add NumPy backend implementation
- [x] Add JAX backend implementation
- [x] Add comprehensive unit tests
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestIntervalPLS`

---

### Story 5.3: Robust PLS / RSIMPLS

**Priority:** P3 - Specialized
**Status:** ✅ Implemented
**Difficulty:** High

**Description:**
Down-weight outliers using robust covariance estimation. Uses RSIMPLS algorithm with
Iteratively Reweighted Least Squares (IRLS) and SIMPLS-style deflation.

**Dependency:** None (pure Python/NumPy implementation, optional JAX for GPU)

**Implementation:**
Full implementation with NumPy and JAX backends. Supports two robust weighting schemes:
- **Huber** (default, c=1.345): Smooth transition between L1/L2, gentle outlier downweighting
- **Tukey bisquare** (c=4.685): Hard outlier rejection, zero weight for extreme outliers

Uses Median Absolute Deviation (MAD) for robust scale estimation.

**Operator:** `nirs4all/operators/models/sklearn/robust_pls.py`

```python
from nirs4all.operators.models import RobustPLS

# Huber weighting (default) - NumPy backend
model_huber = RobustPLS(n_components=10, weighting='huber', c=1.345)
model_huber.fit(X, y)
y_pred = model_huber.predict(X_val)

# Tukey bisquare weighting - more aggressive outlier rejection
model_tukey = RobustPLS(n_components=10, weighting='tukey', c=4.685)
model_tukey.fit(X, y)
y_pred = model_tukey.predict(X_val)

# Outlier detection
outlier_mask = model_huber.get_outlier_mask(threshold=0.5)
print(f"Samples with low weights: {outlier_mask.sum()}")
print(f"Sample weights: {model_huber.sample_weights_}")

# JAX backend for GPU acceleration
model_jax = RobustPLS(n_components=10, weighting='huber', backend='jax')
model_jax.fit(X, y)
```

**Tasks:**
- [x] Research RSIMPLS algorithm details
- [x] Implement Huber and Tukey robust weighting schemes
- [x] Implement MAD robust scale estimation
- [x] Create sklearn-compatible class with NumPy backend
- [x] Add JAX backend for GPU acceleration
- [x] Add get_outlier_mask() method for outlier detection
- [x] Expose sample_weights_ attribute
- [x] Add comprehensive unit tests (31 tests)
- [x] Add to Q19_pls_test.py example

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestRobustPLS`

---

### Story 5.4: Recursive PLS (RPLS)

**Priority:** P3 - Specialized
**Status:** ✅ Implemented
**Difficulty:** Medium-High

**Description:**
Online model updates for drifting processes using exponentially weighted recursive least squares.
Useful for process monitoring and adaptation to instrument drift.

**Dependency:** None (pure Python/NumPy, optional JAX for GPU)

**Implementation:**
Full implementation with NumPy and JAX backends. Uses SIMPLS-style deflation for initial fit
and exponentially weighted covariance updates for partial_fit. Supports forgetting factor
(0-1] for controlling adaptation rate.

**Operator:** `nirs4all/operators/models/sklearn/recursive_pls.py`

```python
from nirs4all.operators.models import RecursivePLS

# NumPy backend (default)
model = RecursivePLS(n_components=5, forgetting_factor=0.99)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Online updates with partial_fit
for X_batch, y_batch in streaming_data:
    model.partial_fit(X_batch, y_batch)
    y_pred = model.predict(X_new)

# Lower forgetting factor for faster adaptation to drift
model_fast_adapt = RecursivePLS(n_components=5, forgetting_factor=0.95)

# JAX backend for GPU acceleration
model_jax = RecursivePLS(n_components=5, forgetting_factor=0.99, backend='jax')
model_jax.fit(X_train, y_train)
y_pred = model_jax.predict(X_test)
```

**Tasks:**
- [x] Implement recursive update algorithm using exponential weighting
- [x] Add partial_fit support for online learning
- [x] Add NumPy backend implementation
- [x] Add JAX backend with JIT compilation
- [x] Add comprehensive unit tests (38 tests)
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestRecursivePLS`

---

## Tier 6: Kernel-Based Methods 🔴

These methods combine kernel techniques with PLS for nonlinear modeling.

### Story 6.1: K-OPLS (Kernel OPLS)

**Priority:** P4 - Specialized
**Status:** ✅ Implemented
**Difficulty:** Very High

**Description:**
Kernel-based Orthogonal Partial Least Squares. Combines kernel methods with OPLS for
nonlinear regression. Uses SIMPLS-style deflation in kernel space with orthogonal
filtering of Y-uncorrelated variation.

**Dependency:** None (pure Python/NumPy, optional JAX for GPU)

**Implementation:**
Full implementation with NumPy and JAX backends. Supports three kernel functions:
- **linear**: K(x,y) = x^T y (equivalent to linear PLS with centering)
- **rbf** (default): K(x,y) = exp(-gamma ||x-y||^2) for nonlinear relationships
- **poly**: K(x,y) = (gamma x^T y + coef0)^degree for polynomial relationships

Uses kernel centering for both training and prediction. Extracts orthogonal components
to filter Y-uncorrelated variation before fitting kernel PLS.

**Operator:** `nirs4all/operators/models/sklearn/kopls.py`

```python
from nirs4all.operators.models import KOPLS

# RBF kernel (default) - NumPy backend
model_rbf = KOPLS(n_components=5, n_ortho_components=2, kernel='rbf', gamma=0.1)
model_rbf.fit(X, y)
y_pred = model_rbf.predict(X_val)

# Polynomial kernel
model_poly = KOPLS(n_components=5, kernel='poly', degree=3, gamma=0.1, coef0=1.0)
model_poly.fit(X, y)

# Linear kernel (comparable to linear OPLS)
model_linear = KOPLS(n_components=5, kernel='linear')
model_linear.fit(X, y)

# Transform to scores
T_ortho = model_rbf.transform(X)  # Returns orthogonal scores
T_pred = model_rbf.transform_predictive(X)  # Returns predictive scores

# JAX backend for GPU acceleration
model_jax = KOPLS(n_components=5, kernel='rbf', gamma=0.1, backend='jax')
model_jax.fit(X, y)
```

**Key Features:**
- sklearn-compatible API (fit/predict/transform/get_params/set_params)
- Automatic kernel parameter defaults (gamma = 1/n_features)
- Optional scaling and centering of input data
- Exposes orthogonal scores, loadings, and kernel regression coefficients

**Tasks:**
- [x] Implement K-OPLS algorithm from Rantalainen et al. 2007
- [x] Create sklearn-compatible class with NumPy backend
- [x] Add JAX backend with JIT compilation
- [x] Support linear, RBF, and polynomial kernels
- [x] Add kernel centering for training and prediction
- [x] Add comprehensive unit tests (40 tests)
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestKOPLS`

**Note:** K-OPLS requires careful hyperparameter tuning (kernel type, gamma, n_components,
n_ortho_components) for optimal performance on NIRS data. Default parameters may not be
optimal for all datasets - grid search or cross-validation is recommended.

**Reference:** Rantalainen et al. 2007 - "Kernel-based orthogonal projections to latent structures"

---

### Story 6.2: KernelPLS / NL-PLS (Nonlinear PLS via Kernel Methods)

**Priority:** P3 - Specialized
**Status:** ✅ Implemented
**Difficulty:** Medium

**Description:**
Kernel PLS (also known as NL-PLS) maps the input data X into a higher-dimensional feature
space using a kernel function (RBF, polynomial, sigmoid) and then fits a PLS model on the
kernel matrix K(X, X). This allows capturing nonlinear relationships between X and Y
while retaining the interpretability of PLS.

**Algorithm:**
1. Compute kernel matrix K = kernel(X_train, X_train)
2. Center the kernel matrix
3. Fit SIMPLS-style PLS on K with target Y
4. For prediction: K_test = kernel(X_test, X_train), center, predict

This is the simple and effective approach described in the user-provided code snippet:
```python
def npls(X_train, y_train, X_test, n_components=2, gamma=1.0):
    K_train = rbf_kernel(X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    pls = PLSRegression(n_components=n_components)
    pls.fit(K_train, y_train)
    y_pred = pls.predict(K_test)
    return y_pred.ravel()
```

**Dependency:** None (pure Python/NumPy, optional JAX for GPU)

**Operator:** `nirs4all/operators/models/sklearn/nlpls.py`

```python
from nirs4all.operators.models import KernelPLS

# RBF kernel (default) - NumPy backend
model_rbf = KernelPLS(n_components=10, kernel='rbf', gamma=0.1)
model_rbf.fit(X, y)
y_pred = model_rbf.predict(X_val)

# Polynomial kernel
model_poly = KernelPLS(n_components=10, kernel='poly', degree=3, gamma=0.1)
model_poly.fit(X, y)

# Sigmoid kernel
model_sigmoid = KernelPLS(n_components=10, kernel='sigmoid', gamma=0.01)
model_sigmoid.fit(X, y)

# Linear kernel (equivalent to standard PLS)
model_linear = KernelPLS(n_components=10, kernel='linear')
model_linear.fit(X, y)

# Transform to kernel PLS score space
T = model_rbf.transform(X)

# JAX backend for GPU acceleration
model_jax = KernelPLS(n_components=10, kernel='rbf', gamma=0.1, backend='jax')
model_jax.fit(X, y)
```

**Key Features:**
- sklearn-compatible API (fit/predict/transform/get_params/set_params)
- Multiple kernel functions: rbf, poly, sigmoid, linear
- Automatic kernel parameter defaults (gamma = 1/n_features)
- Optional kernel centering and Y scaling
- NumPy and JAX backends for CPU/GPU acceleration
- Exposes x_scores_, y_scores_, coef_ for interpretability

**Aliases:**
- `KernelPLS` - Main class name
- `NLPLS` - Alias for Nonlinear PLS
- `KPLS` - Alias for Kernel PLS

**Tasks:**
- [x] Implement Kernel PLS algorithm with SIMPLS on kernel matrix
- [x] Create sklearn-compatible class with NumPy backend
- [x] Add JAX backend with JIT compilation
- [x] Support rbf, poly, sigmoid, and linear kernels
- [x] Add kernel centering for training and prediction
- [x] Add comprehensive unit tests
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestKernelPLS`

**Note:** The gamma parameter significantly affects performance. For NIRS data, start with
small gamma values (0.01-0.1) for RBF kernel. Cross-validation is recommended for tuning.

**Reference:**
- Rosipal & Trejo (2001) - "Kernel partial least squares regression in RKHS"
- Zheng et al. (2024) - "A non-linear PLS based on monotonic inner relation" (MIR-PLS variant)

---

## Tier 7: Specialized/Advanced PLS Methods 🟠

These methods combine PLS with advanced techniques for specific use cases.

### Story 7.1: OKLM-PLS (Online Koopman Latent-Mode PLS)

**Priority:** P3 - Specialized
**Status:** ✅ Implemented
**Difficulty:** High

**Description:**
Online Koopman Latent-Mode PLS combines Koopman operator theory with PLS for time-series
regression. It learns latent dynamics T_{t+1} ≈ F @ T_t while simultaneously fitting
Y_t ≈ T_t @ B. This is useful for spectral data collected over time where temporal
coherence provides additional predictive information.

**Dependency:** None (pure Python/NumPy, optional JAX for GPU)

**Operator:** `nirs4all/operators/models/sklearn/oklmpls.py`

```python
from nirs4all.operators.models import OKLMPLS, PolynomialFeaturizer, RBFFeaturizer

# Basic OKLM-PLS with dynamics constraint
model = OKLMPLS(n_components=10, lambda_dyn=1.0, lambda_reg_y=1.0)
model.fit(X, y)  # X should be temporally ordered
y_pred = model.predict(X)

# Without dynamics (equivalent to featurized PLS)
model_nodyn = OKLMPLS(n_components=10, lambda_dyn=0.0)
model_nodyn.fit(X, y)

# With polynomial featurizer for nonlinearity
model_poly = OKLMPLS(
    n_components=10,
    featurizer=PolynomialFeaturizer(degree=2),
    lambda_dyn=1.0
)
model_poly.fit(X, y)

# With RBF featurizer
model_rbf = OKLMPLS(
    n_components=10,
    featurizer=RBFFeaturizer(n_components=100, gamma=0.1),
    lambda_dyn=1.0
)
model_rbf.fit(X, y)

# Predict future timesteps using learned dynamics
future_preds = model.predict_dynamic(X, n_steps=5)

# JAX backend for GPU acceleration
model_jax = OKLMPLS(n_components=10, lambda_dyn=1.0, backend='jax')
model_jax.fit(X, y)
```

**Key Features:**
- sklearn-compatible API (fit/predict/transform/get_params/set_params)
- Koopman dynamics constraint for temporal coherence
- Pluggable featurizers (Identity, Polynomial, RBF)
- Warm start from standard PLS
- Dynamic prediction for future timesteps
- NumPy and JAX backends for CPU/GPU acceleration

**Tasks:**
- [x] Implement OKLM-PLS algorithm with alternating optimization
- [x] Create sklearn-compatible class with NumPy backend
- [x] Add JAX backend for GPU acceleration
- [x] Implement IdentityFeaturizer, PolynomialFeaturizer, RBFFeaturizer
- [x] Add predict_dynamic for future predictions
- [x] Add comprehensive unit tests
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestOKLMPLS`

**Note:** OKLM-PLS is designed for temporally-ordered data. For non-temporal data, set
lambda_dyn=0 to disable the dynamics constraint.

**Reference:**
- Brunton et al. (2021) - "Modern Koopman theory for dynamical systems"
- Williams et al. (2015) - "A data-driven approximation of the Koopman operator"

---

### Story 7.2: FCK-PLS (Fractional Convolutional Kernel PLS)

**Priority:** P3 - Specialized
**Status:** ✅ Implemented
**Difficulty:** High

**Description:**
Fractional Convolutional Kernel PLS builds spectral features by convolving input spectra
with a bank of fractional order filters, then applies PLS regression. The fractional
filters capture derivative-like information at various fractional orders (0, 0.5, 1, 1.5, 2),
which is useful for identifying different spectral signatures in NIRS data.

**Dependency:** None (pure Python/NumPy/SciPy, optional JAX for GPU)

**Operator:** `nirs4all/operators/models/sklearn/fckpls.py`

```python
from nirs4all.operators.models import FCKPLS, FractionalConvFeaturizer

# Basic FCK-PLS with default fractional orders
model = FCKPLS(
    n_components=10,
    alphas=(0.0, 0.5, 1.0, 1.5, 2.0),  # Fractional orders
    sigmas=(2.0,),                      # Filter scale
    kernel_size=15                       # Filter size
)
model.fit(X, y)
y_pred = model.predict(X)

# Custom fractional orders (smoothing + first + second derivative)
model_custom = FCKPLS(
    n_components=10,
    alphas=(0.0, 1.0, 2.0),
    sigmas=(3.0,),
    kernel_size=21
)
model_custom.fit(X, y)

# Get fractional features for analysis
X_feat = model.get_fractional_features(X)

# Get filter information
info = model.get_filter_info()
print(f"Number of kernels: {info['n_kernels']}")

# Grünwald-Letnikov kernels (more rigorous fractional derivative)
model_gl = FCKPLS(
    n_components=10,
    alphas=(0.0, 1.0),
    kernel_type='grunwald'
)
model_gl.fit(X, y)

# JAX backend for GPU acceleration
model_jax = FCKPLS(n_components=10, alphas=(0.0, 1.0, 2.0), backend='jax')
model_jax.fit(X, y)
```

**Key Features:**
- sklearn-compatible API (fit/predict/transform/get_params/set_params)
- Fractional order filter bank for spectral feature extraction
- Two kernel types: heuristic (Gaussian-modulated) and Grünwald-Letnikov
- Configurable filter scale (sigma) and size
- Access to intermediate fractional features
- NumPy and JAX backends for CPU/GPU acceleration

**Fractional Orders:**
- α ≈ 0: Smoothing (low-pass filtering)
- α ≈ 0.5: Half-derivative (intermediate behavior)
- α ≈ 1: First derivative (highlights slopes)
- α ≈ 1.5: Between first and second derivative
- α ≈ 2: Second derivative (highlights peaks/valleys)

**Tasks:**
- [x] Implement fractional kernel functions (heuristic and Grünwald-Letnikov)
- [x] Create FractionalConvFeaturizer transformer
- [x] Create sklearn-compatible FCKPLS class with NumPy backend
- [x] Add JAX backend for GPU acceleration
- [x] Add get_fractional_features for feature analysis
- [x] Add comprehensive unit tests
- [x] Add to Q19_pls_test.py example
- [x] Export from operators/models/__init__.py

**Test:** `examples/Q19_pls_test.py` and `tests/unit/operators/models/test_sklearn_pls.py::TestFCKPLS`

**Note:** FCK-PLS can be computationally expensive with many filters and large spectra.
The sigma parameter controls filter width - larger sigma captures broader features.
Cross-validation is recommended for tuning alphas and sigmas.

**Aliases:**
- `FCKPLS` - Main class name
- `FractionalPLS` - Alias

**Reference:**
- Podlubny (1999) - "Fractional differential equations"
- Chen et al. (2009) - "Fractional order control - A tutorial"

---

## Summary: Integration Priority

| Priority | Story | Method | Difficulty | Package |
|----------|-------|--------|------------|---------|
| P0 | 1.1 | PLSRegression | ✅ Trivial | sklearn |
| P0 | 1.2 | PLS-DA | ✅ Trivial | sklearn |
| P1 | 2.1 | IKPLS | 🟢 Low | ikpls |
| P1 | 2.2 | OPLS/OPLS-DA | 🟢 Low | pyopls |
| P2 | 2.3 | MB-PLS | 🟢 Low | mbpls |
| P2 | 2.5 | Sparse PLS | 🟢 Low | py-ddspls |
| P2 | 4.1-4.4 | VIP/MCUVE/CARS/SPA | 🟡 Medium | auswahl |
| P2 | 5.2 | iPLS | ✅ Implemented | Custom |
| P3 | 2.4 | DiPLS | 🟢 Low | trendfitter |
| P3 | 3.1 | LW-PLS | ✅ Implemented | Vendored (MIT) |
| P3 | 5.1 | SIMPLS | ✅ Implemented | Custom |
| P3 | 5.3 | Robust PLS | ✅ Implemented | Custom |
| P3 | 5.4 | Recursive PLS | ✅ Implemented | Custom |
| P4 | 6.1 | K-OPLS | ✅ Implemented | Custom |
| P3 | 6.2 | KernelPLS / NL-PLS | ✅ Implemented | Custom |
| P3 | 7.1 | OKLM-PLS | ✅ Implemented | Custom |
| P3 | 7.2 | FCK-PLS | ✅ Implemented | Custom |

---

## Dependency Summary

### Required Dependencies (Core)
```toml
[project.dependencies]
"ikpls>=1.0.0"      # IKPLS - fast PLS with JAX support
"pyopls>=0.1.0"     # OPLS/OPLS-DA
"auswahl>=0.9.0"    # VIP, MCUVE, CARS, SPA variable selection
```

### Optional Dependencies
```toml
[project.optional-dependencies]
pls = [
    "ikpls>=1.0.0",
    "pyopls>=0.1.0",
    "mbpls>=1.0.0",
    "py-ddspls>=0.1.0",
    "trendfitter>=0.1.0",
    "auswahl>=0.9.0",
]

pls-gpu = [
    "ikpls>=1.0.0",
    "jax>=0.4.10",
    "jaxlib>=0.4.10",
]
```

---

## Implementation Notes

### sklearn API Compatibility

All PLS operators should follow the sklearn estimator API:

```python
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class MyPLSModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=10, **kwargs):
        self.n_components = n_components
        # Store all params as attributes

    def fit(self, X, y):
        # Training logic
        # Store learned attributes with trailing underscore
        self.coef_ = ...
        self.x_loadings_ = ...
        return self

    def predict(self, X):
        # Prediction logic
        return y_pred

    def get_params(self, deep=True):
        return {'n_components': self.n_components, ...}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

### Attributes to expose (for compatibility with sklearn and interpretability):
- `coef_`: Regression coefficients
- `x_scores_`: X scores (latent variables)
- `x_loadings_`: X loadings
- `x_weights_`: X weights (for NIPALS/SIMPLS)
- `y_loadings_`: Y loadings
- `n_components_`: Actual number of components used

### Controller Compatibility

The `SklearnModelController` will automatically handle these models since they follow sklearn API.
No custom controller needed unless special handling is required.

---

## Testing Strategy

1. **Unit tests**: Test each PLS variant independently
2. **Integration tests**: Test within nirs4all pipeline
3. **Comparison tests**: Compare results with reference implementations (R packages)
4. **Benchmark tests**: Performance comparison on NIRS datasets

---

## References

- [sklearn PLSRegression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
- [ikpls documentation](https://ikpls.readthedocs.io/)
- [pyopls GitHub](https://github.com/BiRG/pyopls)
- [mbpls documentation](https://mbpls.readthedocs.io/)
- [auswahl documentation](https://auswahl.readthedocs.io/)
- [de Jong 1993 - SIMPLS](https://www.sciencedirect.com/science/article/abs/pii/016974399385002X)

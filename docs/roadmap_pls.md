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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Operator:** `nirs4all/operators/models/sklearn/pls.py`
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

**Operator:** `nirs4all/operators/models/sklearn/pls.py`
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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Operator:** `nirs4all/operators/models/sklearn/pls.py`
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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Operator:** `nirs4all/operators/models/sklearn/pls.py`
```python
from trendfitter import DiPLS

model = DiPLS(n_components=8, lags=5).fit(X_time, y_time)
y_pred = model.predict(X_time_val)
```

**Tasks:**
- [x] Add `trendfitter` to dependencies
- [x] Create `DiPLSOperator` wrapper
- [x] Add tests

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Operator:** `nirs4all/operators/models/sklearn/pls.py`
```python
from ddspls import ddsPLS

model = ddsPLS(n_components=5).fit(X, y)
y_pred = model.predict(X_val)
```

**Tasks:**
- [x] Add `py-ddspls` to dependencies
- [x] Create `SparsePLSOperator` wrapper
- [x] Add tests

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Test:** Add to `examples/Q18_pls_methods.py`

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

**Test:** Add to `examples/Q18_pls_methods.py`

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
- [ ] Implement SIMPLS algorithm from de Jong 1993 paper
- [ ] Create sklearn-compatible class
- [ ] Add unit tests comparing to PLSRegression
- [ ] Benchmark performance

**Test:** Add to `examples/Q18_pls_methods.py`

---

### Story 5.2: iPLS (Interval PLS)

**Priority:** P2 - High Value for NIRS
**Status:** 🟠 Hard
**Difficulty:** Medium

**Description:**
Grid or heuristic search over contiguous wavelength windows with local PLS models.

**Dependency:** None (uses sklearn internally)

**Implementation Plan:**
1. Slice X by intervals
2. Fit PLSRegression per window
3. Select best by CV or stack as ensemble
4. Optionally combine with CARS for interval seeds

**Operator:** `nirs4all/operators/models/sklearn/ipls.py`
```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import numpy as np

class IntervalPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5, n_intervals=10, interval_width=None):
        self.n_components = n_components
        self.n_intervals = n_intervals
        self.interval_width = interval_width

    def fit(self, X, y):
        # Evaluate each interval
        # Select best or combine
        ...
        return self

    def predict(self, X):
        ...
```

**Tasks:**
- [ ] Implement interval grid search logic
- [ ] Support forward/backward interval selection
- [ ] Support ensemble mode
- [ ] Add unit tests
- [ ] Benchmark on NIRS data

**Test:** Add to `examples/Q18_pls_methods.py`

---

### Story 5.3: Robust PLS / RSIMPLS

**Priority:** P3 - Specialized
**Status:** 🟠 Hard
**Difficulty:** High

**Description:**
Down-weight outliers using robust covariance estimation.

**Dependency:** None (pure Python/NumPy/scipy)

**Implementation Plan:**
1. Iterate PLS with robust weights (Huber/Tukey) on X and residuals
2. Compute robust covariance in SIMPLS loop
3. Reweight until convergence
4. Cross-validate weight tuning

**Operator:** `nirs4all/operators/models/sklearn/robust_pls.py`
```python
from sklearn.base import BaseEstimator, RegressorMixin

class RobustPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5, weighting='huber', max_iter=100):
        self.n_components = n_components
        self.weighting = weighting
        self.max_iter = max_iter

    def fit(self, X, y):
        # Implement robust PLS
        ...
        return self
```

**Tasks:**
- [ ] Research RSIMPLS algorithm details
- [ ] Implement robust weighting schemes
- [ ] Create sklearn-compatible class
- [ ] Add unit tests

**Test:** Add to `examples/Q18_pls_methods.py`

---

### Story 5.4: Recursive PLS (RPLS)

**Priority:** P3 - Specialized
**Status:** 🟠 Hard
**Difficulty:** Medium-High

**Description:**
Online model updates for drifting processes. Useful for process monitoring.

**Dependency:** None (pure Python/NumPy)

**Implementation Plan:**
1. Implement online PLS update equations
2. Support forgetting factor for drift adaptation
3. Maintain running statistics

**Operator:** `nirs4all/operators/models/sklearn/recursive_pls.py`
```python
from sklearn.base import BaseEstimator, RegressorMixin

class RecursivePLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5, forgetting_factor=0.99):
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor

    def fit(self, X, y):
        ...

    def partial_fit(self, X, y):
        # Online update
        ...
```

**Tasks:**
- [ ] Implement recursive update algorithm
- [ ] Add partial_fit support
- [ ] Add unit tests
- [ ] Test on streaming data

**Test:** Add to `examples/Q18_pls_methods.py`

---

## Tier 6: No Python Implementation 🔴

These methods have no maintained Python package and require significant effort.

### Story 6.1: K-OPLS (Kernel OPLS)

**Priority:** P4 - Low Priority
**Status:** 🔴 Very Hard
**Difficulty:** Very High

**Description:**
Kernelized OPLS. Benefit over KPLS for NIRS is uncertain.

**Recommendation:** Skip unless specific need arises. Prefer KPLS or OPLS separately.

**Implementation Notes:**
Would require porting MATLAB/R implementations to Python with proper kernel handling.

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
| P2 | 5.2 | iPLS | 🟠 Hard | Custom |
| P3 | 2.4 | DiPLS | 🟢 Low | trendfitter |
| P3 | 3.1 | LW-PLS | ✅ Implemented | Vendored (MIT) |
| P3 | 5.1 | SIMPLS | 🟠 Hard | Custom |
| P3 | 5.3 | Robust PLS | 🟠 Hard | Custom |
| P3 | 5.4 | Recursive PLS | 🟠 Hard | Custom |
| P4 | 6.1 | K-OPLS | 🔴 Very Hard | Custom |

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

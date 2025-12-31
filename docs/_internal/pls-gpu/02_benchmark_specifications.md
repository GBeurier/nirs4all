# pls-gpu: Benchmark and Validation Specifications

**Version:** 1.0.0-draft
**Date:** 2024-12-31
**Status:** Specification Draft

## 1. Overview

This document specifies the complete benchmark suite for validating pls-gpu implementations. The benchmarks serve three purposes:

1. **Correctness validation**: Verify numerical equivalence with reference implementations
2. **Performance comparison**: Measure CPU vs GPU speedups across backends
3. **Publication support**: Generate reproducible results for the accompanying paper

All benchmark scripts will be designed to run using nirs4all's pipeline infrastructure where possible.

## 2. Benchmark Categories

### 2.1 Numerical Equivalence Tests

Verify that pls-gpu produces identical (within tolerance) results to reference implementations.

**Equivalence testing MUST cover three dimensions:**

1. **Parameter Variations**: Test across different parameter configurations
   - `n_components`: [1, 2, 5, 10, 20, 50] (capped by data dimensions)
   - `scale`: [True, False]
   - `center`: [True, False]
   - `algorithm`: All supported algorithms per method
   - Method-specific parameters (e.g., `lambda_in_similarity` for LWPLS, `alpha` for SparsePLS)

2. **Dataset Variations**: Test across different data characteristics
   - Sample sizes: [30, 100, 500, 2000]
   - Feature sizes: [10, 50, 200, 1000]
   - Target types: single (PLS1), multiple (PLS2)
   - Data properties: normalized, non-normalized, with outliers, high collinearity
   - Edge cases: n_samples < n_features, n_samples >> n_features

3. **Numerical Stability**: Test boundary conditions
   - Near-zero variance features
   - Highly correlated features (multicollinearity)
   - Near-singular matrices
   - Mixed scale features (some 0.001, others 1000)

### 2.2 Backend Comparison Tests

Compare performance across NumPy, JAX, PyTorch, and TensorFlow backends.

### 2.3 Method Comparison Tests

Compare prediction accuracy across different PLS methods on real datasets.

### 2.4 Scalability Tests

Measure performance scaling with dataset dimensions.

### 2.5 Component Selection Validation

**Critical**: All method comparisons must use optimal component selection via cross-validation.
Do NOT compare methods with fixed, arbitrary component counts.

**Protocol for Fair Comparison**:

```python
def select_optimal_components(X, y, model_class, max_components=30, cv=5):
    """Select optimal n_components via inner CV (minimize RMSECV)."""
    from sklearn.model_selection import cross_val_score, KFold

    best_score = -np.inf
    best_n = 1

    for n_comp in range(1, min(max_components + 1, X.shape[1], X.shape[0])):
        model = model_class(n_components=n_comp)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_n = n_comp

    return best_n, -best_score
```

Each method comparison must report:
1. Optimal n_components for each method
2. How n_components was selected (CV folds, metric)
3. RMSECV curve showing the minimum

## 3. Reference Implementation Comparison

### 3.1 Reference Implementation Licensing

> **Important**: All reference implementations used for validation have their own licenses.
> This section documents the licensing of each reference package to ensure compliance.

| Reference Package | License | Usage in Benchmarks | Notes |
|------------------|---------|---------------------|-------|
| scikit-learn | BSD-3-Clause | Direct comparison | Permissive, no restrictions |
| ikpls | MIT | Direct comparison | Permissive, no restrictions |
| pyopls | MIT | Direct comparison | Permissive, no restrictions |
| mbpls | MIT | Direct comparison | Permissive, no restrictions |
| sparse-pls | BSD-3-Clause | Direct comparison | Permissive, no restrictions |
| R pls | GPL-2.0+ | Validation only | Comparison via rpy2 |
| R ropls | CeCILL (GPL-compat) | Validation only | Comparison via rpy2 |
| R mixOmics | GPL-3.0 | Validation only | Comparison via rpy2 |
| MATLAB plsregress | Commercial | Pre-computed .mat files | No runtime dependency |

**Benchmark Code Licensing**:
- All benchmark scripts in this repository are licensed under the same license as pls-gpu (MIT/BSD-3)
- Reference implementations are used as **external dependencies** for comparison, not copied
- No GPL-licensed code is included in pls-gpu; GPL packages are called via their public APIs

### 3.3 Python References

| Method | Reference Package | Installation |
|--------|------------------|--------------|
| NIPALS PLS | sklearn | `pip install scikit-learn` |
| SIMPLS | - | Manual implementation per de Jong 1993 |
| IKPLS | ikpls | `pip install ikpls` |
| OPLS | pyopls | `pip install pyopls` |
| MB-PLS | mbpls | `pip install mbpls` |
| LW-PLS | lwpls (GitHub) | Manual install |
| Sparse PLS | sparse-pls | `pip install sparse-pls` |

### 3.4 R References

| Method | Package | Function |
|--------|---------|----------|
| NIPALS | pls | `plsr(method="nipals")` |
| SIMPLS | pls | `plsr(method="simpls")` |
| Kernel PLS | pls | `plsr(method="kernelpls")` |
| OPLS/OPLS-DA | ropls | `opls()` |
| Sparse PLS | mixOmics | `spls()` |
| MB-PLS | ade4 | `mbpls()` |

### 3.5 MATLAB References

| Method | Toolbox/Function |
|--------|-----------------|
| NIPALS | Statistics Toolbox `plsregress` |
| SIMPLS | PLS_Toolbox (Eigenvector) |
| OPLS | Various published codes |

### 3.6 Comparison Script Structure

```python
# bench/comparison/compare_references.py

"""
Compare pls-gpu implementations against reference implementations.

For each method:
1. Generate synthetic data with known properties
2. Fit both pls-gpu and reference implementation
3. Compare: coefficients, scores, loadings, predictions
4. Report maximum deviation and pass/fail status
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

@dataclass
class ComparisonResult:
    method: str
    reference: str
    metric: str
    pls_gpu_value: np.ndarray
    reference_value: np.ndarray
    max_abs_diff: float
    max_rel_diff: float
    passed: bool
    tolerance: float

class ReferenceComparison:
    """Base class for reference implementation comparisons."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
        self.results: List[ComparisonResult] = []

    def compare_arrays(
        self,
        name: str,
        pls_gpu_arr: np.ndarray,
        ref_arr: np.ndarray
    ) -> ComparisonResult:
        """Compare two arrays and return comparison result."""
        max_abs = np.max(np.abs(pls_gpu_arr - ref_arr))
        max_rel = np.max(np.abs(pls_gpu_arr - ref_arr) /
                        (np.abs(ref_arr) + 1e-10))
        passed = np.allclose(pls_gpu_arr, ref_arr,
                            rtol=self.rtol, atol=self.atol)
        return ComparisonResult(
            method=self.method_name,
            reference=self.reference_name,
            metric=name,
            pls_gpu_value=pls_gpu_arr,
            reference_value=ref_arr,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
            passed=passed,
            tolerance=self.rtol
        )

    def run_comparison(self, X: np.ndarray, y: np.ndarray) -> List[ComparisonResult]:
        """Run full comparison. Override in subclasses."""
        raise NotImplementedError
```

### 3.7 Specific Comparison Tests

#### 3.7.0 Comprehensive Equivalence Test Suite

```python
# bench/comparison/comprehensive_equivalence.py

"""
Comprehensive equivalence testing with parameter and dataset variations.

This module ensures pls-gpu matches reference implementations across:
- Multiple parameter configurations
- Diverse dataset characteristics
- Edge cases and boundary conditions
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from itertools import product

@dataclass
class EquivalenceTestCase:
    """Single equivalence test configuration."""
    name: str
    method: str
    params: Dict[str, Any]
    dataset_config: Dict[str, Any]
    reference_impl: str
    tolerance: float = 1e-5

class ParameterGrid:
    """Parameter grids for comprehensive testing."""

    # Core PLS parameters
    PLS_PARAMS = {
        'n_components': [1, 2, 5, 10, 20],
        'scale': [True, False],
        'center': [True, False],
        'algorithm': ['nipals', 'simpls', 'ikpls1', 'ikpls2'],
    }

    # OPLS-specific parameters
    OPLS_PARAMS = {
        'n_components': [1, 2, 3, 5],  # Orthogonal components
        'pls_components': [1, 2, 5, 10],  # Predictive components
        'scale': [True, False],
    }

    # Kernel PLS parameters
    KPLS_PARAMS = {
        'n_components': [1, 5, 10, 20],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': [0.01, 0.1, 1.0, 'auto'],
        'degree': [2, 3, 4],  # For poly kernel
    }

    # Sparse PLS parameters
    SPLS_PARAMS = {
        'n_components': [1, 5, 10],
        'alpha': [0.01, 0.1, 0.5, 1.0],
        'scale': [True, False],
    }

    # LWPLS parameters
    LWPLS_PARAMS = {
        'n_components': [1, 5, 10],
        'lambda_in_similarity': [0.1, 0.5, 1.0, 2.0],
        'kernel': ['rbf', 'tricube'],
    }


class DatasetGenerator:
    """Generate test datasets with various characteristics."""

    @staticmethod
    def generate_suite() -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """Generate comprehensive test dataset suite."""
        datasets = {}

        # Vary sample sizes
        for n_samples in [30, 100, 500, 2000]:
            for n_features in [10, 50, 200, 500]:
                # Skip impractical combinations
                if n_samples < 30 and n_features > 200:
                    continue

                name = f"n{n_samples}_p{n_features}"
                X, y = DatasetGenerator._make_regression(n_samples, n_features)
                datasets[name] = (X, y, {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'type': 'standard'
                })

        # Edge case: n_samples < n_features (wide data)
        X, y = DatasetGenerator._make_regression(50, 500)
        datasets['wide_50x500'] = (X, y, {'type': 'wide'})

        # Edge case: n_samples >> n_features (tall data)
        X, y = DatasetGenerator._make_regression(5000, 20)
        datasets['tall_5000x20'] = (X, y, {'type': 'tall'})

        # Multi-target (PLS2)
        X, y = DatasetGenerator._make_regression(200, 100, n_targets=5)
        datasets['multitarget_5'] = (X, y, {'type': 'pls2', 'n_targets': 5})

        # High collinearity
        X, y = DatasetGenerator._make_collinear(200, 50, collinearity=0.98)
        datasets['high_collinearity'] = (X, y, {'type': 'collinear', 'r': 0.98})

        # Mixed scales (non-normalized)
        X, y = DatasetGenerator._make_mixed_scale(200, 50)
        datasets['mixed_scale'] = (X, y, {'type': 'mixed_scale'})

        # Near-zero variance columns
        X, y = DatasetGenerator._make_low_variance(200, 50, n_low_var=10)
        datasets['low_variance'] = (X, y, {'type': 'low_variance'})

        # With outliers
        X, y = DatasetGenerator._make_with_outliers(200, 50, outlier_frac=0.05)
        datasets['with_outliers'] = (X, y, {'type': 'outliers'})

        return datasets

    @staticmethod
    def _make_regression(n_samples, n_features, n_targets=1, seed=42):
        """Standard regression dataset."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        # y depends on first few features
        n_informative = min(10, n_features)
        coef = np.random.randn(n_informative, n_targets)
        y = X[:, :n_informative] @ coef + 0.1 * np.random.randn(n_samples, n_targets)
        if n_targets == 1:
            y = y.ravel()
        return X, y

    @staticmethod
    def _make_collinear(n_samples, n_features, collinearity=0.95, seed=42):
        """Dataset with high multicollinearity."""
        np.random.seed(seed)
        # Create base features
        n_base = n_features // 5
        X_base = np.random.randn(n_samples, n_base)
        # Create correlated features
        X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            base_idx = i % n_base
            noise = np.sqrt(1 - collinearity**2) * np.random.randn(n_samples)
            X[:, i] = collinearity * X_base[:, base_idx] + noise
        y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def _make_mixed_scale(n_samples, n_features, seed=42):
        """Dataset with mixed feature scales."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        # Apply different scales to different feature groups
        scales = np.array([0.001, 0.01, 1, 100, 1000])
        for i in range(n_features):
            X[:, i] *= scales[i % len(scales)]
        y = (X @ np.random.randn(n_features)) + 0.1 * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def _make_low_variance(n_samples, n_features, n_low_var=5, seed=42):
        """Dataset with some near-constant features."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        # Make some columns near-constant
        for i in range(n_low_var):
            X[:, i] = 0.5 + 1e-8 * np.random.randn(n_samples)
        y = X[:, n_low_var:n_low_var+5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def _make_with_outliers(n_samples, n_features, outlier_frac=0.05, seed=42):
        """Dataset with outliers."""
        np.random.seed(seed)
        X, y = DatasetGenerator._make_regression(n_samples, n_features, seed=seed)
        n_outliers = int(n_samples * outlier_frac)
        outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
        X[outlier_idx] += 10 * np.random.randn(n_outliers, n_features)
        y[outlier_idx] += 10 * np.random.randn(n_outliers)
        return X, y


class ComprehensiveEquivalenceTest:
    """Run comprehensive equivalence tests."""

    def __init__(self, rtol=1e-5, atol=1e-8):
        self.rtol = rtol
        self.atol = atol
        self.results = []

    def run_pls_equivalence(self):
        """Test PLSRegression equivalence across all configurations."""
        from sklearn.cross_decomposition import PLSRegression as SklearnPLS
        from pls_gpu import PLSRegression as PLSGpu

        datasets = DatasetGenerator.generate_suite()

        # Selected parameter combinations (full grid would be too large)
        param_combos = [
            {'n_components': 1, 'scale': True, 'algorithm': 'nipals'},
            {'n_components': 5, 'scale': True, 'algorithm': 'nipals'},
            {'n_components': 10, 'scale': True, 'algorithm': 'nipals'},
            {'n_components': 5, 'scale': False, 'algorithm': 'nipals'},
            {'n_components': 5, 'scale': True, 'algorithm': 'simpls'},
            {'n_components': 10, 'scale': True, 'algorithm': 'simpls'},
            {'n_components': 5, 'scale': True, 'algorithm': 'ikpls1'},
            {'n_components': 5, 'scale': True, 'algorithm': 'ikpls2'},
        ]

        for dataset_name, (X, y, meta) in datasets.items():
            for params in param_combos:
                # Skip if n_components too large for dataset
                max_comp = min(X.shape[0], X.shape[1]) - 1
                if params['n_components'] > max_comp:
                    continue

                result = self._compare_single(
                    method='PLSRegression',
                    dataset_name=dataset_name,
                    X=X, y=y,
                    pls_gpu_class=PLSGpu,
                    reference_class=SklearnPLS,
                    pls_gpu_params=params,
                    reference_params={
                        'n_components': params['n_components'],
                        'scale': params['scale']
                    }
                )
                self.results.append(result)

        return self.results

    def _compare_single(
        self,
        method: str,
        dataset_name: str,
        X: np.ndarray,
        y: np.ndarray,
        pls_gpu_class,
        reference_class,
        pls_gpu_params: Dict,
        reference_params: Dict
    ) -> Dict:
        """Run single comparison test."""
        try:
            # Fit reference
            ref_model = reference_class(**reference_params)
            ref_model.fit(X, y)
            ref_pred = ref_model.predict(X)

            # Fit pls-gpu
            gpu_model = pls_gpu_class(**pls_gpu_params, backend='numpy')
            gpu_model.fit(X, y)
            gpu_pred = gpu_model.predict(X)

            # Compare predictions (most critical)
            pred_match = np.allclose(gpu_pred, ref_pred, rtol=self.rtol, atol=self.atol)
            pred_max_diff = np.max(np.abs(gpu_pred - ref_pred))

            # Compare coefficients
            coef_match = np.allclose(
                gpu_model.coef_, ref_model.coef_,
                rtol=self.rtol, atol=self.atol
            )
            coef_max_diff = np.max(np.abs(gpu_model.coef_ - ref_model.coef_))

            return {
                'method': method,
                'dataset': dataset_name,
                'params': pls_gpu_params,
                'predictions_match': pred_match,
                'predictions_max_diff': pred_max_diff,
                'coefficients_match': coef_match,
                'coefficients_max_diff': coef_max_diff,
                'passed': pred_match and coef_match,
                'error': None
            }

        except Exception as e:
            return {
                'method': method,
                'dataset': dataset_name,
                'params': pls_gpu_params,
                'passed': False,
                'error': str(e)
            }

    def generate_report(self) -> str:
        """Generate equivalence test report."""
        lines = ["# Comprehensive Equivalence Test Report\n"]

        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passed', False))
        lines.append(f"**Total tests**: {total}")
        lines.append(f"**Passed**: {passed} ({100*passed/total:.1f}%)")
        lines.append(f"**Failed**: {total - passed}\n")

        # Failures detail
        failures = [r for r in self.results if not r.get('passed', False)]
        if failures:
            lines.append("## Failed Tests\n")
            for f in failures[:20]:  # Show first 20 failures
                lines.append(f"- **{f['method']}** on {f['dataset']}")
                lines.append(f"  Params: {f['params']}")
                if f.get('error'):
                    lines.append(f"  Error: {f['error']}")
                else:
                    lines.append(f"  Pred diff: {f.get('predictions_max_diff', 'N/A')}")

        return "\n".join(lines)
```

#### 3.7.1 sklearn PLSRegression Comparison

```python
# bench/comparison/compare_sklearn.py

from sklearn.cross_decomposition import PLSRegression as SklearnPLS
from pls_gpu import PLSRegression as PLSGpu

def compare_sklearn_pls(n_samples=100, n_features=50, n_components=10):
    """Compare pls-gpu NIPALS with sklearn PLSRegression."""

    # Generate reproducible data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = X[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)

    # Fit sklearn
    sklearn_pls = SklearnPLS(n_components=n_components, scale=True)
    sklearn_pls.fit(X, y)

    # Fit pls-gpu with NIPALS
    pls_gpu = PLSGpu(n_components=n_components, algorithm='nipals',
                     scale=True, backend='numpy')
    pls_gpu.fit(X, y)

    # Compare attributes
    comparisons = {
        'coef_': (pls_gpu.coef_, sklearn_pls.coef_),
        'x_weights_': (pls_gpu.x_weights_, sklearn_pls.x_weights_),
        'x_loadings_': (pls_gpu.x_loadings_, sklearn_pls.x_loadings_),
        'y_loadings_': (pls_gpu.y_loadings_, sklearn_pls.y_loadings_),
        'x_scores_': (pls_gpu.x_scores_, sklearn_pls.x_scores_),
        'predictions': (pls_gpu.predict(X), sklearn_pls.predict(X)),
    }

    results = []
    for name, (gpu_val, ref_val) in comparisons.items():
        # Handle sign ambiguity in loadings/scores
        if 'loadings' in name or 'weights' in name or 'scores' in name:
            # Try both signs
            diff1 = np.max(np.abs(gpu_val - ref_val))
            diff2 = np.max(np.abs(gpu_val + ref_val))
            if diff2 < diff1:
                gpu_val = -gpu_val

        max_diff = np.max(np.abs(gpu_val - ref_val))
        passed = max_diff < 1e-5
        results.append({
            'metric': name,
            'max_diff': max_diff,
            'passed': passed
        })

    return results
```

#### 3.7.2 R Package Comparison

```python
# bench/comparison/compare_r_pls.py

"""
Compare pls-gpu with R's pls package.

Requires:
- rpy2: pip install rpy2
- R pls package: install.packages("pls")
"""

import numpy as np

def setup_r_comparison():
    """Setup R environment for comparison."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    pls = importr('pls')
    return pls

def compare_r_simpls(X: np.ndarray, y: np.ndarray, n_components: int = 10):
    """Compare SIMPLS implementation with R's pls package."""
    pls_r = setup_r_comparison()

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    # Fit in R
    ro.r.assign('X', X)
    ro.r.assign('y', y)
    ro.r(f'''
        library(pls)
        model <- plsr(y ~ X, ncomp={n_components}, method="simpls",
                     scale=TRUE, center=TRUE)
        coef_r <- coef(model, ncomp={n_components})
        pred_r <- predict(model, X, ncomp={n_components})
    ''')

    coef_r = np.array(ro.r['coef_r'])
    pred_r = np.array(ro.r['pred_r']).ravel()

    # Fit pls-gpu
    from pls_gpu import PLSRegression
    pls_gpu = PLSRegression(n_components=n_components, algorithm='simpls',
                           scale=True, center=True, backend='numpy')
    pls_gpu.fit(X, y)
    pred_gpu = pls_gpu.predict(X)

    # Compare predictions (most important)
    pred_diff = np.max(np.abs(pred_gpu - pred_r))

    return {
        'prediction_max_diff': pred_diff,
        'passed': pred_diff < 1e-5
    }
```

#### 3.7.3 PLS-DA Classification Validation

```python
# bench/comparison/compare_plsda.py

"""
Validate PLS-DA and OPLS-DA classification on real datasets.

Must include classification-specific metrics:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC (binary) or macro-ROC-AUC (multiclass)
- Cross-validated classification accuracy
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pls_gpu.classification import PLSDA, OPLSDA

def validate_plsda(dataset_path: str):
    """Validate PLS-DA on a classification dataset."""
    from pls_gpu.utils import load_classification_dataset

    X, y_labels = load_classification_dataset(dataset_path)

    # Use stratified k-fold for classification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for model_class, name in [(PLSDA, 'PLS-DA'), (OPLSDA, 'OPLS-DA')]:
        y_pred_all = np.zeros_like(y_labels)
        y_proba_all = np.zeros((len(y_labels), len(np.unique(y_labels))))

        for train_idx, test_idx in cv.split(X, y_labels):
            model = model_class(n_components=5)  # Optimize separately
            model.fit(X[train_idx], y_labels[train_idx])
            y_pred_all[test_idx] = model.predict(X[test_idx])
            y_proba_all[test_idx] = model.predict_proba(X[test_idx])

        results.append({
            'method': name,
            'accuracy': accuracy_score(y_labels, y_pred_all),
            'report': classification_report(y_labels, y_pred_all),
            # For binary: roc_auc_score(y_labels, y_proba_all[:, 1])
            # For multiclass: roc_auc_score(y_labels, y_proba_all, multi_class='ovr')
        })

    return results
```

**Classification Datasets for Validation**:

| Dataset | Domain | n | p | Classes | Source |
|---------|--------|---|---|---------|--------|
| Wine | Authentication | 178 | 13 | 3 | UCI |
| Olive Oil | Authentication | 572 | 8 | 9 regions | UCI |
| Breast Cancer | Diagnosis | 569 | 30 | 2 | sklearn |
| Spectral Authenticity | NIR | 200+ | 1000+ | 2 (auth/fraud) | Synthetic or literature |

#### 3.7.4 MATLAB Comparison

```python
# bench/comparison/compare_matlab.py

"""
Compare pls-gpu with MATLAB's plsregress.

Approach: Pre-compute MATLAB results and store as .mat files.
"""

import numpy as np
from scipy.io import loadmat

def load_matlab_reference(dataset_name: str) -> dict:
    """Load pre-computed MATLAB results."""
    data = loadmat(f'bench/reference_data/matlab/{dataset_name}.mat')
    return {
        'X': data['X'],
        'y': data['y'].ravel(),
        'coef': data['coef'],
        'predictions': data['predictions'].ravel(),
        'n_components': int(data['n_components'])
    }

def compare_matlab_pls(dataset_name: str):
    """Compare pls-gpu with MATLAB plsregress results."""
    ref = load_matlab_reference(dataset_name)

    from pls_gpu import PLSRegression
    pls_gpu = PLSRegression(n_components=ref['n_components'],
                           algorithm='nipals', backend='numpy')
    pls_gpu.fit(ref['X'], ref['y'])
    pred_gpu = pls_gpu.predict(ref['X'])

    return {
        'prediction_max_diff': np.max(np.abs(pred_gpu - ref['predictions'])),
        'coef_max_diff': np.max(np.abs(pls_gpu.coef_.ravel() - ref['coef'].ravel()))
    }
```

## 4. Backend Performance Comparison

### 4.1 Benchmark Script

```python
# bench/performance/backend_comparison.py

"""
Compare performance across NumPy, JAX, PyTorch, and TensorFlow backends.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class TimingResult:
    backend: str
    method: str
    n_samples: int
    n_features: int
    n_components: int
    fit_time: float
    predict_time: float
    total_time: float
    speedup_vs_numpy: float

class BackendBenchmark:
    """Benchmark PLS across backends."""

    BACKENDS = ['numpy', 'jax', 'torch', 'tensorflow']

    def __init__(self, warmup_runs: int = 3, timed_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.results: List[TimingResult] = []

    def check_backend_available(self, backend: str) -> bool:
        """Check if backend is available."""
        try:
            if backend == 'jax':
                import jax
            elif backend == 'torch':
                import torch
            elif backend == 'tensorflow':
                import tensorflow
            return True
        except ImportError:
            return False

    def benchmark_single(
        self,
        backend: str,
        method: str,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int
    ) -> Dict:
        """Benchmark a single configuration."""
        from pls_gpu import PLSRegression

        # Create model
        model = PLSRegression(
            n_components=n_components,
            algorithm=method,
            backend=backend
        )

        # Warmup
        for _ in range(self.warmup_runs):
            model.fit(X, y)
            _ = model.predict(X)

        # Timed runs
        fit_times = []
        predict_times = []

        for _ in range(self.timed_runs):
            # Fit timing
            start = time.perf_counter()
            model.fit(X, y)
            fit_times.append(time.perf_counter() - start)

            # Predict timing
            start = time.perf_counter()
            _ = model.predict(X)
            predict_times.append(time.perf_counter() - start)

        return {
            'fit_time': np.median(fit_times),
            'predict_time': np.median(predict_times),
            'total_time': np.median(fit_times) + np.median(predict_times)
        }

    def run_benchmark_suite(
        self,
        methods: List[str] = ['simpls'],
        sample_sizes: List[int] = [100, 500, 1000, 5000, 10000],
        feature_sizes: List[int] = [50, 200, 500, 1000, 2000],
        n_components: int = 10
    ) -> List[TimingResult]:
        """Run full benchmark suite."""

        results = []
        numpy_times = {}  # Cache numpy times for speedup calculation

        for n_samples in sample_sizes:
            for n_features in feature_sizes:
                # Generate data
                np.random.seed(42)
                X = np.random.randn(n_samples, n_features)
                y = np.random.randn(n_samples)

                for method in methods:
                    for backend in self.BACKENDS:
                        if not self.check_backend_available(backend):
                            continue

                        try:
                            timing = self.benchmark_single(
                                backend, method, X, y, n_components
                            )

                            # Calculate speedup vs numpy
                            key = (n_samples, n_features, method)
                            if backend == 'numpy':
                                numpy_times[key] = timing['total_time']
                                speedup = 1.0
                            else:
                                numpy_time = numpy_times.get(key, timing['total_time'])
                                speedup = numpy_time / timing['total_time']

                            result = TimingResult(
                                backend=backend,
                                method=method,
                                n_samples=n_samples,
                                n_features=n_features,
                                n_components=n_components,
                                fit_time=timing['fit_time'],
                                predict_time=timing['predict_time'],
                                total_time=timing['total_time'],
                                speedup_vs_numpy=speedup
                            )
                            results.append(result)

                        except Exception as e:
                            print(f"Error with {backend}/{method}: {e}")

        self.results = results
        return results

    def save_results(self, path: str):
        """Save results to JSON."""
        data = [
            {
                'backend': r.backend,
                'method': r.method,
                'n_samples': r.n_samples,
                'n_features': r.n_features,
                'n_components': r.n_components,
                'fit_time': r.fit_time,
                'predict_time': r.predict_time,
                'total_time': r.total_time,
                'speedup_vs_numpy': r.speedup_vs_numpy
            }
            for r in self.results
        ]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = ["# Backend Performance Comparison\n"]

        # Group by sample/feature size
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in self.results:
            grouped[(r.n_samples, r.n_features)].append(r)

        for (n_samples, n_features), results in grouped.items():
            lines.append(f"\n## {n_samples} samples × {n_features} features\n")
            lines.append("| Backend | Method | Fit (s) | Predict (s) | Speedup |")
            lines.append("|---------|--------|---------|-------------|---------|")

            for r in sorted(results, key=lambda x: (x.method, x.backend)):
                lines.append(
                    f"| {r.backend} | {r.method} | "
                    f"{r.fit_time:.4f} | {r.predict_time:.4f} | "
                    f"{r.speedup_vs_numpy:.2f}x |"
                )

        return "\n".join(lines)
```

### 4.2 GPU vs CPU Comparison

```python
# bench/performance/gpu_vs_cpu.py

"""
Specific comparison of GPU vs CPU performance.
"""

import numpy as np
import time

def benchmark_gpu_scaling():
    """Benchmark GPU speedup as data size increases."""

    sample_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    feature_size = 1000
    n_components = 20

    results = []

    for n_samples in sample_sizes:
        np.random.seed(42)
        X = np.random.randn(n_samples, feature_size)
        y = np.random.randn(n_samples)

        # NumPy timing
        from pls_gpu import PLSRegression

        pls_numpy = PLSRegression(n_components=n_components, backend='numpy')
        start = time.perf_counter()
        pls_numpy.fit(X, y)
        numpy_time = time.perf_counter() - start

        # JAX timing (if available)
        try:
            pls_jax = PLSRegression(n_components=n_components, backend='jax')
            # Warmup
            pls_jax.fit(X[:100], y[:100])

            start = time.perf_counter()
            pls_jax.fit(X, y)
            jax_time = time.perf_counter() - start
            jax_speedup = numpy_time / jax_time
        except:
            jax_time = None
            jax_speedup = None

        # PyTorch timing (if available)
        try:
            pls_torch = PLSRegression(n_components=n_components, backend='torch')
            pls_torch.fit(X[:100], y[:100])

            start = time.perf_counter()
            pls_torch.fit(X, y)
            torch_time = time.perf_counter() - start
            torch_speedup = numpy_time / torch_time
        except:
            torch_time = None
            torch_speedup = None

        results.append({
            'n_samples': n_samples,
            'n_features': feature_size,
            'numpy_time': numpy_time,
            'jax_time': jax_time,
            'jax_speedup': jax_speedup,
            'torch_time': torch_time,
            'torch_speedup': torch_speedup
        })

    return results
```

## 5. Method Comparison on Real Datasets

### 5.1 Dataset Collection

The benchmark uses ~30 contrasted real-world datasets covering:

| Category | Datasets | Samples | Features |
|----------|----------|---------|----------|
| NIR Spectroscopy | Corn, Meat, Tablets, Gasoline | 80-500 | 200-1500 |
| Metabolomics | Breast cancer, Plasma | 100-300 | 500-5000 |
| Process Chemistry | Propylene, Ethanol | 200-1000 | 100-500 |
| Agriculture | Soil, Wheat, Coffee | 100-400 | 400-2000 |
| Food Science | Wine, Olive oil, Honey | 100-300 | 200-1000 |

### 5.2 nirs4all Pipeline for Comparison

```python
# bench/methods/compare_methods_nirs4all.py

"""
Compare PLS methods using nirs4all pipelines.
"""

from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs
from sklearn.model_selection import RepeatedKFold

def create_method_comparison_pipeline():
    """Create pipeline comparing multiple PLS methods."""

    from pls_gpu import PLSRegression, OPLS, KernelPLS, SparsePLS, LWPLS
    from sklearn.preprocessing import StandardScaler

    # Define methods to compare
    methods = {
        'PLS_NIPALS': PLSRegression(n_components=10, algorithm='nipals'),
        'PLS_SIMPLS': PLSRegression(n_components=10, algorithm='simpls'),
        'PLS_IKPLS': PLSRegression(n_components=10, algorithm='ikpls1'),
        'OPLS_1': OPLS(n_components=1, pls_components=5),
        'KernelPLS_RBF': KernelPLS(n_components=10, kernel='rbf'),
        'SparsePLS': SparsePLS(n_components=10, alpha=0.1),
        'LWPLS': LWPLS(n_components=10, lambda_in_similarity=0.5),
    }

    results = {}

    for name, model in methods.items():
        pipeline = [
            StandardScaler(),
            RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
            {"model": model}
        ]
        results[name] = pipeline

    return results

def run_method_comparison(dataset_path: str):
    """Run method comparison on a dataset."""
    from nirs4all.pipeline import PipelineRunner, PipelineConfigs
    from nirs4all.data import DatasetConfigs

    runner = PipelineRunner(verbose=1)

    pipelines = create_method_comparison_pipeline()
    all_results = {}

    for name, pipeline in pipelines.items():
        config = PipelineConfigs(pipeline, name)
        data_config = DatasetConfigs(dataset_path)

        predictions, per_dataset = runner.run(config, data_config)

        all_results[name] = {
            'rmse': predictions.rmse,
            'r2': predictions.r2,
            'predictions': predictions.y_pred
        }

    return all_results
```

### 5.3 Comprehensive Dataset Benchmark

```python
# bench/methods/full_benchmark.py

"""
Full benchmark across all datasets and methods.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

DATASET_PATHS = {
    # NIR Spectroscopy (from nirs4all bench/_datasets/)
    'hiba': 'bench/_datasets/hiba/',           # Real NIR dataset
    'redox': 'bench/_datasets/redox/',         # Redox chemistry dataset

    # Public NIR datasets
    'corn': 'bench/_datasets/corn/',
    'meat': 'bench/_datasets/meat/',
    'tablets': 'bench/_datasets/tablets/',

    # Additional datasets to download
    'tecator': 'bench/_datasets/tecator/',     # StatLib
    'octane': 'bench/_datasets/octane/',       # R pls package
    'gasoline': 'bench/_datasets/gasoline/',   # R pls package

    # Large-scale datasets
    'lucas_soil': 'bench/_datasets/lucas/',    # LUCAS soil spectral library
}

def run_full_benchmark(
    output_dir: str = 'bench/results/',
    methods: List[str] = None,
    datasets: List[str] = None
):
    """Run full benchmark suite."""

    results = []

    for dataset_name, dataset_path in DATASET_PATHS.items():
        if datasets and dataset_name not in datasets:
            continue

        print(f"\nProcessing {dataset_name}...")

        try:
            method_results = run_method_comparison(dataset_path)

            for method_name, metrics in method_results.items():
                results.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'mae': metrics.get('mae', None)
                })
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/method_comparison.csv', index=False)

    # Generate summary
    summary = df.groupby('method').agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).round(4)

    summary.to_csv(f'{output_dir}/method_summary.csv')

    return df, summary
```

## 6. Synthetic Data Benchmarks

### 6.1 Synthetic Data Generation using nirs4all

```python
# bench/synthetic/generate_data.py

"""
Generate synthetic datasets with controlled properties using nirs4all.
"""

from nirs4all.data.synthetic import SyntheticDataGenerator

def generate_benchmark_datasets():
    """Generate suite of synthetic datasets for benchmarking."""

    configs = [
        # Varying complexity
        {
            'name': 'simple_linear',
            'n_samples': 1000,
            'n_features': 100,
            'n_informative': 10,
            'noise_level': 0.1,
            'nonlinearity': 0.0
        },
        {
            'name': 'moderate_nonlinear',
            'n_samples': 1000,
            'n_features': 500,
            'n_informative': 20,
            'noise_level': 0.2,
            'nonlinearity': 0.3
        },
        {
            'name': 'complex_nonlinear',
            'n_samples': 1000,
            'n_features': 1000,
            'n_informative': 30,
            'noise_level': 0.3,
            'nonlinearity': 0.7
        },

        # Varying sample sizes
        {
            'name': 'small_sample',
            'n_samples': 100,
            'n_features': 500,
            'n_informative': 20,
            'noise_level': 0.2
        },
        {
            'name': 'large_sample',
            'n_samples': 10000,
            'n_features': 500,
            'n_informative': 20,
            'noise_level': 0.2
        },

        # High dimensionality
        {
            'name': 'high_dim',
            'n_samples': 500,
            'n_features': 5000,
            'n_informative': 50,
            'noise_level': 0.2
        },

        # Multicollinearity
        {
            'name': 'high_collinearity',
            'n_samples': 500,
            'n_features': 200,
            'n_informative': 20,
            'collinearity': 0.95
        },

        # Multi-target
        {
            'name': 'multi_target',
            'n_samples': 500,
            'n_features': 200,
            'n_targets': 5,
            'n_informative': 20
        }
    ]

    datasets = {}
    for config in configs:
        name = config.pop('name')
        gen = SyntheticDataGenerator(**config)
        X, y = gen.generate()
        datasets[name] = {'X': X, 'y': y, 'config': config}

    return datasets
```

### 6.2 Controlled Comparison Tests

```python
# bench/synthetic/controlled_tests.py

"""
Controlled tests with known ground truth.
"""

import numpy as np

def test_perfect_pls_case():
    """
    Generate data where PLS should achieve perfect prediction.
    Y = X @ true_coef + small_noise
    """
    np.random.seed(42)
    n_samples, n_features = 200, 50
    n_components_true = 5

    # Generate low-rank X
    T_true = np.random.randn(n_samples, n_components_true)
    P_true = np.random.randn(n_features, n_components_true)
    X = T_true @ P_true.T

    # Y is linear combination of same latent variables
    Q_true = np.random.randn(n_components_true, 1)
    y = (T_true @ Q_true).ravel()

    # Add small noise
    y += 0.01 * np.random.randn(n_samples)

    return X, y, n_components_true

def test_orthogonal_variation():
    """
    Generate data with Y-orthogonal variation for OPLS testing.
    """
    np.random.seed(42)
    n_samples, n_features = 200, 50

    # Predictive variation
    T_pred = np.random.randn(n_samples, 3)
    P_pred = np.random.randn(n_features, 3)

    # Orthogonal variation (uncorrelated with Y)
    T_orth = np.random.randn(n_samples, 5)
    P_orth = np.random.randn(n_features, 5)

    # Ensure orthogonality
    T_orth = T_orth - T_pred @ (np.linalg.pinv(T_pred) @ T_orth)

    # Combine
    X = T_pred @ P_pred.T + 2 * T_orth @ P_orth.T

    # Y depends only on predictive part
    y = T_pred[:, 0] + 0.5 * T_pred[:, 1] + 0.1 * np.random.randn(n_samples)

    return X, y, {'n_pred': 3, 'n_orth': 5}
```

## 7. Output Formats and Reporting

### 7.1 Results Directory Structure

> **Separation of Examples and Publication Code**: User-facing examples are in `examples/`.
> Publication-specific code (benchmarks, comparisons, figure generation) is in `publication/`.
> This separation ensures reproducibility of paper results while keeping examples simple.

```
pls-gpu/
├── examples/                     # User-facing examples (NOT publication code)
│   ├── getting_started/
│   │   ├── 01_basic_pls.py           # Simple standalone example
│   │   ├── 02_backend_selection.py   # How to switch backends
│   │   └── 03_cross_validation.py    # CV example
│   ├── sklearn_integration/
│   │   ├── 01_pipeline.py            # sklearn Pipeline integration
│   │   ├── 02_gridsearch.py          # GridSearchCV with pls-gpu
│   │   └── 03_model_selection.py     # cross_val_score example
│   ├── nirs4all_integration/
│   │   ├── 01_nirs4all_pipeline.py   # nirs4all pipeline example
│   │   ├── 02_comparison.py          # Compare with nirs4all operators
│   │   └── 03_advanced.py            # Advanced nirs4all usage
│   ├── backend_specific/
│   │   ├── jax_example.py            # JAX-specific features
│   │   ├── torch_example.py          # PyTorch-specific features
│   │   └── tensorflow_example.py     # TensorFlow-specific features
│   ├── methods/
│   │   ├── 01_opls.py                # OPLS example
│   │   ├── 02_sparse_pls.py          # Sparse PLS example
│   │   ├── 03_kernel_pls.py          # Kernel PLS example
│   │   └── 04_plsda.py               # PLS-DA classification example
│   └── notebooks/
│       ├── tutorial.ipynb            # Interactive tutorial
│       └── visualization.ipynb       # VIP, loadings plots
│
├── publication/                  # Publication-specific code (reproducible)
│   ├── README.md                     # How to reproduce paper results
│   ├── paper/
│   │   ├── manuscript.md             # Paper draft (or LaTeX)
│   │   ├── figures/                  # Publication figures
│   │   ├── tables/                   # Publication tables
│   │   └── supplementary/            # Supplementary material
│   ├── benchmarks/
│   │   ├── run_all.py                # Master script to run all benchmarks
│   │   ├── equivalence/
│   │   │   ├── compare_sklearn.py
│   │   │   ├── compare_r_pls.py
│   │   │   ├── compare_plsda.py
│   │   │   └── compare_matlab.py
│   │   ├── performance/
│   │   │   ├── backend_comparison.py
│   │   │   └── gpu_vs_cpu.py
│   │   ├── methods/
│   │   │   ├── full_benchmark.py
│   │   │   └── method_comparison.py
│   │   └── synthetic/
│   │       ├── generate_data.py
│   │       └── controlled_tests.py
│   ├── datasets/                     # Datasets used in publication
│   │   ├── README.md                 # Dataset sources, licenses
│   │   └── download.py               # Script to download datasets
│   ├── results/                      # Generated results (git-ignored)
│   │   ├── equivalence/
│   │   ├── performance/
│   │   ├── methods/
│   │   └── synthetic/
│   └── generate_figures.py           # Generate all paper figures
```

> **Reproducibility**: Every script in `publication/benchmarks/` must be:
> 1. Self-contained and documented
> 2. Use fixed random seeds
> 3. Log hardware configuration
> 4. Output results in structured format (JSON/CSV)
> 5. Be runnable with a single command

### 7.2 Report Generation

```python
# bench/reporting/generate_reports.py

"""
Generate publication-ready reports and figures.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_speedup_heatmap(results_path: str, output_path: str):
    """Generate heatmap of GPU speedups."""

    with open(results_path) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Pivot for heatmap
    pivot = df.pivot_table(
        values='speedup_vs_numpy',
        index='n_samples',
        columns='n_features',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax)
    ax.set_title('GPU Speedup vs NumPy (JAX backend)')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Number of Samples')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_method_comparison_plot(results_path: str, output_path: str):
    """Generate method comparison bar chart."""

    df = pd.read_csv(results_path)

    # Calculate mean R² per method
    summary = df.groupby('method')['r2'].agg(['mean', 'std']).sort_values('mean')

    fig, ax = plt.subplots(figsize=(12, 6))
    summary['mean'].plot(kind='barh', xerr=summary['std'], ax=ax, capsize=3)
    ax.set_xlabel('R² Score')
    ax.set_title('PLS Method Comparison Across Datasets')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

## 8. Continuous Integration

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/benchmarks.yml

name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly

jobs:
  equivalence-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev,jax]"
          pip install scikit-learn ikpls pyopls
      - name: Run equivalence tests
        run: pytest bench/comparison/ -v --tb=short
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: equivalence-results
          path: bench/results/equivalence/

  performance-tests:
    runs-on: ubuntu-latest
    needs: equivalence-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev,jax,torch]"
      - name: Run performance benchmarks
        run: python bench/performance/backend_comparison.py
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: performance-results
          path: bench/results/performance/

  gpu-benchmarks:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    needs: equivalence-tests
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU benchmarks
        run: python bench/performance/gpu_vs_cpu.py
```

## 9. Summary of Benchmark Scripts

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `compare_sklearn.py` | Validate vs sklearn | sklearn |
| `compare_r_pls.py` | Validate vs R pls | rpy2, R |
| `compare_matlab.py` | Validate vs MATLAB | scipy (for .mat files) |
| `backend_comparison.py` | Benchmark all backends | jax, torch, tf (optional) |
| `gpu_vs_cpu.py` | GPU speedup analysis | jax or torch |
| `compare_methods_nirs4all.py` | Method comparison | nirs4all |
| `full_benchmark.py` | Complete benchmark suite | all |
| `generate_data.py` | Synthetic data generation | nirs4all |
| `generate_reports.py` | Report generation | matplotlib, seaborn |

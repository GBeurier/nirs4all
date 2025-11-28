# Preprocessing Selection Framework

## Overview

This framework provides a **systematic 4-stage approach** to filter and rank preprocessing techniques before running full ML/DL pipelines. The goal is to reduce the preprocessing search space by 10-20× without losing performance, saving 80-95% of exploration time.

The framework:
- **Stage 1**: Exhaustive unsupervised evaluation of all pipeline combinations
- **Stage 2**: Diversity analysis with 6 distance metrics (Grassmann, CKA, RV, Procrustes, Trustworthiness, Covariance)
- **Stage 3**: Proxy model validation (Ridge + KNN) on diverse candidates
- **Stage 4**: Feature augmentation evaluation (2-way and 3-way concatenations)

---

## Quick Start

```bash
# Run with default parameters
python run_selection.py

# Quick test run (minimal, ~1 min)
python run_selection.py --depth 2 --top-stage1 5 --top-stage2 5 --top-stage3 5 --top-stage4 5 --top-final 10 --aug-order 2

# Full evaluation with plots
python run_selection.py --depth 3 --plots
```

---

## Command Line Interface

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth` | 3 | Maximum pipeline stacking depth (1-4) |
| `--top-stage1` | 15 | Number of top candidates from Stage 1 to pass to Stage 2 |
| `--top-stage2` | 15 | Number of diverse candidates from Stage 2 to pass to Stage 3 |
| `--top-stage3` | 15 | Number of top candidates from Stage 3 proxy evaluation |
| `--top-stage4` | 15 | Number of top augmentations from Stage 4 |
| `--top-final` | 30 | Number of final configurations to return |
| `--similarity-ratio` | 0.95 | Similarity threshold for diversity filtering (0-1). Higher = more strict |
| `--aug-order` | 3 | Maximum augmentation order (2=2-way only, 3=2-way and 3-way) |
| `--plots` | False | Show visualization plots |
| `--output` | selection | Output directory |
| `--data` | None | Custom data path |
| `--full` | False | Use full nitro dataset |

### Usage Examples

```bash
# Minimal test (fast, ~1 min)
python run_selection.py --depth 2 --top-stage1 5 --top-stage2 5 --top-stage3 5 --top-stage4 5 --top-final 10 --aug-order 2

# Standard run (balanced, ~5 min)
python run_selection.py --depth 3 --top-stage1 15 --top-stage2 15 --top-stage3 15 --top-stage4 15 --top-final 30

# Comprehensive run (thorough, ~15 min)
python run_selection.py --depth 4 --top-stage1 30 --top-stage2 25 --top-stage3 20 --top-stage4 20 --top-final 50 --aug-order 3

# High diversity focus (more diverse but smaller set)
python run_selection.py --depth 3 --similarity-ratio 0.90 --top-stage1 20 --top-stage2 10

# Custom data with visualization
python run_selection.py --data path/to/my/dataset --output my_results --plots

# Depth 4 with 50 top candidates (as run previously)
python run_selection.py --depth 4 --top-stage1 50 --top-stage2 50 --top-stage3 50 --top-stage4 50 --top-final 100
```

---

## Output Files

| File | Description |
|------|-------------|
| `stage1_unsupervised.csv` | All pipelines with unsupervised metrics |
| `stage2_diversity.csv` | Diversity metrics for top candidates |
| `distance_matrix_combined.csv` | Pairwise combined distances |
| `distance_matrix_subspace.csv` | Pairwise subspace-based distances (Grassmann + CKA + RV) |
| `distance_matrix_geometry.csv` | Pairwise geometry-based distances (Procrustes + Trust + Cov) |
| `stage3_proxy.csv` | Proxy model results for diverse candidates |
| `stage4_augmentation.csv` | Augmentation evaluation results |
| `final_ranking.csv` | Combined ranking of all configurations |
| `systematic_results.png` | Multi-panel visualization |

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: EXHAUSTIVE UNSUPERVISED             │
│  Generate all single + stacked pipelines (depth 1..max_depth)   │
│  Compute: variance_ratio, effective_dim, SNR, roughness, sep    │
│  Output: top_stage1 candidates by total_score                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: DIVERSITY ANALYSIS                   │
│  Compute 6 pairwise distance metrics:                           │
│    Subspace: Grassmann, CKA, RV                                 │
│    Geometry: Procrustes, Trustworthiness, Covariance            │
│  Filter similar pipelines (similarity_ratio threshold)          │
│  Output: top_stage2 diverse candidates                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 3: PROXY MODEL EVALUATION                │
│  Evaluate each candidate with Ridge + KNN (3-fold CV)           │
│  Compute: ridge_r2, knn_score, proxy_score                      │
│  Combine: final_score = 0.4*unsupervised + 0.6*proxy            │
│  Output: top_stage3 best performers                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 4: AUGMENTATION EVALUATION                │
│  Generate 2-way augmentations: [prep1 + prep2]                  │
│  Generate 3-way augmentations: [prep1 + prep2 + prep3]          │
│  Evaluate each with unsupervised metrics + proxy models         │
│  Output: top_stage4 best augmentations                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       FINAL RANKING                              │
│  Combine Stage 3 + Stage 4 results                              │
│  Sort by final_score                                            │
│  Output: top_final configurations                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Using nirs4all for Preprocessing-Only Pipelines

In nirs4all, you can run a pipeline **without models or splits** to only apply preprocessings and retrieve the transformed data. This is essential for evaluating preprocessing quality.

### Example: Preprocessing-Only Pipeline

```python
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import (
    StandardNormalVariate, SavitzkyGolay, MultiplicativeScatterCorrection,
    FirstDerivative, SecondDerivative, Haar, Detrend, Gaussian
)

# Define preprocessings to evaluate
preprocessings = [
    StandardNormalVariate(),
    SavitzkyGolay(window_length=11, polyorder=3),
    MultiplicativeScatterCorrection(),
    FirstDerivative(),
    SecondDerivative(),
    Haar(),
    Detrend(),
    Gaussian(order=1, sigma=2),
]

# Load your data
dataset_config = DatasetConfigs([{
    'folder': 'sample_data/regression/',
    'params': {'has_header': False, 'delimiter': ';', 'decimal_separator': '.'}
}])

# Run each preprocessing and collect transformed data
runner = PipelineRunner(save_files=False, verbose=0, keep_datasets=True)

for pp in preprocessings:
    pipeline = [pp]  # No model, no split - just preprocessing
    pipeline_config = PipelineConfigs(pipeline, f"eval_{pp.__class__.__name__}")

    predictions, _ = runner.run(pipeline_config, dataset_config)

    # Access preprocessed data from runner.pp_data
    # Structure: {dataset_name: {'X': X_preprocessed, 'y': y}}
    for ds_name, data in runner.pp_data.items():
        X_preprocessed = data['X']
        y = data['y']
        # Now apply selection metrics...
```

### Using Feature Augmentation

For testing preprocessing combinations with feature augmentation:

```python
pipeline = [
    {"feature_augmentation": [
        StandardNormalVariate,
        SavitzkyGolay,
        FirstDerivative,
    ]}
]
# This creates multiple transformed versions concatenated as features
```

---

## Selection Methods Specification

### Stage A: Unsupervised Filtering (30-50% elimination)

These methods require **only X data** and filter out obviously poor preprocessings.

---

#### A1. PCA Variance Preserved

**Name:** `pca_variance_filter`

**Description:**
Computes PCA on each preprocessed X and measures cumulative explained variance. Eliminates preprocessings that:
- Destroy too much information (variance << others)
- Produce artifacts (variance concentrated on 1 component)

**Speed:** ~50× faster than model testing

**Inputs:**
- `X_preprocessed`: Transformed spectra (n_samples, n_features)
- `n_components`: Number of PCA components to consider (default: min(10, n_features))
- `min_variance_ratio`: Minimum cumulative variance to keep (default: 0.90)
- `max_first_component_ratio`: Maximum variance for 1st component (default: 0.99)

**Outputs:**
- `variance_score`: Cumulative explained variance (0-1)
- `is_valid`: Boolean (True if passes filters)
- `reason`: String explaining elimination reason if invalid

**Implementation Spec:**
```python
from sklearn.decomposition import PCA
import numpy as np

def pca_variance_filter(X_preprocessed, n_components=10, min_variance_ratio=0.90, max_first_component_ratio=0.99):
    """
    Filter preprocessing based on PCA variance analysis.

    Args:
        X_preprocessed: Transformed data (n_samples, n_features)
        n_components: Number of PCA components
        min_variance_ratio: Minimum cumulative variance threshold
        max_first_component_ratio: Maximum allowed 1st component ratio

    Returns:
        dict with 'variance_score', 'is_valid', 'reason', 'explained_variance_ratio'
    """
    n_comp = min(n_components, X_preprocessed.shape[1], X_preprocessed.shape[0])
    pca = PCA(n_components=n_comp)
    pca.fit(X_preprocessed)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    first_comp_ratio = pca.explained_variance_ratio_[0]
    total_var = cumulative_var[-1]

    is_valid = True
    reason = ""

    if total_var < min_variance_ratio:
        is_valid = False
        reason = f"Variance too low: {total_var:.3f} < {min_variance_ratio}"
    elif first_comp_ratio > max_first_component_ratio:
        is_valid = False
        reason = f"First component too dominant: {first_comp_ratio:.3f} > {max_first_component_ratio}"

    return {
        'variance_score': total_var,
        'is_valid': is_valid,
        'reason': reason,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }
```

---

#### A2. Signal-to-Noise Ratio (SNR)

**Name:** `snr_filter`

**Description:**
Measures the signal-to-noise ratio before/after preprocessing. Eliminates preprocessings that increase relative noise (SNR↓).

**Speed:** Very fast (simple statistics)

**Inputs:**
- `X_original`: Raw spectra
- `X_preprocessed`: Transformed spectra
- `min_snr_ratio`: Minimum SNR_after / SNR_before (default: 0.8)

**Outputs:**
- `snr_before`: SNR of original data
- `snr_after`: SNR of preprocessed data
- `snr_ratio`: Ratio (>1 means improvement)
- `is_valid`: Boolean

**Implementation Spec:**
```python
import numpy as np

def compute_snr(X):
    """
    Compute Signal-to-Noise Ratio.
    SNR = mean(signal) / std(noise)
    Using sample-wise mean as signal and residual std as noise.
    """
    signal = np.mean(X, axis=1)  # Mean spectrum per sample
    residual = X - signal[:, np.newaxis]  # Deviation from mean
    noise = np.std(residual, axis=1)  # Noise per sample

    # Avoid division by zero
    noise = np.where(noise == 0, 1e-10, noise)
    snr = np.mean(np.abs(signal)) / np.mean(noise)
    return snr

def snr_filter(X_original, X_preprocessed, min_snr_ratio=0.8):
    """
    Filter preprocessing based on SNR analysis.

    Args:
        X_original: Original spectra
        X_preprocessed: Transformed spectra
        min_snr_ratio: Minimum acceptable SNR ratio

    Returns:
        dict with 'snr_before', 'snr_after', 'snr_ratio', 'is_valid', 'reason'
    """
    snr_before = compute_snr(X_original)
    snr_after = compute_snr(X_preprocessed)
    snr_ratio = snr_after / snr_before if snr_before != 0 else 0

    is_valid = snr_ratio >= min_snr_ratio
    reason = "" if is_valid else f"SNR degraded: ratio={snr_ratio:.3f} < {min_snr_ratio}"

    return {
        'snr_before': snr_before,
        'snr_after': snr_after,
        'snr_ratio': snr_ratio,
        'is_valid': is_valid,
        'reason': reason
    }
```

---

#### A3. Roughness Score

**Name:** `roughness_filter`

**Description:**
Measures the "roughness" or high-frequency content of spectra. Too aggressive derivatives or smoothing can produce artifacts.

**Speed:** Very fast

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `max_roughness`: Maximum acceptable roughness (default: auto from baseline)

**Outputs:**
- `roughness`: Mean absolute second derivative
- `is_valid`: Boolean

**Implementation Spec:**
```python
import numpy as np

def compute_roughness(X):
    """
    Compute spectral roughness as mean absolute second derivative.
    High values indicate jagged/noisy spectra.
    """
    d2 = np.diff(X, n=2, axis=1)  # Second derivative
    roughness = np.mean(np.abs(d2))
    return roughness

def roughness_filter(X_preprocessed, X_original=None, max_roughness_ratio=10.0):
    """
    Filter preprocessing based on roughness analysis.

    Args:
        X_preprocessed: Transformed spectra
        X_original: Original spectra (for ratio comparison)
        max_roughness_ratio: Maximum acceptable roughness ratio vs original

    Returns:
        dict with 'roughness', 'roughness_ratio', 'is_valid', 'reason'
    """
    roughness = compute_roughness(X_preprocessed)

    if X_original is not None:
        roughness_orig = compute_roughness(X_original)
        roughness_ratio = roughness / roughness_orig if roughness_orig > 0 else float('inf')
    else:
        roughness_ratio = 1.0

    is_valid = roughness_ratio <= max_roughness_ratio
    reason = "" if is_valid else f"Too rough: ratio={roughness_ratio:.3f} > {max_roughness_ratio}"

    return {
        'roughness': roughness,
        'roughness_ratio': roughness_ratio,
        'is_valid': is_valid,
        'reason': reason
    }
```

---

#### A4. Intra/Inter-Sample Distance

**Name:** `distance_separation_filter`

**Description:**
Computes L2 distances intra-sample (similar samples) vs inter-sample (different samples). Good preprocessings should increase separation.

**Speed:** Moderate (distance computations)

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values (for grouping similar/different samples)
- `min_separation_ratio`: Minimum inter/intra distance ratio (default: 1.5)

**Outputs:**
- `intra_distance`: Mean distance within similar samples
- `inter_distance`: Mean distance between different samples
- `separation_ratio`: inter/intra
- `is_valid`: Boolean

**Implementation Spec:**
```python
import numpy as np
from sklearn.metrics import pairwise_distances

def distance_separation_filter(X_preprocessed, y=None, min_separation_ratio=1.0, n_samples=500):
    """
    Filter preprocessing based on sample separation analysis.

    For regression: groups by Y quantiles
    For classification: groups by class

    Args:
        X_preprocessed: Transformed spectra
        y: Target values (optional, uses random pairs if None)
        min_separation_ratio: Minimum inter/intra distance ratio
        n_samples: Number of pairs to sample for efficiency

    Returns:
        dict with 'intra_distance', 'inter_distance', 'separation_ratio', 'is_valid'
    """
    n = X_preprocessed.shape[0]

    if y is not None:
        # Create groups based on Y
        if len(np.unique(y)) < 10:  # Classification
            groups = y
        else:  # Regression: quantile-based groups
            groups = np.digitize(y, np.percentile(y, [25, 50, 75]))

        intra_pairs = []
        inter_pairs = []

        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            indices = np.where(mask)[0]
            if len(indices) >= 2:
                # Sample intra pairs
                for _ in range(min(n_samples // len(unique_groups), len(indices))):
                    i, j = np.random.choice(indices, 2, replace=False)
                    intra_pairs.append((i, j))

            # Sample inter pairs
            other_indices = np.where(~mask)[0]
            if len(other_indices) > 0:
                for idx in indices[:min(n_samples // len(unique_groups), len(indices))]:
                    j = np.random.choice(other_indices)
                    inter_pairs.append((idx, j))

        if intra_pairs:
            intra_dists = [np.linalg.norm(X_preprocessed[i] - X_preprocessed[j]) for i, j in intra_pairs]
            intra_distance = np.mean(intra_dists)
        else:
            intra_distance = 1e-10

        if inter_pairs:
            inter_dists = [np.linalg.norm(X_preprocessed[i] - X_preprocessed[j]) for i, j in inter_pairs]
            inter_distance = np.mean(inter_dists)
        else:
            inter_distance = 0
    else:
        # Without Y, just compute overall variance
        distances = pairwise_distances(X_preprocessed[:min(100, n)])
        intra_distance = np.mean(distances[np.tril_indices(len(distances), -1)])
        inter_distance = intra_distance  # Cannot distinguish without Y

    separation_ratio = inter_distance / intra_distance if intra_distance > 0 else 0
    is_valid = separation_ratio >= min_separation_ratio
    reason = "" if is_valid else f"Poor separation: ratio={separation_ratio:.3f} < {min_separation_ratio}"

    return {
        'intra_distance': intra_distance,
        'inter_distance': inter_distance,
        'separation_ratio': separation_ratio,
        'is_valid': is_valid,
        'reason': reason
    }
```

---

### Stage B: Supervised Ranking (Fast, with Y)

These methods use the target Y but without full model training. They provide reliable preprocessing rankings in seconds.

---

#### B1. RV Coefficient

**Name:** `rv_coefficient`

**Description:**
RV coefficient (Renyi–Van der Waerden) measures similarity between the latent space of X_preprocessed and Y. Works for regression. Ultra-fast (just matrix products).

**Speed:** ~milliseconds per preprocessing

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `center`: Whether to center matrices (default: True)

**Outputs:**
- `rv_score`: RV coefficient (0-1, higher is better)

**Implementation Spec:**
```python
import numpy as np

def rv_coefficient(X_preprocessed, y, center=True):
    """
    Compute RV coefficient between X and Y.

    RV = trace(X'Y Y'X) / sqrt(trace(X'X X'X) * trace(Y'Y Y'Y))

    Args:
        X_preprocessed: Transformed spectra (n_samples, n_features)
        y: Target values (n_samples,) or (n_samples, n_targets)
        center: Whether to center the matrices

    Returns:
        dict with 'rv_score'
    """
    X = X_preprocessed.copy()
    Y = np.atleast_2d(y).T if y.ndim == 1 else y.copy()

    if center:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    # Compute Gram matrices
    XX = X @ X.T  # (n, n)
    YY = Y @ Y.T  # (n, n)

    # RV coefficient
    numerator = np.trace(XX @ YY)
    denominator = np.sqrt(np.trace(XX @ XX) * np.trace(YY @ YY))

    rv_score = numerator / denominator if denominator > 0 else 0

    return {'rv_score': rv_score}
```

---

#### B2. CKA (Centered Kernel Alignment)

**Name:** `cka_score`

**Description:**
CKA is widely used in deep learning to measure X↔Y relationship. One of the best metrics for ranking preprocessings before learning.

**Speed:** Fast (kernel computations)

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `kernel`: Kernel type ('linear' or 'rbf', default: 'linear')

**Outputs:**
- `cka_score`: CKA value (0-1, higher is better)

**Implementation Spec:**
```python
import numpy as np

def centering_matrix(n):
    """Create centering matrix H = I - 1/n * 1*1'"""
    return np.eye(n) - np.ones((n, n)) / n

def hsic(K, L, H):
    """Hilbert-Schmidt Independence Criterion"""
    return np.trace(K @ H @ L @ H) / (K.shape[0] - 1) ** 2

def cka_score(X_preprocessed, y, kernel='linear', gamma=None):
    """
    Compute Centered Kernel Alignment between X and Y.

    CKA = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        kernel: 'linear' or 'rbf'
        gamma: RBF kernel parameter (auto if None)

    Returns:
        dict with 'cka_score'
    """
    Y = np.atleast_2d(y).T if y.ndim == 1 else y
    n = X_preprocessed.shape[0]
    H = centering_matrix(n)

    if kernel == 'linear':
        K_X = X_preprocessed @ X_preprocessed.T
        K_Y = Y @ Y.T
    elif kernel == 'rbf':
        from sklearn.metrics.pairwise import rbf_kernel
        if gamma is None:
            gamma = 1.0 / X_preprocessed.shape[1]
        K_X = rbf_kernel(X_preprocessed, gamma=gamma)
        K_Y = rbf_kernel(Y, gamma=1.0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    hsic_xy = hsic(K_X, K_Y, H)
    hsic_xx = hsic(K_X, K_X, H)
    hsic_yy = hsic(K_Y, K_Y, H)

    denominator = np.sqrt(hsic_xx * hsic_yy)
    cka = hsic_xy / denominator if denominator > 0 else 0

    return {'cka_score': cka}
```

---

#### B3. Correlation with Y

**Name:** `correlation_score`

**Description:**
Compute Pearson correlation of each feature with Y, then aggregate. Preprocessings that increase global correlation are more likely to be useful.

**Speed:** Very fast

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `aggregation`: How to aggregate correlations ('max', 'mean', 'sum', 'l1_norm')

**Outputs:**
- `correlation_score`: Aggregated correlation
- `top_correlations`: Top-k feature correlations

**Implementation Spec:**
```python
import numpy as np

def correlation_score(X_preprocessed, y, aggregation='max', top_k=10):
    """
    Compute feature-wise correlations with target and aggregate.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        aggregation: 'max', 'mean', 'sum', or 'l1_norm'
        top_k: Number of top correlations to return

    Returns:
        dict with 'correlation_score', 'top_correlations', 'all_correlations'
    """
    n_features = X_preprocessed.shape[1]
    correlations = np.zeros(n_features)

    y_centered = y - y.mean()
    y_std = y.std()
    if y_std == 0:
        return {'correlation_score': 0, 'top_correlations': [], 'all_correlations': correlations}

    for j in range(n_features):
        x_j = X_preprocessed[:, j]
        x_centered = x_j - x_j.mean()
        x_std = x_j.std()
        if x_std > 0:
            correlations[j] = np.abs(np.dot(x_centered, y_centered) / (len(y) * x_std * y_std))

    if aggregation == 'max':
        score = np.max(correlations)
    elif aggregation == 'mean':
        score = np.mean(correlations)
    elif aggregation == 'sum':
        score = np.sum(correlations)
    elif aggregation == 'l1_norm':
        score = np.linalg.norm(correlations, ord=1)
    else:
        score = np.max(correlations)

    top_indices = np.argsort(correlations)[-top_k:][::-1]
    top_corrs = [(int(i), float(correlations[i])) for i in top_indices]

    return {
        'correlation_score': score,
        'top_correlations': top_corrs,
        'all_correlations': correlations
    }
```

---

#### B4. Fast PLS Score

**Name:** `pls_score`

**Description:**
Train a very fast PLS with 1-2 latent variables. No CV needed. Rank preprocessings by covariance captured or quick RMSE.

**Speed:** ~1-5 seconds per preprocessing (the fastest supervised method in chemometrics)

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `n_components`: Number of PLS components (default: 2)
- `cv_folds`: Optional quick CV (default: None = no CV)

**Outputs:**
- `pls_r2`: R² score
- `pls_rmse`: RMSE (if cv_folds provided)
- `covariance_captured`: Covariance explained by latent variables

**Implementation Spec:**
```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def pls_score(X_preprocessed, y, n_components=2, cv_folds=None):
    """
    Evaluate preprocessing using fast PLS regression.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        n_components: Number of PLS latent variables (1-2 is fast)
        cv_folds: Number of CV folds (None = no CV, fit on all data)

    Returns:
        dict with 'pls_r2', 'pls_rmse', 'covariance_captured'
    """
    n_comp = min(n_components, X_preprocessed.shape[1], X_preprocessed.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp)

    if cv_folds is not None and cv_folds > 1:
        # Quick cross-validation
        scores = cross_val_score(pls, X_preprocessed, y, cv=cv_folds, scoring='r2')
        pls_r2 = np.mean(scores)

        # Fit on all data for RMSE
        pls.fit(X_preprocessed, y)
        y_pred = pls.predict(X_preprocessed)
        pls_rmse = np.sqrt(mean_squared_error(y, y_pred))
    else:
        # No CV, just fit
        pls.fit(X_preprocessed, y)
        y_pred = pls.predict(X_preprocessed)
        pls_r2 = r2_score(y, y_pred)
        pls_rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Covariance captured by latent variables
    # Sum of squared covariances between X and Y scores
    X_scores = pls.x_scores_
    Y_scores = pls.y_scores_
    covariances = [np.cov(X_scores[:, i], Y_scores[:, i])[0, 1] for i in range(n_comp)]
    covariance_captured = sum(c**2 for c in covariances)

    return {
        'pls_r2': pls_r2,
        'pls_rmse': pls_rmse,
        'covariance_captured': covariance_captured
    }
```

---

### Stage C: Proxy Models (Final Selection)

Quick mini-training for final ranking of top candidates.

---

#### C1. Ridge Regression Proxy

**Name:** `ridge_proxy`

**Description:**
Fast Ridge regression with minimal CV (2-3 folds).

**Speed:** ~5-10 seconds per preprocessing

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `cv_folds`: Number of CV folds (default: 3)
- `alphas`: Ridge alpha values to try

**Outputs:**
- `ridge_r2`: Best R² score
- `best_alpha`: Best regularization parameter

**Implementation Spec:**
```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import numpy as np

def ridge_proxy(X_preprocessed, y, cv_folds=3, alphas=None):
    """
    Quick Ridge regression evaluation.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        cv_folds: Number of CV folds
        alphas: Ridge regularization values

    Returns:
        dict with 'ridge_r2', 'best_alpha'
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    ridge = RidgeCV(alphas=alphas, cv=cv_folds)
    ridge.fit(X_preprocessed, y)

    # Get score with best alpha
    scores = cross_val_score(ridge, X_preprocessed, y, cv=cv_folds, scoring='r2')

    return {
        'ridge_r2': np.mean(scores),
        'best_alpha': ridge.alpha_
    }
```

---

#### C2. KNN Proxy

**Name:** `knn_proxy`

**Description:**
Fast KNN with k=3-5 for quick evaluation.

**Speed:** ~2-5 seconds per preprocessing

**Inputs:**
- `X_preprocessed`: Transformed spectra
- `y`: Target values
- `n_neighbors`: Number of neighbors (default: 3)
- `cv_folds`: Number of CV folds (default: 3)

**Outputs:**
- `knn_score`: R² (regression) or accuracy (classification)

**Implementation Spec:**
```python
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def knn_proxy(X_preprocessed, y, n_neighbors=3, cv_folds=3, task='auto'):
    """
    Quick KNN evaluation.

    Args:
        X_preprocessed: Transformed spectra
        y: Target values
        n_neighbors: Number of neighbors
        cv_folds: Number of CV folds
        task: 'regression', 'classification', or 'auto'

    Returns:
        dict with 'knn_score', 'task'
    """
    if task == 'auto':
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'

    if task == 'regression':
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        scoring = 'r2'
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scoring = 'accuracy'

    scores = cross_val_score(knn, X_preprocessed, y, cv=cv_folds, scoring=scoring)

    return {
        'knn_score': np.mean(scores),
        'task': task
    }
```

---

### Stage D: Combination Analysis

Methods for analyzing preprocessing stacks and combinations.

---

#### D1. Mutual Information Redundancy

**Name:** `mutual_info_redundancy`

**Description:**
Compute MI of each preprocessing with Y, then penalize redundant combinations.

**Speed:** Moderate

**Inputs:**
- `preprocessed_variants`: Dict of {name: X_preprocessed}
- `y`: Target values

**Outputs:**
- `mi_scores`: MI with Y for each preprocessing
- `redundancy_matrix`: Pairwise MI between preprocessings
- `combination_scores`: Scores for combinations (MI - redundancy)

**Implementation Spec:**
```python
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def mutual_info_redundancy(preprocessed_variants, y, task='auto', top_k_features=50):
    """
    Analyze preprocessing combinations using mutual information.

    Args:
        preprocessed_variants: Dict of {name: X_preprocessed}
        y: Target values
        task: 'regression' or 'classification' or 'auto'
        top_k_features: Number of top features to use for redundancy

    Returns:
        dict with 'mi_scores', 'redundancy_matrix', 'combination_scores'
    """
    if task == 'auto':
        task = 'classification' if len(np.unique(y)) < 10 else 'regression'

    mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression

    names = list(preprocessed_variants.keys())
    n_pp = len(names)

    # MI with Y for each preprocessing
    mi_scores = {}
    reduced_X = {}  # Keep top features for redundancy computation

    for name, X in preprocessed_variants.items():
        mi_values = mi_func(X, y)
        mi_scores[name] = np.mean(mi_values)

        # Keep top features for redundancy
        top_indices = np.argsort(mi_values)[-top_k_features:]
        reduced_X[name] = X[:, top_indices]

    # Pairwise redundancy (MI between preprocessings)
    redundancy_matrix = np.zeros((n_pp, n_pp))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                # Use mean correlation as proxy for redundancy
                Xi = reduced_X[name_i]
                Xj = reduced_X[name_j]
                corr = np.abs(np.corrcoef(Xi.mean(axis=1), Xj.mean(axis=1))[0, 1])
                redundancy_matrix[i, j] = corr
                redundancy_matrix[j, i] = corr

    # Score combinations (MI - redundancy)
    combination_scores = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                combo_name = f"{name_i}+{name_j}"
                score = mi_scores[name_i] + mi_scores[name_j] - redundancy_matrix[i, j]
                combination_scores[combo_name] = score

    return {
        'mi_scores': mi_scores,
        'redundancy_matrix': redundancy_matrix,
        'names': names,
        'combination_scores': combination_scores
    }
```

---

#### D2. Grassmann Distance

**Name:** `grassmann_distance`

**Description:**
Compute angles between latent spaces of different preprocessings. Combinations with similar latent spaces are redundant.

**Speed:** Fast (SVD computations)

**Inputs:**
- `preprocessed_variants`: Dict of {name: X_preprocessed}
- `n_components`: Number of latent dimensions (default: 5)

**Outputs:**
- `distance_matrix`: Pairwise Grassmann distances
- `similar_pairs`: Pairs with low distance (redundant)

**Implementation Spec:**
```python
import numpy as np
from scipy.linalg import subspace_angles

def grassmann_distance(preprocessed_variants, n_components=5):
    """
    Compute Grassmann distances between preprocessing latent spaces.

    Args:
        preprocessed_variants: Dict of {name: X_preprocessed}
        n_components: Number of PCA components for subspace

    Returns:
        dict with 'distance_matrix', 'names', 'similar_pairs'
    """
    from sklearn.decomposition import PCA

    names = list(preprocessed_variants.keys())
    n_pp = len(names)

    # Compute subspaces
    subspaces = {}
    for name, X in preprocessed_variants.items():
        n_comp = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        subspaces[name] = pca.components_.T  # (n_features, n_components)

    # Compute pairwise Grassmann distances
    distance_matrix = np.zeros((n_pp, n_pp))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i < j:
                angles = subspace_angles(subspaces[name_i], subspaces[name_j])
                # Grassmann distance = sqrt(sum of squared angles)
                dist = np.sqrt(np.sum(angles ** 2))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

    # Find similar pairs (low distance = redundant)
    similar_pairs = []
    threshold = np.percentile(distance_matrix[distance_matrix > 0], 25)
    for i in range(n_pp):
        for j in range(i + 1, n_pp):
            if distance_matrix[i, j] < threshold:
                similar_pairs.append((names[i], names[j], distance_matrix[i, j]))

    return {
        'distance_matrix': distance_matrix,
        'names': names,
        'similar_pairs': similar_pairs
    }
```

---

## Recommended Pipeline

### Full Selection Workflow

```python
from selection import PreprocessingSelector

# Initialize selector
selector = PreprocessingSelector(verbose=1)

# Define preprocessings to evaluate
from nirs4all.operators.transforms import (
    StandardNormalVariate, SavitzkyGolay, MultiplicativeScatterCorrection,
    FirstDerivative, SecondDerivative, Haar, Detrend, Gaussian, IdentityTransformer
)

preprocessings = {
    'identity': IdentityTransformer(),
    'snv': StandardNormalVariate(),
    'msc': MultiplicativeScatterCorrection(),
    'savgol': SavitzkyGolay(window_length=11, polyorder=3),
    'savgol_d1': SavitzkyGolay(window_length=11, polyorder=3, deriv=1),
    'savgol_d2': SavitzkyGolay(window_length=17, polyorder=2, deriv=2),
    'd1': FirstDerivative(),
    'd2': SecondDerivative(),
    'haar': Haar(),
    'detrend': Detrend(),
    'gaussian1': Gaussian(order=1, sigma=2),
    'gaussian2': Gaussian(order=2, sigma=1),
}

# Run selection
results = selector.select(
    X=X_raw,
    y=y,
    preprocessings=preprocessings,
    stages=['A', 'B', 'C'],  # Run all stages
    top_k=5  # Return top 5 preprocessings
)

# Results
print("Selected preprocessings:")
for name, score in results['ranking']:
    print(f"  {name}: {score:.4f}")

# Recommended combinations
print("\nRecommended 2D combinations:")
for combo in results['combinations_2d'][:5]:
    print(f"  {combo}")
```

---

## Expected Outcomes

| Stage | Methods | Purpose | Typical Time |
|-------|---------|---------|--------------|
| Stage 1 | PCA variance, SNR, Roughness, Separation | Unsupervised ranking | 30-60s |
| Stage 2 | Grassmann, CKA, RV, Procrustes, Trust, Cov | Diversity filtering | 10-30s |
| Stage 3 | Ridge, KNN with 3-fold CV | Proxy model ranking | 30-60s |
| Stage 4 | 2-way and 3-way augmentations | Feature combination | 60-120s |

**Typical reduction:** 4000+ pipeline combinations → 30-50 final candidates

**Ranking correlation with final model:** 0.7-0.9

**Time savings:** 80-95% compared to exhaustive cross-validation

---

## Distance Metrics Reference

### Subspace-Based (compare feature space structure)
- **Grassmann**: Angular distance between PCA subspaces
- **CKA**: Centered Kernel Alignment (representation similarity)
- **RV**: Multivariate correlation structure

### Geometry-Based (compare sample distributions)
- **Procrustes**: Shape alignment distance
- **Trustworthiness**: Neighborhood preservation
- **Covariance**: Distribution shape similarity

---

## See Also

- [METHODOLOGY.md](METHODOLOGY.md) - Detailed mathematical specification
- [systematic/](systematic/) - Implementation modules
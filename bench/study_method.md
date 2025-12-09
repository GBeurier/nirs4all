# Materials and Methods

## Automated Multi-Model Pipeline for NIRS Regression Analysis

---

## 1. Study Design and Workflow Overview

This study implements a comprehensive, automated framework for near-infrared spectroscopy (NIRS) regression analysis. The methodology combines preprocessing optimization, chemometric baselines, machine learning ensembles, deep learning, and transformer architectures into a unified, reproducible workflow.

The framework is designed to process **multiple datasets sequentially**, enabling systematic benchmarking across diverse NIRS applications. When datasets contain repeated measurements per sample, predictions can be **aggregated at the sample level** for robust evaluation.

All experiments were conducted using **nirs4all** (https://github.com/GBeurier/nirs4all), an open-source Python library providing unified pipeline management, automated hyperparameter optimization, and model comparison tools.

### 1.1 Overall Workflow

The analysis follows a four-phase cascading architecture where each phase informs the next:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TRANSFER PREPROCESSING SELECTION                                  │
│  ─────────────────────────────────────────                                  │
│  • Evaluate large preprocessing search space using transfer metrics         │
│  • Select top-K preprocessing pipelines for downstream evaluation           │
│  OUTPUT → Filtered preprocessing list (e.g., top 10 configurations)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: PLS BASELINE WITH FINETUNING                                      │
│  ─────────────────────────────────────                                      │
│  • Train PLS models with automated component tuning                         │
│  • Evaluate preprocessing combinations via feature augmentation             │
│  • Identify best preprocessing + hyperparameter combinations                │
│  OUTPUT → Top preprocessing pipelines + optimal n_components               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: ENSEMBLE AND DEEP LEARNING                                        │
│  ─────────────────────────────────────                                      │
│  • Ridge regression with regularization tuning                              │
│  • Gradient boosting (CatBoost) with multiple configurations               │
│  • 1D-CNN architectures (NICON variants)                                   │
│  • Use validated preprocessing from Phase 2                                 │
│  OUTPUT → Predictions from diverse model families                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: TRANSFORMER-BASED REGRESSION                                      │
│  ─────────────────────────────────────                                      │
│  • Dimensionality reduction (PCA, SVD, Wavelets, Random Projections)       │
│  • Transformer model with custom hyperparameter tuning                      │
│  • Incorporate best preprocessing from Phase 2                              │
│  OUTPUT → Transformer predictions with optimized configurations            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AGGREGATION AND RANKING                                                    │
│  ───────────────────────                                                    │
│  • Combine predictions from all phases                                      │
│  • Rank by RMSE on test partition                                          │
│  • Optional sample-level aggregation for repeated measurements             │
│  OUTPUT → Ranked model comparison across all approaches                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Principles

1. **Cascading knowledge transfer**: Preprocessing validated by simpler models (PLS) is reused by complex models
2. **Automated hyperparameter optimization**: All tunable models use TPE-based Bayesian optimization
3. **Consistent cross-validation**: Same fold structure across all phases for fair comparison
4. **Reproducibility**: Full pipeline serialization enables exact reconstruction

---

## 2. Cross-Validation Strategy

### 2.1 SPXYGFold: Group-Aware SPXY Splitting

Data partitioning employs **SPXYGFold**, a custom K-Fold extension of the SPXY (Sample Partitioning based on joint X-Y distances) algorithm (Galvão et al., 2005). The classical Kennard-Stone algorithm (Kennard & Stone, 1969) selects calibration samples by maximizing the minimum distance between selected points, ensuring uniform coverage of the feature space. SPXY extends this by incorporating target-space information:

$$d_{XY}(i,j) = \frac{d_X(i,j)}{\max(d_X)} + \frac{d_Y(i,j)}{\max(d_Y)}$$

where $d_X(i,j) = \|x_i - x_j\|$ is the Euclidean distance in feature space and $d_Y(i,j) = |y_i - y_j|$ is the distance in target space. The normalization ensures both components contribute equally regardless of scale.

#### SPXYGFold Extensions

The original SPXY produces a single train/test split. SPXYGFold extends this with three key capabilities:

1. **K-Fold cross-validation**: Instead of a single partition, the algorithm generates K non-overlapping folds. The max-min selection iteratively assigns samples to folds in round-robin fashion, ensuring each fold maintains chemometric representativity.

2. **Group-aware splitting**: When samples have repeated measurements (e.g., multiple scans per physical sample), all replicates must remain in the same fold to prevent data leakage. SPXYGFold aggregates group members using their centroid for distance calculations, then assigns entire groups to folds.

3. **Alternating max-min assignment**: For each unassigned sample/group, the algorithm computes the minimum distance to all already-assigned members in each fold, then assigns to the fold where this minimum is maximized. This ensures uniform spectral coverage across all folds.

The algorithm complexity is $O(n^2 \cdot p)$ for distance matrix computation, where $n$ is the number of samples/groups and $p$ the number of features.

### 2.2 Target Normalization

Target values (y) are scaled to the range (0.05, 0.9) using MinMaxScaler. This bounded normalization:
- Prevents extreme gradient values during neural network training
- Ensures consistent loss magnitudes across datasets with different target ranges
- Avoids boundary saturation in sigmoid-based activations
- Maintains numerical stability for Bayesian hyperparameter optimization

---

## 3. Preprocessing Selection

### 3.1 Preprocessing Search Space

The preprocessing search space encompasses established NIRS correction techniques organized in a four-stage Cartesian structure. Each stage addresses a specific spectral artifact:

| Stage | Options | Purpose | Physical Basis |
|-------|---------|---------|----------------|
| **1. Scatter Correction** | None, MSC, SNV, EMSC, RSNV | Remove multiplicative scatter | Particle size, surface texture (Geladi et al., 1985) |
| **2. Smoothing** | None, Savitzky-Golay (w=11,15), Gaussian (σ=1,2) | Reduce high-frequency noise | Instrumental noise, detector artifacts (Savitzky & Golay, 1964) |
| **3. Derivatives** | None, 1st/2nd derivative, SG-derivatives | Remove baseline, enhance peaks | Additive offsets, slope drift (Rinnan et al., 2009) |
| **4. Advanced** | None, Haar, Sym5, Coif3, Detrend, Area-norm | Multi-resolution analysis, normalization | Complex baselines, intensity normalization (Trygg & Wold, 1998) |

**Preprocessing Methods Details:**

- **MSC (Multiplicative Scatter Correction)**: Regresses each spectrum against a reference (mean spectrum), correcting slope and intercept to remove scatter effects
- **SNV (Standard Normal Variate)**: Row-wise standardization (zero mean, unit variance), removing scatter without requiring a reference
- **EMSC (Extended MSC)**: Adds polynomial terms to MSC for complex scatter patterns
- **RSNV (Robust SNV)**: Uses median and MAD instead of mean/std for outlier resistance
- **Savitzky-Golay**: Polynomial smoothing with optional derivative computation
- **Wavelet transforms**: Multi-resolution decomposition for simultaneous denoising and feature extraction

The Cartesian generator produces all valid stage combinations. With 5×9×6×9 = 2,430 possible pipelines, exhaustive evaluation is computationally prohibitive, motivating the transfer-based selection.

### 3.2 Transfer Preprocessing Selection (Phase 1)

The **TransferPreprocessingSelector** implements a rapid preprocessing screening strategy that evaluates pipeline quality without full model training. This approach is inspired by transfer learning principles where spectral transformations that preserve structure tend to generalize well.

#### Selection Algorithm

The selector operates in three stages:

**Stage A: Preprocessing Application**
Each candidate preprocessing pipeline $P_i$ is applied to the dataset, producing transformed spectra $X_{P_i}$. Pipelines that produce invalid outputs (NaN, infinite values, zero variance) are immediately discarded.

**Stage B: Quality Metric Computation**
For each valid preprocessing, the selector computes a battery of spectral quality metrics:

| Metric | Description | Good Values |
|--------|-------------|-------------|
| **Variance Retention** | Fraction of original variance preserved | 0.5–1.0 |
| **Correlation Preservation** | Mean pairwise correlation vs. original | > 0.7 |
| **Noise Ratio** | High-frequency energy / total energy | < 0.3 |
| **PLS Learnability** | Cross-validated R² with few components | > 0.5 |
| **Condition Number** | Numerical stability indicator | < 1000 |

**Stage C: Ranking and Selection**
Metrics are combined into a weighted score. The "balanced" preset uses equal weights across all metrics. The top-K preprocessing pipelines (default: 10) are retained for downstream evaluation.

#### Computational Efficiency

By avoiding full model training, the transfer selector evaluates hundreds of preprocessing combinations in minutes rather than hours. This provides a 10–100× speedup compared to exhaustive cross-validated model comparison.

### 3.3 Preprocessing Fingerprinting

Each preprocessing pipeline receives a unique fingerprint encoding the sequence of transformations and their parameters (e.g., `"SNV|SavitzkyGolay(window=11)|FirstDerivative"`). This enables:
- Deduplication of equivalent pipelines
- Tracking preprocessing provenance across phases
- Reproducible pipeline reconstruction from stored predictions

---

## 4. Hyperparameter Finetuning

### 4.1 Optimization Framework

All tunable models use **Optuna-based Bayesian optimization** (Akiba et al., 2019) with Tree-structured Parzen Estimator (TPE) sampling. TPE models the objective function using kernel density estimation, sampling promising hyperparameter regions more frequently than random search.

#### Finetuning Configuration

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_trials` | Number of optimization iterations | 10–30 |
| `approach` | Fold handling strategy | "grouped" / "single" |
| `eval_mode` | Score aggregation across folds | "avg" / "best" |
| `sample` | Sampling algorithm | "tpe" / "random" |

**Evaluation Approaches:**

- **Grouped evaluation** (`approach="grouped"`): Each trial trains on all K folds. The objective is the average validation score across folds. This provides robust estimates but requires K× training time.

- **Single evaluation** (`approach="single"`): Each trial uses one fold only. Faster but higher variance. Combined with `eval_mode="best"`, this selects the configuration achieving the best single-fold score.

#### Parameter Space Types

The framework supports flexible parameter specifications:

```python
"model_params": {
    "n_components": ("int", 1, 40),           # Integer range
    "alpha": ("log_float", 0.001, 100),       # Log-uniform float
    "learning_rate": ("float", 0.01, 0.3),    # Uniform float
    "model_variant": ("categorical", ["A", "B", "C"])  # Categorical
}
```

### 4.2 Feature Augmentation Strategy

The **feature augmentation** operator enables joint optimization of preprocessing and model hyperparameters. Instead of evaluating each preprocessing separately, the operator samples preprocessing combinations as part of the hyperparameter search:

```python
{"feature_augmentation": {"_or_": preprocessing_list, "pick": [1, 2], "count": 40}}
```

**Parameters:**
- `"_or_"`: List of candidate preprocessing pipelines
- `"pick"`: Range of pipelines to combine (1–2 means use 1 or 2 pipelines)
- `"count"`: Number of random combinations to generate (40 total configurations)

This approach explores the preprocessing × hyperparameter space jointly, potentially finding synergies that grid search would miss. When multiple preprocessings are selected, their outputs are concatenated along the feature axis.

### 4.3 Best Configuration Extraction

After finetuning, the framework extracts winning configurations:
1. **Top preprocessing pipelines**: Ranked by validation performance, deduplicated by fingerprint
2. **Optimal hyperparameters**: Best values found during optimization (e.g., `n_components=12`)

These are propagated to downstream phases, ensuring complex models benefit from preprocessing insights discovered with simpler models.

---

## 5. Model Architectures

### 5.1 Phase 2: Partial Least Squares Regression

**Partial Least Squares** (PLS) regression (Wold et al., 1984) is the foundational chemometric method for NIRS. PLS projects both X and y onto latent variables that maximize covariance, handling collinearity inherent in spectral data.

#### PLS Finetuning

| Parameter | Search Space | Rationale |
|-----------|--------------|-----------|
| `n_components` | 1–40 | More components capture more variance but risk overfitting |

**Optimization Settings:**
- **Trials**: 20 per preprocessing configuration
- **Approach**: Grouped (average across folds)
- **Evaluation**: RMSE on validation partition

#### Role in the Cascade

PLS serves dual purposes:
1. **Baseline establishment**: Provides chemometric reference performance
2. **Preprocessing validation**: Identifies which preprocessing pipelines work well, informing subsequent phases

The top 10 preprocessing pipelines (ranked by PLS validation RMSE) and the optimal component count are retained for Phases 3–4.

### 5.2 Phase 3: Ensemble and Deep Learning Models

Phase 3 evaluates diverse model architectures using the preprocessing configurations validated by PLS.

#### 5.2.1 Ridge Regression

L2-regularized linear regression serves as a regularized baseline:

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|^2$$

| Parameter | Search Space | Trials |
|-----------|--------------|--------|
| `alpha` | Log-uniform (0.001–100) | 20 |

Ridge often outperforms PLS when the optimal number of components is uncertain, as regularization provides smooth dimensionality control.

#### 5.2.2 CatBoost Gradient Boosting

CatBoost (Prokhorenkova et al., 2018) implements gradient boosting with ordered boosting and symmetric trees. Three pre-defined configurations explore the depth/iterations tradeoff:

| Configuration | Iterations | Depth | Learning Rate | Rationale |
|---------------|------------|-------|---------------|-----------|
| **Conservative** | 200 | 6 | 0.10 | Fast training, lower capacity |
| **Balanced** | 400 | 8 | 0.05 | Standard configuration |
| **Deep** | 300 | 10 | 0.08 | Higher capacity, slower |

All configurations use GPU acceleration (`task_type="GPU"`) for efficient training. CatBoost handles spectral data without explicit feature engineering, learning non-linear relationships between wavelengths.

#### 5.2.3 NICON Deep Learning

NICON (NIRS Convolutional Network) is a family of 1D-CNN architectures specifically designed for spectroscopic regression. Three variants are evaluated:

| Variant | Architecture | Parameters | Use Case |
|---------|--------------|------------|----------|
| **nicon** | Standard 1D-CNN | ~50K | General purpose |
| **nicon_VG** | VGG-style deep blocks | ~200K | Complex patterns |
| **thin_nicon** | Reduced width | ~15K | Limited data |

The architectures use:
- 1D convolutions along the wavelength axis
- Batch normalization and dropout for regularization
- Global average pooling before the regression head
- Adam optimizer with learning rate scheduling

### 5.3 Phase 4: Transformer Architecture

Transformer models (Vaswani et al., 2017), originally designed for sequence modeling, have shown strong performance on tabular regression tasks. However, they require careful input preparation for spectroscopic data.

#### 5.3.1 Dimensionality Reduction

NIRS spectra typically contain hundreds to thousands of wavelengths, exceeding practical transformer input sizes. Dimensionality reduction creates compact, information-dense representations:

| Method | Dimensions | Characteristics |
|--------|------------|-----------------|
| **PCA** | 50, 100 | Linear, variance-maximizing, interpretable |
| **Truncated SVD** | 50 | Similar to PCA, handles sparse data |
| **Sparse Random Projection** | 100 | Fast, approximate, preserves distances |
| **Gaussian Random Projection** | 100 | Johnson-Lindenstrauss guarantee |
| **Wavelet + PCA** | 50 | Multi-resolution features + compression |
| **WaveletFeatures** | ~50 | Direct wavelet coefficient extraction |
| **WaveletPCA/SVD** | ~20 | Hierarchical wavelet decomposition |

Additionally, top preprocessing pipelines from Phase 2 are augmented with PCA(n_components=100) and added to the search space, transferring spectral knowledge to the transformer.

#### 5.3.2 Concat Transform

The **concat_transform** operator evaluates multiple dimensionality reductions simultaneously:

```python
{"concat_transform": {"_or_": reduction_list, "pick": [1, 3], "count": 20}}
```

This samples 1–3 reduction methods per trial, concatenating their outputs. For example, combining PCA(50) + WaveletFeatures(50) produces a 100-dimensional input capturing both variance-based and multi-resolution features.

#### 5.3.3 Custom Hyperparameter Tuning

The transformer finetuning optimizes:

| Parameter Type | Options | Description |
|----------------|---------|-------------|
| **Model variant** | 4 pre-trained checkpoints | Different training distributions |
| **Inference config** | Architecture-specific | Attention heads, layers, etc. |
| **Reduction pipeline** | 20 combinations | Dimensionality reduction method |

**Optimization Settings:**
- **Trials**: 10 (computationally expensive)
- **Approach**: Single-fold with best-fold selection
- **Evaluation**: RMSE on validation partition

---

## 6. Evaluation and Aggregation

### 6.1 Primary Metric

Root Mean Square Error (RMSE) on the test partition serves as the primary ranking metric:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

RMSE is preferred over MSE for interpretability (same units as target) and over MAE for sensitivity to large errors.

### 6.2 Sample-Level Aggregation

For datasets with repeated measurements per sample, raw predictions contain multiple values per physical sample. Aggregation computes the sample-level prediction:

$$\hat{y}_{\text{sample}} = \frac{1}{m}\sum_{j=1}^{m}\hat{y}_j$$

where $m$ is the number of replicate scans. Metrics are then computed on aggregated predictions, ensuring:
- Fair comparison regardless of replication scheme
- Reduced prediction variance through averaging
- Alignment with practical use cases (one prediction per sample)

### 6.3 Cross-Phase Ranking

All predictions from Phases 2–4 are combined into a unified ranking:

1. Collect all model predictions (PLS, Ridge, CatBoost variants, NICON variants, Transformer)
2. Compute test RMSE for each model × preprocessing combination
3. Apply sample-level aggregation if applicable
4. Sort by RMSE ascending
5. Report top-K models with their preprocessing, hyperparameters, and scores

This enables direct comparison across fundamentally different model families, identifying whether classical chemometric methods, machine learning ensembles, or deep learning architectures perform best for a given dataset.

---

## 7. Implementation

### 7.1 Software Framework

**nirs4all** (https://github.com/GBeurier/nirs4all) provides the complete infrastructure:

| Component | Functionality |
|-----------|---------------|
| **DatasetConfigs** | Dataset loading, metadata handling, train/test partitioning |
| **PipelineConfigs** | Declarative pipeline specification (JSON/YAML compatible) |
| **PipelineRunner** | Execution engine with automatic parallelization |
| **TransferPreprocessingSelector** | Rapid preprocessing screening |
| **SPXYGFold** | Group-aware SPXY cross-validation |
| **Finetuning integration** | Optuna-based hyperparameter optimization |
| **Prediction storage** | Parquet-based storage with manifest tracking |
| **PredictionAnalyzer** | Ranking, visualization, and reporting |

The framework handles operator instantiation, fold iteration, metric computation, and result serialization automatically. Pipeline configurations are fully reproducible through manifest serialization.

### 7.2 Hardware Requirements

| Component | Requirement | Usage |
|-----------|-------------|-------|
| **GPU** | CUDA-compatible | CatBoost, NICON, Transformer training |
| **CPU** | Multi-core recommended | PLS, Ridge, preprocessing |
| **Memory** | 16GB+ recommended | Large spectral matrices |

GPU acceleration provides 5–50× speedup for gradient-based models compared to CPU-only execution.

---

## References

Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777.

Galvão, R. K. H., et al. (2005). A method for calibration and validation subset partitioning. *Talanta*, 67(4), 736-740.

Geladi, P., MacDougall, D., & Martens, H. (1985). Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, 39(3), 491-500.

Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments. *Technometrics*, 11(1), 137-148.

Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*, 31.

Rinnan, Å., Van Den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC*, 28(10), 1201-1222.

Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639.

Trygg, J., & Wold, S. (1998). PLS regression on wavelet compressed NIR spectra. *Chemometrics and Intelligent Laboratory Systems*, 42(1-2), 209-220.

Wold, S., et al. (1984). The collinearity problem in linear regression. The partial least squares (PLS) approach to generalized inverses. *SIAM J. Sci. Stat. Comput.*, 5(3), 735-743.

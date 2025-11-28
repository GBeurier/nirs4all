# Systematic Preprocessing Selection for NIRS Data

## Abstract

This document describes a systematic framework for selecting optimal preprocessing techniques for Near-Infrared Spectroscopy (NIRS) data analysis. The framework reduces the preprocessing search space by evaluating candidates through four stages: exhaustive unsupervised evaluation, diversity analysis with 6 distance metrics, proxy model validation, and feature augmentation testing.

---

## 1. Introduction

### 1.1 Problem Statement

NIRS data analysis requires preprocessing to remove noise, baseline drift, and scattering effects before building predictive models. The framework evaluates 16 preprocessing techniques:

| Code | Technique | Description |
|------|-----------|-------------|
| `snv` | Standard Normal Variate | Row-wise centering and scaling to remove scatter |
| `rsnv` | Robust SNV | SNV variant robust to outliers |
| `msc` | Multiplicative Scatter Correction | Corrects multiplicative/additive scatter effects |
| `emsc` | Extended MSC | MSC with polynomial baseline modeling |
| `savgol` | Savitzky-Golay (w=11, p=3) | Polynomial smoothing filter |
| `savgol_d1` | Savitzky-Golay + 1st derivative | Smoothing with simultaneous differentiation |
| `d1` | First Derivative | Removes baseline offset |
| `d2` | Second Derivative | Removes linear baseline drift |
| `haar` | Haar Wavelet | Multi-resolution decomposition |
| `detrend` | Detrend | Removes polynomial baseline trends |
| `gaussian` | Gaussian (σ=2, order=1) | Gaussian smoothing with 1st derivative |
| `gaussian2` | Gaussian (σ=2, order=2) | Gaussian smoothing with 2nd derivative |
| `area_norm` | Area Normalization | Normalizes spectra by total area |
| `wav_sym5` | Wavelet Symlet-5 | Symlet wavelet denoising |
| `wav_coif3` | Wavelet Coiflet-3 | Coiflet wavelet denoising |
| `identity` | Identity | No transformation (baseline reference) |

With *n* = 16 preprocessings, testing all combinations up to depth *d* requires evaluating:

$$N = \sum_{k=1}^{d} \frac{n!}{(n-k)!} = n + n(n-1) + n(n-1)(n-2) + ...$$

For 16 preprocessings at depth 3, this yields **4,320 combinations**—each requiring full cross-validation, which becomes computationally prohibitive.

### 1.2 Proposed Solution

Our framework uses a **4-stage funnel approach**:

1. **Stage 1**: Exhaustive unsupervised evaluation of all pipeline combinations
2. **Stage 2**: Diversity analysis with 6 distance metrics to identify complementary preprocessings
3. **Stage 3**: Proxy model validation (Ridge + KNN) on diverse candidates
4. **Stage 4**: Feature augmentation evaluation (2-way and 3-way concatenations)

This reduces full model evaluations from thousands to ~50-100 configurations while maintaining discovery of optimal solutions.

---

## 2. Methodology

### 2.1 Stage 1: Exhaustive Unsupervised Evaluation

All preprocessing configurations (single, stacked depth-2, stacked depth-3, etc.) are evaluated using five unsupervised metrics that correlate with downstream model performance.

#### 2.1.1 PCA Variance Ratio

Measures information preservation after preprocessing:

$$\text{VR} = \sum_{i=1}^{k} \lambda_i / \sum_{i=1}^{p} \lambda_i$$

where $\lambda_i$ are eigenvalues and $k$ is chosen to capture 95% variance. **Higher is better**—indicates preprocessing preserves predictive information.

#### 2.1.2 Effective Dimensionality

Quantifies the intrinsic dimensionality using entropy-based estimation:

$$d_{eff} = \exp\left(-\sum_i p_i \log p_i\right)$$

where $p_i = \lambda_i / \sum_j \lambda_j$ is the normalized eigenvalue. **Interpretation**: A moderate effective dimensionality (5-20) is optimal. Too low suggests over-smoothing; too high indicates noise retention.

#### 2.1.3 Signal-to-Noise Ratio (SNR)

Estimates signal quality as the ratio of structured variance to noise:

$$\text{SNR} = \frac{\text{Var}(\text{mean spectrum})}{\text{mean}(\text{Var per feature})}$$

**Higher is better**—indicates effective noise reduction.

#### 2.1.4 Roughness Score

Measures spectral smoothness via second-order differences:

$$R = \frac{1}{n \cdot p} \sum_{i,j} |x_{i,j+1} - 2x_{i,j} + x_{i,j-1}|$$

**Lower is better**—indicates smooth, well-behaved spectra without high-frequency noise.

#### 2.1.5 Separation Score

Measures inter-sample separation (normalized pairwise distances):

$$S = \frac{\text{mean pairwise distance}}{\text{mean feature std}}$$

**Higher is better**—indicates good sample discrimination.

#### 2.1.6 Combined Score

Metrics are normalized to \[0,1\] and combined with weights:

$$S_{total} = 0.25 \cdot \text{VR} + 0.25 \cdot d_{eff}^{norm} + 0.20 \cdot \text{SNR}^{norm} + 0.15 \cdot R^{norm} + 0.15 \cdot \text{Sep}^{norm}$$

**Output**: `stage1_unsupervised.csv` with all pipelines ranked by total score.

---

### 2.2 Stage 2: Diversity Analysis

To identify complementary preprocessings for feature augmentation, we compute pairwise distances using **6 distance metrics** grouped into two categories.

#### 2.2.1 Subspace-Based Metrics

These metrics compare the principal component structure of transformed data.

##### Grassmann Distance

Measures the angular distance between PCA subspaces:

$$d_G(X_1, X_2) = \sqrt{\sum_{i=1}^{k} \theta_i^2}$$

where $\theta_i$ are the principal angles between the $k$-dimensional subspaces. Normalized to \[0, 1\].

**Interpretation**: High distance indicates preprocessings capture different aspects of the feature space structure.

##### CKA Distance (Centered Kernel Alignment)

Measures representation similarity invariant to orthogonal transformations:

$$\text{CKA}(X_1, X_2) = \frac{\text{HSIC}(K_1, K_2)}{\sqrt{\text{HSIC}(K_1, K_1) \cdot \text{HSIC}(K_2, K_2)}}$$

$$d_{CKA} = 1 - \text{CKA}(X_1, X_2)$$

where HSIC is the Hilbert-Schmidt Independence Criterion with linear kernels.

##### RV Coefficient Distance

Multivariate generalization of squared Pearson correlation:

$$\text{RV}(X_1, X_2) = \frac{\text{trace}(A \cdot B)}{\sqrt{\text{trace}(A^2) \cdot \text{trace}(B^2)}}$$

$$d_{RV} = 1 - \text{RV}$$

where $A = X_1 X_1^T$ and $B = X_2 X_2^T$ are centered Gram matrices.

##### Combined Subspace Distance

$$d_{subspace} = 0.4 \cdot d_G + 0.4 \cdot d_{CKA} + 0.2 \cdot d_{RV}$$

#### 2.2.2 Geometry-Based Metrics

These metrics compare sample distributions and neighborhood structures.

##### Procrustes Distance

Measures residual after optimal alignment (translation, rotation, scaling):

$$d_P = \min_{R,s,t} \|Z_1 - s \cdot Z_2 R - t\|_F^2$$

Applied to PCA projections $Z_1, Z_2$. Captures differences in point cloud shape.

##### Trustworthiness Distance

Measures neighborhood preservation between representations:

$$T = 1 - \frac{2}{nk(2n-3k-1)} \sum_{i=1}^{n} \sum_{j \in U_k(i) \setminus V_k(i)} (r(i,j) - k)$$

where $U_k(i)$ and $V_k(i)$ are k-neighborhoods in each space, and $r(i,j)$ is the rank.

$$d_T = 1 - T$$

##### Covariance Distance

Frobenius distance between (sample) covariance matrices, normalized:

$$d_{cov} = \frac{\|\Sigma_1 - \Sigma_2\|_F}{\sqrt{\|\Sigma_1\|_F \cdot \|\Sigma_2\|_F}}$$

##### Combined Geometry Distance

$$d_{geometry} = 0.4 \cdot d_P + 0.3 \cdot d_T + 0.3 \cdot d_{cov}$$

#### 2.2.3 Overall Combined Distance

$$d_{combined} = 0.5 \cdot d_{subspace} + 0.5 \cdot d_{geometry}$$

#### 2.2.4 Similarity Filtering

Pipelines with combined distance below threshold to a higher-scored pipeline are filtered out:

```
distance_threshold = 1.0 - similarity_ratio
```

With `similarity_ratio=0.70`, pipelines must have at least 5% distance from better-scored ones to be retained.

**Outputs**:
- `stage2_diversity.csv`: Diversity metrics per pipeline
- `distance_matrix_combined.csv`: Pairwise combined distances
- `distance_matrix_subspace.csv`: Pairwise subspace distances
- `distance_matrix_geometry.csv`: Pairwise geometry distances

---

### 2.3 Stage 3: Proxy Model Evaluation

Top diverse candidates from Stage 2 are evaluated with fast proxy models using cross-validation.

#### 2.3.1 Ridge Regression

$$\hat{w} = (X^TX + \alpha I)^{-1}X^Ty$$

Alpha is selected via cross-validation from \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0\}.

#### 2.3.2 K-Nearest Neighbors

$$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i$$

Uses k=3 with distance weighting (or classification mode if target is discrete).

#### 2.3.3 Evaluation Metrics

- **Regression**: R² score via 3-fold cross-validation
- **Classification**: Accuracy via 3-fold cross-validation

$$\text{Proxy Score} = \frac{1}{2}(R^2_{Ridge} + R^2_{KNN})$$

#### 2.3.4 Final Score

$$S_{final} = 0.4 \cdot S_{unsupervised} + 0.6 \cdot S_{proxy}$$

**Output**: `stage3_proxy.csv`

---

### 2.4 Stage 4: Feature Augmentation

Beyond single and stacked pipelines, we test **feature augmentation**—concatenating features from different preprocessings.

#### 2.4.1 2nd-Order Augmentation

$$X_{aug} = \[X_{prep1} \,|\, X_{prep2}\]$$

All pairs from Stage 3 top candidates are tested.

#### 2.4.2 3rd-Order Augmentation

$$X_{aug} = \[X_{prep1} \,|\, X_{prep2} \,|\, X_{prep3}\]$$

Up to 50 triplet combinations from Stage 3 candidates (limited for computational efficiency).

#### 2.4.3 Augmentation Evaluation

Each augmentation is evaluated with the same unsupervised metrics and proxy models as Stage 3:

$$S_{aug} = 0.4 \cdot S_{unsupervised}(X_{aug}) + 0.6 \cdot S_{proxy}(X_{aug})$$

**Output**: `stage4_augmentation.csv`

---

### 2.5 Final Ranking

All configurations from Stage 3 (single/stacked) and Stage 4 (augmented) are combined and ranked by final score.

**Output**: `final_ranking.csv`

---

## 3. Results Interpretation

### 3.1 Output Files

| File | Description |
|------|-------------|
| `stage1_unsupervised.csv` | All pipelines with unsupervised metrics |
| `stage2_diversity.csv` | Diversity metrics for top candidates |
| `distance_matrix_combined.csv` | Pairwise combined distances |
| `distance_matrix_subspace.csv` | Pairwise subspace-based distances |
| `distance_matrix_geometry.csv` | Pairwise geometry-based distances |
| `stage3_proxy.csv` | Proxy model results for diverse candidates |
| `stage4_augmentation.csv` | Augmentation evaluation results |
| `final_ranking.csv` | Combined ranking of all configurations |
| `systematic_results.png` | Multi-panel visualization |

### 3.2 Visualization Panels

1. **Top Pipelines by Unsupervised Score**: Bar chart of best Stage 1 performers
2. **Metric Radar Chart**: Multi-dimensional view of top pipeline characteristics
3. **Pipeline Depth Distribution**: Performance by stacking depth (1, 2, 3)
4. **Distance Heatmap**: Pairwise preprocessing diversity
5. **Proxy Model Performance**: Ridge vs KNN comparison
6. **Unsupervised vs Proxy Correlation**: Validates metric quality

### 3.3 Interpreting Results

| Scenario | Interpretation |
|----------|----------------|
| Single preprocessing dominates | Data benefits from simple, targeted correction |
| Stacked pipeline wins | Sequential corrections address multiple issues |
| Augmentation wins | Data has multi-modal structure benefiting from diverse views |
| High Grassmann distance | Preprocessings capture different principal structures |
| High CKA distance | Representations are fundamentally different |
| High Procrustes distance | Sample point clouds have different shapes |
| High Trustworthiness distance | Neighborhood structures differ significantly |

### 3.4 Diversity Rankings

Stage 2 provides three diversity rankings:

1. **Subspace Diversity**: Prioritizes pipelines with different principal component structures (Grassmann + CKA + RV)
2. **Geometry Diversity**: Prioritizes pipelines with different sample distributions (Procrustes + Trustworthiness + Covariance)
3. **Combined Diversity**: Balanced view of both

These rankings help identify which preprocessings are most complementary for augmentation.

---

## 4. Example Results

Running on the Digestibility dataset (2856 samples, 949 wavelengths):

```
======================================================================
FINAL RANKING
======================================================================

⏱️ Total time: 142.3s

🏆 Top 15 Configurations:
              name           type  unsupervised_score  proxy_score  final_score
0              snv         single               0.421        0.312        0.356
1         rsnv>snv        stacked               0.418        0.308        0.352
2       snv>savgol        stacked               0.415        0.305        0.349
3       savgol>snv        stacked               0.412        0.302        0.346
4  [snv+savgol>snv]    augmented_2               0.398        0.298        0.338
...
```

**Interpretation**: For this dataset:
- SNV-based preprocessing dominates, indicating strong scattering effects
- Stacking SNV with other techniques provides marginal improvements
- 2-way augmentation \[snv+savgol>snv\] appears in top 5
- The marginal differences suggest robustness to preprocessing choice

---

## 5. Computational Complexity

| Stage | Complexity | Typical Time (1000 samples, depth 3) |
|-------|------------|--------------------------------------|
| Stage 1 | O(n³ × samples × features) | 30-60 seconds |
| Stage 2 | O(k² × 6 distance metrics × PCA) | 10-30 seconds |
| Stage 3 | O(k × CV × 2 proxy fits) | 30-60 seconds |
| Stage 4 | O(C(k,2) + C(k,3) × CV × proxy fits) | 60-120 seconds |

**Total**: ~3-5 minutes vs. ~60+ minutes for exhaustive CV evaluation

**Speedup**: 10-20× reduction in computational cost

---

## 6. Recommendations

1. **Start with depth 2-3**: Most gains come from 2-3 step pipelines
2. **Use top_stage1=15-20**: Balances coverage with computational cost
3. **Check diversity matrices**: High-distance pairs often yield augmentation insights
4. **Validate winners**: Run full model training on top 3-5 configurations
5. **Consider data characteristics**:
   - High scattering → SNV, MSC, RSNV
   - Baseline drift → Detrend, derivatives
   - High noise → Savitzky-Golay, Gaussian smoothing
   - Complex structure → Try augmentations

---

## 7. Algorithm Summary

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
│  Evaluate each candidate with Ridge + KNN (cv_folds CV)         │
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

## 8. References

1. Rinnan, Å., van den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

2. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML*.

3. Ye, J., & Janardan, R. (2005). Generalized principal angle between subspaces. *IEEE TPAMI*.

4. Robert, P., & Escoufier, Y. (1976). A unifying tool for linear multivariate statistical methods: the RV-coefficient. *Journal of the Royal Statistical Society: Series C*, 25(3), 257-265.

5. Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study. *ICANN*.

---

*Document generated by nirs4all preprocessing selection framework*
*Version 0.2 - November 2025*

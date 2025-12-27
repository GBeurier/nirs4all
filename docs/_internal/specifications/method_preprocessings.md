# Systematic Preprocessing Selection for NIRS Data

**Authors**: G. Beurier¹*, L. Rouan¹, D. Cornet¹

**Affiliation**: ¹CIRAD, UMR AGAP Institut, F-34398 Montpellier, France

**Corresponding author**: G. Beurier (beurier@cirad.fr)

**Version**: 1.0 — November 2025

---

## Abstract

This document describes a systematic framework for selecting optimal preprocessing techniques for Near-Infrared Spectroscopy (NIRS) data analysis. The framework reduces the preprocessing search space by evaluating candidates through four stages: exhaustive unsupervised evaluation, diversity analysis with six distance metrics, proxy model validation, and feature augmentation testing. The methodology combines established spectroscopic preprocessing techniques (Rinnan et al., 2009) with modern representation comparison methods (Kornblith et al., 2019) to efficiently identify optimal preprocessing configurations while avoiding exhaustive cross-validation of all possible combinations.

**Keywords**: Near-infrared spectroscopy, preprocessing selection, spectral analysis, dimensionality reduction, feature augmentation

---

## 1. Introduction

### 1.1 Problem Statement

NIRS data analysis requires preprocessing to remove noise, baseline drift, and scattering effects before building predictive models (Rinnan et al., 2009). The framework evaluates 16 preprocessing techniques:

| Code | Technique | Description |
|------|-----------|-------------|
| `snv` | Standard Normal Variate | Row-wise centering and scaling to remove scatter (Barnes et al., 1989) |
| `rsnv` | Robust SNV | SNV variant robust to outliers (Guo et al., 1999) |
| `msc` | Multiplicative Scatter Correction | Corrects multiplicative/additive scatter effects (Geladi et al., 1985) |
| `emsc` | Extended MSC | MSC with polynomial baseline modeling (Martens & Stark, 1991) |
| `savgol` | Savitzky-Golay (w=11, p=3) | Polynomial smoothing filter (Savitzky & Golay, 1964) |
| `savgol_d1` | Savitzky-Golay + 1st derivative | Smoothing with simultaneous differentiation (Savitzky & Golay, 1964) |
| `d1` | First Derivative | Removes baseline offset (Norris & Williams, 1984) |
| `d2` | Second Derivative | Removes linear baseline drift (Norris & Williams, 1984) |
| `haar` | Haar Wavelet | Multi-resolution decomposition (Mallat, 1989) |
| `detrend` | Detrend | Removes polynomial baseline trends (Barnes et al., 1989) |
| `gaussian` | Gaussian (σ=2, order=1) | Gaussian smoothing with 1st derivative |
| `gaussian2` | Gaussian (σ=2, order=2) | Gaussian smoothing with 2nd derivative |
| `area_norm` | Area Normalization | Normalizes spectra by total area |
| `wav_sym5` | Wavelet Symlet-5 | Symlet wavelet denoising (Daubechies, 1992) |
| `wav_coif3` | Wavelet Coiflet-3 | Coiflet wavelet denoising (Daubechies, 1992) |
| `identity` | Identity | No transformation (baseline reference) |

With *n* = 16 preprocessings, testing all combinations up to depth *d* requires evaluating:

$$N = \sum_{k=1}^{d} \frac{n!}{(n-k)!} = n + n(n-1) + n(n-1)(n-2) + ...$$

For 16 preprocessings at depth 3, this yields **4,320 combinations**—each requiring full cross-validation, which becomes computationally prohibitive.

### 1.2 Proposed Solution

Our framework uses a **4-stage funnel approach**:

1. **Stage 1**: Exhaustive unsupervised evaluation of all pipeline combinations
2. **Stage 2**: Diversity analysis with 6 distance metrics to identify complementary preprocessings
3. **Stage 3**: Proxy model validation (Ridge + KNN + XGBoost) on diverse candidates
4. **Stage 4**: Feature augmentation evaluation (2-way and 3-way concatenations)

This reduces full model evaluations from thousands to ~50-100 configurations while maintaining discovery of optimal solutions.

---

## 2. Methodology

### 2.1 Stage 1: Exhaustive Unsupervised Evaluation

All preprocessing configurations (single, stacked depth-2, stacked depth-3, etc.) are evaluated using five unsupervised metrics that correlate with downstream model performance.

#### 2.1.1 PCA Variance Ratio (VR)

Measures information preservation after preprocessing by analyzing the explained variance of the first $k$ principal components (Jolliffe, 2002).

**Formula:**
$$\text{VR} = \sum_{i=1}^{k} \lambda_i / \sum_{i=1}^{p} \lambda_i$$

where $\lambda_i$ are the eigenvalues of the covariance matrix of the transformed data $X$, and $k=10$ (fixed to standardize comparison).

**Interpretation:**
*   **Range:** [0, 1]
*   **Goal:** Higher is better.
*   **Meaning:** Indicates how much of the original spectral information is retained in the primary directions of variation. A very low VR suggests the preprocessing has destroyed the signal structure.

#### 2.1.2 Effective Dimensionality ($d_{eff}$)

Quantifies the intrinsic dimensionality of the data using the entropy of the eigenvalue distribution (Roy & Vetterli, 2007).

**Formula:**
$$d_{eff} = \exp\left(-\sum_{i=1}^p p_i \log p_i\right)$$

where $p_i = \lambda_i / \sum_j \lambda_j$ is the normalized eigenvalue (probability distribution).

**Interpretation:**
*   **Range:** [1, p]
*   **Goal:** Moderate (typically 5-20 for NIRS).
*   **Meaning:**
    *   **Too low (< 3):** Over-smoothing or loss of feature detail.
    *   **Too high (> 50):** Retention of high-frequency noise or lack of structure.
    *   **Optimal:** Balances complexity and parsimony.

#### 2.1.3 Signal-to-Noise Ratio (SNR)

Estimates the quality of the spectral signal by comparing the strength of the mean spectral features to the variability across samples (Workman & Weyer, 2007).

**Formula:**
$$\text{SNR} = \frac{\text{Var}(\bar{x})}{\frac{1}{P} \sum_{j=1}^P \text{Var}(x_j)}$$

where $\bar{x}$ is the mean spectrum (vector of means per wavelength), and $\text{Var}(x_j)$ is the variance of the $j$-th wavelength across samples.

**Interpretation:**
*   **Range:** [0, $\infty$)
*   **Goal:** Higher is better.
*   **Meaning:**
    *   **Numerator (Signal):** Variance of the mean spectrum. High values indicate strong spectral peaks and valleys (high contrast).
    *   **Denominator (Noise/Variability):** Average variance per wavelength. High values indicate large spread between samples.
    *   **Ratio:** Favors preprocessing that enhances spectral features (peaks) while reducing random sample-to-sample variability (noise).

#### 2.1.4 Roughness Score ($R$)

Measures the smoothness of the spectra by calculating the mean absolute second derivative. This metric is related to the second derivative preprocessing commonly used in spectroscopy (Savitzky & Golay, 1964; Rinnan et al., 2009).

**Formula:**
$$R = \frac{1}{N(P-2)} \sum_{i=1}^N \sum_{j=2}^{P-1} |x_{i,j+1} - 2x_{i,j} + x_{i,j-1}|$$

where $x_{i,j}$ is the intensity of sample $i$ at wavelength $j$.

**Interpretation:**
*   **Range:** [0, $\infty$)
*   **Goal:** Lower is better.
*   **Meaning:** High roughness indicates the presence of high-frequency noise or jagged artifacts. Smoothing techniques (Savitzky-Golay, Gaussian) explicitly minimize this.

#### 2.1.5 Separation Score ($S$)

Measures the discrimination capability of the data by evaluating the average distance between samples relative to the feature scale. This metric is inspired by the Fisher criterion and class separability measures (Fisher, 1936; Fukunaga, 1990).

**Formula:**
$$S = \frac{\frac{1}{|Pairs|} \sum_{i<j} d(x_i, x_j)}{\frac{1}{P} \sum_{k=1}^P \sigma_k}$$

where $d(x_i, x_j)$ is the Euclidean distance between samples, and $\sigma_k$ is the standard deviation of feature $k$.

**Interpretation:**
*   **Range:** [0, $\infty$)
*   **Goal:** Higher is better.
*   **Meaning:** Indicates how well-separated the samples are in the feature space. Higher separation often correlates with better classification/regression performance.

#### 2.1.6 Combined Unsupervised Score

Metrics are normalized to [0,1] and combined with specific weights to form a single ranking score. This weighted aggregation approach follows multi-criteria decision-making principles (Hwang & Yoon, 1981).

**Normalization:**
*   **Roughness:** $R_{norm} = 1 / (1 + R)$ (Inverted)
*   **SNR:** $SNR_{norm} = \text{clip}(\ln(1 + SNR) / 5, 0, 1)$ (Log-scaled)
*   **Effective Dim:** $D_{norm} = \text{clip}(d_{eff} / 10, 0, 1)$
*   **Separation:** $S_{norm} = \text{clip}(S / 10, 0, 1)$
*   **Variance Ratio:** Already in [0, 1]

**Weighting:**
$$S_{total} = 0.25 \cdot \text{VR} + 0.25 \cdot D_{norm} + 0.20 \cdot SNR_{norm} + 0.15 \cdot R_{norm} + 0.15 \cdot S_{norm}$$

**Output**: `stage1_unsupervised.csv` with all pipelines ranked by total score.

---

### 2.2 Stage 2: Diversity Analysis

To identify complementary preprocessings for feature augmentation, we compute pairwise distances using **6 distance metrics** grouped into two categories.

#### 2.2.1 Subspace-Based Metrics

These metrics compare the principal component structure (linear subspaces) of transformed data.

##### Grassmann Distance ($d_G$)
Measures the angular distance between the principal subspaces of two datasets (Ye & Janardan, 2005).

**Formula:**
$$d_G(X_1, X_2) = \sqrt{\sum_{i=1}^{k} \theta_i^2}$$

where $\theta_i$ are the principal angles between the $k$-dimensional subspaces spanned by the first $k$ principal components of $X_1$ and $X_2$. Normalized by $\sqrt{k(\pi/2)^2}$.

**Interpretation:** High distance implies the preprocessings highlight different linear combinations of features.

##### CKA Distance ($d_{CKA}$)
Centered Kernel Alignment measures representation similarity, invariant to orthogonal transformations and isotropic scaling (Kornblith et al., 2019).

**Formula:**
$$\text{CKA}(K_1, K_2) = \frac{\text{HSIC}(K_1, K_2)}{\sqrt{\text{HSIC}(K_1, K_1) \cdot \text{HSIC}(K_2, K_2)}}$$
$$d_{CKA} = 1 - \text{CKA}$$

where $K_i = X_i X_i^T$ (linear kernel) and HSIC is the Hilbert-Schmidt Independence Criterion.

**Interpretation:** Measures global structural similarity between representations.

##### RV Coefficient Distance ($d_{RV}$)
A multivariate generalization of the squared Pearson correlation coefficient (Robert & Escoufier, 1976).

**Formula:**
$$\text{RV}(X_1, X_2) = \frac{\text{tr}(A^T B)}{\sqrt{\text{tr}(A^T A) \cdot \text{tr}(B^T B)}}$$
$$d_{RV} = 1 - \text{RV}$$

where $A = X_1 X_1^T$ and $B = X_2 X_2^T$ are centered Gram matrices.

**Interpretation:** Measures the correlation between the configuration of sample points.

#### 2.2.2 Geometry-Based Metrics

These metrics compare the local and global geometry of the sample distributions.

##### Procrustes Distance ($d_P$)
Measures the residual sum of squares after optimal linear alignment (translation, rotation, scaling) (Gower, 1975).

**Formula:**
$$d_P = \min_{R,s,t} \|Z_1 - s Z_2 R - t\|_F^2$$

Applied to the PCA scores $Z_1, Z_2$.

**Interpretation:** Captures differences in the "shape" of the point clouds that cannot be resolved by simple linear transformation.

##### Trustworthiness Distance ($d_T$)
Measures how well the local neighborhood structure is preserved (Venna & Kaski, 2001).

**Formula:**
$$T(k) = 1 - \frac{2}{Nk(2N-3k-1)} \sum_{i=1}^N \sum_{j \in U_k(i) \setminus V_k(i)} (r(i,j) - k)$$
$$d_T = 1 - T(k)$$

where $r(i,j)$ is the rank of sample $j$ in the neighborhood of $i$ in the original space, $U_k(i)$ is the set of $k$ nearest neighbors in the transformed space, and $V_k(i)$ is the set of $k$ nearest neighbors in the original space.

**Interpretation:** High distance indicates that the local neighborhood relationships (topology) are significantly different.

##### Covariance Distance ($d_{cov}$)
Measures the difference between the covariance structures using the Frobenius norm (Golub & Van Loan, 2013).

**Formula:**
$$d_{cov} = \frac{\|\Sigma_1 - \Sigma_2\|_F}{\sqrt{\|\Sigma_1\|_F \|\Sigma_2\|_F}}$$

where $\Sigma_i$ is the covariance matrix of $X_i$.

**Interpretation:** Captures differences in feature correlations and variance scales.

#### 2.2.3 Combined Distance & Filtering

**Combined Distance:**
$$d_{combined} = 0.5 \cdot \underbrace{(0.4 d_G + 0.4 d_{CKA} + 0.2 d_{RV})}_{\text{Subspace}} + 0.5 \cdot \underbrace{(0.4 d_P + 0.3 d_T + 0.3 d_{cov})}_{\text{Geometry}}$$

**Similarity Filtering:**
Pipelines are filtered if they are too similar to a better-performing pipeline (based on Stage 1 score).
*   **Threshold:** $1.0 - \text{similarity\_ratio}$ (default ratio 0.95 $\rightarrow$ distance threshold 0.05).
*   **Logic:** If $d_{combined}(P_{better}, P_{candidate}) < \text{threshold}$, discard $P_{candidate}$.

---

### 2.3 Stage 3: Proxy Model Evaluation

Top diverse candidates from Stage 2 are evaluated using fast "proxy" models to estimate downstream performance.

#### 2.3.1 Models

1.  **Ridge Regression:** Linear model with L2 regularization (Hoerl & Kennard, 1970).
    *   $\alpha \in \{0.1, 1.0, 10.0, 100.0\}$ selected via CV.
    *   Captures linear relationships.
2.  **K-Nearest Neighbors (KNN):** Non-parametric instance-based learning (Cover & Hart, 1967).
    *   $k=5$, distance-weighted.
    *   Captures local non-linear manifold structure.
3.  **XGBoost:** Gradient Boosted Decision Trees (Chen & Guestrin, 2016).
    *   50 estimators, max depth 3.
    *   Captures complex non-linear interactions and feature importance.

#### 2.3.2 Evaluation Protocol

*   **Cross-Validation:** 3-fold CV (Kohavi, 1995).
*   **Metrics:**
    *   **Regression:** $R^2$ score (Draper & Smith, 1998).
    *   **Classification:** Accuracy.
*   **Proxy Score:**
    $$S_{proxy} = 0.4 \cdot S_{Ridge} + 0.3 \cdot S_{KNN} + 0.3 \cdot S_{XGB}$$

#### 2.3.3 Final Score Calculation

$$S_{final} = 0.4 \cdot S_{unsupervised} + 0.6 \cdot S_{proxy}$$

**Output**: `stage3_proxy.csv`

---

### 2.4 Stage 4: Feature Augmentation

We test **feature augmentation** by concatenating transformed feature sets from complementary preprocessings. This approach is inspired by multi-view learning and feature fusion strategies (Xu et al., 2013).

#### 2.4.1 Augmentation Strategy

1.  **2nd-Order:** Concatenate all pairs of top Stage 3 candidates.
    $$X_{aug} = [X_{prep1} \parallel X_{prep2}]$$
2.  **3rd-Order:** Concatenate triplets (limited to top 50 combinations).
    $$X_{aug} = [X_{prep1} \parallel X_{prep2} \parallel X_{prep3}]$$

#### 2.4.2 Evaluation

Augmented datasets are evaluated using the same proxy models as Stage 3.
$$S_{aug} = 0.4 \cdot S_{unsupervised}(X_{aug}) + 0.6 \cdot S_{proxy}(X_{aug})$$

**Output**: `stage4_augmentation.csv`

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

## 4. Example Application

The following example illustrates the application of the framework to a digestibility prediction dataset (2856 samples, 949 wavelengths). The final ranking output demonstrates the typical format of results:

```
======================================================================
FINAL RANKING
======================================================================

Top 15 Configurations:
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
- 2-way augmentation [snv+savgol>snv] appears in top 5
- The marginal differences suggest robustness to preprocessing choice

---

## 4. Computational Complexity

The computational complexity of each stage is summarized below. Let $n$ denote the number of preprocessing techniques, $N$ the number of samples, $P$ the number of features (wavelengths), $k$ the number of candidates passed between stages, and $d$ the maximum stacking depth.

| Stage | Complexity | Description |
|-------|------------|-------------|
| Stage 1 | $O(n^d \times N \times P)$ | Exhaustive unsupervised evaluation of all pipeline combinations |
| Stage 2 | $O(k^2 \times 6 \times \text{PCA cost})$ | Pairwise distance computation with six metrics |
| Stage 3 | $O(k \times \text{CV} \times 3 \times \text{model cost})$ | Cross-validated proxy model evaluation |
| Stage 4 | $O(\binom{k}{2} + \binom{k}{3}) \times \text{CV} \times \text{proxy cost}$ | Augmentation evaluation |

The total computational cost is significantly reduced compared to exhaustive cross-validation of all $n^d$ combinations with full model training. The speedup factor depends on the ratio of proxy model complexity to full model complexity and the filtering efficiency of each stage.

---

## 5. Recommendations

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

## 6. Algorithm Summary

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

## 7. References

1. Barnes, R. J., Dhanoa, M. S., & Lister, S. J. (1989). Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy*, 43(5), 772-777. DOI: 10.1366/0003702894202201

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. DOI: 10.1145/2939672.2939785

3. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27. DOI: 10.1109/TIT.1967.1053964

4. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM. DOI: 10.1137/1.9781611970104

5. Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis* (3rd ed.). Wiley. DOI: 10.1002/9781118625590

6. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188. DOI: 10.1111/j.1469-1809.1936.tb02137.x

7. Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition* (2nd ed.). Academic Press. DOI: 10.1016/C2009-0-27872-X

8. Geladi, P., MacDougall, D., & Martens, H. (1985). Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy*, 39(3), 491-500. DOI: 10.1366/0003702854248656

9. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

10. Gower, J. C. (1975). Generalized Procrustes analysis. *Psychometrika*, 40(1), 33-51. DOI: 10.1007/BF02291478

11. Guo, Q., Wu, W., & Massart, D. L. (1999). The robust normal variate transform for pattern recognition with near-infrared data. *Analytica Chimica Acta*, 382(1-2), 87-103. DOI: 10.1016/S0003-2670(98)00737-5

12. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. DOI: 10.1080/00401706.1970.10488634

13. Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer. DOI: 10.1007/978-3-642-48318-9

14. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer. DOI: 10.1007/b98835

15. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 1137-1145.

16. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 3519-3529.

17. Mallat, S. G. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693. DOI: 10.1109/34.192463

18. Martens, H., & Stark, E. (1991). Extended multiplicative signal correction and spectral interference subtraction: New preprocessing methods for near infrared spectroscopy. *Journal of Pharmaceutical and Biomedical Analysis*, 9(8), 625-635. DOI: 10.1016/0731-7085(91)80188-F

19. Norris, K. H., & Williams, P. C. (1984). Optimization of mathematical treatments of raw near-infrared signal in the measurement of protein in hard red spring wheat. I. Influence of particle size. *Cereal Chemistry*, 61(2), 158-165.

20. Rinnan, Å., van den Berg, F., & Engelsen, S. B. (2009). Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222. DOI: 10.1016/j.trac.2009.07.007

21. Robert, P., & Escoufier, Y. (1976). A unifying tool for linear multivariate statistical methods: The RV-coefficient. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 25(3), 257-265. DOI: 10.2307/2347233

22. Roy, O., & Vetterli, M. (2007). The effective rank: A measure of effective dimensionality. *15th European Signal Processing Conference*, 606-610.

23. Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639. DOI: 10.1021/ac60214a047

24. Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study. *Artificial Neural Networks—ICANN 2001*, Lecture Notes in Computer Science, Vol. 2130, 485-491. DOI: 10.1007/3-540-44668-0_68

25. Workman, J., & Weyer, L. (2007). *Practical Guide to Interpretive Near-Infrared Spectroscopy*. CRC Press. DOI: 10.1201/9781420018318

26. Xu, C., Tao, D., & Xu, C. (2013). A survey on multi-view learning. *arXiv preprint arXiv:1304.5634*.

27. Ye, J., & Janardan, R. (2005). Generalized linear discriminant analysis: A unified framework and efficient model selection. *IEEE Transactions on Neural Networks*, 16(2), 241-252. DOI: 10.1109/TNN.2004.841414

---

*Technical report — nirs4all preprocessing selection framework*

*© 2025 CIRAD — UMR AGAP Institut*

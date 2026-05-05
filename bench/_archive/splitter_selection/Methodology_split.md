# Methodology: Splitting Strategy Comparison for NIRS Spectral Data

**Authors**: G. Beurier¹*, L. Rouan¹, D. Cornet¹

**Affiliation**: ¹CIRAD, UMR AGAP Institut, F-34398 Montpellier, France

**Corresponding author**: G. Beurier (beurier@cirad.fr)

**Version**: 1.0 — November 2025

---

## Abstract

This document describes the methodology used to compare different data splitting strategies for Near-Infrared Spectroscopy (NIRS) predictive modeling. The goal is to identify the optimal splitting approach that maximizes model generalization while respecting the inherent structure of repeated measurements. The framework implements 16 splitting strategies from the chemometrics literature, including Kennard-Stone (Kennard & Stone, 1969), SPXY (Galvão et al., 2005), Duplex (Snee, 1977), Puchwein (Puchwein, 1988), and Shenk-Westerhaus (Shenk & Westerhaus, 1991) algorithms. Performance is evaluated using a multi-model ensemble with bootstrap confidence intervals (Efron & Tibshirani, 1993) and representativeness metrics.

**Keywords**: Near-infrared spectroscopy, train-test splitting, cross-validation, chemometrics, calibration transfer

---

## 1. Introduction

## 2. Problem Statement

### 2.1 Context

NIRS spectral data often contains **replicated measurements** (repetitions) of the same physical sample. When building predictive models, a critical constraint is to prevent **data leakage**: all repetitions of a given sample must remain in the same partition (train or test, and within the same fold during cross-validation).

### 2.2 Objectives

1. Compare multiple splitting strategies adapted to spectral data
2. Ensure group integrity (no sample leakage between partitions)
3. Evaluate model performance using an ensemble of diverse regressors
4. Quantify uncertainty via bootstrap confidence intervals
5. Assess split quality via representativeness metrics (spectral coverage, target distribution, leverage)
6. Provide statistical evidence for strategy selection

## 3. Data Structure

| File | Description |
|------|-------------|
| `X.csv` | Spectral features matrix (wavelengths as columns) |
| `Y.csv` | Target variable (e.g., Digestibility) |
| `M.csv` | Metadata including sample ID and repetition number |

### 3.1 Grouping Constraint

Each unique sample ID has multiple repetitions (Rep 1, 2, 3, 4...). The splitting is performed at the **sample level**, not at the observation level, to prevent information leakage.

### 3.2 Aggregation for Splitting Decisions

Before applying any splitting strategy, observations are aggregated by sample ID:

$$\bar{x}_i = \frac{1}{r_i} \sum_{j=1}^{r_i} x_{i,j}$$

where $\bar{x}_i$ is the mean spectrum for sample $i$, $r_i$ is the number of repetitions, and $x_{i,j}$ is the $j$-th repetition of sample $i$.

This aggregation ensures splitting decisions are made at the sample level. After splitting, all repetitions of each sample are assigned to the same partition.

## 4. Splitting Strategies

The splitter selection suite includes **16 splitting strategies** organized into categories. All strategies operate on aggregated sample-level data and respect group integrity.

### 4.1 Baseline Strategies

#### Simple Random Split
**Principle**: Samples are randomly assigned to train/test sets at sample ID level using a permutation of unique sample IDs.

- **Algorithm**: Random permutation followed by sequential split
- **Complexity**: O(n)
- **Limitation**: No guarantee of representative distribution of spectral features or target values

### 4.2 Target-Based Strategies

#### Target Stratified Split
**Principle**: Maintains the target variable distribution across partitions using quantile-based binning (Kohavi, 1995).

**Algorithm**:
1. Compute quantile bins: $q_k = F_Y^{-1}(k/K)$ for $k = 0, 1, ..., K$ where $K$ is the number of bins (default: 5)
2. Assign each sample to bin $b_i = \lfloor K \cdot F_Y(y_i) \rfloor$
3. Within each bin, randomly sample proportionally for train/test

**Advantage**: Ensures similar target distributions in train and test sets, reducing selection bias.

#### Stratified Group KFold
**Principle**: Uses sklearn's `StratifiedGroupKFold` combining stratification by binned target with group-aware splitting (Pedregosa et al., 2011).

**Algorithm**: Internally uses stratified sampling while ensuring all observations from the same group (sample ID) stay together.

### 4.3 Spectral-Based Strategies

#### Spectral PCA Split
**Principle**: Leverages PCA-reduced spectral similarity for clustering, then stratifies by cluster membership (Bro & Smilde, 2014).

**Algorithm**:
1. Apply PCA: $Z = XW_k$ where $W_k$ contains the first $k$ principal components (retaining 95% variance)
2. Apply K-Means clustering in PCA space: $c_i = \arg\min_j \|z_i - \mu_j\|^2$
3. Stratify by cluster: ensure proportional representation of each cluster in train/test

**Advantage**: Ensures spectral diversity in both partitions without requiring explicit distance calculations.

#### Spectral Distance Split (Farthest Point Sampling)
**Principle**: Uses farthest point sampling to select spectrally diverse test samples (Eldar et al., 1997).

**Algorithm**:
1. Apply PCA to reduce dimensionality to $k$ components
2. Initialize: select a random sample as first test point
3. Iteratively select the sample with maximum minimum distance to already selected samples:
   $$s_{t+1} = \arg\max_{i \notin S} \min_{j \in S} d(z_i, z_j)$$
   where $d$ is Euclidean distance in PCA space
4. Continue until desired test set size is reached

**Advantage**: Test set covers extreme regions of spectral space, challenging model extrapolation.

### 4.4 Hybrid Strategies

#### Hybrid Split
**Principle**: Combines spectral clustering with target stratification to balance both sources of variation.

**Algorithm**:
1. Compute spectral clusters via PCA + K-Means ($C$ clusters)
2. Compute target bins via quantiles ($B$ bins)
3. Create combined strata: $s_i = c_i \times B + b_i$ (up to $C \times B$ strata)
4. Stratify by combined strata, ensuring proportional representation

**Advantage**: Balances both spectral and target representativeness; reduces risk of bias from either source alone.

### 4.5 Robustness Strategies

#### Adversarial Split
**Principle**: Creates challenging test sets by including samples that are spectrally distant from the training set centroid.

**Algorithm**:
1. Apply PCA to reduce dimensionality
2. Compute distance to centroid: $d_i = \|z_i - \bar{z}\|$
3. Sort samples by distance (descending)
4. Select top $\alpha \cdot n_{test}$ samples as adversarial (where $\alpha$ is adversarial strength, default: 0.5)
5. Fill remaining test slots randomly

**Advantage**: Tests model robustness to unusual/outlier samples; identifies extrapolation behavior.

### 4.6 Chemometrics Strategies

#### Kennard-Stone Algorithm
**Principle**: Sequential selection maximizing coverage of the spectral feature space (Kennard & Stone, 1969).

**Algorithm**:
1. Compute pairwise Euclidean distances: $D_{ij} = \|x_i - x_j\|$
2. Select the pair with maximum distance: $(s_1, s_2) = \arg\max_{i,j} D_{ij}$
3. Iteratively add the sample with maximum minimum distance to selected set:
   $$s_{t+1} = \arg\max_{i \notin S} \min_{j \in S} D_{ij}$$
4. Continue until training set size is reached

**Advantage**: Optimal coverage of the spectral feature space; ensures training data spans the measurement domain.

#### SPXY (Sample set Partitioning based on joint X-Y distances)
**Principle**: Extends Kennard-Stone by considering both spectral (X) and target (Y) information (Galvão et al., 2005).

**Algorithm**:
1. Normalize X and Y distances to [0, 1] range
2. Compute combined distance: $D_{ij}^{XY} = D_{ij}^X / \max(D^X) + D_{ij}^Y / \max(D^Y)$
3. Apply Kennard-Stone selection using combined distance matrix

**Advantage**: Ensures training set covers both the spectral domain and the target value range.

#### Puchwein Algorithm
**Principle**: Distance-based sample selection using iterative elimination of spectrally similar samples (Puchwein, 1988).

**Algorithm**:
1. Transform data via PCA and scale by eigenvalues (approximating Mahalanobis distance)
2. Sort samples by distance to mean (farthest first)
3. Set initial threshold $d_m = k \cdot (p - 2)$ where $k$ is factor (default: 0.05), $p$ is number of components
4. Iteratively remove samples within threshold distance of selected samples
5. Increase threshold until target training set size is reached

**Advantage**: Systematic coverage with tunable sensitivity; reduces redundancy in training set.

#### Duplex Algorithm
**Principle**: Alternating assignment to train and test sets based on maximum distance, ensuring both sets have good coverage (Snee, 1977).

**Algorithm**:
1. Transform to Mahalanobis-like space (PCA + eigenvalue scaling)
2. Select the two most distant samples, assign to training set
3. Select the next two most distant samples from remainder, assign to test set
4. Alternate: add farthest sample from remaining to training, then to test
5. Continue until all samples are assigned

**Advantage**: Both train and test sets receive representative samples spanning the full spectral space.

#### Shenk-Westerhaus Algorithm
**Principle**: Distance-based selection using adaptive threshold determination via bisection (Shenk & Westerhaus, 1991).

**Algorithm**:
1. Transform to PCA space, standardize
2. Optionally remove outliers (samples with $d_i / p > 3$)
3. Use bisection to find threshold $\theta$ such that selecting samples with pairwise distance $> \theta$ yields target training size
4. Iteratively select samples, removing those within threshold distance

**Advantage**: Well-established method in NIR chemometrics; adaptive threshold selection.

#### Honigs Algorithm
**Principle**: Selection based on spectral uniqueness via orthogonal projection (Honigs et al., 1985).

**Algorithm**:
1. Find the sample-wavelength pair $(i^*, j^*)$ with maximum absolute value: $\arg\max_{i,j} |X_{ij}|$
2. Add sample $i^*$ to training set
3. Remove contribution of $i^*$ from all spectra via projection:
   $$X' = X - \frac{X_{:,j^*}}{X_{i^*,j^*}} \cdot X_{i^*,:}$$
4. Remove row $i^*$ and column $j^*$ from matrix
5. Repeat until training size reached

**Advantage**: Prioritizes spectrally unique samples; natural for spectral data with distinct absorption features.

### 4.7 Clustering Strategies

#### Hierarchical Clustering Split
**Principle**: Agglomerative clustering-based selection using complete linkage (Murtagh & Contreras, 2012).

**Algorithm**:
1. Transform to Mahalanobis-like space (PCA + eigenvalue scaling)
2. Apply agglomerative clustering with complete linkage, $n_{train}$ clusters
3. For each cluster, select the medoid (sample minimizing maximum distance to other cluster members):
   $$m_c = \arg\min_{i \in C_c} \max_{j \in C_c} d_{ij}$$
4. Selected medoids form training set; remaining samples form test set

**Advantage**: Captures hierarchical structure; complete linkage ensures compact clusters.

#### K-Medoids Split
**Principle**: K-Medoids (PAM-like) selection ensuring selected samples are actual observations (Kaufman & Rousseeuw, 1990).

**Algorithm**:
1. Transform to Mahalanobis-like space
2. Apply K-Means to get initial cluster assignments ($n_{train}$ clusters)
3. For each cluster, select the medoid (minimizing maximum distance within cluster)
4. Medoids form training set

**Advantage**: More robust to outliers than K-Means centroids; selected samples are real observations.

## 5. Evaluation Protocol

### 5.1 Train/Test Split

- **Ratio**: 80% training, 20% test (configurable)
- **Constraint**: All repetitions of a sample stay together

### 5.2 Cross-Validation with Repeated Evaluation

- **Method**: K-Fold cross-validation on training set
- **Folds**: 3 (configurable)
- **Repeats**: 3 independent repetitions with different random seeds (42, 123, 456)
- **Constraint**: Group-aware folding ensures no sample leakage across folds

**Rationale**: Repeated CV provides more stable estimates of model variance and reduces sensitivity to a single random fold assignment.

### 5.3 Model Ensemble

The evaluation uses a diverse suite of models spanning different algorithmic families (Wold et al., 2001; Pedregosa et al., 2011):

| Model | Category | Key Hyperparameters | Rationale |
|-------|----------|---------------------|-----------|
| **RidgeCV** | Linear | α ∈ [10⁻⁴, 10⁴] (20 values, log-spaced) | L2 regularization; handles multicollinearity in spectral data |
| **ElasticNetCV** | Linear | α ∈ [10⁻⁴, 10²], ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} | Combines L1 and L2; potential feature selection |
| **PLSRegression** | Linear | n_components = 15 | Standard in chemometrics; latent variable modeling |
| **SVR (RBF)** | Kernel | C = 100, γ = scale, ε = 0.01 | Non-linear; robust to outliers |
| **XGBoostRegressor** | Ensemble | n_est = 200, depth = 6, η = 0.03, L1/L2 regularization | Gradient boosting; captures complex interactions |
| **MLPRegressor** | Deep | Layers: (256, 128, 64), ReLU, Adam, early stopping | Neural network; non-linear feature learning |

**Data preprocessing**: All models use StandardScaler normalization (zero mean, unit variance) applied on training data and transformed on validation/test data to prevent data leakage.

**Ensemble prediction**: Final prediction is the arithmetic mean of all individual model predictions:
$$\hat{y}_{ensemble} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m$$

**Rationale**: Using multiple models reduces dependence on a single algorithm's inductive bias and provides a more robust assessment of split quality.

### 5.4 Performance Metrics

#### 5.4.1 Root Mean Square Error (RMSE)

**Definition**:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Interpretation**: Measures average prediction error in the same units as the target variable. Penalizes large errors more than MAE due to squaring.

#### 5.4.2 Mean Absolute Error (MAE)

**Definition**:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interpretation**: Measures average absolute deviation; more robust to outliers than RMSE.

#### 5.4.3 Coefficient of Determination (R²)

**Definition**:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

**Interpretation**: Proportion of variance explained by the model (Draper & Smith, 1998). Values range from $-\infty$ to 1; negative values indicate worse-than-mean prediction. Higher is better.

#### 5.4.4 Generalization Gap

**Definition**:
$$\Delta_{gen} = \text{RMSE}_{test} - \overline{\text{RMSE}}_{CV}$$

**Interpretation**: Difference between test set performance and cross-validation estimate. A large positive gap indicates overfitting or non-representative split. A negative gap may indicate the test set is "easier" than training folds.

**Ideal value**: Close to zero, indicating CV accurately predicts test performance.

#### 5.4.5 CV Stability (Coefficient of Variation)

**Definition**:
$$\text{CV}_{stability} = \frac{\sigma_{RMSE,CV}}{\mu_{RMSE,CV}}$$

**Interpretation**: Relative variability of RMSE across CV folds and repeats. Lower values indicate more stable, reproducible results. High variability suggests sensitivity to fold composition.

## 6. Representativeness Metrics

These metrics assess the quality of the train/test split independently of model performance.

### 6.1 Spectral Coverage

**Definition**: Proportion of test samples falling within the convex hull of training samples in PCA space.

**Algorithm**:
1. Compute PCA on combined data (10 components)
2. Use first 3 components for convex hull computation
3. Compute Delaunay triangulation of training points
4. Count test points inside the hull

**Formula**:
$$\text{Coverage} = \frac{|\{x_i^{test} : x_i^{test} \in \text{Hull}(X^{train}_{PCA})\}|}{n_{test}}$$

**Interpretation**: Values close to 1 indicate test samples are interpolations; low values indicate extrapolation risk. The convex hull computation uses the Quickhull algorithm (Barber et al., 1996).

**Fallback**: If Delaunay triangulation fails (degenerate case), uses distance-based coverage: proportion of test samples within the maximum training radius from centroid.

### 6.2 Target Distribution Similarity

#### 6.2.1 Wasserstein Distance (Earth Mover's Distance)

**Definition**: Minimum "work" required to transform the training target distribution into the test target distribution.

$$W_1(P_{train}, P_{test}) = \int_{-\infty}^{\infty} |F_{train}(y) - F_{test}(y)| dy$$

**Interpretation**: Lower values indicate more similar distributions (Villani, 2008). Units are the same as the target variable.

#### 6.2.2 Kullback-Leibler Divergence

**Definition**: Information-theoretic measure of difference between discretized target distributions.

**Algorithm**:
1. Discretize targets into 50 bins spanning the combined range
2. Compute normalized histograms $P$ (train) and $Q$ (test)
3. Add smoothing ($\varepsilon = 10^{-10}$) to avoid log(0)
4. Compute $D_{KL}(Q || P)$

**Formula**:
$$D_{KL}(Q || P) = \sum_{i} Q(i) \log \frac{Q(i)}{P(i)}$$

**Interpretation**: Measures information lost when using training distribution to approximate test distribution (Kullback & Leibler, 1951). Lower is better. Asymmetric: $D_{KL}(Q||P) \neq D_{KL}(P||Q)$.

### 6.3 Leverage Analysis (Extrapolation Risk)

Leverage measures how "unusual" a test sample is relative to the training distribution. High-leverage samples may lead to unreliable predictions (extrapolation).

#### 6.3.1 Hotelling's T² Statistic

**Algorithm**:
1. Fit PCA on training data (10 components)
2. Transform test data using fitted PCA
3. Compute training covariance matrix $\Sigma_{train}$ in PCA space
4. Center test data: $z_i^{test} - \bar{z}^{train}$
5. Compute T² for each test sample:

$$T^2_i = (z_i - \bar{z})^T \Sigma^{-1} (z_i - \bar{z})$$

**Interpretation**: Measures multivariate distance from training centroid, accounting for correlations (Hotelling, 1931). Higher values indicate greater risk of extrapolation.

#### 6.3.2 Leverage (Hat Matrix Diagonal)

**Definition**: Approximate leverage from PCA-transformed data.

$$h_i = \frac{1}{n_{train}} + \frac{T^2_i}{n_{train} - 1}$$

**Threshold for high leverage**: $h > \frac{2(p+1)}{n_{train}}$ where $p$ is the number of PCA components.

**Metrics reported**:
- `leverage_mean`: Average leverage of test samples
- `leverage_max`: Maximum leverage (worst-case extrapolation)
- `n_high_leverage`: Number of test samples exceeding threshold
- `hotelling_t2_mean`: Mean T² score

**Interpretation**: Higher mean leverage indicates test set is more "unusual" relative to training (Belsley et al., 1980). Many high-leverage samples suggest potential extrapolation problems.

## 7. Statistical Inference

### 7.1 Bootstrap Confidence Intervals

**Purpose**: Quantify uncertainty in test set metrics without parametric assumptions.

**Algorithm**:
1. For $B = 1000$ bootstrap iterations:
   - Sample $n_{test}$ indices with replacement from test set
   - Compute RMSE, MAE, R² on bootstrap sample
2. Compute percentile confidence intervals at 95% level:
   - Lower bound: 2.5th percentile
   - Upper bound: 97.5th percentile

**Rationale**: Bootstrap CI accounts for sampling variability in the test set and provides interpretable uncertainty ranges (Efron & Tibshirani, 1993).

### 7.2 Pairwise Statistical Tests

For comparing strategies, the following tests are computed using CV fold results (pooled across repeats):

#### 7.2.1 Independent Samples t-test

**Hypothesis**: $H_0: \mu_1 = \mu_2$ (strategies have equal mean RMSE)

**Statistic**:
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

**Interpretation**: p-value < 0.05 indicates statistically significant difference.

#### 7.2.2 Mann-Whitney U Test

**Purpose**: Non-parametric alternative when normality assumption is violated.

**Hypothesis**: $H_0$: distributions are equal.

**Interpretation**: More robust to outliers and non-normal distributions.

#### 7.2.3 Effect Size (Cohen's d)

**Definition**:
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

where $s_{pooled} = \sqrt{(s_1^2 + s_2^2)/2}$

**Interpretation**:
- $|d| < 0.2$: negligible effect
- $0.2 \leq |d| < 0.5$: small effect
- $0.5 \leq |d| < 0.8$: medium effect
- $|d| \geq 0.8$: large effect

These thresholds follow the conventions established by Cohen (1988).

## 8. Analysis Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Data                               │
│  (X: spectra, Y: target, M: metadata with sample IDs)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregate by Sample ID                         │
│  (Mean spectrum per sample for splitting decisions)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Apply Splitting Strategy                          │
│  (16 strategies: Random, Stratified, Spectral, Hybrid,      │
│   Chemometrics, Clustering, Adversarial)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Expand to All Repetitions                      │
│  (Map sample-level splits back to all observations)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Repeated K-Fold Cross-Validation                  │
│  (3 folds × 3 repeats = 9 fold evaluations)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Train Model Ensemble per Fold                     │
│  (Ridge, ElasticNet, PLS, SVR, XGBoost, MLP)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Aggregate Test Predictions                        │
│  (Ensemble of all fold models → test predictions)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│      Compute Metrics + Bootstrap CIs + Representativeness   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Compare and Visualize                          │
│  (Ranking, pairwise tests, visualizations)                  │
└─────────────────────────────────────────────────────────────┘
```

## 9. Strategy Selection Criteria

### 9.1 Primary Criteria

1. **Test RMSE (with 95% CI)**: Lower is better; CIs should not overlap with inferior strategies
2. **Generalization Gap**: Should be close to zero; large positive values indicate overfitting
3. **CV Stability**: Lower coefficient of variation indicates robust splitting

### 9.2 Secondary Criteria (Representativeness)

4. **Spectral Coverage**: Higher indicates test samples are interpolations (safer predictions)
5. **Target Wasserstein Distance**: Lower indicates balanced target distributions
6. **High Leverage Count**: Lower indicates fewer extrapolation samples

### 9.3 Multi-Criteria Decision

The "best" strategy depends on the use case:
- **General use**: Minimize test RMSE while keeping generalization gap small
- **Robustness testing**: Consider adversarial splits with low spectral coverage
- **Conservative deployment**: Maximize spectral coverage and minimize leverage

## 10. Implementation Notes

### 10.1 Files

| File | Purpose |
|------|---------|
| `splitter_strategies.py` | Splitting algorithm implementations (16 strategies) |
| `unsupervised_splitters.py` | Chemometrics algorithms (Puchwein, Duplex, Shenk-West, Honigs, clustering) |
| `splitter_evaluation_enhanced.py` | Multi-model evaluation, bootstrap CIs, representativeness metrics |
| `splitter_visualization.py` | Results visualization |
| `run_splitter_selection.py` | Main execution script |

### 10.2 Usage

```bash
python run_splitter_selection.py --data_dir path/to/data --output_dir results/
python run_splitter_selection.py --data_dir ./data --test_size 0.2
python run_splitter_selection.py --data_dir ./data --n_folds 5
```

### 10.3 Reproducibility

- Random seeds: 42 (primary), 123, 456 (for repeated CV)
- All operations use `np.random.RandomState` for reproducibility
- Split indices are saved for replication

## 11. Limitations and Assumptions

1. **Sample independence**: Assumes samples are independent after controlling for repetitions
2. **Stationarity**: Assumes data distribution does not change over time
3. **Metric choice**: RMSE emphasizes large errors; other metrics may be preferred for specific applications
4. **Model suite**: Results depend on chosen model ensemble; different models may favor different splits
5. **PCA components**: Default 10 components for representativeness metrics; may need adjustment for high-dimensional data
6. **Bootstrap**: Assumes test set is representative of deployment population

## 12. References

### Splitting Algorithms

1. Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments. *Technometrics*, 11(1), 137-148. DOI: 10.1080/00401706.1969.10490666

2. Galvão, R. K. H., Araujo, M. C. U., José, G. E., Pontes, M. J. C., Silva, E. C., & Saldanha, T. C. B. (2005). A method for calibration and validation subset partitioning. *Talanta*, 67(4), 736-740. DOI: 10.1016/j.talanta.2005.03.025

3. Snee, R. D. (1977). Validation of regression models: Methods and examples. *Technometrics*, 19(4), 415-428. DOI: 10.1080/00401706.1977.10489581

4. Puchwein, G. (1988). Selection of calibration samples for near-infrared spectrometry by factor analysis of spectra. *Analytical Chemistry*, 60(6), 569-573. DOI: 10.1021/ac00157a015

5. Shenk, J. S., & Westerhaus, M. O. (1991). Population definition, sample selection, and calibration procedures for near infrared reflectance spectroscopy. *Crop Science*, 31(2), 469-474. DOI: 10.2135/cropsci1991.0011183X003100020049x

6. Honigs, D. E., Hieftje, G. M., Mark, H. L., & Hirschfeld, T. B. (1985). Unique-sample selection via near-infrared spectral subtraction. *Analytical Chemistry*, 57(12), 2299-2303. DOI: 10.1021/ac00289a029

7. Fonseca Diaz, V., De Ketelaere, B., Aernouts, B., & Saeys, W. (2021). Cost-efficient unsupervised sample selection for multivariate calibration. *Chemometrics and Intelligent Laboratory Systems*, 215, 104352. DOI: 10.1016/j.chemolab.2021.104352

### Statistical Methods

8. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. DOI: 10.1007/978-1-4899-4541-9

9. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge. DOI: 10.4324/9780203771587

10. Hotelling, H. (1931). The generalization of Student's ratio. *Annals of Mathematical Statistics*, 2(3), 360-378. DOI: 10.1214/aoms/1177732979

11. Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79-86. DOI: 10.1214/aoms/1177729694

12. Villani, C. (2008). *Optimal Transport: Old and New*. Springer. DOI: 10.1007/978-3-540-71050-9

13. Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis* (3rd ed.). Wiley. DOI: 10.1002/9781118625590

14. Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. Wiley. DOI: 10.1002/0471725153

### Chemometrics

15. Bro, R., & Smilde, A. K. (2014). Principal component analysis. *Analytical Methods*, 6(9), 2812-2831. DOI: 10.1039/C3AY41907J

16. Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: A basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109-130. DOI: 10.1016/S0169-7439(01)00155-1

17. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 1137-1145.

18. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

### Clustering and Geometry

19. Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: An overview. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 2(1), 86-97. DOI: 10.1002/widm.53

20. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. DOI: 10.1002/9780470316801

21. Barber, C. B., Dobkin, D. P., & Huhdanpaa, H. (1996). The Quickhull algorithm for convex hulls. *ACM Transactions on Mathematical Software*, 22(4), 469-483. DOI: 10.1145/235815.235821

22. Eldar, Y., Lindenbaum, M., Porat, M., & Zeevi, Y. Y. (1997). The farthest point strategy for progressive image sampling. *IEEE Transactions on Image Processing*, 6(9), 1305-1315. DOI: 10.1109/83.623193

---

*Technical report — nirs4all splitter selection framework*

*© 2025 CIRAD — UMR AGAP Institut*

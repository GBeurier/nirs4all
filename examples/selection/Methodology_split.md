# Methodology: Splitting Strategy Comparison for NIRS Spectral Data

## 1. Introduction

This document describes the scientific methodology used to compare different data splitting strategies for Near-Infrared Spectroscopy (NIRS) predictive modeling. The goal is to identify the optimal splitting approach that maximizes model generalization while respecting the inherent structure of repeated measurements.

## 2. Problem Statement

### 2.1 Context

NIRS spectral data often contains **replicated measurements** (repetitions) of the same physical sample. When building predictive models, a critical constraint is to prevent **data leakage**: all repetitions of a given sample must remain in the same partition (train or test, and within the same fold during cross-validation).

### 2.2 Objectives

1. Compare multiple splitting strategies adapted to spectral data
2. Ensure group integrity (no sample leakage between partitions)
3. Evaluate model performance using simple baseline regressors
4. Provide statistical evidence for strategy selection

## 3. Data Structure

| File | Description |
|------|-------------|
| `X.csv` | Spectral features matrix (wavelengths as columns) |
| `Y.csv` | Target variable (e.g., Digestibility) |
| `M.csv` | Metadata including sample ID and repetition number |

### 3.1 Grouping Constraint

Each unique sample ID has multiple repetitions (Rep 1, 2, 3, 4...). The splitting is performed at the **sample level**, not at the observation level, to prevent information leakage.

## 4. Splitting Strategies

The splitter selection suite includes **20+ splitting strategies** organized into categories:

### 4.1 Baseline Strategies

#### Simple Random Split
**Principle**: Samples are randomly assigned to train/test sets at sample ID level.

- Simple baseline approach
- May not ensure representative distribution of target values
- Group-aware: random assignment at sample ID level

### 4.2 Target-Based Strategies

#### Target Stratified Split
**Principle**: Maintains the target variable distribution across partitions.

1. Bin target values into quantiles (5 or 10 bins)
2. Assign samples to partitions while preserving bin proportions
3. Group-aware stratification based on sample ID

**Advantage**: Ensures similar target distributions in train and test sets.

#### Stratified Group KFold
**Principle**: Uses sklearn's StratifiedGroupKFold for stratified CV with groups.

- Combines stratification by target with group-aware splitting
- Ensures no sample leakage while maintaining target distribution

### 4.3 Spectral-Based Strategies

#### Spectral PCA Split
**Principle**: Leverages PCA-reduced spectral similarity for clustering.

1. Apply PCA to reduce dimensionality (95% variance retained)
2. Cluster samples using KMeans (5 or 10 clusters)
3. Stratify by cluster membership

**Advantage**: Ensures spectral diversity in both partitions.

#### Spectral Distance Split
**Principle**: Uses farthest point sampling in spectral space.

1. Apply PCA for dimensionality reduction
2. Select test samples to maximize spectral diversity

**Advantage**: Test set covers extreme regions of spectral space.

### 4.4 Hybrid Strategies

#### Hybrid Split
**Principle**: Combines spectral clustering with target stratification.

1. Create spectral clusters (5 or 8 clusters)
2. Bin target values (3 or 5 bins)
3. Create combined strata from cluster × bin combinations

**Advantage**: Balances both spectral and target representativeness.

### 4.5 Robustness Strategies

#### Adversarial Split
**Principle**: Creates challenging test sets with outlier samples.

1. Identify spectral outliers using distance metrics
2. Assign a proportion (30% or 50%) of outliers to test set
3. Fill remaining test slots randomly

**Advantage**: Tests model robustness to unusual samples.

### 4.6 Chemometrics Strategies

#### Kennard-Stone Algorithm
**Principle**: Maximizes spectral coverage using sequential selection.

1. Start with the two samples having maximum Euclidean distance
2. Iteratively add the sample with maximum minimum distance to already selected samples
3. Continue until the desired training set size is reached

**Advantage**: Optimal coverage of the spectral feature space, widely used in chemometrics.

**Reference**: Kennard, R.W. and Stone, L.A. (1969). "Computer Aided Design of Experiments." *Technometrics*, 11(1), 137-148.

#### SPXY (nirs4all)
**Principle**: Sample Partitioning based on joint X-Y distances.

1. Compute distances in both X (spectral) and Y (target) space
2. Combine distances to select representative samples
3. Uses nirs4all library implementation

**Advantage**: Considers both spectral and target information for selection.

#### Puchwein Algorithm
**Principle**: Distance-based sample selection with factor k.

1. Compute PCA-reduced distances between samples
2. Select samples based on distance thresholds
3. Configurable factor k controls selection stringency

**Advantage**: Systematic coverage with tunable parameters.

#### Duplex Algorithm
**Principle**: Alternating assignment to train and test sets.

1. Start with farthest pair, assign one to train, one to test
2. Iteratively select next farthest samples
3. Alternate assignments to ensure balanced coverage

**Advantage**: Both sets get representative samples from the full space.

#### Shenk-Westerhaus Algorithm
**Principle**: Distance-based selection from NIR spectroscopy.

1. Compute spectral distances
2. Select samples based on distance criteria
3. Classic algorithm from NIR chemometrics literature

**Advantage**: Well-established method in NIR applications.

#### Honigs Algorithm
**Principle**: Selection based on spectral uniqueness.

1. Evaluate spectral uniqueness of each sample
2. Select samples that add the most information
3. Iterative selection until test size reached

**Advantage**: Prioritizes spectrally unique samples.

### 4.7 Clustering Strategies

#### Hierarchical Clustering Split
**Principle**: Agglomerative clustering-based selection.

1. Build hierarchical cluster tree using complete linkage
2. Cut tree to form clusters
3. Select representatives from each cluster

**Advantage**: Captures hierarchical structure in spectral data.

#### K-Medoids Split
**Principle**: K-Medoids based sample selection.

1. Find k medoids (actual data points as cluster centers)
2. Use medoid positions to guide selection
3. Ensures selected samples are real observations

**Advantage**: More robust to outliers than K-Means.

## 5. Evaluation Protocol

### 5.1 Train/Test Split

- **Ratio**: 80% training, 20% test
- **Constraint**: All repetitions of a sample stay together

### 5.2 Cross-Validation

- **Method**: K-Fold cross-validation on training set
- **Folds**: 3 (configurable)
- **Constraint**: Group-aware folding (GroupKFold equivalent)

### 5.3 Baseline Models

Simple models are used to compare strategies (not to achieve best performance):

| Model | Description |
|-------|-------------|
| Ridge Regression | L2 regularization, handles multicollinearity |
| PLS | Partial Least Squares, standard in chemometrics |
| KNN | Local neighborhood-based prediction |
| XGBoost | Gradient boosting (default) |

### 5.4 Metrics

- **RMSE** (Root Mean Square Error): Primary metric
- **R²** (Coefficient of Determination): Explained variance
- **MAE** (Mean Absolute Error): Robust to outliers

## 6. Analysis Workflow

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
│  (20+ strategies: Random, Stratified, Spectral, Hybrid,     │
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
│           Evaluate with Baseline Models                     │
│  (Cross-validation on train + final test evaluation)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Compare and Visualize                          │
│  (Statistical comparison, ranking, visualizations)          │
└─────────────────────────────────────────────────────────────┘
```

## 7. Interpretation Guidelines

### 7.1 Strategy Selection Criteria

1. **CV vs Test Performance Gap**: Smaller gap indicates better generalization
2. **Test Set Performance**: Ultimate measure of model quality
3. **Stability**: Low variance across folds indicates robust splitting
4. **Target Distribution**: Similar distributions in train/test suggest fair splitting

### 7.2 Expected Outcomes by Category

- **Baseline (Simple Random)**: Baseline reference, may have high variance
- **Target-Based (Stratified)**: Good target distribution, moderate performance
- **Spectral-Based (PCA, Distance)**: Good spectral diversity, covers feature space
- **Hybrid**: Balanced spectral and target representativeness
- **Chemometrics (Kennard-Stone, SPXY, etc.)**: Excellent spectral coverage, often best for NIRS
- **Clustering (Hierarchical, K-Medoids)**: Captures data structure
- **Robustness (Adversarial)**: Tests worst-case scenarios

## 8. Recommendations

1. **Default Choice**: Kennard-Stone for NIRS data (chemometrics standard)
2. **Alternative**: Stratified split if target distribution is skewed
3. **Validation**: Always verify no sample leakage in final splits
4. **Documentation**: Record chosen strategy and random seed for reproducibility

## 9. Implementation Notes

### 9.1 Files

| File | Purpose |
|------|---------|
| `splitter_strategies.py` | Splitting algorithm implementations |
| `splitter_evaluation.py` | Model training and evaluation |
| `splitter_visualization.py` | Results visualization |
| `run_splitter_selection.py` | Main execution script |

### 9.2 Usage

```bash
python run_splitter_selection.py --data_dir path/to/data --output_dir results/
python run_splitter_selection.py --data_dir ./data --model xgboost --test_size 0.2
python run_splitter_selection.py --data_dir ./data --model ridge --n_folds 3
```

### 9.3 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir`, `-d` | Directory containing X.csv, Y.csv, M.csv | Required |
| `--output_dir`, `-o` | Output directory | data_dir/splitter_selection |
| `--test_size`, `-t` | Fraction for test set | 0.2 |
| `--n_folds`, `-f` | Number of CV folds | 3 |
| `--model`, `-m` | Baseline model (ridge, pls, knn, xgboost) | xgboost |
| `--random_state`, `-r` | Random seed | 42 |
| `--quiet`, `-q` | Suppress verbose output | False |

### 9.4 Reproducibility

- Set random seed for deterministic results
- Save split indices for future reference
- Document all hyperparameters

## 10. References

1. Kennard, R.W. and Stone, L.A. (1969). "Computer Aided Design of Experiments." *Technometrics*, 11(1), 137-148.
2. Galvão, R.K.H., et al. (2005). "A method for calibration and validation subset partitioning." *Talanta*, 67(4), 736-740.
3. Snee, R.D. (1977). "Validation of Regression Models: Methods and Examples." *Technometrics*, 19(4), 415-428.
4. Puchwein, G. (1988). "Selection of Calibration Samples for Near-Infrared Spectrometry by Factor Analysis of Spectra." *Analytical Chemistry*, 60, 569-573.
5. Shenk, J.S. and Westerhaus, M.O. (1991). "Population Definition, Sample Selection, and Calibration Procedures for Near Infrared Reflectance Spectroscopy." *Crop Science*, 31(2), 469-474.
6. Honigs, D.E., et al. (1985). "Unique-Sample Selection via Near-Infrared Spectral Subtraction." *Analytical Chemistry*, 57, 2299-2303.

---

*Generated for the nirs4all splitter selection suite*

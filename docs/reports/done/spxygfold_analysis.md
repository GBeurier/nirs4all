# SPXY Splitter Analysis and SPXYGFold Extension Proposal

## Executive Summary

This report analyzes the current `SPXYSplitter` implementation in nirs4all, identifies its limitations regarding classification tasks, group-aware splitting, and single-fold constraints, and proposes an extended algorithm (`SPXYGFold`) that addresses these issues while maintaining backward compatibility.

---

## 1. Current SPXYSplitter Functioning

### 1.1 Algorithm Overview

The SPXYSplitter implements the **SPXY (Sample Partitioning based on joint X-Y distances)** sampling method, which extends the classic Kennard-Stone algorithm.

#### Historical Background

| Year | Authors | Contribution |
|------|---------|--------------|
| 1969 | Kennard & Stone | Original max-min distance algorithm for uniform sample selection |
| 2005 | Galvão et al. | Extended to SPXY by including Y-space distances |

#### Mathematical Foundation

The SPXY algorithm combines feature-space (X) and target-space (Y) distances:

$$d_{XY}(i,j) = \frac{d_X(i,j)}{\max(d_X)} + \frac{d_Y(i,j)}{\max(d_Y)}$$

Where:
- $d_X(i,j) = \|x_i - x_j\|$ (Euclidean distance in feature space)
- $d_Y(i,j) = \|y_i - y_j\|$ (Euclidean distance in target space)

### 1.2 Current Implementation (nirs4all)

```python
class SPXYSplitter(CustomSplitter):
    def split(self, X, y=None, groups=None):
        # 1. Compute normalized X distance matrix
        distance_features = cdist(X_transformed, X_transformed, metric=self.metric)
        distance_features /= distance_features.max()

        # 2. Compute normalized Y distance matrix
        distance_labels = cdist(y_transformed, y_transformed, metric=self.metric)
        distance_labels /= distance_labels.max()

        # 3. Combine distances
        distance = distance_features + distance_labels

        # 4. Apply max-min selection (Kennard-Stone)
        train_indices, test_indices = self._max_min_distance_split(distance, n_train)
```

### 1.3 Selection Algorithm (Max-Min Distance)

The Kennard-Stone selection proceeds as follows:

1. **Initialization**: Select the two samples with maximum mutual distance
2. **Iteration**: For each remaining sample:
   - Compute minimum distance to all already-selected samples
   - Select the sample with the maximum minimum distance
3. **Termination**: Stop when `n_train` samples are selected

**Theoretical Justification**: This approach minimizes the maximum interpolation error by ensuring uniform coverage of the sample space.

---

## 2. Classification Behavior Analysis

### 2.1 The Problem

**Critical Finding**: SPXY is fundamentally designed for **regression tasks** and produces mathematically incorrect results when applied to classification.

#### Why Classification Fails

When y contains class labels (e.g., [0, 1, 2]):

```python
# Current implementation computes:
distance_labels = cdist(y_transformed, y_transformed, metric="euclidean")
# Results in: d(class_0, class_2) = 2, d(class_0, class_1) = 1
```

This creates **artificial ordinal relationships** between nominal categories:
- Class 0 vs Class 2 is treated as "farther" than Class 0 vs Class 1
- This is **semantically meaningless** for unordered categories
- The normalization preserves these incorrect relationships

### 2.2 Mathematical Analysis

For classification with $K$ classes, the Euclidean distance on encoded labels:

$$d_Y(i,j) = |c_i - c_j|$$

Where $c_i, c_j \in \{0, 1, ..., K-1\}$ are class codes.

**Problems**:
1. **Ordinal assumption**: Treats classes as if they have inherent order
2. **Non-uniform spacing**: Class distances depend on arbitrary encoding
3. **Bias**: Extreme class codes (0 and K-1) are overrepresented in training set

### 2.3 Proposed Solutions for Classification

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **X-only (Kennard-Stone)** | Ignore Y for splitting | Simple, theoretically sound | Ignores class distribution |
| **Hamming distance** | Binary distance (same=0, different=1) | Treats all class differences equally | Loses magnitude information |
| **Stratified + X-distance** | Stratify by class, then SPXY within class | Preserves class balance | Computationally more expensive |
| **One-hot encoding** | Encode classes as vectors, use Euclidean | Uniform class distances | High dimensionality for many classes |

**Recommended Approach**: Use **stratified Kennard-Stone** or **Hamming distance** for Y when task is classification.

---

## 3. Group-Aware Splitting Integration

### 3.1 Use Case

Many spectroscopy datasets contain samples that should be treated as groups:
- **Repeated measurements** of the same physical sample
- **Augmented samples** from the same original
- **Temporal series** from the same experimental unit
- **Biological replicates** from the same subject

Splitting these independently causes **data leakage** between train/test sets.

### 3.2 Proposed Group Aggregation Methods

#### 3.2.1 Group Representation

For a group $G$ containing samples $\{x_1, ..., x_m\}$:

| Method | Formula | Best For |
|--------|---------|----------|
| **Mean (Centroid)** | $\bar{x}_G = \frac{1}{m}\sum_{i=1}^m x_i$ | Normally distributed data |
| **Median** | $\tilde{x}_G = \text{median}(x_1, ..., x_m)$ | Data with outliers |
| **Geometric Median** | $\arg\min_z \sum_{i=1}^m \|x_i - z\|$ | Robust central tendency |
| **First Principal Component** | $\text{PC}_1(\{x_1, ..., x_m\})$ | Capturing main variation |

#### 3.2.2 Inter-Group Distance Metrics

| Method | Definition | Characteristics |
|--------|------------|-----------------|
| **Centroid distance** | $d(\bar{x}_A, \bar{x}_B)$ | Fast, smooth |
| **Single linkage** | $\min_{a \in A, b \in B} d(a,b)$ | Sensitive to outliers |
| **Complete linkage** | $\max_{a \in A, b \in B} d(a,b)$ | Conservative |
| **Average linkage** | $\frac{1}{|A||B|}\sum_{a,b} d(a,b)$ | Balanced |
| **Ward's method** | Increase in total within-group variance | Variance-based |

**Recommended**: **Mean aggregation** with **centroid distance** for computational efficiency.

### 3.3 Group-Aware SPXY Algorithm

```
Algorithm: Group-Aware SPXY Selection

Input: X (n×p), y (n×1), groups (n×1), train_ratio
Output: train_indices, test_indices

1. AGGREGATE
   For each unique group g:
     X_rep[g] = mean(X[groups == g])
     y_rep[g] = mean(y[groups == g])  # or mode for classification
     group_size[g] = count(groups == g)

2. COMPUTE DISTANCES
   D_X = pairwise_distance(X_rep) / max(D_X)
   D_Y = pairwise_distance(y_rep) / max(D_Y)  # or Hamming for classification
   D = D_X + D_Y

3. SELECT GROUPS (max-min on group representatives)
   n_train_groups = ceil(train_ratio * n_groups)
   selected_groups = kennard_stone_select(D, n_train_groups)

4. MAP TO SAMPLES
   train_indices = samples where groups ∈ selected_groups
   test_indices = samples where groups ∉ selected_groups

Return train_indices, test_indices
```

### 3.4 Complexity Analysis

| Phase | Without Groups | With Groups (m groups) |
|-------|---------------|------------------------|
| Distance matrix | $O(n^2 \cdot p)$ | $O(m^2 \cdot p)$ |
| Aggregation | N/A | $O(n \cdot p)$ |
| Selection | $O(n^2)$ | $O(m^2)$ |
| **Total** | $O(n^2 \cdot p)$ | $O(m^2 \cdot p + n \cdot p)$ |

**Benefit**: When $m \ll n$, group-aware approach is significantly faster.

---

## 4. K-Fold Cross-Validation Extension

### 4.1 Current Limitation

`SPXYSplitter` is hardcoded to `n_splits=1`, meaning it can only produce a single train/test split, not K-fold cross-validation.

### 4.2 K-Fold Extension Strategies

#### Strategy A: Round-Robin Assignment

1. Rank all samples/groups by SPXY selection order
2. Assign to folds in round-robin: sample 1→fold 1, sample 2→fold 2, ..., sample k→fold k, sample k+1→fold 1, ...

**Pros**: Simple, deterministic
**Cons**: Adjacent samples in ranking may be similar

#### Strategy B: Alternating Max-Min (Recommended)

1. Initialize k fold representatives (k farthest points from centroid)
2. Iteratively assign remaining samples:
   - For sample s, compute distance to nearest member of each fold
   - Assign to fold with maximum distance (ensures diversity)
   - Cycle through folds to balance sizes

**Mathematical Formulation**:

$$\text{fold}(s) = \arg\max_{f \in \{1,...,k\}} \min_{t \in \text{fold}_f} d(s, t)$$

Subject to: $|\text{fold}_f| \leq \lceil n/k \rceil$ for all $f$

#### Strategy C: Hierarchical Clustering

1. Apply hierarchical clustering on SPXY distance matrix
2. Cut dendrogram at level producing k clusters
3. Each cluster becomes a fold

**Pros**: Preserves neighborhood structure
**Cons**: May produce unbalanced folds

### 4.3 Reference Implementation

The `kennard_stone` Python package (PyPI) provides a working K-Fold implementation:

```python
# From kennard_stone library
class _KennardStone:
    def __init__(self, n_groups: int = 1, ...):
        self.n_groups = n_groups  # Number of folds

    def get_indexes(self, X) -> list[array[int]]:
        # Simultaneously assigns samples to n_groups folds
        # Uses alternating max-min selection
```

Key insight: Their `n_groups` parameter represents the number of folds, and samples are assigned to all folds simultaneously rather than sequentially.

---

## 5. Proposed Implementation: SPXYGFold

### 5.1 Class Design

```python
class SPXYGFold(CustomSplitter):
    """
    SPXY-based K-Fold splitter with group awareness.

    Combines:
    - SPXY (joint X-Y distance) or Kennard-Stone (X-only) selection
    - Group constraints (samples in same group stay together)
    - K-fold cross-validation

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Use 1 for single train/test split.

    test_size : float, default=None
        Proportion of samples for test set. Only used when n_splits=1.
        If None with n_splits=1, defaults to 0.25.

    metric : str, default="euclidean"
        Distance metric for X-space.

    y_metric : str or None, default="euclidean"
        Distance metric for Y-space.
        - "euclidean": For regression (continuous y)
        - "hamming": For classification (categorical y)
        - None: Ignore Y (pure Kennard-Stone)

    aggregation : str, default="mean"
        Method for group aggregation: "mean", "median", "geometric_median"

    random_state : int or None, default=None
        Only used for tie-breaking when multiple samples have equal distances.
    """
```

### 5.2 API Compatibility

```python
# Backward compatible - behaves like current SPXYSplitter
splitter = SPXYGFold(n_splits=1, test_size=0.2)

# K-Fold with SPXY
splitter = SPXYGFold(n_splits=5)

# Group-aware splitting
splitter = SPXYGFold(n_splits=5, groups="sample_id")  # or column name

# Classification mode
splitter = SPXYGFold(n_splits=5, y_metric="hamming")

# Pure Kennard-Stone (X-only)
splitter = SPXYGFold(n_splits=5, y_metric=None)
```

### 5.3 Core Algorithm

```python
def split(self, X, y=None, groups=None):
    # Phase 1: Group handling
    if groups is not None:
        X_rep, y_rep, group_indices = self._aggregate_groups(X, y, groups)
    else:
        X_rep, y_rep = X, y
        group_indices = [[i] for i in range(len(X))]

    # Phase 2: Distance computation
    D = self._compute_combined_distance(X_rep, y_rep)

    # Phase 3: Fold assignment
    fold_assignments = self._assign_to_folds(D, self.n_splits)

    # Phase 4: Generate splits
    for fold_idx in range(self.n_splits):
        test_groups = np.where(fold_assignments == fold_idx)[0]
        test_indices = np.concatenate([group_indices[g] for g in test_groups])
        train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
        yield train_indices, test_indices
```

---

## 6. Mathematical and Implementation Impacts

### 6.1 Distance Matrix Modifications

#### For Classification (Hamming-based Y-distance)

$$d_Y^{(\text{class})}(i,j) = \mathbb{1}[y_i \neq y_j]$$

Normalized: No normalization needed (already in [0,1])

Combined distance:

$$d_{XY}^{(\text{class})}(i,j) = \frac{d_X(i,j)}{\max(d_X)} + \alpha \cdot d_Y^{(\text{class})}(i,j)$$

Where $\alpha \in [0,1]$ controls the weight of class differences.

### 6.2 Group Aggregation for Y

| Task | Aggregation Method |
|------|-------------------|
| Regression | Mean: $\bar{y}_G = \frac{1}{m}\sum_{i \in G} y_i$ |
| Classification | Mode: $\hat{y}_G = \arg\max_c \sum_{i \in G} \mathbb{1}[y_i = c]$ |
| Multi-output | Per-output mean/mode |

### 6.3 Fold Balance Considerations

When groups have different sizes, folds may become unbalanced. Mitigation strategies:

1. **Size-weighted distance**: Weight group representative by $\sqrt{n_g}$
2. **Iterative balancing**: Assign to fold with minimum current size among valid candidates
3. **Post-hoc adjustment**: Move small groups between folds to balance

### 6.4 Computational Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Group aggregation | $O(n \cdot p)$ | $O(m \cdot p)$ |
| X distance matrix | $O(m^2 \cdot p)$ | $O(m^2)$ |
| Y distance matrix | $O(m^2)$ | $O(m^2)$ |
| K-fold assignment | $O(m^2 \cdot k)$ | $O(m \cdot k)$ |
| Index mapping | $O(n)$ | $O(n)$ |
| **Total** | $O(m^2 \cdot p + n \cdot p)$ | $O(m^2 + n)$ |

Where:
- $n$ = number of samples
- $m$ = number of groups (or $n$ if no groups)
- $p$ = number of features
- $k$ = number of folds

---

## 7. Implementation Recommendations

### 7.1 Priority Order

1. **High**: Add classification support (y_metric parameter)
2. **High**: Add K-fold support (n_splits parameter)
3. **Medium**: Add group support (groups parameter)
4. **Low**: Additional aggregation methods and advanced tie-breaking

### 7.2 Testing Strategy

1. **Unit tests**: Verify algorithm correctness on synthetic data
2. **Comparison tests**: Compare single-split mode with current SPXYSplitter
3. **Edge cases**: Empty groups, single-sample groups, highly imbalanced folds
4. **Performance tests**: Benchmark against large datasets

### 7.3 Documentation

- Update docstrings with mathematical formulas
- Add examples for each use case (regression, classification, grouped)
- Include performance comparison with sklearn splitters

---

## 8. References

1. Kennard, R.W. & Stone, L.A. (1969). "Computer Aided Design of Experiments." *Technometrics*, 11(1), 137-148. DOI: 10.1080/00401706.1969.10490666

2. Galvão, R.K.H., Araujo, M.C.U., José, G.E., Pontes, M.J.C., Silva, E.C., & Saldanha, T.C.B. (2005). "A method for calibration and validation subset partitioning." *Talanta*, 67(4), 736-740.

3. Yu, Y. (2021). "kennard-stone: A method for selecting samples by spreading the training data evenly." PyPI. https://pypi.org/project/kennard-stone/

4. Datachemeng. "Training/Test Data Division with Kennard-Stone Algorithm." https://datachemeng.com/trainingtestdivision/

---

## Appendix A: Pseudocode for K-Fold Assignment

```
Algorithm: SPXY_K_Fold_Assignment

Input: D (m×m distance matrix), k (number of folds)
Output: fold_assignment (m-length array)

1. Initialize fold sets: F[1..k] = empty
2. Compute centroid distances: d_centroid[i] = D[i, :].mean()
3. Find k initial samples: init = argsort(d_centroid)[-k:]
4. For j in 1..k:
     F[j].add(init[k-j])
     fold_assignment[init[k-j]] = j

5. remaining = all indices not in init
6. While remaining not empty:
     For j in 1..k:
       If remaining is empty: break

       # Compute min distance to fold j
       min_dist = [min(D[r, f] for f in F[j]) for r in remaining]

       # Select sample with max min-distance
       best = remaining[argmax(min_dist)]
       F[j].add(best)
       fold_assignment[best] = j
       remaining.remove(best)

7. Return fold_assignment
```

## Appendix B: Comparison with Existing Splitters

| Feature | SPXYSplitter | KennardStoneSplitter | sklearn KFold | sklearn GroupKFold | **SPXYGFold** |
|---------|--------------|---------------------|---------------|-------------------|---------------|
| X-space distance | ✓ | ✓ | ✗ | ✗ | ✓ |
| Y-space distance | ✓ | ✗ | ✗ | ✗ | ✓ (optional) |
| Classification support | ✗ | N/A | ✓ | ✓ | ✓ |
| K-fold | ✗ | ✗ | ✓ | ✓ | ✓ |
| Group-aware | ✗ | ✗ | ✗ | ✓ | ✓ |
| Uniform coverage | ✓ | ✓ | ✗ | ✗ | ✓ |

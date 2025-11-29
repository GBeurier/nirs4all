````markdown
# SPXY Splitter: Overview

## 1. Introduction

Random splits often produce over-optimistic results because NIRS spectra are highly autocorrelated, instrument-dependent, and sensitive to small baseline shifts. Kennard–Stone (KS) improves representativity by selecting samples evenly in **X-space**, but ignores **Y**, which is problematic for regression. **SPXY** (Sample set Partitioning based on joint X–Y distances), introduced by **Galvão et al. (2005)**, extends KS by incorporating response variability.
---

## 2. Mathematical Formulation

### 2.1 Normalization

Let
- \( X \in \mathbb{R}^{n \times p} \) be NIRS spectra
- \( y \in \mathbb{R}^{n} \) be the target property (e.g., protein content)

Both are independently standardized:

\[
X'_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j},
\qquad
y'_i = \frac{y_i - \mu_y}{\sigma_y}.
\]

This step is essential because X and Y must contribute on comparable scales.

---

### 2.2 Distance Computation

For samples \( i \) and \( j \):

1. **X-distance (spectral Euclidean distance)**
\[
d_X(i,j) = \| X'_i - X'_j \|_2
\]

2. **Y-distance (absolute difference)**
\[
d_Y(i,j) = |y'_i - y'_j|
\]

3. **Combined SPXY distance**
\[
d_{SPXY}(i,j) = d_X(i,j) + d_Y(i,j)
\]

Thus the final pairwise distance matrix is

\[
D = D_X + D_Y.
\]

---

### 2.3 Sample Selection (Kennard–Stone Style)

1. Select the pair \((i, j)\) that maximizes \( d_{SPXY}(i,j) \).
2. Iteratively select the next sample:

\[
k^\* = \arg\max_k \left( \min_{i \in S} d_{SPXY}(k, i) \right),
\]

where \( S \) is the set of already selected samples.
3. The first \(n_{\text{train}}\) samples form the training set; the rest form the test set.

---

## 3. Why SPXY Works Well for NIRS

### 3.1 Advantages

- Accounts for both **spectral diversity** (X) and **property diversity** (Y).
- Reduces optimistic bias compared to random splits.
- Ensures the training set covers the full dynamic range of Y.
- Particularly effective for:
  - small datasets,
  - heterogeneous Y distributions,
  - spectra with strong local correlations (typical of NIRS),
  - calibration transfer and multi-instrument scenarios.

### 3.2 Limitations

- Normalization is mandatory.
- Very sensitive to extreme outliers in Y.
- Not suitable for classification.
- O(n² p) complexity (acceptable for typical NIRS sizes).

---

## 4. Implementation in NIRS4ALL

The NIRS4ALL implementation follows the mathematical formulation exactly:

1. Standardize X and Y.
2. Compute distance matrices \(D_X\), \(D_Y\).
3. Build combined matrix \(D_{SPXY}\).
4. Apply the KS selection loop.
5. Return train/test indices consistent with:
   - sklearn splitters,
   - DataArray index propagation,
   - augmentation modules,
   - caching and hashing rules.

### Complexity
\[
O(n^2 p)
\]
Compatible with datasets up to several thousand samples and ~2000 wavelengths.

---

## 5. Python Example

```python
from nirs4all.splitters import SPXYSplitter

splitter = SPXYSplitter(test_size=0.2, random_state=42)
train_idx, test_idx = splitter.split(X, y)
````

---

## 6. References

* **Galvão, R. K. H., et al. (2005)**.
  *A method for calibration and validation subset partitioning.*
  Talanta, 67(4), 736–740.

* **Kennard, R. W., & Stone, L. A. (1969)**.
  *Computer aided design of experiments.*
  Technometrics, 11, 137–148.

* **Rinnan, Å., & Rinnan, J. (2007)**.
  *Local regression kernel methods in NIRS.*
  Journal of Chemometrics.

---

# Cross-Dataset Metrics: Complete Explanation

## Overview
This document explains the different types of metrics used to evaluate preprocessing methods for cross-dataset compatibility (transfer learning between different NIRS instruments).

---

## 1. Inter-Dataset Distance Metrics

### What they measure
**Pairwise distances between different datasets in PCA space**

### Types

#### A. Centroid Distance
- **Computation**: Euclidean L2-norm between dataset centers
- **Formula**: `||mean(PCA_A) - mean(PCA_B)||`
- **Interpretation**:
  - How far apart are the "average samples" of dataset A and dataset B?
  - Lower = datasets are centered closer together
  - Better for transfer learning when reduced by preprocessing

#### B. Spread Distance
- **Computation**: Frobenius norm (covariance difference) + sample-wise Euclidean distances
- **Formula**: `||Cov_A - Cov_B||_F + mean(min_distances(samples_A, samples_B))`
- **Interpretation**:
  - How different are the distribution shapes and sample proximities?
  - Lower = distributions overlap more
  - Better for transfer learning when reduced by preprocessing

### Visualization
- **Distance matrices**: Show raw values (scientific notation for tiny numbers)
- **Distance reduction ranking**: Show % improvement and absolute values (with log scale)

### Why these exist
To directly answer: "Does preprocessing make datasets from different machines spatially closer in PCA space?"

---

## 2. Quality Metric Convergence

### What they measure
**Variance of within-dataset quality metrics across all datasets**

### Concept
Instead of measuring distances *between* datasets, we measure if preprocessing makes all datasets achieve **similar quality scores** for their internal structure preservation.

### Metrics tracked
1. **EVR (Explained Variance Ratio)**: Do all datasets retain similar variance after preprocessing?
2. **CKA (Centered Kernel Alignment)**: Do all datasets preserve PCA structure similarly?
3. **RV Coefficient**: Similar to CKA, geometric structure preservation consistency
4. **Procrustes** (inverted): Do all datasets have similar alignment quality? (lower Procrustes = better, so we invert it)
5. **Trustworthiness**: Do all datasets preserve k-NN neighborhoods similarly?
6. **Grassmann** (inverted): Do all datasets have similar subspace angles? (lower = better, so we invert it)

### Formula
```python
# For each metric (with inversion for Grassmann/Procrustes):
raw_variance = variance([dataset1_metric, dataset2_metric, dataset3_metric, ...])
pp_variance = variance([dataset1_pp_metric, dataset2_pp_metric, dataset3_pp_metric, ...])
convergence = (raw_variance - pp_variance) / raw_variance
```

### Interpretation
- **Positive convergence**: Preprocessing reduces variance → datasets more homogeneous in quality
- **Negative convergence**: Preprocessing increases variance → datasets more heterogeneous in quality
- **High convergence = better** for robust cross-dataset models

### Why inversion matters
- **Grassmann and Procrustes are distance metrics**: Lower values = better quality
- **Without inversion**: We'd measure variance of "badness" (distance values)
- **With inversion**: We measure variance of "goodness" (quality scores)
- **Example**:
  - Raw: Dataset A has Procrustes=0.5, Dataset B has Procrustes=0.6 (variance = 0.005)
  - After inversion: Dataset A = -0.5, Dataset B = -0.6 (variance = 0.005, same)
  - But now "more negative = better", consistent with other metrics where "higher = better"

### Visualization
- **6 subplots**: One per quality metric showing convergence for each preprocessing
- **Bar charts**: Green = positive convergence (good), Red = negative convergence (bad)

---

## 3. Why NO "Distance Matrix for Preservation Metrics"?

### The key difference

#### Inter-dataset distances (what we HAVE):
```
Question: "How far is dataset A from dataset B?"
Answer: Centroid distance = 5.2e-10
         Spread distance = 0.008
Interpretation: Direct spatial relationship between datasets
```

#### Preservation metrics (what you CANNOT do):
```
Question: "What is the CKA distance between dataset A and dataset B?"
Answer: ❌ THIS DOESN'T MAKE SENSE
Why: CKA measures "How well does preprocessing preserve dataset A's internal structure?"
     It's a quality score for ONE dataset, not a distance between TWO datasets
```

### Analogy
Think of datasets as students taking a test:

**Inter-dataset distances** = Physical distance between students in the room
- "Student A sits 3 meters from Student B"
- Makes sense to create a distance matrix

**Preservation metrics** = Test scores for each student
- "Student A scored 85%, Student B scored 90%"
- You can compute variance (are scores similar?)
- But you CANNOT compute "score distance between students" - that's just the difference in scores, not a meaningful spatial relationship

### What we use instead
**Quality Convergence**: Measures if all students get similar scores
- "All students scored between 85-90%" = low variance = good convergence
- "Students scored 30%, 60%, 95%" = high variance = bad convergence

---

## 4. Complete Workflow

### For finding best preprocessing for transfer learning:

1. **Spatial proximity** (Inter-dataset distances):
   - Check centroid distance reduction → find methods that bring dataset centers closer
   - Check spread distance reduction → find methods that make distributions overlap more
   - Best: >95% reduction, absolute distance <1e-9

2. **Quality homogeneity** (Convergence):
   - Check EVR convergence → do all datasets retain similar variance?
   - Check CKA/RV convergence → do all datasets preserve structure similarly?
   - Check Procrustes/Grassmann convergence → do all datasets align similarly?
   - Best: Positive convergence (>0.5) across all metrics

3. **Trade-offs**:
   - Some preprocessing brings datasets spatially closer BUT makes quality diverge
   - Some preprocessing makes quality converge BUT increases spatial distance
   - Ideal: High distance reduction + high quality convergence

### Example interpretation
```
Preprocessing: Detr→MSC→Gauss
- Centroid distance reduction: 99.85% ✓ (excellent spatial proximity)
- Spread distance reduction: 99.99% ✓ (excellent distribution overlap)
- Average quality convergence: -11.02 ✗ (quality becomes more divergent)

→ Good for spatial transfer learning
→ But datasets will behave differently in terms of structure preservation
→ May need dataset-specific tuning after transfer
```

---

## 5. Summary Table

| Metric Type | What it measures | Pairwise? | Higher = Better? | Inversion needed? |
|-------------|------------------|-----------|------------------|-------------------|
| **Centroid Distance** | Dataset center separation | Yes (A↔B) | Lower | No |
| **Spread Distance** | Distribution overlap | Yes (A↔B) | Lower | No |
| **EVR Convergence** | Variance consistency | No (variance across all) | Higher | No (already positive metric) |
| **CKA Convergence** | Structure preservation consistency | No (variance across all) | Higher | No (already positive metric) |
| **RV Convergence** | Geometric consistency | No (variance across all) | Higher | No (already positive metric) |
| **Procrustes Convergence** | Alignment quality consistency | No (variance across all) | Higher | **Yes** (distance→quality) |
| **Trustworthiness Convergence** | k-NN preservation consistency | No (variance across all) | Higher | No (already positive metric) |
| **Grassmann Convergence** | Subspace angle consistency | No (variance across all) | Higher | **Yes** (distance→quality) |

---

## 6. Key Takeaways

✅ **Inter-dataset distances**: Tell you if preprocessing brings datasets spatially closer
✅ **Quality convergence**: Tells you if preprocessing makes datasets behave similarly
✅ **Both are needed**: Spatial proximity without quality homogeneity means unreliable transfer
✅ **Inversion is critical**: Grassmann and Procrustes must be inverted to measure "goodness variance" not "badness variance"
✅ **No distance matrices for quality**: Preservation metrics are within-dataset scores, not between-dataset distances

---

## References
- Centroid distance: Standard Euclidean norm
- Spread distance: Frobenius norm + Earth Mover's approximation
- CKA: Kornblith et al. (2019) - Similarity of Neural Network Representations
- RV Coefficient: Escoufier (1973) - Operator related to a data matrix
- Procrustes: Gower & Dijksterhuis (2004) - Procrustes Problems
- Grassmann distance: Subspace angles via SVD

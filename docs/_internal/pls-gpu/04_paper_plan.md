# pls-gpu: Publication Plan for Ranked Journal

**Version:** 1.0.0-draft
**Date:** 2024-12-31
**Status:** Specification Draft

## 1. Target Journals

### 1.1 Primary Targets (Q1 Journals)

| Journal | Impact Factor | Scope | Typical Length | Review Time |
|---------|--------------|-------|----------------|-------------|
| **Chemometrics and Intelligent Laboratory Systems** | 4.0 | Software, chemometrics | 8-15 pages | 2-4 months |
| **Journal of Cheminformatics** | 8.5 | Software, methods | 10-20 pages | 2-3 months |
| **Computer Methods and Programs in Biomedicine** | 6.1 | Software, biomedical | 10-15 pages | 2-4 months |

### 1.2 Alternative Targets

| Journal | Impact Factor | Scope |
|---------|--------------|-------|
| Analytica Chimica Acta | 6.2 | Analytical chemistry |
| Journal of Chemometrics | 2.4 | Chemometrics focus |
| Computational Statistics & Data Analysis | 1.8 | Statistics software |
| SoftwareX | 3.4 | Short software papers |
| JOSS (Journal of Open Source Software) | 0.8 | Open source software |

### 1.3 Recommended Strategy

**Primary**: Chemometrics and Intelligent Laboratory Systems
- Strong software paper tradition
- Established PLS readership
- Good balance of impact and acceptance rate
- Previous successful PLS library publications (pls, mixOmics, SIMCA)

**Backup**: Journal of Cheminformatics (if more technical depth needed) or JOSS (for rapid publication)

## 2. Paper Structure

### 2.1 Proposed Title Options

1. "pls-gpu: A GPU-Accelerated Partial Least Squares Library with Multi-Backend Support"
2. "Unified GPU-Accelerated Partial Least Squares Regression in Python"
3. "pls-gpu: Comprehensive PLS Methods with NumPy, JAX, PyTorch and TensorFlow Backends"

### 2.2 Abstract (250 words target)

**Structure**:
- Background: PLS importance in chemometrics, limitations of current implementations
- Problem: No unified GPU-accelerated PLS library exists
- Solution: pls-gpu with 4 backends, 15+ methods
- Results: Numerical equivalence + speedup factors
- Availability: Open source, PyPI, documentation

### 2.3 Detailed Outline

```
1. Introduction (1.5 pages)
   1.1 PLS in chemometrics and spectroscopy
   1.2 Current software landscape
   1.3 GPU computing in scientific Python
   1.4 Motivation and objectives

2. Mathematical Background (2 pages)
   2.1 Standard PLS formulation
   2.2 Core algorithms
       2.2.1 NIPALS
       2.2.2 SIMPLS
       2.2.3 IKPLS
   2.3 Extended methods overview
       2.3.1 OPLS
       2.3.2 Kernel methods
       2.3.3 Variable selection

3. Software Architecture (2 pages)
   3.1 Design principles
       3.1.1 Separation of algorithm and backend
       3.1.2 sklearn API compatibility
       3.1.3 Lazy backend loading
   3.2 Package structure
   3.3 Backend abstraction layer
   3.4 GPU optimization strategies

4. Implemented Methods (1.5 pages)
   4.1 Table of all methods
   4.2 Algorithm selection guidance
   4.3 Method-specific considerations

5. Validation (2 pages)
   5.1 Numerical equivalence testing
       5.1.1 vs sklearn.cross_decomposition
       5.1.2 vs R pls package
       5.1.3 vs MATLAB plsregress
   5.2 Output comparison methodology
   5.3 Equivalence results
   5.4 Edge case handling

6. Performance Benchmarks (3 pages)
   6.1 Benchmark methodology
   6.2 Backend comparison (CPU vs GPU)
       6.2.1 JAX performance
       6.2.2 PyTorch performance
       6.2.3 TensorFlow performance
   6.3 Scaling analysis
       6.3.1 Sample count scaling
       6.3.2 Feature count scaling
       6.3.3 Component count scaling
   6.4 Memory efficiency
   6.5 Real-time applications potential

7. Case Studies on Real Datasets (3 pages)
   7.1 Dataset descriptions (Table)
   7.2 Spectroscopy applications
       7.2.1 NIR grain protein prediction
       7.2.2 Wine authentication (classification - PLS-DA/OPLS-DA)
       7.2.3 Pharmaceutical API quantification
   7.3 Other domains
       7.3.1 Metabolomics
       7.3.2 Process monitoring (DiPLS or Recursive PLS demo)
   7.4 Comparative analysis with other software
       - Explicit comparison: "pls-gpu obtained equivalent accuracy to MATLAB PLS_Toolbox
         on the gasoline dataset, but with 8x speedup using GPU and open-source availability"
       - Ease-of-use comparison vs R/MATLAB
       - Unified API advantage highlighted

8. Discussion (1 page)
   8.1 Summary of contributions
   8.2 When do advanced variants matter?
       - OPLS: more interpretable models by removing orthogonal variation
       - Sparse PLS: wavelength selection with minimal accuracy loss
       - Kernel PLS: nonlinear relationships
       - When to use each variant (guidance table)
   8.3 Limitations
       - Methods not included (HOPLS, deep learning alternatives)
       - GPU memory constraints for very large data
       - TensorFlow backend lower priority/less tested
   8.4 Future directions

9. Conclusion (0.5 pages)

10. Data and Software Availability

References (40-60 citations)

Supplementary Material
- S1: Complete method comparison tables
- S2: Additional benchmark results
- S3: Installation and usage guide
- S4: Benchmark reproduction code
```

## 3. Scientific Content Details

### 3.1 Introduction Section

**Key Points to Cover**:

1. **PLS Importance**:
   - Dimensionality reduction with prediction focus
   - Wide use in chemometrics, spectroscopy, metabolomics
   - Handles multicollinearity and n < p problems
   - Citations: Wold (1966), Geladi (1986), Mehmood (2012)

2. **Current Software Landscape**:

| Software | Language | GPU | Methods | Limitations |
|----------|----------|-----|---------|-------------|
| sklearn.cross_decomposition | Python | ❌ | 3 | Limited methods |
| R pls | R | ❌ | 4 | No GPU |
| MATLAB PLS_Toolbox | MATLAB | ❌ | Many | Commercial, closed |
| mixOmics | R | ❌ | 5+ | R only |
| pyopls | Python | ❌ | 2 | Limited scope |
| ikpls | Python | ✓ (JAX) | 2 | IKPLS only |

3. **Gap Identification**:
   - No unified GPU-accelerated library
   - Fragmented ecosystem
   - Missing sklearn compatibility in GPU solutions
   - Need for method comparison on same platform

### 3.2 Mathematical Background

#### 3.2.1 Standard PLS Formulation

**Notation**:
- $X \in \mathbb{R}^{n \times p}$: predictor matrix (n samples, p features)
- $Y \in \mathbb{R}^{n \times q}$: response matrix (n samples, q targets)
- $T \in \mathbb{R}^{n \times k}$: X scores
- $U \in \mathbb{R}^{n \times k}$: Y scores
- $P \in \mathbb{R}^{p \times k}$: X loadings
- $Q \in \mathbb{R}^{q \times k}$: Y loadings
- $W \in \mathbb{R}^{p \times k}$: X weights
- $B \in \mathbb{R}^{p \times q}$: regression coefficients

**PLS Decomposition**:
$$X = TP^T + E$$
$$Y = UQ^T + F$$

Where maximizing $\text{cov}(t_i, u_i)$ for each component.

#### 3.2.2 NIPALS Algorithm

```
For each component k = 1, ..., K:
1. Initialize: u = y₁ (first column of Y)
2. Repeat until convergence:
   a. w = X'u / (u'u)
   b. w = w / ||w||
   c. t = Xw
   d. q = Y't / (t't)
   e. u = Yq
3. p = X't / (t't)
4. Deflate: X ← X - tp', Y ← Y - tq'
```

**Complexity**: $O(np + nq)$ per iteration, $O(K \cdot I \cdot (np + nq))$ total.

#### 3.2.3 SIMPLS Algorithm

Direct computation without deflation:
1. Compute $S = X^T Y$
2. For each component, find dominant singular vector of $S$ orthogonal to previous components
3. Project out found directions from $S$

**Complexity**: $O(pq + Kp^2)$, often faster than NIPALS for many components.

#### 3.2.4 IKPLS Algorithm

Improved kernel formulation:
- Algorithm 1: Direct QR-based
- Algorithm 2: Fast matrix multiplication variant

**Key Advantage**: Better numerical stability and consistent behavior across implementations.

### 3.3 Architecture Section

**Key Design Decisions**:

1. **Backend Abstraction**:
   ```python
   # Same code, different backends
   pls_numpy = PLSRegression(backend="numpy")
   pls_jax = PLSRegression(backend="jax")
   pls_torch = PLSRegression(backend="torch")
   ```

2. **Lazy Loading**:
   - JAX/PyTorch/TensorFlow only imported when used
   - Minimal dependencies for basic usage
   - Fast import time

3. **JIT Compilation**:
   - JAX: `@jax.jit` for kernel functions
   - Cached compilation for repeated calls
   - GPU memory management

4. **Numerical Precision**:
   - Float64 default for accuracy
   - Float32 option for memory-limited cases
   - Explicit handling of numerical edge cases

### 3.4 Validation Methodology

**Equivalence Testing Protocol**:

```python
def validate_equivalence(pls_gpu_model, reference_model, X, y, rtol=1e-5):
    """
    Validate numerical equivalence between implementations.

    Checks:
    1. Coefficients (coef_)
    2. X scores (x_scores_)
    3. X loadings (x_loadings_)
    4. Y loadings (y_loadings_)
    5. Predictions (predict(X))
    6. R² score
    """
    pls_gpu_model.fit(X, y)
    reference_model.fit(X, y)

    results = {}

    # Check predictions (most important)
    y_pred_gpu = pls_gpu_model.predict(X)
    y_pred_ref = reference_model.predict(X)
    results['predictions'] = np.allclose(y_pred_gpu, y_pred_ref, rtol=rtol)

    # Check coefficients (may differ by sign per component)
    results['coefficients'] = np.allclose(
        np.abs(pls_gpu_model.coef_),
        np.abs(reference_model.coef_),
        rtol=rtol
    )

    # ... additional checks

    return results
```

**Reference Implementations**:

| Method | Reference | Package |
|--------|-----------|---------|
| PLSRegression | sklearn | scikit-learn 1.3+ |
| SIMPLS | R pls | pls::simpls |
| OPLS | pyopls | pyopls |
| Kernel PLS | sklearn | sklearn KernelRidge + PLS |
| MB-PLS | R ade4 | ade4::mbpls |

### 3.5 Performance Benchmarks

**PLS-Specific Evaluation Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **RMSEP** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root Mean Square Error of Prediction |
| **RMSECV** | Same, via CV | Cross-validated RMSE |
| **Q²** | $1 - \frac{\sum(y_i - \hat{y}_{i,CV})^2}{\sum(y_i - \bar{y})^2}$ | Cross-validated R² |
| **RPD** | $\frac{SD(y)}{RMSEP}$ | Ratio of Performance to Deviation |
| **RER** | $\frac{range(y)}{RMSEP}$ | Ratio of Error Range |
| **BIAS** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$ | Systematic prediction error |
| **SEP** | $\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(e_i - \bar{e})^2}$ | Standard Error of Prediction |
| **RPIQ** | $\frac{IQR(y)}{RMSEP}$ | Ratio of Performance to IQR (preferred for skewed Y) |

> **Essential Chemometrics Metrics**: Q², RPD (or RPIQ), and RMSECV are mandatory for
> chemometrics publications. All real dataset results must report these metrics.
> The library provides `pls_gpu.utils.metrics` with functions to compute them.

**Interpretation Guidelines** (Williams 2014):
- RPD < 1.5: Not usable
- 1.5 ≤ RPD < 2.0: Rough screening
- 2.0 ≤ RPD < 2.5: Screening
- 2.5 ≤ RPD < 3.0: Quality control
- RPD ≥ 3.0: Process control

**Benchmark Protocol**:

1. **Hardware Configuration**:
   - CPU: AMD EPYC 7742 (64 cores) or Intel Xeon Gold
   - GPU: NVIDIA A100 40GB or RTX 4090
   - Memory: 256GB RAM
   - Storage: NVMe SSD

2. **Benchmark Parameters**:
   - n_samples: [100, 500, 1000, 5000, 10000, 50000]
   - n_features: [100, 500, 1000, 5000]
   - n_components: [5, 10, 20, 50]
   - repetitions: 10 per configuration

3. **Timing Methodology**:
   - Warm-up runs (discard first 3)
   - Mean ± std of 10 runs
   - Separate fit and predict timing
   - Include data transfer time for GPU

**Expected Results**:

| n_samples | n_features | NumPy (s) | JAX CPU (s) | JAX GPU (s) | Speedup |
|-----------|------------|-----------|-------------|-------------|---------|
| 1000 | 500 | 0.05 | 0.08 | 0.02 | 2.5x |
| 5000 | 1000 | 0.8 | 0.9 | 0.08 | 10x |
| 10000 | 2000 | 4.2 | 4.5 | 0.25 | 17x |
| 50000 | 1000 | 15.0 | 16.0 | 0.6 | 25x |

### 3.6 Real Dataset Case Studies

**Dataset Selection Criteria**:
- Diverse application domains
- Varied sizes (n, p)
- Different Y types (continuous, multivariate)
- Publicly available or reproducible

**Proposed Datasets**:

| Dataset | Domain | n | p | Targets | Source |
|---------|--------|---|---|---------|--------|
| Corn | NIR | 80 | 700 | 4 | Eigenvector |
| Meat | NIR | 215 | 100 | 1 (fat) | chemometrics |
| Wine | MIR | 124 | 842 | 3 | UCI |
| Tecator | NIR | 240 | 100 | 3 (fat, water, protein) | StatLib |
| Diesel | NIR | 245 | 401 | 4 | Eigenvector |
| Tablet | NIR | 655 | 404 | 1 (API) | Pharma |
| Octane | NIR | 39 | 226 | 1 (octane) | pls package |
| Soil | MIR | 3000+ | 3578 | Multiple | LUCAS |
| Metabolomics | NMR | 500+ | 10000+ | Disease | MTBLS |
| Multi-block | NIR+MIR | 200+ | 1000+ (2 blocks) | 1+ | Synthetic or literature |
| Process Time-Series | NIR | 500+ | 200+ | 1 (dynamic) | Process monitoring |

**Case Study Template**:

```
Dataset: [Name]
Domain: [Application area]
Samples: [n], Features: [p], Targets: [description]

Objective: [What is being predicted/classified]

Methods Compared:
- PLSRegression (simpls, nipals, ikpls)
- OPLS
- Sparse PLS (if many features)

Results:
- RMSECV / RMSEP (both reported)
- Q² (cross-validated)
- R² (calibration)
- RPD or RPIQ
- Optimal n_components (with selection method)
- GPU speedup (if n > 1000)

Interpretability (at least one case study):
- VIP score plot across wavelengths/features
- Identification of important spectral regions

Insights:
- Which method performed best
- Preprocessing importance
- Component interpretation
```

## 4. Figures and Tables

### 4.1 Required Figures

1. **Figure 1: Software Architecture**
   - Package structure diagram
   - Backend abstraction visualization
   - Data flow through pipeline

2. **Figure 2: Algorithm Comparison**
   - NIPALS vs SIMPLS convergence
   - Computational complexity comparison
   - Memory usage comparison

3. **Figure 3: Backend Performance Scaling**
   - Log-log plot: time vs n_samples
   - Multiple lines per backend
   - CPU vs GPU regions

4. **Figure 4: GPU Speedup Analysis**
   - Heatmap: speedup vs (n_samples, n_features)
   - Crossover point identification
   - Memory efficiency plot

5. **Figure 5: Numerical Equivalence**
   - Scatter: pls-gpu vs sklearn predictions
   - Histogram of differences
   - Per-component comparison

6. **Figure 6: Real Dataset Results**
   - Bar plot: R² by method and dataset
   - Radar chart: method characteristics
   - Timing comparison

7. **Figure 7: VIP Interpretation Example**
   - VIP score plot across wavelengths for one NIR case study
   - Identification of important spectral regions
   - Optional: VIP stability across CV folds

### 4.2 Required Tables

1. **Table 1: Method Overview**
   - All implemented methods
   - Algorithm variants
   - Complexity
   - Use cases

2. **Table 2: Backend Comparison**
   - Supported backends
   - GPU support
   - Platform availability
   - Dependencies

3. **Table 3: Numerical Validation**
   - Reference implementations
   - Max absolute difference
   - PASS/FAIL status

4. **Table 4: Performance Benchmarks**
   - Timing for standard sizes
   - Memory usage
   - Speedup factors

5. **Table 5: Real Dataset Summary**
   - Dataset characteristics
   - Best method per dataset
   - R² / RMSE values

## 5. Supplementary Material

### 5.1 Supplementary Methods

1. **S1: Extended Algorithm Details**
   - Full mathematical derivations
   - Pseudocode for all methods
   - Numerical stability considerations

2. **S2: Backend Implementation Details**
   - JAX JIT compilation strategy
   - PyTorch CUDA optimization
   - TensorFlow graph mode usage

### 5.2 Supplementary Results

3. **S3: Complete Benchmark Tables**
   - All tested configurations
   - Raw timing data
   - Statistical analysis

4. **S4: Extended Dataset Results**
   - Per-dataset detailed results
   - Cross-validation curves
   - Feature importance analyses

### 5.3 Supplementary Code

5. **S5: Reproduction Scripts**
   - Benchmark reproduction
   - Dataset preprocessing
   - Figure generation code

## 6. References (Core Citations)

### 6.1 Foundational PLS

1. Wold, H. (1966). Estimation of principal components. Multivariate Analysis.
2. Wold, S., et al. (1984). The collinearity problem in linear regression. SIAM JSC.
3. Geladi, P., Kowalski, B.R. (1986). Partial least-squares regression. Analytica Chimica Acta.
4. de Jong, S. (1993). SIMPLS: An alternative approach to PLS. Chemometrics and Intelligent Laboratory Systems.
5. Dayal, B.S., MacGregor, J.F. (1997). Improved PLS algorithms. Journal of Chemometrics.

### 6.2 PLS Variants

6. Trygg, J., Wold, S. (2002). Orthogonal projections to latent structures (O-PLS). Journal of Chemometrics.
7. Rosipal, R., Trejo, L.J. (2001). Kernel partial least squares. Machine Learning.
8. Westerhuis, J.A., et al. (1998). Analysis of multiblock data. Journal of Chemometrics.
9. Kim, S., et al. (2005). Locally weighted PLS. Chemometrics and Intelligent Laboratory Systems.
10. Chun, H., Keleş, S. (2010). Sparse PLS. JRSS-B.
11. Bouhaddani, S.E., et al. (2016). Evaluation of O2PLS in Omics data integration. BMC Bioinformatics.

### 6.3 Software Papers

11. Mevik, B.H., Wehrens, R. (2007). The pls package. Journal of Statistical Software.
12. Rohart, F., et al. (2017). mixOmics. PLOS Computational Biology.
13. Pedregosa, F., et al. (2011). Scikit-learn. JMLR.
14. Engstrøm, O., et al. (2024). ikpls. Journal of Open Source Software.

### 6.4 GPU Computing

15. Bradbury, J., et al. (2018). JAX: composable transformations. GitHub.
16. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
17. Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning. OSDI.

## 7. Author Contributions

**Proposed CRediT Statement**:

- **Conceptualization**: [Author 1]
- **Methodology**: [Author 1, Author 2]
- **Software**: [Author 1, Author 2, Author 3]
- **Validation**: [Author 2, Author 3]
- **Formal Analysis**: [Author 1, Author 2]
- **Investigation**: [Author 1, Author 2]
- **Data Curation**: [Author 3]
- **Writing - Original Draft**: [Author 1]
- **Writing - Review & Editing**: [All]
- **Visualization**: [Author 2]
- **Supervision**: [Author 1]
- **Project Administration**: [Author 1]

## 8. Timeline for Publication

### 8.1 Writing Schedule

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1-2 | Outline refinement, figure drafts | Detailed outline |
| 3-4 | Introduction, Background | Draft sections 1-2 |
| 5-6 | Architecture, Methods | Draft sections 3-4 |
| 7-8 | Validation, Performance | Draft sections 5-6 |
| 9-10 | Case Studies, Discussion | Draft sections 7-8 |
| 11 | Conclusion, Abstract, Supplementary | Complete draft |
| 12 | Internal review, revisions | Revised draft |
| 13 | Co-author review | Approved draft |
| 14 | Final formatting, submission | Submitted manuscript |

### 8.2 Post-Submission

- Expected review time: 2-4 months
- Revision time: 2-4 weeks
- Total to publication: ~6 months from submission

## 9. Data and Code Availability Statement

**Template**:

> All source code for pls-gpu is available under the MIT license at https://github.com/[username]/pls-gpu. The package can be installed via pip: `pip install pls-gpu`. Documentation is available at https://pls-gpu.readthedocs.io. Benchmark scripts and datasets are included in the repository under `bench/`. All results presented in this paper can be reproduced using the scripts in `paper/reproduce.py`.

## 10. Checklist Before Submission

### 10.1 Technical

- [ ] All methods implemented and tested
- [ ] Numerical validation complete
- [ ] Benchmarks run and documented
- [ ] Code publicly available
- [ ] Documentation complete
- [ ] PyPI release published

### 10.2 Manuscript

- [ ] Abstract ≤ 250 words
- [ ] Keywords selected (5-7)
- [ ] All figures high resolution (300+ DPI)
- [ ] Tables properly formatted
- [ ] References complete and consistent
- [ ] Supplementary material prepared
- [ ] Co-author approvals obtained

### 10.3 Submission

- [ ] Cover letter prepared
- [ ] Highlights prepared (if required)
- [ ] Graphical abstract prepared (if required)
- [ ] Author contributions statement
- [ ] Conflict of interest statement
- [ ] Data availability statement
- [ ] ORCID IDs for all authors

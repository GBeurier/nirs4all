# PLS methods for NIRS in Python

Scope: NIR spectroscopy calibration and classification. Focus on PLS-family and NIR-specific wrappers. Each entry gives method, Python lib, short explanation, and a minimal usage snippet. Where no maintained Python exists, I give a concise dev plan.

---

## Core regressors

### PLSRegression (NIPALS PLS1/PLS2)

**Python:** `scikit-learn`
**Why:** Baseline for NIRS calibration and multivariate Y. Stable and fast. ([Scikit-learn][1])
**Example**

```python
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=10)   # tune by CV
pls.fit(Xcal, ycal)                    # X: spectra, y: lab values
ypred = pls.predict(Xval)
```

### SIMPLS

**Python:** no standard maintained package
**Why:** Same predictions as PLSRegression with different deflation; common in chemometrics literature. Implementable. ([ScienceDirect][2])
**Dev plan:** Implement de Jong 1993 SIMPLS: iterative weight via dominant eigenvector of (X^\top Y Y^\top X), deflate covariance, accumulate loadings; mirror `sklearn` API for fit/predict; add CV.

### Kernel PLS (KPLS)

**Python:** `ikpls` (NumPy + JAX backends)
**Why:** Nonlinear latent space. Useful with complex spectra–property relations; supports fast CV and GPU. ([Ikpls][3])
**Example**

```python
from ikpls.numpy_ikpls import PLS as KPLS
model = KPLS(algorithm=1)              # IKPLS Alg. #1
model.fit(Xcal, ycal, A=15)            # A = max comps
yp = model.predict(Xval, n_components=8)
```

### Improved Kernel PLS (IKPLS + fast CV)

**Python:** `ikpls.fast_cross_validation` (NumPy) and JAX variants
**Why:** Orders-of-magnitude faster CV for PLS/IKPLS with correct centering/scaling; weighted CV supported. ([Ikpls][3])
**Example**

```python
from ikpls.fast_cross_validation.numpy_ikpls import PLS as FastKPLS
fk = FastKPLS()
fk.cross_validate(Xcal, ycal, A=20, center_X=True, scale_X=True)  # returns CV RMSE by comps
```

---

## Discriminant analysis

### PLS-DA

**Python:** use `sklearn` PLSRegression with one-hot Y; or `pypls` (research)
**Why:** Standard for classifying categories from spectra. ([Scikit-learn][1])
**Example**

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression

enc = OneHotEncoder(sparse_output=False).fit(ycls.reshape(-1,1))
Y = enc.transform(ycls.reshape(-1,1))  # one-hot
plsda = PLSRegression(n_components=8).fit(X, Y)
yhat = enc.inverse_transform(plsda.predict(X))
```

### OPLS / OPLS-DA

**Python:** `pyopls`
**Why:** Removes Y-orthogonal X-variation to sharpen interpretability; then use 1-comp PLS on filtered scores. Common in metabolomics and applicable to NIRS. ([GitHub][4])
**Example**

```python
from pyopls import OPLS
from sklearn.cross_decomposition import PLSRegression

Z = OPLS(k=1).fit_transform(X, y)     # strip orthogonal variance
clf = PLSRegression(n_components=1).fit(Z, y)
```

---

## Block-structured and dynamic

### MB-PLS (Multiblock PLS)

**Python:** `mbpls`
**Why:** Fuse multiple X blocks (sensor sets, preprocessing variants). scikit-learn style. ([mbpls.readthedocs.io][5])
**Example**

```python
from mbpls.mbpls import MBPLS
mb = MBPLS(n_components=6).fit([Xraw, Xsnv], y)
yp = mb.predict([Xraw_val, Xsnv_val])
```

### DiPLS (Dynamic PLS) and SMB-PLS

**Python:** `trendfitter`
**Why:** Handle time-lagged process/NIR streams via Hankelization; SPC metrics included. ([PyPI][6])
**Example**

```python
from trendfitter import DiPLS
dpls = DiPLS(n_components=8, lags=5).fit(Xtime, ytime)
yhat = dpls.predict(Xtime_val)
```

### Locally-Weighted PLS (LW-PLS)

**Python:** `lwpls` (GitHub)
**Why:** Just-in-time models near each query sample. Useful with drift or local nonlinearity. ([GitHub][7])
**Example**

```python
from lwpls.Python.lwpls import LWPLS  # repo layout
lw = LWPLS(n_components=6, kernel_width=0.5).fit(X, y)
yp = lw.predict(Xval)
```

---

## Multiway/tensor

### N-PLS (multiway PLS)

**Python:** `npls`
**Why:** For 3-way data (sample × wavelength × time/repetition). Use if NIRS has intrinsic multiway structure. ([PyPI][8])
**Example**

```python
from npls import NPLS
npls = NPLS(n_components=3, l1=0.0, l2=0.0).fit(X3d, y)
yp = npls.predict(X3d_val)
```

### HOPLS

**Python:** no maintained package
**Why:** Higher-order Tucker blocks; rarely needed for standard 2-D NIRS.
**Dev plan:** Implement as alternating least squares on Tucker cores with PLS-style covariance objective; follow original HOPLS paper; wrap like `sklearn`.

---

## Variable/interval selection wrappers for PLS

### VIP (Variable Importance in Projection)

**Python:** in `auswahl` as a selector
**Why:** Rank wavelengths from a fitted PLS; quick filter. ([auswahl.readthedocs.io][9])
**Example**

```python
from auswahl import VIP
vip = VIP(pls_kwargs=dict(n_components=8)).fit(X, y)
idx = vip.get_support(indices=True)
X_sel = X[:, idx]
```

### MC-UVE / UVE-PLS

**Python:** `auswahl.MCUVE`
**Why:** Stability of regression weights under resampling; robust wavelength culling. ([auswahl.readthedocs.io][9])
**Example**

```python
from auswahl import MCUVE
mcuve = MCUVE(pls_kwargs=dict(n_components=8), n_iter=200).fit(X, y)
mask = mcuve.get_support()
```

### CARS, SPA, Random Frog, VISSA

**Python:** `auswahl`
**Why:** Strong NIR track record for wavelength selection; all provided as selectors that wrap PLS scoring. ([auswahl.readthedocs.io][9])
**Example**

```python
from auswahl import CARS
cars = CARS(pls_kwargs=dict(n_components=8), n_runs=50, n_select=80).fit(X, y)
X_sel = X[:, cars.get_support(indices=True)]
```

### iPLS (interval PLS)

**Python:** no single canonical lib; build with `sklearn`
**Why:** Grid or heuristic search over contiguous wavelength windows with local PLS.
**Dev plan:** Slice X by intervals, fit `PLSRegression` per window, select best by CV or stack as ensemble; optionally combine with CARS for interval seeds. (Use `sklearn` CV utilities.)

---

## Robust and sparse PLS

### Robust PLS / RSIMPLS

**Python:** no standard maintained lib
**Why:** Down-weight outliers. Useful if spectra contain leverage outliers.
**Dev plan:** Iterate PLS with robust weights (Huber/Tukey) on X and residuals; or compute robust covariance in SIMPLS loop; reweight until convergence; cross-validate weight tuning.

### Sparse PLS

**Python:** `py-ddspls` (ddsPLS); `sparse-pls` (emerging)
**Why:** Joint prediction and variable selection by penalized loadings. Useful when many wavelengths. ([PyPI][10])
**Example (ddsPLS)**

```python
from ddsPLS import ddsPLS
model = ddsPLS(n_components=5).fit(X, y)     # auto-tunes lambda
yp = model.predict(Xval)
```

---

## Methods to include cautiously or avoid for standard NIRS

* **PLSCanonical / CCA-style PLS**
  **Why not primary:** Maximizes symmetric correlation between X and Y scores. NIRS calibration targets prediction of Y from X. Use `PLSRegression` instead. Keep for exploratory X–Y association only. ([Scikit-learn][11])

* **K-OPLS**
  **Status:** Kernelized OPLS is documented in literature, but no maintained Python. Benefit over KPLS for NIRS is uncertain and adds complexity. Prefer KPLS or OPLS unless you need kernelized orthogonal filtering. ([PMC][12])

* **HOPLS**
  **Why rarely needed:** Overkill for 2-D NIRS unless you have genuine tensor structure (e.g., spatial–spectral cubes). See N-PLS first.

* **Neuroimaging-specific PLS toolkits (`pyls`)**
  **Why not:** Focus on cross-covariance brain–behavior analysis, not NIR calibration workflows. Prefer `PLSRegression`. ([GitHub][13])

---

## Minimal build signatures (standardize to scikit-learn style)

All custom implementations should follow:

```python
class MyPLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=10, **kwargs): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
```

* Add `.x_scores_`, `.x_loadings_`, `.x_weights_`, `.coef_` to match `sklearn` for interoperability. ([Scikit-learn][1])

---

## Quick selection of what to ship first for NIRS

1. `PLSRegression` (baseline), 2) `OPLS` preprocessor + 1-comp PLS, 3) `KPLS/IKPLS` with fast CV, 4) `MB-PLS` if multiple X blocks, 5) wrappers: `VIP`, `MC-UVE`, `CARS`, `SPA`, 6) `DiPLS` for time-dependence, 7) `sPLS` when variable selection is needed. ([Scikit-learn][1])

---

## References and docs

* `PLSRegression` API and attributes. ([Scikit-learn][1])
* SIMPLS original paper. ([ScienceDirect][2])
* `ikpls` docs and API. ([Ikpls][3])
* `pyopls` usage. ([GitHub][4])
* `mbpls` docs and examples. ([mbpls.readthedocs.io][5])
* `trendfitter` DiPLS and SMB-PLS. ([PyPI][6])
* `lwpls` repo. ([GitHub][7])
* `npls` package. ([PyPI][8])
* `auswahl` selectors (VIP, MC-UVE, CARS, SPA, VISSA). ([auswahl.readthedocs.io][9])
* Sparse PLS options (`py-ddspls`, `sparse-pls`). ([PyPI][10])
* Why PLSCanonical is not primary for NIRS calibration. ([Scikit-learn][11])


[1]: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html "PLSRegression — scikit-learn 1.7.2 documentation"
[2]: https://www.sciencedirect.com/science/article/abs/pii/016974399385002X?utm_source=chatgpt.com "An alternative approach to partial least squares regression"
[3]: https://ikpls.readthedocs.io/ "Improved Kernel Partial Least Squares (IKPLS) and Fast Cross-Validation — IKPLS 3.0.0.post1 documentation"
[4]: https://github.com/BiRG/pyopls "GitHub - BiRG/pyopls: A Python 3 implementation of orthogonal projection to latent structures"
[5]: https://mbpls.readthedocs.io/ "Multiblock Partial Least Squares Package — mbpls 1.0.2 documentation"
[6]: https://pypi.org/project/trendfitter/ "trendfitter · PyPI"
[7]: https://github.com/hkaneko1985/lwpls "GitHub - hkaneko1985/lwpls: Locally-Weighted Partial Least Squares (LWPLS)"
[8]: https://pypi.org/project/npls/?utm_source=chatgpt.com "npls"
[9]: https://auswahl.readthedocs.io/en/latest/point_selection.html "1. Wavelength Point Selection — auswahl 0.9.0 documentation"
[10]: https://pypi.org/project/py-ddspls/?utm_source=chatgpt.com "py-ddspls"
[11]: https://scikit-learn.org/stable/modules/cross_decomposition.html?utm_source=chatgpt.com "1.8. Cross decomposition"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2323673/?utm_source=chatgpt.com "K-OPLS package: Kernel-based orthogonal projections to ..."
[13]: https://github.com/rmarkello/pyls?utm_source=chatgpt.com "rmarkello/pyls: A Python implementation of Partial Least ..."






| Method                        | Std implementation (R / etc.)      | Python implementation(s)                                                 | Implementation difficulty                                                        |
| ----------------------------- | ---------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| PLS1/PLS2 (NIPALS)            | R: `pls::plsr(method="nipals")`    | `scikit-learn` PLSRegression                                             | Low. Direct use or light wrappers. ([CRAN][1])                                   |
| SIMPLS                        | R: `pls::plsr(method="simpls")`    | None standard; reimplement from de Jong (1993)                           | Low–Medium. Closed-form deflation, well documented. ([CRAN][1])                  |
| Kernel PLS (KPLS)             | R: `pls::plsr(method="kernelpls")` | `ikpls` (NumPy/JAX)                                                      | Medium. Kernelization and cross-val speedups handled in `ikpls`. ([CRAN][1])     |
| Improved Kernel PLS (IKPLS)   | —                                  | `ikpls` (NumPy/JAX, CPU/GPU)                                             | Low. Use package API. ([JOSS][2])                                                |
| PLS-DA                        | R: `mixOmics::plsda`               | Use `scikit-learn` PLSRegression with one-hot Y; `pypls` includes PLS-DA | Low. Dummy-coded Y. ([mixomics.org][3])                                          |
| OPLS / OPLS-DA                | R: `ropls::opls`                   | `pyopls`; `pypls` (OPLS-DA)                                              | Medium. Orthogonal filter + 1-comp PLS core. ([GitHub][4])                       |
| Kernel OPLS (K-OPLS)          | MATLAB/JS code in papers/toolboxes | No maintained Python known                                               | Medium–High. Need kernel OPLS derivation + centering/deflation tests. ([PMC][5]) |
| MB-PLS (multiblock)           | R: `ade4::mbpls`                   | `mbpls` (scikit-learn API)                                               | Low–Medium. Use `mbpls`. ([rdocumentation.org][6])                               |
| SO-PLS / PO-PLS               | R: `multiblock::sopls`             | No stable general Python; implement on top of PLS                        | Medium. Sequencing + orthogonalisation bookkeeping. ([rdrr.io][7])               |
| O2PLS / O2PLS-DA              | R: `OmicsPLS`, `o2plsda`           | No mainstream Python; niche repos exist                                  | Medium–High. Bi-directional orthogonal parts and CV. ([BioMed Central][8])       |
| N-PLS (multiway)              | MATLAB/R toolboxes                 | `npls` (N-PLS1, PyPI)                                                    | Medium. Tensor shapes and deflation details. ([PyPI][9])                         |
| HOPLS                         | —                                  | No maintained Python package found                                       | High. Tensor Tucker blocks + deflation. ([arXiv][10])                            |
| iPLS (interval PLS)           | R: `mdatools::ipls`                | `auswahl` has IPLS example utilities                                     | Medium. Interval CV loop around PLS. ([mda.tools][11])                           |
| UVE-PLS / MC-UVE              | R: `plsVarSel::mcuve_pls`          | `pynir` includes MC-UVE; `auswahl` docs                                  | Medium. Stability selection around PLS. ([rdrr.io][12])                          |
| GA-PLS                        | R/Matlab toolboxes                 | No standard Python                                                       | Medium–High. GA wrapper around PLS. ([GitHub][13])                               |
| Moving-window PLS             | —                                  | No standard Python; implement                                            | Medium. Sliding window modelling + CV. ([Nan Xiao | 肖楠][14])                     |
| Robust PLS (e.g., RSIMPLS)    | R: `rpls`                          | No standard Python                                                       | Medium–High. Robust covariances + weighting. ([CRAN][15])                        |
| Recursive PLS (RPLS)          | MATLAB/R in literature             | No standard Python; implement                                            | Medium. Online updates + forgetting. ([ScienceDirect][16])                       |
| Dynamic PLS (DiPLS)           | MATLAB/R toolboxes                 | `trendfitter` (DiPLS, SMB-PLS)                                           | Medium. Time-lagged Hankel X; package exists. ([PyPI][17])                       |
| Locally-Weighted PLS (LW-PLS) | —                                  | `lwpls` (GitHub)                                                         | Medium. Local kernel + small PLS per query. ([GitHub][18])                       |
| Sparse PLS (sPLS) / sPLS-DA   | R: `mixOmics::spls`, `splsda`      | Emerging: `sparse-pls`, `py-ddspls`                                      | Medium. Penalised loadings; tune sparsity. ([rdrr.io][19])                       |
| Neuroimaging PLS toolkits     | MATLAB reference toolboxes         | `pyls`, `plspy`                                                          | Low–Medium. Domain-specific, usable components. ([pyls.readthedocs.io][20])      |

Notes:

* General PLS in Python: `scikit-learn` PLSRegression is the baseline. ([Scikit-learn][21])
* High-performance kernels and JAX backends: `ikpls`. ([JOSS][2])
* Multiblock coverage in Python: `mbpls` and `trendfitter` cover common chemometric workflows. ([mbpls.readthedocs.io][22])

If you want, I can turn this into a build plan: packages to vendor vs. re-code, with test references and JAX targets.

[1]: https://cran.r-project.org/web/packages/pls/vignettes/pls-manual.pdf?utm_source=chatgpt.com "Introduction to the pls Package"
[2]: https://joss.theoj.org/papers/10.21105/joss.06533?utm_source=chatgpt.com "IKPLS: Improved Kernel Partial Least Squares and Fast Cross ..."
[3]: https://mixomics.org/methods/spls-da/?utm_source=chatgpt.com "(s)PLS-DA"
[4]: https://github.com/BiRG/pyopls?utm_source=chatgpt.com "BiRG/pyopls: A Python 3 implementation of orthogonal ..."
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2323673/?utm_source=chatgpt.com "K-OPLS package: Kernel-based orthogonal projections to ..."
[6]: https://www.rdocumentation.org/packages/ade4/versions/1.7-23/topics/mbpls?utm_source=chatgpt.com "mbpls function - Multiblock partial least squares"
[7]: https://rdrr.io/cran/multiblock/man/sopls.html?utm_source=chatgpt.com "sopls: Sequential and Orthogonalized PLS (SO-PLS)"
[8]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2371-3?utm_source=chatgpt.com "Integrating omics datasets with the OmicsPLS package"
[9]: https://pypi.org/project/npls/?utm_source=chatgpt.com "npls"
[10]: https://arxiv.org/abs/1207.1230?utm_source=chatgpt.com "Higher-Order Partial Least Squares (HOPLS): A Generalized Multi-Linear Regression Method"
[11]: https://mda.tools/docs/ipls.html?utm_source=chatgpt.com "Interval PLS | Getting started with mdatools for R"
[12]: https://rdrr.io/cran/plsVarSel/man/mcuve_pls.html?utm_source=chatgpt.com "mcuve_pls: Uninformative variable elimination in PLS ..."
[13]: https://github.com/khliland/plsVarSel?utm_source=chatgpt.com "khliland/plsVarSel: Variable selection methods for Partial ..."
[14]: https://nanx.me/papers/interval-variable-selection.pdf?utm_source=chatgpt.com "interval-variable-selection.pdf"
[15]: https://cran.r-project.org/web/packages/rpls/rpls.pdf?utm_source=chatgpt.com "Package 'rpls' - Robust Partial Least Squares"
[16]: https://www.sciencedirect.com/science/article/abs/pii/S0967066102000965?utm_source=chatgpt.com "Recursive partial least squares algorithms for monitoring ..."
[17]: https://pypi.org/project/trendfitter/?utm_source=chatgpt.com "trendfitter"
[18]: https://github.com/hkaneko1985/lwpls?utm_source=chatgpt.com "Locally-Weighted Partial Least Squares (LWPLS)"
[19]: https://rdrr.io/bioc/mixOmics/man/spls.html?utm_source=chatgpt.com "Sparse Partial Least Squares (sPLS) in mixOmics"
[20]: https://pyls.readthedocs.io/en/stable/?utm_source=chatgpt.com "pyls: Partial Least Squares in Python"
[21]: https://scikit-learn.org/stable/modules/cross_decomposition.html?utm_source=chatgpt.com "1.8. Cross decomposition"
[22]: https://mbpls.readthedocs.io/?utm_source=chatgpt.com "Multiblock Partial Least Squares Package — mbpls 1.0.2 ..."









Here is a field-tested catalog of PLS variants used in chemometrics, with canonical sources and current implementations (R / Python incl. NumPy & JAX / C++ when known).

# Core regression algorithms

* **PLS1 / PLS2 via NIPALS** — original iterative scheme. R: `pls::plsr(method="nipals")`. Python: `sklearn.cross_decomposition.PLSRegression` (NIPALS). Source: Wold (NIPALS), scikit-learn docs. ([scikit-learn.org][1])
* **SIMPLS** — covariance-maximizing, deflation without augmented data. R: `pls::plsr(method="simpls")`. Source: de Jong 1993.
* **Kernel PLS (KPLS)** — nonlinear PLS in RKHS. R: `pls::plsr(method="kernelpls")`. Source: Rosipal & Trejo 2001 (JMLR).
* **Improved Kernel PLS (IKPLS)** — fast KPLS variants. Python: `ikpls` (CPU NumPy + GPU/TPU JAX). Method source + package/JOSS refs.

# Classification and orthogonalized variants

* **PLS-DA** — PLS with dummy-coded Y. R: `mixOmics::plsda`. Python: implement with `PLSRegression` + one-hot Y. Source: mixOmics guide; general PLS-DA notes.
* **OPLS / OPLS-DA** — splits predictive vs Y-orthogonal X variation. R: `ropls::opls`. Python: `pyopls`, `pypls` (PLS-DA/OPLS-DA). Method source overview.
* **Kernel OPLS (K-OPLS)** — kernelized OPLS. R/MATLAB implementation (open-source).
* **OPLS-HDA** — recent multiclass extension of OPLS-DA. (method paper).

# Multi-block / data-fusion PLS

* **MB-PLS** — multiblock PLS. R: `ade4::mbpls` (+ `mbclusterwise::mbpls.fast`).
* **SO-PLS / PO-PLS** — sequential or parallel orthogonalized PLS for multiple X-blocks. R: `multiblock::sopls` (and vignette with SO-PLS, PO-PLS, sparse MB-PLS).
* **O2PLS** — bidirectional orthogonalized two-block PLS. (method overview and references).

# Multiway / tensor PLS

* **N-PLS (multiway PLS)** — for 3-way+ arrays; used in spectroscopy and process analytics. (N-way PLS references).
* **HOPLS** — higher-order PLS via Tucker blocks. Open-access method paper.

# Interval and wavelength-selection PLS

* **iPLS** — interval PLS; local models on contiguous spectral windows. Method + implementation context (R ecosystem). ([opg.optica.org][2])
* **UVE-PLS / MC-UVE family** — uninformative variable elimination with PLS; Monte-Carlo extensions.
* **GA-PLS** — genetic-algorithm selection wrapped around PLS. Toolbox (MATLAB) and seminal paper.
* **Moving-window PLS/selection** — moving window variants over spectra (often paired with UVE/SPA).

# Robust PLS

* **Robust PLS / RSIMPLS** — down-weights outliers; algorithmic and statistical treatments. (series overview).

# Dynamic, adaptive, and local PLS

* **Recursive PLS (RPLS)** — online model updates for drifting processes. Classic algorithms.
* **Dynamic PLS (DPLS / DKPLS)** — PLS on time-lagged (Hankel) X; kernelized dynamic variants exist. Survey and DKPLS paper.
* **Locally Weighted PLS (LW-PLS)** — just-in-time local models around each query. Foundations and recent analysis. ([pubs.acs.org][3])

# Sparse and penalized PLS

* **sPLS / sPLS-DA** — lasso-penalized loadings for selection + prediction. R: `mixOmics::spls`, `splsda`. Recent dual-sPLS variant in R (`dual.spls`).

# Tooling snapshots (what you can use today)

* **R (broadest coverage)**: `pls` (NIPALS/SIMPLS/kernelPLS); `mixOmics` (PLS, PLS-DA, sPLS, sPLS-DA, multi-block DIABLO/MINT); `ropls` (OPLS/OPLS-DA); `ade4` (MB-PLS); `multiblock` (SO-PLS/PO-PLS, sparse MB-PLS).
* **Python (production-ready pieces)**: `scikit-learn` PLSRegression (NIPALS); `ikpls` (KPLS/IKPLS, NumPy+JAX, CPU/GPU); `pyopls` / `pypls` (OPLS / OPLS-DA / PLS-DA). ([scikit-learn.org][1])
* **C++**: no widely used standalone chemometric PLS libraries; many R packages rely on Rcpp/Armadillo backends; MATLAB toolboxes dominate for legacy workflows. ([cran.r-project.org][4])

# Notes for implementation coverage in your library

* You already get **PLS1/PLS2 (NIPALS)** from scikit-learn. Add **SIMPLS** and **KPLS/IKPLS** for speed and nonlinearity (reuse `ikpls` JAX/NumPy backends).
* For chemometric practice, prioritize: **PLS-DA**, **OPLS-(DA)**, **iPLS / UVE-PLS / GA-PLS**, **MB-PLS / SO-PLS**, **RPLS/DPLS**, **sPLS**. The cited packages and papers give algorithms and testbeds for re-implementation. ([opg.optica.org][2])

If you want, I can map these into a concrete backlog: minimal API for each variant, refs to reproduce unit tests, and which ones to wrap vs re-code (with JAX paths where useful).

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html?utm_source=chatgpt.com "PLSRegression — scikit-learn 1.7.2 documentation"
[2]: https://opg.optica.org/abstract.cfm?uri=as-54-3-413&utm_source=chatgpt.com "Interval Partial Least-Squares Regression (iPLS)"
[3]: https://pubs.acs.org/doi/abs/10.1021/ac980208r?utm_source=chatgpt.com "Optimization in Locally Weighted Regression - ACS Publications"
[4]: https://cran.r-project.org/package%3Dprospectr?utm_source=chatgpt.com "CRAN: Package prospectr - R Project"


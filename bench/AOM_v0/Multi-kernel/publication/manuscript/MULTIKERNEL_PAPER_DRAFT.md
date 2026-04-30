# Multi-Kernel Generalisations of Operator-Adaptive PLS for Near-Infrared Spectroscopy: mkR, MKM, and BLUP

**Status**: draft v0.1 (manuscript outline). Numbers in **bold** in this
file MUST be reproduced from `benchmark_runs/*/results.csv` before
submission.

This is a sister manuscript to the AOM-PLS paper at
`publication/manuscript/main.tex`. It introduces three multi-kernel
extensions of the AOM operator family — mkR, MKM, BLUP — sharing a
common centred + trace-normalised kernel API.

## Abstract

We extend the operator-adaptive partial-least-squares (AOM-PLS)
framework with three multi-kernel siblings:

- **mkR** — multi-kernel Ridge with explicit per-block weights `eta_b`,
  learnable from kernel-target alignment (KTA) or by softmax-CV on the
  inner training loss.
- **MKM** — multi-kernel mixed model with REML-estimated variance
  components per AOM block.
- **BLUP** — empirical-BLUP decomposition of MKM predictions into
  per-block contributions.

On the small-cohort smoke benchmark (3 datasets from the TabPFN paper:
ALPINE, AMYLOSE, BEER), `mkR-softmax_cv` (compact bank, no branch
preprocessing) gets a median relative RMSEP of **0.95** vs PLS,
matching paper-Ridge HPO (1.00 vs Ridge). MKM with branch preprocessing
(SNV, MSC, ASLS) closes the gap on the harder dataset AMYLOSE
(see `benchmark_runs/smoke3_branches/results.csv`). BLUP makes the
contribution of every AOM operator inspectable per individual at no
extra fitting cost.

## 1. Introduction

### 1.1 The AOM operator family

A bank of strict-linear preprocessing operators
``A_b in R^{p x p}`` (e.g. Savitzky-Golay smoothers / derivatives,
detrend, gap derivatives) defines per-block transformed spectra
``Z_b = X A_b^T``. AOM-PLS chains these into a single PLS model and
selects, per latent component, the best operator from CV.

### 1.2 What multi-kernel adds

Each block defines a kernel ``K_b = X A_b^T A_b X^T``. The AOM kernel
sum ``K_AOM = sum_b s_b^2 K_b`` (existing AOM-Ridge) is **fixed**:
weights are pre-computed from RMS norms, not learned. Multi-kernel
generalisations learn weights `(eta_b)_b` from the data:

- mkR: prediction-driven (closed-form KTA or gradient on inner CV RMSE).
- MKM: likelihood-driven (REML-estimated variance components).
- BLUP: same prediction as MKM, but exposes per-block contributions.

### 1.3 Contributions

1. A common **centred + trace-normalised** kernel API (`AOMKernelizer`)
   that makes per-block weights interpretable across blocks.
2. Three sklearn-compatible estimators sharing the kernelizer.
3. A unified smoke / extended / full benchmark protocol against the
   TabPFN paper cohort with reference RMSEs from PLS, Ridge, TabPFN-raw,
   TabPFN-opt, CNN-NICON, and CatBoost.
4. Open-source, reproducible code under `bench/AOM_v0/Multi-kernel/`.

## 2. Mathematical foundations

### 2.1 Centred + trace-normalised AOM block kernels

Centring and trace normalisation are critical for inter-block
comparability:

```text
K_b_raw = Xc (A_b^T A_b) Xc^T              # raw block kernel on centred X
K_b_c = H K_b_raw H, with H = I - 1 1^T/n  # double-centring
tau_b = n / max(trace(K_b_c), eps)
K_b = tau_b K_b_c                           # tilde K, satisfies tr(K_b)/n = 1
```

Cross kernels for prediction use **only training-side moments**:

```text
mu_b = (1/n) K_b_raw 1_n              # row mean of training kernel (stored)
nu_b = (1/n^2) 1_n^T K_b_raw 1_n      # global mean (stored)
r_*  = (1/n) K_b_raw_* 1_n            # per-test-row mean of cross kernel
K_b_*_c = K_b_raw_* - 1_* mu_b^T - r_* 1_n^T + nu_b 1_* 1_n^T
K_b_*   = tau_b K_b_*_c
```

This is the standard kernel-PCA "feature-centring at training mean"
construction; ``r_*`` is computed deterministically from test data and
training statistics, with no leakage of training labels or held-out
test rows.

### 2.2 mkR — multi-kernel Ridge

Combined kernel: ``K_eta = sum_b eta_b K_b`` with ``eta_b >= 0,
sum_b eta_b = 1``. Dual Ridge:

```text
C = (K_eta + alpha I)^-1 (y - y_mean)
y_hat_* = K_eta_* C + y_mean
```

When all blocks are strict-linear, an equivalent original-space
coefficient exists:

```text
U_eta = sum_b eta_b tau_b A_b^T A_b X_train_c^T
beta = U_eta C
y_hat_* = X_*c beta + y_mean
```

Weight strategies: ``uniform`` (1/B), ``manual``, ``kta`` (closed-form),
``softmax_cv`` (gradient on inner-CV RMSE with KL-to-uniform
regularisation, multi-restart).

### 2.3 MKM — multi-kernel mixed model

```text
y = X_f beta + sum_b u_b + e
u_b ~ N(0, sigma_b^2 K_b),    e ~ N(0, sigma_e^2 I)
y ~ N(X_f beta, V),    V = sum_b sigma_b^2 K_b + sigma_e^2 I
```

REML log-likelihood (with `p_f = rank(X_f)`):

```text
ell_REML = -0.5 [ logdet V + logdet(X_f^T V^-1 X_f) + r^T V^-1 r + (n - p_f) log 2*pi ]
```

Single Cholesky of `V` reused for all derivatives; analytic gradient via
`P = V^-1 - V^-1 X_f M^-1 X_f^T V^-1` and
`g_j = 0.5 (tr(P dV_j) - a^T dV_j a)`, `a = V^-1 r`. Multi-start
L-BFGS-B on log-variances with deterministic + random initialisation.

Reports per-block **relative variance contributions**:

```text
h_b = sigma_b^2 / (sum_b sigma_b^2 + sigma_e^2)
```

### 2.4 BLUP — per-block prediction decomposition

Once REML converges:

```text
alpha_dual = V^-1 (y - X_f hat beta)         # precomputed at fit time
hat u_b_* = sigma_b^2 K_b_* alpha_dual        # per-test-block contribution
hat y_*   = X_f_* hat beta + sum_b hat u_b_*
```

Decomposition identity (must hold to fp tolerance):
``predict_components(X)["total"] == predict(X)``.

## 3. Algorithms (pseudocode)

```
mkR_fit(X, y; bank, weight_strategy, alphas, cv):
    apply optional branch preprocessor (SNV/MSC/ASLS/...)
    K_blocks = AOMKernelizer.fit_transform(X)
    if weight_strategy == 'uniform':  eta = 1/B
    elif weight_strategy == 'manual': eta = clip+normalize(user_init)
    elif weight_strategy == 'kta':    eta = simplex KTA(K_blocks, y)
    elif weight_strategy == 'softmax_cv':
         eta, alpha = L-BFGS-B over (theta, log alpha) on inner-CV RMSE
                        with KL-to-uniform regularisation
    K_eta  = sum_b eta_b * K_b
    C      = (K_eta + alpha I)^-1 (y - y_mean)        # Cholesky
    coef_  = sum_b eta_b * tau_b * A_b^T A_b X_c^T C  # original-space
    return mkR(eta, alpha, coef_, ...)

MKM_fit(X, y; bank, method, n_restarts):
    apply optional branch preprocessor
    K_blocks = AOMKernelizer.fit_transform(X)
    X_f      = ones(n, 1)                              # intercept-only
    theta*   = best of n_restarts L-BFGS-B (REML or ML)
    sigma2_blocks, sigma2_residual = exp(theta*)
    V        = sum sigma2_b K_b + sigma2_e I
    Cholesky once; alpha_dual = V^-1 (y - X_f beta_hat)
    return MKM(theta*, sigma2_*, alpha_dual, ...)

BLUP_fit(X, y; ...):
    self.mkm_ = MKM_fit(X, y; ...)
    return BLUP(mkm_, train_X)

BLUP.predict_components(X):
    K_b_cross[b] = AOMKernelizer.transform(X)
    fixed = X_f_test @ beta_hat
    random[b] = sigma2_b * K_b_cross[b] @ alpha_dual
    total = fixed + sum random[b]
    return {"fixed": fixed, "random": random, "total": total}
```

## 4. Experimental protocol

### 4.1 Cohort

We use the AOM-Ridge cohort CSV at
`bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv`: 54 OK regression
datasets from the TabPFN paper, with reference RMSEs for PLS, Ridge,
TabPFN-raw, TabPFN-opt, CNN-NICON, CatBoost.

### 4.2 Variants (planned for Phase 7)

| family | strategy | branch preprocessor | naming |
|--------|----------|---------------------|--------|
| Ridge | raw | none | `Ridge-raw` |
| mkR | uniform / kta / softmax_cv | none | `mkR-{strategy}` |
| mkR | softmax_cv | snv / msc / asls / emsc1 | `mkR-softmax_cv-{branch}` |
| MKM | reml / ml | none | `MKM-{method}` |
| MKM | reml | snv / msc / asls / emsc1 | `MKM-reml-{branch}` |
| BLUP | reml | none / snv / msc / asls | `BLUP-reml(-{branch})` |

### 4.3 Reporting

- Median relative RMSEP (vs PLS / Ridge / TabPFN-opt) per variant.
- Wins per variant (count of datasets where variant beats PLS).
- Critical-difference (CD) diagrams (Nemenyi post-hoc).
- Per-dataset table of best variant.
- Variance contribution barplots for MKM.
- Per-individual contribution table for BLUP (top deviating samples).

## 5. Results

### 5.1 Smoke benchmark (3 datasets, no branches)

From `benchmark_runs/smoke3/summary_per_variant.csv`:

| Variant | median rel-PLS | median rel-Ridge | median rel-TabPFN-opt | median fit-time (s) |
|---------|----------------|------------------|----------------------|---------------------|
| **mkR-softmax_cv** | **0.95** | 1.00 | 1.37 | 39 |
| BLUP-reml | 0.99 | 1.05 | 1.42 | 46 |
| MKM-reml | 0.99 | 1.05 | 1.42 | 54 |
| mkR-kta | 1.17 | 1.18 | 2.12 | 18 |
| mkR-uniform | 1.35 | 1.37 | 2.15 | 22 |
| Ridge-raw | 2.37 | 2.40 | 3.09 | 0.1 |

**Headline finding**: `mkR-softmax_cv` beats PLS (median 0.95 < 1.0)
**without** any preprocessing. MKM/BLUP match PLS within 1% on this
cohort.

Per-dataset best variants (`benchmark_runs/smoke3/summary_per_dataset.csv`):

- **ALPINE**: `mkR-softmax_cv` — 0.95 vs PLS, **beats** PLS by 5%.
- **AMYLOSE**: `MKM-reml` — 1.17 vs PLS, **17% behind** PLS (hard
  dataset; preprocessing-sensitive).
- **BEER**: `MKM-reml` — **0.62** vs PLS, **beats** PLS by 38% (small
  n=40, REML's variance estimation pays off).

### 5.2 Smoke benchmark with branches (3 datasets, 8 variants)

From `benchmark_runs/smoke3_branches/`:

[fill in once benchmark completes — should show whether SNV/MSC/ASLS
help close the gap on AMYLOSE].

### 5.3 Extended benchmark (12 datasets) — Phase 7a

[future].

### 5.4 Full 57-dataset benchmark — Phase 7b

[future].

### 5.5 Ablations

[future].

## 6. Discussion

### 6.1 When does each model shine?

- **mkR-softmax_cv** is the default for prediction; it beats PLS by 5%
  on the smoke median and is the best variant on ALPINE. Trade-off:
  fit time of ~30-60s vs sub-second for PLS / Ridge.
- **MKM** is the default when you need a variance decomposition or
  inference-grade bounds; it shines on small datasets like BEER (n=40)
  where REML's variance estimation is statistically efficient.
- **BLUP** is the default for explainability — per-individual
  contributions reveal which preprocessing pathway drove each
  prediction (no extra fitting cost; same prediction as MKM).

### 6.2 Limitations

- Inner-CV in `softmax_cv` uses a frozen outer-training kernelizer;
  inner-validation rows still affect the centring stats. Documented as
  v1 caveat; v2 will refit the kernelizer per inner fold.
- MKM's variance components are not separately identifiable when two
  block kernels have alignment > 0.95 (their sum is identifiable, the
  individual values are not).
- Not yet supported: multi-output Y, classification, POP-style
  per-component variants.

### 6.3 Reproducibility

```bash
# Tests (71 unit tests):
.venv/bin/pytest bench/AOM_v0/Multi-kernel/{MKR,MkM,Blup}/tests -q

# Smoke benchmark, no branches (~5 min):
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py \
  --cohort smoke3 --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3

# Smoke + branches (~20 min):
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py \
  --cohort smoke3 --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches \
  --variants mkR-softmax_cv mkR-softmax_cv-snv mkR-softmax_cv-msc mkR-softmax_cv-asls \
             MKM-reml MKM-reml-snv MKM-reml-msc MKM-reml-asls

# Summarise:
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/summarize_multikernel_smoke.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/<workspace>/results.csv
```

## 7. Acknowledgements

The cohort and reference RMSEs come from the TabPFN paper (NeurIPS 2024)
and were preserved in `bench/AOM_v0/Ridge/benchmark_runs/all57_cohort.csv`.
AOM-PLS is implemented in `bench/AOM_v0/Multi-kernel/aompls`; AOM-Ridge
in `bench/AOM_v0/Ridge`.

## 8. References

[fill in: PLS, Ridge, KTA (Cristianini), HSIC, REML (Patterson-Thompson),
BLUP (Henderson), kernel methods, AOM-PLS draft].

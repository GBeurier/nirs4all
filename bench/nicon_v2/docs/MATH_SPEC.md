# nicon_v2 — Mathematical Specification

This document defines the model classes, loss functions, regularisers, and uncertainty estimators used in `nicon_v2`. It is the contract that the Codex math reviewer checks against the implementation.

## 1. Notation

Let `X ∈ ℝ^{n × p}` be the spectra (n samples, p wavelengths) and `y ∈ ℝ^n` the target. We assume p is known per dataset (no padding across datasets); each model is built per-dataset with the proper input shape.

* Train fold indices `T ⊂ {1, …, n}`.
* Validation indices `V ⊂ T` (held-out for early stopping).
* Calibration indices `C ⊂ T \ V` (held-out for conformal calibration in Phase 5+).
* **Optimisation set** `A = T \ (V ∪ C)` — *all* training losses, fitted preprocessing statistics, augmentation statistics, learned EMSC reference, etc. are computed using only `A`.
* Test indices `S = {1, …, n} \ T`.

Disjointness invariants enforced by the implementation: `A`, `V`, `C`, `S` are mutually disjoint; `A ∪ V ∪ C = T`; `T ∪ S = {1, …, n}`. A spy operator (à la `bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py::SpyOperator`) verifies that no fitted statistic ever sees a non-`A` index.

## 2. Preprocessing layers

### 2.1 Standard Normal Variate (SNV)
For each spectrum row `xᵢ`,
```
SNV(xᵢ) = (xᵢ − mean(xᵢ)) / std(xᵢ).
```

### 2.2 Multiplicative Scatter Correction (MSC)
Let `r ∈ ℝ^p` be a reference spectrum (mean of train fold). Fit per-row scalars (a, b) by least squares to `xᵢ ≈ a + b·r`. Then `MSC(xᵢ) = (xᵢ − a) / b`.

### 2.3 Savitzky-Golay derivative (fixed)
Window length `w` (odd), polynomial order `m`, derivative order `d ∈ {0, 1, 2}`. The SG kernel `K_{w,m,d} ∈ ℝ^w` is computed once and applied as `xᵢ * K_{w,m,d}` (1-D convolution, mode='nearest'). The differentiable implementation is `Conv1d(in=1, out=1, kernel_size=w, padding=w//2, bias=False, weight=K)` with `weight.requires_grad=False`.

### 2.4 Concat-derivatives front
Output channels: `[ x, SG_{w_1, m, 1}(x), SG_{w_2, m, 2}(x) ]` stacked along the channel dimension (default `w_1 = 11, w_2 = 11, m = 2`). Optionally extend with `[ SNV(x), MSC(x) ]`.

### 2.5 Learnable EMSC layer (Phase 6)
Following Helin et al. 2022. Let `r̂ ∈ ℝ^p` be a learnable reference, and let `B ∈ ℝ^{p × k}` be a basis (Vandermonde polynomials of degree ≤ d, learnable coefficients). For each spectrum `xᵢ`:
```
xᵢ ≈ a + b · r̂ + B · cᵢ + εᵢ
```
where `(a, b, cᵢ)` are obtained by closed-form least squares on the Vandermonde + r̂ design matrix. The corrected output is `xᵢ_corr = xᵢ − (a + B·cᵢ) − (b − 1)·r̂`. The trainable parameters are `r̂` and the basis coefficients of `B`. A smoothness penalty `λ_S ‖∇²r̂‖² ` is added to the loss.

## 3. Backbone architecture (V1)

Block `i ∈ {1, …, K}`:
```
Conv1D(filters=Cᵢ, kernel=kᵢ, stride=1, padding='same')
LayerNorm  or  GroupNorm(groups=g)
Activation(GELU | ELU)
SpatialDropout1D(p_drop)
MaxPool1D(pool=2, stride=2)
```
Default: `K = 4`, kernels `(7, 5, 3, 3)`, channels `(16, 32, 64, 128)`. After block K:
```
GAP1D                                # global average pooling
Dropout(p_dense)                     # 0.2
Dense(num_classes for cls / 1 for reg, identity)
```

For regression the output is `μ̂(x) = Wᵀ · h(x) + b` where `h(x)` is the post-GAP feature vector. For classification the output is logits passed to `softmax` in the loss.

## 4. Loss functions

### 4.1 Regression
Squared error is computed on the y-processing-scaled scale (for numerical stability) and inverse-transformed at evaluation. The optimisation runs over `A`, **not** the full train fold:
```
L_reg = (1/|A|) Σᵢ∈A (ỹᵢ − μ̂ᵢ)² + λ_W · ‖θ‖²
```
where `ỹᵢ = (yᵢ − m_y)/s_y` is the y-scaled target with `(m_y, s_y) = (mean, std)` fitted on `A`, and `λ_W` is the AdamW weight decay (default `1e-4`).

For uncertainty (Phase 5+) we may train with Gaussian NLL on the **scaled** target:
```
L_nll = (1/|A|) Σᵢ∈A 0.5 · (((ỹᵢ − μ̂ᵢ) / σ̂ᵢ)² + 2 log σ̂ᵢ + log 2π)
```
where the network has a second output head producing `log σ̂` (clamped to `[log 1e-3, log 1e3]`). At evaluation we inverse-transform: `μ̂_orig = s_y · μ̂ + m_y`, `σ̂_orig = s_y · σ̂`. RMSEP, NLL and coverage are reported on the **original `y` scale** in the result CSV.

### 4.2 Classification
Cross-entropy over softmax logits. Optional label smoothing α=0.05 (small).

## 5. Augmentation operators

### 5.1 Bjerrum (offset / slope / multiplicative)
Per training mini-batch:
```
u ~ U[-σ_u, σ_u]
m ~ U[1 − σ_m, 1 + σ_m]
s ~ U[-σ_s, σ_s]
x' = u + m · x + s · w(λ)            # w(λ) = (λ − λ_0) / (λ_max − λ_0)
```
with default `(σ_u, σ_m, σ_s) = (0.05·range(x_train), 0.05, 0.05·range(x_train))`.

### 5.2 Mixup
Two indices `i, j` drawn uniformly. Sample `α ~ Beta(a, a)` (default `a = 0.2`).
```
x' = α · xᵢ + (1 − α) · xⱼ
y' = α · yᵢ + (1 − α) · yⱼ
```

### 5.3 C-Mixup (Yao 2022)
Sample `j` for each `i` proportional to a Gaussian kernel on `|yᵢ − yⱼ|`:
```
p(j | i) ∝ exp( − (yᵢ − yⱼ)² / (2 · σ_y²) )
```
The bandwidth `σ_y` is **chosen fold-locally** from a small grid (default `{0.05, 0.1, 0.2, 0.5, 1.0} · std(y_A)`) by inner CV inside `A`. We also test a robust variant `σ_y = 1.06 · std(y_A) · |A|^{-1/5}` (Silverman's rule). Ablations in Phase 3 compare:
* no mixup (control),
* vanilla mixup (uniform `j`),
* C-Mixup with fixed `σ_y = 0.5 · std(y_A)`,
* C-Mixup with fold-local grid-tuned `σ_y`.

## 6. Training

* Optimizer: **AdamW**, `lr = 1e-3`, `betas = (0.9, 0.999)`, `weight_decay = 1e-4`.
* Schedule: `OneCycleLR` over `n_epochs · n_train_batches` total steps; `pct_start = 0.1`.
* Early stopping on validation loss with patience 20.
* Batch size: `min(64, max(8, n_train // 8))`.
* Number of epochs: 200 (with early stopping).
* Mixed precision (fp16) on CUDA.
* Multiple seeds for ensembles (default `[0, 1, 2, 3, 4]`).

## 7. Cross-validation and aggregation

5-fold SPXYFold; predictions on each test fold are concatenated. Metrics reported per fold and as the median across folds (with IQR). When the dataset has predefined train/test splits (TabPFN paper convention) we honour the predefined split *and* run 5-fold CV inside the training set for early stopping.

## 8. Conformal calibration (Phase 5+)

Split-conformal procedure (Vovk et al. 2005; Lei et al. 2018). On the calibration set `C`,
```
qᵢ = | yᵢ − μ̂ᵢ | / max(σ̂ᵢ, ε)        (locally-residualised score, ε=1e-3)
k  = ⌈(|C|+1)(1−α)⌉
q̂_{1−α} = q_(k) (k-th order statistic of {qᵢ})
PI(x) = [μ̂(x) − q̂_{1−α} · σ̂(x), μ̂(x) + q̂_{1−α} · σ̂(x)]
```
Coverage `1−α = 0.9` by default.

**Finite-sample handling.** The marginal coverage guarantee `P(y ∈ PI(x)) ≥ 1−α` requires `k ≤ |C|`, i.e.
`⌈(|C|+1)(1−α)⌉ ≤ |C|`. When the calibration set is too small (e.g. `|C| < 10` for α = 0.1), the formula becomes invalid; the implementation flags `coverage_status = "calibration_too_small"`, sets `PI = (-∞, +∞)`, and reports `coverage_90 = NaN`. We require `|C| ≥ ⌈1/α⌉ + 5` (≥ 15 for α = 0.1) before reporting calibrated intervals.

`σ̂` is taken from the network's variance head (or set to a constant when only RMSE is trained). The residualised score scales the interval by the model's per-sample uncertainty, which gives narrower bands where the model is confident.

## 9. Deep ensembles (Phase 5)

`M = 5` independently-initialised networks (different seeds). Inference (Lakshminarayanan et al. 2017):
```
μ̂_ens(x)    = (1/M) Σ_m μ̂_m(x)                                                 # mean
aleatoric²(x) = (1/M) Σ_m σ̂_m²(x)                                                 # mean of per-net variances
epistemic²(x) = (1/M) Σ_m μ̂_m²(x) − μ̂_ens(x)²                                    # variance of means
σ̂_ens²(x)  = aleatoric²(x) + epistemic²(x)                                       # canonical total predictive variance
```
The result CSV stores `aleatoric_var` and `epistemic_var` separately so they can be reported in the manuscript. When the network is regression-only (no σ̂ head), `aleatoric²` is 0 and the ensemble variance reduces to the empirical variance of the M means.

## 10. Hyperparameter defaults table

| Parameter | Default |
|-----------|---------|
| Backbone kernels | (7, 5, 3, 3) |
| Backbone channels | (16, 32, 64, 128) |
| Norm | LayerNorm |
| Activation | GELU |
| `p_drop` (spatial) | 0.2 |
| `p_dense` | 0.2 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-4 |
| `batch_size` | min(64, max(8, n // 8)) |
| `epochs` | 200 (early-stop patience 20) |
| `seed_list` | [0, 1, 2, 3, 4] |
| `M` (ensemble) | 5 |
| `1 − α` (conformal) | 0.9 |
| `σ_y` (C-Mixup) | 0.5 · std(y_train) |
| Bjerrum amplitudes | 0.05 · range(x_train) for offset / slope; 0.05 mult |

# mkR — Mathematical Specification

## Notation

```text
X in R^{n x p}        spectra (training)
y in R^{n x q}        targets, q in {1, ...} (we focus on q = 1)
A_b in R^{p x p}      strict-linear operators (b = 1, ..., B)
H = I_n - (1/n) 1 1^T centring matrix (training-mean projector orthogonal complement)
```

Models centre `X` and `y`:

```text
Xc = X - x_mean          # x_mean is column-mean of training X
yc = y - y_mean
```

## Block Kernels

For each operator:

```text
Z_b = Xc A_b^T               # (n, p)
K_b_raw = Z_b Z_b^T           # (n, n) symmetric PSD
        = Xc (A_b^T A_b) Xc^T
```

Without further processing, raw block kernels can have wildly different
scales that mix into the weight estimation problem. mkR enforces two
standardisation steps:

### Centering

```text
K_b_c = H K_b_raw H
```

Centred kernels satisfy `K_b_c 1 = 0`, removing the constant (intercept)
direction. Equivalent to working on `Xc` (already centred) but applied at
the kernel level so the convention extends to nonlinear branch kernels.

### Trace normalisation

```text
tau_b = n / max(trace(K_b_c), eps)
K_b = tau_b * K_b_c          # tilde{K}_b in user spec
```

Now `trace(K_b)/n = 1`. Block scales `s_b` from existing AOM-Ridge are
**replaced** by trace scales `tau_b`, not multiplied by them.

**Zero-trace blocks**: if `trace(K_b_c) < eps_zero` (default
`eps_zero = 1e-12 * n`), the block carries essentially no signal;
``AOMKernelizer`` raises a clear error. Do **not** apply
`tau_b = n / eps`, which would amplify floating-point noise into a
spurious dominant kernel.

### Cross kernels (for prediction)

For new samples `X_*` (with `X_*c = X_* - x_mean_train`):

```text
K_b_raw_* = X_*c (A_b^T A_b) Xc^T   # (n_*, n)
```

Centring with **training-only** moments. Define:

```text
c_train = (1/n) 1_n^T K_b_raw          # (n,)       column / row mean of training kernel
nu_b    = (1/n^2) 1_n^T K_b_raw 1_n    # scalar     global mean of training kernel
r_*     = (1/n) K_b_raw_* 1_n          # (n_*,)     per-test-row mean of cross kernel
                                         # against training set
```

The double-centred cross kernel is:

```text
K_b_c_* = K_b_raw_* - 1_* c_train - r_* 1_n^T + nu_b 1_* 1_n^T
K_b_*   = tau_b * K_b_c_*
```

This matches the standard kernel-PCA "feature-centring at training mean"
construction. `c_train, nu_b, tau_b` are computed at `fit` time (training
data only). `r_*` is computed at `transform` time but is a deterministic
function of the test sample and the training set; this is **not** test-set
leakage.

**Batch invariance** (required test): `transform([x])[0]` must equal the
row of `transform([x, x2, ...])` corresponding to `x`, within fp tolerance.
A batched implementation that uses test-batch row means
(`H_test K H_train` with `H_test` defined on the test batch) would violate
this invariant.

## Combined Kernel

```text
K_eta = sum_b eta_b * K_b
K_eta_* = sum_b eta_b * K_b_*
```

with `eta_b >= 0`. We do **not** require `sum eta_b = 1` in general (this is
absorbed into `alpha`), but in practice we constrain to the simplex via
softmax to avoid identifiability issues with `alpha`.

## Dual Ridge

Solve:

```text
C = (K_eta + alpha I_n)^-1 yc
```

Predict:

```text
y_hat_c = K_eta_* C
y_hat = y_hat_c + y_mean
```

## Original-Space Coefficient (Strict-Linear, No Nonlinear Branch)

When all blocks are strict-linear and no branch preproc is used,
`coef_` exists in the original feature space. Substituting:

```text
K_b = tau_b * H Xc (A_b^T A_b) Xc^T H
    = tau_b * Xc (A_b^T A_b) Xc^T   (H acts as identity on already centred Xc)
```

We can pull weights inside:

```text
U_eta = sum_b eta_b tau_b A_b^T A_b Xc^T   # (p, n)
K_eta = Xc U_eta
beta = U_eta @ C                            # (p, q)
y_hat_c = X_*c @ beta
```

Equivalence test: `K_eta @ C == Xc @ beta` for any test row.

## Weight Strategies

### Uniform

```text
eta_b = 1 / B
```

### Manual

User supplies `eta_b >= 0`; we project onto the simplex:
`eta = max(eta, 0); eta /= sum(eta)`.

### KTA-simplex (closed-form, supervised)

```text
align_b = <K_b, yc yc^T>_F / (||K_b||_F * ||yc yc^T||_F)
```

with optional centring `align_b = HSIC(K_b, yy^T) / sqrt(HSIC(K_b, K_b) * HSIC(yy^T, yy^T))`
(equivalent if both kernels are already centred). Then:

```text
eta_b = max(align_b, 0) / sum_b max(align_b, 0)   # top_k mask if specified
```

This recovers the existing `mkl.learn_block_weights` algorithm but applied
to centred / trace-normalised kernels (so weights are comparable in scale).

### Softmax-CV (gradient, supervised, K-fold)

Parameterise:

```text
eta = softmax(theta) = exp(theta) / sum_b exp(theta_b)
```

so `eta` lives on the simplex unconstrained. Define inner CV loss:

```text
L(theta, alpha) = (1/n) sum_{i=1}^n (y_i - y_hat_i^{-fold(i)})^2
```

where `y_hat_i^{-fold(i)}` is the prediction of the model with weights
`eta(theta)` and ridge `alpha` trained on the fold not containing `i`.
Optimise `theta` by L-BFGS-B with finite-difference gradient,
optionally jointly with `alpha` (or alternating). Multi-start with `n_restarts=3`
random `theta_0` from `Dirichlet(1, ...)` and pick lowest `L`.

To stabilise, regularise toward uniform:

```text
L_reg(theta, alpha) = L(theta, alpha) + lambda_eta * KL(softmax(theta) || uniform)
```

with `lambda_eta` small (default `1e-3`).

## Block Diagnostics

Pairwise alignment:

```text
A_ij = <K_i, K_j>_F / (||K_i||_F * ||K_j||_F)
```

If `A_ij > 0.95` for `i != j`, weights `eta_i, eta_j` are weakly identifiable
individually (the sum `eta_i + eta_j` is identifiable). mkR diagnostics
report `max_i!=j A_ij` and the corresponding pair.

Per-fold weight stability:

```text
eta_bar_b = mean_f eta_{b,f}
sigma_b   = std_f eta_{b,f}
stability_b = 1 - sigma_b / (eta_bar_b + eps)
```

## Equivalence Claims

1. **Uniform-trace mkR** (centred + trace-normalised kernels with
   `eta_b = 1/B`) is **proportional** to AOM-Ridge `block_scaling="rms"`
   superblock for strict-linear blocks:

   ```text
   K_super_rms = sum_b s_b^2 K_b_raw
   K_uniform_trace = (1/B) sum_b tau_b K_b_c   (centred + trace-normalised)
   ```

   The proportionality constant comes from the relation between `s_b^2`
   (RMS scale) and `tau_b` (trace scale): `s_b^2 = 1/(RMS(X_b)^2 + eps)`
   and `tau_b = n / tr(K_b_c)`. For centred blocks where the mean
   subtracted by `H` is small (`x_mean ~ 0` when X is already centred),
   the relation simplifies to:

   ```text
   K_super_rms ≈ p * sum_b tau_b K_b_raw = p * B * K_uniform_trace
   ```

   So uniform-trace mkR ≡ AOM-Ridge `"rms"` superblock when alpha is
   rescaled by `p * B`. **The previous claim (with `block_scaling="none"`)
   is wrong unless there is a single block or all block traces are equal.**

   Test target: equivalence vs RMS superblock (rescaled alpha), not "none".

2. **KTA mkR** with centred + trace-normalised kernels does **not**
   reproduce the existing `selection="mkl"` (which uses raw kernels with
   `scales=ones`, see `estimators.py:719`); the alignment scores differ
   because trace normalisation changes `||K_b||_F`. The two are sister
   variants, not identical.

3. **Softmax-CV mkR** is a strict generalisation of uniform: setting
   `lambda_eta -> +inf` collapses to uniform; on noiseless oracle data
   with sufficient `n`, it converges to the oracle simplex weights
   (within `~ 0.1` L1 distance for `n >= 200, B = 4`).

## Simplex Convention

mkR weights are **always projected onto the simplex** (`eta_b >= 0,
sum_b eta_b = 1`). Manual weights with negative entries are clipped;
all-zero manual weights raise. The constraint `sum eta_b = 1` is absorbed
into `alpha`: scaling `eta` by a constant `c > 0` is equivalent to scaling
`alpha` by `1/c`. Reporting simplex `eta` removes this degeneracy.

## Sanity Tests

- `K_b @ 1 == 0` (centring).
- `trace(K_b) / n == 1` (trace normalisation).
- `K_eta == sum_b eta_b * K_b` (linearity).
- Cross-kernel symmetry: `K_test_train(X_test1, X_test2)` symmetric to
  `K_test_train(X_test2, X_test1)` modulo transpose.
- `predict(X_train) == X_train @ coef + intercept` (primal/dual agreement).
- One-block uniform mkR equals Ridge on `Xc A_b^T` (after the trace scaling
  of `alpha`).

## Numerical Conventions

- Cholesky for `(K_eta + alpha I)`, with adaptive jitter (existing
  `_cholesky_solve_with_jitter`).
- Eigendecomposition path for alpha-grid sweeps (existing
  `solve_dual_ridge_path_eigh`).
- `eps = 1e-12` for trace normalisation denominator.
- Block alignment matrix uses `||K||_F` (no jitter; centred kernels are PSD
  with rank `<= n - 1`).

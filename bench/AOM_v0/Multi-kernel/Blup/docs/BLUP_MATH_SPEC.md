# BLUP / E-BLUP — Mathematical Specification

## Notation

Same as MKM_MATH_SPEC.md. Recall the model:

```text
y = X_f beta + sum_{b=1}^B u_b + e,    u_b ~ N(0, sigma_b^2 K_b),    e ~ N(0, sigma_e^2 I)
```

Marginal:

```text
y ~ N(X_f beta, V),    V = sum_b sigma_b^2 K_b + sigma_e^2 I_n
```

We assume MKM has provided estimates `(hat sigma_b^2)_b, hat sigma_e^2,
hat beta`. We use these in BLUP formulae below; this is technically
**E-BLUP** (Empirical BLUP).

## BLUP of Random Effects

### Henderson's mixed-model equations (training)

For known variances:

```text
hat u_b = sigma_b^2 K_b V^-1 (y - X_f beta)
```

with `beta` plugged in as `hat beta` (GLS estimator from MKM):

```text
hat beta = (X_f^T V^-1 X_f)^-1 X_f^T V^-1 y
```

Define the dual coefficient:

```text
alpha_dual = V^-1 (y - X_f hat beta)        # (n,)
```

Then on training:

```text
hat u_b      = sigma_b^2 K_b alpha_dual    # (n_train,) per block b
hat y_train  = X_f hat beta + sum_b hat u_b
```

### Cross-prediction on test individuals

For new samples with kernel `K_b_*` (cross-kernel from test to train,
shape `(n_test, n_train)`) and fixed-effect design `X_*f`:

```text
hat u_b_*   = sigma_b^2 K_b_* alpha_dual    # (n_test,)
hat y_*     = X_*f hat beta + sum_b hat u_b_*
```

In words: each block predicts its part of the response by projecting the
test sample's similarity to the training set (`K_b_*`) onto the dual
coefficient `alpha_dual`, scaled by its variance.

## Decomposition Identity

By construction:

```text
hat y_* = X_*f hat beta + sum_b sigma_b^2 K_b_* alpha_dual
```

This can be regrouped as:

```text
hat y_* = sum_{b=0}^B contribution_b(x_*)
```

with `contribution_0 = X_*f hat beta` (fixed effects) and `contribution_b =
sigma_b^2 K_b_* alpha_dual` for `b = 1, ..., B`.

The decomposition is **linear in the test sample**: each component is the
projection of `x_*` (through the kernel `K_b_*`) onto a fixed direction
(`alpha_dual`), so each contribution is a linear function of `x_*` (when
`K_b` is strict-linear).

### Sanity Test

```text
sum over b of contribution_b(X) == predict(X)     up to floating-point tolerance
```

This must hold for **any** `X`, including training data, test data, and
random unseen data.

## Relationship to Ridge / mkR

For fixed variances, BLUP prediction equals mkR with:

```text
eta_b = sigma_b^2,    alpha = sigma_e^2
```

So BLUP prediction differs from mkR only in **how `(eta, alpha)` are
chosen** (by REML in BLUP, by inner-CV in mkR's `softmax_cv`). The
**decomposition into per-block contributions** is unique to BLUP.

## Shrinkage Diagnostic

Each `hat u_b` is shrunk toward zero by a factor proportional to the
"signal-to-residual" ratio for block `b`. A useful scalar shrinkage
measure:

```text
shrink_b = sigma_b^2 lambda_max(K_b) / (sigma_b^2 lambda_max(K_b) + sigma_e^2)
```

Values close to 1 mean little shrinkage (high block effect), close to 0
mean strong shrinkage (block effectively zeroed).

A more accurate per-individual diagnostic uses `sigma_b^2 K_b V^-1`
restricted to the leading eigenvectors of `K_b`. Defer to Round 2.

## Prediction Variance (Optional)

For test individual `x_*`:

```text
Var(hat y_*) = x_*^T (X_f^T V^-1 X_f)^-1 x_*               # fixed-effect
             + sum_b sigma_b^2 [k_b(x_*, x_*) - k_b_*^T V^-1 k_b_*]   # block b conditional variance
```

where `k_b(x_*, x_*) = sigma_b^2 K_b(x_*, x_*)` (training-side trace
normalisation extends to test). This is approximate (assumes variance
components are known); for E-BLUP an exact correction would inflate the
variance to account for variance-component estimation error (Kackar-Harville
correction). We document the approximation but don't ship the correction
in Round 1.

## Numerical Recipe (Prediction)

Given fitted MKM:

1. `K_b_*` for each `b`: cross kernel using stored training statistics.
2. `dual_alpha = V^-1 (y - X_f hat beta)` (precomputed during fit).
3. `contribution_b = (hat sigma_b^2) * (K_b_* @ dual_alpha)`.
4. `contribution_fixed = X_*f @ hat beta`.
5. `hat y_* = contribution_fixed + sum_b contribution_b`.

All operations are `O(n_test n_train)` per block; no test-time matrix
factorisation needed.

## Edge Cases

- **Block at boundary** (`hat sigma_b^2` near 0): `hat u_b ~ 0`, the
  contribution is negligible. Reported in diagnostics.
- **Highly correlated blocks** (`align(K_i, K_j) > 0.95`): individual
  contributions `hat u_i, hat u_j` are not separately identifiable but
  their sum is. Diagnostic flag.
- **Singular fixed-effect design** (`rank(X_f) < r`): use Moore-Penrose
  pseudoinverse for `M = X_f^T V^-1 X_f`. Default `X_f = ones(n, 1)` so
  this doesn't happen in practice.

## Sanity Tests (Synthetic R1, R2, R3)

- **R1 (1 active block out of 4)**: `corr(hat u_active, u_active_truth) > 0.8`,
  `||hat u_inactive||_2 / ||hat u_active||_2 < 0.2`.
- **R2 (3 active blocks)**: per-block correlation with truth `> 0.6` per
  active block.
- **R3 (correlated blocks)**: sum of correlated-pair contributions is
  recovered (`> 0.8` correlation with sum of truth), individual
  contributions may not be.
- **Decomposition sum**: `sum components == predict` to `< 1e-10` absolute
  tolerance for `n < 1000`.

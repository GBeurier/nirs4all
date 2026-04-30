# MKM — Mathematical Specification

## Notation

```text
n        number of training samples
p        number of features (spectral channels)
B        number of AOM operator blocks
X        in R^{n x p}     spectra
y        in R^n           targets (q = 1; multi-trait deferred)
X_f      in R^{n x r}     fixed-effect design (default = ones intercept)
A_b      in R^{p x p}     strict-linear AOM operator b
K_b      in R^{n x n}     centred + trace-normalised block kernel (see MKR_MATH_SPEC.md)
```

## Model

```text
y = X_f beta + sum_{b=1}^B u_b + e
u_b ~ N(0, sigma_b^2 K_b)             (independent across b)
e   ~ N(0, sigma_e^2 I_n)
```

Marginal:

```text
y ~ N(X_f beta, V),    V = sum_{b=1}^B sigma_b^2 K_b + sigma_e^2 I_n
```

We parametrise variances on the log scale:

```text
theta_b = log sigma_b^2,    theta_e = log sigma_e^2
theta = (theta_1, ..., theta_B, theta_e) in R^{B+1}
```

so that `sigma_b^2 = exp(theta_b) > 0` is enforced.

## Likelihood

### ML

For known `beta`:

```text
ell_ML(theta, beta) = -0.5 [ logdet V + (y - X_f beta)^T V^-1 (y - X_f beta) + n log 2*pi ]
```

Profiling out `beta` via GLS:

```text
beta_hat(theta) = (X_f^T V^-1 X_f)^-1 X_f^T V^-1 y
r = y - X_f beta_hat
ell_ML*(theta) = -0.5 [ logdet V + r^T V^-1 r + n log 2*pi ]
```

### REML

```text
ell_REML(theta) = -0.5 [
    logdet V
  + logdet(X_f^T V^-1 X_f)
  + r^T V^-1 r
  + (n - r) log 2*pi
]
```

REML is preferred when `r > 0` because it reduces bias in variance
estimates (corrects for fixed-effect degrees of freedom).

## Gradient (Analytic)

Define the projection matrix:

```text
P = V^-1 - V^-1 X_f (X_f^T V^-1 X_f)^-1 X_f^T V^-1
```

Useful identities:
- `P y = V^-1 r`.
- `P V P = P`.
- `P X_f = 0`.

For component `theta_j`:

```text
dV/dtheta_b = sigma_b^2 K_b      (b = 1, ..., B)
dV/dtheta_e = sigma_e^2 I_n
```

Gradient of negative log-REML:

```text
- d ell_REML / d theta_j  = 0.5 [ tr(P dV/dtheta_j) - r^T P dV/dtheta_j P r ]
                          = 0.5 [ tr(P dV/dtheta_j) - (P r)^T (dV/dtheta_j)(P r) ]
```

For ML, replace `P` by `V^-1` and drop the `logdet(X_f^T V^-1 X_f)` term;
the gradient becomes:

```text
- d ell_ML* / d theta_j  = 0.5 [ tr(V^-1 dV/dtheta_j) - r^T V^-1 dV/dtheta_j V^-1 r ]
                         (with r = y - X_f beta_hat(theta))
```

## Numerical Recipe (per evaluation)

Given `theta`:

1. Build `V = sum_b exp(theta_b) K_b + exp(theta_e) I_n + jitter * I_n`.
2. Cholesky `V = L L^T` (one factorisation; bail out with progressively
   larger jitter on failure, mirroring AOM-Ridge `_cholesky_solve_with_jitter`).
3. `V^-1 X_f` = `cho_solve(L, X_f)`.
4. `M = X_f^T V^-1 X_f`. Cholesky `M = L_M L_M^T`.
5. `beta_hat = cho_solve(L_M, X_f^T V^-1 y)`.
6. `r = y - X_f beta_hat`.
7. `V^-1 r = cho_solve(L, r)`.
8. `logdet V = 2 * sum log diag(L)`.
9. `logdet M = 2 * sum log diag(L_M)`.
10. `ell_REML = -0.5 (logdet V + logdet M + r^T V^-1 r + (n - r) log 2*pi)`.

For gradient:
- For each block `b`, compute `tr(P K_b) = tr(V^-1 K_b) - tr((V^-1 X_f)^T K_b (V^-1 X_f) M^-1)`.
  In practice, form `Q = V^-1 X_f`, `Q_M = Q M^-1`, then
  `tr(P K_b) = sum(diag(V^-1 K_b)) - sum(K_b * (Q_M Q^T))` (one
  `(n, n)` Hadamard sum).
- `r^T P K_b P r = (V^-1 r)^T K_b (V^-1 r) - (V^-1 r)^T X_f M^-1 (V^-1 X_f)^T K_b (V^-1 r) - sym`
  — but with `P r = V^-1 r` (since `r perp X_f V^-1`), simplifies to
  `(P r)^T K_b (P r)` directly, where we precompute `P r = V^-1 r` (REML
  property: `P r = V^-1 r` because `r` is the GLS residual and is in the
  null space of `X_f^T V^-1`).
- For `theta_e`: `dV/dtheta_e = sigma_e^2 I_n`, so `tr(P) = tr(V^-1) - tr((V^-1 X_f) M^-1 (V^-1 X_f)^T)`.

## Multi-Start Optimisation

Starting points:

```text
theta_0_b = log( var(y) / (B + 1) )   for b = 1, ..., B
theta_0_e = log( var(y) / (B + 1) )
```

Plus `n_restarts - 1` random starts:

```text
theta_0 = base_theta + N(0, sigma_init^2 I)
sigma_init = 1.0
```

Each restart calls L-BFGS-B with bounds `(-15, 15)` per component. Pick
best by lowest `-ell_REML`. Report mean / std of converged solutions to
flag multimodality.

## Boundary Diagnostics

If `theta_b - lower_bound < 0.5` we flag `block b` as **at boundary** —
its variance is essentially zero. This is a legitimate REML solution
(some blocks contribute nothing) but should be reported clearly.

## Identifiability Diagnostic

For each pair `(i, j)`:

```text
align_ij = <K_i, K_j>_F / (||K_i||_F * ||K_j||_F)
```

If `align_ij > 0.95`, MKM cannot reliably split `sigma_i^2` and `sigma_j^2`.
Report the worst pair and recommend either pruning or aggregation.

## Relative Contribution

Once converged:

```text
total_var = sum_b sigma_b^2 + sigma_e^2
h_b      = sigma_b^2 / total_var
h_e      = sigma_e^2 / total_var
```

`h_b` is interpretable as a **relative variance contribution** and is the
primary scientific output of MKM. Note this is not a heritability in the
genetic sense — but if `K_b` are normalised genetic relationship matrices,
it would be one.

## Equivalence with mkR (at fixed theta)

For fixed `(sigma_b^2)_b, sigma_e^2`, the MKM mean prediction equals an
mkR prediction with:

```text
eta_b = sigma_b^2,   alpha = sigma_e^2
```

So MKM ≡ mkR with **likelihood-derived weights** (instead of CV-derived
weights).

Test: with fixed `theta`, MKM `predict(X_test)` matches mkR `predict(X_test)`
to floating-point tolerance.

## Sanity Tests (Synthetic)

- **R1 (1 active block out of 4)**: REML recovers `h_active in [0.5, 1.0]`,
  `h_inactive < 0.1` each. Random-restart consensus stable.
- **R2 (3 active blocks)**: relative ordering of `(h_b)` matches truth.
- **R3 (2 quasi-identical blocks)**: sum `h_1 + h_2` matches truth, but
  individual `h_b` may not. Boundary diagnostic flags one of them.
- **At-fixed-theta**: MKM prediction equals mkR prediction (numerical).
- **Single kernel**: REML on `K_1` alone matches Ridge with `alpha = sigma_e^2 / sigma_1^2`.

## Numerical Conventions

- All log-scale variables in `(log sigma^2, ...)`.
- Bounds `[-15, 15]` (so `sigma^2 in [3e-7, 3e6]`).
- Jitter: start at `1e-8 * trace(V) / n`, escalate by factor 10 on failure.
- Convergence: `||grad||_inf < 1e-5` or `|delta theta|_inf < 1e-7`.
- Maximum iterations: `200` per restart.

## Stability Considerations

When two kernels are highly aligned, the Hessian becomes near-singular.
L-BFGS-B handles this gracefully but the converged `theta` may sit on a
ridge of equal-likelihood values. The diagnostic `align_ij > 0.95` and the
multi-restart variance flag this for the user.

If `sum_b sigma_b^2 K_b` is near-rank-deficient relative to `sigma_e^2 I`,
likelihood becomes flat and convergence is slow; the multi-start protocol
mitigates this.

# AOM-Ridge Mathematical Specification

## Matrix Conventions

```text
X in R^{n x p}
Y in R^{n x q}
A_b in R^{p x p}
X_b = X A_b^T
```

Models center `X` and `Y` during fit:

```text
Xc = X - x_mean
Yc = Y - y_mean
Y_hat = (X - x_mean) beta + y_mean
```

## Single-Operator Ridge

For one strict linear operator:

```text
Z_b = Xc A_b^T
K_b = Z_b Z_b^T = Xc A_b^T A_b Xc^T
C_b = (K_b + alpha I_n)^-1 Yc
beta_b = A_b^T A_b Xc^T C_b
```

Prediction:

```text
Y_hat_c = X_*c beta_b
Y_hat_c = K_*b C_b
K_*b = X_*c A_b^T A_b Xc^T
```

These two forms must be tested for equality.

## Superblock AOM-Ridge

Define weighted blocks:

```text
Z_b = s_b Xc A_b^T
Phi(Xc) = [Z_1 | Z_2 | ... | Z_B]
```

The explicit problem is:

```text
min_gamma ||Yc - Phi(Xc) gamma||_F^2 + alpha ||gamma||_F^2
```

The dual kernel is:

```text
K_super = Phi(Xc) Phi(Xc)^T
        = sum_b s_b^2 Xc A_b^T A_b Xc^T
```

Define:

```text
M = sum_b s_b^2 A_b^T A_b
U = M Xc^T
K = Xc U
C = (K + alpha I_n)^-1 Yc
beta = U C = M Xc^T C
```

The estimator must expose `coef_ = beta` with shape `(p, q)`, never a wide
superblock coefficient with shape `(B p, q)`.

## Block Scaling

Default:

```text
block_scaling="rms"
s_b = 1 / (RMS(Xc A_b^T) + eps)
RMS(Z) = ||Z||_F / sqrt(n p)
```

Efficient computation:

```text
AXt = A_b Xc^T
RMS(Xc A_b^T) = ||AXt||_F / sqrt(n p)
```

Scaling must be fitted only on training data. In CV, every fold has its own
`x_mean_f`, `y_mean_f`, and `s_b,f`.

## Alpha Grid

For a train kernel:

```text
base = max(trace(K) / n, eps)
alpha_j = base * 10^t_j
t_j in [-6, 6]
```

Default `n_grid=50`.

## Global Hard AOM-Ridge

`selection="global"` selects:

```text
(b*, alpha*) = argmin CVRMSE(b, alpha)
```

Every candidate must be evaluated with fold-local centering, operator fitting,
block scaling, and kernels.

## Active Superblock

For large banks:

```text
large strict-linear bank
  -> fold-local normalized screening
  -> diversity pruning
  -> top_m active operators
  -> superblock dual Ridge
```

Fast screening:

```text
S = Xc^T Yc
R_b = s_b A_b S
score_b = ||R_b||_F^2
```

Optional alignment diagnostic:

```text
align_b = <K_b, Yc Yc^T>_F / (||K_b||_F ||Yc Yc^T||_F)
```

## Nonlinear Or Fitted Branches

SNV, MSC, EMSC, OSC, ASLS, and other data-dependent preprocessors are branch
transformers, not strict fixed `A_b` operators in phase 1.

For branch Ridge:

```text
Z_b = T_b(X)
K = sum_b s_b^2 Z_b Z_b^T
```

Every branch parameter must be fold-local. Original-space `coef_` is not
available unless the branch exposes a fixed linear adjoint.


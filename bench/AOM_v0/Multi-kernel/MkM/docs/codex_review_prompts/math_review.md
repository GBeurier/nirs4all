# Codex Math Review: MKM (Multi-Kernel Mixed Model)

Review mathematical correctness of the MKM implementation.

Read:

```text
bench/aom_v0/Multi-kernel/MkM/docs/MKM_MATH_SPEC.md
bench/aom_v0/Multi-kernel/MkM/docs/IMPLEMENTATION_PLAN.md
bench/aom_v0/Multi-kernel/MkM/mkm/kernelizer.py
bench/aom_v0/Multi-kernel/MkM/mkm/likelihood.py
bench/aom_v0/Multi-kernel/MkM/tests/test_likelihood.py
bench/aom_v0/Multi-kernel/MkM/tests/test_single_kernel_reml.py
bench/aom_v0/Multi-kernel/MkM/tests/test_multi_kernel_reml.py
```

Verify the central formulae:

```text
V(theta)        = sum_b exp(theta_b) K_b + exp(theta_e) I_n + jitter * I_n
beta_hat        = (X_f^T V^-1 X_f)^-1 X_f^T V^-1 y
r               = y - X_f beta_hat
ell_REML(theta) = -0.5 [ logdet V + logdet(X_f^T V^-1 X_f) + r^T V^-1 r + (n - r) log 2*pi ]
```

Check:

1. K_b are centred (`K_b @ 1 == 0`) and trace-normalised (`tr(K_b)/n == 1`).
2. Gradient formula uses `P = V^-1 - V^-1 X_f (X_f^T V^-1 X_f)^-1 X_f^T V^-1`
   correctly. For each component:
   `dell/dtheta_j = 0.5 [ -tr(P dV/dtheta_j) + (P r)^T dV/dtheta_j (P r) ]`
   (sign convention for **negative** REML).
3. `dV/dtheta_b = sigma_b^2 K_b`, `dV/dtheta_e = sigma_e^2 I` (chain rule from
   log-variance parametrisation).
4. Cholesky of `V` is reused for `V^-1 X_f`, `V^-1 y`, `V^-1 r`.
5. ML and REML differ only in `logdet(X_f^T V^-1 X_f)` term and
   normalisation `(n vs n - r)`.
6. Bounds `theta in [-15, 15]` keep `sigma^2 in [3e-7, 3e6]`.
7. Multi-start protocol properly reports best run, all endpoints, and
   detects boundary solutions (`theta_b - lower_bound < 0.5`).
8. Single-kernel limit: REML reduces to known closed form for `K + sigma_e^2 I`.

Return findings ordered by severity with file and line references. If a
formula is wrong, give the corrected formula and a minimal test that would
catch it.

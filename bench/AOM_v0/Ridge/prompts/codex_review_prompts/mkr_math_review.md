# Codex Math Review: mkR

Read:

```text
bench/AOM_v0/Ridge/docs/MKR_MATH_SPEC.md
bench/AOM_v0/Ridge/aomridge/kernelizer.py
bench/AOM_v0/Ridge/aomridge/weights.py
bench/AOM_v0/Ridge/tests/test_mkr_kernelizer.py
bench/AOM_v0/Ridge/tests/test_mkr_weights.py
bench/AOM_v0/Ridge/tests/test_mkr_equivalences.py
```

Verify the central formulae:

```text
K_b_raw  = X (A_b^T A_b) X^T
H        = I - (1/n) 1 1^T
K_b_c    = H K_b_raw H
tau_b    = n / max(trace(K_b_c), eps)
K_b      = tau_b K_b_c                          # symmetric, centred, normalised
K_eta    = sum_b eta_b K_b
beta     = U_eta @ (K_eta + alpha I)^-1 yc      # original-space coefficient
U_eta    = sum_b eta_b tau_b A_b^T A_b X^T      # for strict-linear blocks
```

For cross-kernels:

```text
K_b_raw_*  = X_*c (A_b^T A_b) Xc^T
mu_b       = (1/n) K_b_raw 1                   # row mean of training kernel
nu_b       = (1/n^2) 1^T K_b_raw 1             # global mean
K_b_c_*    = K_b_raw_* - mu_b^T(broadcast) - row_mean(K_b_raw_*) + nu_b
K_b_*      = tau_b K_b_c_*
```

Check:

1. Centring uses **training-side moments only**. Test or production code
   that recomputes `mu_b, nu_b` from test data is a bug.
2. Trace normalisation: `trace(K_b_c) / n == 1` after normalisation. Also
   `tr(K_b) = n`, *not* 1 (some texts use `tr/n = 1`, others `tr = 1`;
   spec says `tr/n = 1`).
3. softmax-CV gradient (if implemented analytically): chain-rule through
   `eta = softmax(theta)` is `d eta_b / d theta_j = eta_b (delta_bj - eta_j)`.
4. `coef_` is original-space when no nonlinear branch is used.
5. `predict(X) == K_eta_* @ C` matches `X_c @ coef_ + intercept` to fp tol.
6. Identifiability warning when `align(K_i, K_j) > 0.95`.

Return findings first, ordered by severity.

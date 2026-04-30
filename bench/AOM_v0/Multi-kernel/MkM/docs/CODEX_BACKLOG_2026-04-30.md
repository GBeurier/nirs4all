# MkM Codex Backlog — Round 2 (2026-04-30)

Source: `/tmp/codex_mkm_math.md` (Codex math/code review of likelihood,
optimisation, estimator, and tests).

## Verified OK

- For full-rank fixed effects, REML objective and analytic gradient match
  the spec.
- Single Cholesky factorisation of `V` is reused (correctly) for `S`, `Q`,
  `M^-1`, and `P`.
- E-BLUP mean prediction formula is correct.
- Kernelizer transform uses training-side stats and ignores `y` (no target
  leakage).
- `pytest -q tests/` → 12 passed.

## High-severity findings (applied)

### H1 — Rank-deficient fixed effects could silently produce wrong likelihood

**Spec**: `p_f = rank(X_f)`, then use `(n - p_f) log 2π` and
`logdet(X_f^T V^-1 X_f)`.

**Bug**: `compute_neg_log_reml` uses `X_f.shape[1]` and Cholesky-factorises
`M` directly. A collinear `X_f` produces a Cholesky failure (good) **or**
a finite but bogus likelihood with wrong degrees of freedom (bad).

**Fix applied**: added a defensive `_rank_of(X_f)` SVD-based check at the
top of `compute_neg_log_reml` that raises a clear error if rank < column
count. The estimator's fit path already runs `fit_fixed_effects(X_f)` to
project to leading SVD directions; this guard catches direct callers
(tests / benchmarks / users) that pass raw `X_f`.

Files: `mkm/likelihood.py:212`, `mkm/likelihood.py:255` (new `_rank_of`).

## Medium-severity backlog (deferred)

### M1 — Boundary detection: projected-gradient sign

**Issue**: For minimisation with a lower bound, an active lower-bound
solution has `grad >= 0`. Current code uses `max(grad[j], 0.0)` which
keeps positive gradients (correct for minimisation) but the docstring
claims "outward direction"; clarify or invert sign for a bound-from-above
test (none in current spec).

**Defer**: behaviour is correct for our lower-bound-only setup; only the
docstring is misleading.

### M2 — Boundary diagnostics use REML gradient even in ML mode

When `method="ml"`, the boundary diagnostic still calls
`compute_neg_log_reml_grad` (which uses `P`). This doesn't affect the ML
objective value but makes ML boundary reports unreliable. Action: branch
the gradient computation on `method` and use `V^-1` for ML.

**Defer**: REML is the default; ML mode is a comparison baseline.

## Test additions

The defensive rank check has no new dedicated test yet (covered indirectly
by the existing `fit_fixed_effects` rank tests). Add a unit test that
calls `compute_neg_log_reml` with a rank-deficient `X_f` and asserts the
clear error message.

**Tracking**: `tests/test_likelihood.py` to add `test_rank_deficient_xf_raises`.

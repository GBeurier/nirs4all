# MKM Plan Review Corrections (Codex roadmap review, 2026-04-30)

Source: `/tmp/codex_mkm_roadmap.md`.

## High-Severity Findings

### H1. Symbol overload `r`

**Issue**: `MKM_MATH_SPEC.md:11` uses `r` for fixed-effect dimension;
`:53` reuses `r` for the residual vector; REML formula uses `(n-r)`.
Standard convention requires `(n - p_f)` where `p_f = rank(X_f)`.

**Fix**:
- Rename residual to `resid`.
- Define `p_f = rank(X_f)`.
- Validate / drop rank-deficient fixed-effect columns.
- Use `(n - p_f) log 2*pi` and `logdet(X_f^T V^-1 X_f)` only after rank check.

### H2. Fixed-theta equivalence with mkR overclaimed

**Issue**: `MKM_MATH_SPEC.md:186` says MkM equals mkR with
`eta_b = sigma_b^2, alpha = sigma_e^2`. But mkR uses simplex-normalised
`eta`, and the trace-normalisation factor on each kernel changes the
denominator.

**Fix**: define two equivalence statements:
- **Absolute-eta path** (mkR with `weight_strategy="manual"` and no
  simplex projection): `eta_b = sigma_b^2`, `alpha = sigma_e^2`.
- **Simplex path** (mkR default): with `s = sum_b sigma_b^2`,
  `eta_b = sigma_b^2 / s`, `alpha = sigma_e^2 / s`.

Tests must use one of these explicitly.

### H3. Single-kernel Ridge equivalence: trace factor in wrong place

**Issue**: `TEST_PLAN.md:9` says raw Ridge should use
`alpha = sigma_e^2 / sigma_g^2 * tau`. If `K_norm = tau * Z Z^T`, the
correct equivalent on raw Z is `alpha = sigma_e^2 / (sigma_g^2 * tau)`.

**Fix**: state three cases explicitly:
- Normalised kernel `alpha = sigma_e^2 / sigma_g^2`.
- Raw feature/kernel `alpha = sigma_e^2 / (sigma_g^2 * tau)`.
- Absolute-eta mkR `alpha = sigma_e^2` when `eta = sigma_g^2`.

### H4. Gradient form mixes notation

**Issue**: Spec mixes `r^T P dotV P r` and `r^T V^-1 dotV V^-1 r` forms.
Equivalent only when `r = resid` (GLS residual).

**Fix**: define once and use uniformly:

```text
a = P y = V^-1 resid
g_j = 0.5 * (tr(P dotV_j) - a^T dotV_j a)
dotV_j = exp(theta_j) K_j   (b = 1, ..., B)
dotV_e = exp(theta_e) I_n
```

### H5. Trace computation underspecified

**Issue**: `tr(V^-1 K_b)` per block per evaluation is too slow for
100-op bank.

**Fix**: per evaluation:
1. Cholesky of `V` → `L`.
2. `Q = V^-1 X_f = cho_solve(L, X_f)`.
3. `M = X_f^T Q`; `L_M = chol(M)`.
4. `Q_M = Q M^-1` (one cho_solve).
5. `S = V^-1 = cho_solve(L, I_n)` once.
6. `P = S - Q M^-1 Q^T`.
7. For each block: `tr(P dotV_j) = sum(P * dotV_j)` (Hadamard sum,
   one `(n, n)` operation). For `theta_e`: `tr(P) = sum(diag(P))`.

If 100-op bank is too slow, insert an alignment-pruning phase before
estimator benchmarks.

## Medium-Severity Findings

### M1. Bounds and boundary detection scale-dependent

- Bounds `theta in [-15, 15]` are absolute; boundary detection is
  inconsistent across docs (`at bound` vs `within 0.5`).
- **Fix**: standardise `y` internally so `var(y) ~ 1`, OR define bounds
  relative to `var(y)`. Boundary KKT detection: project gradient on the
  active set; flag as boundary if projected gradient near zero AND
  `theta_b` near `lower_bound + tol` AND relative contribution `< eps`.

### M2. Multi-restart strategy too weak

- `IMPLEMENTATION_PLAN.md` starts from `N(0, I)`; `MATH_SPEC.md` from
  `log(var(y)/(B+1))`.
- **Fix**: use deterministic starts:
  - uniform variance.
  - residual-only.
  - each single-block active.
  - small active sets (top-2, top-3 by KTA).
  - random perturbations around `log(var(y)/(B+1))`.
- Report best objective, endpoint spread, projected gradient,
  convergence per restart.

### M3. Identifiability diagnostics incomplete

Pairwise alignment misses multi-kernel linear dependence. Spec expects
"one aligned component to hit a boundary" which need not happen.

**Fix**: add condition number on the Gram matrix of vectorised kernels
(`G_ij = <vec(K_i), vec(K_j)> / (||K_i||_F ||K_j||_F)`). Flag if
`cond(G) > 1e6`. Drop zero-trace kernels. Adjust R3 acceptance to
"sum stability only" when alignment > 0.95.

### M4. sklearn API ambiguity

- `predict_components` listed but not shipped.
- Fixed-effects argument `X_fixed=None` complicates Pipeline / CV.

**Fix**:
- Make `fit(X, y)` and `predict(X)` the primary path.
- Default fixed effects to intercept only.
- For non-intercept fixed effects, accept via constructor (a column subset
  spec) or `fit_params` keyword; document sklearn limitations.
- Remove `predict_components` from MKM (only ship in BLUP).

### M5. Test tolerances too loose

- Likelihood test allows `1e-4` gradient error.
- **Fix**: require objective agreement vs brute force `1e-8`–`1e-10`;
  gradient finite-difference combined abs/rel tolerance `1e-6`. Add
  explicit tests for: rank-deficient `X_f`, boundary KKT,
  zero-trace kernels, aligned-kernel grouped recovery, and raw-vs-normalised
  Ridge equivalence.

### M6. Benchmark success criteria overclaim

- Median `< 0.99` labelled "true win" without paired uncertainty.
- **Fix**: pre-register variants, include per-fit timeout/failure rules,
  require convergence status in summary, use paired bootstrap or
  Wilcoxon-style intervals. Otherwise call it "matched within noise".

## Low-Severity Findings

### L1. Phase ordering leaves blockers too late

**Fix**: insert diagnostics/pruning phase before estimator benchmarks.
Make shared kernelizer a hard prerequisite before mkR equivalence or
benchmark comparison.

## Action Items

| # | Severity | Action | Status |
|---|----------|--------|--------|
| H1 | High | Rename `r` → `p_f` (rank) and `resid` (residual); rank-check `X_f` | applied to spec |
| H2 | High | Specify both absolute and simplex equivalence paths with mkR | applied to spec |
| H3 | High | Fix single-kernel Ridge equivalence formula | applied to test plan |
| H4 | High | Standardise gradient form using `a = V^-1 resid` | applied to spec |
| H5 | High | Specify exact `S = V^-1`, `P = S - Q M^-1 Q^T` recipe | applied to spec |
| M1 | Med | Standardise `y`; KKT boundary detection | applied to spec |
| M2 | Med | Multi-restart with deterministic + random starts | applied to plan |
| M3 | Med | Add condition number diagnostic; drop zero-trace blocks | applied to spec |
| M4 | Med | Remove `predict_components` from MKM (BLUP only); `fit(X, y)` primary | applied to plan |
| M5 | Med | Tighten gradient tolerance; add boundary / rank-deficient tests | applied to test plan |
| M6 | Med | Pre-registered variants, paired bootstrap, convergence filter | applied to benchmark |
| L1 | Low | Diagnostics-before-benchmark phase ordering | applied to plan |

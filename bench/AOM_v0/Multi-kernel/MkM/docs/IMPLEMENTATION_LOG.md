# MKM Implementation Log

This log is append-only. Each phase should add:

```text
date
phase
files changed
tests run
Codex review prompt used
findings fixed
findings deferred
```

## 2026-04-30: Planning Documents Created

Created the AOM-MKM documentation scaffold under `bench/aom_v0/Multi-kernel/MkM`.

Key decisions:

- AOM-MKM is the probabilistic counterpart of AOM-Ridge / mkR.
- Variance components are parameterised on the log scale to enforce positivity.
- REML is the default method; ML is shipped as a comparison baseline.
- L-BFGS-B optimiser with 10 random restarts (Dirichlet/normal init).
- Single Cholesky factorisation of `V` reused for all per-evaluation
  derivatives.
- Centred + trace-normalised block kernels (same convention as mkR).
- Equivalence with mkR at fixed `(sigma_b^2, sigma_e^2)` is a required test.
- Multi-trait, classification, POP variants are out of scope for v1.

Next steps:

- Codex roadmap review of `IMPLEMENTATION_PLAN.md` + `MKM_MATH_SPEC.md`.
- Phase 0 synthetic ground-truth.
- Phase 1 kernelizer (mirror mkR; share via adapter once mkR ships).
- Phase 2 likelihood + analytic gradient.
- Phase 3 optimisation (multi-start L-BFGS-B).
- Phase 4 estimator (sklearn API).
- Phase 5 smoke benchmark.

## 2026-04-30: Phases 0-5 complete (Claude pilot, after consolidation)

After the consolidation under `bench/AOM_v0/Multi-kernel/MkM`, all phases
through smoke benchmark are complete.

Files added:

- `mkm/__init__.py`, `kernelizer.py`, `likelihood.py`, `optimisation.py`,
  `estimator.py`.
- `tests/conftest.py`, `synthetic.py`, `test_likelihood.py`,
  `test_estimator.py`.
- `docs/PLAN_REVIEW_CORRECTIONS.md` (Codex roadmap review).
- `docs/CODEX_BACKLOG_2026-04-30.md` (Codex math/code review round 2).

Phase summary:

- **Phase 0** (synthetic) — uses Multi-kernel/MKR/tests/synthetic_mkr.py
  via the local conftest path setup. R1/R2/R3 generators verified.
- **Phase 1** (kernelizer) — `mkm.kernelizer` is a thin re-export of
  `aomridge.kernelizer.AOMKernelizer` (no duplication).
- **Phase 2** (likelihood) — REML / ML log-likelihood and analytic
  gradient implemented per spec. Single Cholesky reused for `S = V^-1`,
  `Q = V^-1 X_f`, `M^-1`, `P = S - Q M^-1 Q^T`. Analytic gradient =
  `0.5 (tr(P dV_j) - a^T dV_j a)` with `a = V^-1 resid`.
- **Phase 3** (optimisation) — multi-start L-BFGS-B with deterministic
  seeds (uniform, residual-only, single-block) plus random perturbations
  around `log(var(y)/(B+1))`. Boundary detection via projected-gradient
  KKT. Reports endpoint variance for multimodality detection.
- **Phase 4** (estimator) — sklearn-compliant `AOMMultiKernelMixedModel`
  with `fit(X, y) -> self`, `predict(X)`, `score(X, y)`. Stored
  attributes: `sigma2_blocks_`, `sigma2_residual_`,
  `relative_contributions_`, `beta_fixed_`, `alpha_dual_`, `theta_`,
  `log_likelihood_`, `converged_`, `boundary_components_`,
  `kernel_alignment_max_`, `kernel_alignment_matrix_`.
- **Phase 5** (smoke benchmark) — runs via
  `bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py`. On the
  3-dataset smoke cohort, MKM-reml beats PLS on BEER (rel-RMSEP 0.62) and
  matches PLS on ALPINE (0.99); on AMYLOSE it's at 1.17 (still better
  than Ridge-raw at 2.64).

Tests: 12 / 12 passing.

```
PYTHONPATH=bench/AOM_v0/Multi-kernel/MKR:bench/AOM_v0/Multi-kernel/MkM \
  pytest bench/AOM_v0/Multi-kernel/MkM/tests -q
```

## 2026-04-30: Codex review round 2 + applied fixes

Codex math/code review (`/tmp/codex_mkm_math.md`):

- HIGH (applied): rank-deficient `X_f` was not defensively checked inside
  `compute_neg_log_reml`. Added an SVD-based `_rank_of(X_f)` guard; raises
  a clear error when `rank(X_f) < X_f.shape[1]`. The estimator's fit path
  already runs `fit_fixed_effects(X_f)`, but direct callers
  (tests, benchmarks, users) get the safer error now.
- MEDIUM (deferred to backlog):
  - Boundary detection projected-gradient sign clarification.
  - ML-mode boundary diagnostics use REML gradient (correct for objective,
    but not ideal for boundary status flag).

Findings logged in `docs/CODEX_BACKLOG_2026-04-30.md`.

## 2026-04-30: Phase 6 — branch_preproc parameter added

Extended `AOMMultiKernelMixedModel.__init__` and `.fit` / `.predict` with
a `branch_preproc` parameter (`"none"`, `"snv"`, `"msc"`, `"asls"`,
`"osc"`, `"emsc1"`). Branch transformer is fitted on training data only,
stored as `self._branch_`, then applied to test data at predict time.
Imports `aomridge.branches.make_branch_preproc` lazily at fit time.

Tests: 12 / 12 still passing (branch_preproc is optional, default
`"none"`, no behaviour change for existing users).

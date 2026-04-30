# BLUP Implementation Log

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

Created the AOM-BLUP documentation scaffold under `bench/aom_v0/Multi-kernel/Blup`.

Key decisions:

- AOM-BLUP wraps AOM-MKM and adds **per-block prediction decomposition**.
- E-BLUP is what we ship (variances estimated by REML, not assumed known).
- `predict_components(X)` returns `{"fixed": ..., "<op_b>": ..., "total": ...}`.
- Decomposition sum identity is a primary test: `sum components == predict`.
- BLUP delegates fitting to MKM; no duplicated REML logic.
- `alpha_dual = V^-1 (y - X_f beta_hat)` is precomputed at fit time.
- Highly-aligned blocks (`align > 0.95`) flagged as non-identifiable
  individually but their sum is identifiable.

Next steps:

- Codex roadmap review.
- Phase 0 synthetic ground-truth.
- Phase 1 estimator scaffold (delegating to MKM).
- Phase 2 decomposition module.
- Phase 3 diagnostics.
- Phase 4 smoke benchmark.

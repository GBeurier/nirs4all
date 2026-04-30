# Multi-Kernel Status — 2026-04-30

Snapshot of the consolidated `bench/AOM_v0/Multi-kernel/` workspace.

## Layout

```
bench/AOM_v0/
├── Ridge/                          (preserved, original AOM-Ridge — untouched)
└── Multi-kernel/                   (consolidated home for new work)
    ├── aompls/                     (AOM-PLS package, shared via PYTHONPATH)
    ├── tests/                      (AOM-PLS originals: 97 ok, 12 skipped)
    ├── benchmarks/                 (build_cohorts, run_smoke, summarize, run_multikernel_smoke)
    ├── benchmark_runs/             (results.csv per cohort)
    ├── docs/                       (AOM-PLS docs, untouched)
    ├── publication/                (manuscript, figures, tables, scripts)
    │   ├── manuscript/
    │   │   ├── main.tex                             (AOM-PLS manuscript, untouched)
    │   │   └── MULTIKERNEL_PAPER_DRAFT.md           (NEW — multi-kernel sister manuscript)
    │   ├── figures/                                 (AOM-PLS figures + new mk-* figures)
    │   ├── tables/                                  (AOM-PLS tables + new mk-* tables)
    │   └── scripts/
    │       ├── make_figures.py                      (AOM-PLS, untouched)
    │       ├── make_multikernel_figures.py          (NEW)
    │       └── make_multikernel_tables.py           (NEW)
    ├── source_materials/           (AOM-PLS reference docs, untouched)
    │
    ├── MKR/                        (mkR package: multi-kernel Ridge)
    │   ├── aomridge/               (kernelizer, weights, mkr_estimator + Ridge support files)
    │   ├── tests/                  (48 tests passing)
    │   ├── benchmarks/             (created, empty for now)
    │   ├── benchmark_runs/
    │   ├── docs/                   (MKR_IMPLEMENTATION_PLAN, MKR_MATH_SPEC, MKR_TEST_PLAN, MKR_BENCHMARK_PROTOCOL, MKR_PLAN_REVIEW_CORRECTIONS)
    │   └── prompts/codex_review_prompts/  (mkr_{roadmap,math,code,test,publication}_review.md)
    │
    ├── MkM/                        (MKM package: multi-kernel mixed model with REML)
    │   ├── mkm/                    (kernelizer, likelihood, optimisation, estimator)
    │   ├── tests/                  (12 tests passing)
    │   ├── benchmarks/
    │   ├── benchmark_runs/
    │   ├── docs/                   (IMPLEMENTATION_PLAN, MKM_MATH_SPEC, TEST_PLAN, BENCHMARK_PROTOCOL,
    │   │                            CODEX_REVIEW_WORKFLOW, IMPLEMENTATION_LOG, PLAN_REVIEW_CORRECTIONS,
    │   │                            CODEX_BACKLOG_2026-04-30, codex_review_prompts/)
    │   └── prompts/
    │
    └── Blup/                       (BLUP package: per-block decomposition)
        ├── blup/                   (estimator only — wraps MkM)
        ├── tests/                  (10 tests passing)
        ├── benchmarks/
        ├── benchmark_runs/
        ├── docs/                   (IMPLEMENTATION_PLAN, BLUP_MATH_SPEC, TEST_PLAN, BENCHMARK_PROTOCOL,
        │                            CODEX_REVIEW_WORKFLOW, IMPLEMENTATION_LOG, PLAN_REVIEW_CORRECTIONS,
        │                            CODEX_BACKLOG_2026-04-30, codex_review_prompts/)
        └── prompts/
```

## Phases completed

| Phase | Description | Status | Tests passing |
|-------|-------------|--------|---------------|
| 0 | Master plans + scaffolding (mkR, MkM, Blup) | ✅ | n/a |
| 1 | mkR implementation (kernelizer, weights, estimator) | ✅ | 48 |
| 2 | MkM implementation (REML likelihood, optimiser, estimator) | ✅ | 12 |
| 3 | Blup implementation (decomposition wrapper around MkM) | ✅ | 10 |
| 4 | Codex round 2 (math + code reviews) — high-severity fixes applied | ✅ | 71 (all) |
| 5 | Smoke benchmark on 3 datasets (no branches) | ✅ | n/a |
| 6 | `branch_preproc` parameter (SNV, MSC, ASLS, OSC, EMSC1) added to all three estimators | ✅ | 71 (all) |
| 7 | Smoke benchmark with branch variants | running | — |
| 8 | Full 57-dataset benchmark | pending | — |
| 9 | Publication scaffolding | partly ✅ | — |

## Key results (smoke3, no branches)

From `benchmark_runs/smoke3/summary_per_variant.csv`:

| Variant | median rel-PLS | median rel-Ridge | median rel-TabPFN-opt | median fit-time |
|---------|----------------|-------------------|----------------------|------------------|
| **mkR-softmax_cv** | **0.95** ✓ | 1.00 | 1.37 | 39 s |
| BLUP-reml | 0.99 | 1.05 | 1.42 | 46 s |
| MKM-reml | 0.99 | 1.05 | 1.42 | 54 s |
| mkR-kta | 1.17 | 1.18 | 2.12 | 18 s |
| mkR-uniform | 1.35 | 1.37 | 2.15 | 22 s |
| Ridge-raw | 2.37 | 2.40 | 3.09 | 0.1 s |

**Headline**: `mkR-softmax_cv` matches/beats PLS on smoke median (0.95).
**MKM-reml** beats PLS by 38% on the small BEER dataset (rel 0.62) and
matches PLS on ALPINE (rel 0.99). All three new models close the
Ridge-vs-PLS gap (Ridge-raw is at rel 2.37, mkR/MKM/BLUP are around 1.0).

Per-dataset best (`benchmark_runs/smoke3/summary_per_dataset.csv`):

| Dataset | Best variant | RMSEP | rel-PLS |
|---------|--------------|-------|---------|
| ALPINE | mkR-softmax_cv | 0.0592 | **0.95** |
| AMYLOSE | MKM-reml | 2.238 | 1.17 |
| BEER | MKM-reml | 0.234 | **0.62** |

## Reproducibility

```bash
# Tests
.venv/bin/pytest bench/AOM_v0/Multi-kernel/{MKR,MkM,Blup}/tests -q
# 71 passed

# Smoke benchmark, no branches (~5 min)
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py \
  --cohort smoke3 --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3

# Smoke benchmark, with branches (~25 min)
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/run_multikernel_smoke.py \
  --cohort smoke3 --workspace bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches \
  --variants mkR-softmax_cv mkR-softmax_cv-snv mkR-softmax_cv-msc mkR-softmax_cv-asls \
             MKM-reml MKM-reml-snv MKM-reml-msc MKM-reml-asls

# Summarise
.venv/bin/python bench/AOM_v0/Multi-kernel/benchmarks/summarize_multikernel_smoke.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv

# Figures + tables
.venv/bin/python bench/AOM_v0/Multi-kernel/publication/scripts/make_multikernel_figures.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv
.venv/bin/python bench/AOM_v0/Multi-kernel/publication/scripts/make_multikernel_tables.py \
  bench/AOM_v0/Multi-kernel/benchmark_runs/smoke3_branches/results.csv
```

## Codex review log

Three Codex code+math reviews were run after the implementation:

| Review | Output | Status | Key fixes applied |
|--------|--------|--------|--------------------|
| mkR | `/tmp/codex_mkr_math.md` | done | Spy log shared via class attribute (no-leakage tests now non-vacuous) |
| MkM | `/tmp/codex_mkm_math.md` | done | Defensive `_rank_of(X_f)` guard in `compute_neg_log_reml` |
| Blup | `/tmp/codex_blup_math.md` | done | `total` accumulated independently of dict keys (handles duplicate block names) |

Plus 3 round-1 reviews on the plans (`/tmp/codex_*_roadmap.md`).

## Open items (deferred to future rounds)

- v2 fold-local kernelizer in `softmax_cv` to remove the inner-CV
  centring caveat (currently flagged as v1 limitation).
- ML-mode boundary diagnostics use REML gradient (functional but
  non-ideal).
- Full 57-dataset benchmark (Phase 7).
- LaTeX manuscript, CD diagrams, full ablation tables.
- POP-style per-component variants.
- Multi-output / classification.

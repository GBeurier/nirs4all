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

## Iteration cohort (user-curated 11 diverse datasets)

For rapid hypothesis-validation we use a hand-picked 11-dataset subset
that covers the diversity of the TabPFN paper (different sample sizes,
spectral resolutions, scientific domains). Saved at:

```
bench/AOM_v0/Multi-kernel/benchmark_runs/curated11_cohort.csv
```

Datasets (n_train range 81 → 2925):
- GRAPEVINE/An_spxyG70_30_byCultivar_MicroNIR
- DIESEL/DIESEL_bp50_246_b-a, DIESEL_bp50_246_hla-b
- MALARIA/Malaria_Sporozoite_229_Maia
- WOOD_density/WOOD_N_402_Olale
- MANURE21/All_manure_{CaO,P2O5}_SPXY_strat_Manure_type
- FUSARIUM/Fv_Fm_grp70_30
- BEEFMARBLING/Beef_Marbling_RandomSplit
- BERRY/ta_groupSampleID_stratDateVar_balRows
- ECOSIS_LeafTraits/Chla+b_spxyG_block2deg

The full 54-dataset cohort runs once we are converged on the iteration
cohort.

## Phases completed

| Phase | Description | Status | Tests passing |
|-------|-------------|--------|---------------|
| 0 | Master plans + scaffolding (mkR, MkM, Blup) | ✅ | n/a |
| 1 | mkR implementation (kernelizer, weights, estimator) | ✅ | 48 |
| 2 | MkM implementation (REML likelihood, optimiser, estimator) | ✅ | 12 |
| 3 | Blup implementation (decomposition wrapper around MkM) | ✅ | 10 |
| 4 | Codex round 2 (math + code reviews) — high-severity fixes applied | ✅ | 71 (all) |
| 5 | Smoke benchmark on 3 datasets (no branches) | ✅ | n/a |
| 6 | `branch_preproc` parameter (SNV, MSC, ASLS, OSC, EMSC1) added; smoke benchmark with branch variants | ✅ | 71 (all) |
| 6b | Codex round 3 review of branch results + Phase 7 plan | ✅ | n/a |
| 7a | Curated 11-dataset benchmark (iteration cohort) | running | — |
| 7b | Full 54-dataset cohort benchmark | pending | — |
| 8 | Publication scaffolding (manuscript draft, figures, tables, scripts) | in progress | — |

## Key results — DIESEL preview (curated11 partial, 2/11 datasets done)

When curated11 was first launched, the 2 fast DIESEL datasets completed
fully before workers got hung on the larger ones. Already a strong
signal:

**DIESEL_bp50_246_b-a** (n_train=113, p=401, ref_PLS=3.29, ref_TabPFN-opt=4.33):

| Variant | RMSEP | rel-PLS | rel-Ridge | rel-TabPFN-opt | fit-time |
|---------|-------|---------|-----------|----------------|----------|
| Ridge-raw | 14.85 | 4.52 | 5.23 | 3.43 | 0.03 s |
| mkR-softmax_cv | 2.86 | 0.87 | 1.01 | **0.66** | 5.5 s |
| mkR-softmax_cv-snv | 2.79 | **0.85** | 0.98 | **0.65** | 4.7 s |
| mkR-softmax_cv-msc | 2.92 | 0.89 | 1.03 | **0.67** | 6.8 s |
| MKM-reml | 2.83 | 0.86 | 1.00 | **0.65** | 11 s |
| MKM-reml-asls | 2.85 | 0.87 | 1.00 | **0.66** | 11 s |
| MKM-reml-msc | 2.79 | **0.85** | 0.98 | **0.64** | 9.8 s |

**DIESEL_bp50_246_hla-b** (n_train=133, p=401, ref_PLS=2.96, ref_TabPFN-opt=4.20):

| Variant | RMSEP | rel-PLS | rel-Ridge | rel-TabPFN-opt | fit-time |
|---------|-------|---------|-----------|----------------|----------|
| Ridge-raw | 17.04 | 5.76 | 6.26 | 4.05 | 0.04 s |
| mkR-softmax_cv | 2.69 | 0.91 | 0.99 | **0.64** | 8.2 s |
| mkR-softmax_cv-snv | 2.80 | 0.94 | 1.03 | **0.66** | 8.3 s |
| mkR-softmax_cv-msc | 2.83 | 0.95 | 1.04 | **0.67** | 7.5 s |
| MKM-reml | 2.60 | **0.88** | 0.96 | **0.62** | 3.5 s |
| MKM-reml-asls | 2.71 | 0.92 | 1.00 | **0.64** | 11 s |
| MKM-reml-msc | 2.64 | 0.89 | 0.97 | **0.63** | 10 s |

**Headline**: on both DIESEL datasets, **every multi-kernel variant beats
both PLS (-5 to -15%) and TabPFN-opt (-33 to -38%)**. Ridge-raw is
catastrophic (4–6× worse than PLS), confirming AOM kernel mixing is
essential for these high-dimensional, low-n problems.

Source: `benchmark_runs/curated11/results.csv` (n=2 datasets, partial run).

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

## Codex Round 3 review of Phase 6 results (2026-04-30)

Source: `/tmp/codex_phase6_review.md`. Key findings:

- HIGH: smoke n=3 is too small for inferential ranking — treat as QA.
- HIGH: don't report oracle median on small cohorts.
- MEDIUM (applied): convergence + boundary_components columns added to
  the runner CSV schema.
- MEDIUM: variant naming conflates branch + solver axes; report
  branch-lift WITHIN solver family (mkR-asls vs mkR, MKM-asls vs MKM).

**Phase 7 variant set (Codex-recommended)**:

```
Ridge-raw, mkR-softmax_cv, mkR-softmax_cv-snv, mkR-softmax_cv-msc,
MKM-reml, MKM-reml-asls, MKM-reml-msc.
```

Drops `mkR-softmax_cv-asls` (only helped BEER) and
`MKM-reml-snv` (MSC was slightly stronger / faster).

**Phase 7 statistical tests** (to add in summarizer):

- Wilcoxon signed-rank on per-dataset log RMSEP ratios.
- Sign / binomial tests for wins / losses / ties.
- Holm correction for planned pairwise comparisons.
- Effect sizes with bootstrap CIs.
- Failure / non-convergence counts.
- Sensitivity analysis where failures are ranked last.

## Open items (deferred to future rounds)

- v2 fold-local kernelizer in `softmax_cv` to remove the inner-CV
  centring caveat (currently flagged as v1 limitation).
- ML-mode boundary diagnostics use REML gradient (functional but
  non-ideal).
- Phase 7 full 54-dataset benchmark (running).
- Phase 8 LaTeX manuscript, CD diagrams, full ablation tables.
- POP-style per-component variants.
- Multi-output / classification.

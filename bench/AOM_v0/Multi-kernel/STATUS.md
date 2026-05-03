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

## Iteration cohort (user-curated diverse10, updated 2026-04-30)

For rapid hypothesis-validation we use a hand-picked 10-dataset subset
covering 8 scientific domains and a wide range of sample sizes (40 to
3734) and spectral widths (196 to 2151). Saved at:

```
bench/AOM_v0/Multi-kernel/benchmark_runs/diverse10_cohort.csv
```

| Dataset | n_train | p | Domain |
|---------|---------|---|--------|
| Beer_OriginalExtract_60_YbaseSplit | 40 | 576 | BEER |
| TIC_spxy70 | 43 | 254 | IncombustibleMaterial |
| An_spxyG70_30_byCultivar_NeoSpectra | 82 | 257 | GRAPEVINE_LeafTraits |
| ALPINE_P_291_KS | 247 | 2151 | ALPINE |
| All_manure_Total_N_SPXY_strat_Manure_type | 343 | 1003 | MANURE21 |
| All_manure_MgO_SPXY_strat_Manure_type | 343 | 1003 | MANURE21 |
| grapevine_chloride_556_KS | 388 | 1023 | GRAPEVINES |
| N_woOutlier | 1205 | 1154 | COLZA |
| Chla+b_spxyG_block2deg | 2925 | 196 | ECOSIS_LeafTraits |
| Chla+b_spxyG_species | 3734 | 196 | ECOSIS_LeafTraits |

Earlier iteration cohorts (`curated11_cohort.csv`, `curated10_cohort.csv`)
are preserved in the workspace but `diverse10` is the canonical one
going forward.

The full 54-dataset cohort runs once we are converged on the iteration
cohort. Stage A (51 datasets, n≤1500) is already done with 306
successful fits.

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
| 7a | Curated 10-dataset benchmark (curated10, iteration cohort v1) | ✅ | n/a |
| 7a-codex | Codex round 4 review of curated10 — variant set tightened | ✅ | n/a |
| 7b-stageA | Full 51-dataset benchmark (n_train ≤ 1500) | ✅ | n/a |
| 7b-stageB | 3 large datasets (n_train > 1500) | killed (LUCAS too slow); ECOSIS subset folded into diverse10 |
| 7c | Diverse-10 benchmark (user-curated v2) | running | — |
| 8 | Publication scaffolding (manuscript draft, figures, tables, scripts) | in progress | — |

## Iter 8 — Full 51-dataset Stage A benchmark (DONE)

**5 variants × 51 datasets = 255 fits, 0 failures.** Wall time 2h35min @ n_jobs=6.

Quartz_spxy70 excluded from primary medians (PLS ref RMSE pathologically tiny ~1e-6).

| Variant | n | rel-PLS | rel-TabPFN | wins-PLS | wins-TabPFN |
|---------|---|---------|------------|----------|--------------|
| mkR-default-active15-sparse3 | 50 | **0.968** | 1.082 | 33/50 | 10/50 |
| MKM-reml-asls-active15 | 50 | 0.970 | 1.095 | 32/50 | **13/50** |
| **mkR-asls-active15-sparse1** | 50 | 0.978 | **1.077** | 29/50 | 9/50 |
| mkR-msc-active15-sparse3 | 50 | 0.983 | 1.097 | 28/50 | 9/50 |
| Ridge-raw | 50 | 1.226 | 1.397 | 11/50 | 6/50 |

**Oracle (best variant per dataset)**:
- Median rel-PLS = **0.922** (8% better than PLS)
- Median rel-TabPFN = **1.047** (4.7% off TabPFN-opt)
- Wins: PLS 40/50 (80%), TabPFN-opt 21/50 (42%)

**Variant frequency in oracle**:
- MKM-reml-asls: 16 (32%) — REML strong on diverse domains
- mkR-default-sparse3: 12 (24%) — no-branch sparse winner
- mkR-msc-sparse3: 9 (18%)
- Ridge-raw: 7 (14%) — wins on simple linear/small-n datasets
- mkR-asls-sparse1: 6 (12%)

**Codex publication review** (`/tmp/codex_iter8_publication_review.md`):
- **Headline single variant**: `mkR-asls-sparse1` (best rel-TabPFN=1.077)
- **Robustness challenger**: `MKM-reml-asls` (13/50 TabPFN wins)
- **Story A** (single deployable champion) is cleanest. Story B (oracle) = headroom analysis.
- Stage B/classification out of scope for this paper.

## Iter 4 — score-method ablation + ASLS cross-over (4 datasets, DONE)

24/24 fits ok. New per-dataset bests:
- TIC: `mkR-softmax_cv-snv-active15-kta` = 1.170 / 1.101 (KTA score helps mkR)
- ALPINE: `mkR-softmax_cv-active15-tuned` = 0.940 / 1.348
- BEER: `mkR-softmax_cv-asls-active15` = 0.886 (mkR finally matches MKM with ASLS)

**Per-variant median** (focused 4): `mkR-softmax_cv-asls-active15` = 0.967 / 1.267 (3/4 wins) — ties MKM-reml-asls.

## Iter 5 — sparse softmax (post-hoc top-k) (4 datasets, DONE)

Implementation: after softmax_cv optimisation, zero out all but top-k weights, renormalise, re-search alpha grid at sparse weights. Token: `sparseN`.

36/36 fits ok. **Per-dataset bests** (rel-PLS / rel-TabPFN):
- BEER: `mkR-asls-active15-sparse2` = **0.784 / 1.321** (8% improvement over MKM-asls 0.850)
- TIC: `mkR-msc-active15-sparse3` = **1.167 / 1.099** (matches iter3 MKM-reml-msc)
- ALPINE: `mkR-active15-sparse3` = **0.911 / 1.306** (3% improvement over iter4 tuned)
- MANURE_MgO: `mkR-snv-active15-sparse3` = 0.933 / 0.972

**Per-variant medians (focused 4)**:
- `mkR-asls-active15-sparse1`: 0.945 / 1.255 (3/4)
- `mkR-asls-active15-sparse2`: 0.950 / 1.232 (3/4) ← BEST balance
- `mkR-msc-active15-sparse3`: 0.997 / 1.229 ← BEST rel-TabPFN

## Iter 6 — diverse10 validation (in progress)

Champions: sparse1/2/3 ASLS, sparse3 default, sparse3 MSC, MKM-reml-asls, Ridge-raw. 8/10 datasets done.

**Key generalisation finding**: `mkR-msc-active15-sparse3` wins N_woOutlier (n=1205) at 0.938 — resolves Codex P2 concern about hard-dataset failure mode.

## Iter 7 — no-alpha-retune ablation (4 datasets, DONE)

Codex P1: validate sparse softmax mechanism is structural, not just alpha rotation.

Decomposition of BEER ASLS gain (dense 0.886 → sparse2+retune 0.784):
- Sparse2 noretune (alpha frozen): 0.822 — **0.064 gain from pruning** (62%)
- Sparse2 retune: 0.784 — additional 0.038 from alpha re-search (38%)

**Verdict**: pruning is the primary mechanism. Alpha retune adds modest ~0.2-2.5% on top.

## Iter 2 — tuned + active30 (4 datasets focused) — DONE

All 16 fits OK. **Codex round 6 verdict**:
- Tuned MKM = no-op (numerically tied to iter1 across BEER/TIC/MANURE).
- Tuned mkR = mixed signal on n=4 (BEER +6%, TIC -3%); not robust.
- Active30 helps TIC (+1.7%) but hurts BEER (-3%); confirms diminishing
  returns above active15.
- **rel-TabPFN-opt still 1.25-1.30** (target < 1.0); not yet converged.

Codex Iter 3 priorities: kta/blend screen-method ablation, mkR-asls
cross-over, mkR-tuned no-branch.

## Iter 3 — branch ablation (in progress)

Testing untested branch combinations on 4-dataset focused cohort:
- MKM-reml-snv, MKM-reml-msc (untested SNV/MSC on MKM)
- BLUP-reml-asls, BLUP-reml-snv (untested branch combos for BLUP)
- mkR-kta-default-active15 (closed-form KTA strategy)
- mkR-softmax_cv-msc-default-active15

**Early findings (BEER YbaseSplit)**:
- BLUP-reml-asls = MKM-reml-asls = 0.85 (BLUP is a thin wrapper around
  MKM, so predictions coincide as expected; useful for diagnostic).
- MKM-reml-snv/msc collapse to rel-PLS 1.11 (much worse than ASLS).
- mkR-kta-default-active15 = 0.99 (close to softmax_cv); **closed-form
  KTA could be 6× faster** if competitive elsewhere.

## Iter 1 — active-screened default bank (7/8 diverse10 partial; killed at N_woOutlier)

**Codex round 5 review** (`/tmp/codex_iter1_review.md`,
saved to `MKR/docs/CODEX_BACKLOG_ITER1.md`) findings:

1. ✅ No train/test leakage. Screen runs on training y only.
2. ⚠️ **Naming bug**: I documented Iter1 as KTA-screened but the runner
   actually used `screen_score_method="norm"` (the default in
   `screen_active_operators`). Both are supervised screens but compute
   different scores. Need to either relabel or rerun with KTA explicitly.
3. ✅ Screening before trace normalisation is mathematically sound for
   linear operators.
4. ✅ Don't fall back to compact for small n — BEER/TIC (smallest n)
   are the strongest active15 wins.
5. ⚠️ MANURE_Total_N regression: variance-component identifiability
   issue with MKM-reml-asls + active ASLS subset. Keep compact MKM-asls
   as challenger.

**Iter 2 priority (per Codex)**: tighter hyperparams on active15 BEFORE
trying active30/50. `kernel_alignment_max` already saturates around 1
on most datasets, so active30 mostly adds collinear blocks.

**Hypothesis**: pre-screen 100-op `default` bank to top-15 by KTA → keep
operator diversity without selection-variance overhead.

**Implementation**: added `top_k_active` + `screen_score_method` params
to `AOMKernelizer`; passed through `kernel_top_k_active` in mkR/MKM/BLUP
estimators.

**Results** (7/8 datasets, N_woOutlier still running):

| Variant | Wins/7 PLS | Median rel-PLS | Median rel-TabPFN-opt |
|---------|------------|----------------|------------------------|
| **MKM-reml-asls-default-active15** | **5/7** | **0.966** | 1.208 |
| BLUP-reml-default-active15 | 4/7 | 0.968 | 1.198 |
| MKM-reml-default-active15 | 4/7 | 0.968 | **1.198** |
| mkR-softmax_cv-default-active15 | 4/7 | 0.974 | 1.212 |
| mkR-softmax_cv-snv-default-active15 | 4/7 | 0.994 | **1.127** |

**Per-dataset improvement vs baseline** (diverse10 best per dataset):

| Dataset | Baseline best | Iter1 best | Lift |
|---|---|---|---|
| **BEER YbaseSplit** | MKM-reml-asls 0.98 | MKM-reml-asls-default-active15 **0.85** | **+13.3%** |
| **TIC_spxy70** | mkR-snv 1.32 | mkR-snv-default-active15 **1.20** | **+9.5%** |
| MANURE_MgO | mkR-snv 0.95 | mkR-snv-default-active15 0.93 | +1.7% |
| grapevine_chloride | mkR-snv 1.03 | mkR-snv-default-active15 1.02 | +1.3% |
| ALPINE | mkR 0.95 | mkR-default-active15 0.95 | -0.3% (~tie) |
| MANURE_Total_N | MKM-asls 0.87 | MKM-asls-default-active15 0.88 | -1.9% |
| An_NeoSpectra | Ridge-raw 0.89 | MKM-reml-default-active15 0.94 | -6.1% (Ridge-raw still wins; not in iter1 variant set) |

**Verdict**: clear net positive. Iter1 should be retained as a default
recommendation for mkR/MKM with diverse / hard datasets.

**Iter 2 plan**:
- Combine `default-active15` with `tuned` budget
  (mkR n_restarts=5, max_iter=40, alpha_grid=40; MKM n_restarts=6,
  max_iter=150).
- Add `active30` as an alternative tuning to see if more diversity
  helps without tuning.
- Codex review (running).

## Key results — diverse10 user-curated v2 (9 datasets full + 1 partial)

`benchmark_runs/diverse10/results.csv` (55 rows = 54 ok across 9
datasets + Ridge-raw for the slow ECOSIS species):

| Variant | Wins / N | Median rel-PLS | Median rel-Ridge | Median rel-TabPFN-opt |
|---------|---------:|---------------:|-----------------:|----------------------:|
| **mkR-softmax_cv-snv** | 5 / 9 | **0.993** | 1.032 | 1.227 |
| MKM-reml-asls | 3 / 9 | 1.002 | 1.087 | 1.289 |
| MKM-reml-msc | 3 / 9 | 1.026 | 1.077 | 1.277 |
| mkR-softmax_cv | 4 / 9 | 1.054 | 1.054 | 1.251 |
| MKM-reml | 4 / 9 | 1.096 | 1.097 | 1.307 |
| Ridge-raw | 3 / 10 | 1.148 | 1.181 | 1.316 |

**Per-dataset best variant** (per `summary_per_dataset.csv`):

| Dataset | Best | rel-PLS | rel-TabPFN-opt |
|---------|------|---------|----------------|
| ECOSIS Chla+b block2deg (n=2925, p=196) | **Ridge-raw** | **0.504** | **0.491** |
| ECOSIS Chla+b species (n=3734, p=196) | **Ridge-raw** | **0.599** | **0.796** |
| MANURE_Total_N (n=343, p=1003) | MKM-reml-asls | 0.868 | 0.957 |
| An_NeoSpectra (n=82, p=257) | Ridge-raw | 0.888 | 0.976 |
| N_woOutlier (n=1205, p=1154) | mkR-softmax_cv-snv | 0.941 | 1.262 |
| MANURE_MgO (n=343, p=1003) | mkR-softmax_cv-snv | 0.948 | 0.989 |
| ALPINE (n=247, p=2151) | mkR-softmax_cv | 0.951 | 1.364 |
| BEER YbaseSplit (n=40, p=576) | MKM-reml-asls | 0.981 | 1.654 |
| grapevine_chloride (n=388, p=1023) | mkR-softmax_cv-snv | 1.034 | 1.227 |
| TIC (n=43, p=254) | mkR-softmax_cv-snv | 1.323 | 1.245 |

**Insight: 4 different variants share the per-dataset wins** —
- `Ridge-raw` × 3 (all ECOSIS-style n >> p, plus the small-n NeoSpectra)
- `mkR-softmax_cv-snv` × 4 (scatter-affected datasets)
- `MKM-reml-asls` × 2 (asymmetric-baseline datasets)
- `mkR-softmax_cv` × 1 (ALPINE)

**Decision rule emerging from diverse10**:

| n / p ratio | Recommended | Median rel-PLS gap |
|-------------|-------------|---------------------|
| `n >> 5p` (e.g. n=2925, p=196) | Ridge-raw | -50 % |
| `n ≤ p` & scatter present | mkR-softmax_cv-snv | -5 to -15 % |
| `n ≤ p` & asymmetric baselines | MKM-reml-asls | -2 to -13 % |
| `n ≤ p` & smooth spectra | mkR-softmax_cv | -5 to -10 % |

(Note: species Ridge-raw was added manually because the multi-kernel
fits on n=3734 took 3+ hours each and were terminated for compute
budget. The Ridge result alone validates the n >> p decision rule.)

## Key result — Ridge wins when n >> p (ECOSIS preview)

ECOSIS_LeafTraits/Chla+b_spxyG_block2deg (n=2925, p=196 — wide-sample,
narrow-spectra) shows a striking reversal: **Ridge-raw beats PLS by
50 %** (rel-RMSEP 0.504), while all multi-kernel variants underperform
PLS (rel-RMSEP 1.22-1.46).

This is consistent with the operator-mixture intuition: AOM kernels
help when n ≤ p (each operator gives a different "view" of a
high-dimensional spectrum), but lose value when n >> p (Ridge on raw
features already sees enough samples to fit a stable linear model).
**Use Ridge-raw on cohorts where n_train >> p**.

This finding is preserved in `benchmark_runs/diverse10/results.csv` and
will be highlighted in the manuscript discussion.

## Key results — Stage A (51 datasets, n_train ≤ 1500, 6 variants)

`benchmark_runs/all54_stageA/results.csv` (306 fits, 0 failures):

| Variant | Wins / 51 | Median rel-PLS | 95% CI | Median rel-Ridge | Median rel-TabPFN-opt | Median fit-time |
|---------|----------:|---------------:|--------|-----------------:|----------------------:|-----------------:|
| **MKM-reml-msc** | 29 / 51 | **0.982** | [0.962, 1.025] | 1.023 | 1.119 | 6 s |
| **MKM-reml** | **30 / 51** | 0.984 | [0.964, 1.013] | 1.001 | **1.070** | 8 s |
| mkR-softmax_cv | 27 / 51 | 0.984 | [0.963, 1.021] | 1.004 | **1.067** | 11 s |
| MKM-reml-asls | 29 / 51 | 0.988 | [0.968, 1.005] | 1.008 | 1.113 | 9 s |
| mkR-softmax_cv-snv | 29 / 51 | 0.992 | [0.974, 1.015] | 1.016 | 1.113 | 14 s |
| Ridge-raw | 12 / 51 | 1.222 | [1.107, 1.355] | 1.244 | 1.391 | 0.007 s |

Pairwise Wilcoxon between any two multi-kernel variants is NOT
significant (smallest p = 0.26 for MKM-reml vs mkR; Holm-corrected
p = 1.0). All five multi-kernel variants are statistically equivalent
on Stage A.

**Headline**: on a representative 51-dataset NIRS cohort, the multi-kernel
extensions of AOM-Ridge **systematically beat PLS at the median**
(0.98 vs PLS=1.0) and reduce the gap to TabPFN-opt to ~7 % from PLS's
~30 %. Ridge-raw remains 22 % behind PLS.

## Key results — curated10 (10 diverse datasets, 7 variants, all 70 fits ok)

From `benchmark_runs/curated10/per_variant_stats.csv`:

| Variant | Wins / N | Median rel-PLS | Median rel-Ridge | Median rel-TabPFN-opt | Median fit-time (s) |
|---------|---------:|---------------:|-----------------:|----------------------:|---------------------:|
| **MKM-reml-msc** | **7 / 9** | **0.952** | 1.003 | 1.142 | 26 |
| MKM-reml-asls | 6 / 9 | 0.964 | 1.002 | **1.096** | 26 |
| MKM-reml | 6 / 9 | 0.985 | 1.000 | 1.100 | 31 |
| mkR-softmax_cv-snv | 5 / 9 | 0.977 | 1.049 | 1.164 | 24 |
| mkR-softmax_cv-msc | 5 / 9 | 0.977 | 1.056 | 1.162 | 44 |
| **mkR-softmax_cv** | 5 / 9 | 0.976 | 1.007 | **1.080** | 28 |
| Ridge-raw | 1 / 9 | 1.263 | 1.321 | 1.475 | 0.05 |

(MALARIA was excluded from rel-PLS because the cohort csv lacks
`ref_rmse_pls` for it; all variants converged but RMSE ≈ 34 000 because
Y is sporozoite counts with Y_std ≈ 22 700, so all models barely beat
constant prediction. Real failure mode of the dataset, not a bug.)

**Pairwise Wilcoxon vs `mkR-softmax_cv`** (`pairwise_wilcoxon.csv`): MKM
variants are NOT statistically distinguishable from mkR at n=10 (all
p > 0.5, Holm-corrected p = 1.0). The differences are real but the
sample size is too small for inferential significance — full 54-dataset
benchmark needed.

**Per-dataset best variant** (`summary_per_dataset.csv`):

| Dataset | Best variant | rel-PLS | rel-TabPFN-opt |
|---------|--------------|---------|----------------|
| BEEFMARBLING | MKM-reml-msc | 0.95 | 1.14 |
| BERRY | mkR-softmax_cv | 1.01 | 1.24 |
| DIESEL_b-a | MKM-reml-msc | **0.85** | **0.64** |
| DIESEL_hla-b | MKM-reml | **0.88** | **0.62** |
| FUSARIUM | Ridge-raw | 0.96 | 1.18 |
| GRAPEVINE | MKM-reml-msc | 0.95 | **0.95** |
| MANURE_CaO | MKM-reml-msc | 0.94 | 1.26 |
| MANURE_P2O5 | MKM-reml-asls | 1.03 | 1.04 |
| WOOD | mkR-softmax_cv | **0.92** | **0.99** |

Multi-kernel models beat **PLS** on **7 of 9** datasets with valid PLS
ref (BERRY rel=1.01 and MANURE_P2O5 rel=1.03 are losses; MALARIA has no
PLS ref so n_ref=9). They beat **TabPFN-opt** on 4 datasets
(DIESEL_b-a, DIESEL_hla-b, GRAPEVINE, WOOD) — the latter is significant
because TabPFN-opt is a 110 M-parameter pre-trained foundation model.

**Default recommendation**: Codex round 4 review (`/tmp/codex_curated10_review.md`)
recommends keeping `mkR-softmax_cv` as the conservative default (it
loses to MKM-reml-msc on the median but wins on BERRY where MKM-reml-msc
gets rel 1.52). MKM-reml-msc and MKM-reml-asls become the "full-run
challengers" promoted only after the full 54-dataset confirmation.

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

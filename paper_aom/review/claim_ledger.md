# AOM paper claim ledger

Hand-curated ledger of every headline claim that appears, or is intended to
appear, in the Talanta manuscript. One row per claim. The ledger is the source
of truth for which numbers in the paper are **provisional**, which are
**verified**, and which are **blocked** on a specific experimental deliverable.

Maintained manually. Update on every promotion / launch / tag change. The
upstream specification for the claims is `paper_aom/review/experiments_needed.md`
lines 224-313 (Claims A through E). Denominators MUST come from
`paper_aom/review/cohort_manifest.csv` / `.md`.

Status legend:

- `provisional` -- value is in the manuscript today but rests on single-seed,
  partial, or exploratory evidence; must not be cited as final.
- `verified`    -- multi-seed (>= 3) result with paired statistics, timing
  audit, and reviewer-ready supplement entry.
- `blocked-on:<short-id>` -- value cannot be finalised until the named
  experiment / artefact lands.

| Claim | Text | Current provisional value | Denominator | Required evidence to finalise | Source artefact (today) | Target artefact (final) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A** | AOM gives fast auditable calibration: it moves preprocessing/model search inside a coefficient-bearing linear model, reducing HPO time vs PLS/Ridge HPO while preserving or improving RMSEP. | No headline number yet; current narrative says "AOM-PLS compact CV-5 ~= PLS HPO RMSEP at a fraction of the wall-clock". | regression: 57 paired (has_pls AND has_aom_pls AND has_tabpfn_hpo intersection); time table needs same denominator across all rows | (i) Multi-seed (0/1/2) PLS-tabpfn-hpo-25trials + Ridge-tabpfn-hpo-60trials runs with `fit_time_s`, `predict_time_s`, `search_time_s`. (ii) Matching AOM-PLS-compact-cv5 / ASLS-AOM-compact-cv5 / AOMRidge-AutoSelect timings on identical splits. (iii) Pareto figure + `table_time_budget.tex`. | `bench/AOM_v0/benchmark_runs/full/results.csv` has fit_time_s/predict_time_s for AOM-PLS variants (seed 0); no comparable PLS/Ridge HPO timing exists. | `bench/scenarios/runs/paper_aom_linear_hpo/results.csv` + `paper_aom/tables/table_time_budget.tex` + `paper_aom/figures/fig_accuracy_time_pareto.{pdf,svg}` + `paper_aom/review/linear_hpo_time_audit.md` | `blocked-on:P1-launch` (linear-HPO timing run) |
| **B** | AOM-PLS is a robust fast calibration layer; `ASLS-AOM-compact-cv5-numpy` is the headline variant. | Median RMSEP/PLS = 0.960 with 42/57 wins (seed 0). `AOM-compact-cv5-numpy` 0.992, 38/57 wins. `nirs4all-AOM-PLS-default` 0.999, 29/57 wins. | 57 paired regression datasets (has_pls AND has_aom_pls; QUARTZ excluded from pairwise ratio per manifest rule). | (i) Seeds 0/1/2 of AOM-PLS-compact-cv5 + ASLS-AOM-compact-cv5 on the 61-row cohort. (ii) Paired Holm-adjusted Wilcoxon vs PLS-tuned-cv5 (multi-seed). (iii) Ablation: compact vs default bank; CV-3 vs CV-5 vs rep-CV-3; one-SE rule on/off. | `bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv` (single-seed n=57 median 0.960 for ASLS-AOM-compact-cv5-numpy). | `bench/scenarios/runs/paper_aom_aompls_seeds012/results.csv` + `paper_aom/tables/table_regression_aompls.tex` + `paper_aom/review/aompls_multiseed_audit.md` | `blocked-on:P3-launch-aompls` |
| **C** | AOM-Ridge is the second instantiation: operator adaptation is not PLS-specific. Deployable selectors (Blender / AutoSelect) beat tuned Ridge on median. | `AOMRidge-Blender-headline-spxy3` median -2.22% vs Ridge, 35/52 wins. `AOMRidge-AutoSelect-headline-spxy3` median -0.61%, 27/52 wins. audit20 x seeds 0/1/2 already Holm-significant vs Ridge but mixed vs ASLS-AOM-compact. | 52 paired regression datasets (has_ridge AND has_aom_ridge minus the local fail set; QUARTZ excluded from pairwise ratio). The 52 vs 57 gap is due to AOMRidge-global allowability + Local k=50 failure on a few small-N datasets. | (i) Seeds 0/1/2 of AOMRidge-Blender + AutoSelect on the 61-row cohort. (ii) Holm-adjusted Wilcoxon vs Ridge-tuned-cv5 (multi-seed). (iii) Holm-adjusted Wilcoxon vs ASLS-AOM-compact-cv5 (cross-claim test). (iv) Failure-mode table for the 5-9 datasets without a deployable AOMRidge row. | `bench/AOM_v0/Ridge/publication/tables/table_per_method_summary.tex` (single-seed). `bench/AOM_v0/Ridge/docs/D_A_001_AUDIT20_PAIRED_STATS.md` (audit20 x 3 seeds). | `bench/scenarios/runs/paper_aom_aomridge_seeds012/results.csv` + `paper_aom/tables/table_regression_aomridge.tex` + `paper_aom/review/aomridge_multiseed_audit.md` | `blocked-on:P3-launch-aomridge` |
| **D** | Classification generalises the scoring setup: same operator-adaptive idea, different loss family, 17-dataset cohort. | None as headline; orientation only: master_classif `PLS-DA` ~0.757, `TabPFN-opt` ~0.732, `Catboost` ~0.717, `NICON` ~0.500 balanced accuracy (seed 42, 15 datasets). AOM-Ridge classifier results exist on 14 datasets (seed 0). | Classification cohort denominator from manifest: 16 (manifest `task==classification, status==include`); only 14 currently have AOMRidgeCls results; AOM-PLS-DA on 0 datasets so far. | (i) AOM-PLS-DA (nipals-adjoint and simpls-covariance) on 17-dataset cohort, seeds 0/1/2. (ii) AOM-Ridge classifier on full 17 cohort (currently 14), seeds 0/1/2. (iii) Add balanced accuracy, macro-F1, log-loss, ECE columns. (iv) Confirm or remove the 3 missing classification datasets (Genotype10_250 / Group9_1856 / Strawberry2C / Species_56_Bagnall) -- the Bagnall dataset is already excluded due to missing files. | `bench/master_results_classif.csv` (15 datasets, seed 42); `bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv` (14 datasets x 5 AOMRidgeCls variants, seed 0). | `bench/scenarios/runs/paper_aom_classification_seeds012/results.csv` + `paper_aom/tables/table_classification_main.tex` + `paper_aom/review/classification_multiseed_audit.md` | `blocked-on:P4-launch-classification` |
| **E** | FCK and related extensions were explored but not promoted as the recommended default; reported as negative diagnostic. | `AOMPLS-compact-with-fck-full57`: FCK selected on 17/57 datasets, strict AOM-Ridge gate fails: median +8.7%, q90 +35.8%, worst +136.6% (seed 0). `AOMRidgePLSCV-compact-with-fck` also fails strict promotion gates. | Datasets where FCK selector activated: 17 out of 57 (single seed) -- this denominator must be cited as "17/57 (seed 0)" until reseeded. | (i) Seeds 0/1/2 of AOMPLS-compact-with-fck-full57. (ii) FCK-on-vs-FCK-off paired table on the 17-of-57 activation subset. (iii) One paragraph in the Supplement framing as failed promotion (not negative result). No further headline runs required. | `bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv` (AOMPLS-compact-with-fck-full57, AOMRidgePLSCV-compact-with-fck rows). | `paper_aom/supplement.tex` Section "FCK exploration" + `paper_aom/tables/table_fck_diagnostic.tex` (seeds 0/1/2 only). | `provisional` (low priority; finalise after Claim A/B/C/D are verified) |

## Cross-cutting blockers

These items are not claims themselves but block multiple of the above:

| Blocker id | Description | Blocks claims |
| --- | --- | --- |
| `P1-launch` | Rebuild PLS-default-cv5, PLS-tabpfn-hpo-25trials, Ridge-default-cv5, Ridge-tabpfn-hpo-60trials with wall-clock + search-time + final-fit-time on the frozen cohort/splits. Same machine for all rows in the time table. | A (timing baseline), B (paired vs PLS), C (paired vs Ridge) |
| `P3-launch-aompls` | AOM-PLS-compact-cv5 + ASLS-AOM-compact-cv5 + nirs4all-AOM-PLS-default on cohort_manifest rows where `task==regression, status_in_primary_analysis==include`, seeds 0/1/2, with timing columns. | A, B |
| `P3-launch-aomridge` | AOMRidge-global-compact-none/snv, AOMRidge-Local-compact-knn50, AOMRidge-Blender-headline-spxy3, AOMRidge-AutoSelect-headline-spxy3 on the same cohort/seeds, with timing. Route through Local/MKL only for datasets where `aomridge_global_allowed==False`. | A, C |
| `P4-launch-classification` | AOM-PLS-DA + AOM-Ridge classifier variants on cohort_manifest rows where `task==classification, status_in_primary_analysis==include`, seeds 0/1/2, with balanced accuracy / macro-F1 / log-loss / ECE. | D |
| `P5-stats` | Aggregate paired Wilcoxon + Holm correction across seeds; emit `paper_aom/review/paired_stats.csv`; update `paper_aom/tables/table_regression_main.tex` and `table_regression_aomridge.tex`. | A, B, C, D |

## Bookkeeping

- Provisional numbers in the paper PDF (`paper_aom/AOM-paper.pdf` if present, else `main.tex`) MUST cite this ledger row + the manifest denominator until the row flips to `verified`.
- After every multi-seed run completes, edit this file in the same commit that introduces the artefact, and update the corresponding row's Status column.
- Do not delete rows when claims are dropped; mark them as `provisional` and add a "Dropped: <reason>" note. The audit trail matters for Talanta reviewers.

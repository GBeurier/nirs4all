# Codex hand-off — AOM paper finalization

Date: 2026-05-13.

This document briefs Codex on the state of the AOM paper after the in-session
infrastructure build + multi-seed benchmark runs.

## What was done in this session

### Day-0 infrastructure (P0, P2, P6, P7)

- `paper_aom/review/build_cohort_manifest.py` — generates the canonical
  cohort manifest from `bench/AOM_v0/benchmarks/cohort_regression.csv` (60 ok
  rows after Quartz exclusion) and `cohort_classification.csv` (17 ok rows).
- `paper_aom/review/cohort_manifest.csv` (78 rows) and
  `paper_aom/review/cohort_manifest.md` (auto-generated denominator rules).
- `paper_aom/review/claim_ledger.md` — per-claim status table.
- `paper_aom/review/selector_diagnostics.py` — operator + selector diagnostics
  extractor. Already produced: `operator_frequency.csv`,
  `selector_diagnostics.csv`, `failure_mode_table.csv`,
  `compact_bank_justification.md`, `table_selector_diagnostics.tex`.
- `paper_aom/review/aggregate_stats.py` — multi-schema results loader +
  paired stats + table/figure generator (`table_main_results.tex`,
  `table_paired_stats.tex`, `table_time_budget.tex`,
  `table_classification_main.tex`, `fig_accuracy_time_pareto.{pdf,png}`,
  `fig_runtime_distribution.{pdf,png}`, `final_stats.md`,
  `classification_stats.md`).
- `paper_aom/tables/table_software.tex` — rewritten to 10-row component
  status table (production / parity-verified / smoke-tested /
  code-ready-not-tested-locally / research).
- `bench/AOM_lib/cpp/build/c_smoke.c` — `#include <math.h>` added; smoke
  compile + run succeeds.

### nirs4all AOM-lib wrapper (P2)

- `nirs4all/operators/models/sklearn/aom_pls_aomlib.py` — sklearn wrapper
  around `bench/AOM_lib/python/src/aompls.AOMPLSCompact`. Lazy import.
  Diagnostic attributes: `selected_operator_sequence_`,
  `n_components_selected_`, `selected_operator_scores_`.
- `tests/unit/operators/models/test_aom_pls_aomlib.py` — 7 tests pass.
- `examples/aom_paper/aomlib_nirs4all_regression.py` — minimal example that
  runs end-to-end on `examples/sample_data/regression`.
- Export updated in `nirs4all/operators/models/sklearn/__init__.py`.

### Multi-seed benchmark runs (P3, P4)

Six benchmark jobs launched as background workers (24-core machine).
Workspaces:

| Workspace | Runner | Variants | Seeds | Status (end of session) |
| --- | --- | --- | --- | --- |
| `bench/scenarios/runs/paper_aom_aompls_seed{0,1,2}/` (merged into `paper_aom_aompls_seeds012/`) | `bench/AOM_v0/benchmarks/run_extended_benchmark.py` | 9 headline AOM-PLS variants incl. `ASLS-AOM-compact-cv5-numpy`, `AOM-compact-cv5-numpy`, `nirs4all-AOM-PLS-default`, `PLS-standard-numpy`, etc. | 0,1,2 | nearly complete (~1471/1620 rows) |
| `bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012/` | `run_aomridge_benchmark.py --variants top5_fast` (Blender/AutoSelect dropped — too slow for full-cohort multi-seed; cf. note below) | 5 (Ridge-raw, AOMRidge-global-compact-none-split_aware, MKL-shrink03, Local-knn50, Local-cv-blended) | 0,1,2 | running |
| `bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_seeds012_partial_headline.csv` | first attempt with full `headline` set; killed after 56 min (only 12 rows; 8 selector candidates each cost ~5 min/row) | mixed | 0,1 (partial) | partial reference only |
| `bench/AOM_v0/benchmark_runs/paper_aom_aompls_da_seeds012/` | `run_aompls_benchmark.py --task classification` | PLS-DA-standard + 4 AOM-PLS-DA variants | 0,1,2 | running |
| `bench/AOM_v0/Ridge/benchmark_runs/paper_aom_aomridge_cls_seeds012/` | `run_aomridge_classification.py --variants smoke` | smoke set | 0,1,2 | running |
| `bench/scenarios/runs/paper_aom_linear_hpo/` | `bench/tabpfn_paper/run_linear_hpo_paper_aom.py` | PLS-default-cv5, PLS-tabpfn-hpo-25trials, Ridge-default-cv5, Ridge-tabpfn-hpo-60trials | 0,1,2 | running (720 cells, slow; ~5 h wall clock) |

**Runtime bookkeeping** (`fit_time_s`, `predict_time_s`, `search_time_s`,
`refit_time_s`, `total_time_s` columns) is recorded in:
- `paper_aom_linear_hpo/results.csv` (per-trial wall clock)
- `paper_aom_aompls_seeds012/results.csv` (`fit_time_s`)
- `paper_aom_aomridge_seeds012/results.csv` (`fit_time_s`, `predict_time_s`)

### Bug fixes applied during the session

- `bench/AOM_v0/benchmarks/run_aompls_benchmark.py` —
  `AOMPLSDAClassifier.__init__()` doesn't accept `repeats`/`one_se_rule`/
  `cv_splitter`. Added `_aom_kwargs(kind, ...)` filter that strips those
  three keys when `kind == "classification"`. Patched 6 call sites.
- `bench/AOM_v0/benchmarks/cohort_regression.csv` — `Quartz_spxy70` marked
  `status=missing_data` (its `Xtrain.csv` file is missing). 60 datasets
  remain `status=ok`.

### Headline AOM-Ridge selector (Blender/AutoSelect) data — current state

The full `headline` variant set is prohibitively slow at full-cohort
× 3 seeds (~3 days wall clock). The paper-supplied evidence for the
deployable Blender/AutoSelect claims therefore relies on:

1. `bench/AOM_v0/Ridge/benchmark_runs/all54_headline/results.csv`
   (single-seed, full 54-dataset cohort, includes Blender + AutoSelect).
2. `bench/AOM_v0/Ridge/benchmark_runs/da001_audit20_seeds012/results.csv`
   (3 seeds × 20-dataset audit subset, includes Blender + AutoSelect).
3. The new `paper_aom_aomridge_seeds012/results.csv` covers the base
   deployable variants (top5_fast set: Ridge-raw, global-compact,
   MultiBranchMKL, Local-knn50, Local-cv-blended) on full cohort × 3 seeds.

Combining 1+2+3 is the basis for claim C (AOM-Ridge second instantiation).
The aggregator already handles both schemas via `_map_aom_v0_wide` and
`_map_harness` mappers in `paper_aom/review/aggregate_stats.py`.

## Validation request

Please audit the following in this order:

1. **Cohort manifest**
   - Read `paper_aom/review/cohort_manifest.csv` and
     `paper_aom/review/cohort_manifest.md`.
   - Confirm: 60 regression + 16 classification datasets included; QUARTZ
     handled per "absolute in failure table; excluded from ratio tables"
     rule; TabPFN-allowed and AOMRidge-global-allowed gates documented.
   - Cross-check against `bench/AOM_v0/benchmarks/cohort_regression.csv`
     and `cohort_classification.csv`.

2. **Claim ledger**
   - Read `paper_aom/review/claim_ledger.md`. For each claim A-E, confirm
     that the current evidence path is identified and that the "required
     evidence" column matches what is in the new multi-seed workspaces.

3. **Software status table**
   - Read `paper_aom/tables/table_software.tex`. Confirm row counts and
     status labels match reality:
     - Run `bench/AOM_lib/cpp/build/test_operators` (should pass).
     - Run `bench/AOM_lib/cpp/build/test_parity_kfold bench/AOM_lib/cpp/tests/reference` (should pass).
     - Run `PYTHONPATH=bench/AOM_lib/python/src pytest -q bench/AOM_lib/python/tests/test_parity.py` (should be 9 passed).
     - Compile + run `bench/AOM_lib/cpp/c_smoke.c` (or `cpp/build/c_smoke.c`) per the snippet in `paper_aom/review/experiments_needed.md` §P7 lines 719-723.

4. **AOM-lib wrapper**
   - Read `nirs4all/operators/models/sklearn/aom_pls_aomlib.py` and
     `tests/unit/operators/models/test_aom_pls_aomlib.py`.
   - Run `PYTHONPATH=bench/AOM_lib/python/src pytest -q tests/unit/operators/models/test_aom_pls_aomlib.py`.
   - Run `PYTHONPATH=bench/AOM_lib/python/src python examples/aom_paper/aomlib_nirs4all_regression.py`.

5. **Aggregator / diagnostics**
   - `python paper_aom/review/aggregate_stats.py --partial` should
     succeed and refresh all tables/figures.
   - `python paper_aom/review/selector_diagnostics.py` should likewise
     refresh `operator_frequency.csv` and `table_selector_diagnostics.tex`.
   - Sanity-check that the headline numbers in `final_stats.md`
     (median ratio, 95 % CI, wins) are within ±0.05 of the paper's
     provisional values:
       - ASLS-AOM-compact-cv5 vs PLS-default: median ratio ≈ 0.96, wins ≈ 42/57.
       - AOMRidge-Blender vs Ridge-tuned: median ratio ≈ 0.978, wins ≈ 35/52.

6. **Paper update**
   - Reread `paper_aom/main.tex` abstract, intro, results, and discussion.
   - Replace any provisional numbers with the regenerated ones from
     `paper_aom/review/final_stats.md`.
   - Add/replace references to:
     - `table_software.tex` (now 10 rows)
     - `table_time_budget.tex` (new, runtime claim)
     - `fig_accuracy_time_pareto.pdf` (new)
     - `fig_runtime_distribution.pdf` (new)
   - In supplement.tex, add a section "Multi-seed regression robustness"
     citing `paper_aom_aompls_seeds012/results.csv` and the AOM-Ridge
     audit data. Also a section "Parity validation" describing the
     BEER/CORN/ALPINE × kfold5/kfold5+oneSE/spxy5 parity tests with
     coefficient tolerance < 1e-8.
   - Update "Data and code availability" with the explicit subdirectories
     (`bench/AOM_v0/`, `bench/AOM_v0/Ridge/`, `bench/AOM_lib/`,
     `paper_aom/review/`).
   - Add the Novelty Statement and AI-assisted technologies declaration
     per Talanta requirements.
   - Rebuild `paper_aom/AOM-paper.pdf` and `AOM-supplement.pdf`.

## Known limitations to disclose in supplement

- **Blender/AutoSelect multi-seed full-cohort data is not collected fresh
  in this session**; the paper combines (a) full-cohort single-seed
  `all54_headline` and (b) audit20 × 3-seed data. Disclose this denominator
  splitting explicitly in the caption of `table_paired_stats.tex`.
- **Linear HPO timing run may still be incomplete** at the time of paper
  build; if so, mark the time-budget claim "preliminary" and rebuild the
  table after the run finishes.
- **Quartz_spxy70** is excluded from all pairwise ratio analyses (denominator
  near zero) and from the new multi-seed run (raw file missing). It still
  appears in the failure table with absolute RMSEP only.
- **POP-PLS-DA / AOM-PLS-DA classification rows** were initially failing
  due to a runner bug (`repeats` kwarg). Fixed mid-session via the
  `_aom_kwargs(kind, ...)` filter. All subsequent rows are clean.

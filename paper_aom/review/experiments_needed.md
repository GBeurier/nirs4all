# AOM paper: experiments to finish before Talanta submission

Date: 2026-05-13.

Target manuscript: `paper_aom/` / `AOM-paper.pdf`.

Target journal: Talanta. Talanta's Guide for Authors asks for demonstrated
analytical applicability, validation against established methods, statistical
treatment, and data/code availability. Current guide:
https://www.sciencedirect.com/journal/talanta/publish/guide-for-authors

## Position

Keep the broad AOM paper. Do not reduce the manuscript to a smaller
Ridge-only, PLS-only, or "scope down" story. The publishable idea is the
linear exploration/operator-adaptive calibration layer:

- PLS and Ridge are two instantiations of the same operator-adaptive principle.
- The central value is not only a small RMSEP gain; it is a fast, auditable
  alternative to large preprocessing/HPO searches.
- Classification can be added as a secondary validation axis if its results
  are generated with the same cohort discipline and metrics.
- The paper must distinguish current production `nirs4all`, the benchmark
  target under `bench/`, and the dedicated `AOM_lib` package.

The missing work is therefore:

1. freeze and characterize datasets/tasks cleanly;
2. rerun fair PLS/Ridge HPO baselines from the TabPFN paper setup with timing;
3. validate robustness across seeds for the models actually presented;
4. add classification scores as secondary evidence;
5. connect the dedicated C++ implementation and bindings as real software
   artefacts;
6. provide `nirs4all` example/wrapper code that uses `AOM_lib` and explains
   how it supports the paper.

## Current state verified locally

### Paper narrative

`paper_aom/main.tex` already frames the broad contribution correctly:

- operator-adaptive calibration, not only a new preprocessing filter;
- strict linear operators inside the calibration model;
- SNV, MSC, EMSC, OSC and ASLS treated as fold-local branches, not fixed
  operators;
- AOM-PLS and AOM-Ridge as complementary linear calibration models;
- current manuscript is regression-only, with classification not yet integrated.

Current abstract-level numbers should be treated as provisional until the
final reruns:

- ASLS + compact AOM-PLS CV-5: median RMSEP/PLS = 0.960, 42/57 wins;
- deployable AOM-Ridge selector: median -2.22% vs tuned Ridge, 35/52 wins;
- oracle AOM-Ridge quantities are separated from deployable results.

This is the right high-level structure for Talanta if the evidence is made
cleaner and the compute-time story is explicitly measured.

### Bench target vs current nirs4all

Earlier recommendations mixed the production state of `nirs4all` with the
benchmark target. The paper should use `bench/` as the scientific evidence
source and use `nirs4all` as a user-facing integration path.

Relevant current targets:

- `bench/AOM_v0/Summary.md`: AOM-PLS evidence; the benchmark champion is
  `ASLS-AOM-compact-cv5-numpy`, not the older
  `nirs4all-AOM-PLS-default`.
- `bench/scenarios/model_registry.yaml`: current canonical candidate names
  and preset membership.
- `bench/scenarios/{fast_reliable,strong_practical,best_current,exhaustive_research}.json`:
  generated scenario manifests as of 2026-05-12.
- `bench/scenarios/runs/*_full57_seed0/results.csv`: recent production
  harness runs, seed 0 only, useful for dispatch sanity but not sufficient for
  journal-level robustness.
- `bench/AOM_lib/`: dedicated AOM-PLS C++/Python/R/MATLAB/Julia/JS package
  artefacts.

The manuscript must explicitly say which numbers come from the bench target
and which software is production `nirs4all` versus dedicated `AOM_lib`.

### Regression evidence map

| Artefact | What it currently proves | Limitation before Talanta |
| --- | --- | --- |
| `bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv` | AOM-PLS champion: `ASLS-AOM-compact-cv5-numpy`, 57 datasets, median 0.960, 42 wins, median fit 1.36 s. | Single seed; cohort properties not fully surfaced; prediction artefacts incomplete. |
| `bench/AOM_v0/benchmark_runs/full/results.csv` | Broad AOM-PLS grid: 7888 rows, 59 datasets, 134 variants, seed 0. | Too heterogeneous for direct claims unless filtered. |
| `bench/AOM_v0/Ridge/publication/tables/table_per_method_summary.tex` | Deployable AOM-Ridge variants: Blender 35/52, median -2.22% vs Ridge; AutoSelect 27/52, median -0.61%. | Single-seed publication table; selectors still need full multi-seed support. |
| `bench/AOM_v0/Ridge/docs/HEADLINE_SPXY3_NESTED_AUDIT.md` | AutoSelect/Blender selector mechanism is nested with respect to external test set. | Audit does not replace full-cohort multi-seed evidence. |
| `bench/AOM_v0/Ridge/docs/D_A_001_AUDIT20_PAIRED_STATS.md` | 20 datasets x 3 seeds; AutoSelect/Blender beat Ridge and global compact; mixed versus ASLS-AOM. | Still not full 57/61; relative ratios have extreme tails on QUARTZ-like rows. |
| `bench/scenarios/runs/best_current_full57_seed0/results.csv` | Recent production harness sanity: 456 rows, 57 datasets, 8 candidates, seed 0, 428 OK. | Old preset membership; single seed; failures/timeouts on several datasets. |
| `bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv` | Broad dispatch sanity: 1823 rows, 61 datasets, 31 candidates, seed 0, includes FCK and selectors. | Exploratory; many terminal failures, failures and skips. |
| `bench/AOM_lib/README.md` | Dedicated package exists: C++ core, C API, Python, R, MATLAB, Julia, JS/WASM. | R/Julia/MATLAB/JS tests need matching runtimes or honest tested-status labels. |

Local verification done for this review:

- `bench/AOM_lib/cpp/build/test_operators`: pass.
- `bench/AOM_lib/cpp/build/test_parity_kfold bench/AOM_lib/cpp/tests/reference`: pass on BEER/CORN/ALPINE x kfold5/kfold5+oneSE/spxy5.
- C ABI smoke compiled and ran against `libaompls.so`; the smoke source needs
  `#include <math.h>` to remove the `sqrt` warning.
- `PYTHONPATH=bench/AOM_lib/python/src pytest -q bench/AOM_lib/python/tests/test_parity.py`: 9 passed.
- `ctest`, `R`, `julia` and `matlab` are not installed in this local shell, so
  those package gates were not run here.

### TabPFN-paper PLS/Ridge HPO state

The old TabPFN paper setup contains exactly the baseline budget that should be
recreated for the AOM paper's time-gain claim.

Verified files:

- `bench/tabpfn_paper/run_reg_pls.py`
- `bench/tabpfn_paper/full_run.py`
- `bench/tabpfn_paper/run_reg_tabpfn.py`

Important setup details:

- external split: `SPXYFold(n_splits=3, random_state=42)`;
- PLS preprocessing search:
  - `[None, SNV, MSC, EMSC(degree=1), EMSC(degree=2)]`
  - `[None, SG(11,2,1), SG(15,2,1), SG(21,2,1), SG(31,2,1), SG(15,3,2), SG(21,3,2), SG(31,3,2), Gaussian(0,1), Gaussian(0,2)]`
  - `[None, ASLSBaseline, Detrend]`
  - `[None, OSC(1), OSC(2), OSC(3)]`
  - `StandardScaler(with_mean=True, with_std=False)`
  - `PLSRegression(scale=False)`, `n_components` 1..25, 25 trials;
- Ridge HPO from `full_run.py`:
  - same linear preprocessing space;
  - `Ridge()`, 60 TPE trials;
  - `alpha` log-uniform 1e-5..1e4;
  - `fit_intercept` bool;
  - `solver` in `auto`, `svd`, `cholesky`, `lsqr`;
  - `tol=1e-4`, `positive=False`.

Current aggregate result files:

- `bench/1_master_results.csv`: regression baselines, 335 rows, seed 42.
  PLS and Ridge are partial; no usable timing columns for the time-gain claim.
- `bench/master_results_classif.csv`: classification baselines, 71 rows,
  15 datasets, seed 42.

Conclusion: do not merely import the old TabPFN CSVs. Recreate PLS/Ridge HPO
with the current paper cohort, fixed splits, result artefacts and wall-clock
timing. This is essential because "AOM saves time" is a stronger and cleaner
Talanta claim than only "AOM slightly improves RMSEP".

### Classification state

Classification is already partly implemented and should be added as secondary
evidence, not as a new main claim unless the final results are strong.

Verified artefacts:

- `bench/master_results_classif.csv`: 15 classification datasets.
  Models include `TabPFN-opt`, `Catboost`, `PLS-DA`, `TabPFN-Raw`, `NICON`.
  Primary metric is balanced accuracy; macro-F1 is also present.
- `bench/AOM_v0/benchmarks/cohort_classification.csv`: 17 classification
  datasets with train/test paths and properties.
- `bench/AOM_v0/benchmarks/run_aompls_benchmark.py`: supports classification
  variants:
  - `PLS-DA-standard`
  - `AOM-PLS-DA-global-nipals-adjoint`
  - `POP-PLS-DA-nipals-adjoint`
  - `AOM-PLS-DA-global-simpls-covariance`
  - `POP-PLS-DA-simpls-covariance`
- `bench/AOM_v0/Ridge/benchmarks/run_aomridge_classification.py`: supports
  AOM-Ridge classifier variants and reports balanced accuracy, macro-F1,
  log loss and ECE.
- `bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv`:
  70 rows, 14 datasets, 5 variants, seed 0 only.

Current classification baselines should be used only as orientation:

- `bench/master_results_classif.csv` median balanced accuracy:
  `PLS-DA` about 0.757, `TabPFN-opt` about 0.732, `TabPFN-Raw` about 0.718,
  `Catboost` about 0.717, `NICON` about 0.500.

Conclusion: add classification scores, but rerun AOM-PLS-DA and AOM-Ridge
classification on the 17-dataset cohort for seeds 0/1/2 before making claims.

### nirs4all and AOM_lib integration state

Current `nirs4all` does not yet use `AOM_lib` as its AOM implementation.

Verified files:

- `nirs4all/operators/models/sklearn/aom_pls.py`: legacy Python/Numpy AOM-PLS
  implementation.
- `nirs4all/operators/models/sklearn/aom_pls_classifier.py`: classifier wrapper
  around the legacy regressor.
- `nirs4all/operators/models/sklearn/__init__.py`: exports the current
  legacy classes.
- `tests/unit/operators/models/test_aom_pls.py`
- `tests/unit/operators/models/test_aom_pls_classifier.py`
- `tests/unit/operators/models/test_pop_pls.py`

`bench/AOM_lib/python/src/aompls` provides a tested Python package binding to
the dedicated implementation. The safer integration path is to add a new
optional wrapper or backend instead of changing existing `nirs4all` behaviour:

- new wrapper: `nirs4all/operators/models/sklearn/aom_pls_aomlib.py`; or
- new parameter on the existing estimator: `backend="legacy" | "aomlib"`.

For the paper, prefer the new wrapper first. It gives a clean example without
risking regressions in existing `nirs4all` users.

### AOM_lib and AOM-Ridge scope

`bench/AOM_lib` currently implements AOM-PLS, not AOM-Ridge. A search for
Ridge/AOMRidge in `bench/AOM_lib` returns no AOM-Ridge implementation.

Therefore:

- do not present `AOM_lib` as the full PLS+Ridge implementation yet;
- do present it as the dedicated AOM-PLS implementation with C++ core and
  multi-language bindings;
- keep AOM-Ridge code referenced under `bench/AOM_v0/Ridge`;
- if adding the paper to the `AOM_lib` repo/subdir, label it as a companion
  paper for operator-adaptive calibration, with AOM-PLS implemented in
  `AOM_lib` and AOM-Ridge still in the benchmark research code.

## Claims to keep

### Primary claim A: AOM gives fast auditable calibration

This should become the main Talanta story:

> AOM moves part of the preprocessing/model-search burden inside a linear
> coefficient-bearing calibration model, reducing the need for expensive
> external preprocessing HPO while preserving or improving accuracy on many
> NIRS datasets.

This claim requires a new timing table:

- PLS default/light CV;
- PLS TabPFN-style preprocessing HPO;
- Ridge default/light CV;
- Ridge TabPFN-style preprocessing HPO;
- AOM-PLS compact CV-5;
- ASLS-AOM compact CV-5;
- AOM-Ridge deployable variants;
- TabPFN-Raw/HPO only where allowed.

Report both:

- calibration accuracy: RMSEP, MAE, R2, paired ratios/deltas;
- time budget: preprocessing search time, final fit time, total wall-clock,
  failed/timeout rows.

### Primary claim B: AOM-PLS is a robust fast calibration layer

Use `ASLS-AOM-compact-cv5-numpy` as the AOM-PLS headline, but separate:

- ASLS as branch preprocessing;
- strict operator contribution from `AOM-compact-cv5`,
  `AOM-compact-cv3`, `AOM-compact-repcv3`;
- compact-vs-default bank ablations.

Numbers currently usable as provisional references:

- `ASLS-AOM-compact-cv5-numpy`: 57 datasets, median RMSEP/PLS 0.960,
  42/57 wins.
- `AOM-compact-cv5-numpy`: 57 datasets, median RMSEP/PLS 0.992,
  38/57 wins.
- `nirs4all-AOM-PLS-default`: 57 datasets, median around 0.999,
  29/57 wins.

### Primary claim C: AOM-Ridge is the second instantiation

Use AOM-Ridge to show that operator adaptation is not PLS-specific.

Current conservative deployable numbers:

- `AOMRidge-Blender-headline-spxy3`: 52 paired datasets, median -2.22% vs
  tuned Ridge, 35/52 wins.
- `AOMRidge-AutoSelect-headline-spxy3`: 52 paired datasets, median -0.61% vs
  tuned Ridge, 27/52 wins.

Current subset robustness:

- audit20 x seeds 0/1/2: AutoSelect and Blender beat Ridge-tuned-cv5 with
  Holm-adjusted p < 0.05;
- they do not consistently clear the Holm gate versus
  `ASLS-AOM-compact-cv5-numpy`.

Do not lead with oracle results. Keep oracle tables as retrospective upper
bounds only.

### Secondary claim D: classification generalizes the scoring setup

Classification can strengthen the paper if framed as secondary validation:

- same operator-adaptive idea;
- different loss/metric family;
- balanced accuracy, macro-F1, log loss and ECE;
- 17-dataset cohort with explicit denominators;
- seeds 0/1/2 minimum.

Do not let classification distract from the regression/HPO/time story. It
belongs in one concise main-text table or figure plus Supplement details.

### Secondary claim E: FCK and other extensions were explored, not promoted

FCK is useful to show that the exploration was not hand-picked, but it should
not be promoted as a recommended default:

- `AOMPLS-compact-with-fck-full57`: FCK selected on 17/57 datasets, but strict
  AOM-Ridge gate fails: median +8.7% vs curated AOM-Ridge best, q90 +35.8%,
  worst +136.6%.
- `AOMRidgePLSCV-compact-with-fck`: also fails strict promotion gates.

Mention in Supplement as negative/diagnostic exploration.

## Ordered actionable roadmap

### P0 - freeze the paper object and cohorts

Goal: stop denominator drift before launching expensive runs.

Deliverables:

1. `paper_aom/review/cohort_manifest.csv`
2. `paper_aom/review/cohort_manifest.md`
3. `paper_aom/review/claim_ledger.md`
4. regenerated `paper_aom/tables/table_benchmark_diversity.tex`

The manifest must have one row per dataset/task/split with at least:

- `dataset`
- `task`
- `source_family`
- `source_run`
- `domain_group`
- `response_or_trait`
- `split_type`
- `n_train`
- `n_test`
- `n_features`
- `p_over_n_train`
- `has_pls`
- `has_ridge`
- `has_aom_pls`
- `has_aom_ridge`
- `has_pls_da`
- `has_aom_pls_da`
- `has_aom_ridge_cls`
- `has_tabpfn_raw`
- `has_tabpfn_hpo`
- `tabpfn_allowed`
- `aomridge_global_allowed`
- `status_in_primary_analysis`
- `exclusion_reason`

Current dimension sanity from recent harness outputs:

- `exhaustive_research_full57_seed0`: 61 datasets observed, OK dims for 60;
  `n_train` 28-39225, median 227;
  `n_test` 12-6192, median 146;
  `n_features` 125-4200, median 1003;
  median `p/n_train` about 3.50.
- split markers include SPXY/spxy, KS, y-based, random, group, by-cultivar,
  block2deg, NocitaKS and Maia.

Rules:

- Broad regression corpus can remain 57/61 datasets.
- Classification corpus is 17 datasets unless failures reduce a paired table.
- Pairwise tables may have different denominators, but each denominator must
  be explicit in the caption and derived from the manifest.
- QUARTZ needs an explicit treatment rule: include absolute RMSEP in the
  failure table, but exclude or cap relative ratios when the denominator is
  near zero.

Suggested command skeleton:

```bash
python paper_aom/review/build_cohort_manifest.py \
  --sources \
    bench/1_master_results.csv \
    bench/master_results_classif.csv \
    bench/AOM_v0/benchmark_runs/full/results.csv \
    bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv \
    bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv \
    bench/scenarios/runs/best_current_full57_seed0/results.csv \
  --out paper_aom/review/cohort_manifest.csv
```

If the script does not exist, create it before any heavy compute.

### P1 - rebuild fair PLS/Ridge HPO baselines with timing

Goal: support the time-gain claim against the strongest classical baselines.

This is the first heavy experiment to run because it changes the paper's main
argument from "AOM can win some datasets" to "AOM provides competitive
calibration with much lower search budget".

Deliverables:

1. `bench/scenarios/runs/paper_aom_linear_hpo/results.csv`
2. `bench/scenarios/runs/paper_aom_linear_hpo/search_artifacts/`
3. `paper_aom/review/linear_hpo_time_audit.md`
4. `paper_aom/tables/table_time_budget.tex`
5. `paper_aom/figures/fig_accuracy_time_pareto.*`

Required variants:

- `PLS-default-cv5`
- `PLS-tabpfn-hpo-25trials`
- `Ridge-default-cv5`
- `Ridge-tabpfn-hpo-60trials`
- `AOM-PLS-compact-cv5-numpy`
- `ASLS-AOM-compact-cv5-numpy`
- `AOMRidge-global-compact-none`
- `AOMRidge-Local-compact-knn50`
- `AOMRidge-Blender-headline-spxy3`
- `AOMRidge-AutoSelect-headline-spxy3`

Timing requirements:

- record total wall-clock;
- record HPO/search time separately from final refit time;
- record number of preprocessing candidates/trials evaluated;
- record fit time and predict time for AOM methods;
- record timeout and failure rows without silently dropping them;
- run on the same machine/environment for all methods in the time table.

Use the TabPFN paper preprocessing/HPO spaces described above, but do not reuse
old CSVs without timing. The old setup used `SPXYFold(n_splits=3,
random_state=42)`; for the AOM paper, either:

- reproduce that exact split as a historical TabPFN-paper baseline, and label
  it as such; or
- map the same HPO budget onto the frozen AOM paper splits and seeds.

Preferred paper route: map the same HPO budget onto the frozen AOM paper
cohort/splits, because the paired table then answers the current paper.

### P2 - integrate AOM_lib into nirs4all examples

Goal: make the software claim real and explainable to readers.

This can be done before or in parallel with long runs because it is mostly
code integration and documentation.

Deliverables:

1. `nirs4all/operators/models/sklearn/aom_pls_aomlib.py`
2. export in `nirs4all/operators/models/sklearn/__init__.py`
3. optional dependency/extra in `pyproject.toml` once package naming is fixed
4. `examples/aom_paper/aomlib_nirs4all_regression.py`
5. `examples/aom_paper/linear_hpo_vs_aom_time.py`
6. tests under `tests/unit/operators/models/`

Recommended implementation:

- add a new estimator wrapper, not a silent replacement of legacy AOM-PLS;
- import `aompls` lazily with a helpful error if the optional package is not
  installed;
- expose sklearn-compatible `fit`, `predict`, `get_params`, `set_params`;
- document the mapping between paper terminology and code:
  - compact bank;
  - CV selector;
  - one-SE rule if supported;
  - selected operator sequence;
  - component count.

Required tests:

- wrapper imports cleanly when `aompls` is available through
  `PYTHONPATH=bench/AOM_lib/python/src`;
- fit/predict returns shape-compatible predictions on a tiny regression
  fixture;
- wrapper diagnostics expose selected operator/component information;
- existing legacy AOM-PLS tests still pass.

This code is also the right place to show how `nirs4all` uses `AOM_lib` for
the paper. Keep the example small and executable.

### P3 - run final regression robustness seeds

Goal: robustness on the actual models presented in the paper.

Minimum final candidate set:

- `PLS-default-cv5`
- `PLS-tabpfn-hpo-25trials`
- `Ridge-default-cv5`
- `Ridge-tabpfn-hpo-60trials`
- `AOM-PLS-compact-cv5-numpy`
- `ASLS-AOM-compact-cv5-numpy`
- `AOMRidge-global-compact-none`
- `AOMRidge-global-compact-snv` only if kept in the text;
- `AOMRidge-Local-compact-knn50`
- `AOMRidge-Blender-headline-spxy3`
- `AOMRidge-AutoSelect-headline-spxy3`
- `TabPFN-Raw`
- `TabPFN-HPO-preprocessing` where allowed by `tabpfn_allowed`

Seed policy:

- required: seeds 0/1/2 on the final cohort;
- preferred for Talanta: seeds 0/1/2/3/4 for the final table;
- primary statistical unit: dataset-level seed mean;
- sensitivity: row-level dataset x seed table.

Do not use preset pools as paper claims. Presets are for software workflows.
The paper needs a fixed comparison table.

Suggested starting point, after verifying manifest membership:

```bash
python bench/harness/run_benchmark.py \
  --cohort full57 \
  --pipeline bench/scenarios/best_current.json \
  --workspace bench/scenarios/runs/paper_aom_best_current_full57_seeds012 \
  --seeds 0,1,2 \
  --stats
```

Then add selectors if they are not in the chosen manifest:

```bash
python bench/harness/run_benchmark.py \
  --cohort full57 \
  --pipeline bench/scenarios/exhaustive_research.json \
  --workspace bench/scenarios/runs/paper_aom_selectors_full57_seeds012 \
  --seeds 0,1,2 \
  --stats \
  --max-models 12
```

Expected outputs:

- one consolidated `results.csv` with all final candidates;
- `stats.json`;
- per-dataset failure/timeout table;
- exact list of candidate/dataset/seed rows not run because of constraints;
- sidecar artefacts for predictions/diagnostics where available.

### P4 - add classification score block

Goal: add classification without diluting the regression paper.

Deliverables:

1. `bench/AOM_v0/benchmark_runs/classification_aompls_seeds012/results.csv`
2. `bench/AOM_v0/Ridge/benchmark_runs/classification_all17_seeds012/results.csv`
3. `paper_aom/tables/table_classification_main.tex`
4. `paper_aom/review/classification_stats.md`

Run AOM-PLS-DA/POP-PLS-DA:

```bash
PYTHONPATH=bench/AOM_v0 python bench/AOM_v0/benchmarks/run_aompls_benchmark.py \
  --task classification \
  --cohort bench/AOM_v0/benchmarks/cohort_classification.csv \
  --master bench/master_results_classif.csv \
  --workspace bench/AOM_v0/benchmark_runs/classification_aompls_seeds012 \
  --seeds 0,1,2 \
  --cv 5
```

Run AOM-Ridge classifiers:

```bash
PYTHONPATH=bench/AOM_v0:bench/AOM_v0/Ridge python \
  bench/AOM_v0/Ridge/benchmarks/run_aomridge_classification.py \
  --workspace bench/AOM_v0/Ridge/benchmark_runs/classification_all17_seeds012 \
  --cohort full \
  --variants smoke \
  --cv 3 \
  --seeds 0 1 2 \
  --cohort-path bench/AOM_v0/benchmarks/cohort_classification.csv
```

Metrics:

- primary: balanced accuracy;
- secondary: macro-F1;
- calibration: log loss and ECE;
- higher-is-better stats for balanced accuracy/macro-F1;
- lower-is-better stats for log loss/ECE.

Stats:

- paired dataset-level seed means;
- win/tie/loss;
- bootstrap CI;
- sign/Wilcoxon with Holm correction for pre-registered comparisons;
- no-harm tails for decreases in balanced accuracy;
- failure/timeout table.

Classification should appear as secondary validation unless it clearly passes
all gates.

### P5 - final statistical tables and figures

Goal: make the paper statistically defensible.

Generate:

- `paper_aom/review/final_stats.md`
- `paper_aom/tables/table_main_results.tex`
- `paper_aom/tables/table_paired_stats.tex`
- `paper_aom/tables/table_time_budget.tex`
- `paper_aom/tables/table_classification_main.tex`
- `paper_aom/figures/fig_accuracy_time_pareto.*`
- `paper_aom/figures/fig_runtime_distribution.*`

Required regression analyses:

1. Median paired RMSEP ratios or deltas with 95% bootstrap CI.
2. Win/tie/loss counts per paired comparison.
3. Sign test and Wilcoxon signed-rank on log RMSEP ratios.
4. Holm correction over pre-registered primary comparisons.
5. Friedman test and Nemenyi/critical-difference summary on the complete
   candidate subset.
6. Effect sizes: Cliff's delta or paired rank-biserial.
7. No-harm tails: q75, q90, worst ratio, named worst dataset.
8. Stratification by split type and size regime.
9. Seed stability:
   - per-method median and IQR across seeds;
   - number of datasets where the winner changes across seeds;
   - selector variance for AutoSelect/Blender.
10. Runtime:
   - median/q75/q90 fit time;
   - median/q75/q90 total HPO+fit time;
   - timeout and failure counts;
   - speedup ratio of AOM methods versus TabPFN-style PLS/Ridge HPO.

Primary comparisons to pre-register:

- `ASLS-AOM-compact-cv5` vs `PLS-tabpfn-hpo-25trials`.
- `ASLS-AOM-compact-cv5` vs `PLS-default-cv5`.
- `AOM-compact-cv5` vs `PLS-default-cv5`.
- `AOMRidge-global-compact-none` vs `Ridge-tabpfn-hpo-60trials`.
- `AOMRidge-Local-compact-knn50` vs `Ridge-tabpfn-hpo-60trials` on compatible rows.
- `AOMRidge-Blender` vs `Ridge-tabpfn-hpo-60trials`.
- `AOMRidge-AutoSelect` vs `Ridge-tabpfn-hpo-60trials`.
- AOM-Ridge selectors vs `ASLS-AOM-compact-cv5` as cross-family context,
  not the central claim.
- TabPFN-HPO-preprocessing as a reference baseline only on allowed datasets.

Do not claim "beats TabPFN-opt" unless the deployable paired allowed-cohort
result supports it after correction.

### P6 - selector, operator and failure diagnostics

Goal: support the mathematical story with inspectable behaviour.

Required outputs:

- `paper_aom/review/selector_diagnostics.csv`
- `paper_aom/tables/table_selector_diagnostics.tex`
- `paper_aom/review/compact_bank_justification.md`
- generated Supplement section/table for the empirical path from broad AOM
  exploration to the compact preprocessing set;
- selected operator frequency for AOM-PLS compact CV-5;
- selected component count distribution;
- strict-linear AOM vs ASLS-branch contribution:
  `AOM-compact-cv5` vs `ASLS-AOM-compact-cv5`;
- compact vs default bank:
  `AOM-compact-*` vs `nirs4all-AOM-PLS-default`;
- AutoSelect chosen-candidate counts;
- Blender weight mean/std per candidate;
- cases where selector choice changes across seeds;
- failure-mode table for QUARTZ, Brix, Tleaf, FinalScore, LMA,
  LUCAS_SOC_all, LUCAS_SOC_Cropland, y-based splits and outlier splits.

Supplement rule: keep only the clean, source-backed compact-bank evidence in
the submitted Supplement. Anything still hand-assembled, not regenerated from
scripts, or not backed by final multi-seed runs should stay in
`paper_aom/review/compact_bank_justification.md` or this checklist until it is
ready.

The clean Supplement version should justify:

- exact compact bank contents and ordering, matching `bench/AOM_lib`;
- bank-size genealogy: compact, family-pruned, response-dedup, default,
  deep3/deep4 and compact+FCK;
- candidate-count argument for selection variance;
- ablation table from
  `bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv`;
- selected-operator frequencies from final result files;
- why OSC/EMSC/deep banks/FCK are exploratory or not promoted.

### P7 - software/package linkage and paper placement

Goal: the paper must stop saying only "planned". The implementation status
must be precise.

Required manuscript edits:

- `main.tex`: replace "under development" with the actual status:
  "A dedicated C++17 AOM-PLS implementation with C ABI and
  Python/R/MATLAB/Julia bindings is provided under `bench/AOM_lib`; C++ and
  Python parity were verified against the Python AOM_v0 reference."
- `tables/table_software.tex`: add rows for `AOM_lib/cpp`, `AOM_lib/python`,
  `AOM_lib/r`, `AOM_lib/matlab`, `AOM_lib/julia`; mark tested/not tested.
- `supplement.tex`: add the parity fixture design:
  BEER/CORN/ALPINE x kfold5/kfold5+oneSE/spxy5; coefficient tolerance < 1e-8,
  prediction tolerance < 1e-8, RMSE-curve tolerance < 1e-9.
- Data/code availability: include repository URL, exact subdirectory, release
  tag or archive hash.

Package/test gates:

```bash
# C++ direct tests without ctest
bench/AOM_lib/cpp/build/test_operators
bench/AOM_lib/cpp/build/test_parity_kfold bench/AOM_lib/cpp/tests/reference

# Python binding
PYTHONPATH=bench/AOM_lib/python/src pytest -q bench/AOM_lib/python/tests/test_parity.py

# C smoke after adding #include <math.h>
gcc bench/AOM_lib/cpp/build/c_smoke.c \
  -Ibench/AOM_lib/cpp/include \
  -Lbench/AOM_lib/cpp/build -laompls \
  -Wl,-rpath,$PWD/bench/AOM_lib/cpp/build \
  -lm -lstdc++ -o /tmp/aompls_c_smoke && /tmp/aompls_c_smoke
```

For the `AOM_lib` repo/subdir:

- add `bench/AOM_lib/paper/README.md` pointing to `paper_aom/` or the final
  DOI/preprint;
- add `CITATION.cff` once authors/title/DOI are fixed;
- if copying the manuscript source, state clearly that `AOM_lib` currently
  implements the AOM-PLS part, while AOM-Ridge remains in
  `bench/AOM_v0/Ridge`.

## Execution order

### Day 0: no heavy compute

1. Create `cohort_manifest.csv` and `claim_ledger.md`.
2. Freeze final regression and classification candidate lists.
3. Decide whether the HPO timing rerun uses exact old SPXYFold(3, seed 42) or
   the frozen paper splits. Preferred: frozen paper splits.
4. Add the `nirs4all` AOM-lib wrapper skeleton and tiny example.
5. Add/fix software status table and C smoke `math.h` warning.

### Days 1-3: heavy runs

1. Launch PLS/Ridge HPO timing rerun first.
2. Launch final regression seeds 0/1/2 for the fixed candidate set.
3. Launch classification AOM-PLS-DA and AOM-Ridge seeds 0/1/2.
4. Monitor timeouts and failures; never silently drop rows.
5. If time permits, extend final regression to seeds 3/4.

### Day 4: aggregation

1. Build final paired regression stats.
2. Build time-budget table and accuracy-time Pareto figure.
3. Build classification stats.
4. Aggregate selector/operator diagnostics.
5. Regenerate LaTeX tables and figures.

### Day 5: manuscript sync

1. Update `main.tex`, `supplement.tex`, `table_main_results.tex`,
   `table_paired_stats.tex`, `table_benchmark_diversity.tex`,
   `table_time_budget.tex`, `table_classification_main.tex`,
   `table_software.tex`.
2. Rebuild `paper_aom/AOM-paper.pdf` and `AOM-supplement.pdf`.
3. Add data/code availability, Novelty Statement, and AI-assisted technologies
   declaration if applicable.

## Reviewer risks and mitigation

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Dataset denominator drift | Current files mention 52, 57, 59, 61 and other denominators depending on source. | Manifest plus pairwise denominators in every caption. |
| Time-gain overclaim | Old CSVs do not contain enough timing detail. | Recompute PLS/Ridge HPO and AOM runs on the same machine with wall-clock/search/refit timing. |
| "AOM" means too many things | AOM-PLS, AOM-Ridge, branches, FCK and oracles can blur together. | Define strict operators, branch preprocessors, deployable selectors and oracle envelopes separately. |
| Single-seed full-cohort results | Current headline full cohorts are mostly seed 0; audit subsets have seeds. | Final seed 0/1/2 minimum, 0..4 preferred. |
| Selector overfitting | AOM-Ridge selectors are powerful and expensive. | Use nested audit plus multi-seed selector diagnostics. |
| Oracle confused with method | Oracle numbers are stronger than deployable ones. | Keep oracle in Supplement or clearly label as retrospective upper bound. |
| Classification distracts from regression | Classification is only partly run today. | Keep it as secondary validation unless it passes all gates. |
| AOM_lib overclaim | AOM_lib implements AOM-PLS, not AOM-Ridge. | State exact implementation scope and tested language status. |
| nirs4all confusion | Current nirs4all AOM is legacy Python, not AOM_lib. | Add explicit wrapper/example and state which code produced paper numbers. |
| TabPFN fairness | TabPFN rows are constrained by n/p gates and old splits. | Pair only where allowed; document preprocessing/HPO budget; do not claim broad superiority unless corrected stats support it. |

## Minimum acceptable submission framing

Use this framing if the final multi-seed and timing runs are completed:

> We introduce operator-adaptive calibration as a linear, coefficient-bearing
> way to move preprocessing/model selection inside NIRS calibration. The
> framework is instantiated in PLS and Ridge, evaluated across heterogeneous
> NIRS regression datasets, extended to classification scores, and compared
> against classical HPO and modern baselines with paired statistics, seed
> sensitivity, runtime budgets and software parity checks.

Avoid:

- "state of the art" unless the final deployable paired table supports it;
- "AOM beats TabPFN" unless the allowed paired cohort supports it;
- presenting ASLS/SNV/MSC gains as strict linear operator gains;
- presenting FCK or oracle results as recommended defaults;
- presenting `AOM_lib` as the full AOM-Ridge implementation;
- using old `nirs4all-AOM-PLS-default` performance as if it were the bench
  target.

## Open implementation tasks

- [ ] Create `paper_aom/review/build_cohort_manifest.py`.
- [ ] Create `paper_aom/review/cohort_manifest.csv`.
- [ ] Create `paper_aom/review/claim_ledger.md`.
- [ ] Regenerate `paper_aom/tables/table_benchmark_diversity.tex`.
- [ ] Freeze final regression candidate list.
- [ ] Freeze final classification candidate list.
- [ ] Recompute TabPFN-paper-style PLS HPO with timing.
- [ ] Recompute TabPFN-paper-style Ridge HPO with timing.
- [ ] Run final regression seeds 0/1/2, preferably 0..4.
- [ ] Run AOM-PLS-DA classification seeds 0/1/2.
- [ ] Run AOM-Ridge classification seeds 0/1/2.
- [ ] Add `nirs4all` optional AOM-lib wrapper/example.
- [ ] Add tests for the `nirs4all` AOM-lib wrapper.
- [ ] Aggregate final stats into `paper_aom/review/final_stats.md`.
- [ ] Aggregate classification stats into `paper_aom/review/classification_stats.md`.
- [ ] Export selector diagnostics.
- [ ] Finalize compact-bank justification for Supplement, or keep provisional
  material in `paper_aom/review/compact_bank_justification.md`.
- [ ] Export or document prediction artefacts.
- [ ] Update `table_time_budget.tex`.
- [ ] Update `table_classification_main.tex`.
- [ ] Update `table_software.tex` with `bench/AOM_lib` package status.
- [ ] Run R/Julia/MATLAB package parity where available.
- [ ] Add `bench/AOM_lib/paper/README.md` or equivalent paper pointer.
- [ ] Rebuild manuscript and supplement.

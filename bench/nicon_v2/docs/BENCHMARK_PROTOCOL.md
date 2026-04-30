# nicon_v2 — Benchmark Protocol

This document defines the dataset cohort, data-loading conventions, train/test split rules, metrics, baselines, output schema, and acceptance gates that govern every benchmark run in `bench/nicon_v2/benchmarks/`.

## 1. Cohort

The cohort mirrors the AOM-Ridge curated cohort plus the TabPFN-paper datasets so we can compare nicon_v2 directly to the existing benchmark rows.

* **Source root.** `/home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/<group>/<dataset>/Xcal.csv` etc.
* **Curated regression cohort (39 datasets).** Same as `bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv`. The cohort manifest is duplicated in `nicon_v2/datasets.py::CURATED_REGRESSION_COHORT`.
* **Smoke cohort (3 datasets).** `[ALPINE_P_291_KS, Beer_OriginalExtract_60_KS, Rice_Amylose_313_YbasedSplit]`. Used in CI / fast iteration.
* **Extended cohort (61 datasets).** Same as `bench/AOM_v0/benchmarks/cohort_regression.csv`. Used for the publication run.
* **Classification cohort (15 datasets).** Same as `bench/AOM_v0/benchmarks/cohort_classification.csv`. Optional in early phases.

Each dataset entry in the cohort manifest carries `(database_name, dataset, task, group, n_train, n_test, n_features, ref_rmse_pls, ref_rmse_paper_ridge, ref_rmse_tabpfn, ref_rmse_aom_ridge_branch_global, ref_rmse_aom_pls_global)`.

## 2. Data loading

* Use `pd.read_csv(path, sep=";")` with `_coerce_numeric` and column-mean imputation for residual NaNs (same logic as `bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py`).
* Train/test files are pre-split at the dataset level by the TabPFN paper convention (Kennard-Stone / SPXY / random / Y-based). We never re-split train+test.
* When a dataset has multiple measurement repetitions for one physical sample, splits are at the *physical-sample* level (the metadata column `Sample_ID` is honored). Failure to do so inflates R² and is the chronic mistake of the deep-NIR literature (Walsh et al. 2023). nicon_v2 always honours `Sample_ID` when present.

## 3. Splits and inner cross-validation

The TabPFN-paper / AOM cohort distributes each dataset with a **predefined train/test split** (Kennard-Stone, SPXY, random, or Y-based, depending on the dataset). The published reference numbers (PLS / Ridge / TabPFN raw / TabPFN opt / CNN / CatBoost — see `cohort_regression.csv`) are obtained on those predefined splits. To stay comparable we use the **predefined split as the primary evaluation protocol**:

* **Primary (publication) metric.** Fit on `Xtrain.csv` / `Ytrain.csv`, evaluate on `Xtest.csv` / `Ytest.csv`. Reported once per `(dataset, variant, seed)`.
* **Inner CV** (early stopping, α / n-component selection, σ_y bandwidth selection, conformal calibration) operates **inside** the training split:
  * Phase 0 / 1: 5-fold `KFold` (sklearn) for sklearn baselines; random 80/20 train/val for torch models.
  * Phase 2+: 5-fold `SPXYFold` (`nirs4all.operators.splitters.SPXYFold`) inside the train.
  * Phase 5+: an additional 20 % of the inner-train is held out as the conformal calibration set `C`.
* **Inner CV is never used as the publication metric.** Inner CV scores are diagnostics (`cv_min_score` column) but the row's `rmsep` always refers to the predefined test set.
* **Repeated runs.** For variance estimates we run each `(dataset, variant)` with `seeds = [0, 1, 2]` and report the median + IQR; the publication tables use the median.

Datasets without a predefined split (rare in this cohort) use a single deterministic SPXYFold with a held-out 30 % test set, recorded as `cv_protocol = "spxy_30_holdout"`.

## 4. Metrics

All metrics are reported on the **original (un-scaled) y** scale.

| Metric | Definition | When |
|--------|------------|------|
| `rmsep` | √mean((y_true − y_pred)²) | always |
| `mae`   | mean(|y_true − y_pred|) | always |
| `r2`    | sklearn `r2_score` | always |
| `bias`  | mean(y_pred − y_true) | always |
| `nll`   | mean Gaussian NLL (eq. §4 of MATH_SPEC) | UQ phase |
| `aleatoric_var` | mean of per-net σ̂²(x) | UQ phase |
| `epistemic_var` | variance across ensemble means | UQ phase |
| `coverage_90` | fraction of test samples in the 90 % conformal PI | UQ phase |
| `width_90`    | mean width of the 90 % PI | UQ phase |
| `interval_score_90` | mean Winkler / interval score = `(hi − lo) + (2/α)·max(0, lo − y) + (2/α)·max(0, y − hi)` (Gneiting & Raftery 2007) | UQ phase |
| `crps`  | empirical CRPS (closed-form for Gaussian forecasts) | UQ phase |
| `calibration_curve_csv` | path to a CSV with `(target_coverage, observed_coverage)` rows for nominal coverages 0.1, 0.2, …, 0.9 | UQ phase |
| `fit_time_s`, `predict_time_s` | wall-clock seconds | always |
| `total_params`, `peak_vram_mb` | model footprint | always |

## 5. Reference baselines

The reference numbers are stamped on every result row from `cohort_regression.csv` and `bench/AOM_v0/Ridge/benchmark_runs/curated/results.csv`:

| Column | Meaning | Source CSV |
|--------|---------|------------|
| `ref_rmse_pls`                | TabPFN-paper PLS reference | cohort_regression.csv |
| `ref_rmse_paper_ridge`        | TabPFN-paper Ridge reference | cohort_regression.csv |
| `ref_rmse_tabpfn_raw`         | TabPFN raw (no preproc) | cohort_regression.csv |
| `ref_rmse_tabpfn_opt`         | TabPFN with paper preprocessing | cohort_regression.csv |
| `ref_rmse_cnn`                | TabPFN-paper CNN baseline | cohort_regression.csv |
| `ref_rmse_catboost`           | TabPFN-paper CatBoost baseline | cohort_regression.csv |
| `ref_rmse_aom_ridge_curated_best` | best AOM-Ridge variant (per dataset) | curated/results.csv |
| `ref_rmse_aom_pls_global`     | best AOM-PLS variant | AOM-PLS full/results.csv (when available) |

Missing reference cells are stored as **empty** (NaN) — never imputed. Relative metrics are computed only when the reference is finite and positive.

In addition, every nicon_v2 benchmark **run** also produces *first-class* baseline rows for the existing `nirs4all` models so that the comparison is internal-consistent (without depending on cohort-CSV ref values that may pre-date our hardware/seed):

| Variant label | Family | Implementation source |
|---------------|--------|------------------------|
| `Ridge-baseline`     | sklearn Ridge + StandardScaler + 5-fold α-CV | `nicon_v2.models.baseline.RidgeBaseline` |
| `PLS-baseline`       | sklearn PLSRegression + StandardScaler + 5-fold n-comp CV | `nicon_v2.models.baseline.PLSBaseline` |
| `NICON-baseline`     | upstream `nirs4all.operators.models.pytorch.nicon` (read-only import) | `_build_nicon` |
| `DECON-baseline`     | upstream `nirs4all.operators.models.pytorch.nicon._build_decon` | `_build_decon` |
| `DeepSpectra-baseline` | (Phase 1) reimplementation of Zhang et al. 2019 | `nicon_v2.models.deepspectra` |
| `nicon_v2-V1`, `V2`, … | improved variants per phase | `nicon_v2.models.v{1,2,…}` |

Derived columns:
* `relative_rmsep_vs_<ref>` = `(rmsep − ref) / ref` for every reference column listed above.
* `relative_rmsep_vs_internal_pls` = `(rmsep − rmsep_pls_internal) / rmsep_pls_internal` where `rmsep_pls_internal` is our own `PLS-baseline` row on the same `(dataset, seed)`.

## 6. Result CSV schema

The aggregate CSV (`benchmark_runs/<workspace>/results.csv`) has columns:

```
dataset_group, dataset, task, n_train, n_test, n_features,
variant, model_version, seed, cv_fold, cv_protocol,
status, error_message,
rmsep, mae, r2, bias,
nll, aleatoric_var, epistemic_var, coverage_90, width_90, interval_score_90, crps,
calibration_curve_csv,
ref_rmse_pls, ref_rmse_paper_ridge, ref_rmse_tabpfn_raw, ref_rmse_tabpfn_opt,
ref_rmse_cnn, ref_rmse_catboost, ref_rmse_aom_ridge_curated_best,
relative_rmsep_vs_pls, relative_rmsep_vs_paper_ridge,
relative_rmsep_vs_tabpfn_raw, relative_rmsep_vs_tabpfn_opt,
relative_rmsep_vs_cnn, relative_rmsep_vs_aom_ridge_curated_best,
fit_time_s, predict_time_s, total_params, peak_vram_mb,
hyperparams_json, python_version, torch_version, cuda_version,
git_sha, host
```

In parallel, every `(dataset, variant, seed)` writes a **per-sample predictions parquet** at `benchmark_runs/<workspace>/predictions/<variant>__<dataset>__seed<seed>.parquet` with columns:

```
sample_id (int)              — index into Xtest.csv
y_true (float)
y_pred (float)
y_pred_sigma (float | NaN)   — predictive σ when UQ enabled
pi_lo_90, pi_hi_90 (float | NaN)
residual = y_pred − y_true
fold (int)                   — −1 for predefined split
seed (int)
variant (str), dataset (str)
```

This lets downstream scripts compute per-dataset paired tests, residual analyses, and wavelength-resolved attribution without re-running the model.

## 7. Acceptance gates by phase

Every phase produces a **paired** comparison between two variants (e.g. nicon_v2-V1 vs NICON-baseline) on the same `(dataset, seed)` rows. The accept rule is uniform:

* **Effect.** Median Δrmsep across the cohort ≤ −X % (X listed in the phase's hypothesis).
* **Significance.** Paired Wilcoxon signed-rank p < 0.05 (two-sided) across the cohort, or `n < 8` cohort entries (smoke runs are exempt from significance).
* **Per-dataset safety.** No dataset regresses by > 5 % rmsep vs the previous phase's accepted variant, *and* no dataset regresses by > 10 % vs any cohort reference (PLS / paper Ridge / TabPFN / CNN).
* **Length robustness.** The new model passes the forward/backward sanity tests on the spectrum-length set `{401 (DIESEL), 576 (Beer), 700 (CORN/BISCUIT), 1154 (AMYLOSE), 2151 (ECOSIS)}` — see `tests/test_length_robustness.py`.
* **Capacity guard.** `total_params ≤ 5 · n_features × n_train_typical` heuristic, *or* `total_params ≤ 1e6` — whichever is larger; recorded in the row.

**Phase-0 gate** (this iteration): the harness can reproduce the AOM-Ridge `Ridge-raw-stdscale` numbers within ±2 % on the smoke cohort, and the published `paper Ridge` numbers within ±5 % on at least one of the smoke datasets.

**Final gate** (Prompt.md):
* **Leaderboard success.** Median relative RMSEP vs `aom_ridge_curated_best` ≤ −2 %, paired Wilcoxon p < 0.05 on the curated 39-dataset cohort, ≥ 50 % wins, no dataset regresses > 10 % vs any reference.
* **Scientific success** (fallback). nicon_v2-best beats `NICON-baseline` and `DECON-baseline` on ≥ 75 % of the curated cohort with paired Wilcoxon p < 0.05; even if it does not clear the leaderboard gate, this is enough for a publishable contribution comparing CNN variants.

Both gates are reported in the manuscript; the leaderboard claim is contingent on both holding.

## 8. Resumability and reproducibility

* The benchmark runner appends rows to a shared CSV; if `(dataset, variant, seed)` is present with `status="OK"`, it is skipped (`--resume`, on by default).
* Random state is propagated through (i) numpy / torch / python global seeds, (ii) `random_state` of CV splitter, (iii) DataLoader workers, (iv) `cudnn.deterministic = True`, `cudnn.benchmark = False`.
* The full hyper-parameters and the git commit SHA are stored on every row so that any historical benchmark can be replayed.

### 8.1 Environment pin

Pinned in `bench/nicon_v2/environment.lock.json` (generated alongside the first benchmark run on a host). Records:

* python version (`sys.version`)
* package versions: `torch`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `nirs4all`, `pyarrow`
* CUDA toolkit version, GPU name, GPU driver
* OS / kernel string

Each result row also stamps `python_version`, `torch_version`, `cuda_version`, `host`, `git_sha`. When a phase is reproduced the script reads the lockfile and warns if any package version differs.

### 8.2 Replication packet

A reproducible publication run produces:

* `benchmark_runs/full/results.csv` (aggregate metrics per row)
* `benchmark_runs/full/predictions/*.parquet` (per-sample y_true, y_pred, σ, residuals)
* `benchmark_runs/full/manifest.json` (cohort hashes, file SHA-256s, environment lock, exact CLI commands, host info, total wall-clock)
* `benchmark_runs/full/log.txt` (command log; copies stdout/stderr per phase)
* `benchmark_runs/full/error_rows.csv` (any `status=ERROR` rows with full traceback)
* `publication/scripts/run_publication.sh` (the exact end-to-end script)

The replication checklist (`docs/REPLICATION_CHECKLIST.md`, written at publication time) lists each artefact, expected SHA-256s and the canonical command to regenerate it from a clean checkout.

## 9. Reporting commands

The publication scripts in `publication/scripts/` consume `benchmark_runs/full/results.csv` and produce:

* `publication/tables/summary_per_variant.csv`
* `publication/tables/summary_per_dataset.csv`
* `publication/figures/fig_critical_difference.pdf`
* `publication/figures/fig_per_dataset_delta_vs_pls.pdf`
* `publication/figures/fig_cumulative_rmsep.pdf`

## 10. Appendix — Bibliographic citation conventions for results

* When citing **paper Ridge** numbers we record TabPFN-paper Hollmann et al. 2025 (Nature) values.
* When citing **AOM** numbers we record the column from the corresponding `bench/AOM_v0/.../results.csv`.
* Where a reference is missing we leave the field empty (NaN); never imputed.

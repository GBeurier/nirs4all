# Evaluation Audit (Scoring and Reporting)

Date: 2026-02-14  
Scope: end-to-end scoring flow in `nirs4all` from model evaluation to persisted scores and tab reports.

---

## Current logic and functionning: detailed view of the scoring method in nirs4all

### 1) Training-time score computation

The primary score path is:

1. `BaseModelController.launch_training(...)` computes predictions per partition (`train`, `val`, `test`) on unscaled targets (`nirs4all/controllers/models/base_model.py:1226`, `nirs4all/controllers/models/base_model.py:1249`).
2. `ScoreCalculator.calculate(...)` selects task metric via `ModelControllerUtils.get_best_score_metric(...)`:
   - Regression => metric is `rmse` (`nirs4all/controllers/models/utilities.py:180`).
3. Scores are computed with `evaluator.eval(...)`, where `rmse = sqrt(mean_squared_error)` (`nirs4all/controllers/models/components/score_calculator.py:62`, `nirs4all/core/metrics.py:153`).
4. Prediction payload stores:
   - scalar `val_score`, `test_score`, `train_score`
   - full metric dict `scores[partition][metric]`
   - full arrays (`y_true`, `y_pred`) by partition.

### 2) CV fold and averaged-fold logic

For CV pipelines:

1. Each fold emits entries with `fold_id in {'0','1','2',...}`.
2. `_create_fold_averages(...)` creates:
   - `fold_id='avg'`
   - `fold_id='w_avg'`
3. Validation for `avg/w_avg` is built from concatenated OOF predictions (not from averaging fold RMSE scalars) (`nirs4all/controllers/models/base_model.py:1665`).

Important consequence:

- `avg` validation score (OOF RMSE) is **not exactly** `mean(fold RMSE)` in general.

### 3) In-memory prediction representation

`Predictions.add_prediction(...)` stores one row per partition in memory (`nirs4all/data/predictions.py:252`), with:

- identity fields (`dataset_name`, `config_name`, `model_name`, `fold_id`, `step_idx`, etc.)
- scalar scores (`val_score`, `test_score`, `train_score`)
- arrays (`y_true`, `y_pred`) for the partition.

Ranking API:

- `Predictions.top(...)` supports `score_scope` (`cv`, `final`, `mix`, `flat`) (`nirs4all/data/predictions.py:487`).
- Final entries rank by `cv_rank_score` when available (`nirs4all/data/predictions.py:629`).

### 4) Store persistence (DuckDB)

`Predictions.flush(...)` persists buffered rows through `WorkspaceStore.save_prediction(...)` (`nirs4all/data/predictions.py:388`).

Store notes:

- `predictions` table does **not** persist `config_name`, `step_idx`, or `cv_rank_score` as first-class columns.
- For persisted rows, `val_score` is written as `row.get("val_score") or 0.0` (`nirs4all/data/predictions.py:445`), so `None` becomes `0.0`.

### 5) Pipeline-level best score used for refit selection

When a pipeline completes:

- `executor` computes pipeline `best_val` from `prediction_store.get_best(...)` and stores it in `pipelines.best_val` (`nirs4all/pipeline/execution/executor.py:253`).
- `get_best()` ranks on `rank_partition='val'` by default (`nirs4all/data/predictions.py:856`).

Because fold rows, `avg`, and `w_avg` all exist in val partition, `pipelines.best_val` often becomes the **best single fold val score**, not the CV aggregate score users expect.

### 6) Refit selection and final metadata

`extract_top_configs(...)` supports:

- `ranking='rmsecv'`: uses `pipelines.best_val` (`nirs4all/pipeline/execution/refit/config_extractor.py:186`)
- `ranking='mean_val'`: computes arithmetic mean of per-fold val scores (`nirs4all/pipeline/execution/refit/config_extractor.py:279`)

But when building `RefitConfig`, `best_score` is always set from `completed.best_val` (`nirs4all/pipeline/execution/refit/config_extractor.py:258`), regardless of ranking mode.

During refit relabel:

- `fold_id='final'`
- `refit_context='standalone'`
- `val_score=None`
- `cv_rank_score=refit_config.best_score` (`nirs4all/pipeline/execution/refit/executor.py:393`).

### 7) Refit storage and run-level aggregation

Refit predictions are merged into the per-dataset store (`run_dataset_predictions`), but not into the global `run_predictions` path in orchestrator refit dispatch.

Effect:

- dataset-level final reports work,
- global final ranking helpers relying on `run_predictions` are incomplete.

### 8) Tab report generation logic

`_print_refit_report(...)` (orchestrator) drives:

1. Final summary table via `TabReportManager.generate_per_model_summary(...)`.
2. Top CV chains table from `predictions.top(..., score_scope='cv', fold_id='avg', rank_partition='val')`.

Per-model summary columns are computed as:

- `Test RMSE`: final entry `test_score`
- `CV_test_avg`: mean fold test score
- `CV_test_wavg`: weighted mean fold test score
- `Avg_val`: entry `cv_rank_score`
- `RMSE_CV`: `_compute_rmse_cv_indexed(...)` over concatenated OOF arrays.

### 9) Concrete reproduction from your case

Dataset: `rice_amylose_313_ybasedsplit`  
Run: `c46cf581-3c4e-4ab4-af31-ad9642a120d5` (DuckDB `bench/tabpfn_paper/workspace/store.duckdb`)

For pipeline `TABPFN_Paper_5901efe6` (`SNV>Detr`):

- Fold val RMSE: `2.9217`, `3.3546`, `3.3642`
- Mean fold RMSE: `3.2135`
- `avg` (OOF RMSE): `3.2194`
- OOF MSE: `10.3646`
- Stored `pipelines.best_val`: `2.9217`
- Final refit test RMSE: `4.0758`

For pipeline `TABPFN_Paper_442cdbfc` (`raw`):

- Fold val RMSE: `3.4518`, `3.9359`, `3.9894`
- Mean fold RMSE: `3.7923`
- `avg` (OOF RMSE): `3.7991`
- OOF MSE: `14.4330`
- Stored `pipelines.best_val`: `3.4518`
- Final refit test RMSE: `5.1228`

These numbers match the confusing report you shared:

- `Avg_val` aligns with stored `best_val` (best fold), not average CV value.
- `RMSE_CV` aligns with OOF MSE, not OOF RMSE.

---

## Current status: what's missing, what's wrong, what's ok and ready

### What is wrong (confirmed defects)

1. `RMSE_CV` column can print MSE instead of RMSE for regression runs already using `rmse`.
   - Root cause: `display_as_rmse` flag wired to `metric == "mse"` before calling `_compute_rmse_cv_indexed(...)` (`nirs4all/visualization/reports.py:167`, `nirs4all/visualization/reports.py:202`, `nirs4all/visualization/reports.py:551`).
   - Impact: large misleading values (`10.3646`, `14.4330`) in a column named RMSE.

2. `Avg_val` in final table is not “average CV validation score”.
   - It prints `cv_rank_score`, which is set from `RefitConfig.best_score`.
   - In current multi-criteria flow, `best_score` is populated from `pipelines.best_val` (`nirs4all/pipeline/execution/refit/config_extractor.py:258`), often the best single fold score.
   - Impact: ranking column meaning is ambiguous and inconsistent with CV table.

3. Refit criterion semantics are collapsed.
   - `mean_val` criterion is computed during selection, but selected config score metadata still uses `best_val`.
   - Impact: reported “Selected by” and reported `Avg_val` are semantically disconnected.

4. Refit entries are not first-class in global result prediction buffer.
   - Refit runs populate `run_dataset_predictions`; global `run_predictions` is not updated in refit path.
   - Impact: APIs depending on global predictions (e.g., `best_final` path using `self.predictions.top(score_scope='final')`) can be empty/inconsistent.

5. Refit `val_score=None` is lost when persisted.
   - `flush()` converts falsy to `0.0` (`row.get("val_score") or 0.0`).
   - Impact: store-level data suggests a real validation score for refit rows where none exists.

6. Score naming and user-facing labels are overloaded.
   - At least four different “CV validation” concepts exist:
     - best fold RMSE
     - mean fold RMSE
     - OOF RMSE (`fold_id='avg'`)
     - OOF MSE
   - Impact: users cannot infer ranking basis directly from current tables.

### What is missing

1. A single explicit scoring contract across training, selection, storage, and reporting.
2. Persisted fields for refit ranking provenance (`cv_rank_score`, criterion used, score type).
3. Stable score glossary in user docs and report headers.
4. Regression tests asserting numeric invariants between:
   - fold scores
   - avg/w_avg scores
   - final summary columns.
5. Explicit per-table “sorted by” indicator in printed reports.

### What is OK and ready

1. Base metric computation path (RMSE on unscaled values) is coherent.
2. OOF construction for `fold_id='avg'`/`'w_avg'` is correct and unbiased.
3. Multi-criteria selection mechanism (top-k by multiple criteria + dedup) works functionally.
4. Report indexing (`_build_prediction_index`) is performant and structurally clean.
5. Dataset-level refit reporting path is operational and already richer than earlier versions.

---

## Backlog of what should be done

### P0 (must fix first: correctness and trust)

1. Fix `RMSE_CV` computation/display wiring.
   - `RMSE_CV` must always display RMSE when column name is RMSE.
2. Make `best_val` semantic explicit and consistent.
   - Decide one canonical definition for `rmsecv` ranking (recommended: OOF RMSE from `fold_id='avg'` val score).
   - Update pipeline completion logic accordingly.
3. Split/rename `Avg_val` into explicit score types.
   - Example columns:
     - `CV_rank_score`
     - `CV_oof_rmse`
     - `CV_mean_fold_rmse`
4. Preserve criterion-specific refit score.
   - `RefitConfig.best_score` should correspond to the criterion that selected the config, or store per-criterion scores.
5. Merge refit entries into global run result path (or make all APIs consume per-dataset union consistently).
6. Preserve `val_score=None` for refit persistence (no `or 0.0` coercion for nullable score fields).

### P1 (stabilization and maintainability)

1. Persist score provenance fields in store schema:
   - `score_type`, `cv_rank_score`, `selection_criterion`.
2. Add a typed score payload/model to replace implicit ad-hoc fields.
3. Add report legends and sorting basis line in every summary table.
4. Add a “score audit” utility to print all score variants for one model chain.

### P2 (quality and UX)

1. Harden chain summary SQL:
   - aggregate CV summaries from relevant partitions only (avoid accidental weighting by duplicated rows).
2. Add documentation page in user guide that explains each score and when to use it.
3. Add CI tests using a tiny deterministic dataset for full scoring-path snapshots.

---

## Your overral point of view

The scoring system is close to being robust, but today it is failing at one critical boundary: **semantic consistency** across layers.

Model evaluation itself is mostly correct; the major defects are in how score variants are selected, named, propagated, and displayed. Right now, different modules treat `best_val` as different concepts (best fold, mean fold, OOF CV), and the final report mixes those concepts under ambiguous headers.

The right direction is not a patchwork of display tweaks. It is to define and enforce a strict score contract with explicit named score types, then wire selection/reporting/storage to those exact types. Once that contract is in place, ranking becomes explicit, reports become trustworthy, and debugging becomes straightforward.

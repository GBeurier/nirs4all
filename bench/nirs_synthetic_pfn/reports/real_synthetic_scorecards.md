# Real/Synthetic Scorecards

## Objective

Standardize Phase B2 scorecards for local real benchmark cohorts against A2 synthetic preset datasets.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py --n-synthetic-samples 80 --max-real-samples 80 --max-real-datasets 0 --seed 20260429`

## Outputs

- Markdown: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md`
- CSV metrics summary: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`

## Phase A Gate Override

- `phase_a_gate_override`: `A3_failed_documented`
- A3 fitted-only real-fit gate remains failed/documented and is not hidden by this B2 report.
- B2 comparisons are realism diagnostics only; they do not establish downstream transfer benefit.

## Config

- Seed: 20260429
- A2 synthetic presets generated: 74
- Synthetic samples per preset: 80
- Real samples per scored dataset cap: 80
- Real dataset cap: all runnable rows
- Comparison spaces: uncalibrated_raw, calibrated_raw_diagnostic, snv
- Thresholds are provisional smoke thresholds, not calibrated domain gates.
- Primary decisions and gates use only `comparison_space == "uncalibrated_raw"`; the historical `comparison_space == "raw"` lane is retired and `calibrated_raw_diagnostic` plus `snv` are diagnostics that cannot override an uncalibrated_raw failure.
- The uncalibrated_raw lane is scored on synthetic spectra before any marginal calibration is fitted or applied, so it remains the authoritative gate.
- Marginal calibration is applied only to the calibrated_raw_diagnostic and snv lanes; calibration does not change metric definitions or thresholds.

## Git Status

- `git status --short` lines: 65
- First entries:
  - ` M bench/AOM_v0/Ridge/aomridge/estimators.py`
  - ` M bench/AOM_v0/Ridge/aomridge/kernels.py`
  - ` M bench/AOM_v0/Ridge/aomridge/selection.py`
  - ` M bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv`
  - ` M bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py`
  - ` M bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md`
  - ` M bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py`
  - ` M bench/AOM_v0/aompls/estimators.py`
  - ` M bench/AOM_v0/aompls/preprocessing.py`
  - ` M bench/AOM_v0/aompls/scorers.py`
  - ` M bench/AOM_v0/aompls/selection.py`
  - ` M bench/AOM_v0/benchmark_runs/full/results.csv`
  - ` M bench/AOM_v0/benchmarks/run_aompls_benchmark.py`
  - ` M bench/AOM_v0/benchmarks/run_extended_benchmark.py`
  - ` M bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv`
  - ` M bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py`
  - ` M bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py`
  - ` M bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py`
  - ` M bench/nirs_synthetic_pfn/reports/integration_gate_status.md`
  - ` M bench/nirs_synthetic_pfn/reports/nirs_context_query_sampler_contract.md`
  - ` M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`
  - ` M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md`
  - ` M bench/nirs_synthetic_pfn/reports/transfer_validation.csv`
  - ` M bench/nirs_synthetic_pfn/reports/transfer_validation.md`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/prior_adapter.py`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/realism.py`
  - ` M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/transfer.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_prior_adapter.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_realism_scorecards.py`
  - ` M bench/nirs_synthetic_pfn/tests/test_transfer_validation.py`
  - `?? .claude/`
  - `?? .codex`
  - `?? bench/AOM_v0/Ridge/aomridge/branches.py`
  - `?? bench/AOM_v0/Ridge/aomridge/cv.py`
  - `?? bench/AOM_v0/Ridge/aomridge/mkl.py`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated_cohort.csv`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/curated_v2/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/smoke6/`
  - `?? bench/AOM_v0/Ridge/benchmark_runs/smoke_cv5/`
  - `?? bench/AOM_v0/Ridge/docs/CODEX_BACKLOG_round2_2026-04-29.md`
  - `?? bench/AOM_v0/Ridge/publication/`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_branch_global.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_mkl.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_one_se_and_repeated_cv.py`
  - `?? bench/AOM_v0/Ridge/tests/test_ridge_round3_fixes.py`
  - `?? bench/AOM_v0/docs/CV_SPLITTER_DESIGN.md`
  - `?? bench/AOM_v0/publication/tables/table_top15_score_time.tex`
  - `?? bench/nirs_synthetic_pfn/docs/06_SYNTHETIC_REALISM_REMEDIATION_ROADMAP.md`
  - `?? bench/nirs_synthetic_pfn/experiments/exp04_adversarial_auc.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp05_minimal_ablation_attribution.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp06_encoder_tabpfn_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/experiments/exp07_nirs_icl_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/reports/adversarial_auc.csv`
  - `?? bench/nirs_synthetic_pfn/reports/adversarial_auc.md`
  - `?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.csv`
  - `?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.md`
  - `?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv`
  - `?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md`
  - `?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.csv`
  - `?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.md`
  - `?? bench/nirs_synthetic_pfn/tests/test_adversarial_auc_report.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_encoder_tabpfn_gate_precheck.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_minimal_ablation_attribution.py`
  - `?? bench/nirs_synthetic_pfn/tests/test_nirs_icl_gate_precheck.py`

## Real Cohort Inventory

| source | cohort path | exists | total rows | status ok rows | runnable rows | rows with missing paths |
|---|---|---|---:|---:|---:|---:|
| `AOM_regression` | `/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_regression.csv` | `True` | 61 | 61 | 61 | 0 |
| `AOM_classification` | `/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_classification.csv` | `True` | 17 | 16 | 16 | 0 |

## Missing Paths

No missing paths among `status == ok` cohort rows.

## Scorecard Summary

- Report status: `done`
- Runnable real rows discovered: 77
- Selected real rows attempted: 77
- Uncalibrated real/synthetic comparison rows written (authoritative): 71
- Calibrated raw diagnostic comparison rows written: 71
- SNV diagnostic comparison rows written: 71
- Blocked selected rows written: 6
- Synthetic-only dry-run rows written: 0
- Load/score failures after selection: 6

## Synthetic Mapping Strategy

Deterministic matrix-first dataset mapping is used first; stable SHA-256 fallback is used only when no matrix rule matches. No dataset-index round-robin selection is used.
Rules use only source/task/database/dataset identifiers, never y values, labels, or split contents.

Strategy counts over uncalibrated_raw rows:
- `matrix_first_dataset`: 58
- `stable_hash_fallback`: 19

Synthetic preset counts over uncalibrated_raw rows:
- `baking`: 3
- `dairy`: 5
- `forage`: 20
- `fruit`: 4
- `fuel`: 4
- `grain`: 8
- `juice`: 5
- `meat`: 5
- `oilseeds`: 5
- `powders`: 1
- `soil`: 11
- `tablets`: 1
- `wine`: 5

## Real Marginal Calibration

Strong provisional marginal calibration is applied to synthetic spectra after the authoritative `uncalibrated_raw` lane is scored, and only before the `calibrated_raw_diagnostic` and `snv` diagnostic lanes; the `uncalibrated_raw` gate is computed on uncalibrated synthetic spectra.
Covariance calibration is disabled by default because the R5 covariance calibration worsened adversarial AUC.
Fit inputs are limited to `real_X` and real wavelengths; apply inputs are limited to synthetic X and synthetic wavelengths, with real calibration interpolated to that grid when needed.
No y/target/labels/splits or source oracle inputs are used for calibration; metadata is marked `oracle=false`, `label_inputs_used=false`, `target_inputs_used=false`, `split_inputs_used=false`, and `source_oracle_used=false`.
The marginal calibration uses per-wavelength robust affine scaling, quantile mapping, and high-pass residual scaling.
Covariance metadata is still emitted with `enabled=false` and `reason=disabled_by_default_auc_regression`.
Thresholds are not changed and metric definitions are not weakened by calibration (`thresholds_modified=false`, `metrics_modified=false`).
These calibrations are intentionally strong and provisional; they must not be interpreted as proof of downstream transfer.

Marginal calibration grid strategy counts over compared rows:
- `interpolated_real_calibration_to_synthetic_grid`: 124
- `not_recorded`: 71
- `same_grid`: 18

Covariance calibration coverage over compared rows:
- enabled: 0
- disabled: 213

- Warning: Strong provisional marginal calibration for B2 diagnostics only; not a calibrated domain gate or transfer-benefit claim.

## Finite Sanitation

Audit-aware non-finite sanitation is applied before alignment, calibration, and scoring; synthetic spectra are sanitized both after generation and after marginal calibration.
No imputation is used; rows or wavelength columns containing non-finite values are dropped, and the policy is recorded per side/stage.
Sanitation requires `>=8` finite samples and `>=3` finite wavelengths; otherwise the row is blocked rather than scored.
Sanitation never modifies thresholds or metric definitions (`thresholds_modified=false`, `metrics_modified=false`, `imputed=false`).

Real-side sanitation actions:
- `drop_columns`: 6
- `drop_rows`: 12
- `no_op`: 198
- `skipped_due_to_wavelength_grid_unknown`: 3

Synthetic-side sanitation actions:
- `no_op`: 74
- `not_run`: 3
- `post_generation:no_op|post_marginal:no_op`: 142

Total real rows dropped across audited rows: 2118
Total real columns dropped across audited rows: 21

## Grid Remapping

Grid-compatible preset remapping is applied only when the original preset has fewer than three real-grid overlap points, the real wavelengths are numeric, and the dataset mapping came from stable hash fallback.
Index-fallback grids (those parsed via `np.arange` because their CSV header is not numeric) are blocked with `wavelength_grid_unknown` because their physical wavelength scale is unknown.
Semantic matrix-first matches with no supported physical wavelength overlap are blocked with `domain_wavelength_support` instead of being remapped cross-domain for wavelength compatibility.
Every remap records `original_preset`, `selected_preset`, and a `reason` token; remapping never modifies thresholds or metric definitions.

Remap reason counts:
- `grid_compatible_fallback`: 6
- `no_grid_compatible_alternative`: 3
- `no_remap_needed`: 207
- `wavelength_grid_unknown`: 3
- index-fallback rows skipped: 3

Datasets remapped to a grid-compatible preset:
- `PEACH/Brix_spxy70`: `fruit` -> `grain`
- `PLUMS/Firmness_spxy70`: `fruit` -> `grain`

## Source Overrides

Bench source overrides are emitted only from the B2 scorecard path and are recorded under `generation.source_overrides`.
Rules use dataset source/task/database/dataset tokens plus physical real wavelengths when needed; they do not read y values, labels, splits, targets, or performance metrics.
The wavelength support override remains opt-in at canonicalization time and disabled by default outside explicit B2 on-demand generation.

Source override audits recorded: 213
Source override enabled rows: 102
Wavelength support override requested rows: 6
Wavelength support override applied rows: 6

Source override rule counts:
- `beer_wine_liquid_transmittance`: 6
- `beer_wine_real_grid_support`: 6
- `corn_grain_matrix_physics`: 6
- `diesel_fuel_matrix_physics`: 9
- `instrument_token_micronir`: 9
- `instrument_token_neospectra`: 3
- `milk_dairy_matrix_physics`: 12
- `preset_juice_bench_defaults`: 15
- `preset_soil_bench_defaults`: 33
- `preset_wine_bench_defaults`: 15
- `puree_emulsion_matrix_physics`: 3

## Audit Flags

All sanitation, marginal calibration, covariance calibration, and grid remap steps record the following audit flags:

- `oracle=false`
- `label_inputs_used=false`
- `target_inputs_used=false`
- `split_inputs_used=false`
- `source_oracle_used=false`
- `replays_real_rows=false`
- `thresholds_modified=false`
- `metrics_modified=false`
- `imputed=false`

## Diagnostic Outcome

- Uncalibrated_raw adversarial AUC smoke failures: 71/71 compared rows.
- Uncalibrated_raw mean adversarial AUC: 0.9999 (lower is better; provisional smoke threshold 0.85).
- Uncalibrated_raw PCA overlap smoke failures: 68/71 compared rows.
- Uncalibrated_raw gate remains authoritative: these scorecards do not claim realism success when uncalibrated_raw adversarial AUC exceeds the smoke threshold; calibrated_raw_diagnostic and snv cannot override.

## R9 Gap Summary

R9 is a partial diagnostic improvement, not a B2 pass.
Raw adversarial AUC gaps remain: 71/71 compared rows.
Raw PCA overlap gaps remain: 68/71 compared rows.
BEER, DIESEL, and CORN rows remain named gaps when they appear below; no downstream transfer or realism-pass claim is made for them.

Named persistent gaps over raw rows:
- `BEER/Beer_OriginalExtract_60_KS`: preset `wine`, adv AUC 1, PCA overlap 0, decision `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `BEER/Beer_OriginalExtract_60_YbaseSplit`: preset `wine`, adv AUC 1, PCA overlap 0, decision `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit`: preset `grain`, adv AUC 1, PCA overlap 0, decision `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit`: preset `grain`, adv AUC 1, PCA overlap 0, decision `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `DIESEL/DIESEL_bp50_246_b-a`: preset `fuel`, adv AUC 1, PCA overlap 0, decision `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `DIESEL/DIESEL_bp50_246_hla-b`: preset `fuel`, adv AUC 1, PCA overlap 0, decision `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio`
- `DIESEL/DIESEL_bp50_246_hlb-a`: preset `fuel`, adv AUC 1, PCA overlap 0, decision `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio`

## Calibrated Raw Diagnostic

- Calibrated_raw_diagnostic adversarial AUC smoke failures: 30/71 diagnostic rows.
- Marginal calibration is applied only on the calibrated_raw_diagnostic copy; the uncalibrated_raw lane stays the authoritative gate.
- Calibrated_raw_diagnostic results cannot override an uncalibrated_raw failure and must not be interpreted as B2 success when uncalibrated_raw fails.

## SNV Diagnostic

- SNV adversarial AUC smoke failures: 43/71 diagnostic rows.
- SNV is applied only after wavelength-grid alignment and only on the calibrated comparison copies.
- SNV diagnostics do not override uncalibrated_raw decisions and must not be interpreted as B2 success when uncalibrated_raw fails.

## Metrics Table

| space | dataset | task | synthetic preset | n real | n synthetic | adv AUC | PCA overlap | NN ratio | derivative gap | decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `uncalibrated_raw` | `ALPINE/ALPINE_P_291_KS` | `regression` | `forage` | 80 | 80 | 1 | 0 | 5.098 | 1.771 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ALPINE/ALPINE_P_291_KS` | `regression` | `forage` | 80 | 80 | 0.3672 | 0.4125 | 1.471 | 0.886 | `provisional_pass` |
| `snv` | `ALPINE/ALPINE_P_291_KS` | `regression` | `forage` | 80 | 80 | 0.7625 | 0.2625 | 2.086 | 0.8693 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `AMYLOSE/Rice_Amylose_313_YbasedSplit` | `regression` | `grain` | 80 | 80 | 1 | 0 | 39.87 | 1.802 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `AMYLOSE/Rice_Amylose_313_YbasedSplit` | `regression` | `grain` | 80 | 80 | 1 | 0.4 | 2.251 | 0.3582 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `AMYLOSE/Rice_Amylose_313_YbasedSplit` | `regression` | `grain` | 80 | 80 | 1 | 0.1625 | 4.429 | 0.3494 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `BEEFMARBLING/Beef_Marbling_RandomSplit` | `regression` | `meat` | 80 | 80 | 1 | 0 | 49.08 | 0.008901 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BEEFMARBLING/Beef_Marbling_RandomSplit` | `regression` | `meat` | 80 | 80 | 0.2609 | 0.4625 | 1.512 | 0.05393 | `provisional_pass` |
| `snv` | `BEEFMARBLING/Beef_Marbling_RandomSplit` | `regression` | `meat` | 80 | 80 | 0.7016 | 0.575 | 2.776 | 0.0517 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `BEER/Beer_OriginalExtract_60_KS` | `regression` | `wine` | 60 | 80 | 1 | 0 | 149.8 | 0.5038 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BEER/Beer_OriginalExtract_60_KS` | `regression` | `wine` | 60 | 80 | 1 | 0 | 1.12 | 0.06188 | `provisional_review:pca_overlap,adversarial_auc` |
| `snv` | `BEER/Beer_OriginalExtract_60_KS` | `regression` | `wine` | 60 | 80 | 1 | 0 | 1.116 | 0.06655 | `provisional_review:pca_overlap,adversarial_auc` |
| `uncalibrated_raw` | `BEER/Beer_OriginalExtract_60_YbaseSplit` | `regression` | `wine` | 60 | 80 | 1 | 0 | 147.2 | 0.5239 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BEER/Beer_OriginalExtract_60_YbaseSplit` | `regression` | `wine` | 60 | 80 | 1 | 0 | 1.118 | 0.06139 | `provisional_review:pca_overlap,adversarial_auc` |
| `snv` | `BEER/Beer_OriginalExtract_60_YbaseSplit` | `regression` | `wine` | 60 | 80 | 1 | 0 | 1.111 | 0.06498 | `provisional_review:pca_overlap,adversarial_auc` |
| `uncalibrated_raw` | `BERRY/brix_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 1 | 0 | 88.43 | 4.124 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BERRY/brix_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.3047 | 0.4625 | 2.506 | 0.1431 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `BERRY/brix_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.932 | 0.3875 | 1.625 | 0.5461 | `provisional_review:adversarial_auc` |
| `uncalibrated_raw` | `BERRY/ph_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 1 | 0 | 35.74 | 4.098 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BERRY/ph_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.3281 | 0.3625 | 1.579 | 0.2194 | `provisional_pass` |
| `snv` | `BERRY/ph_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.9453 | 0.3 | 1.695 | 0.5802 | `provisional_review:adversarial_auc` |
| `uncalibrated_raw` | `BERRY/ta_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 1 | 0 | 85.14 | 4.15 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BERRY/ta_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.2891 | 0.4 | 2.689 | 0.4503 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `BERRY/ta_groupSampleID_stratDateVar_balRows` | `regression` | `juice` | 80 | 80 | 0.9508 | 0.2875 | 1.596 | 0.6421 | `provisional_review:adversarial_auc` |
| `uncalibrated_raw` | `BISCUIT/Biscuit_Fat_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 1 | 0 | 51.58 | 0.2832 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BISCUIT/Biscuit_Fat_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 0.8689 | 0.3556 | 2.106 | 0.09334 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `BISCUIT/Biscuit_Fat_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 0.9407 | 0.2306 | 2.856 | 0.09053 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `BISCUIT/Biscuit_Sucrose_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 1 | 0 | 49.52 | 0.2208 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BISCUIT/Biscuit_Sucrose_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 0.9119 | 0.3653 | 2.03 | 0.09667 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `BISCUIT/Biscuit_Sucrose_40_RandomSplit` | `regression` | `baking` | 72 | 80 | 0.9408 | 0.2431 | 2.479 | 0.06616 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `COLZA/C_woOutlier` | `regression` | `oilseeds` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `COLZA/N_wOutlier` | `regression` | `oilseeds` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `COLZA/N_woOutlier` | `regression` | `oilseeds` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 1 | 0 | 33.56 | 1.054 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 0.9992 | 0.3625 | 2.472 | 0.08932 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 1 | 0.125 | 3.428 | 0.08391 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 1 | 0 | 37.34 | 1.014 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 0.9984 | 0.4375 | 2.45 | 0.09868 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 80 | 80 | 1 | 0.175 | 3.595 | 0.08878 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DIESEL/DIESEL_bp50_246_b-a` | `regression` | `fuel` | 80 | 80 | 1 | 0 | 449.8 | 1.501 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DIESEL/DIESEL_bp50_246_b-a` | `regression` | `fuel` | 80 | 80 | 1 | 0.2 | 2.111 | 0.04233 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `DIESEL/DIESEL_bp50_246_b-a` | `regression` | `fuel` | 80 | 80 | 1 | 0.25 | 2.426 | 0.02769 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DIESEL/DIESEL_bp50_246_hla-b` | `regression` | `fuel` | 80 | 80 | 1 | 0 | 389 | 1.535 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DIESEL/DIESEL_bp50_246_hla-b` | `regression` | `fuel` | 80 | 80 | 1 | 0.35 | 2.089 | 0.0472 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `DIESEL/DIESEL_bp50_246_hla-b` | `regression` | `fuel` | 80 | 80 | 1 | 0.275 | 2.293 | 0.03222 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DIESEL/DIESEL_bp50_246_hlb-a` | `regression` | `fuel` | 80 | 80 | 1 | 0 | 416 | 1.53 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DIESEL/DIESEL_bp50_246_hlb-a` | `regression` | `fuel` | 80 | 80 | 1 | 0.2 | 2.004 | 0.0427 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `DIESEL/DIESEL_bp50_246_hlb-a` | `regression` | `fuel` | 80 | 80 | 1 | 0.1625 | 2.302 | 0.03044 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DarkResp/Rd25_CBtestSite` | `regression` | `forage` | 80 | 80 | 1 | 0 | 31.16 | 1.479 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DarkResp/Rd25_CBtestSite` | `regression` | `forage` | 80 | 80 | 0.8742 | 0.3875 | 1.936 | 0.2637 | `provisional_review:adversarial_auc` |
| `snv` | `DarkResp/Rd25_CBtestSite` | `regression` | `forage` | 80 | 80 | 0.9 | 0.275 | 3.538 | 0.2563 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DarkResp/Rd25_GTtestSite` | `regression` | `forage` | 80 | 80 | 1 | 0 | 31.63 | 1.489 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DarkResp/Rd25_GTtestSite` | `regression` | `forage` | 80 | 80 | 1 | 0.4 | 2.016 | 0.241 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `DarkResp/Rd25_GTtestSite` | `regression` | `forage` | 80 | 80 | 0.9875 | 0.2625 | 3.465 | 0.2289 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DarkResp/Rd25_XSBNtestSite` | `regression` | `forage` | 80 | 80 | 1 | 0 | 28.73 | 1.43 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DarkResp/Rd25_XSBNtestSite` | `regression` | `forage` | 80 | 80 | 0.8828 | 0.3875 | 1.946 | 0.2712 | `provisional_review:adversarial_auc` |
| `snv` | `DarkResp/Rd25_XSBNtestSite` | `regression` | `forage` | 80 | 80 | 0.9219 | 0.275 | 3.511 | 0.2616 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `DarkResp/Rd25_spxy70` | `regression` | `forage` | 80 | 80 | 1 | 0 | 28.85 | 1.502 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `DarkResp/Rd25_spxy70` | `regression` | `forage` | 80 | 80 | 0.6602 | 0.4 | 1.878 | 0.2317 | `provisional_pass` |
| `snv` | `DarkResp/Rd25_spxy70` | `regression` | `forage` | 80 | 80 | 0.9109 | 0.2625 | 3.353 | 0.2287 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ECOSIS_LeafTraits/Ccar_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 1 | 0 | 14.42 | 1.186 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ECOSIS_LeafTraits/Ccar_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.6195 | 0.5 | 1.373 | 0.01619 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Ccar_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.5602 | 0.3375 | 2.114 | 0.03173 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 1 | 0 | 16.16 | 1.174 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.793 | 0.525 | 1.732 | 0.01962 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.7891 | 0.3125 | 2.395 | 0.005234 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ECOSIS_LeafTraits/Chla+b_spxyG_species` | `regression` | `forage` | 80 | 80 | 1 | 0 | 14.59 | 1.133 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ECOSIS_LeafTraits/Chla+b_spxyG_species` | `regression` | `forage` | 80 | 80 | 0.6594 | 0.5 | 1.44 | 0.0007714 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Chla+b_spxyG_species` | `regression` | `forage` | 80 | 80 | 0.575 | 0.375 | 2.071 | 0.02222 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ECOSIS_LeafTraits/LMA_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 1 | 0 | 13.74 | 1.118 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ECOSIS_LeafTraits/LMA_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.8102 | 0.5875 | 1.688 | 0.00905 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/LMA_spxyG_block2deg` | `regression` | `forage` | 80 | 80 | 0.8086 | 0.5125 | 2.527 | 0.0063 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FUSARIUM/FinalScore_grp70_30_scoreQ` | `regression` | `juice` | 80 | 80 | 1 | 0 | 143.5 | 4.417 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FUSARIUM/FinalScore_grp70_30_scoreQ` | `regression` | `juice` | 80 | 80 | 1 | 0.325 | 2.067 | 0.395 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `FUSARIUM/FinalScore_grp70_30_scoreQ` | `regression` | `juice` | 80 | 80 | 1 | 0.325 | 2.665 | 0.06449 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FUSARIUM/Fv_Fm_grp70_30` | `regression` | `dairy` | 80 | 80 | 1 | 0 | 33.15 | 2.874 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FUSARIUM/Fv_Fm_grp70_30` | `regression` | `dairy` | 80 | 80 | 0.4977 | 0.3875 | 1.427 | 0.1704 | `provisional_pass` |
| `snv` | `FUSARIUM/Fv_Fm_grp70_30` | `regression` | `dairy` | 80 | 80 | 0.3484 | 0.3875 | 2.11 | 0.1698 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FUSARIUM/Tleaf_grp70_30` | `regression` | `forage` | 80 | 80 | 1 | 0 | 27.67 | 2.796 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FUSARIUM/Tleaf_grp70_30` | `regression` | `forage` | 80 | 80 | 0.9586 | 0.4375 | 1.34 | 0.1247 | `provisional_review:adversarial_auc` |
| `snv` | `FUSARIUM/Tleaf_grp70_30` | `regression` | `forage` | 80 | 80 | 0.7047 | 0.3625 | 2.402 | 0.1258 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINES/grapevine_chloride_556_KS` | `regression` | `fruit` | 80 | 80 | 1 | 0 | 55.41 | 4.749 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINES/grapevine_chloride_556_KS` | `regression` | `fruit` | 80 | 80 | 1 | 0.5125 | 1.914 | 0.03812 | `provisional_review:adversarial_auc` |
| `snv` | `GRAPEVINES/grapevine_chloride_556_KS` | `regression` | `fruit` | 80 | 80 | 1 | 0.25 | 2.218 | 0.02906 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 1 | 0 | 17.18 | 1.489 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 0.5039 | 0.425 | 1.322 | 0.4521 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 0.5359 | 0.3375 | 1.973 | 0.4326 | `provisional_pass` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR` | `regression` | `forage` | 80 | 80 | 1 | 0 | 68.37 | 0.4436 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR` | `regression` | `forage` | 80 | 80 | 0.9516 | 0.3625 | 1.943 | 0.02024 | `provisional_review:adversarial_auc` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR` | `regression` | `forage` | 80 | 80 | 1 | 0.225 | 3.701 | 0.00871 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 1 | 0 | 63.03 | 0.4901 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 0.3742 | 0.4 | 1.805 | 0.01466 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 0.7125 | 0.25 | 3.131 | 0.02187 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` | `regression` | `forage` | 80 | 80 | 1 | 0 | 28.33 | 0.684 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` | `regression` | `forage` | 80 | 80 | 1 | 0.325 | 2.13 | 0.01305 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` | `regression` | `forage` | 80 | 80 | 1 | 0.25 | 3.071 | 0.01781 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 1 | 0 | 16.08 | 1.467 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 0.5055 | 0.375 | 1.496 | 0.3991 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD` | `regression` | `forage` | 80 | 80 | 0.6711 | 0.35 | 2.017 | 0.3722 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 1 | 0 | 69.98 | 0.4619 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 0.375 | 0.3875 | 1.762 | 0.01654 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `forage` | 80 | 80 | 0.5742 | 0.175 | 3.767 | 0.0134 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `IncombustibleMaterial/TIC_spxy70` | `regression` | `soil` | 62 | 80 | 0.9958 | 0 | 8.912 | 1.943 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `IncombustibleMaterial/TIC_spxy70` | `regression` | `soil` | 62 | 80 | 0.9942 | 0.2448 | 1.533 | 0.3802 | `provisional_review:adversarial_auc` |
| `snv` | `IncombustibleMaterial/TIC_spxy70` | `regression` | `soil` | 62 | 80 | 0.9659 | 0.1948 | 2.424 | 0.235 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS` | `regression` | `soil` | 80 | 80 | 1 | 0.0125 | 12.35 | 2.38 | `provisional_review:derivative_gap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS` | `regression` | `soil` | 80 | 80 | 0.4648 | 0.375 | 2.458 | 1.193 | `provisional_review:derivative_gap,nearest_neighbor_ratio` |
| `snv` | `LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS` | `regression` | `soil` | 80 | 80 | 0.95 | 0.15 | 3.623 | 0.5748 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `LUCAS/LUCAS_SOC_all_26650_NocitaKS` | `regression` | `soil` | 80 | 80 | 1 | 0.0125 | 12.55 | 2.37 | `provisional_review:derivative_gap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `LUCAS/LUCAS_SOC_all_26650_NocitaKS` | `regression` | `soil` | 80 | 80 | 0.3 | 0.2625 | 2.592 | 1.465 | `provisional_review:derivative_gap,nearest_neighbor_ratio` |
| `snv` | `LUCAS/LUCAS_SOC_all_26650_NocitaKS` | `regression` | `soil` | 80 | 80 | 0.9688 | 0.1875 | 4.221 | 0.7031 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic` | `regression` | `soil` | 80 | 80 | 1 | 0 | 14.6 | 2.139 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic` | `regression` | `soil` | 80 | 80 | 0.3422 | 0.45 | 2.401 | 1.27 | `provisional_review:derivative_gap,nearest_neighbor_ratio` |
| `snv` | `LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic` | `regression` | `soil` | 80 | 80 | 0.9609 | 0.175 | 3.443 | 0.6753 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MALARIA/Malaria_Oocist_333_Maia` | `regression` | `soil` | 80 | 80 | 1 | 0 | 17.53 | 0.911 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MALARIA/Malaria_Oocist_333_Maia` | `regression` | `soil` | 80 | 80 | 0.643 | 0.45 | 2.269 | 0.4762 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MALARIA/Malaria_Oocist_333_Maia` | `regression` | `soil` | 80 | 80 | 0.9578 | 0.1 | 2.65 | 0.3902 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MALARIA/Malaria_Sporozoite_229_Maia` | `regression` | `meat` | 80 | 80 | 1 | 0 | 15.21 | 1.089 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MALARIA/Malaria_Sporozoite_229_Maia` | `regression` | `meat` | 80 | 80 | 0.3477 | 0.4 | 1.942 | 0.6187 | `provisional_pass` |
| `snv` | `MALARIA/Malaria_Sporozoite_229_Maia` | `regression` | `meat` | 80 | 80 | 0.7633 | 0.3375 | 3.573 | 0.4396 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MANURE21/All_manure_CaO_SPXY_strat_Manure_type` | `regression` | `grain` | 80 | 80 | 1 | 0 | 32.39 | 0.8557 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MANURE21/All_manure_CaO_SPXY_strat_Manure_type` | `regression` | `grain` | 80 | 80 | 0.4383 | 0.3875 | 2.028 | 0.2992 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MANURE21/All_manure_CaO_SPXY_strat_Manure_type` | `regression` | `grain` | 80 | 80 | 0.9094 | 0.225 | 2.664 | 0.3044 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MANURE21/All_manure_K2O_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 1 | 0 | 25.65 | 0.742 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MANURE21/All_manure_K2O_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.3406 | 0.375 | 1.818 | 0.5267 | `provisional_pass` |
| `snv` | `MANURE21/All_manure_K2O_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.8695 | 0.5125 | 2.574 | 0.5252 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 1 | 0 | 22.4 | 0.8085 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.3344 | 0.4625 | 1.867 | 0.5037 | `provisional_pass` |
| `snv` | `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.7883 | 0.3875 | 3.665 | 0.5255 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MANURE21/All_manure_P2O5_SPXY_strat_Manure_type` | `regression` | `baking` | 80 | 80 | 1 | 0 | 26.72 | 0.7126 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MANURE21/All_manure_P2O5_SPXY_strat_Manure_type` | `regression` | `baking` | 80 | 80 | 0.3461 | 0.5125 | 2.407 | 0.3531 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MANURE21/All_manure_P2O5_SPXY_strat_Manure_type` | `regression` | `baking` | 80 | 80 | 0.9695 | 0.225 | 3.319 | 0.2953 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 1 | 0 | 23.2 | 0.794 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.4633 | 0.475 | 1.426 | 0.2661 | `provisional_pass` |
| `snv` | `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` | `regression` | `wine` | 80 | 80 | 0.7734 | 0.525 | 2.682 | 0.2438 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MILK/Milk_Fat_1224_KS` | `regression` | `dairy` | 80 | 80 | 1 | 0 | 43.44 | 1.44 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MILK/Milk_Fat_1224_KS` | `regression` | `dairy` | 80 | 80 | 1 | 0.375 | 2.101 | 0.001115 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `MILK/Milk_Fat_1224_KS` | `regression` | `dairy` | 80 | 80 | 0.8812 | 0.25 | 3.492 | 0.0212 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MILK/Milk_Lactose_1224_KS` | `regression` | `dairy` | 80 | 80 | 1 | 0 | 45.14 | 1.368 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MILK/Milk_Lactose_1224_KS` | `regression` | `dairy` | 80 | 80 | 0.9984 | 0.4 | 1.928 | 0.005083 | `provisional_review:adversarial_auc` |
| `snv` | `MILK/Milk_Lactose_1224_KS` | `regression` | `dairy` | 80 | 80 | 0.8492 | 0.275 | 3.29 | 0.01705 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MILK/Milk_Urea_1224_KS` | `regression` | `dairy` | 80 | 80 | 1 | 0 | 44.35 | 1.403 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MILK/Milk_Urea_1224_KS` | `regression` | `dairy` | 80 | 80 | 0.9758 | 0.5125 | 1.84 | 0.02504 | `provisional_review:adversarial_auc` |
| `snv` | `MILK/Milk_Urea_1224_KS` | `regression` | `dairy` | 80 | 80 | 0.5242 | 0.375 | 3.271 | 0.02477 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PEACH/Brix_spxy70` | `regression` | `grain` | 50 | 80 | 1 | 0 | 28.76 | 0.1896 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PEACH/Brix_spxy70` | `regression` | `grain` | 50 | 80 | 1 | 0.2275 | 2.257 | 0.0003654 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `PEACH/Brix_spxy70` | `regression` | `grain` | 50 | 80 | 1 | 0.235 | 1.886 | 0.0004784 | `provisional_review:adversarial_auc` |
| `uncalibrated_raw` | `PHOSPHORUS/LP_spxyG` | `regression` | `soil` | 80 | 80 | 1 | 0 | 26.07 | 1.315 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PHOSPHORUS/LP_spxyG` | `regression` | `soil` | 80 | 80 | 0.9172 | 0.4125 | 1.906 | 0.2368 | `provisional_review:adversarial_auc` |
| `snv` | `PHOSPHORUS/LP_spxyG` | `regression` | `soil` | 80 | 80 | 0.8508 | 0.1875 | 3.166 | 0.2225 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PHOSPHORUS/MP_spxyG` | `regression` | `soil` | 80 | 80 | 1 | 0 | 28.08 | 1.329 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PHOSPHORUS/MP_spxyG` | `regression` | `soil` | 80 | 80 | 0.8469 | 0.4375 | 1.763 | 0.2493 | `provisional_pass` |
| `snv` | `PHOSPHORUS/MP_spxyG` | `regression` | `soil` | 80 | 80 | 0.5797 | 0.25 | 3.269 | 0.2493 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PHOSPHORUS/NP_spxyG` | `regression` | `soil` | 80 | 80 | 1 | 0 | 28.21 | 1.317 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PHOSPHORUS/NP_spxyG` | `regression` | `soil` | 80 | 80 | 0.9281 | 0.325 | 1.939 | 0.2385 | `provisional_review:adversarial_auc` |
| `snv` | `PHOSPHORUS/NP_spxyG` | `regression` | `soil` | 80 | 80 | 0.8836 | 0.2 | 3.207 | 0.2446 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PHOSPHORUS/Pi_spxyG` | `regression` | `soil` | 80 | 80 | 1 | 0 | 29.07 | 1.366 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PHOSPHORUS/Pi_spxyG` | `regression` | `soil` | 80 | 80 | 0.7523 | 0.3375 | 2.223 | 0.2578 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `PHOSPHORUS/Pi_spxyG` | `regression` | `soil` | 80 | 80 | 0.8766 | 0.1375 | 3.741 | 0.2551 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PHOSPHORUS/V25_spxyG` | `regression` | `soil` | 80 | 80 | 1 | 0 | 25.96 | 1.344 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PHOSPHORUS/V25_spxyG` | `regression` | `soil` | 80 | 80 | 0.8422 | 0.4 | 1.632 | 0.236 | `provisional_pass` |
| `snv` | `PHOSPHORUS/V25_spxyG` | `regression` | `soil` | 80 | 80 | 0.8 | 0.2125 | 2.8 | 0.2077 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PLUMS/Firmness_spxy70` | `regression` | `grain` | 40 | 80 | 1 | 0 | 24.73 | 1.077 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PLUMS/Firmness_spxy70` | `regression` | `grain` | 40 | 80 | 0.7328 | 0.2375 | 1.338 | 0.0685 | `provisional_pass` |
| `snv` | `PLUMS/Firmness_spxy70` | `regression` | `grain` | 40 | 80 | 0.9953 | 0.1375 | 2.681 | 0.08195 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `QUARTZ/Quartz_spxy70` | `regression` | `soil` | 80 | 80 | 1 | 0 | 62.9 | 5.197 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `QUARTZ/Quartz_spxy70` | `regression` | `soil` | 80 | 80 | 1 | 0.125 | 8.275 | 0.2992 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `QUARTZ/Quartz_spxy70` | `regression` | `soil` | 80 | 80 | 0.9859 | 0.0125 | 14.79 | 0.1612 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `TABLET/Escitalopramt_310_Zhao` | `regression` | `tablets` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `WOOD_density/WOOD_Density_402_Olale` | `regression` | `fruit` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `WOOD_density/WOOD_N_402_Olale` | `regression` | `meat` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `uncalibrated_raw` | `ARABIDOPSIS_CEFE/Genotype10_250` | `classification` | `forage` | 80 | 80 | 1 | 0 | 31.09 | 0.9062 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ARABIDOPSIS_CEFE/Genotype10_250` | `classification` | `forage` | 80 | 80 | 0.5984 | 0.4375 | 1.661 | 0.2012 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/Genotype10_250` | `classification` | `forage` | 80 | 80 | 0.4062 | 0.2 | 3.509 | 0.2069 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ARABIDOPSIS_CEFE/Group9_1856` | `classification` | `forage` | 80 | 80 | 1 | 0 | 30.31 | 0.8884 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ARABIDOPSIS_CEFE/Group9_1856` | `classification` | `forage` | 80 | 80 | 0.6469 | 0.4375 | 1.635 | 0.2316 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/Group9_1856` | `classification` | `forage` | 80 | 80 | 0.4344 | 0.1375 | 3.295 | 0.2237 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ARABIDOPSIS_CEFE/Group_2185` | `classification` | `forage` | 80 | 80 | 1 | 0 | 26.19 | 0.8892 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ARABIDOPSIS_CEFE/Group_2185` | `classification` | `forage` | 80 | 80 | 0.2977 | 0.4375 | 1.574 | 0.2246 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/Group_2185` | `classification` | `forage` | 80 | 80 | 0.2344 | 0.325 | 3.153 | 0.2084 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `ARABIDOPSIS_CEFE/InOut_1264` | `classification` | `forage` | 80 | 80 | 1 | 0 | 19.14 | 0.9413 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `ARABIDOPSIS_CEFE/InOut_1264` | `classification` | `forage` | 80 | 80 | 0.4656 | 0.7625 | 1.711 | 0.3805 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/InOut_1264` | `classification` | `forage` | 80 | 80 | 0.3937 | 0.2125 | 2.948 | 0.3714 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `BEEF_Impurity/Beef_Impurity_60_AlJowder` | `classification` | `meat` | 60 | 80 | 1 | 0 | 4.863 | 1.721 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `BEEF_Impurity/Beef_Impurity_60_AlJowder` | `classification` | `meat` | 60 | 80 | 0.2521 | 0.2208 | 2.276 | 0.09092 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `BEEF_Impurity/Beef_Impurity_60_AlJowder` | `classification` | `meat` | 60 | 80 | 0.5833 | 0.2917 | 1.954 | 0.1376 | `provisional_pass` |
| `uncalibrated_raw` | `COFFEE_orig/CoffeeType_kenstone70_strat` | `classification` | `fuel` | 70 | 80 | 1 | 0 | 56.81 | 1.327 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `COFFEE_orig/CoffeeType_kenstone70_strat` | `classification` | `fuel` | 70 | 80 | 0.3152 | 0.1429 | 4.021 | 0.6534 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `COFFEE_orig/CoffeeType_kenstone70_strat` | `classification` | `fuel` | 70 | 80 | 0.9884 | 0.0375 | 18.49 | 0.5199 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `Cassava/CT2C_1057_CIAT_Acc` | `classification` | `grain` | 80 | 80 | 1 | 0 | 103.4 | 0.1456 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `Cassava/CT2C_1057_CIAT_Acc` | `classification` | `grain` | 80 | 80 | 0.9453 | 0.25 | 2.115 | 0.08896 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `Cassava/CT2C_1057_CIAT_Acc` | `classification` | `grain` | 80 | 80 | 0.6281 | 0.1375 | 3.472 | 0.08852 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FUSARIUM/FinalScoreBin_grp70_30_classStrat` | `classification` | `fruit` | 80 | 80 | 1 | 0 | 67.69 | 4.62 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FUSARIUM/FinalScoreBin_grp70_30_classStrat` | `classification` | `fruit` | 80 | 80 | 1 | 0.4125 | 1.614 | 0.317 | `provisional_review:adversarial_auc` |
| `snv` | `FUSARIUM/FinalScoreBin_grp70_30_classStrat` | `classification` | `fruit` | 80 | 80 | 1 | 0.3125 | 2.277 | 0.3352 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FUSARIUM/ScoreBin_grp70_30_classStrat` | `classification` | `oilseeds` | 80 | 80 | 1 | 0 | 30.85 | 2.768 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FUSARIUM/ScoreBin_grp70_30_classStrat` | `classification` | `oilseeds` | 80 | 80 | 0.95 | 0.3375 | 1.578 | 0.148 | `provisional_review:adversarial_auc` |
| `snv` | `FUSARIUM/ScoreBin_grp70_30_classStrat` | `classification` | `oilseeds` | 80 | 80 | 0.9367 | 0.325 | 2.389 | 0.1589 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `FruitPuree/Strawberry2C_983_Holland_Acc94.3` | `classification` | `juice` | 80 | 80 | 0.9938 | 0 | 39.54 | 1.744 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `FruitPuree/Strawberry2C_983_Holland_Acc94.3` | `classification` | `juice` | 80 | 80 | 0.9773 | 0.4125 | 1.814 | 0.08322 | `provisional_review:adversarial_auc` |
| `snv` | `FruitPuree/Strawberry2C_983_Holland_Acc94.3` | `classification` | `juice` | 80 | 80 | 0.9945 | 0.5125 | 2.355 | 0.07263 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MALARIA/Oocist2C_333_Maia_Acc87.6` | `classification` | `powders` | 80 | 80 | 1 | 0 | 19.43 | 1.038 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MALARIA/Oocist2C_333_Maia_Acc87.6` | `classification` | `powders` | 80 | 80 | 0.5695 | 0.3875 | 2.881 | 0.403 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MALARIA/Oocist2C_333_Maia_Acc87.6` | `classification` | `powders` | 80 | 80 | 1 | 0.1625 | 3.803 | 0.4191 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MALARIA/Sporozoite2C_229_Maia_Acc94.5` | `classification` | `oilseeds` | 80 | 80 | 1 | 0 | 13.51 | 1.256 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MALARIA/Sporozoite2C_229_Maia_Acc94.5` | `classification` | `oilseeds` | 80 | 80 | 0.5289 | 0.425 | 1.726 | 0.6695 | `provisional_pass` |
| `snv` | `MALARIA/Sporozoite2C_229_Maia_Acc94.5` | `classification` | `oilseeds` | 80 | 80 | 0.9594 | 0.175 | 2.315 | 0.5645 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `MILK/labels_kenstone70_strat` | `classification` | `dairy` | 80 | 80 | 1 | 0 | 163.8 | 0.06057 | `provisional_review:pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `MILK/labels_kenstone70_strat` | `classification` | `dairy` | 80 | 80 | 0.7547 | 0.3375 | 2.522 | 0.001578 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MILK/labels_kenstone70_strat` | `classification` | `dairy` | 80 | 80 | 0.8984 | 0.175 | 2.613 | 0.001349 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `uncalibrated_raw` | `PISTACIA/Species_code_grpStrat70_30_bySpecimen` | `classification` | `fruit` | 80 | 80 | 1 | 0.025 | 12.69 | 1.006 | `provisional_review:derivative_gap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `PISTACIA/Species_code_grpStrat70_30_bySpecimen` | `classification` | `fruit` | 80 | 80 | 0.2891 | 0.2875 | 1.611 | 0.5207 | `provisional_pass` |
| `snv` | `PISTACIA/Species_code_grpStrat70_30_bySpecimen` | `classification` | `fruit` | 80 | 80 | 0.6547 | 0.2625 | 3.187 | 0.5584 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `Wood_Sustainability/C2_511_Davrieux_Acc82` | `classification` | `grain` | 80 | 80 | 1 | 0 | 25.88 | 1.349 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `Wood_Sustainability/C2_511_Davrieux_Acc82` | `classification` | `grain` | 80 | 80 | 0.55 | 0.2375 | 1.877 | 0.3328 | `provisional_pass` |
| `snv` | `Wood_Sustainability/C2_511_Davrieux_Acc82` | `classification` | `grain` | 80 | 80 | 0.8477 | 0.275 | 3.644 | 0.2787 | `provisional_review:nearest_neighbor_ratio` |
| `uncalibrated_raw` | `Wood_Sustainability/C5_511_Davrieux_Acc82` | `classification` | `meat` | 80 | 80 | 1 | 0 | 18.43 | 1.163 | `provisional_review:derivative_gap,pca_overlap,adversarial_auc,nearest_neighbor_ratio` |
| `calibrated_raw_diagnostic` | `Wood_Sustainability/C5_511_Davrieux_Acc82` | `classification` | `meat` | 80 | 80 | 0.2234 | 0.3375 | 2.181 | 0.8454 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `Wood_Sustainability/C5_511_Davrieux_Acc82` | `classification` | `meat` | 80 | 80 | 0.5813 | 0.2 | 4.281 | 0.8386 | `provisional_review:nearest_neighbor_ratio` |

## Metric Route

- Spectral mean/variance: reported as mean profile averages for real and synthetic spectra.
- Derivative mean/variance: first derivative per sample, summarized by median.
- Correlation length: first autocorrelation lag below 1/e, summarized by median.
- SNR: high-pass residual estimate from `nirs4all.synthesis.validation.compute_snr`.
- Baseline curvature: degree-3 polynomial residual standard deviation.
- Peak density: peaks per 100 nm using the shared validation helper.
- PCA overlap: 2D PCA histogram intersection when sklearn and sample counts permit.
- Adversarial AUC: PCA plus logistic real/synthetic classifier when sample counts permit.
- Nearest-neighbor ratio: real-to-synthetic nearest distance divided by real-to-real nearest distance.

## Provisional Thresholds

| metric | threshold | interpretation |
|---|---:|---|
| adversarial AUC smoke | 0.85 | lower is better |
| adversarial AUC stretch | 0.7 | stronger future target |
| derivative log10 gap | 1.0 | no order-of-magnitude gap |
| PCA overlap min | 0.01 | non-empty overlap |
| NN ratio max | 2.0 | synthetic not much farther than real neighbors |

## Load Failures

- `AOM_regression` `COLZA/C_woOutlier` [wavelength_grid_unknown]: wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring
- `AOM_regression` `COLZA/N_wOutlier` [wavelength_grid_unknown]: wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring
- `AOM_regression` `COLZA/N_woOutlier` [wavelength_grid_unknown]: wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring
- `AOM_regression` `TABLET/Escitalopramt_310_Zhao` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `WOOD_density/WOOD_Density_402_Olale` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `WOOD_density/WOOD_N_402_Olale` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points

## Decision

B2 scorecard route is runnable, but realism smoke success is not established: raw adversarial AUC failed for all raw compared rows, so synthetic spectra are trivially separable. Blocked selected rows retained in CSV: 6.

## Raw Summary JSON

```json
{
  "git_status": {
    "line_count": 65,
    "lines": [
      " M bench/AOM_v0/Ridge/aomridge/estimators.py",
      " M bench/AOM_v0/Ridge/aomridge/kernels.py",
      " M bench/AOM_v0/Ridge/aomridge/selection.py",
      " M bench/AOM_v0/Ridge/benchmark_runs/smoke/results.csv",
      " M bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py",
      " M bench/AOM_v0/Ridge/docs/IMPLEMENTATION_LOG.md",
      " M bench/AOM_v0/Ridge/tests/test_ridge_cv_no_leakage.py",
      " M bench/AOM_v0/aompls/estimators.py",
      " M bench/AOM_v0/aompls/preprocessing.py",
      " M bench/AOM_v0/aompls/scorers.py",
      " M bench/AOM_v0/aompls/selection.py",
      " M bench/AOM_v0/benchmark_runs/full/results.csv",
      " M bench/AOM_v0/benchmarks/run_aompls_benchmark.py",
      " M bench/AOM_v0/benchmarks/run_extended_benchmark.py",
      " M bench/AOM_v0/publication/tables/relative_rmsep_per_variant.csv",
      " M bench/nirs_synthetic_pfn/experiments/exp00_smoke_prior_dataset.py",
      " M bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py",
      " M bench/nirs_synthetic_pfn/experiments/exp03_transfer_validation.py",
      " M bench/nirs_synthetic_pfn/reports/integration_gate_status.md",
      " M bench/nirs_synthetic_pfn/reports/nirs_context_query_sampler_contract.md",
      " M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv",
      " M bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md",
      " M bench/nirs_synthetic_pfn/reports/transfer_validation.csv",
      " M bench/nirs_synthetic_pfn/reports/transfer_validation.md",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/adapters/prior_adapter.py",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/realism.py",
      " M bench/nirs_synthetic_pfn/src/nirsyntheticpfn/evaluation/transfer.py",
      " M bench/nirs_synthetic_pfn/tests/test_prior_adapter.py",
      " M bench/nirs_synthetic_pfn/tests/test_realism_scorecards.py",
      " M bench/nirs_synthetic_pfn/tests/test_transfer_validation.py",
      "?? .claude/",
      "?? .codex",
      "?? bench/AOM_v0/Ridge/aomridge/branches.py",
      "?? bench/AOM_v0/Ridge/aomridge/cv.py",
      "?? bench/AOM_v0/Ridge/aomridge/mkl.py",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated_cohort.csv",
      "?? bench/AOM_v0/Ridge/benchmark_runs/curated_v2/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/smoke6/",
      "?? bench/AOM_v0/Ridge/benchmark_runs/smoke_cv5/",
      "?? bench/AOM_v0/Ridge/docs/CODEX_BACKLOG_round2_2026-04-29.md",
      "?? bench/AOM_v0/Ridge/publication/",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_branch_global.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_mkl.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_one_se_and_repeated_cv.py",
      "?? bench/AOM_v0/Ridge/tests/test_ridge_round3_fixes.py",
      "?? bench/AOM_v0/docs/CV_SPLITTER_DESIGN.md",
      "?? bench/AOM_v0/publication/tables/table_top15_score_time.tex",
      "?? bench/nirs_synthetic_pfn/docs/06_SYNTHETIC_REALISM_REMEDIATION_ROADMAP.md",
      "?? bench/nirs_synthetic_pfn/experiments/exp04_adversarial_auc.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp05_minimal_ablation_attribution.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp06_encoder_tabpfn_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/experiments/exp07_nirs_icl_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/reports/adversarial_auc.csv",
      "?? bench/nirs_synthetic_pfn/reports/adversarial_auc.md",
      "?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.csv",
      "?? bench/nirs_synthetic_pfn/reports/encoder_tabpfn_gate.md",
      "?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.csv",
      "?? bench/nirs_synthetic_pfn/reports/minimal_ablation_attribution.md",
      "?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.csv",
      "?? bench/nirs_synthetic_pfn/reports/nirs_icl_gate_precheck.md",
      "?? bench/nirs_synthetic_pfn/tests/test_adversarial_auc_report.py",
      "?? bench/nirs_synthetic_pfn/tests/test_encoder_tabpfn_gate_precheck.py",
      "?? bench/nirs_synthetic_pfn/tests/test_minimal_ablation_attribution.py",
      "?? bench/nirs_synthetic_pfn/tests/test_nirs_icl_gate_precheck.py"
    ],
    "returncode": 0,
    "truncated": false
  },
  "inventories": [
    {
      "exists": true,
      "missing_paths": [],
      "missing_rows": 0,
      "ok_rows": 61,
      "path": "/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_regression.csv",
      "runnable_rows": 61,
      "source": "AOM_regression",
      "total_rows": 61
    },
    {
      "exists": true,
      "missing_paths": [],
      "missing_rows": 0,
      "ok_rows": 16,
      "path": "/home/delete/nirs4all/nirs4all/bench/AOM_v0/benchmarks/cohort_classification.csv",
      "runnable_rows": 16,
      "source": "AOM_classification",
      "total_rows": 17
    }
  ],
  "load_failures": [
    {
      "dataset": "COLZA/C_woOutlier",
      "failure_class": "wavelength_grid_unknown",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/COLZA/C_woOutlier/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/COLZA/C_woOutlier/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/COLZA/C_woOutlier/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/COLZA/C_woOutlier/Ytrain.csv"
      },
      "reason": "wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "COLZA/N_wOutlier",
      "failure_class": "wavelength_grid_unknown",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/COLZA/N_wOutlier/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/COLZA/N_wOutlier/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/COLZA/N_wOutlier/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/COLZA/N_wOutlier/Ytrain.csv"
      },
      "reason": "wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "COLZA/N_woOutlier",
      "failure_class": "wavelength_grid_unknown",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/COLZA/N_woOutlier/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/COLZA/N_woOutlier/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/COLZA/N_woOutlier/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/COLZA/N_woOutlier/Ytrain.csv"
      },
      "reason": "wavelength_grid_unknown: real wavelength headers were not parsed as a physical wavelength grid; refusing synthetic generation, clipping, and scoring",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "TABLET/Escitalopramt_310_Zhao",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/TABLET/Escitalopramt_310_Zhao/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/TABLET/Escitalopramt_310_Zhao/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/TABLET/Escitalopramt_310_Zhao/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/TABLET/Escitalopramt_310_Zhao/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "WOOD_density/WOOD_Density_402_Olale",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_Density_402_Olale/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_Density_402_Olale/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_Density_402_Olale/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_Density_402_Olale/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "WOOD_density/WOOD_N_402_Olale",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_N_402_Olale/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_N_402_Olale/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_N_402_Olale/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/WOOD_density/WOOD_N_402_Olale/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    }
  ],
  "row_count": 219,
  "status": "done"
}
```

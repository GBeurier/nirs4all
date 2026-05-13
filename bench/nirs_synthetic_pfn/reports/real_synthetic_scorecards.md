# Real/Synthetic Scorecards

## Objective

Standardize Phase B2 scorecards for local real benchmark cohorts against A2 synthetic preset datasets.

## Command

`PYTHONPATH=bench/nirs_synthetic_pfn/src python bench/nirs_synthetic_pfn/experiments/exp02_real_synthetic_scorecards.py --n-synthetic-samples 40 --max-real-samples 40 --max-real-datasets 0 --seed 20260429`

## Outputs

- Markdown: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.md`
- CSV metrics summary: `bench/nirs_synthetic_pfn/reports/real_synthetic_scorecards.csv`

## Phase A Gate Override

- `phase_a_gate_override`: `A3_failed_documented`
- A3 fitted-only real-fit gate remains failed/documented and is not hidden by this B2 report.
- B2 comparisons are realism diagnostics only; they do not establish downstream transfer benefit.

## Config

- Seed: 20260429
- A2 synthetic presets generated: 71
- Synthetic samples per preset: 40
- Real samples per scored dataset cap: 40
- Real dataset cap: all runnable rows
- Comparison spaces: raw, snv
- Thresholds are provisional smoke thresholds, not calibrated domain gates.
- Primary decisions and gates use only `comparison_space == "raw"`; SNV is an additional diagnostic.
- Synthetic spectra are calibrated before scoring; calibration does not change metric definitions or thresholds.

## Git Status

- `git status --short` lines: 6
- First entries:
  - ` M bench/AOM_v0/benchmark_runs/full/results.csv`
  - `?? .claude/`
  - `?? .codex`
  - `?? bench/AOM_v0/Ridge/`
  - `?? bench/nirs_synthetic_pfn/`
  - `?? docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md`

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
- Raw real/synthetic comparison rows written: 60
- SNV diagnostic comparison rows written: 60
- Blocked selected rows written: 17
- Synthetic-only dry-run rows written: 0
- Load/score failures after selection: 17

## Synthetic Mapping Strategy

Deterministic dataset-aware token mapping is used first; stable SHA-256 fallback is used only when no token rule matches. No dataset-index round-robin selection is used.

Strategy counts over raw rows:
- `dataset_aware_token`: 45
- `stable_hash_fallback`: 32

Synthetic preset counts over raw rows:
- `baking`: 6
- `dairy`: 9
- `forage`: 5
- `fruit`: 15
- `fuel`: 6
- `grain`: 4
- `meat`: 5
- `oilseeds`: 8
- `powders`: 14
- `tablets`: 5

## Real Marginal Calibration

Strong provisional marginal calibration is applied to synthetic spectra before raw and SNV scoring.
Fit inputs are limited to `real_X` and real wavelengths; apply inputs are limited to synthetic X and synthetic wavelengths.
No y/target/labels/splits or source oracle inputs are used for calibration metadata marked `oracle=false`, `label_inputs_used=false`, `target_inputs_used=false`, `split_inputs_used=false`, and `source_oracle_used=false`.
The calibration uses per-wavelength robust affine scaling, quantile mapping, and high-pass residual scaling; it is intentionally strong and provisional.
Thresholds are not changed and metric definitions are not weakened by calibration (`thresholds_modified=false`, `metrics_modified=false`).
Quantile mapping is column-wise and metadata records `replays_real_rows=false`; it must not be interpreted as proof of downstream transfer.

Calibration grid strategy counts over compared rows:
- `interpolated_real_calibration_to_synthetic_grid`: 104
- `same_grid`: 16

- Warning: Strong provisional marginal calibration for B2 diagnostics only; not a calibrated domain gate or transfer-benefit claim.

## Diagnostic Outcome

- Raw adversarial AUC smoke failures: 22/60 compared rows.
- Raw PCA overlap smoke failures: 2/60 compared rows.
- Raw gate remains visible: these scorecards do not claim realism success when raw adversarial AUC exceeds the smoke threshold.

## SNV Diagnostic

- SNV adversarial AUC smoke failures: 31/60 diagnostic rows.
- SNV is applied only after wavelength-grid alignment and only on comparison copies.
- SNV diagnostics do not override raw decisions and must not be interpreted as B2 success when raw fails.

## Metrics Table

| space | dataset | task | synthetic preset | n real | n synthetic | adv AUC | PCA overlap | NN ratio | derivative gap | decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `raw` | `ALPINE/ALPINE_P_291_KS` | `regression` | `oilseeds` | 40 | 40 | 0.2406 | 0.375 | 1.233 | 0.9784 | `provisional_pass` |
| `snv` | `ALPINE/ALPINE_P_291_KS` | `regression` | `oilseeds` | 40 | 40 | 0.7 | 0.125 | 1.943 | 0.9771 | `provisional_pass` |
| `raw` | `AMYLOSE/Rice_Amylose_313_YbasedSplit` | `regression` | `grain` | 40 | 40 | 0.8375 | 0.325 | 2.091 | 0.2876 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `AMYLOSE/Rice_Amylose_313_YbasedSplit` | `regression` | `grain` | 40 | 40 | 0.9688 | 0.175 | 3.916 | 0.2515 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `BEEFMARBLING/Beef_Marbling_RandomSplit` | `regression` | `meat` | 40 | 40 | 0.275 | 0.325 | 1.525 | 0.1162 | `provisional_pass` |
| `snv` | `BEEFMARBLING/Beef_Marbling_RandomSplit` | `regression` | `meat` | 40 | 40 | 0.6719 | 0.15 | 2.615 | 0.06581 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `BEER/Beer_OriginalExtract_60_KS` | `regression` | `baking` | 40 | 40 | 1 | 0 | 1.116 | 0.06767 | `provisional_review:pca_overlap,adversarial_auc` |
| `snv` | `BEER/Beer_OriginalExtract_60_KS` | `regression` | `baking` | 40 | 40 | 1 | 0 | 1.131 | 0.07429 | `provisional_review:pca_overlap,adversarial_auc` |
| `raw` | `BEER/Beer_OriginalExtract_60_YbaseSplit` | `regression` | `baking` | 40 | 40 | 1 | 0 | 1.098 | 0.06683 | `provisional_review:pca_overlap,adversarial_auc` |
| `snv` | `BEER/Beer_OriginalExtract_60_YbaseSplit` | `regression` | `baking` | 40 | 40 | 1 | 0 | 1.123 | 0.07075 | `provisional_review:pca_overlap,adversarial_auc` |
| `raw` | `BERRY/brix_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.1531 | 0.45 | 1.291 | 0.3341 | `provisional_pass` |
| `snv` | `BERRY/brix_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.4 | 0.325 | 1.829 | 0.2383 | `provisional_pass` |
| `raw` | `BERRY/ph_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.2 | 0.5 | 1.624 | 0.1212 | `provisional_pass` |
| `snv` | `BERRY/ph_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.5281 | 0.35 | 2.318 | 0.1393 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `BERRY/ta_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.2437 | 0.35 | 1.545 | 0.1591 | `provisional_pass` |
| `snv` | `BERRY/ta_groupSampleID_stratDateVar_balRows` | `regression` | `fruit` | 40 | 40 | 0.4188 | 0.325 | 2.285 | 0.2464 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `BISCUIT/Biscuit_Fat_40_RandomSplit` | `regression` | `baking` | 40 | 40 | 0.6719 | 0.275 | 1.878 | 0.08526 | `provisional_pass` |
| `snv` | `BISCUIT/Biscuit_Fat_40_RandomSplit` | `regression` | `baking` | 40 | 40 | 0.9094 | 0.275 | 2.754 | 0.08128 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `BISCUIT/Biscuit_Sucrose_40_RandomSplit` | `regression` | `baking` | 40 | 40 | 0.7125 | 0.225 | 1.765 | 0.08864 | `provisional_pass` |
| `snv` | `BISCUIT/Biscuit_Sucrose_40_RandomSplit` | `regression` | `baking` | 40 | 40 | 0.8438 | 0.175 | 2.484 | 0.08266 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `COLZA/C_woOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.7063 | 0.3 | 1.264 | 0.05302 | `provisional_pass` |
| `snv` | `COLZA/C_woOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.8906 | 0.2 | 2.282 | 0.05083 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `COLZA/N_wOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.8281 | 0.325 | 1.362 | 0.07094 | `provisional_pass` |
| `snv` | `COLZA/N_wOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.9625 | 0.15 | 2.406 | 0.06733 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `COLZA/N_woOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.8625 | 0.375 | 1.268 | 0.01373 | `provisional_review:adversarial_auc` |
| `snv` | `COLZA/N_woOutlier` | `regression` | `oilseeds` | 40 | 40 | 0.9594 | 0.2 | 2.263 | 0.04958 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit` | `regression` | `oilseeds` | 40 | 40 | 0.9031 | 0.3 | 1.766 | 0.09287 | `provisional_review:adversarial_auc` |
| `snv` | `CORN/Corn_Oil_80_ZhengChenPelegYbaseSplit` | `regression` | `oilseeds` | 40 | 40 | 0.9875 | 0.075 | 2.918 | 0.0895 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 40 | 40 | 0.975 | 0.3 | 1.731 | 0.1068 | `provisional_review:adversarial_auc` |
| `snv` | `CORN/Corn_Starch_80_ZhengChenPelegYbaseSplit` | `regression` | `grain` | 40 | 40 | 1 | 0.15 | 3.306 | 0.1043 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `DIESEL/DIESEL_bp50_246_b-a` | `regression` | `fuel` | 40 | 40 | 1 | 0.125 | 1.928 | 0.04273 | `provisional_review:adversarial_auc` |
| `snv` | `DIESEL/DIESEL_bp50_246_b-a` | `regression` | `fuel` | 40 | 40 | 1 | 0.1 | 2.333 | 0.02907 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `DIESEL/DIESEL_bp50_246_hla-b` | `regression` | `fuel` | 40 | 40 | 1 | 0.075 | 1.704 | 0.04137 | `provisional_review:adversarial_auc` |
| `snv` | `DIESEL/DIESEL_bp50_246_hla-b` | `regression` | `fuel` | 40 | 40 | 1 | 0.125 | 1.974 | 0.02671 | `provisional_review:adversarial_auc` |
| `raw` | `DIESEL/DIESEL_bp50_246_hlb-a` | `regression` | `fuel` | 40 | 40 | 1 | 0.1 | 1.765 | 0.03965 | `provisional_review:adversarial_auc` |
| `snv` | `DIESEL/DIESEL_bp50_246_hlb-a` | `regression` | `fuel` | 40 | 40 | 1 | 0.075 | 2.004 | 0.0269 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `DarkResp/Rd25_CBtestSite` | `regression` | `dairy` | 40 | 40 | 0.9437 | 0.375 | 1.568 | 0.2909 | `provisional_review:adversarial_auc` |
| `snv` | `DarkResp/Rd25_CBtestSite` | `regression` | `dairy` | 40 | 40 | 0.9906 | 0.175 | 2.642 | 0.3161 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `DarkResp/Rd25_GTtestSite` | `regression` | `grain` | 40 | 40 | 0.6312 | 0.2 | 1.286 | 0.2907 | `provisional_pass` |
| `snv` | `DarkResp/Rd25_GTtestSite` | `regression` | `grain` | 40 | 40 | 0.6562 | 0.1 | 2.453 | 0.2973 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `DarkResp/Rd25_XSBNtestSite` | `regression` | `dairy` | 40 | 40 | 0.7562 | 0.25 | 1.558 | 0.3069 | `provisional_pass` |
| `snv` | `DarkResp/Rd25_XSBNtestSite` | `regression` | `dairy` | 40 | 40 | 0.9156 | 0.125 | 2.775 | 0.3139 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `DarkResp/Rd25_spxy70` | `regression` | `fuel` | 40 | 40 | 0.1969 | 0.25 | 1.788 | 0.415 | `provisional_pass` |
| `snv` | `DarkResp/Rd25_spxy70` | `regression` | `fuel` | 40 | 40 | 0.5344 | 0.15 | 3.299 | 0.4103 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `ECOSIS_LeafTraits/Ccar_spxyG_block2deg` | `regression` | `forage` | 40 | 40 | 0.5 | 0.375 | 1.06 | 0.02651 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Ccar_spxyG_block2deg` | `regression` | `forage` | 40 | 40 | 0.4281 | 0.3 | 1.616 | 0.01571 | `provisional_pass` |
| `raw` | `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | `regression` | `baking` | 40 | 40 | 0.5375 | 0.375 | 1.345 | 0.05341 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | `regression` | `baking` | 40 | 40 | 0.6469 | 0.275 | 2.032 | 0.03598 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `ECOSIS_LeafTraits/Chla+b_spxyG_species` | `regression` | `tablets` | 40 | 40 | 0.65 | 0.275 | 1.292 | 0.01777 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/Chla+b_spxyG_species` | `regression` | `tablets` | 40 | 40 | 0.725 | 0.25 | 2.229 | 0.01742 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `ECOSIS_LeafTraits/LMA_spxyG_block2deg` | `regression` | `meat` | 40 | 40 | 0.4656 | 0.475 | 1.091 | 0.01157 | `provisional_pass` |
| `snv` | `ECOSIS_LeafTraits/LMA_spxyG_block2deg` | `regression` | `meat` | 40 | 40 | 0.4062 | 0.325 | 2.299 | 0.01019 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `FUSARIUM/FinalScore_grp70_30_scoreQ` | `regression` | `fuel` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `FUSARIUM/Fv_Fm_grp70_30` | `regression` | `meat` | 40 | 40 | 1 | 0.125 | 1.42 | 0.1349 | `provisional_review:adversarial_auc` |
| `snv` | `FUSARIUM/Fv_Fm_grp70_30` | `regression` | `meat` | 40 | 40 | 0.6813 | 0.125 | 2.996 | 0.1398 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `FUSARIUM/Tleaf_grp70_30` | `regression` | `powders` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `GRAPEVINES/grapevine_chloride_556_KS` | `regression` | `fruit` | 40 | 40 | 1 | 0.275 | 1.591 | 0.08099 | `provisional_review:adversarial_auc` |
| `snv` | `GRAPEVINES/grapevine_chloride_556_KS` | `regression` | `fruit` | 40 | 40 | 1 | 0.325 | 1.94 | 0.03656 | `provisional_review:adversarial_auc` |
| `raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD` | `regression` | `fruit` | 40 | 40 | 0.4531 | 0.2 | 1.027 | 0.2704 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_ASD` | `regression` | `fruit` | 40 | 40 | 0.2375 | 0.225 | 1.969 | 0.2617 | `provisional_pass` |
| `raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR` | `regression` | `fruit` | 40 | 40 | 0.6969 | 0.175 | 1.383 | 0.1121 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR` | `regression` | `fruit` | 40 | 40 | 0.8969 | 0.2 | 2.841 | 0.08536 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `fruit` | 40 | 40 | 0.4781 | 0.35 | 1.383 | 0.07189 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `fruit` | 40 | 40 | 0.6406 | 0.1 | 2.297 | 0.05416 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` | `regression` | `fruit` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD` | `regression` | `fruit` | 40 | 40 | 0.6406 | 0.325 | 1.096 | 0.2387 | `provisional_pass` |
| `snv` | `GRAPEVINE_LeafTraits/LMA_spxyG70_30_byCultivar_ASD` | `regression` | `fruit` | 40 | 40 | 0.4719 | 0.35 | 2.053 | 0.3074 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `fruit` | 40 | 40 | 0.9469 | 0.375 | 1.261 | 0.1662 | `provisional_review:adversarial_auc` |
| `snv` | `GRAPEVINE_LeafTraits/WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` | `regression` | `fruit` | 40 | 40 | 0.9406 | 0.025 | 5.201 | 0.1265 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `IncombustibleMaterial/TIC_spxy70` | `regression` | `powders` | 40 | 40 | 0.9375 | 0.225 | 1.629 | 0.3851 | `provisional_review:adversarial_auc` |
| `snv` | `IncombustibleMaterial/TIC_spxy70` | `regression` | `powders` | 40 | 40 | 0.8688 | 0.075 | 2.519 | 0.3065 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS` | `regression` | `powders` | 40 | 40 | 0.2375 | 0.075 | 1.72 | 1.122 | `provisional_review:derivative_gap` |
| `snv` | `LUCAS/LUCAS_SOC_Cropland_8731_NocitaKS` | `regression` | `powders` | 40 | 40 | 0.9187 | 0.1 | 3.245 | 0.6964 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `LUCAS/LUCAS_SOC_all_26650_NocitaKS` | `regression` | `powders` | 40 | 40 | 0.2 | 0.3 | 2.68 | 1.835 | `provisional_review:derivative_gap,nearest_neighbor_ratio` |
| `snv` | `LUCAS/LUCAS_SOC_all_26650_NocitaKS` | `regression` | `powders` | 40 | 40 | 0.8938 | 0.075 | 3.498 | 1.025 | `provisional_review:derivative_gap,adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic` | `regression` | `powders` | 40 | 40 | 0.175 | 0.2 | 2.835 | 1.659 | `provisional_review:derivative_gap,nearest_neighbor_ratio` |
| `snv` | `LUCAS/LUCAS_pH_Organic_1763_LiuRandomOrganic` | `regression` | `powders` | 40 | 40 | 0.9812 | 0.1 | 2.976 | 0.8509 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `MALARIA/Malaria_Oocist_333_Maia` | `regression` | `dairy` | 40 | 40 | 0.675 | 0.4 | 1.433 | 0.3711 | `provisional_pass` |
| `snv` | `MALARIA/Malaria_Oocist_333_Maia` | `regression` | `dairy` | 40 | 40 | 0.9906 | 0.125 | 2.167 | 0.4573 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `MALARIA/Malaria_Sporozoite_229_Maia` | `regression` | `meat` | 40 | 40 | 0.2437 | 0.175 | 1.547 | 0.5121 | `provisional_pass` |
| `snv` | `MALARIA/Malaria_Sporozoite_229_Maia` | `regression` | `meat` | 40 | 40 | 0.6594 | 0.225 | 3.132 | 0.4387 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `MANURE21/All_manure_CaO_SPXY_strat_Manure_type` | `regression` | `dairy` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `MANURE21/All_manure_K2O_SPXY_strat_Manure_type` | `regression` | `oilseeds` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` | `regression` | `tablets` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `MANURE21/All_manure_P2O5_SPXY_strat_Manure_type` | `regression` | `forage` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` | `regression` | `oilseeds` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `MILK/Milk_Fat_1224_KS` | `regression` | `dairy` | 40 | 40 | 1 | 0.275 | 1.7 | 0.004079 | `provisional_review:adversarial_auc` |
| `snv` | `MILK/Milk_Fat_1224_KS` | `regression` | `dairy` | 40 | 40 | 0.8094 | 0.225 | 2.351 | 0.006503 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `MILK/Milk_Lactose_1224_KS` | `regression` | `dairy` | 40 | 40 | 0.9906 | 0.3 | 1.955 | 0.004277 | `provisional_review:adversarial_auc` |
| `snv` | `MILK/Milk_Lactose_1224_KS` | `regression` | `dairy` | 40 | 40 | 0.6906 | 0.175 | 3.463 | 0.01221 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `MILK/Milk_Urea_1224_KS` | `regression` | `dairy` | 40 | 40 | 0.9688 | 0.275 | 1.446 | 0.01768 | `provisional_review:adversarial_auc` |
| `snv` | `MILK/Milk_Urea_1224_KS` | `regression` | `dairy` | 40 | 40 | 0.3906 | 0.225 | 2.296 | 0.04044 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `PEACH/Brix_spxy70` | `regression` | `fruit` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `PHOSPHORUS/LP_spxyG` | `regression` | `powders` | 40 | 40 | 0.8344 | 0.225 | 1.58 | 0.2406 | `provisional_pass` |
| `snv` | `PHOSPHORUS/LP_spxyG` | `regression` | `powders` | 40 | 40 | 0.8125 | 0.175 | 3.169 | 0.247 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `PHOSPHORUS/MP_spxyG` | `regression` | `powders` | 40 | 40 | 0.9313 | 0.275 | 1.751 | 0.2198 | `provisional_review:adversarial_auc` |
| `snv` | `PHOSPHORUS/MP_spxyG` | `regression` | `powders` | 40 | 40 | 0.95 | 0.1 | 2.825 | 0.196 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `PHOSPHORUS/NP_spxyG` | `regression` | `powders` | 40 | 40 | 0.9125 | 0.325 | 1.566 | 0.2498 | `provisional_review:adversarial_auc` |
| `snv` | `PHOSPHORUS/NP_spxyG` | `regression` | `powders` | 40 | 40 | 0.8094 | 0.125 | 3.156 | 0.2521 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `PHOSPHORUS/Pi_spxyG` | `regression` | `powders` | 40 | 40 | 0.9688 | 0.1 | 1.7 | 0.2059 | `provisional_review:adversarial_auc` |
| `snv` | `PHOSPHORUS/Pi_spxyG` | `regression` | `powders` | 40 | 40 | 0.9875 | 0.1 | 2.943 | 0.1974 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `PHOSPHORUS/V25_spxyG` | `regression` | `powders` | 40 | 40 | 0.6875 | 0.275 | 1.541 | 0.2508 | `provisional_pass` |
| `snv` | `PHOSPHORUS/V25_spxyG` | `regression` | `powders` | 40 | 40 | 0.7656 | 0.225 | 2.952 | 0.2598 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `PLUMS/Firmness_spxy70` | `regression` | `fruit` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `QUARTZ/Quartz_spxy70` | `regression` | `powders` | 40 | 40 | 0.9469 | 0.125 | 5.466 | 0.2051 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `snv` | `QUARTZ/Quartz_spxy70` | `regression` | `powders` | 40 | 40 | 0.9219 | 0.075 | 11.53 | 0.09269 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `TABLET/Escitalopramt_310_Zhao` | `regression` | `tablets` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `WOOD_density/WOOD_Density_402_Olale` | `regression` | `tablets` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `WOOD_density/WOOD_N_402_Olale` | `regression` | `tablets` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `ARABIDOPSIS_CEFE/Genotype10_250` | `classification` | `fuel` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `ARABIDOPSIS_CEFE/Group9_1856` | `classification` | `grain` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `ARABIDOPSIS_CEFE/Group_2185` | `classification` | `forage` | 40 | 40 | 0.2062 | 0.175 | 1.407 | 0.2327 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/Group_2185` | `classification` | `forage` | 40 | 40 | 0.2781 | 0.25 | 2.926 | 0.232 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `ARABIDOPSIS_CEFE/InOut_1264` | `classification` | `baking` | 40 | 40 | 0.2938 | 0.525 | 1.441 | 0.3009 | `provisional_pass` |
| `snv` | `ARABIDOPSIS_CEFE/InOut_1264` | `classification` | `baking` | 40 | 40 | 0.2719 | 0.2 | 2.394 | 0.3041 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `BEEF_Impurity/Beef_Impurity_60_AlJowder` | `classification` | `meat` | 40 | 40 | 0.2281 | 0.125 | 1.776 | 0.04406 | `provisional_pass` |
| `snv` | `BEEF_Impurity/Beef_Impurity_60_AlJowder` | `classification` | `meat` | 40 | 40 | 0.5469 | 0.3 | 1.592 | 0.09265 | `provisional_pass` |
| `raw` | `COFFEE_orig/CoffeeType_kenstone70_strat` | `classification` | `forage` | 40 | 40 | 0.6344 | 0.325 | 2.286 | 0.3141 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `COFFEE_orig/CoffeeType_kenstone70_strat` | `classification` | `forage` | 40 | 40 | 0.9219 | 0.125 | 5.419 | 0.2596 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `Cassava/CT2C_1057_CIAT_Acc` | `classification` | `fruit` | 40 | 40 | 0.9719 | 0.4 | 1.56 | 0.3087 | `provisional_review:adversarial_auc` |
| `snv` | `Cassava/CT2C_1057_CIAT_Acc` | `classification` | `fruit` | 40 | 40 | 0.9594 | 0.125 | 2.94 | 0.291 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `FUSARIUM/FinalScoreBin_grp70_30_classStrat` | `classification` | `powders` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `FUSARIUM/ScoreBin_grp70_30_classStrat` | `classification` | `powders` | 0 | 0 | n/a | n/a | n/a | n/a | `blocked_score_failure` |
| `raw` | `FruitPuree/Strawberry2C_983_Holland_Acc94.3` | `classification` | `fruit` | 40 | 40 | 0.8625 | 0.275 | 1.065 | 0.05886 | `provisional_review:adversarial_auc` |
| `snv` | `FruitPuree/Strawberry2C_983_Holland_Acc94.3` | `classification` | `fruit` | 40 | 40 | 0.9875 | 0.4 | 1.578 | 0.07594 | `provisional_review:adversarial_auc` |
| `raw` | `MALARIA/Oocist2C_333_Maia_Acc87.6` | `classification` | `dairy` | 40 | 40 | 0.1719 | 0.3 | 1.464 | 0.6559 | `provisional_pass` |
| `snv` | `MALARIA/Oocist2C_333_Maia_Acc87.6` | `classification` | `dairy` | 40 | 40 | 0.9844 | 0.125 | 2.378 | 0.5956 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `MALARIA/Sporozoite2C_229_Maia_Acc94.5` | `classification` | `oilseeds` | 40 | 40 | 0.3125 | 0.15 | 1.71 | 0.6535 | `provisional_pass` |
| `snv` | `MALARIA/Sporozoite2C_229_Maia_Acc94.5` | `classification` | `oilseeds` | 40 | 40 | 0.9594 | 0.1 | 3.018 | 0.5664 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |
| `raw` | `MILK/labels_kenstone70_strat` | `classification` | `dairy` | 40 | 40 | 0.6906 | 0.175 | 2.046 | 0.001429 | `provisional_review:nearest_neighbor_ratio` |
| `snv` | `MILK/labels_kenstone70_strat` | `classification` | `dairy` | 40 | 40 | 0.6813 | 0.125 | 2.513 | 0.001388 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `PISTACIA/Species_code_grpStrat70_30_bySpecimen` | `classification` | `fruit` | 40 | 40 | 0.2656 | 0.125 | 1.643 | 0.5161 | `provisional_pass` |
| `snv` | `PISTACIA/Species_code_grpStrat70_30_bySpecimen` | `classification` | `fruit` | 40 | 40 | 0.65 | 0.275 | 3.031 | 0.3875 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `Wood_Sustainability/C2_511_Davrieux_Acc82` | `classification` | `forage` | 40 | 40 | 0.2656 | 0.275 | 1.619 | 0.3427 | `provisional_pass` |
| `snv` | `Wood_Sustainability/C2_511_Davrieux_Acc82` | `classification` | `forage` | 40 | 40 | 0.6531 | 0.15 | 3.998 | 0.2982 | `provisional_review:nearest_neighbor_ratio` |
| `raw` | `Wood_Sustainability/C5_511_Davrieux_Acc82` | `classification` | `powders` | 40 | 40 | 0.2781 | 0.25 | 1.997 | 0.3334 | `provisional_pass` |
| `snv` | `Wood_Sustainability/C5_511_Davrieux_Acc82` | `classification` | `powders` | 40 | 40 | 0.8688 | 0.175 | 2.945 | 0.3064 | `provisional_review:adversarial_auc,nearest_neighbor_ratio` |

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

- `AOM_regression` `FUSARIUM/FinalScore_grp70_30_scoreQ` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Xtrain.csv
- `AOM_regression` `FUSARIUM/Tleaf_grp70_30` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Xtrain.csv
- `AOM_regression` `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `MANURE21/All_manure_CaO_SPXY_strat_Manure_type` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `MANURE21/All_manure_K2O_SPXY_strat_Manure_type` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `MANURE21/All_manure_P2O5_SPXY_strat_Manure_type` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `PEACH/Brix_spxy70` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `PLUMS/Firmness_spxy70` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `TABLET/Escitalopramt_310_Zhao` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `WOOD_density/WOOD_Density_402_Olale` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_regression` `WOOD_density/WOOD_N_402_Olale` [wavelength_grid_overlap]: real/synthetic wavelength grids have fewer than three overlapping points
- `AOM_classification` `ARABIDOPSIS_CEFE/Genotype10_250` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Xtrain.csv
- `AOM_classification` `ARABIDOPSIS_CEFE/Group9_1856` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Xtrain.csv
- `AOM_classification` `FUSARIUM/FinalScoreBin_grp70_30_classStrat` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Xtrain.csv
- `AOM_classification` `FUSARIUM/ScoreBin_grp70_30_classStrat` [non_finite_spectra]: non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Xtrain.csv

## Decision

B2 scorecard route is runnable and writes standardized markdown plus CSV metrics. Blocked selected rows retained in CSV: 17.

## Raw Summary JSON

```json
{
  "git_status": {
    "line_count": 6,
    "lines": [
      " M bench/AOM_v0/benchmark_runs/full/results.csv",
      "?? .claude/",
      "?? .codex",
      "?? bench/AOM_v0/Ridge/",
      "?? bench/nirs_synthetic_pfn/",
      "?? docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md"
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
      "dataset": "FUSARIUM/FinalScore_grp70_30_scoreQ",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/regression/FUSARIUM/FinalScore_grp70_30_scoreQ/Xtrain.csv",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "FUSARIUM/Tleaf_grp70_30",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/regression/FUSARIUM/Tleaf_grp70_30/Xtrain.csv",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "MANURE21/All_manure_CaO_SPXY_strat_Manure_type",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_CaO_SPXY_strat_Manure_type/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_CaO_SPXY_strat_Manure_type/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_CaO_SPXY_strat_Manure_type/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_CaO_SPXY_strat_Manure_type/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "MANURE21/All_manure_K2O_SPXY_strat_Manure_type",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_K2O_SPXY_strat_Manure_type/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_K2O_SPXY_strat_Manure_type/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_K2O_SPXY_strat_Manure_type/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_K2O_SPXY_strat_Manure_type/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "MANURE21/All_manure_MgO_SPXY_strat_Manure_type",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_MgO_SPXY_strat_Manure_type/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_MgO_SPXY_strat_Manure_type/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_MgO_SPXY_strat_Manure_type/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_MgO_SPXY_strat_Manure_type/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "MANURE21/All_manure_P2O5_SPXY_strat_Manure_type",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_P2O5_SPXY_strat_Manure_type/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_P2O5_SPXY_strat_Manure_type/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_P2O5_SPXY_strat_Manure_type/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_P2O5_SPXY_strat_Manure_type/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "MANURE21/All_manure_Total_N_SPXY_strat_Manure_type",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_Total_N_SPXY_strat_Manure_type/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_Total_N_SPXY_strat_Manure_type/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_Total_N_SPXY_strat_Manure_type/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/MANURE21/All_manure_Total_N_SPXY_strat_Manure_type/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "PEACH/Brix_spxy70",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/PEACH/Brix_spxy70/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/PEACH/Brix_spxy70/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/PEACH/Brix_spxy70/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/PEACH/Brix_spxy70/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
      "source": "AOM_regression",
      "task": "regression"
    },
    {
      "dataset": "PLUMS/Firmness_spxy70",
      "failure_class": "wavelength_grid_overlap",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/regression/PLUMS/Firmness_spxy70/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/regression/PLUMS/Firmness_spxy70/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/regression/PLUMS/Firmness_spxy70/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/regression/PLUMS/Firmness_spxy70/Ytrain.csv"
      },
      "reason": "real/synthetic wavelength grids have fewer than three overlapping points",
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
    },
    {
      "dataset": "ARABIDOPSIS_CEFE/Genotype10_250",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Genotype10_250/Xtrain.csv",
      "source": "AOM_classification",
      "task": "classification"
    },
    {
      "dataset": "ARABIDOPSIS_CEFE/Group9_1856",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/ARABIDOPSIS_CEFE/Group9_1856/Xtrain.csv",
      "source": "AOM_classification",
      "task": "classification"
    },
    {
      "dataset": "FUSARIUM/FinalScoreBin_grp70_30_classStrat",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/FUSARIUM/FinalScoreBin_grp70_30_classStrat/Xtrain.csv",
      "source": "AOM_classification",
      "task": "classification"
    },
    {
      "dataset": "FUSARIUM/ScoreBin_grp70_30_classStrat",
      "failure_class": "non_finite_spectra",
      "paths": {
        "test_path": "bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Xtest.csv",
        "train_path": "bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Xtrain.csv",
        "ytest_path": "bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Ytest.csv",
        "ytrain_path": "bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Ytrain.csv"
      },
      "reason": "non-finite spectra in /home/delete/nirs4all/nirs4all/bench/tabpfn_paper/data/classification/FUSARIUM/ScoreBin_grp70_30_classStrat/Xtrain.csv",
      "source": "AOM_classification",
      "task": "classification"
    }
  ],
  "row_count": 137,
  "status": "done"
}
```

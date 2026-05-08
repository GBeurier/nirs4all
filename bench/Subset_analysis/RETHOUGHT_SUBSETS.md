# Rethought Representative Subsets

Generated from `bench/benchmark_master_results.csv`.

## Method

- Universe: the 57 regression datasets already used by `main_class_score_matrix.csv`.
- Rows: `record_type in {observed, reference_paper}`, regression, test split, `status=ok`.
- Score: `score_ratio_vs_dataset_pls`; lower is better.
- Candidate: `source_family | model_class | model_name | variant`, aggregated by median per dataset.
- Eligibility: at least 90% global coverage and complete subset coverage.
- Objective: preserve concrete candidate transfer across all-candidate, no-TabPFN, linear-core, AOM-Ridge, AOM-PLS, multi-kernel, TabPFN, and nonlinear scopes.

## Recommended Subsets

### fast12_transfer_core

Primary fast iteration gate. One dataset per group, AOM-Ridge-compatible, balanced between linear wins, nonlinear wins, and ties.

- Size: 12
- Dataset groups: 12
- Behavior counts: `{'linear': 5, 'nonlinear_strong': 5, 'tie': 2}`

- `DIESEL_bp50_246_hlb-a` (DIESEL; tie; best=Multi-kernel ridge)
- `Corn_Oil_80_ZhengChenPelegYbaseSplit` (CORN; linear; best=AOM-Ridge)
- `MP_spxyG` (PHOSPHORUS; tie; best=AOM-PLS)
- `TIC_spxy70` (IncombustibleMaterial; linear; best=Ridge)
- `WUEinst_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` (GRAPEVINE_LeafTraits; linear; best=AOM-PLS)
- `brix_groupSampleID_stratDateVar_balRows` (BERRY; nonlinear_strong; best=TabPFN)
- `All_manure_K2O_SPXY_strat_Manure_type` (MANURE21; linear; best=AOM-Ridge)
- `Biscuit_Sucrose_40_RandomSplit` (BISCUIT; linear; best=AOM-PLS)
- `Ccar_spxyG_block2deg` (ECOSIS_LeafTraits; nonlinear_strong; best=NICON/CNN)
- `LUCAS_pH_Organic_1763_LiuRandomOrganic` (LUCAS; nonlinear_strong; best=TabPFN)
- `N_woOutlier` (COLZA; nonlinear_strong; best=TabPFN)
- `Beer_OriginalExtract_60_KS` (BEER; nonlinear_strong; best=TabPFN)

### audit20_transfer_core

Second-pass audit subset. Still AOM-Ridge-compatible, broader dataset-group and behavior coverage before running the full 57.

- Size: 20
- Dataset groups: 20
- Behavior counts: `{'linear': 6, 'nonlinear_mild': 3, 'nonlinear_strong': 8, 'tie': 3}`

- `All_manure_K2O_SPXY_strat_Manure_type` (MANURE21; linear; best=AOM-Ridge)
- `Rd25_GTtestSite` (DarkResp; linear; best=AOM-Ridge)
- `MP_spxyG` (PHOSPHORUS; tie; best=AOM-PLS)
- `An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra` (GRAPEVINE_LeafTraits; linear; best=PLS)
- `Biscuit_Sucrose_40_RandomSplit` (BISCUIT; linear; best=AOM-PLS)
- `Ccar_spxyG_block2deg` (ECOSIS_LeafTraits; nonlinear_strong; best=NICON/CNN)
- `DIESEL_bp50_246_b-a` (DIESEL; linear; best=AOM-Ridge)
- `Milk_Urea_1224_KS` (MILK; nonlinear_mild; best=TabPFN)
- `ALPINE_P_291_KS` (ALPINE; nonlinear_strong; best=TabPFN)
- `Rice_Amylose_313_YbasedSplit` (AMYLOSE; nonlinear_strong; best=TabPFN)
- `Quartz_spxy70` (QUARTZ; tie; best=Ridge)
- `LUCAS_pH_Organic_1763_LiuRandomOrganic` (LUCAS; nonlinear_strong; best=TabPFN)
- `Fv_Fm_grp70_30` (FUSARIUM; nonlinear_mild; best=TabPFN)
- `Escitalopramt_310_Zhao` (TABLET; nonlinear_mild; best=TabPFN)
- `WOOD_N_402_Olale` (WOOD_density; tie; best=Multi-kernel ridge)
- `ph_groupSampleID_stratDateVar_balRows` (BERRY; nonlinear_strong; best=TabPFN)
- `C_woOutlier` (COLZA; nonlinear_strong; best=TabPFN)
- `TIC_spxy70` (IncombustibleMaterial; linear; best=Ridge)
- `Beer_OriginalExtract_60_YbaseSplit` (BEER; nonlinear_strong; best=Meta-selector/MoE)
- `Firmness_spxy70` (PLUMS; nonlinear_strong; best=NICON/CNN)

## Transfer Check

| Subset | Scope | Eligible | Spearman | Winner Rank | Regret | Winner Class |
|---|---|---:|---:|---:|---:|---|
| `fast12_transfer_core` | `all_candidates` | 186 | 0.948 | 1 | 0.0000 | `TabPFN` |
| `fast12_transfer_core` | `no_tabpfn` | 182 | 0.945 | 1 | 0.0000 | `AOM-Ridge` |
| `fast12_transfer_core` | `linear_core` | 166 | 0.943 | 1 | 0.0000 | `AOM-Ridge` |
| `fast12_transfer_core` | `aom_ridge_only` | 13 | 0.978 | 1 | 0.0000 | `AOM-Ridge` |
| `fast12_transfer_core` | `aom_pls_only` | 133 | 0.922 | 1 | 0.0000 | `AOM-PLS` |
| `fast12_transfer_core` | `multi_kernel_only` | 5 | 0.900 | 1 | 0.0000 | `Multi-kernel ridge` |
| `fast12_transfer_core` | `tabpfn_only` | 4 | 1.000 | 1 | 0.0000 | `TabPFN` |
| `fast12_transfer_core` | `nonlinear_core` | 10 | 0.988 | 1 | 0.0000 | `TabPFN` |
| `audit20_transfer_core` | `all_candidates` | 186 | 0.970 | 1 | 0.0000 | `TabPFN` |
| `audit20_transfer_core` | `no_tabpfn` | 182 | 0.968 | 1 | 0.0000 | `AOM-Ridge` |
| `audit20_transfer_core` | `linear_core` | 166 | 0.970 | 1 | 0.0000 | `AOM-Ridge` |
| `audit20_transfer_core` | `aom_ridge_only` | 13 | 0.967 | 1 | 0.0000 | `AOM-Ridge` |
| `audit20_transfer_core` | `aom_pls_only` | 133 | 0.960 | 1 | 0.0000 | `AOM-PLS` |
| `audit20_transfer_core` | `multi_kernel_only` | 5 | 0.900 | 1 | 0.0000 | `Multi-kernel ridge` |
| `audit20_transfer_core` | `tabpfn_only` | 4 | 1.000 | 1 | 0.0000 | `TabPFN` |
| `audit20_transfer_core` | `nonlinear_core` | 10 | 0.976 | 1 | 0.0000 | `TabPFN` |

## Coverage Caution

The following core datasets are useful stress cases, but current high-coverage AOM-Ridge rows are missing there, so they are not part of the default subsets:

- `Brix_spxy70`
- `LUCAS_SOC_Cropland_8731_NocitaKS`
- `Malaria_Oocist_333_Maia`
- `Malaria_Sporozoite_229_Maia`


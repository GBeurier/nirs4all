# Multidataset Synthetic Realism Replan

Date: 2026-05-01

Scope: corrected planning for synthetic NIRS realism after recognizing that the
previous P2 work was DIESEL-only. This document supersedes any interpretation
that the DIESEL stop review is a global stop for `bench/nirs_synthetic_pfn`.
It adds no generator profile, no gate, no promotion, no threshold or metric
change, no statistics/PCA/noise capture step, no ML/DL step, and no
`nirs4all/` integration.

## Correction

The previous R9/P2 sequence is valid only as a DIESEL case study.

It should not be used to conclude that the mechanistic approach is exhausted
for NIRS spectra in general. DIESEL is narrow and atypical compared with the
available benchmark pool:

- one relatively pure fuel/liquid domain;
- one local support regime around `750-1550 nm` in the audited cohort;
- low matrix diversity compared with leaves, manure, beer, minerals, and crop
  tissues;
- possible prior preprocessing particularities that must not be generalized.

The correct global conclusion is therefore:

- DIESEL/R3d remains a useful case study and cautionary example;
- the global realism roadmap must restart from a multi-dataset real-data atlas;
- no cross-dataset synthetic realism claim has been established yet.

## Representative Panel

The first multidataset panel is a planning seed, not a realism conclusion:

| Dataset | Domain | Train/Test x Features | Spectral Axis | Metadata | Notes |
|---|---|---:|---|---|---|
| `All_manure_MgO_SPXY_strat_Manure_type` | manure | `343/147 x 1003` | `852.78-2502.37 nm` | yes | MgO target, SPXY split stratified by manure type |
| `All_manure_Total_N_SPXY_strat_Manure_type` | manure | `343/147 x 1003` | `852.78-2502.37 nm` | yes | Total N target, same support as MgO |
| `An_spxyG70_30_byCultivar_NeoSpectra` | grapevine leaves | `82/37 x 257` | `1350-2550 nm` | yes | NeoSpectra, cultivar split signal |
| `TIC_spxy70` | incombustible material | `43/19 x 254` | `868-1764 nm` | minimal | TIC target, some negative X values |
| `Chla+b_spxyG_species` | ECOSIS leaves | `3734/3116 x 196` | `450-2400 nm` | yes | species split, `-999` target sentinel observed |
| `Chla+b_spxyG_block2deg` | ECOSIS leaves | `2925/3925 x 196` | `450-2400 nm` | yes | block split, `-999` target sentinel observed |
| `ALPINE_P_291_KS` | alpine vegetation | `247/44 x 2151` | `350-2500 nm` | yes | P target, KS split, some negative X values |
| `Beer_OriginalExtract_60_YbaseSplit` | beer | `40/20 x 576` | `1100-2250 nm` | no | liquid/food, Y-based split |
| `N_woOutlier` | colza | `1205/1207 x 1154` | `12489.6 -> 3594.9 cm-1` | yes | wavenumber axis, not nm; outlier-filtered |
| `grapevine_chloride_556_KS` | grapevine | `388/167 x 1023` | `338.9-2515.3 nm` | no | chloride target, KS split, some negative X values |

Important consequence: a single DIESEL-style support window, profile family, or
failure mode is not a valid organizing principle for the full project.

The table is a file-level inventory seed from the current `Xtrain.csv`,
`Xtest.csv`, `Ytrain.csv`, `Ytest.csv`, and optional `Mtrain.csv`/`Mtest.csv`
files. It is not a preprocessing classification and does not establish any
synthetic realism result.

## Immediate Planning Change

The next active plan is no longer "make another DIESEL mechanism".

The next active plan is:

1. build a multidataset spectral atlas;
2. classify datasets by spectral axis, domain, instrument/resolution,
   preprocessing status, metadata availability, split policy, and target
   quality;
3. only then design mechanistic generator families.

## Phase M0: Dataset Contract Inventory

Goal: make the real data auditable before touching the generator again.

Deliverables:

- a bench-only inventory script for the representative panel;
- a table per dataset with:
  - path;
  - task type;
  - train/test rows;
  - feature count;
  - spectral axis type: nm, wavenumber, unknown;
  - axis direction and range;
  - feature separator/format;
  - target column;
  - sentinel/missing target flags, especially `-999`;
  - metadata presence and key metadata fields;
  - split policy inferred from name and files;
  - rough value range and negative-value presence;
  - explicit preprocessing evidence if present.

Rules:

- do not infer first derivative, SNV, MSC, absorbance, or reflectance unless a
  file, doc, or metadata field proves it;
- detect wavenumber axes separately from wavelength axes;
- do not drop sentinel values silently;
- do not use labels, targets, or splits to tune synthetic parameters.

## Phase M1: Real Spectral Atlas

Goal: understand what realism means across the panel.

Required analyses:

- support/range/resolution maps;
- value distribution maps per domain and instrument;
- smoothness and derivative-scale diagnostics, but as observation only;
- train/test split maps: KS, SPXY, species, cultivar, block, Y-base;
- metadata availability maps;
- target quality flags and sentinel handling.

Outputs:

- `reports/multidataset_real_spectral_atlas.md`;
- machine-readable CSV summary;
- explicit "unknown preprocessing" labels where status is not proven.

Blocking rule:

- no synthetic benchmark or generator tuning may proceed until the atlas
  distinguishes at least:
  - nm vs wavenumber datasets;
  - raw-like positive spectra vs negative/processed spectra;
  - metadata-rich vs metadata-poor datasets;
  - broad VIS-NIR/SWIR vs narrow instrument-specific supports.

## Phase M2: Generator Regime Design

Goal: design mechanistic families before coding profiles.

Candidate families:

- plant/leaf VIS-NIR-SWIR:
  pigments, water, dry matter, leaf structure, scattering;
- manure/organic-mineral mixtures:
  water, organic matter, nitrogen-related chemistry, mineral/ash effects,
  particle scattering;
- liquid food/fuel:
  pathlength/reference, CH/OH/NH bands, temperature/batch effects;
- mineral/incombustible material:
  broad mineral bands, particle size/scattering, low-organic spectra;
- wavenumber-domain datasets:
  explicit axis conversion or a separate wavenumber-native path.

Rules:

- one generator mechanism must declare which regime it targets;
- no profile can be evaluated only on DIESEL unless explicitly labeled
  "DIESEL case study";
- any preprocessing-like output must declare whether it is raw, absorbance,
  reflectance, derivative, or unknown.

## Phase M3: Multidataset Mechanistic Smoke Bench

Goal: test whether mechanistic generation covers multiple regimes without
statistical capture.

Minimum smoke set:

- one manure target;
- one ECOSIS leaf target;
- one grapevine/NeoSpectra target;
- one beer/liquid target;
- one mineral/material target;
- one wavenumber-domain target or an explicit exclusion report.

Required outputs:

- per-dataset validation report;
- per-regime failure taxonomy;
- no aggregate-only score that hides per-dataset failures;
- no promotion from a single dataset.

## Phase M4: Multidataset Scientific Validation

Goal: make the realism claim auditable.

Metrics should remain diagnostic until explicitly validated:

- support and scale compatibility;
- smoothness and derivative behavior;
- band-position plausibility;
- train/test leakage checks;
- real/synthetic scorecards per dataset;
- adversarial AUC only as a realism diagnostic, not as a tuning oracle;
- TSTR/RTSR only after synthetic generation is frozen for that run.

Blocking rule:

- a generator can only be called broadly useful if it passes multiple
  independent regimes;
- a generator that passes only DIESEL remains a domain-specific case study.

## Phase M5: Later Statistical And ML Layers

Only after the multidataset mechanistic path is blocked with evidence may the
project discuss:

1. mechanistic plus statistical/noise capture;
2. then, if still insufficient, ML/DL hybrid generation.

The current multidataset work has not reached that point.

## How To Use The DIESEL Work Now

DIESEL work remains useful for:

- demonstrating how to run strict mechanism isolation;
- showing how easy it is to overfit a narrow support/domain;
- providing reusable review/test discipline;
- providing exp28/exp29 patterns for data-support preflights.

DIESEL work is not useful for:

- claiming global NIRS realism;
- deciding that all mechanistic paths are exhausted;
- selecting a universal support window;
- promoting R9/P2-style pathlength/damping mechanisms across domains.

## Immediate Next Actions

1. Add an exp30-style multidataset inventory for the representative panel.
2. Generate `reports/multidataset_real_spectral_atlas.md`.
3. Update the handoff once exp30 exists.
4. Only after the atlas, write a new mechanism design document organized by
   dataset families.

No generator profile should be coded before Phase M0/M1 is complete.

# Phase M2 Generator Regime Design (docs-only)

Date: 2026-05-01

Scope: docs-only design step for the multidataset synthetic NIRS realism
plan. This document maps the representative panel from
`14_MULTIDATASET_REALISM_REPLAN.md` to candidate uncalibrated mechanistic
regime families, lists candidate phenomena per family from prior NIR
literature, and records per-family evidence gaps. It selects no profile,
no parameter, no constant, no threshold, no metric, and no gate.

## Contract

- This is a docs-only design deliverable. No generator, profile, mechanism,
  gate, promotion, threshold, metric, or integration change is added.
- No statistics, no PCA, no covariance, no quantile/marginal/noise capture,
  no calibration, no ML, no DL.
- No labels, targets, splits, downstream metrics, adversarial AUC,
  transfer scores, or PCA/cluster outputs are used as a tuning oracle for
  any candidate mechanism listed here. Targets and split policies are
  referenced only for identity, never to choose mechanisms or constants.
- `nirs4all/` is not modified or required by this design step.
- DIESEL R3d/R9/P2 conclusions are not generalized. Mechanisms used to
  describe the DIESEL case study remain a case study and are not promoted
  to other families.
- Preprocessing status of each panel dataset stays `unknown` unless an
  on-disk documentary source has been cited per row in the exp30 atlas.
  No SNV, MSC, derivative, absorbance, or reflectance status is inferred
  here without that proof.

## Roadmap Position

This document is Phase M2 of the multidataset replan
(`14_MULTIDATASET_REALISM_REPLAN.md`):

- Phase M0 (Dataset Contract Inventory) and Phase M1 (Real Spectral Atlas)
  are covered by exp30 and the atlas report
  `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`.
- Phase M2 (Generator Regime Design) is this file. Its purpose is to map
  data families to candidate mechanistic phenomena before any code is
  written.
- Phase M3 (Multidataset Mechanistic Smoke Bench) requires a separate,
  later, code deliverable. It is blocked by this document and by the
  per-family evidence gaps listed below.
- Phases M4 and M5 are blocked behind M3.

The DIESEL-only mechanistic stop documented in
`11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md` and the data-support
preflight in `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` remain in force for the
DIESEL case study lane and are not weakened by this document.

## Atlas Inputs

This design uses only the descriptive atlas fields produced by
`bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`:

- panel: 10 datasets, 10 inspectable rows;
- axis classes: 9 datasets on a `nm` axis, 1 dataset (`COLZA/N_woOutlier`)
  on a `wavenumber (cm-1)` axis (descending, forced by the panel rule);
- documentary preprocessing evidence: only `COLZA/N_woOutlier` carries a
  documentary `absorbance` declaration sourced from
  `bench/tabpfn_paper/data/regression/COLZA/README.txt`;
- negative X values observed in sampled rows for
  `IncombustibleMaterial/TIC_spxy70` and `ALPINE/ALPINE_P_291_KS`;
- target sentinel `-999` rows present in train for both
  `ECOSIS_LeafTraits/Chla+b_spxyG_species` and
  `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg`;
- metadata-rich (>=3 metadata columns) for 7 datasets, metadata-poor for
  3 (`BEER/Beer_OriginalExtract_60_YbaseSplit` and
  `GRAPEVINES/grapevine_chloride_556_KS` ship without metadata files;
  `IncombustibleMaterial/TIC_spxy70` only has a `Row_ID` column).

No atlas field that depends on labels, targets, splits, or downstream
metrics is used for regime selection. The split policy column is
identity-only.

## Dataset-To-Regime Assignment

The five candidate regime families come from
`14_MULTIDATASET_REALISM_REPLAN.md`. Wavenumber-domain is treated as a
distinct family because the axis transform must be declared, not inferred.

| dataset | atlas axis | atlas span | regime family | reason |
|---|---|---|---|---|
| `MANURE21/All_manure_MgO_SPXY_strat_Manure_type` | nm (852.78–2502.37) | 1003 features | manure / organic-mineral | manure samples with MgO/Total_N/CaO/K2O/OM metadata |
| `MANURE21/All_manure_Total_N_SPXY_strat_Manure_type` | nm (852.78–2502.37) | 1003 features | manure / organic-mineral | same physical cohort, different target column |
| `ECOSIS_LeafTraits/Chla+b_spxyG_species` | nm (450–2400) | 196 features | plant / leaf VIS-NIR-SWIR | leaf reflectance database, broad VIS+SWIR span |
| `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg` | nm (450–2400) | 196 features | plant / leaf VIS-NIR-SWIR | same instrument family, different split policy |
| `ALPINE/ALPINE_P_291_KS` | nm (350–2500) | 2151 features | plant / leaf VIS-NIR-SWIR | alpine vegetation, broadest nm span in panel |
| `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra` | nm (1350–2550) | 257 features | plant / leaf VIS-NIR-SWIR | NeoSpectra MicroNIR-style narrow SWIR window |
| `GRAPEVINES/grapevine_chloride_556_KS` | nm (338.9–2515.3) | 1023 features | plant / leaf VIS-NIR-SWIR | grapevine VIS+SWIR cohort with chloride target |
| `BEER/Beer_OriginalExtract_60_YbaseSplit` | nm (1100–2250) | 576 features | liquid food / fuel | brewing wort liquid in SWIR |
| `IncombustibleMaterial/TIC_spxy70` | nm (868–1764) | 254 features | mineral / incombustible material | low-organic mineral matrix, narrow NIR window |
| `COLZA/N_woOutlier` | wavenumber (cm-1, 3594.9–12489.6) | 1154 features | wavenumber-domain | documented absorbance on cm-1 axis |

Notes:

- The DIESEL/R3d case study is not a panel dataset and is not assigned to
  any family here. It remains a strict case study in
  `11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`.
- A single dataset may belong to more than one regime in a future
  revision. For now each dataset has exactly one assignment to keep the
  Phase M3 smoke set unambiguous.

## Candidate Mechanistic Phenomena By Family

These are reading-list candidates only. They are not selected, not
parameterized, and not promoted. Constants, ranges, gate thresholds, and
metric definitions are intentionally omitted.

### Plant / leaf VIS-NIR-SWIR

Datasets:

- `ECOSIS_LeafTraits/Chla+b_spxyG_species`,
  `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg`,
  `ALPINE/ALPINE_P_291_KS`,
  `GRAPEVINE_LeafTraits/An_spxyG70_30_byCultivar_NeoSpectra`,
  `GRAPEVINES/grapevine_chloride_556_KS`.

Candidate phenomena from prior leaf optics literature:

- pigment absorption in the visible region;
- water absorption combination/overtone bands;
- CH/OH/NH overtones for dry matter and protein;
- leaf internal scattering and structural geometry effects (PROSPECT-class
  formulations are reading-list candidates only);
- red-edge transition for full VIS+NIR cohorts;
- broad SWIR slope for narrow-window instruments such as MicroNIR or
  NeoSpectra.

Out of scope here: any specific band center, any constant, any combiner,
any gate, any metric, and any pigment/water mixture model selection.

### Manure / organic-mineral mixtures

Datasets: `MANURE21/All_manure_MgO_SPXY_strat_Manure_type`,
`MANURE21/All_manure_Total_N_SPXY_strat_Manure_type`.

Candidate phenomena:

- water and organic CH/OH/NH overtones;
- nitrogen-related chemistry effects;
- mineral and ash components inferred only from documented metadata
  columns such as `MgO`, `K2O`, `CaO`, `Total_N`, `P2O5`;
- particle-size and packing scattering effects of solid bulk samples.

Note: the metadata columns are descriptive identity. They are not used
here as a tuning oracle and are not used as targets-as-oracle in any
proposed mechanism. Use of any of these columns to drive a generator
parameter is forbidden under this document.

### Liquid food / fuel

Datasets: `BEER/Beer_OriginalExtract_60_YbaseSplit`.

Candidate phenomena:

- pathlength and reference behavior under Beer-Lambert framing;
- CH overtone bands for ethanol/sugar systems;
- OH water bands;
- batch and temperature effects.

DIESEL is not assigned here. The DIESEL case study has its own data-support
gate (`11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`,
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`) and must not be merged into the
generic liquid food/fuel family.

### Mineral / incombustible material

Datasets: `IncombustibleMaterial/TIC_spxy70`.

Candidate phenomena:

- broad mineral overtone bands;
- particle size and packing scattering;
- low-organic baseline behavior.

Important: the atlas reports negative X values in sampled rows for this
dataset. Until a documentary source explains those negatives (preprocessed
input, baseline-corrected input, instrument convention), no mechanism in
this family may assume raw positive intensities or raw reflectance.

### Wavenumber-domain

Datasets: `COLZA/N_woOutlier`.

Candidate phenomena:

- absorbance-domain Beer-Lambert framing on the cm-1 axis as documented
  in `bench/tabpfn_paper/data/regression/COLZA/README.txt`;
- water and CH/NH combination bands expressed in cm-1;
- explicit axis-conversion contract for any later cross-axis comparison;
  no implicit nm-as-default assumption.

## Per-Family Evidence Gaps

Each family has at least one missing input that must be resolved before
any Phase M3 mechanism for it can be coded. These are gap statements, not
backlog items, and they are not permission to start coding.

| family | gap | what would close the gap |
|---|---|---|
| plant / leaf VIS-NIR-SWIR | preprocessing status of all five datasets is `unknown`; ECOSIS train sets carry `-999` target sentinels with no on-disk explanation; ALPINE shows negative X values in sampled rows | a per-dataset documentary source for raw vs preprocessed status, and a documentary statement on the `-999` semantics and on the negative-X rows |
| manure / organic-mineral | no documentary source for raw vs preprocessed; bulk solid acquisition geometry is not row-bound to spectra | a documentary source for the acquisition mode and a row-bound geometry/instrument metadata reference (analogous to the DIESEL geometry requirement in `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` but adapted to manure) |
| liquid food / fuel | BEER has no metadata file at all; no documented pathlength, batch, or temperature | a documentary source for the BEER acquisition (cuvette pathlength, temperature, batch) |
| mineral / incombustible | TIC has only a `Row_ID` metadata column; negative X values are present in sampled rows; preprocessing status is `unknown` | a documentary source for TIC acquisition mode and for the negative-X semantics (preprocessed input vs raw with offset removal) |
| wavenumber-domain | only 1 dataset in the panel; broader cohort breadth needed before any wavenumber-native mechanism can be evaluated as multi-dataset | at least one additional documented wavenumber-axis dataset in the panel before a wavenumber-native Phase M3 mechanism is designed |

These gaps are not treated as label/target/split signal. They are
documentary metadata gaps about the real data acquisition.

## Status

- This document does not introduce a profile, mechanism, gate, promotion,
  threshold, metric, parameter, constant, or oracle.
- No code, no test, and no report under
  `bench/nirs_synthetic_pfn/reports/` is added or modified by this design
  step.
- `nirs4all/` is not modified or required.
- The exp30 atlas remains the only allowed input for regime assignment at
  this point.

## Next Allowed Actions

In strict order, with no skipping:

1. Document the per-family evidence gaps listed above on disk, per
   dataset, before any Phase M3 mechanism is coded. The accepted shape is
   the manifest pattern from
   `12_DATA_SUPPORT_MANIFEST_SCHEMA.md`, generalized away from DIESEL
   identity tokens to per-family identity tokens. A future preflight
   analogous to exp29 may then be designed to check those manifests for
   each family.
2. Only after a family's evidence gap is closed by a documented
   manifest, write a separate Phase M3 mechanism design document for
   that family. It must be docs-only at first, must declare which atlas
   rows it targets, and must declare which atlas rows it explicitly does
   not generalize to.
3. Only after that Phase M3 mechanism design exists for at least the
   manure family, the plant/leaf VIS-NIR-SWIR family, the liquid food
   family, the mineral family, and the wavenumber-domain family, may a
   multidataset mechanistic smoke bench be coded under
   `bench/nirs_synthetic_pfn/`. A single-family smoke bench is allowed
   only for diagnostic isolation and may not produce a global realism
   claim.
4. Phase M4 (multidataset scientific validation) and Phase M5
   (statistical/ML layers) remain blocked behind Phase M3 and behind
   their respective doctrine clauses in
   `14_MULTIDATASET_REALISM_REPLAN.md`.

## Cross-References

- `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
- `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
- `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`
- `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`

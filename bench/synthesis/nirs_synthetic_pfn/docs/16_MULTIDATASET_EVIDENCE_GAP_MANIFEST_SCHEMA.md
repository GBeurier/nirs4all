# Multidataset Evidence-Gap Manifest Schema (docs-only)

Date: 2026-05-01

Scope: docs-only schema for human-authored CSV/JSON manifests that close
the per-family evidence gaps recorded in
`15_M2_GENERATOR_REGIME_DESIGN.md`. This schema generalizes the DIESEL
manifest pattern in `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` to the five
multidataset regime families. It introduces no generator, profile,
mechanism, gate, promotion, threshold, metric, parameter, statistic,
PCA, covariance, noise capture, calibration, ML, or DL step. It does not
modify `nirs4all/`. It does not modify the DIESEL contract in
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`, and it does not modify
`exp29_data_support_manifest_preflight.py`. A future preflight analogous
to exp29, when designed, would consume manifests in this schema.

## Contract

- Docs-only specification: no code, no test, no preflight, no profile,
  no mechanism, no gate, no promotion, no threshold, no metric, no
  parameter, no constant.
- No statistics, no PCA, no covariance, no quantile/marginal/noise
  capture, no calibration, no ML, no DL.
- No labels, targets, splits, downstream metrics, adversarial AUC,
  transfer scores, or PCA/cluster outputs are accepted as manifest
  inputs. Manifests describe acquisition and preprocessing evidence
  only, never label or model feedback.
- DIESEL/fuel rows continue to be handled by
  `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` and exp29. This schema does not
  redefine, weaken, or replace them.
- Each manifest row targets exactly one regime family. A future
  preflight would refuse rows that match more than one family identity
  token set or that target an undeclared family.
- Closing an evidence gap with this schema does not authorize a Phase M3
  mechanism for that family. It only documents that the data inputs are
  available. A separate Phase M3 design document, per family, is still
  required afterwards.

## Roadmap Position

This document is the deliverable for Phase M2 next-action #1 in
`15_M2_GENERATOR_REGIME_DESIGN.md`. It precedes any Phase M3 mechanism
work and is independent of the DIESEL case study in
`11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md` and
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`.

## Regime Families And Identity Tokens

Each manifest row must declare its target family in the
`regime_family` field, and the row identity text must contain at least
one accepted token for that family. Tokens are matched case-folded
against the row identity fields `source`, `task`, `database_name`,
`dataset`, and `regime_family`.

| `regime_family` value | accepted identity tokens |
|---|---|
| `plant_leaf_visnir_swir` | `leaf`, `leaves`, `foliage`, `vegetation`, `plant`, `grapevine`, `vine`, `grape`, `alpine`, `ecosis`, `neospectra`, `microNIR` (case-folded) |
| `manure_organic_mineral` | `manure`, `slurry`, `dung`, `compost`, `digestate`, `fertilizer`, `organic_amendment`, `livestock_waste` |
| `liquid_food` | `beer`, `wort`, `brewing`, `juice`, `wine`, `milk`, `whey`, `beverage`, `liquid_food` |
| `mineral_incombustible` | `mineral`, `ore`, `rock`, `regolith`, `sediment`, `incombustible`, `ash`, `inorganic`, `soil_inorganic` |
| `wavenumber_domain` | `wavenumber`, `cm-1`, `cm_inv`, `ftir`, `mid_ir`, `mid-infrared`, `nir_wavenumber`, `colza_wavenumber` |

Notes:

- DIESEL/fuel identity tokens (`diesel`, `btex`, `petro`, `gasoline`,
  `fuel`, `hydrocarbon`, `alkane`, `aromatic`, `crude_oil`, `kerosene`)
  remain reserved for the DIESEL contract in
  `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` and are explicitly excluded from
  the `liquid_food` family identity tokens here.
- A single token set may not span two families. A row must not, for
  example, declare both `wavenumber_domain` and `plant_leaf_visnir_swir`
  in the same row even when the underlying dataset is plant tissue on a
  cm-1 axis. The wavenumber-domain family takes precedence for
  cm-1-axis rows because the axis transform is the dominant evidence
  gap.

## Required Identity Fields (All Families)

Every accepted row must populate every field in this set:

- `regime_family` (one of the values in the table above);
- `source`;
- `task`;
- `database_name`;
- `dataset`;
- `axis_unit` (one of `nm`, `cm-1`, `unknown`);
- `axis_min_value`;
- `axis_max_value`;
- `n_features_after_alignment`;
- `n_train_rows`;
- `n_test_rows`.

These are descriptive identity fields. They must agree with the
`exp30_multidataset_real_spectral_atlas` row for the same dataset where
one exists.

## Required Evidence Fields (Per Family)

In addition to the identity fields, every accepted row must populate
every evidence field for its family. Empty values are not accepted.

### plant_leaf_visnir_swir

- `preprocessing_status_documented_source` (path or DOI to a documentary
  source);
- `preprocessing_status_value` (one of `raw`, `absorbance`,
  `reflectance`, `transmittance`, `derivative`, `snv`, `msc`,
  `corrected_other`, `unknown`);
- `target_sentinel_value_documented_source` (required only when the
  atlas reports `-999`/`-9999`/`-99` train rows);
- `target_sentinel_semantics_value` (one of `not_measured`,
  `out_of_range`, `quality_flag`, `placeholder`, `other`; required only
  when the sentinel source is required);
- `negative_x_semantics_documented_source` (required only when the
  atlas reports negative X values in sampled rows);
- `negative_x_semantics_value` (one of `baseline_subtracted`,
  `derivative_output`, `instrument_offset`, `corrected_other`,
  `raw_with_offset`; required only when the negative-X source is
  required);
- `instrument_class` (free text descriptor of the instrument family,
  for example `ASD_FieldSpec`, `MicroNIR`, `NeoSpectra`,
  `LeafIntegratingSphere`).

### manure_organic_mineral

- `preprocessing_status_documented_source`;
- `preprocessing_status_value` (same enum as above);
- `acquisition_geometry_documented_source`;
- `acquisition_geometry_kind` (one of `row_bound_real_metadata`,
  `real_cohort_metadata_header`, `documented_constant`,
  `generic`);
- at least one of `cup_diameter_mm`, `sample_thickness_mm`,
  `presentation_mode` (free text), `instrument_class`;
- `bulk_packing_documentation_source` (required when
  `acquisition_geometry_kind` is `documented_constant`).

Generic geometry (`acquisition_geometry_kind=generic`) does not close
the manure evidence gap and remains blocked.

### liquid_food

- `preprocessing_status_documented_source`;
- `preprocessing_status_value` (same enum);
- `pathlength_documented_source`;
- `pathlength_mm` (numeric, optical pathlength of the cuvette);
- `temperature_documented_source`;
- `temperature_field_value_or_range` (free text descriptor of the
  documented sample temperature);
- `batch_documented_source`;
- `batch_field_or_descriptor` (free text descriptor of how batches are
  recorded for the cohort).

DIESEL/fuel rows must use the DIESEL contract in
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`. A `liquid_food` row containing a
DIESEL/fuel identity token is rejected by this schema.

### mineral_incombustible

- `preprocessing_status_documented_source`;
- `preprocessing_status_value` (same enum);
- `negative_x_semantics_documented_source` (required when the atlas
  reports negative X values in sampled rows);
- `negative_x_semantics_value` (same enum as for plant_leaf);
- `acquisition_geometry_documented_source`;
- `acquisition_geometry_kind` (same enum as for manure);
- `instrument_class`.

### wavenumber_domain

- `preprocessing_status_documented_source`;
- `preprocessing_status_value` (same enum);
- `axis_unit_documented_source` (required and must declare cm-1);
- `axis_direction_documented_source` (required, must explicitly declare
  ascending or descending cm-1);
- `axis_conversion_contract_source` (required, must point to a
  documented procedure for any future cross-axis comparison; absence of
  any planned cross-axis comparison is acceptable when explicitly
  declared in the same source);
- at least two distinct cohort sources documented in the panel for this
  family before any Phase M3 mechanism for the family is designed
  (recorded as `panel_breadth_documented_sources` containing two or
  more comma-separated source paths/DOIs).

## Forbidden Leakage Fields (All Families)

A manifest row containing any populated value for any field in this set
is rejected by this schema. The set is identical to
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`:

`label`, `labels`, `target`, `targets`, `target_value`, `class`,
`class_label`, `split`, `splits`, `fold`, `metric`, `metrics`, `score`,
`auc`, `auroc`, `roc_auc`, `adversarial`, `adversarial_score`,
`adversarial_metric`, `adversarial_auc`, `transfer`, `transfer_score`,
`transfer_metric`, `transfer_auc`, `downstream`, `downstream_metric`,
`downstream_score`, `downstream_auc`, `downstream_feedback`,
`performance_metric`, `validation_score`, `test_score`, `train_score`,
`threshold`, `gate_threshold`, `gate`, `pca`, `covariance`,
`noise_capture`, `ml_model`, `dl_model`, `calibration`, `profile`,
`promotion`.

Identity fields such as `regime_family`, `task`, and `database_name`
are descriptive identity, not metric or label feedback, and are not in
this list.

## Accepted File Shapes

- CSV: one manifest row per cohort or panel-dataset summary, with the
  identity and evidence fields above as columns.
- JSON: a single object, a list of objects, or an object containing a
  `rows` or `records` array of objects.

Relative paths in `*_documented_source` fields and in any future
spectral path field are resolved from a `--root` argument when a future
preflight script consumes the manifest.

## Decisions (Per Family)

A future preflight reading manifests in this schema would emit, per
family, exactly one of:

- `ready_for_phase_m3_mechanism_design_<family>`: at least one accepted
  row provides every required evidence field for that family with
  populated values, and no leakage field is populated;
- `blocked_pending_<family>_evidence_no_stats_ml`: no accepted row
  provides the full required evidence set for that family.

A `ready` decision for a family is not permission to code a Phase M3
mechanism for that family. It only authorizes writing the per-family
Phase M3 mechanism design document under `bench/nirs_synthetic_pfn/docs/`.

## Minimal CSV Examples

### Plant/leaf with documented absorbance and sentinel semantics

```csv
regime_family,source,task,database_name,dataset,axis_unit,axis_min_value,axis_max_value,n_features_after_alignment,n_train_rows,n_test_rows,preprocessing_status_documented_source,preprocessing_status_value,target_sentinel_value_documented_source,target_sentinel_semantics_value,instrument_class
plant_leaf_visnir_swir,future_lab,regression,ECOSIS_LeafTraits,Chla+b_documented_2026,nm,450,2400,196,3734,3116,docs/sources/ecosis_chla_release.txt,reflectance,docs/sources/ecosis_chla_sentinels.txt,not_measured,LeafIntegratingSphere
```

### Manure with row-bound geometry

```csv
regime_family,source,task,database_name,dataset,axis_unit,axis_min_value,axis_max_value,n_features_after_alignment,n_train_rows,n_test_rows,preprocessing_status_documented_source,preprocessing_status_value,acquisition_geometry_documented_source,acquisition_geometry_kind,cup_diameter_mm,instrument_class
manure_organic_mineral,future_lab,regression,MANURE21,All_manure_documented_2026,nm,852.78,2502.37,1003,343,147,docs/sources/manure21_acquisition.pdf,reflectance,docs/sources/manure21_geometry.csv,row_bound_real_metadata,30,FOSS_NIRSystems_5000
```

### Wavenumber-domain with axis-conversion contract

```csv
regime_family,source,task,database_name,dataset,axis_unit,axis_min_value,axis_max_value,n_features_after_alignment,n_train_rows,n_test_rows,preprocessing_status_documented_source,preprocessing_status_value,axis_unit_documented_source,axis_direction_documented_source,axis_conversion_contract_source,panel_breadth_documented_sources
wavenumber_domain,future_lab,regression,COLZA,N_woOutlier_documented_2026,cm-1,3594.9,12489.6,1154,1205,1207,bench/tabpfn_paper/data/regression/COLZA/README.txt,absorbance,bench/tabpfn_paper/data/regression/COLZA/README.txt,bench/tabpfn_paper/data/regression/COLZA/README.txt,docs/sources/colza_wavenumber_axis_conversion.md,bench/tabpfn_paper/data/regression/COLZA/README.txt,docs/sources/second_wavenumber_cohort_release.txt
```

## Minimal JSON Example

### Liquid food with documented pathlength

```json
{
  "rows": [
    {
      "regime_family": "liquid_food",
      "source": "future_lab",
      "task": "regression",
      "database_name": "BEER",
      "dataset": "Beer_OriginalExtract_60_documented_2026",
      "axis_unit": "nm",
      "axis_min_value": 1100,
      "axis_max_value": 2250,
      "n_features_after_alignment": 576,
      "n_train_rows": 40,
      "n_test_rows": 20,
      "preprocessing_status_documented_source": "docs/sources/beer_acquisition.pdf",
      "preprocessing_status_value": "absorbance",
      "pathlength_documented_source": "docs/sources/beer_acquisition.pdf",
      "pathlength_mm": 5.0,
      "temperature_documented_source": "docs/sources/beer_acquisition.pdf",
      "temperature_field_value_or_range": "20 +/- 1 C",
      "batch_documented_source": "docs/sources/beer_acquisition.pdf",
      "batch_field_or_descriptor": "fermentation_lot_id"
    }
  ]
}
```

## Status

- Docs-only specification: no script, no preflight, no test added.
- Targets, splits, labels, downstream metrics, adversarial AUC, transfer
  scores, PCA, covariance, noise capture, ML, and DL inputs are not
  accepted by this schema and are explicitly listed as forbidden.
- A future preflight implementing this schema would be a separate
  bench-only deliverable, scoped per family. Until that preflight
  exists, manifests in this schema are descriptive only and do not
  authorize a Phase M3 mechanism for any family.
- `nirs4all/` is not modified or required.

## Cross-References

- `bench/nirs_synthetic_pfn/docs/15_M2_GENERATOR_REGIME_DESIGN.md`
- `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
- `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
- `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`
- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`

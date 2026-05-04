# Data Support Manifest Schema For exp29

Date: 2026-05-01

Scope: human-authored CSV/JSON manifests for
`bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py`.
This document describes how to declare that a new real DIESEL/fuel corpus has
either wider wavelength support or row-bound geometry metadata. It adds no
generator profile, mechanism, gate, promotion, threshold, metric, stats, PCA,
noise capture, ML, or DL.

## Accepted File Shapes

exp29 accepts either CSV or JSON.

- CSV: one manifest row per corpus or cohort row summary.
- JSON: either a single object, a list of objects, or an object containing
  `rows` or `records`.

Relative file paths in path fields are resolved from `--root`.

## Wider Support Rows

A wider-support row is ready only when all of these are true:

- identity fields are present: `source`, `task`, `database_name`, `dataset`;
- the identity text contains a DIESEL/fuel token such as `diesel`, `btex`,
  `petro`, `gasoline`, `fuel`, `hydrocarbon`, `alkane`, `aromatic`,
  `crude_oil`, or `kerosene`;
- accepted wavelength evidence proves at least one aligned wavelength is
  outside `750-1550 nm`;
- no forbidden leakage field is populated.

Accepted wavelength evidence can be supplied in one of three ways:

- direct wavelength list in one of: `wavelengths`, `wavelength_nm`,
  `wavelength_headers`, `spectral_headers`, `aligned_wavelengths_nm`;
- summary fields: one min field, one max field, one count field, and either a
  support count or off-support count;
- spectral CSV header paths in one or more of: `train_path`, `xtrain_path`,
  `test_path`, `xtest_path`, `x_path`, `spectral_path`, `spectrum_path`.

Summary field aliases:

- min: `wavelength_min`, `wavelength_min_nm`, `min_wavelength`,
  `min_wavelength_nm`;
- max: `wavelength_max`, `wavelength_max_nm`, `max_wavelength`,
  `max_wavelength_nm`;
- count: `n_wavelengths_after_alignment`, `n_wavelengths`,
  `wavelength_count`, `wavelength_count_after_alignment`;
- support count: `support_count_after_alignment`,
  `support_wavelength_count_after_alignment`;
- off-support count: `off_support_count_after_alignment`,
  `off_support_wavelength_count_after_alignment`.

Min/max/count summaries must be internally consistent. For example, a range
outside `750-1550 nm` must also have a positive off-support count.

## Row-Bound Geometry Rows

A row-bound geometry row is ready only when all wider-support identity
requirements are met, and all binding fields are present:

- `source`;
- `task`;
- `database_name`;
- `dataset`;
- `row_binding_key`;
- `metadata_source`.

It must also include a row-bound metadata declaration, supplied through
`geometry_metadata_kind` or its accepted alias `geometry_scope`, and at least
one parseable geometry field.

Accepted row-bound declarations:

- `geometry_metadata_kind=real_cohort_metadata_header`;
- `geometry_metadata_kind=real_row_bound`;
- `geometry_metadata_kind=row_bound_real_metadata`;
- `geometry_metadata_kind=row_bound_real_cohort_metadata`;
- the same values in `geometry_scope`;
- or `real_row_bound=true`.

Accepted numeric geometry fields:

- `source_detector_distance_mm`;
- `source_detector_mm`;
- `source_detector_distance_cm`;
- `pathlength_mm`;
- `path_length_mm`;
- `optical_path_mm`;
- `collection_angle_deg`;
- `illumination_angle_deg`;
- `incidence_angle_deg`.

Accepted text geometry fields:

- `collection_geometry`;
- `measurement_geometry`;
- `geometry_description`.

Generic geometry does not unblock exp29. Rows marked with
`geometry_metadata_kind=generic`, `generic_geometry`, `generic_synthesis_model`,
`constant`, `builder_constant`, the same values in `geometry_scope`, or
`generic_geometry=true` remain blocked.

## Forbidden Leakage Fields

Do not include populated label, target, split, metric, gate, model, calibration,
profile, promotion, PCA, covariance, noise, downstream, adversarial, or transfer
feedback fields. exp29 rejects rows containing any populated field from this
set:

`label`, `labels`, `target`, `targets`, `target_value`, `class`,
`class_label`, `split`, `splits`, `fold`, `metric`, `metrics`, `score`, `auc`,
`auroc`, `roc_auc`, `adversarial`, `adversarial_score`,
`adversarial_metric`, `adversarial_auc`, `transfer`, `transfer_score`,
`transfer_metric`, `transfer_auc`, `downstream`, `downstream_metric`,
`downstream_score`, `downstream_auc`, `downstream_feedback`,
`performance_metric`, `validation_score`, `test_score`, `train_score`,
`threshold`, `gate_threshold`, `gate`, `pca`, `covariance`, `noise_capture`,
`ml_model`, `dl_model`, `calibration`, `profile`, `promotion`.

## Minimal CSV Examples

Wider real support from direct wavelengths:

```csv
source,task,database_name,dataset,wavelengths
field_lab,regression,DIESEL,DIESEL_wide_2026,700;750;900;1550;1600
```

Wider real support from summary counts:

```csv
source,task,database_name,dataset,wavelength_min,wavelength_max,n_wavelengths_after_alignment,off_support_count_after_alignment
field_lab,regression,DIESEL,DIESEL_summary_2026,700,1600,5,2
```

Row-bound geometry:

```csv
source,task,database_name,dataset,row_binding_key,metadata_source,geometry_metadata_kind,source_detector_distance_mm,collection_angle_deg
field_lab,regression,DIESEL,DIESEL_geometry_2026,sample_id,Mtrain.csv,real_cohort_metadata_header,4.0,45
```

## Minimal JSON Examples

Wider real support:

```json
{
  "rows": [
    {
      "source": "field_lab",
      "task": "regression",
      "database_name": "DIESEL",
      "dataset": "DIESEL_wide_2026",
      "wavelengths": "700;750;900;1550;1600"
    }
  ]
}
```

Row-bound geometry:

```json
{
  "rows": [
    {
      "source": "field_lab",
      "task": "regression",
      "database_name": "DIESEL",
      "dataset": "DIESEL_geometry_2026",
      "row_binding_key": "sample_id",
      "metadata_source": "Mtrain.csv",
      "geometry_metadata_kind": "real_cohort_metadata_header",
      "source_detector_distance_mm": 4.0,
      "collection_angle_deg": 45
    }
  ]
}
```

## Preflight Commands

Run exp29 against a manifest:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py \
  --root . \
  --manifest /path/to/manifest.csv \
  --report /tmp/exp29_data_support_manifest_preflight.md \
  --csv /tmp/exp29_data_support_manifest_preflight.csv
```

Run exp29 against the current default AOM cohort files:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py \
  --report /tmp/exp29_data_support_manifest_preflight.md \
  --csv /tmp/exp29_data_support_manifest_preflight.csv
```

## Decisions

exp29 has only two top-level recommendations:

- `blocked_pending_manifest_data_support_no_stats_ml`: no accepted row provides
  wider real DIESEL/fuel support or row-bound real geometry metadata.
- `ready_for_mechanistic_audit_design`: at least one accepted row provides
  wider real DIESEL/fuel support or row-bound real geometry metadata.

`ready_for_mechanistic_audit_design` is not permission to code a generator or
profile. It means the next step is a separate mechanistic audit design in
`uncalibrated_raw`, with unchanged metrics, unchanged gates, and explicit
anti-leakage constraints.

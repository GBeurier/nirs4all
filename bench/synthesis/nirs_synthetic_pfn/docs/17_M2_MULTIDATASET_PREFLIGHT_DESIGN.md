# Multidataset Evidence-Gap Preflight Design (docs-only)

Date: 2026-05-01

Scope: docs-only design for a future bench-only preflight script that
consumes manifests authored against
`16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md`. This document
describes the CLI surface, per-row state machine, per-family aggregation,
output report shape, and required test coverage. It does not implement
the preflight, does not add a profile, mechanism, gate, promotion,
threshold, metric, parameter, statistic, PCA, covariance, noise capture,
calibration, ML, or DL step. It does not modify `nirs4all/`. It does
not modify `12_DATA_SUPPORT_MANIFEST_SCHEMA.md`, the DIESEL preflight
`exp29_data_support_manifest_preflight.py`, or the atlas
`exp30_multidataset_real_spectral_atlas.py`.

When implemented, the future script must live under
`bench/nirs_synthetic_pfn/experiments/` and ship with a sibling test
file under `bench/nirs_synthetic_pfn/tests/`, both following the
exp28/exp29/exp30 file-loading and module-loading conventions.

## Contract

- Bench-only preflight: descriptive checks against schema 16 only.
  No spectra are built, no calibration is applied, no statistics or PCA
  or covariance or noise capture is computed, no ML or DL is trained.
- The preflight reads identity and evidence fields from a manifest. It
  reads no `Xtrain`, `Xtest`, `Ytrain`, `Ytest`, `Mtrain`, or `Mtest`
  contents. The atlas script `exp30` already covers file-level reads.
- Targets, splits, labels, downstream metrics, adversarial AUC,
  transfer scores, PCA, covariance, noise capture, ML, and DL inputs
  are forbidden. A row containing any populated forbidden field listed
  in schema 16 must be rejected.
- A `ready_for_phase_m3_mechanism_design_<family>` decision authorizes
  only the writing of a per-family Phase M3 mechanism design document.
  It does not authorize a generator profile, a gate, a promotion, a
  threshold, a metric change, or any code under `nirs4all/`.
- Default behavior with no manifest is empty-result and a blocked
  recommendation per family.

## Roadmap Position

This is the design step for Phase M2 next-action #2 in
`15_M2_GENERATOR_REGIME_DESIGN.md`. It is downstream of:

- `16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md` (input contract);
- `exp30_multidataset_real_spectral_atlas` (the descriptive
  identity reference).

It precedes any Phase M3 mechanism design document and any Phase M3 code.
The DIESEL preflight in `12`/`exp29` is unaffected and remains the
authoritative DIESEL contract.

## CLI Surface

The future script must accept exactly these arguments:

- `--root PATH` (default: `.`): root for resolving relative
  `*_documented_source` paths inside the manifest;
- `--manifest PATH` (required when running against a real manifest;
  optional for empty-default behavior);
- `--regime-family FAMILY` (optional, repeatable): restrict accepted
  rows to one or more of `plant_leaf_visnir_swir`,
  `manure_organic_mineral`, `liquid_food`, `mineral_incombustible`,
  `wavenumber_domain`. Unspecified means all five families.
- `--report PATH`: markdown report output path
  (default: `bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.md`);
- `--csv PATH`: CSV row dump output path
  (default: `bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv`).

The script must print the report path, the CSV path, the row count, and
a per-family summary dict to stdout on completion, exactly like
`exp29` and `exp30`.

## Per-Row State Machine

For each manifest row the future script must, in order:

1. **Parse the row** as a flat string-keyed mapping. CSV values are
   already strings; JSON values must be coerced to strings before field
   matching. Empty values are treated as missing.
2. **Reject leakage.** If any populated value exists for any field name
   in the schema-16 forbidden-leakage set, mark the row
   `rejected_leakage_fields` and emit no further checks. The row is
   ineligible to satisfy any family.
3. **Check required identity fields.** If any of `regime_family`,
   `source`, `task`, `database_name`, `dataset`, `axis_unit`,
   `axis_min_value`, `axis_max_value`, `n_features_after_alignment`,
   `n_train_rows`, `n_test_rows` is missing, mark the row
   `blocked_missing_identity` and emit no family-specific evidence
   checks for that row.
4. **Validate `regime_family`** against the five accepted family values.
   Unknown family names mark the row `blocked_unknown_regime_family`.
5. **Apply the `--regime-family` filter** if any was passed. Rows whose
   declared family is not in the filter set are marked `filtered_out`
   and do not contribute to per-family aggregation.
6. **Check identity tokens.** Case-fold the row identity text
   (concatenation of `source`, `task`, `database_name`, `dataset`, and
   `regime_family`) and require at least one match against the family's
   identity token set in schema 16. Reject rows that match tokens of a
   different family with `blocked_cross_family_identity_token`.
7. **Reject reserved DIESEL/fuel tokens.** Rows that contain any DIESEL
   reserved token from `12_DATA_SUPPORT_MANIFEST_SCHEMA.md` are marked
   `blocked_reserved_diesel_fuel_token` regardless of declared family.
8. **Check the per-family required evidence fields** from schema 16. If
   any required field is missing or empty, mark the row
   `blocked_missing_<family>_evidence_no_stats_ml` with the missing
   field names recorded in a `missing_evidence_fields` column.
9. **Validate enum values.** `preprocessing_status_value`,
   `target_sentinel_semantics_value`, `negative_x_semantics_value`,
   `acquisition_geometry_kind`, and `axis_unit` must each be in their
   schema-16 enum. Out-of-enum values mark the row
   `blocked_invalid_enum_value`.
10. **Validate numeric fields.** `axis_min_value`, `axis_max_value`,
    `n_features_after_alignment`, `n_train_rows`, `n_test_rows`, and
    any numeric geometry field (e.g. `pathlength_mm`,
    `cup_diameter_mm`, `sample_thickness_mm`) must parse as floats and
    must satisfy `axis_min_value < axis_max_value`,
    `n_features_after_alignment > 0`, `n_train_rows >= 0`,
    `n_test_rows >= 0`. Failures mark the row
    `blocked_invalid_numeric_field`.
11. **Validate documented sources.** Every `*_documented_source` field
    must either be a populated free-text DOI/URL or resolve to an
    existing path under `--root`. Missing referenced paths mark the row
    `blocked_documented_source_not_found`.
12. **Emit `accepted_for_<family>`** when none of the above blocks
    fired for a row whose declared family passed the filter.

A single row may only contribute to one family. The first matching
family decides; cross-family identity tokens are rejected explicitly in
step 6 and 7 above.

## Per-Family Aggregation And Decisions

After all rows are processed, the script must aggregate per family:

- count of `accepted_for_<family>` rows;
- count of rows blocked at each stage above, broken down by status;
- the missing-evidence-field histogram (which required field is most
  often missing).

For each of the five families the script emits exactly one decision:

- `ready_for_phase_m3_mechanism_design_<family>`: at least one
  `accepted_for_<family>` row exists for that family;
- `blocked_pending_<family>_evidence_no_stats_ml`: no
  `accepted_for_<family>` row exists for that family.

A `ready` decision for one family does not affect the decision for
another family. The script must not emit a single global decision.

## Output Report Shape

The markdown report must contain, in order:

1. header with audit scope, comparison space (`uncalibrated_raw_or_unknown`),
   manifest source, report path, csv path, total rows, and one
   recommendation line per family;
2. Contract section identical in posture to `exp29` and `exp30`,
   including the explicit no-stats / no-ML / no-oracle text;
3. Per-family summary table with columns: family, accepted rows,
   blocked rows by status, missing-evidence-field top three,
   recommendation;
4. Per-row table with columns: row index, status, regime_family,
   source/task/database_name/dataset, missing evidence fields,
   rejected leakage fields, evidence path resolution status;
5. Reproduce block with the exact CLI invocation used.

The CSV row dump must contain one row per manifest row, with at minimum
the columns: status, regime_family, source, task, database_name,
dataset, axis_unit, axis_min_value, axis_max_value,
n_features_after_alignment, n_train_rows, n_test_rows,
preprocessing_status_value, missing_evidence_fields,
rejected_leakage_fields, recommendation_signal.

A `not_inspected_rows` count must surface at the top of the report when
any row failed to parse at all.

## Required Test Coverage

The future test file under
`bench/nirs_synthetic_pfn/tests/test_exp31_<name>.py` must cover at
least:

1. empty-default behavior emits five `blocked_pending_*` decisions;
2. wider-than-schema-16 leakage fields cause `rejected_leakage_fields`
   regardless of identity completeness;
3. each of the five families has at least one positive
   `ready_for_phase_m3_mechanism_design_<family>` test from a minimal
   well-formed CSV manifest;
4. each of the five families has at least one negative
   `blocked_missing_<family>_evidence_no_stats_ml` test that drops one
   required evidence field;
5. cross-family token leakage (e.g. a `liquid_food` row containing the
   token `manure`) is blocked with `blocked_cross_family_identity_token`;
6. reserved DIESEL/fuel tokens (e.g. `diesel`) in any non-DIESEL family
   are blocked with `blocked_reserved_diesel_fuel_token`;
7. invalid enum values for `preprocessing_status_value`,
   `acquisition_geometry_kind`, and `axis_unit` are blocked with
   `blocked_invalid_enum_value`;
8. invalid numeric fields (negative `axis_min_value`,
   `axis_min_value >= axis_max_value`, non-numeric `pathlength_mm`)
   are blocked with `blocked_invalid_numeric_field`;
9. a documented source path that does not exist under `--root` is
   blocked with `blocked_documented_source_not_found`;
10. CSV and JSON manifests carrying the same logical row produce
    identical per-row decisions and identical per-family
    recommendations;
11. the `--regime-family` filter restricts per-family aggregation to
    the specified families and leaves the other families with their
    default `blocked_pending_*` decision unchanged;
12. the report markdown contains the exact strings:
    `no statistics`, `no PCA`, `no calibration`, `no ML`, `no DL`,
    `no labels`, `no targets`, `no splits`, and
    `nirs4all/` is not modified;
13. the per-row CSV contains the schema-16 mandatory columns above and
    the row count matches the manifest row count.

## Out Of Scope

The future script must NOT:

- read any spectral file content (`X*.csv`, `Y*.csv`, `M*.csv`);
- compute any statistic, percentile, derivative, smoothness, PCA,
  covariance, noise estimate, or any function of label/target/split;
- access any nirs4all module or function;
- emit any aggregate cross-family score or single global decision;
- emit any field that could be interpreted as a parameter, gate, or
  threshold for a future generator;
- modify any file outside `bench/nirs_synthetic_pfn/`.

## Reproduce (After Implementation)

After the future script and test file exist, reproduction must look
like:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp31_multidataset_evidence_gap_manifest_preflight.py \
  --root . \
  --manifest /path/to/manifest.csv \
  --report bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.md \
  --csv bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv
```

and:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python -m pytest \
  bench/nirs_synthetic_pfn/tests/test_exp31_multidataset_evidence_gap_manifest_preflight.py -q
```

The exact filename `exp31_multidataset_evidence_gap_manifest_preflight.py`
is a placeholder. The implementer may pick a different filename
provided it follows the existing `exp<NN>_<short_name>.py` convention
and the sibling test file follows the matching
`test_exp<NN>_<short_name>.py` convention.

## Status

- Docs-only design: no script, no test, no preflight added.
- `nirs4all/` not modified or required.
- `12_DATA_SUPPORT_MANIFEST_SCHEMA.md`, `exp29`, and `exp30` are
  unchanged.
- A `ready` decision from the future preflight does not authorize a
  Phase M3 mechanism. It only authorizes the per-family Phase M3
  mechanism design document.

## Cross-References

- `bench/nirs_synthetic_pfn/docs/16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/docs/15_M2_GENERATOR_REGIME_DESIGN.md`
- `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
- `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
- `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`
- `bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`

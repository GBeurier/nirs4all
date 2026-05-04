# Mechanistic Stop Review And Data Requirements

Date: 2026-05-01

Scope: P2-07 stop review for `bench/nirs_synthetic_pfn`, DIESEL
`uncalibrated_raw` realism only. This document consolidates existing
R9e-R9m, P2a, and exp24-exp28 evidence. It creates no profile, no R9n, no
gate, no promotion, no threshold or metric change, and no `nirs4all/`
integration.

## Decision

Status: `mechanistic_uncalibrated_blocked_pending_data_support`

The current uncalibrated mechanistic route is stopped. The stop is not because
statistics, PCA, noise capture, ML, or DL are now approved. The stop is because
the next plausible mechanistic questions need data support that the current
bench does not contain:

- no current real DIESEL/fuel cohort has non-empty wavelength support outside
  `750-1550 nm` after real-grid alignment;
- no real cohort metadata row binds source-detector distance, pathlength,
  collection angle, illumination angle, or equivalent collection geometry to
  the audited spectra.

R3d (`r3d_diesel_matrix_v1`) remains the accepted DIESEL baseline. R9e, R9f,
R9h, R9i, R9j, R9k, R9l, R9m, P2a, and exp24-exp28 are diagnostic evidence
only.

## Evidence Summary

R9e-R9m bounded the support-local and component-isolation space:

- R9e: clean support-only pathlength/reference attenuation is non-null and
  lowers level and morphology gap versus R3d without guard clipping, but it is
  diagnostic-only and does not supersede R3d.
- R9f: moving attenuation before offset and restricting it to the path
  component is close to R3d and weaker than R9e.
- R9h: support CH center/drop-1720 isolation is effectively null versus R3d.
- R9i: CH width/gain isolation is non-null but too small to explain the
  R4b/R4c behavior.
- R9j: residual damping is the strongest isolated component in R9i-R9k, but it
  remains below the stronger R4-family behavior and is not a standalone
  promotion.
- R9k: continuum-hump-only is effectively neutral to slightly worse versus
  R3d.
- R9l: residual damping plus clean attenuation is the strongest controlled
  R9 combination before R9m, but it remains behind R4b and has aggregate vs
  paired ambiguity against R4c.
- R9m: width/gain plus residual damping plus clean attenuation is NO-GO for
  consolidation. It is only slightly better than R9l on morphology gap and
  introduces derivative-under regression, so no R9n or retune follows.

P2a and exp24-exp28 then tested whether the next mechanistic layer could be
audited with current data:

- exp24: the render-stage map shows all compared profiles remain dominated by
  `support_mean_drives_global_mean`; it is a map, not a remediation.
- exp25/P2a: full-row row-level pathlength/reference attenuation is
  technically valid as a diagnostic profile, but collapses to R9e on the
  current cohort because all aligned wavelengths are inside `750-1550 nm`.
- exp26: the current real-aligned cohort cannot distinguish R9e support-only
  attenuation from P2a full-row attenuation; a generated prior-grid
  counterfactual with 38 off-support wavelengths can distinguish them.
- exp27: fixed readout maps are report-only counterfactual diagnostics and
  degrade versus identity; source-detector geometry cannot be audited because
  no row-bound geometry metadata is present.
- exp28: inventory confirms zero current real DIESEL/fuel rows extending
  outside `750-1550 nm` after alignment and zero real cohort metadata headers
  carrying source-detector/pathlength/collection geometry. Recommendation:
  `blocked_pending_metadata_or_wider_real_cohort_no_stats_ml`.

## Science Gate Versus Diagnostic Evidence

Science/gate facts:

- R3d is the accepted DIESEL baseline.
- The comparison space remains `uncalibrated_raw`.
- The current uncalibrated mechanistic path is blocked pending data support.
- No profile after R3d is promoted, gated, or integrated.
- The next mechanistic audit requires either wider real spectral support or
  row-bound geometry metadata, as defined below.

Diagnostic-only facts:

- R9e-R9m and P2a are mechanism isolations or controlled combinations.
- exp24-exp28 are report-side audits and inventories.
- R4b/R4c remain useful comparators for behavior, not promoted replacements
  through this stop review.
- Aggregate and paired deltas, support/off-support counts, readout
  counterfactuals, and inventory rows are evidence for decisions, not new
  metrics or thresholds.

## Required Data To Resume Mechanistic Work

For human-authored exp29 CSV/JSON manifest fields and examples, see
`12_DATA_SUPPORT_MANIFEST_SCHEMA.md`.

At least one of the following prerequisites must be satisfied before another
uncalibrated mechanistic audit is proposed:

1. Wider real cohort support:
   - ingest a real DIESEL/fuel/BTEX/petrochemical cohort whose numeric spectral
     headers retain non-empty off-support wavelengths after the same real-grid
     alignment semantics used by exp28;
   - exp28 must report at least one real-grid row with
     `extends_outside_750_1550_after_alignment=True` and
     `off_support_count_after_alignment > 0`;
   - the cohort source, task, dataset name, train/test header merge behavior,
     and support counts must be written to the exp28 CSV/report.

2. Row-bound geometry metadata:
   - add real cohort manifest or row metadata that binds each audited spectrum
     to source-detector distance or equivalent source-detector geometry;
   - include pathlength or optical path where applicable;
   - include collection, illumination, or incidence angle where applicable;
   - distinguish row-bound real metadata from generic builder/domain constants;
   - exp28 must report `geometry_metadata_kind=real_cohort_metadata_header`, or
     a documented replacement enum with the same row-bound semantics, for the
     relevant DIESEL/fuel rows;
   - the exp28 CSV/report must identify the cohort, row binding key, metadata
     source, and at least one parsed source-detector/pathlength/collection or
     illumination geometry field, not only
     `generic_geometry_available_not_real_row_bound`.

Generic mechanistic laws or constants in code/docs are not sufficient on their
own. A future optical-depth, geometry, or readout mechanism must be tied to
new real support or row-bound metadata before it is auditable.

## Explicit Prohibitions While Blocked

Until exp28 is re-run and shows that at least one prerequisite above is met,
the following are prohibited in this project lane:

- stats, PCA, covariance, quantile, marginal, residual, or noise capture;
- ML/DL generation, learned residuals, learned priors, or source-oracle use;
- label, target, split, downstream metric, adversarial score, transfer score,
  or gate-driven parameter selection;
- retuning R9e/R9f/R9j/R9l/R9m/P2a constants after reading morphology results;
- creating R9n or a new P2 profile to bypass the missing data support;
- changing thresholds, metrics, gates, or integration status;
- editing `nirs4all/` as part of this bench stop review.

These prohibitions do not ban read-only inventory updates or documentation that
clarifies the block. They do ban using statistical or learned layers as the
next move before the mechanistic data prerequisites are satisfied.

## Resume Path

The only approved restart sequence is:

1. Ingest a wider real DIESEL/fuel cohort, or add row-bound real geometry
   metadata at the cohort/manifest layer.
2. Re-run exp28 unchanged or with narrowly documented inventory coverage
   updates, preserving the no-stats/no-ML contract.
3. If exp28 confirms the new support, write a new mechanistic audit design
   before coding any generator profile.
4. Run the mechanistic audit in `uncalibrated_raw` with unchanged metrics,
   unchanged gates, and explicit anti-leakage flags.
5. Only after a later documented mechanistic block with the new data support
   may a separate plan discuss statistical noise/residual capture. ML/DL stays
   later than that.

## Final Disposition

P2-07 is complete as a stop review. The project has exhausted the current
uncalibrated mechanistic path as far as the current data can support it, but it
has not exhausted all possible mechanistic science. The next requirement is
data support, not another support-local tweak and not a statistical or learned
fallback.

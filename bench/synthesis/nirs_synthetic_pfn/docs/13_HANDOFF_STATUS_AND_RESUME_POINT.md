# Handoff Status And Resume Point

Date: 2026-05-01

Scope: resume document for the current `bench/nirs_synthetic_pfn` work
on synthetic NIRS realism.

Active goal as of 2026-05-01 (this document supersedes the earlier
content/Y-realism interpretation): produce synthetic spectra whose
distribution is indistinguishable from a chosen real dataset's spectra,
evaluated by an adversarial discriminator on spectra alone. The goal is
defined in `18_X_REALISM_DISCRIMINATOR_STRATEGY.md`. The implementation
is in `experiments/exp32_hybrid_xrealism_discriminator.py` (single
dataset) and `experiments/exp33_panel_xrealism_discriminator.py`
(multi-dataset runner over `bench/tabpfn_paper/data`).

Doctrine note: under the new active goal, statistics and PCA and
per-channel noise capture are **explicitly part of the hybrid
generator**, and adversarial AUC IS the tuning oracle. Targets,
labels, splits-as-oracle, downstream metrics, and transfer scores
remain forbidden as inputs. `nirs4all/` is still not edited in this
lane. See `18_X_REALISM_DISCRIMINATOR_STRATEGY.md` section "Doctrine
revisions" for the full table.

Multidataset correction (still valid): the R9/P2 work summarized below
was DIESEL-only. It is useful as a strict case study, not a global
conclusion. The earlier global planning document
`14_MULTIDATASET_REALISM_REPLAN.md` and its M2 stack
(`15_M2_GENERATOR_REGIME_DESIGN.md`,
`16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md`,
`17_M2_MULTIDATASET_PREFLIGHT_DESIGN.md`,
`exp31_multidataset_evidence_gap_manifest_preflight.py`) is now a
parallel content-realism lane, off the critical path.

## Current State

The completed lane is DIESEL `uncalibrated_raw` synthetic realism. The next
lane is a multidataset realism atlas over representative datasets from
`bench/tabpfn_paper/data`.

Status:

- `R3d` / `r3d_diesel_matrix_v1` remains the accepted DIESEL baseline.
- R9e-R9m, P2a, and exp24-exp29 are diagnostic evidence only.
- The current uncalibrated mechanistic path is stopped at:
  `mechanistic_uncalibrated_blocked_pending_data_support`.
- That stop applies to the DIESEL case study only. It is a data-support stop,
  not approval to move to stats/PCA/noise capture or ML/DL.
- No global multidataset realism claim has been established yet.
- `nirs4all/` has not been modified in this lane.

Immediate scientific blocker:

- current real DIESEL/fuel rows do not expose non-empty off-support wavelength
  support outside `750-1550 nm` after alignment;
- current real cohort metadata does not bind source-detector distance,
  pathlength, collection angle, illumination angle, or equivalent geometry to
  audited spectra.

## Doctrine To Preserve

The agreed progression is:

1. exhaust explicit mechanistic uncalibrated generation first;
2. only after a data-supported mechanistic stop, consider mechanistic
   generation plus statistical/noise capture;
3. only after that is insufficient, consider ML/DL hybrid generation.

The current stop is not a data-supported mechanistic failure. It is a
missing-data-support failure. Therefore the next move is data/metadata support,
not stats/PCA/noise capture and not ML/DL.

## What Was Done

### Palier 1: R9 Mechanistic Iterations

The R9 sequence tested support-local and component-isolation variants while
keeping R3d as the reference baseline.

Main result:

- R9e showed a useful clean support-only pathlength/reference attenuation
  signal.
- R9j showed the strongest isolated residual damping signal.
- R9l combined residual damping plus clean attenuation and improved some
  diagnostics.
- R9m combined width/gain, residual damping, and clean attenuation, but was
  declared NO-GO because it gave only a small morphology improvement over R9l
  while introducing derivative-under regression.
- No R9n was created.
- No R9 profile superseded R3d.

Key report:

- `bench/nirs_synthetic_pfn/reports/r9m_tester_verification.md`

### P2-01 / exp24: Render-Stage Failure Map

Added a read-only audit:

- `bench/nirs_synthetic_pfn/experiments/exp24_diesel_render_stage_failure_map.py`
- `bench/nirs_synthetic_pfn/tests/test_exp24_diesel_render_stage_failure_map.py`

Result:

- all compared stage/profile failures remained dominated by
  `support_mean_drives_global_mean`;
- no derivative/correlation-dominant cluster justified a new mechanism;
- no profile, gate, or promotion was created.

Reports:

- `bench/nirs_synthetic_pfn/reports/exp24_diesel_render_stage_failure_map.md`
- `bench/nirs_synthetic_pfn/reports/exp24_tester_verification.md`

### P2-02 / exp25: Row-Level Pathlength/Reference Diagnostic

Added diagnostic-only P2a support:

- `p2a_diesel_row_pathlength_reference_v1` in the bench builder adapter;
- `bench/nirs_synthetic_pfn/experiments/exp25_diesel_row_pathlength_reference_audit.py`
- `bench/nirs_synthetic_pfn/tests/test_exp25_diesel_row_pathlength_reference_audit.py`

Result:

- P2a was technically valid as a diagnostic;
- on the current cohort it collapsed close to R9e because all aligned
  wavelengths are inside `750-1550 nm`;
- P2a was not promoted.

Report:

- `bench/nirs_synthetic_pfn/reports/exp25_tester_verification.md`

### P2-03 / exp26: Support/Off-Support Discriminability

Added a read-only audit:

- `bench/nirs_synthetic_pfn/experiments/exp26_diesel_support_offsupport_discriminability_audit.py`
- `bench/nirs_synthetic_pfn/tests/test_exp26_diesel_support_offsupport_discriminability_audit.py`

Result:

- current real-aligned cohort has zero off-support wavelengths, so it cannot
  distinguish R9e support-only attenuation from P2a full-row attenuation;
- a generated counterfactual prior grid with off-support wavelengths can
  distinguish them, proving the current real cohort is the limitation.

Reports:

- `bench/nirs_synthetic_pfn/reports/exp26_diesel_support_offsupport_discriminability_audit.md`
- `bench/nirs_synthetic_pfn/reports/exp26_tester_verification.md`

### P2-04: Coupled Optical-Depth/Damping Decision

Added a docs-only decision:

- `bench/nirs_synthetic_pfn/docs/10_P2B_COUPLED_OPTICAL_DEPTH_DAMPING_DECISION.md`

Result:

- P2b was not coded;
- the obvious coupling was judged too close to a covariance between known
  diagnostic knobs without a predeclared optical law;
- before any future P2b-like work, the project requires a single optical
  latent, deterministic damping, a broken-coupling ablation, anti-leakage
  flags, and real data support.

### P2-05 / exp27: Readout/Geometry Audit

Added a read-only audit:

- `bench/nirs_synthetic_pfn/experiments/exp27_diesel_readout_geometry_audit.py`
- `bench/nirs_synthetic_pfn/tests/test_exp27_diesel_readout_geometry_audit.py`

Result:

- fixed readout transforms were report-only counterfactuals;
- non-identity readouts degraded versus identity on the current raw
  comparison;
- geometry could not be audited because no row-bound source-detector or
  pathlength metadata was present.

Reports:

- `bench/nirs_synthetic_pfn/reports/exp27_diesel_readout_geometry_audit.md`
- `bench/nirs_synthetic_pfn/reports/exp27_tester_verification.md`

### P2-06 / exp28: Mechanistic Data-Support Inventory

Added a read-only inventory:

- `bench/nirs_synthetic_pfn/experiments/exp28_mechanistic_data_support_inventory.py`
- `bench/nirs_synthetic_pfn/tests/test_exp28_mechanistic_data_support_inventory.py`

Result:

- inventory rows: 415;
- 3 AOM DIESEL cohort rows and 32 local fuel file/grid rows were found;
- no parsed real-grid row extends outside `750-1550 nm` after alignment;
- no real cohort metadata header carries source-detector/pathlength/collection
  geometry;
- generic mechanistic laws and geometry terms exist in code/docs but are not
  row-bound to the audited real spectra.

Decision:

- `blocked_pending_metadata_or_wider_real_cohort_no_stats_ml`

Report:

- `bench/nirs_synthetic_pfn/reports/exp28_tester_verification.md`

### P2-07: Mechanistic Stop Review

Added stop-review documentation:

- `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`

Result:

- formally records:
  `mechanistic_uncalibrated_blocked_pending_data_support`;
- freezes the older Palier 2 queue until new data support is present;
- explicitly prohibits stats/PCA/noise capture/ML/DL while blocked;
- defines the only accepted restart path.

Related roadmap:

- `bench/nirs_synthetic_pfn/docs/09_SYNTHETIC_REALISM_ROADMAP.md`

### P2-08 / exp29: Data-Support Manifest Preflight

Added a bench-only preflight:

- `bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/tests/test_exp29_data_support_manifest_preflight.py`

Purpose:

- check a future human-authored CSV/JSON manifest before resuming mechanistic
  work;
- return only:
  - `blocked_pending_manifest_data_support_no_stats_ml`;
  - `ready_for_mechanistic_audit_design`.

Current-state result:

- rows checked: 3;
- recommendation: `blocked_pending_manifest_data_support_no_stats_ml`;
- wider-support rows: 0;
- row-bound geometry rows: 0.

Important safeguards:

- non-fuel rows cannot unblock;
- generic geometry cannot unblock;
- geometry requires row binding and real row-bound declaration;
- populated label/target/split/downstream/adversarial/AUC/transfer/gate/PCA/
  noise/ML/DL/calibration/profile/promotion fields reject rows;
- min/max/count summaries are blocked unless support/off-support counts are
  explicit and internally consistent.

Reports:

- `bench/nirs_synthetic_pfn/reports/exp29_data_support_manifest_preflight.md`
- `bench/nirs_synthetic_pfn/reports/exp29_tester_verification.md`

### M0/M1: Multidataset Real Spectral Atlas (exp30)

Added a bench-only inventory and atlas:

- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/tests/test_exp30_multidataset_real_spectral_atlas.py`

Output (regenerated locally; reports directory is gitignored):

- `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`
- `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.csv`

Scope and rules:

- file-level inventory only over the 10-dataset representative panel from
  `bench/tabpfn_paper/data/regression`;
- records per-dataset path, train/test rows, feature count, axis type (nm vs
  wavenumber vs unknown), axis direction, range, median resolution,
  separator, sampled X min/max, presence of negative or non-finite X
  tokens, target column, sentinel rows (`-999`, `-9999`, `-99`),
  metadata column inventory, split policy inferred from the dataset name,
  and preprocessing evidence;
- preprocessing status stays `unknown` unless an explicit on-disk
  documentary source is cited per row (currently only
  `COLZA/N_woOutlier`, with absorbance evidence quoted from
  `bench/tabpfn_paper/data/regression/COLZA/README.txt`);
- `N_woOutlier` is forced to `wavenumber (cm-1)` and reported as a
  descending axis per the panel rule;
- targets and split policies are descriptive identity fields only and are
  not used as a tuning oracle;
- no generator profile, mechanism, gate, promotion, threshold, metric,
  PCA, covariance, noise capture, ML, or DL step is added.

Result snapshot:

- panel size: 10; ok rows: 10;
- nm axis: 9; wavenumber axis: 1; unknown axis: 0;
- rows with negative X values in sampled rows:
  `IncombustibleMaterial/TIC_spxy70` and `ALPINE/ALPINE_P_291_KS`;
- target sentinel `-999` rows in train: 10 each in
  `ECOSIS_LeafTraits/Chla+b_spxyG_species` and
  `ECOSIS_LeafTraits/Chla+b_spxyG_block2deg`;
- metadata-rich (>=3 columns): 7; metadata-poor: 3 (BEER and
  `grapevine_chloride_556_KS` ship without metadata files);
- Phase M1 distinguishability checklist passes for nm vs wavenumber,
  raw-positive vs negative-or-processed, metadata-rich vs metadata-poor,
  and broad VIS-NIR/SWIR vs narrow instrument supports.

Validation:

- exp30 tests: `13 passed`;
- full bench tests with `PYTHONPATH=bench/nirs_synthetic_pfn/src`:
  `1355 passed` (was `1342` plus the 13 new exp30 tests);
- `ruff check bench/nirs_synthetic_pfn`: passed;
- targeted mypy on the exp30 script and test: passed;
- `git status --short nirs4all/`: empty.

Reproduce:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py \
  --panel-root bench/tabpfn_paper/data/regression \
  --report bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md \
  --csv bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.csv
```

### M2: Multidataset Evidence-Gap Preflight (exp31)

Added a bench-only preflight that consumes manifests authored against
`16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md` and follows the
contract in `17_M2_MULTIDATASET_PREFLIGHT_DESIGN.md`:

- `bench/nirs_synthetic_pfn/experiments/exp31_multidataset_evidence_gap_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/tests/test_exp31_multidataset_evidence_gap_manifest_preflight.py`

Output (regenerated locally; reports directory is gitignored):

- `bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.md`
- `bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv`

Scope and rules:

- bench-only preflight; no spectral file content read; no `nirs4all/`
  module required;
- 12-step per-row state machine in design 17 order: leakage rejection
  -> required identity -> regime_family validation ->
  `--regime-family` filter -> family-token check (substring matching
  against source/task/database_name/dataset/regime_family) ->
  cross-family token rejection (substring matching against
  source/task/database_name/dataset only, excluding the regime_family
  value itself to avoid false positives such as `mineral` inside
  `manure_organic_mineral`) -> reserved DIESEL/fuel token rejection
  -> per-family required evidence and family-specific extras (manure
  geometry descriptor or-of, manure documented-constant bulk packing,
  generic-geometry block, wavenumber breadth >= 2 sources, conditional
  pair consistency for sentinel/negative-x evidence) -> enum
  validation -> numeric validation -> documented-source path
  resolution (remote identifier prefixes accepted: http, https, ftp,
  doi, arxiv, urn);
- per-family decisions: `ready_for_phase_m3_mechanism_design_<family>`
  or `blocked_pending_<family>_evidence_no_stats_ml`; no global
  cross-family score is emitted;
- forbidden leakage field set is identical to
  `12_DATA_SUPPORT_MANIFEST_SCHEMA.md`;
- DIESEL contract in 12 and exp29 are unchanged.

Default empty-state result:

- rows checked: 0;
- per-family decisions: all five families return
  `blocked_pending_<family>_evidence_no_stats_ml`;
- leakage rejected rows: 0.

Validation:

- exp31 tests: `13 passed` (covering all 13 required test items in
  design 17);
- full bench tests with `PYTHONPATH=bench/nirs_synthetic_pfn/src`:
  `1368 passed` (was `1355` plus the 13 new exp31 tests);
- `ruff check bench/nirs_synthetic_pfn`: passed;
- targeted mypy on the exp31 script and test: passed;
- `git status --short nirs4all/`: empty.

Reproduce:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src python \
  bench/nirs_synthetic_pfn/experiments/exp31_multidataset_evidence_gap_manifest_preflight.py \
  --root . \
  --manifest /path/to/manifest.csv \
  --report bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.md \
  --csv bench/nirs_synthetic_pfn/reports/multidataset_evidence_gap_preflight.csv
```

### R0: X-Realism Hybrid Generator + Adversarial Discriminator (exp32, exp33)

Active deliverables for the new goal:

- `bench/nirs_synthetic_pfn/docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`
  -- the active strategy document.
- `bench/nirs_synthetic_pfn/experiments/exp32_hybrid_xrealism_discriminator.py`
  -- single-dataset evaluator. Hybrid generator (polynomial baseline +
  parametric peaks fit to mean spectrum; PCA on residuals from that
  mechanistic skeleton; per-channel residual noise) + adversarial AUC
  harness (RandomForest canonical, LogisticRegression linear sanity)
  with stratified shuffle splits. CLI sweeps PCA rank.
- `bench/nirs_synthetic_pfn/experiments/exp33_panel_xrealism_discriminator.py`
  -- multi-dataset runner that walks any `--root` (default
  `bench/tabpfn_paper/data`) and applies exp32 to every leaf
  `Xtrain.csv` directory. Per-dataset best-AUC table, per-dataset
  status; no global cross-dataset score is interpreted as content
  realism.
- `bench/nirs_synthetic_pfn/tests/test_exp32_hybrid_xrealism_discriminator.py`
  -- 11 tests covering the data loader (quoted-int and nm-suffix
  headers, train/test axis-mismatch rejection), generator
  fit/sample/zero-pca-mode behavior, AUC sanity (≈0.5 for two random
  halves, high for a constant synthetic pool), per-PCA-rank trend, and
  markdown/CSV contract anchors (`no labels`, `no targets`,
  `no splits-as-oracle`, no `nirs4all/` import).

Output (regenerated locally; reports directory is gitignored):

- `bench/nirs_synthetic_pfn/reports/xrealism_<dataset>.{md,csv}`
- `bench/nirs_synthetic_pfn/reports/xrealism_panel.{md,csv}`

Iteration plan (per `18_X_REALISM_DISCRIMINATOR_STRATEGY.md`):

- smoke on one dataset first, then the panel runner;
- if RF AUC plateaus high on a dataset, upgrade the generator (more
  peaks, non-Gaussian PCA score sampling, multiplicative scattering
  augment, derivative features, etc.) before adding ML/DL.

Validation:

- exp32 tests: 12 passed (includes the empirical and joint_bootstrap sampling-mode test);
- ruff and targeted mypy on exp32, exp33, exp32 tests: passed;
- `git status --short nirs4all/`: empty.

Iteration log (2026-05-02): the recommended generator is now
**knn_mixup score sampling with k=5 nearest neighbors,
Dirichlet alpha=1, Gaussian per-channel noise tail, PCA rank ~30-50**.
This is the configuration that drove RandomForest AUC closest to 0.5 on
the smoke datasets without entering the bootstrap-leakage regime
(RF AUC under 0.5).

Smoke results (subsampled ECOSIS Chla+b leaves, 1500 spectra, 196 features):

| variant | RF AUC | LR AUC | notes |
|---|---|---|---|
| v0 Gaussian | 0.9186 | 0.4694 | Linear matched, RF easy. |
| v2 GMM(K=10, full) on PCA scores | 0.7816 | 0.4655 | Best parametric joint fit. |
| v6 knn_mixup k=5 alpha=1 + jb noise | **0.5783** | 0.4776 | **Best honest result** on ECOSIS smoke. |
| v5 joint_bootstrap (jitter=0.05) | 0.2816 | 0.4840 | Bootstrap leakage (synthetic ~ real). |

Cross-dataset confirmation (manure, 343 spectra x 1003 features):

| variant | RF AUC | LR AUC | notes |
|---|---|---|---|
| v0 Gaussian | 0.7185 | 0.4562 | Manure already easier for v0 than ECOSIS. |
| v6 knn_mixup k=5 alpha=1 + jb noise | 0.5194 | 0.4711 | Within 2 points of 0.5. |
| v7 knn_mixup k=5 alpha=1 + Gauss noise | **0.5093** | 0.4724 | **Within 1 point of 0.5.** |

The full mode comparison and the chosen recipe rationale are documented
in `docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`. The full panel sweep
across `bench/tabpfn_paper/data/regression` is run via `exp33` with the
winning configuration; per-dataset best-AUC results land in
`bench/nirs_synthetic_pfn/reports/xrealism_panel_knn_winner.{md,csv}`
(reports directory is gitignored, regenerable).

### P2-09: Manifest Schema Documentation

Added the human-facing manifest schema:

- `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`

It documents:

- accepted CSV/JSON shapes;
- wider-support fields;
- row-bound geometry fields;
- forbidden leakage fields;
- minimal CSV and JSON examples;
- exp29 preflight commands;
- the meaning of the two exp29 decisions.

## Validation State

Latest verified results from the exp29 tester:

- exp29 tests: `10 passed`;
- exp28 + exp29 tests: `15 passed`;
- full bench tests: `1342 passed`, with 4 pre-existing sklearn PLS warnings;
- targeted ruff for exp29: passed;
- full `ruff check bench/nirs_synthetic_pfn`: passed;
- targeted mypy for exp29 script/test: passed;
- `git status --short nirs4all/`: empty.

Known tooling note:

- global `mypy .` is not a reliable signal right now because unrelated
  environment/Sphinx syntax issues can fail outside this bench scope;
- scoped mypy on touched exp29 files passed.

## Files To Inspect First When Resuming

Read these first:

1. `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
2. `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
3. `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`
4. `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`
5. `bench/nirs_synthetic_pfn/reports/exp29_tester_verification.md`
6. `bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py`
7. `bench/nirs_synthetic_pfn/tests/test_exp29_data_support_manifest_preflight.py`
8. `bench/nirs_synthetic_pfn/reports/exp28_tester_verification.md`
9. `bench/nirs_synthetic_pfn/experiments/exp28_mechanistic_data_support_inventory.py`

## How To Resume Globally

Do not start by coding a generator profile.

The next global step is the multidataset plan in
`14_MULTIDATASET_REALISM_REPLAN.md`:

1. build an exp30-style inventory for the representative panel — done;
2. generate a real spectral atlas over support, axis type, preprocessing
   evidence, metadata, splits, target sentinels, and value ranges — done
   (see `bench/nirs_synthetic_pfn/reports/multidataset_real_spectral_atlas.md`);
3. only after that, design generator regimes by data family — done,
   docs-only deliverable in
   `bench/nirs_synthetic_pfn/docs/15_M2_GENERATOR_REGIME_DESIGN.md`,
   no profile/gate/promotion.

Phase M2 has assigned each of the 10 panel datasets to one of five
candidate regime families (plant/leaf VIS-NIR-SWIR, manure/organic-mineral,
liquid food/fuel, mineral/incombustible material, wavenumber-domain),
listed candidate uncalibrated mechanistic phenomena per family, and
recorded per-family evidence gaps that must be closed before any Phase M3
mechanism is coded. It introduces no generator, profile, gate, promotion,
threshold, metric, PCA, covariance, noise capture, ML, or DL step.

Representative datasets for the next pass:

- `All_manure_MgO_SPXY_strat_Manure_type`
- `An_spxyG70_30_byCultivar_NeoSpectra`
- `TIC_spxy70`
- `Chla+b_spxyG_species`
- `ALPINE_P_291_KS`
- `Beer_OriginalExtract_60_YbaseSplit`
- `All_manure_Total_N_SPXY_strat_Manure_type`
- `Chla+b_spxyG_block2deg`
- `N_woOutlier`
- `grapevine_chloride_556_KS`

## How To Resume The DIESEL Case Study

The old DIESEL-specific restart sequence is still valid only if the goal is to
continue the DIESEL case study:

Approved restart sequence:

1. Provide or ingest a CSV/JSON manifest following
   `12_DATA_SUPPORT_MANIFEST_SCHEMA.md`.
2. Run exp29:

   ```bash
   PYTHONPATH=bench/nirs_synthetic_pfn/src python \
     bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py \
     --root . \
     --manifest /path/to/manifest.csv \
     --report /tmp/exp29_data_support_manifest_preflight.md \
     --csv /tmp/exp29_data_support_manifest_preflight.csv
   ```

3. If exp29 remains
   `blocked_pending_manifest_data_support_no_stats_ml`, stop. Do not move to
   stats/PCA/noise capture or ML/DL.
4. If exp29 returns `ready_for_mechanistic_audit_design`, write a separate
   mechanistic audit design document before coding any generator/profile
   change.
5. Re-run exp28 or a narrowly updated exp28 inventory to confirm the same data
   support in the broader inventory.
6. Only then implement a new mechanistic audit in `uncalibrated_raw`, with
   unchanged metrics/gates and explicit anti-leakage flags.

## What Not To Do Next

Do not:

- create R9n;
- retune R9e/R9j/R9l/R9m/P2a constants;
- create a new P2 profile to bypass missing data support;
- generalize the DIESEL stop review to all NIRS datasets;
- evaluate a new mechanism on DIESEL only and call it globally realistic;
- use labels, targets, splits, downstream metrics, adversarial AUC, transfer
  scores, PCA, covariance, quantiles, marginal matching, residual/noise
  capture, ML, or DL;
- change thresholds, metrics, gates, integration status, or production APIs;
- edit `nirs4all/` for this lane.

## Current Worktree Notes

The bench contains multiple generated or untracked artifacts from this
investigation. Important untracked source/docs currently include:

- `bench/nirs_synthetic_pfn/docs/09_SYNTHETIC_REALISM_ROADMAP.md`
- `bench/nirs_synthetic_pfn/docs/10_P2B_COUPLED_OPTICAL_DEPTH_DAMPING_DECISION.md`
- `bench/nirs_synthetic_pfn/docs/11_MECHANISTIC_STOP_REVIEW_AND_DATA_REQUIREMENTS.md`
- `bench/nirs_synthetic_pfn/docs/12_DATA_SUPPORT_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/docs/13_HANDOFF_STATUS_AND_RESUME_POINT.md`
- `bench/nirs_synthetic_pfn/docs/14_MULTIDATASET_REALISM_REPLAN.md`
- `bench/nirs_synthetic_pfn/docs/15_M2_GENERATOR_REGIME_DESIGN.md`
- `bench/nirs_synthetic_pfn/docs/16_MULTIDATASET_EVIDENCE_GAP_MANIFEST_SCHEMA.md`
- `bench/nirs_synthetic_pfn/docs/17_M2_MULTIDATASET_PREFLIGHT_DESIGN.md`
- `bench/nirs_synthetic_pfn/docs/18_X_REALISM_DISCRIMINATOR_STRATEGY.md`
- `bench/nirs_synthetic_pfn/experiments/exp28_mechanistic_data_support_inventory.py`
- `bench/nirs_synthetic_pfn/experiments/exp29_data_support_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/experiments/exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/experiments/exp31_multidataset_evidence_gap_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/experiments/exp32_hybrid_xrealism_discriminator.py`
- `bench/nirs_synthetic_pfn/experiments/exp33_panel_xrealism_discriminator.py`
- `bench/nirs_synthetic_pfn/tests/test_exp28_mechanistic_data_support_inventory.py`
- `bench/nirs_synthetic_pfn/tests/test_exp29_data_support_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/tests/test_exp30_multidataset_real_spectral_atlas.py`
- `bench/nirs_synthetic_pfn/tests/test_exp31_multidataset_evidence_gap_manifest_preflight.py`
- `bench/nirs_synthetic_pfn/tests/test_exp32_hybrid_xrealism_discriminator.py`

Reports under `bench/nirs_synthetic_pfn/reports/` are local artifacts and may
be ignored by git. Re-run the corresponding experiment scripts when exact
report regeneration matters.

## Bottom Line

Everything currently points to two conclusions:

1. For DIESEL, the mechanistic uncalibrated approach has been pushed as far as
   the current real DIESEL/fuel support allows. The next DIESEL input is either:

- a wider real DIESEL/fuel cohort with real aligned wavelength support outside
  `750-1550 nm`; or
- row-bound source-detector/pathlength/collection geometry metadata tied to
  the audited spectra.

2. For the project as a whole, the DIESEL work is too narrow. The next global
   input is a multidataset real spectral atlas over the representative panel.

Once a DIESEL-specific data input exists, exp29 is the first DIESEL gate to
run. For the active X-realism goal, start with
`18_X_REALISM_DISCRIMINATOR_STRATEGY.md`. The hybrid generator + AUC
harness in `experiments/exp32_hybrid_xrealism_discriminator.py` and the
panel runner in `experiments/exp33_panel_xrealism_discriminator.py` are
in place. The next global step is to iterate on the generator until
RandomForest AUC drops near 0.5 across a chosen smoke dataset (the
default ECOSIS Chla+b leaves), then run the panel runner across all of
`bench/tabpfn_paper/data` and document per-dataset failure modes for
the datasets where AUC stays high.

For the parallel content-realism lane, the older route is still
available via `14_MULTIDATASET_REALISM_REPLAN.md` and the M2 stack
(`15`/`16`/`17`/`exp31`). It is off the critical path under the active
goal but remains valid if the project later returns to content
realism.

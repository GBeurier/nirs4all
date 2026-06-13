# Heterogeneous Source Repetitions

This page describes the experimental relation pipeline for datasets where each
physical sample has a different number of spectral repetitions per source, for
example `MIR=2`, `RAMAN=3`, and `NIRS=2`.

The legacy `repetition=` option still describes one repetition axis shared by
the loaded feature matrix. It is the right choice when every source is already
aligned row by row. Source-aware repetitions are different: each source has its
own observation table, and nirs4all must first resolve the physical sample key
before building an aligned feature representation.

## Required Blocks

Use `experimental_relation_pipeline: true` to make the relation contract
explicit. The public YAML blocks are:

```yaml
experimental_relation_pipeline: true

repetition_spec:
  sample_id: sample_id
  link_by: sample_id
  target_level: physical_sample
  rep_order: exchangeable
  strict_cardinality: true
  missing_repetition_policy: strict
  missing_source_policy: strict
  sources:
    MIR: {rep_col: rep, expected: 2}
    RAMAN: {rep_col: rep, expected: 3}
    NIRS: {rep_col: rep, expected: 2}

representations:
  - name: per_source_aggregate
    stage: raw_before_preprocessing
    unit_level: sample

reducers:
  - role: score
    axis: unit
    input_level: observation
    output_level: sample
    method: mean

fit_influence:
  mode: auto

meta_features:
  meta_row_domain: sample
  alignment_key: physical_sample_id

refit_slots:
  - slot_id: best_sample
    mode: refit_one
    selection_level: sample
```

The schema preserves these blocks and passes them to the relation helpers. The
loader refuses ambiguous heterogeneous multisource inputs that would otherwise
be treated as positionally aligned features.

## Representation Choices

`per_source_aggregate` reduces each source to a fixed-width sample-level feature
view before normal model training. This is the simplest deployable default and
corresponds to modeling `f(E[x])`.

`per_source_observation` keeps source observations separate and is useful before
late prediction fusion. Branch predictions must then be aligned by
`physical_sample_id`, not by row position.

`cartesian_full` and `cartesian_mc` create derived combo observations. They are
valid when feature-level interactions across sources matter, but require a
bounded `CombinationPlan`, grouped CV by `physical_sample_id`, sample-level
scoring, and a declared `fit_influence` policy.

`stack_fixed` requires stable ordered repetitions. Use `stack_padded_masked`
when prediction may have missing repetitions or sources and the model can handle
mask/padding features.

## Fit Influence

Fit influence is separate from prediction reduction. `FitInfluencePolicy`
controls how much each physical sample contributes during model fitting when a
representation creates multiple derived rows.

Use:

- `uniform_rows` when every physical sample creates the same number of derived
  rows and row influence is acceptable.
- `equal_sample_influence` when cardinalities vary and the backend supports
  sample weights.
- `resample_equalized` when the backend does not support weights but samples
  must contribute equally.
- `strict_weight_support` when the run must fail instead of choosing a fallback.

Reducers still define how predictions are reduced back to the sample level for
scoring, ranking, final prediction, and export.
For `axis: fold`, `axis: model`, and `axis: metric`, the reducer is still a
grouped reduction: nirs4all supplies the appropriate output key, such as
`physical_sample_id`, and applies the declared method within each key.

## Stacking And Missing Sources

For production late fusion, prefer `meta_row_domain: sample`. It gives the meta
model one row per physical sample and keeps missing-source behavior explicit via
`missing_prediction_policy`.

`meta_row_domain: combo` is experimental. It requires an explicit final reducer
from `combo` to `sample`, strict OOF alignment, and a declared fit influence
policy. Validation-final profiles reject `reuse_oof` because it is optimistic.

## Replay And Explainability

Relation-aware runs must carry a `relation_replay_manifest` in exported `.n4a`
bundles. The manifest records staging, representation, reducer, missingness,
stacking, and fit-influence contracts with fingerprints. `BundleLoader` exposes
the loaded manifest as `loader.relation_replay_manifest`.

The high-level API also surfaces the same provenance. `RunResult` exposes
`relation_replay_manifest`, `relation_replay_manifests`,
`relation_materialization_manifest`, and `get_feature_lineage(...)` for the
best or selected relation-aware chain. `PredictResult` carries relation
metadata in `metadata` and exposes the same `relation_replay_manifest`,
`relation_materialization_manifest`, `feature_lineage`, `lineage_warning`, and
`get_feature_lineage(...)` accessors after workspace-chain, bundle, or session
prediction.

When explanations are computed on aggregated or derived features, use
`ExplainResult.explanation_level` and `ExplainResult.feature_lineage` to keep the
scientific meaning visible. A feature aggregated from MIR repetitions should not
be presented as if it were a raw wavelength from one observation.
Use `ExplainResult.get_feature_lineage("MIR:1000")` to inspect the source-level
provenance for one explained feature.
For `stack_padded_masked`, model-space headers include both value columns and
presence-mask columns such as `mask:MIR:1000`. The lineage keeps the source as
`MIR` and marks the mask column with `feature_role: presence_mask`.
For bundle replay, the materialization fingerprint describes the saved model's
feature-space contract; if a freshly materialized dataset carries its own
relation manifest, that dataset manifest takes precedence.

## Minimal Examples

For a step-by-step walkthrough that reads the shipped CSV fixtures, builds a
`RawMultiSourceDataset`, materializes `per_source_aggregate` and
`cartesian_full`, and connects the staged dataset to `rep_fusion`, see
{doc}`heterogeneous_repetitions_tutorial`.

See:

- `examples/configs/datasets/heterogeneous_repetitions_per_source_aggregate.yaml`
- `examples/configs/datasets/heterogeneous_repetitions_late_fusion.yaml`
- `examples/configs/datasets/heterogeneous_repetitions_cartesian_full.yaml`
- `examples/configs/datasets/heterogeneous_repetitions_missing_source.yaml`

Sample-config mirrors are also available for test-dataset discovery:

- `examples/sample_configs/datasets/G01_heterogeneous_per_source_aggregate.yaml`
- `examples/sample_configs/datasets/G02_heterogeneous_late_fusion.yaml`
- `examples/sample_configs/datasets/G03_heterogeneous_cartesian_full.yaml`
- `examples/sample_configs/datasets/G04_heterogeneous_missing_source.yaml`

# Repetition Nodes: `rep_to_sources`, `rep_to_pp`, and `rep_fusion`

NIRS datasets often contain repeated spectra for the same physical sample. These nodes reshape repetition-aware data so downstream steps can process repetitions as sources, preprocessing views, or relation-aware fusion inputs.

## `rep_to_sources`

Convert repetition groups into a multi-source representation.

```yaml
pipeline:
  - rep_to_sources: Sample_ID

  - branch:
      by_source: true
      steps:
        source_0:
          - class: nirs4all.operators.transforms.StandardNormalVariate
        source_1:
          - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection

  - merge:
      sources: concat

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

JSON:

```json
{
  "pipeline": [
    {
      "rep_to_sources": "Sample_ID"
    }
  ]
}
```

Python:

```python
pipeline = [
    {"rep_to_sources": "Sample_ID"},
    {"merge": {"sources": "concat"}},
    {"model": model},
]
```

## `rep_to_pp`

Convert repetition groups into preprocessing-pipeline views.

```yaml
pipeline:
  - rep_to_pp: Sample_ID

  - feature_augmentation:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

## `rep_fusion`

`rep_fusion` materializes a relation-aware repetition/source fusion plan. It is intended for advanced relation-aware staging workflows rather than ordinary `SpectroDataset` pipelines.

```yaml
pipeline:
  - rep_fusion:
      strategy: late
      aggregate: mean
```

If you only need common repetition handling, start with `rep_to_sources`, `rep_to_pp`, grouped splitters, or dataset-level aggregation.

## Related Dataset Configuration

Repetition-aware behavior depends on metadata columns that identify the physical sample. Dataset configs can provide metadata files such as `train_group`, `test_group`, `train_m`, and `test_m`.

See {doc}`/user_guide/data/aggregation` and {doc}`/user_guide/data/heterogeneous_repetitions`.

# `merge`, `merge_sources`, and `merge_predictions`

`merge` combines outputs from branches or sources. It is the node users reach for when they ask:

- how do I merge two sources?
- how do I concatenate branch features?
- how do I build stacking from branch predictions?
- how do I reassemble samples after separation branches?

## Quick Recipes

| Task | Node |
| --- | --- |
| Concatenate branch feature matrices | `{"merge": "features"}` |
| Stack branch out-of-fold predictions | `{"merge": "predictions"}` |
| Combine branch features and predictions | `{"merge": "all"}` |
| Reassemble separation branches | `{"merge": "concat"}` |
| Concatenate multi-source features | `{"merge": {"sources": "concat"}}` |
| Stack sources along a source axis | `{"merge": {"sources": "stack"}}` |
| Keep sources as a dictionary | `{"merge": {"sources": "dict"}}` |

## Merge Two Sources

```yaml
pipeline:
  - branch:
      by_source: true
      steps:
        NIR:
          - class: nirs4all.operators.transforms.StandardNormalVariate
        MIR:
          - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection

  - merge:
      sources: concat

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 12
```

With explicit source options:

```yaml
pipeline:
  - merge:
      sources:
        strategy: concat
        sources: [NIR, MIR]
        on_incompatible: error
        output_name: fused_spectra
        preserve_source_info: true
```

## Merge Branch Features

```yaml
pipeline:
  - branch:
      snv:
        - class: nirs4all.operators.transforms.StandardNormalVariate
      msc:
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection

  - merge: features

  - model:
      class: sklearn.linear_model.Ridge
```

Equivalent explicit form:

```yaml
pipeline:
  - merge:
      features: all
      include_original: false
      output_as: features
      on_missing: error
      on_shape_mismatch: error
```

## Merge Branch Predictions for Stacking

```yaml
pipeline:
  - split:
      class: sklearn.model_selection.KFold
      params:
        n_splits: 5
        shuffle: true
        random_state: 42

  - branch:
      pls:
        - class: nirs4all.operators.transforms.StandardNormalVariate
        - model:
            class: sklearn.cross_decomposition.PLSRegression
            params:
              n_components: 8
      ridge:
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
        - model:
            class: sklearn.linear_model.Ridge

  - merge: predictions

  - model:
      class: sklearn.linear_model.Ridge
      params:
        alpha: 0.1
```

The prediction merge uses out-of-fold predictions by default to avoid leakage.

## Mixed Feature + Prediction Merge

```yaml
pipeline:
  - merge:
      features:
        branches: [0, 1]
      predictions:
        branches: [2]
        models: all
      include_original: true
      output_as: features
```

Per-branch prediction config:

```yaml
pipeline:
  - merge:
      predictions:
        - branch: 0
          select: best
          metric: rmse
        - branch: 1
          select:
            top_k: 2
          metric: r2
          aggregate: mean
```

## JSON

```json
{
  "pipeline": [
    {
      "merge": {
        "sources": {
          "strategy": "concat",
          "sources": ["NIR", "MIR"],
          "on_incompatible": "error",
          "output_name": "fused_spectra",
          "preserve_source_info": true
        }
      }
    }
  ]
}
```

## Python

```python
pipeline = [
    {"branch": {"by_source": True, "steps": {
        "NIR": [SNV()],
        "MIR": [MSC()],
    }}},
    {"merge": {"sources": "concat"}},
    {"model": model},
]
```

## Supported Keys

| Key | Meaning |
| --- | --- |
| `features` | Branch features to collect: `all`, `true`, list of branch indices, or `{branches: [...]}`. |
| `predictions` | Branch predictions to collect: `all`, `true`, branch list, or per-branch configs. |
| `sources` | Source merge spec: `concat`, `stack`, `dict`, or full dict. |
| `concat` | `true` for separation-branch reassembly. |
| `include_original` | Include pre-branch features in the merged output. |
| `on_missing` | `error`, `warn`, or `skip`. |
| `on_shape_mismatch` | Shape mismatch handling. |
| `unsafe` | Disable safe OOF reconstruction for prediction merges. Use only when you understand leakage risk. |
| `output_as` | Output layout, commonly `features`. |
| `source_names` | Names for outputs when the merge creates/keeps sources. |
| `meta_feature_plan`, `stacking_fit_contract` | Advanced replayable late-fusion metadata. |

Aliases `merge_sources` and `merge_predictions` are routed to the same merge controller.

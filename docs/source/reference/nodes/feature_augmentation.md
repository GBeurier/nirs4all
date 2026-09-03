# `feature_augmentation`

`feature_augmentation` creates multiple feature views from one input. It is useful for trying several preprocessing representations of the same spectra.

## Action Modes

| `action` | Behavior |
| --- | --- |
| `extend` | Run each operator independently on the base view and keep the original views. This is the default. |
| `add` | Chain each operator on all existing views and keep previous views. |
| `replace` | Chain each operator on all existing views and discard previous views. |

## YAML

```yaml
pipeline:
  - feature_augmentation:
      - class: nirs4all.operators.transforms.StandardNormalVariate
      - class: nirs4all.operators.transforms.SavitzkyGolay
        params:
          window_length: 15
          polyorder: 2
          deriv: 1
    action: extend

  - model:
      class: sklearn.cross_decomposition.PLSRegression
      params:
        n_components: 10
```

Generate alternative feature views:

```yaml
pipeline:
  - feature_augmentation:
      _or_:
        - class: nirs4all.operators.transforms.StandardNormalVariate
        - class: nirs4all.operators.transforms.MultiplicativeScatterCorrection
        - class: nirs4all.operators.transforms.SavitzkyGolay
          params:
            deriv: 1
      pick: 2
    action: extend
```

## JSON

```json
{
  "pipeline": [
    {
      "feature_augmentation": [
        {
          "class": "nirs4all.operators.transforms.StandardNormalVariate"
        },
        {
          "class": "nirs4all.operators.transforms.SavitzkyGolay",
          "params": {
            "window_length": 15,
            "polyorder": 2,
            "deriv": 1
          }
        }
      ],
      "action": "extend"
    }
  ]
}
```

## Python

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay

pipeline = [
    {"feature_augmentation": [SNV(), SavitzkyGolay(deriv=1)], "action": "extend"},
    {"model": model},
]
```

## Related Nodes

- Use {doc}`concat_transform` when you want one wider feature matrix.
- Use {doc}`generators` when you want a search space of preprocessing combinations.

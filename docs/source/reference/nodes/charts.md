# Chart Nodes

Chart nodes save visual outputs during training and are skipped during prediction/explanation modes.

## Supported Keywords

| Keyword | Purpose |
| --- | --- |
| `chart_2d`, `2d_chart` | 2D spectra/projection chart. |
| `chart_3d`, `3d_chart` | 3D spectra/projection chart. |
| `y_chart`, `chart_y` | Target distribution chart. |
| `fold_chart`, `chart_fold`, `fold_*` | Fold visualization chart. |
| `spectra_dist`, `spectral_distribution`, `spectra_envelope` | Spectral envelope/distribution chart. |
| `augment_chart`, `augmentation_chart` | Original vs augmented spectra chart. |
| `augment_details_chart`, `augmentation_details_chart` | Detailed augmentation chart. |
| `exclusion_chart`, `chart_exclusion` | Included/excluded sample visualization. |

## YAML

```yaml
pipeline:
  - chart_2d:
      method: pca
      color_by: y

  - y_chart:
      include_excluded: true
      highlight_excluded: true
      layout: standard

  - spectral_distribution:
      max_samples: 200

  - model:
      class: sklearn.cross_decomposition.PLSRegression
```

Augmentation charts:

```yaml
pipeline:
  - sample_augmentation:
      class: nirs4all.operators.augmentation.GaussianAdditiveNoise
      params:
        sigma: 0.01

  - augment_chart:
      max_samples: 50
      alpha_original: 0.8
      alpha_augmented: 0.4
```

Exclusion chart:

```yaml
pipeline:
  - exclude:
      class: nirs4all.operators.filters.YOutlierFilter
      params:
        method: iqr

  - exclusion_chart:
      color_by: reason
      n_components: 2
```

## JSON

```json
{
  "pipeline": [
    {
      "fold_chart": {
        "color_by": "y"
      }
    },
    {
      "y_chart": {
        "include_excluded": true,
        "layout": "standard"
      }
    }
  ]
}
```

## Python

```python
pipeline = [
    {"chart_2d": {"method": "pca", "color_by": "y"}},
    {"fold_chart": {"color_by": "y"}},
    {"model": model},
]
```

See {doc}`/user_guide/visualization/index`.

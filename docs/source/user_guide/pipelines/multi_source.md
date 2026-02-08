# Multi-Source Pipelines

Work with multiple data sources (sensors, instruments, modalities) in a single pipeline.

## Overview

Multi-source datasets combine data from different origins:
- **NIR + Raman**: Complementary spectroscopy techniques
- **Portable + Benchtop**: Same sensor type, different instruments
- **Spectra + Metadata**: Spectral data with chemical markers or sensor readings

nirs4all provides two key constructs for multi-source workflows:
- **`source_branch`**: Apply different preprocessing to each source
- **`merge_sources`**: Combine source features into a unified representation

## Loading Multi-Source Data

### From Multiple Files

```python
from nirs4all.data import DatasetConfigs

dataset = DatasetConfigs([
    {"path": "nir_spectra.csv", "source_name": "NIR"},
    {"path": "raman_spectra.csv", "source_name": "Raman"},
    {"path": "markers.csv", "source_name": "markers"},
])
```

### Source Properties

Each source can have different:
- Number of features (wavelengths, channels)
- Headers (wavelength values)
- Preprocessing requirements

## Source Branching

Apply source-specific preprocessing pipelines:

```python
pipeline = [
    ShuffleSplit(n_splits=5, random_state=42),

    # Different preprocessing per source
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "Raman": [MSC(), SavitzkyGolay(window_length=11, polyorder=2)],
        "markers": [VarianceThreshold(), StandardScaler()],
    }},

    # Sources are merged automatically after source_branch
    PLSRegression(n_components=15),
]
```

### How source_branch Works

1. **Isolation**: Each source is processed independently
2. **Parallel execution**: Source pipelines run in parallel (conceptually)
3. **Type-specific steps**: Each source gets its own transformer chain
4. **Auto-merge**: By default, sources are concatenated after processing

```
Input Data
    │
    ├── NIR ──────► SNV → FirstDerivative ────┐
    │                                          │
    ├── Raman ────► MSC → SavitzkyGolay ──────├──► Merged Features
    │                                          │
    └── markers ──► VarianceThreshold ────────┘
                    → StandardScaler
```

### source_branch Syntax Variants

#### Named Sources (Recommended)
```python
{"source_branch": {
    "NIR": [SNV(), FirstDerivative()],
    "markers": [MinMaxScaler()],
}}
```

#### Indexed Sources
```python
{"source_branch": {
    0: [SNV(), FirstDerivative()],
    1: [MinMaxScaler()],
}}
```

#### Auto Mode (Same Processing Per Source)
```python
{"source_branch": "auto"}  # Each source processed independently with empty pipeline
```

#### Default Pipeline for Unlisted Sources
```python
{"source_branch": {
    "NIR": [SNV()],
    "_default_": [MinMaxScaler()],  # Applied to other sources
}}
```

### Disabling Auto-Merge

By default, sources are merged after `source_branch`. To keep them separate:

```python
{"source_branch": {
    "NIR": [SNV()],
    "markers": [StandardScaler()],
    "_merge_after_": False  # Keep sources separate
}}
```

## Merging Sources

Explicitly combine features from multiple sources:

```python
pipeline = [
    ShuffleSplit(n_splits=5, random_state=42),

    {"source_branch": {
        "NIR": [SNV()],
        "Raman": [MSC()],
    }},

    # Explicit merge with options
    {"merge_sources": "concat"},  # Horizontal concatenation

    PLSRegression(n_components=15),
]
```

### Merge Strategies

#### Concatenation (Default)
```python
{"merge_sources": "concat"}
```
Horizontally concatenates all sources: `[NIR_features | Raman_features | ...]`

#### Stacking
```python
{"merge_sources": "stack"}
```
Creates 3D array `(samples, sources, features)`. Requires uniform feature dimensions.

#### Averaging
```python
{"merge_sources": "average"}
```
Element-wise average of sources. Requires identical dimensions.

### Advanced Merge Options

#### Weighted Merging
```python
{"merge_sources": {
    "mode": "concat",
    "weights": {"NIR": 1.0, "Raman": 0.5}  # Scale Raman features by 0.5
}}
```

#### Selective Merging
```python
{"merge_sources": {
    "sources": ["NIR", "markers"],  # Exclude Raman
    "mode": "concat"
}}
```

## Complete Examples

### Example 1: NIR + Chemical Markers

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from nirs4all.operators.transforms import SNV, FirstDerivative
from nirs4all.data import DatasetConfigs
import nirs4all

# Multi-source dataset
dataset = DatasetConfigs([
    {"path": "nir_spectra.csv", "source_name": "NIR"},
    {"path": "chemical_markers.csv", "source_name": "markers"},
])

pipeline = [
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Source-specific preprocessing
    {"source_branch": {
        "NIR": [SNV(), FirstDerivative()],
        "markers": [VarianceThreshold(threshold=0.01), StandardScaler()],
    }},

    # Merge and model
    {"merge_sources": "concat"},
    PLSRegression(n_components=15),
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset)
print(f"Multi-source RMSE: {result.best_score:.4f}")
```

### Example 2: Portable vs Benchtop Instruments

```python
pipeline = [
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Instrument-specific calibration
    {"source_branch": {
        "portable": [
            # Portable needs more aggressive preprocessing
            SNV(),
            SavitzkyGolay(window_length=15, polyorder=2),
            FirstDerivative(),
        ],
        "benchtop": [
            # Benchtop is more stable
            SNV(),
        ],
    }},

    # Weighted merge (trust benchtop more)
    {"merge_sources": {
        "mode": "concat",
        "weights": {"portable": 0.7, "benchtop": 1.0}
    }},

    PLSRegression(n_components=10),
]
```

### Example 3: Hybrid Branching (Sources + Preprocessing Variants)

Combine source branching with regular pipeline branching:

```python
pipeline = [
    KFold(n_splits=3, shuffle=True, random_state=42),

    # Step 1: Source-level preprocessing
    {"source_branch": {
        "NIR": [SNV()],
        "Raman": [MSC()],
    }},

    # Step 2: Feature scaling (applied to merged features)
    MinMaxScaler(),

    # Step 3: Compare models via regular branching
    {"branch": {
        "pls": [PLSRegression(n_components=10)],
        "rf": [RandomForestRegressor(n_estimators=100)],
    }},
]

result = nirs4all.run(pipeline=pipeline, dataset=dataset)
print(f"Branches: {result.predictions.get_unique_values('branch_name')}")
```

## Sources vs Branches

Understanding the difference between sources and branches:

| Concept | Sources | Branches |
|---------|---------|----------|
| **Dimension** | Data provenance | Processing strategy |
| **Origin** | Different sensors/files | Same data, different pipelines |
| **Created by** | `DatasetConfigs` | `branch` keyword |
| **Merged by** | `merge_sources` | `merge` keyword |
| **Use case** | Multi-instrument fusion | Algorithm comparison |

```
Multi-Source Data                    Pipeline Branching

NIR data ──────┐                     Input ─────┬── SNV → PLS
               ├──► source_branch                │
Raman data ────┘                                 ├── MSC → RF
                                                 │
                                                 └── Detrend → SVR
```

## Best Practices

1. **Name your sources**: Use descriptive names like `"NIR"`, `"Raman"` instead of indices
2. **Match preprocessing to source**: Each source type has different noise characteristics
3. **Consider feature scaling**: Sources may have very different scales
4. **Test weighted merging**: Sometimes weighting sources improves performance
5. **Use variance thresholding**: Remove uninformative features from metadata sources
6. **Monitor source contributions**: Use SHAP or feature importance to understand each source's contribution

## Troubleshooting

### "merge_sources requires a dataset with feature sources"
Your dataset has only one source. Check your `DatasetConfigs` definition.

### Sources have different sample counts
All sources must have the same number of samples. Ensure your files are aligned by sample ID.

### Feature dimension mismatch for stacking
`"stack"` mode requires all sources to have the same number of features. Use `"concat"` for heterogeneous sources.

## See Also

- {doc}`/user_guide/data/loading_data` - Loading multi-source datasets
- {doc}`/user_guide/pipelines/branching` - Regular pipeline branching
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- [D04_merge_sources.py](https://github.com/GBeurier/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py) - Full example

```{seealso}
**Related Examples:**
- [U03: Multi-Source](../../../examples/user/02_data_handling/U03_multi_source.py) - Basic multi-source data handling
- [D04: Merge Sources](../../../examples/developer/01_advanced_pipelines/D04_merge_sources.py) - Advanced multi-source branching and merging
```

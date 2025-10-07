# Cross-Dataset Analysis for Multi-Machine Compatibility

## Overview

The `PreprocPCAEvaluator` now includes advanced cross-dataset analysis capabilities to evaluate which preprocessing methods bring datasets from different machines closer together, enabling better cross-prediction and transfer learning.

## New Features

### 1. Inter-Dataset Distance Metrics

The evaluator now computes three key distance metrics between dataset pairs:

- **Centroid Distance**: Euclidean distance between dataset centroids in PCA space
- **Spread Distance**: Combined covariance and sample-wise distribution distance
- **Subspace Angle**: Grassmann distance between PCA subspaces

These metrics are computed for both **raw** and **preprocessed** data, allowing comparison.

### 2. Preprocessing Improvement Scores

For each preprocessing method, the system computes:

- **Centroid Improvement**: `(raw_distance - preprocessed_distance) / raw_distance`
  - Positive values = datasets got closer
  - Negative values = datasets got farther apart

- **Spread Improvement**: Similar calculation for distribution overlap

### 3. New Visualization Methods

#### `plot_cross_dataset_distances()`
Shows how preprocessing affects inter-dataset distances with 4 subplots:
- Centroid improvement (bar chart showing which methods bring datasets closer)
- Spread improvement (similar for distribution overlap)
- Absolute centroid distances (before/after comparison)
- Absolute spread distances (before/after comparison)

#### `plot_cross_dataset_heatmap(metric='centroid_improvement')`
Creates heatmaps showing pairwise dataset distances for each preprocessing method.
- Useful for identifying which dataset pairs benefit most from each preprocessing
- Color-coded: green (closer) to red (farther apart)

#### `plot_cross_dataset_network(preproc=None)`
Network visualization showing dataset proximity:
- Nodes = datasets
- Edges = similarity (thicker = more similar/closer)
- Compare raw data vs. best preprocessing to visualize improvement

### 4. New Data Attributes

- `evaluator.cross_dataset_df_`: DataFrame with all pairwise dataset distances
- `evaluator.pp_pcas_`: Dictionary storing preprocessed PCA results

### 5. New Methods

```python
# Get summary of cross-dataset performance
summary = evaluator.get_cross_dataset_summary(metric='centroid_improvement')

# Returns DataFrame sorted by best preprocessing for cross-compatibility
```

## Usage Example

```python
from nirs4all.utils.PCA_analyzer import PreprocPCAEvaluator
from nirs4all.pipeline import PipelineRunner

# Run your pipeline with multiple datasets
runner = PipelineRunner(save_files=False, verbose=0, keep_datasets=True)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get raw and preprocessed data
datasets_raw = runner.raw_data  # {dataset_name: array}
datasets_pp = runner.pp_data    # {dataset_name: {preproc: array}}

# Create evaluator and analyze
evaluator = PreprocPCAEvaluator(r_components=12, knn=10)
evaluator.fit(datasets_raw, datasets_pp)

# Display cross-dataset analysis
if not evaluator.cross_dataset_df_.empty:
    summary = evaluator.get_cross_dataset_summary()
    print(summary)

    # Visualize
    evaluator.plot_cross_dataset_distances()
    evaluator.plot_cross_dataset_heatmap(metric='centroid_improvement')
    evaluator.plot_cross_dataset_network(preproc=None)  # Raw data

    # Show best preprocessing
    best_pp = summary.iloc[0]['preproc']
    evaluator.plot_cross_dataset_network(preproc=best_pp)
```

## Interpretation Guide

### For Cross-Machine Compatibility:

**Good indicators:**
- ✅ High `centroid_improv_mean` (>0.5)
- ✅ High `spread_improv_mean` (>0.5)
- ✅ Low `centroid_dist_pp` (<0.1)
- ✅ Low `spread_dist_pp` (<0.1)

**Bad indicators:**
- ❌ Negative improvement values (datasets got farther apart)
- ❌ High absolute distances after preprocessing
- ❌ High standard deviation in improvements (inconsistent effect)

### Recommended Preprocessing Selection:

1. **For transfer learning**: Choose preprocessing with highest centroid improvement
2. **For data harmonization**: Choose preprocessing with lowest absolute distances
3. **For robust cross-prediction**: Choose preprocessing with high improvement AND low variance

## Example Output

```
CROSS-DATASET DISTANCE SUMMARY:
        preproc  centroid_improv_mean  spread_improv_mean  centroid_dist_pp
Standardization              0.8523                0.7758             0.0234
      Centering              0.4521                0.3891             0.0891
      NoPreproc              0.0000               -0.0015             0.3253

TOP 3 PREPROCESSING FOR CROSS-DATASET COMPATIBILITY:
1. Standardization - Improvement: +0.8523
2. Centering       - Improvement: +0.4521
3. NoPreproc       - Improvement: +0.0000
```

## Technical Details

### Distance Computation

The cross-dataset distance computation uses:

1. **PCA projection**: All datasets are projected into common PCA spaces
2. **Centroid distance**: L2 norm between mean PCA scores
3. **Spread distance**: Combination of:
   - Frobenius norm of covariance difference
   - Mean minimum pairwise distance between samples

### Computational Complexity

- Cross-dataset computation: O(n_datasets² × n_preproc)
- Each distance calculation uses sampling for efficiency
- Default: 100 samples for spread distance computation

## Real-World Applications

1. **Multi-instrument studies**: Evaluate which preprocessing allows combining data from different NIRS instruments
2. **Temporal studies**: Check if preprocessing compensates for instrument drift over time
3. **Multi-site studies**: Ensure data from different laboratories can be combined
4. **Transfer learning**: Identify preprocessing that enables model trained on one instrument to predict on another

## See Also

- Original within-dataset analysis methods (preserved)
- `Q9_data_analysis.py` example script
- `test_cross_dataset_analysis.py` for synthetic data testing

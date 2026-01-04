# Pipeline Samples for Testing

This directory contains 10 comprehensive pipeline definitions (JSON and YAML) that demonstrate all nirs4all pipeline features. These samples serve as:

1. **Test cases** for the Pipeline Editor in the webapp
2. **Reference implementations** for complex pipeline configurations
3. **Validation tests** for the nirs4all pipeline engine

## Pipeline Overview

| File | Format | Features Covered |
|------|--------|------------------|
| `01_basic_regression.yaml` | YAML | y_processing, split with group, finetune_params |
| `02_feature_augmentation.json` | JSON | feature_augmentation, _or_, pick, count generators |
| `03_sample_augmentation.yaml` | YAML | sample_augmentation, multiple augmenters, count, selection |
| `04_branching_basic.json` | JSON | Named branches, multiple models per branch |
| `05_stacking_merge.yaml` | YAML | merge predictions, MetaModel, per-branch model selection |
| `06_generator_syntax.json` | JSON | _range_, _or_, _log_range_, _grid_ generators |
| `07_concat_transform.yaml` | YAML | concat_transform for multi-view feature fusion |
| `08_complex_finetune.json` | JSON | Advanced finetune_params, train_params, hyperband |
| `09_filters_splits.yaml` | YAML | sample_filter, group splits, metadata_partitioner |
| `10_complete_all_features.json` | JSON | All features combined in a single pipeline |

## Feature Coverage

### Core Keywords
- ✅ `preprocessing` - Explicit feature preprocessing
- ✅ `y_processing` - Target variable scaling/transformation
- ✅ `model` - Model training step with naming
- ✅ `name` - Custom model/step naming

### Augmentation
- ✅ `feature_augmentation` - Multiple preprocessing views with extend/add/replace
- ✅ `sample_augmentation` - Training-time data augmentation
  - transformers list
  - count parameter
  - selection (random/all/sequential)
  - random_state

### Branching & Merging
- ✅ `branch` - Named and indexed branches
- ✅ `merge` - Merge features or predictions
  - predictions selection (best, top_k, all)
  - features selection
  - output_as (features/sources)
  - on_missing handling
- ✅ `source_branch` - Per-source preprocessing
- ✅ `concat_transform` - Horizontal feature concatenation

### Splitting
- ✅ `split` keyword with explicit splitter
- ✅ `group` parameter for grouped splits
- ✅ GroupKFold, SPXYGFold, KFold, ShuffleSplit

### Filtering
- ✅ `sample_filter` with filters list
- ✅ YOutlierFilter, SpectralQualityFilter
- ✅ mode (any/all), report

### Metadata
- ✅ `metadata_partitioner` with column, branches, default

### Generators
- ✅ `_or_` - Alternative choices
- ✅ `_range_` - Numeric sequences
- ✅ `_log_range_` - Logarithmic sequences
- ✅ `_grid_` - Cartesian product
- ✅ `pick` - Combinations (unordered)
- ✅ `arrange` - Permutations (ordered)
- ✅ `count` - Limit results

### Model Configuration
- ✅ `finetune_params` with:
  - n_trials
  - approach (single/cross)
  - eval_mode (best/mean)
  - sample (grid/random/hyperband)
  - model_params (int/float ranges, categorical lists)
  - train_params (for neural networks)
- ✅ `train_params` for final training

### Stacking
- ✅ `MetaModel` with source_models
- ✅ Multi-level stacking

### Visualization
- ✅ `chart_2d`, `chart_y` with include_excluded, highlight_excluded

## Running Tests

```bash
# From examples/pipeline_samples directory
cd examples/pipeline_samples

# Run all pipeline tests
python test_all_pipelines.py

# Quick mode (skip slow pipelines)
python test_all_pipelines.py -q

# Verbose output
python test_all_pipelines.py -v 2

# Test specific pipeline
python test_all_pipelines.py -p 01_basic
```

## Using in Pipeline Editor

These samples are designed to test the webapp's Pipeline Editor capabilities:

1. **Import Test**: Try importing each JSON/YAML file
2. **Visual Verification**: Check that all steps render correctly
3. **Parameter Editing**: Verify all parameters are editable
4. **Export Test**: Export and compare with original
5. **Execution Test**: Run the pipeline from the editor

## Known Limitations

- `08_complex_finetune.json` requires TensorFlow for NICON model
- `09_filters_splits.yaml` requires dataset with metadata columns
- `10_complete_all_features.json` ideally needs multi-source data

## File Formats

### JSON Format
```json
{
  "name": "Pipeline Name",
  "description": "Description",
  "pipeline": [
    {"class": "sklearn.preprocessing.MinMaxScaler"},
    {"y_processing": {"class": "sklearn.preprocessing.StandardScaler"}},
    ...
  ]
}
```

### YAML Format
```yaml
pipeline:
  - class: sklearn.preprocessing.MinMaxScaler
  - y_processing:
      class: sklearn.preprocessing.StandardScaler
  ...
```

## Contributing

When adding new pipeline samples:

1. Cover features not yet tested
2. Include comments explaining complex steps
3. Add to this README
4. Test with `test_all_pipelines.py`

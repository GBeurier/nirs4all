# Reference Documentation

This section provides comprehensive reference documentation for nirs4all's pipeline syntax, generator syntax, and API patterns.

## Reference Examples

The reference examples are located in `examples/reference/` and provide detailed documentation of all supported syntaxes and patterns.

### R01 - Pipeline Syntax Reference

[R01_pipeline_syntax.py](https://github.com/GBeurier/nirs4all/blob/main/examples/reference/R01_pipeline_syntax.py)

Complete documentation of all pipeline declaration formats:

- **Step Formats**: Class, instance, string, dictionary
- **Step Keywords**: preprocessing, y_processing, model, feature_augmentation, sample_augmentation, concat_transform
- **Branching**: branch, merge, source_branch, merge_sources
- **Cross-validation**: Splitter configuration
- **Models**: Basic, named, finetuned, saved

Example step formats:

```python
# All equivalent ways to add MinMaxScaler
MinMaxScaler                                      # Class
MinMaxScaler()                                    # Instance
{"preprocessing": MinMaxScaler()}                 # Dict wrapper
{"class": "sklearn.preprocessing.MinMaxScaler"}   # Class path
```

### R02 - Generator Syntax Reference

[R02_generator_reference.py](https://github.com/GBeurier/nirs4all/blob/main/examples/reference/R02_generator_reference.py)

Complete documentation of generator syntax for creating multiple pipeline variants:

- **Core Generators**: `_or_`, `_range_`, `count`
- **Selection**: `pick` (combinations), `arrange` (permutations)
- **Advanced**: `_log_range_`, `_grid_`, `_zip_`, `_chain_`, `_sample_`
- **Constraints**: `_mutex_`, `_requires_`, `_exclude_`
- **Presets**: Reusable configuration templates

Example generator syntax:

```python
# Generate variants with different preprocessing
{"feature_augmentation": {
    "_or_": [SNV, MSC, Detrend, FirstDerivative],
    "pick": 2,    # Combinations of 2
    "count": 5    # Limit to 5 variants
}}

# Model parameter sweep
{"_range_": [2, 20, 2], "param": "n_components", "model": PLSRegression}
```

### R03 - All Keywords Test

[R03_all_keywords.py](https://github.com/GBeurier/nirs4all/blob/main/examples/reference/R03_all_keywords.py)

Integration test exercising ALL pipeline keywords in a single complex pipeline. Useful for:

- Verifying all keywords work together
- Understanding complex pipeline structures
- Testing after modifications

### R04 - Legacy API Reference

[R04_legacy_api.py](https://github.com/GBeurier/nirs4all/blob/main/examples/reference/R04_legacy_api.py)

Reference for the original PipelineRunner/PipelineConfigs API:

```python
# Legacy API (still supported)
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.data import DatasetConfigs

runner = PipelineRunner(verbose=1)
predictions, _ = runner.run(
    PipelineConfigs(pipeline, "MyPipeline"),
    DatasetConfigs("path/to/data")
)

# New API (recommended)
import nirs4all

result = nirs4all.run(pipeline, "path/to/data", name="MyPipeline")
```

## Quick Reference

### Pipeline Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `preprocessing` | Apply to features (X) | `{"preprocessing": StandardScaler()}` |
| `y_processing` | Apply to targets (y) | `{"y_processing": MinMaxScaler()}` |
| `model` | Model training | `{"model": PLSRegression(n_components=10)}` |
| `feature_augmentation` | Create feature views | `{"feature_augmentation": [SNV, MSC]}` |
| `sample_augmentation` | Augment samples | `{"sample_augmentation": {...}}` |
| `concat_transform` | Concatenate features | `{"concat_transform": [PCA(10), SVD(10)]}` |
| `branch` | Parallel execution | `{"branch": [[...], [...]]}` |
| `merge` | Combine branches | `{"merge": "predictions"}` |
| `source_branch` | Per-source processing | `{"source_branch": [...]}` |
| `merge_sources` | Combine sources | `{"merge_sources": "concat"}` |

### Generator Keywords

| Keyword | Purpose | Example |
|---------|---------|---------|
| `_or_` | Alternatives | `{"_or_": [A, B, C]}` |
| `_range_` | Numeric sequence | `{"_range_": [1, 20, 2]}` |
| `pick` | Combinations | `{"_or_": [...], "pick": 2}` |
| `arrange` | Permutations | `{"_or_": [...], "arrange": 2}` |
| `count` | Limit results | `{"_or_": [...], "count": 5}` |
| `_grid_` | Cartesian product | `{"_grid_": {...}}` |
| `_sample_` | Statistical sampling | `{"_sample_": {...}}` |

## Running Reference Examples

```bash
# Run all reference examples
./run.sh -c reference

# Run specific reference example
./run.sh -n "R01*.py"
./run.sh -n "R02*.py"
```

## See Also

- [Module API](api/module_api.md) - `nirs4all.run()`, `nirs4all.predict()`, `nirs4all.session()`
- [sklearn Integration](api/sklearn_integration.md) - `NIRSPipeline` wrapper
- [Examples README](https://github.com/GBeurier/nirs4all/blob/main/examples/README.md) - Complete examples index

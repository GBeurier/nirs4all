# Examples

NIRS4ALL includes comprehensive examples organized into two learning paths.

## Overview

Examples are located in the [`examples/`](https://github.com/GBeurier/nirs4all/tree/main/examples) directory of the repository.

**Important**: Examples use **per-folder numbering** - each folder starts from 01 (e.g., U01-U04 in getting_started, U01-U06 in data_handling).

::::{grid} 2
:gutter: 3

:::{grid-item-card} 👤 User Path
:class-card: sd-bg-light

Progressive examples for learning NIRS4ALL from basics to advanced usage.

+++
{bdg-primary}`Beginner` {bdg-success}`Step-by-step`
:::

:::{grid-item-card} 🔧 Developer Path
:class-card: sd-bg-light

Advanced examples for power users and contributors.

+++
{bdg-warning}`Advanced` {bdg-danger}`Internals`
:::

::::

## User Path

Progressive examples for learning NIRS4ALL. Each folder has its own numbering (U01, U02, etc.):

| Folder | Topic | Examples | Description |
|--------|-------|----------|-------------|
| `01_getting_started/` | Getting Started | U01-U04 | First steps with pipelines |
| `02_data_handling/` | Data Handling | U01-U06 | Loading, managing, and generating data |
| `03_preprocessing/` | Preprocessing | U01-U04 | Spectral preprocessing techniques |
| `04_models/` | Models | U01-U04 | Model training and comparison |
| `05_cross_validation/` | Cross-Validation | U01-U04 | Validation strategies |
| `06_deployment/` | Deployment | U01-U04 | Export and production usage |
| `07_explainability/` | Explainability | U01-U03 | SHAP and feature importance |

### Getting Started Examples (01_getting_started/)

- **U01_hello_world.py** - Simplest possible pipeline
- **U02_basic_regression.py** - Complete regression workflow
- **U03_basic_classification.py** - Classification with NIRS data
- **U04_visualization.py** - Plotting predictions and residuals

### Data Handling Examples (02_data_handling/)

- **U01_flexible_inputs.py** - Different input formats
- **U02_multi_datasets.py** - Analyze multiple datasets
- **U03_multi_source.py** - Handle multiple data sources
- **U04_wavelength_handling.py** - Wavelength interpolation and units
- **U05_synthetic_data.py** - Generate synthetic NIRS spectra
- **U06_synthetic_advanced.py** - Advanced synthetic data with builder API

### Preprocessing Examples (03_preprocessing/)

- **U01_preprocessing_basics.py** - SNV, MSC, derivatives
- **U02_feature_augmentation.py** - Multiple preprocessing views
- **U03_sample_augmentation.py** - Data augmentation for training
- **U04_signal_conversion.py** - Reflectance to absorbance

## Developer Path

Advanced examples for power users. Each folder has its own numbering (D01, D02, etc.):

| Folder | Topic | Examples | Description |
|--------|-------|----------|-------------|
| `01_advanced_pipelines/` | Advanced Pipelines | D01-D05 | Branching, merging, stacking |
| `02_generators/` | Generators & Synthetic | D01-D06 | Hyperparameter sweeps, synthetic data |
| `03_deep_learning/` | Deep Learning | D01-D04 | TF/PyTorch/JAX integration |
| `04_transfer_learning/` | Transfer Learning | D01-D03 | Model adaptation |
| `05_advanced_features/` | Advanced Features | D01-D03 | Custom extensions |
| `06_internals/` | Internals | D01-D02 | Understanding the engine |

### Branching and Stacking Examples (01_advanced_pipelines/)

- **D01_branching_basics.py** - Creating parallel preprocessing paths
- **D02_branching_advanced.py** - Complex branching patterns
- **D03_merge_basics.py** - Combining branch outputs
- **D04_merge_sources.py** - Multi-source data handling
- **D05_meta_stacking.py** - Meta-model stacking ensembles

### Generator & Synthetic Data Examples (02_generators/)

- **D01_generator_syntax.py** - Basic generator syntax
- **D02_generator_advanced.py** - Advanced generator patterns
- **D03_generator_iterators.py** - Generator iterators
- **D04_nested_generators.py** - Nested generators
- **D05_synthetic_custom_components.py** - Custom spectral components for synthetic data
- **D06_synthetic_testing.py** - Using synthetic data for testing

## Reference Examples (R01-R04)

Standalone reference examples for testing and demonstration:

| Example | Topic | Description |
|---------|-------|-------------|
| **R01_pipeline_syntax.py** | Pipeline Syntax | Complete syntax demonstration |
| **R02_generator_reference.py** | Generators | All generator keywords |
| **R03_all_keywords.py** | Keywords | All pipeline keywords |
| **R04_legacy_api.py** | Legacy API | Deprecated API examples |

## Running Examples

```bash
cd examples

# Run all examples
./run.sh

# Run single example by index
./run.sh -i 1

# Run by name pattern
./run.sh -n U01*.py

# Run user examples only
./run.sh -c user

# Run developer examples only
./run.sh -c developer

# Enable logging to log.txt
./run.sh -l

# Enable plots and show
./run.sh -p -s
```

:::{tip}
Start with the user path examples (U01-U04) to learn the basics, then explore specific topics as needed.
:::

## Example Template

All examples follow this structure:

```python
"""
Example: U01_hello_world.py
Description: Simplest possible pipeline example.
Topics: pipeline basics, running a model
"""

import nirs4all
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Define pipeline
pipeline = [
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

# Run with sample data
result = nirs4all.run(
    pipeline=pipeline,
    dataset="sample_data/regression",
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
```

## See Also

- {doc}`/getting_started/installation` - Installation guide
- {doc}`/getting_started/quickstart` - Quick start in 5 minutes
- {doc}`/getting_started/concepts` - Core concepts
- {doc}`/user_guide/index` - Detailed how-to guides
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax

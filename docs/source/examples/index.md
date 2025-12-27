# Examples

NIRS4ALL includes comprehensive examples organized into two learning paths.

## Overview

Examples are located in the [`examples/`](https://github.com/GBeurier/nirs4all/tree/main/examples) directory of the repository.

::::{grid} 2
:gutter: 3

:::{grid-item-card} 👤 User Path (U01-U27)
:class-card: sd-bg-light

Progressive examples for learning NIRS4ALL from basics to advanced usage.

+++
{bdg-primary}`Beginner` {bdg-success}`Step-by-step`
:::

:::{grid-item-card} 🔧 Developer Path (D01-D22)
:class-card: sd-bg-light

Advanced examples for power users and contributors.

+++
{bdg-warning}`Advanced` {bdg-danger}`Internals`
:::

::::

## User Path (U01-U27)

Progressive examples for learning NIRS4ALL:

| Range | Folder | Topic | Description |
|-------|--------|-------|-------------|
| U01-U04 | `01_getting_started/` | Getting Started | First steps with pipelines |
| U05-U08 | `02_data_handling/` | Data Handling | Loading and managing data |
| U09-U12 | `03_preprocessing/` | Preprocessing | Spectral preprocessing techniques |
| U13-U16 | `04_models/` | Models | Model training and comparison |
| U17-U20 | `05_cross_validation/` | Cross-Validation | Validation strategies |
| U21-U24 | `06_deployment/` | Deployment | Export and production usage |
| U25-U27 | `07_explainability/` | Explainability | SHAP and feature importance |

### Getting Started Examples

- **U01_hello_world.py** - Simplest possible pipeline
- **U02_basic_regression.py** - Complete regression workflow
- **U03_basic_classification.py** - Classification with NIRS data
- **U04_visualization.py** - Plotting predictions and residuals

### Preprocessing Examples

- **U09_preprocessing_basics.py** - SNV, MSC, derivatives
- **U10_feature_augmentation.py** - Multiple preprocessing views
- **U11_sample_augmentation.py** - Data augmentation for training
- **U12_signal_conversion.py** - Reflectance to absorbance

## Developer Path (D01-D22)

Advanced examples for power users:

| Range | Folder | Topic | Description |
|-------|--------|-------|-------------|
| D01-D05 | `01_advanced_pipelines/` | Advanced Pipelines | Branching, merging, stacking |
| D06-D09 | `02_generators/` | Generators | Hyperparameter sweeps |
| D10-D13 | `03_deep_learning/` | Deep Learning | TF/PyTorch/JAX integration |
| D14-D16 | `04_transfer_learning/` | Transfer Learning | Model adaptation |
| D17-D19 | `05_advanced_features/` | Advanced Features | Custom extensions |
| D20-D22 | `06_internals/` | Internals | Understanding the engine |

### Branching and Stacking Examples

- **D01_branching_basics.py** - Creating parallel preprocessing paths
- **D02_branching_advanced.py** - Complex branching patterns
- **D03_merge_basics.py** - Combining branch outputs
- **D04_merge_sources.py** - Multi-source data handling
- **D05_meta_stacking.py** - Meta-model stacking ensembles

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

- {doc}`/getting_started/index` - Installation and quick start
- {doc}`/user_guide/index` - Detailed how-to guides
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax

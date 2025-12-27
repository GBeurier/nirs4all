# Getting Started

Welcome to NIRS4ALL! This section will help you get up and running quickly.

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
concepts
```

## Overview

NIRS4ALL is designed to make Near-Infrared Spectroscopy data analysis accessible to everyone. Whether you're a spectroscopy expert or new to the field, this guide will help you:

1. **Install** the library and its dependencies
2. **Run** your first pipeline in minutes
3. **Understand** the core concepts
4. **Explore** what's possible

::::{grid} 3
:gutter: 3

:::{grid-item-card} 📦 Installation
:link: installation
:link-type: doc

Install NIRS4ALL and verify your setup.

+++
{bdg-primary}`First Step`
:::

:::{grid-item-card} 🚀 Quickstart
:link: quickstart
:link-type: doc

Your first pipeline in 5 minutes.

+++
{bdg-success}`5 Minutes`
:::

:::{grid-item-card} 💡 Core Concepts
:link: concepts
:link-type: doc

Understand pipelines, datasets, and results.

+++
{bdg-info}`Essential`
:::

::::

## Installation

### Basic Installation

```bash
pip install nirs4all
```

This installs the core library with scikit-learn support. Deep learning frameworks are optional.

### With Additional ML Frameworks

```bash
# With TensorFlow support (CPU)
pip install nirs4all[tensorflow]

# With TensorFlow support (GPU)
pip install nirs4all[gpu]

# With PyTorch support
pip install nirs4all[torch]

# With all ML frameworks
pip install nirs4all[all]
```

### Development Installation

For developers who want to contribute:

```bash
git clone https://github.com/gbeurier/nirs4all.git
cd nirs4all
pip install -e .[dev]
```

### Verify Installation

```bash
nirs4all --test-install
```

## Quick Start (5 Minutes)

Here's a complete example to get you started:

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# Define and run a pipeline in one step
result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),                          # Scale features to [0, 1]
        ShuffleSplit(n_splits=3),                # 3-fold cross-validation
        {"model": PLSRegression(n_components=10)} # PLS model
    ],
    dataset="path/to/your/data",                 # Your spectral data
    verbose=1                                     # Show progress
)

# Check the results
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")

# Export the best model for deployment
result.export("exports/my_first_model.n4a")
```

## Core Concepts

### 1. Pipelines

A pipeline is a sequence of operations applied to your data:

```python
pipeline = [
    MinMaxScaler(),                    # Preprocessing
    SNV(),                             # NIRS-specific preprocessing
    KFold(n_splits=5),                 # Cross-validation
    {"model": PLSRegression(10)}       # Model training
]
```

### 2. Datasets

NIRS4ALL automatically loads data from various formats:

```python
# From a folder containing CSV files
result = nirs4all.run(pipeline, dataset="data/wheat/")

# From specific files
result = nirs4all.run(pipeline, dataset="data/spectra.csv")

# With explicit configuration
from nirs4all.data import DatasetConfigs
dataset = DatasetConfigs(
    path="data/wheat/",
    target_column="protein",
    wavelength_start=900,
    wavelength_end=2500
)
```

### 3. Results

The `result` object contains everything about your run:

```python
# Access metrics
print(result.best_rmse)
print(result.best_r2)

# Get predictions
predictions = result.predictions

# Export for deployment
result.export("exports/best_model.n4a")

# Use for new predictions
y_pred = result.predict(X_new)
```

## Next Steps

::::{grid} 2
:gutter: 3

:::{grid-item-card} 📖 User Guide
:link: /user_guide/index
:link-type: doc

Learn about preprocessing, stacking, and deployment.
:::

:::{grid-item-card} 📚 Reference
:link: /reference/pipeline_syntax
:link-type: doc

Complete pipeline syntax and operator catalog.
:::

:::{grid-item-card} 📝 Examples
:link: /examples/index
:link-type: doc

50+ working examples organized by topic.
:::

:::{grid-item-card} 🔧 Developer Guide
:link: /developer/index
:link-type: doc

Architecture and extending the library.
:::

::::

## See Also

- {doc}`/user_guide/preprocessing/index` - NIRS-specific preprocessing techniques
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference
- {doc}`/examples/index` - Working examples for all features

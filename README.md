<div align="center">

<img src="docs/source/assets/nirs4all_logo.png" width="300" alt="NIRS4ALL Logo">
<img src="docs/source/assets/logo-cirad-en.jpg" width="300" alt="CIRAD Logo">

# NIRS4ALL

**A comprehensive Python library for Near-Infrared Spectroscopy data analysis**

[![PyPI version](https://img.shields.io/pypi/v/nirs4all.svg)](https://pypi.org/project/nirs4all/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: CeCILL-2.1](https://img.shields.io/badge/license-CeCILL--2.1-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[Documentation](https://nirs4all.readthedocs.io) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Examples](examples/) •
[Contributing](CONTRIBUTING.md)

</div>

---

## Overview

NIRS4ALL bridges the gap between spectroscopic data and machine learning by providing a unified framework for **data loading**, **preprocessing**, **model training**, and **evaluation**. Built for researchers and practitioners working with Near-Infrared Spectroscopy data.

<div align="center">
<img src="docs/source/assets/pipeline.jpg" width="400" alt="Performance Heatmap">
</div>

### Key Features

- **NIRS-Specific Preprocessing** — SNV, MSC, Savitzky-Golay, derivatives, and 25+ spectral transforms
- **Multi-Backend ML** — Seamless integration with scikit-learn, TensorFlow, PyTorch, and JAX
- **Declarative Pipelines** — Define complex workflows with simple, readable syntax
- **Hyperparameter Tuning** — Built-in Optuna integration for automated optimization
- **Rich Visualizations** — Performance heatmaps, candlestick plots, SHAP explanations
- **Model Deployment** — Export trained pipelines as portable `.n4a` bundles
- **sklearn Compatible** — `NIRSPipeline` wrapper for SHAP, cross-validation, and more

<div align="center">
<img src="docs/source/assets/heatmap.png" width="400" alt="Performance Heatmap">
<img src="docs/source/assets/candlestick.png" width="400" alt="Performance Distribution">
<img src="docs/source/assets/stacking.png" width="400" alt="Regression Scatter Plot">
<br><em>Advanced visualization capabilities for model performance analysis</em>
</div>

---

## Installation

### Basic Installation

```bash
pip install nirs4all
```

This installs the core library with scikit-learn support. Deep learning frameworks are optional.

### With ML Backends

```bash
# TensorFlow
pip install nirs4all[tensorflow]

# PyTorch
pip install nirs4all[torch]

# JAX
pip install nirs4all[jax]

# All frameworks
pip install nirs4all[all]

# All frameworks with GPU support
pip install nirs4all[all-gpu]
```

### Development Installation

```bash
git clone https://github.com/GBeurier/nirs4all.git
cd nirs4all
pip install -e ".[dev]"
```

### Verify Installation

```bash
nirs4all --test-install      # Check dependencies
nirs4all --test-integration  # Run integration tests
nirs4all --version           # Check version
```

---

## Quick Start

### Simple API (Recommended)

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression

# Define your pipeline
pipeline = [
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# Train and evaluate
result = nirs4all.run(
    pipeline=pipeline,
    dataset="path/to/your/data",
    name="MyPipeline",
    verbose=1
)

# Access results
print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")

# Export for deployment
result.export("exports/best_model.n4a")
```

### Session for Multiple Runs

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

with nirs4all.session(verbose=1, save_artifacts=True) as s:
    # Compare models with shared configuration
    pls_result = nirs4all.run(
        pipeline=[MinMaxScaler(), PLSRegression(n_components=10)],
        dataset="data/wheat.csv",
        name="PLS",
        session=s
    )

    rf_result = nirs4all.run(
        pipeline=[MinMaxScaler(), RandomForestRegressor(n_estimators=100)],
        dataset="data/wheat.csv",
        name="RandomForest",
        session=s
    )

    print(f"PLS: {pls_result.best_rmse:.4f} | RF: {rf_result.best_rmse:.4f}")
```

### sklearn Integration with SHAP

```python
import nirs4all
from nirs4all.sklearn import NIRSPipeline
import shap

# Train with nirs4all
result = nirs4all.run(pipeline, dataset)

# Wrap for sklearn compatibility
pipe = NIRSPipeline.from_result(result)

# Use with SHAP
explainer = shap.Explainer(pipe.predict, X_background)
shap_values = explainer(X_test)
shap.summary_plot(shap_values)
```

---

## Pipeline Syntax

NIRS4ALL uses a declarative syntax for defining pipelines:

```python
from nirs4all.operators.transforms import SNV, SavitzkyGolay, FirstDerivative

pipeline = [
    # Preprocessing
    MinMaxScaler(),
    SNV(),
    SavitzkyGolay(window_length=11, polyorder=2),

    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Cross-validation
    ShuffleSplit(n_splits=5, test_size=0.2),

    # Models to compare
    {"model": PLSRegression(n_components=10)},
    {"model": RandomForestRegressor(n_estimators=100)},

    # Neural network with training parameters
    {
        "model": nicon,
        "name": "NICON-CNN",
        "train_params": {"epochs": 100, "patience": 20}
    }
]
```

### Advanced Features

```python
# Feature augmentation - generate preprocessing combinations
{
    "feature_augmentation": {
        "_or_": [SNV, FirstDerivative, SavitzkyGolay],
        "size": [1, (1, 2)],
        "count": 5
    }
}

# Hyperparameter optimization
{
    "model": PLSRegression(),
    "finetune_params": {
        "n_trials": 50,
        "model_params": {"n_components": ("int", 1, 30)}
    }
}

# Branching for parallel preprocessing paths
{
    "branch": [
        [SNV(), PLSRegression(n_components=10)],
        [MSC(), RandomForestRegressor()]
    ]
}

# Merge branch outputs (stacking)
{"merge": "predictions"}
```

---

## Available Transforms

### NIRS-Specific

| Transform | Description |
|-----------|-------------|
| `SNV` / `StandardNormalVariate` | Standard Normal Variate normalization |
| `RNV` / `RobustNormalVariate` | Robust Normal Variate (outlier-resistant) |
| `MSC` / `MultiplicativeScatterCorrection` | Multiplicative Scatter Correction |
| `SavitzkyGolay` | Smoothing and derivative computation |
| `FirstDerivative` / `SecondDerivative` | Spectral derivatives |
| `Detrend` | Remove linear/polynomial trends |
| `Gaussian` | Gaussian smoothing |
| `Haar` | Haar wavelet decomposition |

### Signal Processing

| Transform | Description |
|-----------|-------------|
| `Baseline` | Baseline correction |
| `ReflectanceToAbsorbance` | Convert R to A using Beer-Lambert |
| `Resampler` | Wavelength interpolation |
| `CARS` / `MCUVE` | Feature selection methods |

### Splitting Methods

| Splitter | Description |
|----------|-------------|
| `KennardStone` | Kennard-Stone algorithm |
| `SPXY` | Sample set Partitioning based on X and Y |
| `KMeansSplit` | K-means clustering based split |

See [Preprocessing Guide](docs/user_guide/preprocessing.md) for complete reference.

---

## Examples

The `examples/` directory is organized by topic:

### User Examples (`examples/user/`)

| Category | Examples |
|----------|----------|
| **Getting Started** | Hello world, basic regression, classification, visualization |
| **Data Handling** | Multi-source, data loading, metadata |
| **Preprocessing** | SNV, MSC, derivatives, custom transforms |
| **Models** | Multi-model, hyperparameter tuning, stacking, PLS variants |
| **Cross-Validation** | KFold, group splits, nested CV |
| **Deployment** | Export, prediction, workspace management |
| **Explainability** | SHAP basics, sklearn integration, feature selection |

### Reference Examples (`examples/reference/`)

Complete syntax reference and advanced pipeline patterns.

Run examples:

```bash
cd examples
./run.sh              # Run all
./run.sh -i 1         # Run by index
./run.sh -n "U01*"    # Run by pattern
```

---

## Documentation

| Section | Description |
|---------|-------------|
| [**User Guide**](docs/user_guide/) | Preprocessing, API migration, augmentation |
| [**API Reference**](docs/api/) | Module-level API, sklearn integration, data handling |
| [**Specifications**](docs/specifications/) | Pipeline syntax, config format, metrics |
| [**Explanations**](docs/explanations/) | SHAP, resampling, SNV theory |

Full documentation: [nirs4all.readthedocs.io](https://nirs4all.readthedocs.io)

---

## Research Applications

NIRS4ALL has been used in published research:

> **Houngbo, M. E., et al. (2024).** *Convolutional neural network allows amylose content prediction in yam (Dioscorea alata L.) flour using near infrared spectroscopy.* **Journal of the Science of Food and Agriculture, 104(8), 4915-4921.** John Wiley & Sons, Ltd.

---

## Citation

If you use NIRS4ALL in your research, please cite:

```bibtex
@software{beurier2025nirs4all,
  author = {Gregory Beurier and Denis Cornet and Lauriane Rouan},
  title = {NIRS4ALL: Open spectroscopy for everyone},
  url = {https://github.com/GBeurier/nirs4all},
  version = {0.6.0},
  year = {2025},
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- [Report bugs](https://github.com/GBeurier/nirs4all/issues)
- [Request features](https://github.com/GBeurier/nirs4all/issues)
- [Improve documentation](docs/)

---

## License

This project is licensed under the [CeCILL-2.1 License](LICENSE) — a French free software license compatible with GPL.

---

## Acknowledgments

- [CIRAD](https://www.cirad.fr/) for supporting this research
- The open-source scientific Python community

<div align="center">
<br>
<strong>Made for the spectroscopy community</strong>
</div>

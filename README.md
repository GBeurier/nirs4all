<div align="center">
<img src="docs/nirs4all_logo.png" width="300" alt="NIRS4ALL Logo">

[![PyPI version](https://img.shields.io/pypi/v/nirs4all.svg)](https://pypi.org/project/nirs4all/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: CECILL-2.1](https://img.shields.io/badge/license-CECILL--2.1-blue.svg)](LICENSE)
</div>
<!-- [![Build](https://github.com/gbeurier/nirs4all/actions/workflows/CI.yml/badge.svg)](https://github.com/gbeurier/nirs4all/actions/workflows/CI.yml) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/nirs4all/badge/?version=latest)](https://nirs4all.readthedocs.io/) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567) -->

NIRS4ALL is a comprehensive machine learning library specifically designed for Near-Infrared Spectroscopy (NIRS) data analysis. It bridges the gap between spectroscopic data and machine learning by providing a unified framework for data loading, preprocessing, model training, and evaluation.

<!-- <img src="docs/pipeline.jpg" width="700" alt="NIRS4ALL Pipeline"> -->

## What is Near-Infrared Spectroscopy (NIRS)?

Near-Infrared Spectroscopy (NIRS) is a rapid and non-destructive analytical technique that uses the near-infrared region of the electromagnetic spectrum (approximately 700-2500 nm). NIRS measures how near-infrared light interacts with the molecular bonds in materials, particularly C-H, N-H, and O-H bonds, providing information about the chemical composition of samples.

### Key advantages of NIRS:
- Non-destructive analysis
- Minimal sample preparation
- Rapid results (seconds to minutes)
- Potential for on-line/in-line implementation
- Simultaneous measurement of multiple parameters

### Common applications:
- Agriculture: soil analysis, crop quality assessment
- Food industry: quality control, authenticity verification
- Pharmaceutical: raw material verification, process monitoring
- Medical: tissue monitoring, brain imaging
- Environmental: pollutant detection, water quality monitoring

### Notes:

NIRS4All is in active development; APIs and docs are subject to change.
Pre-1.0 notice: interfaces and documentation may change without notice.

## Features

NIRS4ALL offers a wide range of functionalities:

1. **Spectrum Preprocessing**:
   - Baseline correction
   - Standard normal variate (SNV)
   - Robust normal variate
   - Savitzky-Golay filtering
   - Normalization
   - Detrending
   - Multiplicative scatter correction
   - Derivative computation
   - Gaussian filtering
   - Haar wavelet transformation
   - And more...

2. **Data Splitting Methods**:
   - Kennard Stone
   - SPXY
   - Random sampling
   - Stratified sampling
   - K-means
   - And more...

3. **Model Integration**:
   - Scikit-learn models
   - TensorFlow/Keras models
   - Pre-configured neural networks dedicated to the NIRS: nicon & decon (see publication below)
   - PyTorch models (via extensions)
   - JAX models (via extensions)

4. **Model Fine-tuning**:
   - Hyperparameter optimization with Optuna
   - Grid search and random search
   - Cross-validation strategies

5. **Visualization**:
   - Preprocessing effect visualization
   - Model performance visualization
   - Feature importance analysis
   - Classification metrics
   - Residual analysis

<div align="center">
<img src="docs/heatmap.png" width="400" alt="Performance Heatmap">
<img src="docs/candlestick.png" width="400" alt="Performance Distribution">
<br><em>Advanced visualization capabilities for model performance analysis</em>
</div>

## Installation

### Basic Installation

```bash
pip install nirs4all
```
# Install TensorFlow cpu support by default

### With Additional ML Frameworks

```bash


# With PyTorch support
pip install nirs4all[torch]

# With Keras support
pip install nirs4all[keras]

# With JAX support
pip install nirs4all[jax]

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

## Installation Testing

After installing `nirs4all`, you can verify your installation and environment using the built-in CLI test commands:

```bash
# Basic installation test: checks required dependencies and versions
nirs4all --test-install

# Integration test: runs sklearn, tensorflow, and optuna pipelines on sample data
nirs4all --test-integration

# Check version
nirs4all --version
```

Each command will print a summary of the test results and alert you to any missing dependencies or issues with your environment.


## Quick Start

### Basic Pipeline Example

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from nirs4all.dataset import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transformations import (
    StandardNormalVariate, SavitzkyGolay, MultiplicativeScatterCorrection
)

# Define your processing pipeline
pipeline = [
    MinMaxScaler(),                    # Scale features
    StandardNormalVariate(),           # Apply SNV transformation
    ShuffleSplit(n_splits=3),         # 3-fold cross-validation
    {"y_processing": MinMaxScaler()}, # Scale target values
    {"model": PLSRegression(n_components=10)},
    {"model": RandomForestRegressor(n_estimators=100)},
]

# Create configurations
pipeline_config = PipelineConfigs(pipeline, name="MyPipeline")
dataset_config = DatasetConfigs("path/to/your/data")

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1)
predictions, predictions_per_datasets = runner.run(pipeline_config, dataset_config)

# Analyze results
top_models = predictions.top_k(5, 'rmse')
print("Top 5 models by RMSE:")
for i, model in enumerate(top_models):
    print(f"{i+1}. {model['model_name']}: RMSE = {model['rmse']:.4f}")
```

### Advanced Pipeline with Feature Augmentation

```python
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, Gaussian, Haar
)

# Define multiple preprocessing options
preprocessors = [Detrend, FirstDerivative, Gaussian, StandardNormalVariate]

# Advanced pipeline with feature augmentation
pipeline = [
    "chart_2d",  # Generate visualization
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {
        "feature_augmentation": {
            "_or_": preprocessors,
            "size": [1, (1, 2)],  # Single and paired transformations
            "count": 7           # Generate 7 different combinations
        }
    },
    ShuffleSplit(n_splits=3, test_size=0.25),
]

# Add multiple PLS models with different components
for n_comp in range(5, 31, 5):
    pipeline.append({
        "name": f"PLS-{n_comp}_components",
        "model": PLSRegression(n_components=n_comp)
    })

# Run and analyze
pipeline_config = PipelineConfigs(pipeline, "AdvancedPipeline")
runner = PipelineRunner(save_files=False)
predictions, _ = runner.run(pipeline_config, dataset_config)
```

### Neural Network Integration

```python
from nirs4all.operators.models.cirad_tf import nicon

# Pipeline with pre-configured neural network
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=3),
    {"y_processing": MinMaxScaler()},
    {"model": PLSRegression(n_components=15)},
    {
        "model": nicon,  # Pre-configured convolutional neural network
        "name": "NICON-CNN",
        "train_params": {
            "epochs": 100,
            "patience": 20,
            "verbose": 1
        }
    }
]

pipeline_config = PipelineConfigs(pipeline, "NeuralNetworkPipeline")
runner = PipelineRunner(save_files=False, verbose=1)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Compare neural network with traditional models
top_models = predictions.top_k(3, 'rmse')
for i, model in enumerate(top_models):
    print(f"{i+1}. {model['model_name']}: RMSE = {model['rmse']:.4f}")
```

### Hyperparameter Optimization

```python
# Pipeline with automated hyperparameter tuning
pipeline = [
    MinMaxScaler(),
    StandardNormalVariate(),
    ShuffleSplit(n_splits=3),
    {"y_processing": MinMaxScaler()},
    {
        "model": PLSRegression(),
        "name": "PLS-Optimized",
        "finetune_params": {
            "n_trials": 50,
            "verbose": 1,
            "approach": "single",  # "grouped" or "single"
            "model_params": {
                'n_components': ('int', 1, 30),
            },
        }
    }
]

pipeline_config = PipelineConfigs(pipeline, "OptimizedPipeline")
runner = PipelineRunner(save_files=False, verbose=1)
predictions, _ = runner.run(pipeline_config, dataset_config)

# Get the best optimized model
best_model = predictions.top_k(1, 'rmse')[0]
print(f"Best model: {best_model['model_name']} with RMSE: {best_model['rmse']:.4f}")
```

### Visualization and Analysis

```python
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
import matplotlib.pyplot as plt

# Create analyzer for your predictions
analyzer = PredictionAnalyzer(predictions)

# Plot top performing models
fig1 = analyzer.plot_top_k_comparison(k=5, rank_metric='rmse')
plt.title('Top 5 Models Comparison')

# Create heatmap of model performance across preprocessing methods
fig2 = analyzer.plot_variable_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    metric='rmse'
)
plt.title('Model Performance Heatmap')

# Candlestick plot for model variability
fig3 = analyzer.plot_variable_candlestick(
    filters={"partition": "test"},
    variable="model_name"
)
plt.title('Model Performance Variability')

plt.show(block=False)
```

## Tutorials

NIRS4ALL provides comprehensive tutorials to help you master NIRS data analysis:

### 🚀 [Tutorial 1: Beginner's Guide](examples/Tutorial_1_Beginners_Guide.ipynb)
Perfect for getting started with NIRS4ALL! This tutorial covers:
- **Basic PLS Regression** - Your first NIRS pipeline
- **Enhanced Preprocessing** - Spectral data preprocessing techniques
- **Classification** - Random Forest classification examples
- **Model Persistence** - Save and reuse trained models
- **Multiple Datasets** - Cross-dataset validation and analysis
- **Data Visualization** - Create meaningful plots and charts

Start here if you're new to NIRS analysis or the NIRS4ALL framework.

### 🔬 [Tutorial 2: Advanced Analysis](examples/Tutorial_2_Advanced_Analysis.ipynb)
For experienced users ready for sophisticated techniques:
- **Multi-Source Analysis** - Multi-target regression with single datasets
- **Hyperparameter Optimization** - Automated model tuning with Optuna
- **Custom Components** - Build your own transformers and models
- **Configuration Generation** - Dynamic pipeline customization
- **Advanced Visualizations** - Professional-grade analysis dashboards
- **Neural Networks** - Deep learning with pre-configured models (nicon, decon)
- **Complete Workflows** - End-to-end professional analysis

These tutorials demonstrate real-world workflows and best practices for production-ready NIRS analysis.

## Examples

Ready-to-run example scripts demonstrating common NIRS workflows:

- **[Q1_regression.py](examples/Q1_regression.py)** - Basic regression with PLS models and preprocessing combinations
- **[Q1_classif.py](examples/Q1_classif.py)** - Classification pipeline with Random Forest and preprocessing
- **[Q2_multimodel.py](examples/Q2_multimodel.py)** - Compare multiple model types (PLS, RF, SVM) in one run
- **[Q3_finetune.py](examples/Q3_finetune.py)** - Hyperparameter optimization with Optuna
- **[Q4_multidatasets.py](examples/Q4_multidatasets.py)** - Cross-dataset validation and transfer learning
- **[Q5_predict.py](examples/Q5_predict.py)** - Load saved models and predict on new data
- **[Q6_multisource.py](examples/Q6_multisource.py)** - Multi-target regression from single dataset
- **[Q7_discretization.py](examples/Q7_discretization.py)** - Convert continuous targets to categorical
- **[Q8_shap.py](examples/Q8_shap.py)** - SHAP analysis for model interpretability
- **[Q9_acp_spread.py](examples/Q9_acp_spread.py)** - PCA-based dataset analysis and visualization
- **[Q10_resampler.py](examples/Q10_resampler.py)** - Wavelength resampling and interpolation techniques
- **[Q13_nm_headers.py](examples/Q13_nm_headers.py)** - Working with nanometer (nm) wavelength headers instead of wavenumbers (cm⁻¹)

Run any example with: `python examples/<example_name>.py`

## Documentation

### Core Documentation

- **[Preprocessing.md](docs/Preprocessing.md)** - Complete reference of transformers (nirs4all, sklearn, scipy) with usage examples
- **[CONFIG_FORMAT.md](docs/CONFIG_FORMAT.md)** - Pipeline configuration file format and structure
- **[NESTED_CROSS_VALIDATION.md](docs/NESTED_CROSS_VALIDATION.md)** - Nested CV for unbiased hyperparameter tuning
- **[PREDICTION_RESULTS_LIST.md](docs/PREDICTION_RESULTS_LIST.md)** - Understanding prediction results and metrics
- **[SHAP_EXPLANATION.md](docs/SHAP_EXPLANATION.md)** - Model interpretability with SHAP values
- **[RESAMPLER.md](docs/RESAMPLER.md)** - Wavelength resampling strategies
- **[COMBINATION_GENERATOR.md](docs/COMBINATION_GENERATOR.md)** - Feature augmentation and preprocessing combinations
- **[CROSS_DATASET_METRICS_EXPLANATION.md](docs/CROSS_DATASET_METRICS_EXPLANATION.md)** - Cross-dataset validation metrics

Full documentation will be available at [https://nirs4all.readthedocs.io/](https://nirs4all.readthedocs.io/)

## Dependencies

- numpy (>=1.20.0)
- pandas (>=1.0.0)
- scipy (>=1.5.0)
- scikit-learn (>=0.24.0)
- PyWavelets (>=1.1.0)
- joblib (>=0.16.0)
- jsonschema (>=3.2.0)
- kennard-stone (>=0.5.0)
- twinning (>=0.0.5)
- optuna (>=2.0.0)

## Optional Dependencies

- tensorflow (>=2.10.0) - For TensorFlow models
- torch (>=2.0.0) - For PyTorch models
- keras (>=3.0.0) - For Keras models
- jax (>=0.4.10) & jaxlib (>=0.4.10) - For JAX models

## Research Applications

NIRS4ALL has been successfully used in published research:

**Houngbo, M. E., Desfontaines, L., Diman, J. L., Arnau, G., Mestres, C., Davrieux, F., Rouan, L., Beurier, G., Marie‐Magdeleine, C., Meghar, K., Alamu, E. O., Otegbayo, B. O., & Cornet, D. (2024).** *Convolutional neural network allows amylose content prediction in yam (Dioscorea alata L.) flour using near infrared spectroscopy.* **Journal of the Science of Food and Agriculture, 104(8), 4915-4921.** John Wiley & Sons, Ltd.

## How to Cite

If you use NIRS4ALL in your research, please cite:

```
@software{beurier2025nirs4all,
  author = {Gregory Beurier and Denis Cornet and Camille Noûs and Lauriane Rouan},
  title = {nirs4all is all your nirs: Open spectroscopy for everyone},
  url = {https://github.com/gbeurier/nirs4all},
  version = {0.2.1},
  year = {2025},
}
```

## License

This project is licensed under the CECILL-2.1 License - see the LICENSE file for details.

## Acknowledgments

- [CIRAD](https://www.cirad.fr/) for supporting this research
- [LLMs] for providing fast documentation, nice charts, emojis in logs 😭, and plenty of useless tests, booby-trapped source code, and misleading specifications.
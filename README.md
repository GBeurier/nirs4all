<img src="docs/nirs4all_logo.png" width="300" alt="NIRS4ALL Logo">

[![PyPI version](https://img.shields.io/pypi/v/nirs4all.svg)](https://pypi.org/project/nirs4all/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: CECILL-2.1](https://img.shields.io/badge/license-CECILL--2.1-blue.svg)](LICENSE)
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
   - And more

2. **Data Splitting Methods**: 
   - Kennard Stone
   - SPXY
   - Random sampling
   - Stratified sampling
   - K-means
   - And more

3. **Model Integration**:
   - Scikit-learn models
   - TensorFlow/Keras models
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

## Installation

### Basic Installation

```bash
pip install nirs4all
```

### With Additional ML Frameworks

```bash
# With TensorFlow support
pip install nirs4all[tf]

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

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from nirs4all.data.dataset_loader import get_dataset
from nirs4all.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG
from nirs4all.core.runner import ExperimentRunner
from nirs4all.core.config import Config
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression

# Define a simple processing pipeline
pipeline = [
    RobustScaler(),  # Scale the data
    {"split": RepeatedKFold(n_splits=3, n_repeats=1)},  # Define cross-validation splits
    {"features": [None, SG, SNV, [SG, SNV]},  # Provide 4 versions of the spectra (original, Savitzky-Golay, SNV, Savgol then SNV)
    MinMaxScaler()  # Scale the data again after splitting
]

# Define scaler for y
y_scaler = MinMaxScaler()

# Create a configuration
config = Config("path/to/your/data", pipeline, y_scaler, PLSRegression(n_components=10), None, 42)

# Run the experiment
runner = ExperimentRunner(config)
datasets, predictions, scores, _ = runner.run()

# Print results
print("Model Performance:")
for i, score in enumerate(scores):
    print(f"Model {i+1}:")
    for j, fold_score in enumerate(score[:-3]):
        print(f"  Fold {j+1}: {fold_score}")
    print(f"  Mean: {score[-3]}")
    print(f"  Best: {score[-2]}")
    print(f"  Weighted Mean: {score[-1]}")
```

## Advanced Usage

For more advanced usage, please refer to the [comprehensive walkthrough notebook](examples/nirs4all_walkthrough.ipynb) which covers:

1. Data Loading and Exploration
2. Basic Processing Pipeline
3. Training scikit-learn Models
4. Training TensorFlow Models
5. Fine-tuning Models
6. Advanced Pipeline with Custom Transformations
7. Running Multiple Configurations in Parallel
8. Advanced Data Visualization
9. Transformation Effects Visualization
10. Model Performance Analysis
11. Feature Importance Analysis
12. Prediction Visualization
13. Classification Metrics
14. Residual Analysis
15. Model Deployment

## Documentation

Detailed documentation will be soon available at [https://nirs4all.readthedocs.io/](https://nirs4all.readthedocs.io/)

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

## How to Cite

If you use NIRS4ALL in your research, please cite:

```
@software{beurier2025nirs4all,
  author = {Gregory Beurier and Denis Cornet and Lauriane Rouan},
  title = {NIRS4ALL: Unlocking Spectroscopy for Everyone},
  url = {https://github.com/gbeurier/nirs4all},
  version = {0.0.1},
  year = {2025},
}
```

## License

This project is licensed under the CECILL-2.1 License - see the LICENSE file for details.

## Acknowledgments

- [CIRAD](https://www.cirad.fr/) for supporting this research
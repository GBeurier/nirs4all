# Welcome to NIRS4ALL's documentation!

<div align="center">
<img src="assets/nirs4all_logo.png" width="300" alt="NIRS4ALL Logo">
</div>

NIRS4ALL is a comprehensive machine learning library specifically designed for Near-Infrared Spectroscopy (NIRS) data analysis. It bridges the gap between spectroscopic data and machine learning by providing a unified framework for data loading, preprocessing, model training, and evaluation.

```{toctree}
:maxdepth: 2
:caption: Contents:

architecture
api/modules
```

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

# With Keras support
pip install nirs4all[keras]

# With JAX support
pip install nirs4all[jax]

# With all ML frameworks
pip install nirs4all[all]

# With all ML frameworks and GPU support for TensorFlow
pip install nirs4all[all-gpu]
```

### Development Installation

For developers who want to contribute:

```bash
git clone https://github.com/gbeurier/nirs4all.git
cd nirs4all
pip install -e .[dev]
```

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

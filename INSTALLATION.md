# NIRS4ALL Installation Guide

This guide provides step-by-step instructions for installing NIRS4ALL from scratch, including Python setup, GPU support, and installation verification.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Installation](#python-installation)
3. [NIRS4ALL Installation](#nirs4all-installation)
4. [GPU Support (Optional)](#gpu-support-optional)
5. [Installation Verification](#installation-verification)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing NIRS4ALL, you'll need:
- A computer running Windows, macOS, or Linux
- Internet connection for downloading packages
- Administrative privileges (for some installations)

## Python Installation

NIRS4ALL requires Python 3.9 or higher. If you don't have Python installed:

### Windows

1. **Download Python:**
   - Visit [python.org](https://www.python.org/downloads/)
   - Download the latest Python 3.x version (3.9+ recommended)

2. **Install Python:**
   - Run the downloaded installer
   - ⚠️ **Important:** Check "Add Python to PATH" during installation
   - Choose "Install Now" or customize installation location

3. **Verify Installation:**
   ```cmd
   python --version
   pip --version
   ```

### macOS

1. **Using Homebrew (Recommended):**
   ```bash
   # Install Homebrew if you don't have it
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Python
   brew install python
   ```

2. **Alternative - Official Installer:**
   - Download from [python.org](https://www.python.org/downloads/)
   - Run the `.pkg` installer

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Creating a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python -m venv nirs4all_env

# Activate virtual environment
# Windows:
nirs4all_env\Scripts\activate
# macOS/Linux:
source nirs4all_env/bin/activate

# Your prompt should now show (nirs4all_env)
```

## NIRS4ALL Installation

### Basic Installation

For most users, the basic installation includes all required dependencies and TensorFlow CPU support:

```bash
pip install nirs4all
```

This installs:
- All required dependencies (NumPy, Pandas, Scikit-learn, etc.)
- TensorFlow (CPU version)
- Core NIRS4ALL functionality

### Installation with Additional ML Frameworks

Choose the installation that matches your needs:

```bash
# With PyTorch support
pip install nirs4all[torch]

# With Keras support (standalone)
pip install nirs4all[keras]

# With JAX support
pip install nirs4all[jax]

# With all ML frameworks (CPU versions)
pip install nirs4all[all]
```

### Development Installation

For developers who want to contribute to NIRS4ALL:

```bash
# Clone the repository
git clone https://github.com/gbeurier/nirs4all.git
cd nirs4all

# Install in development mode
pip install -e .[dev]
```

## GPU Support (Optional)

GPU acceleration can significantly speed up deep learning models. GPU support is available for TensorFlow and PyTorch.

### Prerequisites for GPU Support

1. **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
2. **NVIDIA drivers** (latest recommended)
3. **CUDA Toolkit** (version depends on the framework)

### TensorFlow GPU Installation

For TensorFlow with GPU support:

```bash
# Option 1: Install NIRS4ALL with GPU-enabled TensorFlow
pip install nirs4all[gpu]

# Option 2: Install TensorFlow GPU separately (if you already have NIRS4ALL)
pip install tensorflow[and-cuda]
```

⚠️ **Windows GPU Warning:** Starting from TensorFlow 2.11, official GPU support for Windows has been discontinued. For Windows users with NVIDIA GPUs:
- **TensorFlow 2.10** is the last version with native Windows GPU support
- Consider using **Windows Subsystem for Linux (WSL2)** for newer TensorFlow versions with GPU
- Alternative: Use **tensorflow-cpu** and leverage other frameworks like PyTorch for GPU acceleration

```bash
# For Windows with GPU - use TensorFlow 2.10
pip install "tensorflow-gpu==2.10.*"
# OR use WSL2 with latest TensorFlow
```

### PyTorch GPU Installation

For PyTorch with GPU support, visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the exact command for your system. Example:

```bash
# Example for CUDA 11.8 (check PyTorch website for latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install NIRS4ALL
pip install nirs4all
```

### Verifying GPU Support

After installation, verify GPU detection:

```python
# TensorFlow GPU check
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# PyTorch GPU check (if installed)
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## Installation Verification

NIRS4ALL provides built-in CLI commands to verify your installation. These commands test different aspects of the installation and help identify any issues.

### Basic Installation Test

Test core dependencies and basic functionality:

```bash
nirs4all --test_install
```

This command:
- ✅ Checks Python version (requires ≥3.7)
- ✅ Verifies all required dependencies with correct versions
- ✅ Checks for optional ML frameworks (TensorFlow, PyTorch, Keras, JAX)
- ✅ Tests NIRS4ALL component imports
- ✅ Shows installation summary

**Expected output for successful installation:**
```
🔍 Testing NIRS4ALL Installation...
==================================================
✓ Python: 3.10.11

📦 Required Dependencies:
  ✓ numpy: 2.2.5
  ✓ pandas: 2.2.3
  ✓ scipy: 1.15.3
  ✓ sklearn: 1.6.1
  ✓ pywt: 1.8.0
  ✓ joblib: 1.5.0
  ✓ jsonschema: 4.23.0

🔧 Optional ML Frameworks:
  ✓ tensorflow: 2.20.0
  ⚠️ torch: Not installed
  ✓ keras: 3.11.3
  ⚠️ jax: Not installed

🎯 NIRS4ALL Components:
  ✓ nirs4all.utils.backend_utils: OK
  ✓ nirs4all.core.runner: OK
  ✓ nirs4all.data.dataset_loader: OK
  ✓ nirs4all.transformations: OK

🎉 Basic installation test PASSED!
✓ All required dependencies are available
✓ Available ML frameworks: tensorflow, keras
```

### Full Installation Test

Comprehensive test including framework functionality:

```bash
nirs4all --full_test_install
```

This command:
- 🔄 Runs the basic installation test first
- 🧠 Tests TensorFlow functionality (model creation and inference)
- 🔥 Tests PyTorch functionality (if installed)
- 📊 Tests scikit-learn integration
- 🔧 Tests NIRS4ALL transformations
- 🖥️ Checks GPU availability
- 📋 Provides detailed functionality report

### Integration Test

Complete pipeline test with real sample data:

```bash
nirs4all --test_integration
```

This command runs three different model types:
- 🌳 **Random Forest Classification** - Tests scikit-learn integration
- 📈 **PLS Regression with Fine-tuning** - Tests hyperparameter optimization (5-15 components)
- 🧠 **Simple CNN (3 epochs)** - Tests neural network functionality

**Expected output for successful integration test:**
```
🧪 NIRS4ALL Integration Test...
==================================================
Running Full Pipeline Integration Test...
==================================================

Test 1: Random Forest Classification
  ✓ Data shapes: X_train(300, 1665), Y_train(300, 1)
  ✓ Execution time: 1.25 seconds
  ✓ Model scores: {'accuracy': 0.94}
  ✓ Random Forest Classification completed successfully!

Test 2: PLS Fine-tuning
  ✓ Data shapes: X_train(130, 2151), Y_train(130, 1)
  ✓ Best parameters: {'n_components': 12}
  ✓ Model scores: {'mse': 285.3, 'r2': 0.27}
  ✓ PLS Fine-tuning completed successfully!

Test 3: Simple CNN
  ✓ Data shapes: X_train(300, 1665), Y_train(300, 1)
  ✓ Execution time: 3.45 seconds
  ✓ Model scores: {'accuracy': 0.89}
  ✓ Simple CNN completed successfully!

🎉 Integration test PASSED!
✓ All 3 pipeline tests completed successfully
✓ NIRS4ALL is ready for use!
```

### Other Useful Commands

```bash
# Check NIRS4ALL version
nirs4all --version

# Show all available CLI commands
nirs4all --help
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Not Found
```
'python' is not recognized as an internal or external command
```
**Solution:** Python not in PATH. Reinstall Python and check "Add Python to PATH", or use `python3` instead of `python`.

#### 2. Permission Denied
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```
**Solution:** Use `--user` flag or virtual environment:
```bash
pip install --user nirs4all
# OR create virtual environment (recommended)
```

#### 3. Package Conflicts
```
ERROR: pip's dependency resolver does not currently consider all the ways that packages could conflict
```
**Solution:** Use a fresh virtual environment:
```bash
python -m venv fresh_env
# Activate and install NIRS4ALL
```

#### 4. TensorFlow Installation Issues
**For Windows users with old CPUs:**
```bash
# If you get "Your CPU supports instructions that this TensorFlow binary was not compiled to use"
pip install tensorflow-cpu==2.10.0
```

#### 5. GPU Not Detected
**Check NVIDIA driver and CUDA:**
```bash
nvidia-smi  # Should show GPU info
nvcc --version  # Should show CUDA version
```

### Installation Test Failures

If `nirs4all -test_install` fails:

1. **Missing dependencies:**
   ```bash
   pip install --upgrade nirs4all
   ```

2. **Version conflicts:**
   ```bash
   pip install --upgrade --force-reinstall nirs4all
   ```

3. **Create fresh environment:**
   ```bash
   python -m venv clean_env
   # Activate environment
   pip install nirs4all
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check the test output:** Run `nirs4all -test_install` for detailed diagnostics
2. **GitHub Issues:** Report bugs at [https://github.com/gbeurier/nirs4all/issues](https://github.com/gbeurier/nirs4all/issues)
3. **Documentation:** Visit [https://nirs4all.readthedocs.io/](https://nirs4all.readthedocs.io/) (coming soon)

### Performance Tips

1. **Use virtual environments** to avoid conflicts
2. **Install GPU support** for faster deep learning models
3. **Use SSD storage** for faster data loading
4. **Ensure sufficient RAM** (8GB+ recommended for large datasets)

## Quick Start After Installation

Once installed, verify everything works:

```python
import nirs4all
from nirs4all.data.dataset_loader import get_dataset
from nirs4all.transformations import StandardNormalVariate

# Load sample data
dataset = get_dataset("binary")
print(f"Dataset loaded: {dataset.X.shape}")

# Apply transformation
snv = StandardNormalVariate()
X_transformed = snv.fit_transform(dataset.X)
print("Transformation applied successfully!")
```

🎉 **Congratulations!** NIRS4ALL is now installed and ready to use. Check out the [walkthrough notebook](examples/nirs4all_walkthrough.ipynb) for a comprehensive tutorial.
# Installation

This guide covers installing NIRS4ALL and verifying your setup.

## Quick Install

For most users, installation is a single command:

```bash
pip install nirs4all
```

This installs:
- All required dependencies (NumPy, Pandas, Scikit-learn, etc.)
- Core NIRS4ALL functionality

TensorFlow is optional and available through extras (for example:
`pip install nirs4all[tensorflow]` or `pip install nirs4all[all]`).

## Requirements

- **Python 3.11 or higher**
- pip (Python package manager)

:::{tip}
We recommend using a virtual environment to avoid package conflicts:
```bash
python -m venv nirs4all_env
source nirs4all_env/bin/activate  # Linux/macOS
# or: nirs4all_env\Scripts\activate  # Windows
pip install nirs4all
```
:::

## Installation Options

### With Additional ML Frameworks

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

### GPU Support (TensorFlow)

For GPU-accelerated deep learning:

```bash
pip install nirs4all[gpu]

# Or install TensorFlow GPU separately
pip install tensorflow[and-cuda]
```

:::{warning}
**Windows GPU Note**: Starting from TensorFlow 2.11, official GPU support for Windows has been discontinued. Windows users should either:
- Use TensorFlow 2.10: `pip install tensorflow-gpu==2.10.*`
- Use Windows Subsystem for Linux (WSL2)
- Use PyTorch for GPU acceleration instead
:::

### GPU Support (PyTorch)

For PyTorch with GPU, visit [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command for your system:

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install nirs4all
pip install nirs4all
```

### Development Installation

For contributors who want to modify the source code:

```bash
git clone https://github.com/gbeurier/nirs4all.git
cd nirs4all
pip install -e .[dev]
```

## Verify Installation

### Basic Installation Test

Test that all dependencies are correctly installed:

```bash
nirs4all --test-install
```

**Expected output** (successful installation):
```
🔍 Testing NIRS4ALL Installation...
==================================================
✓ Python: 3.11.5

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
  ✓ All components loaded successfully

🎉 Basic installation test PASSED!
```

### Integration Test

Run a complete pipeline test with real sample data:

```bash
nirs4all --test-integration
```

This runs three different pipeline types:
- **Sklearn Pipeline** - Tests PLS and RandomForest with preprocessing
- **TensorFlow Pipeline** - Tests neural network functionality
- **Optuna Pipeline** - Tests hyperparameter optimization

**Expected output**:
```
🧪 NIRS4ALL Integration Test...
==================================================
✅ PLSRegression - completed successfully (5.4s)
✅ NICON Neural Network - completed successfully (8.0s)
✅ Optuna Optimization - completed successfully (1.2s)

🎉 Integration test PASSED!
🚀 NIRS4ALL is ready for use!
```

### Verify GPU Support

To check if GPU acceleration is available:

```python
# TensorFlow GPU check
import tensorflow as tf
print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))

# PyTorch GPU check
import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
```

## Troubleshooting

### Common Installation Issues

#### Python Not Found

```
'python' is not recognized as an internal or external command
```

**Solution**:
- Reinstall Python and check "Add Python to PATH"
- Or use `python3` instead of `python`

#### Permission Denied

```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solution**: Use a virtual environment (recommended) or `--user` flag:
```bash
pip install --user nirs4all
```

#### Package Conflicts

```
ERROR: pip's dependency resolver does not currently consider all packages
```

**Solution**: Use a fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install nirs4all
```

#### TensorFlow Issues on Old CPUs

```
Your CPU supports instructions that this TensorFlow binary was not compiled to use
```

**Solution**: Install CPU-specific TensorFlow:
```bash
pip install tensorflow-cpu==2.10.0
```

#### GPU Not Detected

**Check your NVIDIA driver and CUDA**:
```bash
nvidia-smi        # Should show GPU info
nvcc --version    # Should show CUDA version
```

**Common fixes**:
1. Update NVIDIA drivers
2. Ensure CUDA version matches TensorFlow/PyTorch requirements
3. Check that `cudnn` is installed

### Installation Test Failures

If `nirs4all --test-install` fails:

1. **Upgrade nirs4all**:
   ```bash
   pip install --upgrade nirs4all
   ```

2. **Force reinstall**:
   ```bash
   pip install --upgrade --force-reinstall nirs4all
   ```

3. **Clean install** in new environment:
   ```bash
   python -m venv clean_env
   source clean_env/bin/activate
   pip install nirs4all
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the test output: `nirs4all --test-install`
2. Review [GitHub Issues](https://github.com/gbeurier/nirs4all/issues)
3. Open a new issue with your error message and system info

## Performance Tips

- **Use virtual environments** to avoid package conflicts
- **Install GPU support** for deep learning models (10-100x faster)
- **Use SSD storage** for faster data loading with large datasets
- **Ensure sufficient RAM** (8GB+ recommended for large spectral datasets)

## Next Steps

Once installed, continue to:

- {doc}`quickstart` - Run your first pipeline in 5 minutes
- {doc}`concepts` - Understand the core concepts
- {doc}`/examples/index` - Explore working examples

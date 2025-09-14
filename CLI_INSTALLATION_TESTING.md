# NIRS4ALL CLI Installation Testing

This document describes the CLI commands available for testing the NIRS4ALL installation.

## Commands

### Basic Installation Test

Test basic installation and show dependency versions:

```bash
nirs4all -test_install
```

This command will:
- Check Python version (requires >=3.7)
- Verify all required dependencies are installed with correct versions
- Check for optional ML frameworks (TensorFlow, PyTorch, Keras, JAX)
- Test that NIRS4ALL components can be imported successfully
- Display a summary of the installation status

### Full Installation Test

Perform a comprehensive test including framework functionality:

```bash
nirs4all -full_test_install
```

This command will:
- Run the basic installation test first
- Test TensorFlow functionality with model creation and inference
- Test PyTorch functionality with model creation and inference
- Test scikit-learn with model training and prediction
- Test NIRS4ALL integration with backend detection and transformations
- Check GPU availability for supported frameworks
- Provide detailed functionality test results

### Other Commands

Show version information:
```bash
nirs4all --version
```

Show help and available commands:
```bash
nirs4all --help
```

## Example Output

### Basic Installation Test Success
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

### Installation Failure Example
```
🔍 Testing NIRS4ALL Installation...
==================================================
✓ Python: 3.10.11

📦 Required Dependencies:
  ✓ numpy: 2.2.5
  ✓ pandas: 2.2.3
  ✓ scipy: 1.15.3
  ❌ sklearn: Not installed
  ❌ pywt: Not installed
  ✓ joblib: 1.5.0
  ✓ jsonschema: 4.23.0

❌ Basic installation test FAILED!
Please install missing dependencies using:
  pip install nirs4all
```

## Exit Codes

- **0**: All tests passed successfully
- **1**: Some tests failed or required dependencies are missing

## Troubleshooting

If the installation test fails:

1. **Install missing dependencies**: Run `pip install nirs4all` to install all required dependencies
2. **Check Python version**: Ensure you're using Python 3.7 or higher
3. **Virtual environment**: Consider using a virtual environment to avoid conflicts
4. **Update packages**: Some packages may need updating: `pip install --upgrade nirs4all`

For the full test (`-full_test_install`), additional optional dependencies may be needed:
- **TensorFlow**: `pip install tensorflow`
- **PyTorch**: `pip install torch`
- **JAX**: `pip install jax jaxlib`

## Implementation Details

The CLI commands are implemented in:
- Entry point: `nirs4all.cli.main:main`
- Installation testing: `nirs4all.cli.test_install`
- Tests: `tests/cli/test_test_install.py`

The commands use proper module import names (e.g., `sklearn` for scikit-learn, `pywt` for PyWavelets) and provide detailed feedback about the installation status.
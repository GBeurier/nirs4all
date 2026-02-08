# Testing Guide

Run and write tests for nirs4all development.

## Overview

nirs4all uses **pytest** for testing, with a comprehensive test suite covering:
- Unit tests for individual modules
- Integration tests for combined functionality
- End-to-end workflow tests
- Framework-specific tests (TensorFlow, PyTorch, JAX)

## Running Tests

### Run All Tests

```bash
pytest tests/
```

By default, `pytest` uses parallel workers (`-n auto --dist worksteal`) from
`pyproject.toml`.

Install test dependencies (including `pytest-xdist`) with:

```bash
pip install -r requirements-test.txt
```

### Parallel Execution

```bash
# Use all available CPU cores
pytest tests/ -n auto --dist worksteal

# Use a fixed worker count
pytest tests/ -n 4 --dist worksteal

# Disable parallel workers for local debugging
pytest tests/ -n 0
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Full integration tests (end-to-end)
pytest tests/integration_tests/
```

### Run Tests for Specific Modules

```bash
# Data module tests
pytest tests/unit/data/

# Pipeline module tests
pytest tests/unit/pipeline/

# Transform tests
pytest tests/unit/transforms/

# Model tests
pytest tests/unit/models/

# Controller tests
pytest tests/unit/controllers/

# Utility tests
pytest tests/unit/utils/
```

### Run Specific Test File

```bash
pytest tests/unit/data/test_metadata.py -v
```

### Run Specific Test Function

```bash
pytest tests/unit/data/test_metadata.py::test_metadata_creation -v
```

## Test Markers

nirs4all uses pytest markers to categorize tests by framework requirements:

### Available Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.sklearn` | Tests using scikit-learn only |
| `@pytest.mark.tensorflow` | Tests requiring TensorFlow |
| `@pytest.mark.torch` | Tests requiring PyTorch |
| `@pytest.mark.jax` | Tests requiring JAX |
| `@pytest.mark.slow` | Longer-running tests |
| `@pytest.mark.gpu` | Tests requiring GPU |

### Running Tests by Marker

```bash
# Run only sklearn tests
pytest -m sklearn

# Run TensorFlow tests
pytest -m tensorflow

# Run PyTorch tests
pytest -m torch

# Skip GPU tests
pytest -m "not gpu"

# Run sklearn OR tensorflow tests
pytest -m "sklearn or tensorflow"
```

### Skipping Framework Tests

If a framework isn't installed, its tests are automatically skipped:

```bash
$ pytest -m tensorflow
# If TensorFlow not installed: "X tests skipped"
```

## Test Coverage

### Run with Coverage Report

```bash
# Generate coverage report
pytest tests/unit/ --cov=nirs4all --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage for Specific Module

```bash
pytest tests/unit/data/ --cov=nirs4all.data --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures, pytest config
├── run_tests.py                   # Test runner script
├── README.md                      # Test documentation
│
├── unit/                          # Unit tests
│   ├── data/                      # Dataset, metadata, loaders
│   ├── pipeline/                  # Runner, config, serialization
│   ├── transforms/                # Signal processing, NIRS transforms
│   ├── models/                    # TensorFlow, PyTorch models
│   ├── controllers/               # Augmentation, split, transformer
│   └── utils/                     # Utility functions
│
├── integration/                   # Integration tests
│   └── augmentation/              # Multi-component integration
│
├── integration_tests/             # End-to-end tests
│   ├── test_basic_pipeline.py
│   ├── test_classification.py
│   ├── test_regression.py
│   └── ...
│
└── fixtures/                      # Test data
    ├── datasets/                  # Sample datasets
    └── pipelines/                 # Test configurations
```

## Writing Tests

### Basic Test Structure

```python
"""Tests for my_module."""

import pytest
from nirs4all.my_module import MyClass


class TestMyClass:
    """Test suite for MyClass."""

    def test_initialization(self):
        """Test basic initialization."""
        obj = MyClass(param=1)
        assert obj.param == 1

    def test_transform(self):
        """Test transform method."""
        obj = MyClass()
        result = obj.transform([1, 2, 3])
        assert len(result) == 3

    def test_invalid_input(self):
        """Test that invalid input raises error."""
        obj = MyClass()
        with pytest.raises(ValueError, match="Invalid"):
            obj.transform(None)
```

### Using Fixtures

```python
import pytest
import numpy as np


@pytest.fixture
def sample_spectra():
    """Create sample spectral data."""
    return np.random.randn(100, 500)


@pytest.fixture
def sample_targets():
    """Create sample target values."""
    return np.random.randn(100)


def test_with_fixtures(sample_spectra, sample_targets):
    """Test using fixtures."""
    assert sample_spectra.shape == (100, 500)
    assert len(sample_targets) == 100
```

### Framework-Specific Tests

```python
import pytest
from nirs4all.utils.backend import TF_AVAILABLE, TORCH_AVAILABLE


@pytest.mark.tensorflow
@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
def test_tensorflow_model():
    """Test TensorFlow model."""
    from nirs4all.operators.models.tensorflow.nicon import nicon
    model = nicon((100,))
    assert model is not None


@pytest.mark.torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_model():
    """Test PyTorch model."""
    from nirs4all.operators.models.pytorch.nicon import nicon
    model = nicon((100,))
    assert model is not None
```

### Testing Pipeline Execution

```python
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
import nirs4all


def test_basic_pipeline():
    """Test basic pipeline execution."""
    pipeline = [
        ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        PLSRegression(n_components=5),
    ]

    result = nirs4all.run(
        pipeline=pipeline,
        dataset="sample_data/regression",
        verbose=0
    )

    assert result.num_predictions > 0
    assert result.best_score > 0
```

### Parametrized Tests

```python
import pytest


@pytest.mark.parametrize("n_components", [1, 5, 10, 20])
def test_pls_components(n_components):
    """Test PLS with various component counts."""
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=n_components)
    assert model.n_components == n_components


@pytest.mark.parametrize("transform,expected", [
    ("SNV", "StandardNormalVariate"),
    ("MSC", "MultiplicativeScatterCorrection"),
    ("Detrend", "Detrend"),
])
def test_transform_names(transform, expected):
    """Test transform naming."""
    from nirs4all.operators import transforms
    cls = getattr(transforms, transform)
    assert expected in str(cls)
```

## Running Examples as Tests

Examples in `examples/` serve as integration tests:

```bash
cd examples

# Run all examples
./run.sh

# Run single example by index
./run.sh -i 1

# Run by name pattern
./run.sh -n "U01*.py"

# Enable logging
./run.sh -l

# Enable plots (for visual inspection)
./run.sh -p -s
```

## Test Configuration

### pyproject.toml

`pytest` is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short -n auto --dist worksteal"
markers = [
    "sklearn: sklearn-only tests",
    "tensorflow: TensorFlow tests",
    "torch: PyTorch tests",
    "jax: JAX tests",
    "slow: slow running tests",
    "gpu: GPU-requiring tests",
]
```

### VS Code Test Explorer

To make VS Code run tests in parallel, add this local workspace config:

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests",
    "-n",
    "auto",
    "--dist",
    "worksteal"
  ]
}
```

Create or update `.vscode/settings.json` with this content.
`/.vscode/` is gitignored, so this is a local developer setting.

### conftest.py

The `tests/conftest.py` configures the test environment:

```python
import matplotlib

def pytest_configure(config):
    """Configure pytest environment."""
    # Use non-interactive backend for headless testing
    matplotlib.use('Agg')
```

## Common Test Patterns

### Testing Transformers

```python
def test_snv_transform():
    """Test SNV transformer."""
    from nirs4all.operators.transforms import SNV
    import numpy as np

    X = np.random.randn(10, 100)
    snv = SNV()

    # Fit and transform
    X_transformed = snv.fit_transform(X)

    # Check shape preserved
    assert X_transformed.shape == X.shape

    # Check SNV properties (mean=0, std=1 per sample)
    np.testing.assert_array_almost_equal(
        X_transformed.mean(axis=1),
        np.zeros(10),
        decimal=10
    )
```

### Testing Controllers

```python
def test_controller_matches():
    """Test controller matching."""
    from nirs4all.controllers.transforms import TransformerController
    from sklearn.preprocessing import StandardScaler

    # Should match sklearn transformers
    assert TransformerController.matches(
        step=StandardScaler(),
        operator=StandardScaler(),
        keyword=""
    )
```

### Testing Serialization

```python
def test_pipeline_serialization():
    """Test pipeline can be serialized and deserialized."""
    from nirs4all.pipeline import PipelineConfigs
    import yaml

    pipeline = [
        StandardScaler(),
        PLSRegression(n_components=10),
    ]

    config = PipelineConfigs(pipeline, "TestPipeline")

    # Serialize
    yaml_str = yaml.dump(config.to_dict())

    # Deserialize
    loaded = yaml.safe_load(yaml_str)
    assert loaded['name'] == "TestPipeline"
```

## Debugging Tests

### Verbose Output

```bash
pytest tests/unit/data/test_metadata.py -v --tb=long
```

### Stop on First Failure

```bash
pytest tests/ -x
```

### Print Output

```bash
pytest tests/ -s
```

### Debug with pdb

```bash
pytest tests/ --pdb
```

Or in code:
```python
def test_with_debug():
    import pdb; pdb.set_trace()
    # debugging here
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Release builds

See `.github/workflows/` for CI configuration.

## Best Practices

1. **Test names**: Use descriptive names like `test_snv_normalizes_spectra`
2. **One assertion per test**: Keep tests focused
3. **Use fixtures**: Share common setup code
4. **Mark framework tests**: Use `@pytest.mark.tensorflow` etc.
5. **Test edge cases**: Empty inputs, single samples, large datasets
6. **Document test purpose**: Use docstrings
7. **Clean up**: Don't leave test files or outputs

## See Also

- {doc}`architecture` - System architecture
- [pytest documentation](https://docs.pytest.org/)

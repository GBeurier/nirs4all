# Test Suite Organization

This document describes the organization of the nirs4all test suite.

## Structure Overview

```
tests/
├── unit/                          # Unit tests for individual modules
│   ├── data/                      # Tests for data module (datasets, metadata, loaders)
│   ├── pipeline/                  # Tests for pipeline module (runner, config, serialization)
│   ├── transforms/                # Tests for transforms module (signal, NIRS, augmentation)
│   ├── models/                    # Tests for models module (TensorFlow, PyTorch)
│   ├── controllers/               # Tests for controllers (augmentation, split, transformer)
│   └── utils/                     # Tests for utilities (binning, balancing, data generation)
├── integration/                   # Integration tests for combined functionality
├── integration_tests/             # Full integration tests (end-to-end workflows)
├── workspace/                     # Tests for workspace management
└── fixtures/                      # Test data and fixtures
    ├── datasets/                  # Sample datasets
    └── pipelines/                 # Test pipeline configurations
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation:

- **`data/`** - Tests for dataset management, configuration, metadata, loaders, predictions, and header units handling
- **`pipeline/`** - Tests for pipeline runner, configuration, serialization, binary loading, and manifest management
- **`transforms/`** - Tests for signal processing, NIRS transformations, augmentation, and data splitters
- **`models/`** - Tests for TensorFlow and PyTorch model implementations
- **`controllers/`** - Tests for augmentation controllers (indexer, sample, split, transformer)
- **`utils/`** - Tests for utility functions (binning, balancing, data generation)

### Integration Tests (`tests/integration/`)

Integration tests verify that multiple components work together correctly:

- Augmentation integration workflows
- Dataset augmentation end-to-end
- Multi-component integration scenarios

### Full Integration Tests (`tests/integration_tests/`)

Complete end-to-end tests simulating real-world usage:

- Basic pipeline workflows
- Classification pipelines
- Regression pipelines
- Multi-source analysis
- Group-based splitting
- SHAP analysis
- PCA analysis
- Hyperparameter tuning
- Sample augmentation
- Prediction reuse
- Resampler functionality

### Workspace Tests (`tests/workspace/`)

Tests for workspace management and organization:

- Catalog export functionality
- Library manager operations
- Query and reporting features

### Fixtures (`tests/fixtures/`)

Reusable test data and configurations:

- Sample datasets for testing
- Pipeline configurations
- Test data generators

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Unit Tests Only
```bash
pytest tests/unit/
```

### Run Integration Tests
```bash
pytest tests/integration/
pytest tests/integration_tests/
```

### Run Specific Module Tests
```bash
pytest tests/unit/data/          # Data module tests
pytest tests/unit/pipeline/      # Pipeline module tests
pytest tests/unit/transforms/    # Transforms module tests
pytest tests/unit/models/        # Models module tests
pytest tests/unit/controllers/   # Controllers module tests
pytest tests/unit/utils/         # Utilities module tests
```

### Run with Coverage
```bash
pytest tests/unit/ --cov=nirs4all --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/unit/data/test_metadata.py -v
```

## Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<specific_behavior>`

## Adding New Tests

When adding new tests:

1. **Unit tests**: Place in appropriate subdirectory under `tests/unit/`
2. **Integration tests**: Place in `tests/integration/` or `tests/integration_tests/`
3. **Test fixtures**: Add to `tests/fixtures/` if data is reusable
4. **Documentation**: Update this README if adding new test categories

## Test Data Management

- Use `tests/fixtures/` for shared test data
- Keep test data minimal and focused
- Document test data purpose and structure
- Use `TestDataManager` from `tests/unit/utils/test_data_generator.py` for generating test data

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Commits to main branches
- Release builds

See `.github/workflows/` for CI configuration.

## Test Statistics

As of the last restructuring:
- **Total tests**: 676
- **Unit tests**: ~500
- **Integration tests**: ~150
- **Test files**: ~70

## Key Test Files

- `conftest.py` - Pytest configuration and shared fixtures
- `run_tests.py` - Test runner script
- `run_runner_tests.py` - Pipeline runner test script
- `unit/utils/test_data_generator.py` - Test data generation utilities

## Migration Notes

The test suite was reorganized in October 2025 to improve maintainability and clarity:

### Directory Structure Changes
- Moved `tests/dataset/` → `tests/unit/data/`
- Moved `tests/pipeline/` → `tests/unit/pipeline/`
- Moved `tests/serialization/` → `tests/unit/pipeline/`
- Moved `tests/pipeline_runner/` → `tests/unit/pipeline/`
- Moved `tests/utils/` → `tests/unit/utils/`
- Created `tests/unit/controllers/` for controller tests
- Created `tests/unit/transforms/` for transform tests
- Created `tests/unit/models/` for model tests
- Created `tests/fixtures/` for test data

### File Naming Improvements
- Renamed `test_header_units_step*.py` → descriptive names (e.g., `test_feature_source_header_units.py`)
- Renamed `test_phase*_*.py` → removed phase numbers (e.g., `test_catalog_export.py`)
- Renamed controller tests for consistency (e.g., `test_sample_augmentation_controller.py` → `test_sample_augmentation.py`)
- Renamed `test_resampler.py` → `test_resampler_integration.py` in integration_tests

All import paths were updated to reflect the new structure.

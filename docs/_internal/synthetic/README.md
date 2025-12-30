# Synthetic Data Generator - Documentation Index

**Project**: nirs4all Synthetic NIRS Data Generator Integration
**Status**: Phase 5 Complete - Ready for Phase 6
**Last Updated**: 2024-12-30

---

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Module Migration | ✅ **Complete** | 98% test coverage, 108 tests |
| Phase 2: API Integration | ✅ **Complete** | `nirs4all.generate()` working, 93 new tests |
| Phase 3: Feature Enhancements | ✅ **Complete** | 237 tests, metadata/targets/sources modules |
| Phase 4: Export & Fitting | ✅ **Complete** | 287 tests, exporter/fitter modules |
| Phase 5: Test Migration | ✅ **Complete** | 122 new fixture tests, pytest fixtures in conftest.py |
| Phase 6: Documentation | 🔲 Ready | Next up: Complete documentation |

---

## Quick Links

| Document | Description |
|----------|-------------|
| [Full Specification](./synthetic_generator_specification.md) | Complete technical specification |
| [Implementation Roadmap](./implementation_roadmap.md) | Detailed implementation plan with code patterns |
| [Test Integration Strategy](./test_integration_strategy.md) | Testing fixtures and migration guide |

---

## Executive Summary

This initiative integrates and enhances the synthetic NIRS spectra generator into nirs4all's public API. The project delivers:

1. **Clean API Integration**: `nirs4all.generate()` as a first-class API function
2. **Enhanced Generator**: More realistic spectra with metadata, multi-source, and classification support
3. **Export Capabilities**: Generate files compatible with DatasetConfigs
4. **Test Modernization**: Replace static fixtures with reproducible synthetic data

---

## Scope Summary

### In Scope

| Feature | Priority | Phase | Status |
|---------|----------|-------|--------|
| Move generator to `nirs4all/data/synthetic/` | P0 | 1 | ✅ Complete |
| `nirs4all.generate()` API function | P0 | 2 | ✅ Complete |
| `SyntheticDatasetBuilder` fluent interface | P0 | 2 | ✅ Complete |
| Metadata generation (sample_id, groups, repetitions) | P0 | 3 | ✅ Complete |
| Classification support | P0 | 3 | ✅ Complete |
| Multi-source generation | P0 | 3 | ✅ Complete |
| CSV export for testing | P0 | 4 | ✅ Complete |
| Real data fitting | P1 | 4 | ✅ Complete |
| Pytest fixtures | P0 | 5 | ✅ Complete |
| Documentation | P0 | 6 | 🔲 Pending |

### Out of Scope (Future Work)

- Time-series simulation
- Real-time streaming generation
- GPU acceleration
- External database integration

---

## Phase 5 Deliverables

### Core Pytest Fixtures (tests/conftest.py)

The following fixtures are now available for all tests:

#### Session-Scoped Fixtures (Shared, Read-Only)

| Fixture | Purpose | Samples |
|---------|---------|---------|
| `standard_regression_dataset` | Basic regression | 200 |
| `standard_classification_dataset` | 3-class classification | 150 |
| `standard_binary_dataset` | Binary classification | 100 |
| `multi_target_dataset` | Multi-output regression | 200 |
| `multi_source_dataset` | Multiple data sources | 200 |
| `dataset_with_metadata` | Groups & repetitions | 300 |
| `dataset_with_batch_effects` | Batch simulation | 300 |
| `small_regression_arrays` | Quick test arrays | 50 |
| `sample_wavelengths` | Standard wavelength grid | - |

#### Function-Scoped Fixtures (Fresh, Modifiable)

| Fixture | Purpose | Returns |
|---------|---------|---------|
| `fresh_regression_dataset` | Modifiable regression | SpectroDataset |
| `fresh_classification_dataset` | Modifiable classification | SpectroDataset |
| `regression_arrays` | Raw arrays | (X, y) |
| `classification_arrays` | Classification arrays | (X, y) |
| `synthetic_dataset_folder` | Temp file folder | Path |
| `synthetic_single_file_folder` | Single-file folder | Path |
| `synthetic_csv_file` | Single CSV file | Path |

#### Parametrized Fixtures

| Fixture | Variations | Use Case |
|---------|------------|----------|
| `dataset_all_complexities` | simple, realistic, complex | Complexity testing |
| `classification_n_classes` | 2, 3, 5 classes | Class count testing |

#### Factory Fixtures

| Fixture | Purpose |
|---------|---------|
| `synthetic_builder_factory` | Create custom builders |
| `synthetic_generator_factory` | Create custom generators |
| `csv_variation_generator` | Create CSV format variations |

### CSV Loader Test Fixtures (tests/unit/data/loaders/conftest.py)

Specialized fixtures for testing data loaders:

| Fixture | Format Description |
|---------|-------------------|
| `csv_standard_format` | Semicolon delimiter, standard files |
| `csv_comma_delimiter` | Comma-separated |
| `csv_tab_delimiter` | Tab-separated (.tsv) |
| `csv_no_headers` | No column headers |
| `csv_with_index` | Row index column |
| `csv_single_file` | All data in one file |
| `csv_fragmented` | Multiple small files |
| `csv_low_precision` | 2 decimal places |
| `csv_high_precision` | 10 decimal places |
| `csv_with_missing_values` | NaN values |
| `csv_with_text_headers` | Text column names |
| `csv_european_decimals` | Comma as decimal separator |
| `csv_multi_source` | X split across files |
| `csv_single_file_all_data` | All data + metadata |

### Migration Guide

Old approach:
```python
from tests.fixtures.data_generators import SyntheticNIRSDataGenerator
gen = SyntheticNIRSDataGenerator(random_state=42)
X, y = gen.generate_regression_data(n_samples=100)
```

New approach (recommended):
```python
# Using fixtures in tests
def test_my_feature(standard_regression_dataset):
    X = standard_regression_dataset.x({"partition": "train"})
    y = standard_regression_dataset.y({"partition": "train"})

# For file-based tests
def test_loader(synthetic_dataset_folder):
    from nirs4all.data import DatasetConfigs
    dataset = DatasetConfigs(synthetic_dataset_folder).get_datasets()[0]

# For custom configurations
def test_custom(synthetic_builder_factory):
    dataset = synthetic_builder_factory(n_samples=50).with_classification(n_classes=5).build()
```

---

## Currently Available (Phase 1 & 2)

After Phase 2 completion, the following is available:

### Top-Level API (NEW in Phase 2)

```python
import nirs4all

# Simple generation - returns SpectroDataset
dataset = nirs4all.generate(n_samples=1000, random_state=42)

# Get arrays instead of dataset
X, y = nirs4all.generate(n_samples=500, as_dataset=False, random_state=42)

# Convenience functions
dataset = nirs4all.generate.regression(n_samples=500, target_range=(0, 100))
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)

# Builder for full control
dataset = (
    nirs4all.generate.builder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic", components=["water", "protein"])
    .with_targets(distribution="lognormal", range=(0, 100))
    .with_partitions(train_ratio=0.8)
    .build()
)

# Integration with pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

result = nirs4all.run(
    pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=3), PLSRegression(10)],
    dataset=nirs4all.generate(n_samples=500, random_state=42)
)
```

### Core Generator (Phase 1)

```python
from nirs4all.data.synthetic import SyntheticNIRSGenerator

# Basic generation
generator = SyntheticNIRSGenerator(random_state=42)
X, Y, E = generator.generate(n_samples=1000)

# Create a SpectroDataset directly
dataset = generator.create_dataset(n_train=800, n_test=200)

# Use predefined components
from nirs4all.data.synthetic import ComponentLibrary
library = ComponentLibrary.from_predefined(["water", "protein", "lipid"])
generator = SyntheticNIRSGenerator(component_library=library, random_state=42)

# Configuration classes
from nirs4all.data.synthetic import SyntheticDatasetConfig, FeatureConfig
config = SyntheticDatasetConfig(n_samples=1000, complexity="realistic")
```

### Builder Pattern (Phase 2)

```python
from nirs4all.data.synthetic import SyntheticDatasetBuilder

dataset = (
    SyntheticDatasetBuilder(n_samples=1000, random_state=42)
    .with_features(
        wavelength_range=(1000, 2500),
        complexity="realistic",
        components=["water", "protein", "lipid"]
    )
    .with_targets(
        distribution="lognormal",
        range=(5, 50),
        component="protein"  # Single target
    )
    .with_classification(n_classes=3)  # Or classification
    .with_metadata(
        n_groups=3,
        n_repetitions=(2, 5)
    )
    .with_partitions(train_ratio=0.8)
    .with_batch_effects(n_batches=3)
    .build()
)

# Get raw arrays instead
X, y = builder.build_arrays()

# Get configuration object
config = builder.get_config()

# Create from config
builder = SyntheticDatasetBuilder.from_config(config)
```

---

## API Quick Reference (All Phases Complete)

### Available Now (Phases 1-4)

```python
import nirs4all

# Simple generation - returns SpectroDataset
dataset = nirs4all.generate(n_samples=1000, random_state=42)

# Get arrays instead of dataset
X, y = nirs4all.generate(n_samples=500, as_dataset=False, random_state=42)

# Convenience functions
dataset = nirs4all.generate.regression(n_samples=500, target_range=(0, 100))
dataset = nirs4all.generate.classification(n_samples=300, n_classes=3)

# Multi-source generation
dataset = nirs4all.generate.multi_source(n_samples=400, sources=[
    {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
    {"name": "markers", "type": "aux", "n_features": 15}
])

# Template fitting - generate data that matches real data characteristics
dataset = nirs4all.generate.from_template(X_real, wavelengths=wavelengths)

# Export functions - generate and save to files
path = nirs4all.generate.to_folder("output/", n_samples=1000)
path = nirs4all.generate.to_csv("output/data.csv", n_samples=500)

# Builder for full control
dataset = (
    nirs4all.generate.builder(n_samples=1000, random_state=42)
    .with_features(complexity="realistic", components=["water", "protein"])
    .with_targets(distribution="lognormal", range=(0, 100))
    .with_partitions(train_ratio=0.8)
    .fit_to(X_real, wavelengths=wavelengths)  # Match real data
    .build()
)

# Export from builder
builder = (
    nirs4all.generate.builder(n_samples=500)
    .with_partitions(train_ratio=0.8)
)
builder.export("output/folder")      # Export to folder
builder.export_to_csv("output.csv")  # Export to single CSV

# Use fitter directly for analysis
from nirs4all.data.synthetic import RealDataFitter

fitter = RealDataFitter()
params = fitter.fit(X_real, wavelengths=wavelengths)
params.save("fitted_params.json")

# Generate with fitted parameters
generator = SyntheticNIRSGenerator(**params.to_generator_kwargs())
X_synth, y_synth, _ = generator.generate(500)

# Evaluate similarity
metrics = fitter.evaluate_similarity(X_synth, wavelengths)
print(f"Similarity score: {metrics['overall_score']:.2f}")

# Get tuning recommendations
for rec in fitter.get_tuning_recommendations():
    print(rec)

# Integration with pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

result = nirs4all.run(
    pipeline=[StandardScaler(), PLSRegression(10)],
    dataset=nirs4all.generate(n_samples=1000, random_state=42)
)
```

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Core Module | Week 1-2 | Generator in nirs4all, unit tests |
| Phase 2: API Integration | Week 2-3 | `nirs4all.generate()` working |
| Phase 3: Enhancements | Week 3-4 | Metadata, classification, multi-source |
| Phase 4: Export & Fitting | Week 4-5 | File export, real data fitting |
| Phase 5: Test Migration | Week 5-6 | All tests using synthetic data |
| Phase 6: Documentation | Week 6 | Complete documentation |

**Total Duration**: ~6 weeks

---

## Key Design Decisions

### 1. Module Location

**Decision**: Place in `nirs4all/data/synthetic/`

**Rationale**:
- Consistent with data module structure
- Close to `SpectroDataset` and loaders
- Natural import path: `from nirs4all.data.synthetic import ...`

### 2. API Style

**Decision**: Callable namespace (`nirs4all.generate` is both function and namespace)

**Rationale**:
- Simple default: `nirs4all.generate(n_samples=1000)`
- Discoverable methods: `nirs4all.generate.regression(...)`
- Matches existing pattern of `nirs4all.run()`

### 3. Builder Pattern

**Decision**: Fluent builder for complex configurations

**Rationale**:
- Handles many optional parameters cleanly
- Self-documenting method chains
- Familiar pattern for Python users

### 4. Backward Compatibility

**Decision**: Keep `bench/synthetic/` with deprecation warnings

**Rationale**:
- Existing scripts continue to work
- Clear migration path
- Remove in v1.0

---

## Design Decisions (Confirmed)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Where to place module? | `nirs4all/data/synthetic/` | Consistent with data module structure |
| How to expose API? | `nirs4all.generate()` | ML-standard naming (like sklearn's make_*) |
| Keep old generator? | Yes, with deprecation warnings | Backward compatibility |
| Default complexity level | `"simple"` for unit tests, `"realistic"` for integration tests | Balance speed (unit) vs realism (integration), max 5s overhead |
| Random state handling | Per-call | Explicit reproducibility |
| Visualization integration | **Merge with `nirs4all.visualization`** | Synthetic spectra are spectra - use existing tools |

---

## File Structure After Implementation

```
nirs4all/
├── __init__.py                 # +generate export
├── api/
│   ├── __init__.py             # +generate export
│   └── generate.py             # NEW: Top-level API
├── data/
│   ├── __init__.py             # +synthetic export
│   └── synthetic/              # NEW: Entire module
│       ├── __init__.py
│       ├── generator.py
│       ├── components.py
│       ├── config.py
│       ├── builder.py
│       ├── targets.py
│       ├── metadata.py
│       ├── sources.py
│       ├── exporter.py
│       ├── fitter.py
│       ├── validation.py
│       └── _constants.py
└── visualization/
    └── synthetic.py            # NEW: Merge visualizer here

tests/
├── conftest.py                 # +synthetic fixtures
└── unit/data/synthetic/        # NEW: Unit tests
    ├── __init__.py
    ├── test_generator.py
    ├── test_builder.py
    ├── test_metadata.py
    ├── test_targets.py
    └── test_exporter.py

docs/
├── _internal/synthetic/        # Current location
│   ├── README.md               # This file
│   ├── synthetic_generator_specification.md
│   ├── implementation_roadmap.md
│   └── test_integration_strategy.md
└── source/
    ├── api/synthetic.rst       # NEW: API docs
    └── user_guide/
        └── synthetic_data.md   # NEW: User guide
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Usability | 5 min to first dataset | User testing |
| Test Coverage | >90% for synthetic module | pytest-cov |
| Performance | 10K samples < 1s | Benchmarks |
| Test Migration | >80% tests using fixtures | Code analysis |
| Documentation | All public APIs documented | Doc coverage |

---

## Next Steps

1. **Review** this specification with team
2. **Confirm** design decisions and priorities
3. **Begin Phase 1** implementation
4. **Weekly check-ins** to track progress

---

## Contact

For questions about this specification, refer to:
- Main specification: [synthetic_generator_specification.md](./synthetic_generator_specification.md)
- Implementation details: [implementation_roadmap.md](./implementation_roadmap.md)
- Testing strategy: [test_integration_strategy.md](./test_integration_strategy.md)

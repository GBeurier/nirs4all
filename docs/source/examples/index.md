# Interactive Example Browser

NIRS4ALL ships with **67 comprehensive examples** organized into three categories, each demonstrating real-world spectroscopy workflows. Browse by difficulty, topic, or learning path to find the perfect starting point for your use case.

```{toctree}
:maxdepth: 2
:hidden:

user/getting_started
user/data_handling
user/preprocessing
user/models
user/cross_validation
user/deployment
user/explainability
developer
```

## Quick Start

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 First Time Here?
:link: #beginner-learning-path
:link-type: ref

Start with the beginner learning path: 4 examples covering basics to get you analyzing spectra in 30 minutes.

+++
Recommended for new users
:::

:::{grid-item-card} 🎯 Looking for Something Specific?
:link: #browse-by-topic
:link-type: ref

Jump directly to examples organized by topic: data handling, preprocessing, models, pipelines, etc.

+++
For experienced users
:::

::::

## Running Examples

All examples are executable Python scripts located in `examples/` with sample datasets included:

```bash
# Run from project root
cd examples

# Run all user examples
./run.sh -c user

# Run specific example
python user/01_getting_started/U01_hello_world.py

# Run with visualizations
python user/01_getting_started/U02_basic_regression.py --plots --show

# Quick mode (skip deep learning examples)
./run.sh -q
```

---

(beginner-learning-path)=
## Beginner Learning Path

Start here if you're new to nirs4all. Follow this 4-step path to understand core concepts:

::::{grid} 2
:gutter: 2

:::{grid-item-card} 1️⃣ Hello World
★☆☆☆☆ | ~1 minute

[examples/user/01_getting_started/U01_hello_world.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U01_hello_world.py)

Your first pipeline in 20 lines. Train a PLS model on NIRS data and get predictions.

**Topics**: `nirs4all.run()` basics, pipeline structure, reading results
:::

:::{grid-item-card} 2️⃣ Basic Regression
★★☆☆☆ | ~3 minutes

[examples/user/01_getting_started/U02_basic_regression.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U02_basic_regression.py)

Add preprocessing, feature augmentation, and visualization to your pipeline.

**Topics**: SNV, derivatives, `PredictionAnalyzer`, comparing models
:::

:::{grid-item-card} 3️⃣ Flexible Data Inputs
★☆☆☆☆ | ~2 minutes

[examples/user/02_data_handling/U01_flexible_inputs.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U01_flexible_inputs.py)

Learn all the ways to load data: numpy arrays, dictionaries, SpectroDataset objects.

**Topics**: Data formats, partition specification, dataset configuration
:::

:::{grid-item-card} 4️⃣ Preprocessing Basics
★★☆☆☆ | ~3 minutes

[examples/user/03_preprocessing/U01_preprocessing_basics.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U01_preprocessing_basics.py)

Explore NIRS-specific preprocessing: scatter correction, derivatives, smoothing.

**Topics**: SNV, MSC, Detrend, Savitzky-Golay, wavelets
:::

::::

**Next Steps**: After completing this path, explore {ref}`intermediate-learning-path` or browse examples by topic below.

---

(intermediate-learning-path)=
## Intermediate Learning Path

Ready to build production-ready models? Follow this path:

::::{grid} 2
:gutter: 2

:::{grid-item-card} 5️⃣ Multi-Model Comparison
★★☆☆☆ | ~3 minutes

[examples/user/04_models/U01_multi_model.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U01_multi_model.py)

Compare multiple models in one run using the `_or_` generator syntax.

**Topics**: Model selection, generator syntax, performance comparison
:::

:::{grid-item-card} 6️⃣ Cross-Validation Strategies
★★☆☆☆ | ~4 minutes

[examples/user/05_cross_validation/U01_cv_strategies.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U01_cv_strategies.py)

Proper validation with K-Fold, stratified splits, and group-aware CV.

**Topics**: KFold, ShuffleSplit, StratifiedKFold, custom splits
:::

:::{grid-item-card} 7️⃣ Save, Load, and Predict
★★☆☆☆ | ~3 minutes

[examples/user/06_deployment/U01_save_load_predict.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U01_save_load_predict.py)

Export trained models and make predictions on new data.

**Topics**: Model persistence, prediction API, workspace management
:::

:::{grid-item-card} 8️⃣ SHAP Basics
★★☆☆☆ | ~4 minutes

[examples/user/07_explainability/U01_shap_basics.py](https://github.com/nirs4all/nirs4all/blob/main/examples/user/07_explainability/U01_shap_basics.py)

Understand model predictions with SHAP waterfall and spectral plots.

**Topics**: SHAP integration, feature importance, visualization
:::

::::

**Next Steps**: Explore {ref}`advanced-learning-path` or dive into specific topics in the sections below.

---

(advanced-learning-path)=
## Advanced Learning Path

Master advanced features for research and production systems:

::::{grid} 2
:gutter: 2

:::{grid-item-card} 9️⃣ Branching Basics
★★★☆☆ | ~5 minutes

[examples/developer/01_advanced_pipelines/D01_branching_basics.py](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D01_branching_basics.py)

Create parallel preprocessing branches and compare results.

**Topics**: Duplication branches, separation branches, branch comparison
:::

:::{grid-item-card} 🔟 Generator Syntax
★★★☆☆ | ~5 minutes

[examples/developer/02_generators/D01_generator_syntax.py](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py)

Master the generator syntax for automated pipeline exploration.

**Topics**: `_or_`, `_range_`, `_grid_`, `_zip_`, `_chain_`
:::

:::{grid-item-card} 1️⃣1️⃣ Session Workflow
★★★★☆ | ~6 minutes

[examples/developer/06_internals/D01_session_workflow.py](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D01_session_workflow.py)

Use sessions for stateful multi-step workflows with training, prediction, and retraining.

**Topics**: Session API, persistence, multi-run workflows
:::

:::{grid-item-card} 1️⃣2️⃣ Cache Performance
★★★★☆ | ~6 minutes

[examples/developer/06_internals/D03_cache_performance.py](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D03_cache_performance.py)

Optimize large pipeline runs with step caching and CoW snapshots.

**Topics**: CacheConfig, memory optimization, performance tuning
:::

::::

---

(browse-by-topic)=
## Browse by Topic

### 📚 Getting Started (4 examples)

Perfect first examples for understanding nirs4all basics.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: Hello World](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U01_hello_world.py) | ★☆☆☆☆ | Your first pipeline in 20 lines |
| [U02: Basic Regression](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U02_basic_regression.py) | ★★☆☆☆ | Preprocessing, feature augmentation, visualization |
| [U03: Basic Classification](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U03_basic_classification.py) | ★★☆☆☆ | Classification pipelines and confusion matrices |
| [U04: Visualization](https://github.com/nirs4all/nirs4all/blob/main/examples/user/01_getting_started/U04_visualization.py) | ★★☆☆☆ | Heatmaps, candlestick charts, prediction analysis |

**Key APIs**: `nirs4all.run()`, `PredictionAnalyzer`, `RunResult`

---

### 💾 Data Handling (6 examples)

Loading data, multi-source datasets, synthetic data generation.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: Flexible Inputs](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U01_flexible_inputs.py) | ★☆☆☆☆ | All data input formats: arrays, dicts, SpectroDataset |
| [U02: Multi-Datasets](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U02_multi_datasets.py) | ★★☆☆☆ | Analyze multiple datasets in one run |
| [U03: Multi-Source](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U03_multi_source.py) | ★★★☆☆ | Combine NIR, markers, and other data sources |
| [U04: Wavelength Handling](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U04_wavelength_handling.py) | ★★☆☆☆ | Resample, downsample, focus on spectral regions |
| [U05: Synthetic Data](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U05_synthetic_data.py) | ★★☆☆☆ | Generate synthetic datasets with `nirs4all.generate()` |
| [U06: Advanced Synthetic Data](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U06_synthetic_advanced.py) | ★★★☆☆ | SyntheticDatasetBuilder with metadata and batch effects |

**Key APIs**: `SpectroDataset`, `DatasetConfigs`, `nirs4all.generate()`, `SyntheticDatasetBuilder`

---

### 🔬 Preprocessing (4 examples)

NIRS-specific preprocessing operators and signal conversions.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: Preprocessing Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U01_preprocessing_basics.py) | ★★☆☆☆ | SNV, MSC, Detrend, derivatives, smoothing, wavelets |
| [U02: Feature Augmentation](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U02_feature_augmentation.py) | ★★★☆☆ | Explore preprocessing combinations with `feature_augmentation` |
| [U03: Sample Augmentation](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U03_sample_augmentation.py) | ★★★☆☆ | Data augmentation for training set expansion |
| [U04: Signal Conversion](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U04_signal_conversion.py) | ★★★☆☆ | Convert between absorbance, reflectance, transmittance |

**Key APIs**: `SNV`, `MSC`, `Detrend`, `SavitzkyGolay`, `Derivative`, `ToAbsorbance`, `FromAbsorbance`

---

### 🤖 Models (4 examples)

Model training, hyperparameter tuning, ensembles, PLS variants.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: Multi-Model](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U01_multi_model.py) | ★★☆☆☆ | Compare multiple models with `_or_` generator |
| [U02: Hyperparameter Tuning](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U02_hyperparameter_tuning.py) | ★★★☆☆ | Automated tuning with grid, random, Bayesian search |
| [U03: Stacking Ensembles](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U03_stacking_ensembles.py) | ★★★☆☆ | StackingRegressor, VotingRegressor, meta-learners |
| [U04: PLS Variants](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U04_pls_variants.py) | ★★★★☆ | PLSRegression, IKPLS, OPLS, SparsePLS, IntervalPLS |

**Key APIs**: `finetune_params`, `StackingRegressor`, PLS variants, hyperparameter search

---

### ✅ Cross-Validation (6 examples)

Validation strategies, group splitting, outlier filtering, aggregation.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: CV Strategies](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U01_cv_strategies.py) | ★★☆☆☆ | KFold, ShuffleSplit, StratifiedKFold, custom splits |
| [U02: Group Splitting](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U02_group_splitting.py) | ★★★☆☆ | GroupKFold, repetition-aware splitting |
| [U03: Sample Exclusion](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U03_sample_filtering.py) | ★★★☆☆ | Outlier detection with `exclude` keyword |
| [U04: Repetition Aggregation](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U04_aggregation.py) | ★★★☆☆ | Aggregate predictions from repeated measurements |
| [U05: Tagging Analysis](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U05_tagging_analysis.py) | ★★★☆☆ | Mark samples with `tag` for downstream analysis |
| [U06: Exclusion Strategies](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U06_exclusion_strategies.py) | ★★★★☆ | Advanced exclusion with "any" vs "all" modes |

**Key APIs**: `GroupKFold`, `YOutlierFilter`, `XOutlierFilter`, `tag`, `exclude`, `repetition`

---

### 🚀 Deployment (4 examples)

Model persistence, bundle export, sklearn integration, workspace management.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: Save, Load, Predict](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U01_save_load_predict.py) | ★★☆☆☆ | Basic prediction workflow with workspace |
| [U02: Export Bundle](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U02_export_bundle.py) | ★★☆☆☆ | Export to .n4a and .n4a.py portable bundles |
| [U03: Workspace Management](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U03_workspace_management.py) | ★★☆☆☆ | Session context, DuckDB storage, library management |
| [U04: sklearn Integration](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U04_sklearn_integration.py) | ★★☆☆☆ | Use NIRSPipeline as sklearn estimator for GridSearchCV |

**Key APIs**: `nirs4all.predict()`, `export()`, `NIRSPipeline`, `nirs4all.session()`

---

### 🔍 Explainability (3 examples)

SHAP-based feature importance and variable selection.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [U01: SHAP Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/user/07_explainability/U01_shap_basics.py) | ★★☆☆☆ | SHAP waterfall, spectral, and beeswarm plots |
| [U02: SHAP with sklearn](https://github.com/nirs4all/nirs4all/blob/main/examples/user/07_explainability/U02_shap_sklearn.py) | ★★★☆☆ | Custom SHAP explainers with NIRSPipeline wrapper |
| [U03: Feature Selection](https://github.com/nirs4all/nirs4all/blob/main/examples/user/07_explainability/U03_feature_selection.py) | ★★★☆☆ | CARS, MC-UVE for variable selection |

**Key APIs**: `nirs4all.explain()`, `NIRSPipeline`, SHAP integration, feature selection

---

## Developer Examples (30 examples)

Advanced examples for extending nirs4all and building complex workflows.

### 🌿 Advanced Pipelines (7 examples)

Branching, merging, separation, stacking, value mapping.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: Branching Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D01_branching_basics.py) | ★★★☆☆ | Duplication and separation branches |
| [D02: Branching Advanced](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D02_branching_advanced.py) | ★★★★☆ | BranchAnalyzer for statistical comparison |
| [D03: Merge Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D03_merge_basics.py) | ★★★★☆ | Feature merge, prediction merge, concat merge |
| [D04: Merge Sources](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py) | ★★★★☆ | Multi-source branching and merging |
| [D05: Meta Stacking](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D05_meta_stacking.py) | ★★★★★ | MetaModel, StackingConfig, multi-level stacking |
| [D06: Separation Branches](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D06_separation_branches.py) | ★★★★☆ | by_tag, by_metadata, by_filter, by_source |
| [D07: Value Mapping](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D07_value_mapping.py) | ★★★★☆ | Group metadata values for custom branching |

**Key APIs**: `branch`, `merge`, `by_tag`, `by_metadata`, `by_source`, `MetaModel`, `StackingConfig`

---

### 🎲 Generators & Synthetic Data (9 examples)

Generator syntax, synthetic data generation, application domains, instruments.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: Generator Syntax](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py) | ★★★☆☆ | `_or_`, `_range_`, `pick`, `arrange` |
| [D02: Generator Advanced](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D02_generator_advanced.py) | ★★★★☆ | `_mutex_`, `_requires_`, `_grid_`, `_zip_`, `_chain_`, `_sample_` |
| [D03: Generator Iterators](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D03_generator_iterators.py) | ★★★★☆ | Lazy iteration, batch processing, variant inspection |
| [D04: Nested Generators](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D04_nested_generators.py) | ★★★★★ | Hierarchical parameter spaces |
| [D05: Custom Components](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D05_synthetic_custom_components.py) | ★★★★☆ | Build custom spectral component libraries |
| [D06: Synthetic Testing](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D06_synthetic_testing.py) | ★★★★☆ | Use synthetic data for reproducible testing |
| [D07: Wavenumber Procedural](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D07_synthetic_wavenumber_procedural.py) | ★★★★☆ | Overtone calculation, combination bands, H-bonding |
| [D08: Application Domains](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D08_synthetic_application_domains.py) | ★★★★☆ | Agriculture, food, pharma, environmental domains |
| [D09: Instruments](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D09_synthetic_instruments.py) | ★★★★☆ | Simulate FOSS, Bruker, SCiO, multi-sensor instruments |

**Key APIs**: `_or_`, `_range_`, `_grid_`, `_zip_`, generator constraints, `SyntheticNIRSGenerator`, application domains

---

### 🧠 Deep Learning (4 examples)

PyTorch, JAX, TensorFlow integration with the `@framework` decorator.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: PyTorch Models](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/03_deep_learning/D01_pytorch_models.py) | ★★★★☆ | Built-in nicon, decon, transformer, custom models |
| [D02: JAX Models](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/03_deep_learning/D02_jax_models.py) | ★★★★★ | JAX/Flax integration with JIT compilation |
| [D03: TensorFlow Models](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/03_deep_learning/D03_tensorflow_models.py) | ★★★★☆ | TensorFlow/Keras nicon and decon architectures |
| [D04: Framework Comparison](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/03_deep_learning/D04_framework_comparison.py) | ★★★★☆ | Same model across PyTorch, JAX, TensorFlow |

**Key APIs**: `@framework` decorator, `nicon`, `decon`, `transformer`, custom DL models

---

### 🔄 Transfer Learning (3 examples)

Calibration transfer, retraining modes, PCA geometry analysis.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: Transfer Analysis](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/04_transfer_learning/D01_transfer_analysis.py) | ★★★★☆ | TransferPreprocessingSelector for transfer problems |
| [D02: Retrain Modes](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/04_transfer_learning/D02_retrain_modes.py) | ★★★★☆ | full, transfer, finetune retraining modes |
| [D03: PCA Geometry](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/04_transfer_learning/D03_pca_geometry.py) | ★★★☆☆ | PreprocPCAEvaluator for geometry preservation |

**Key APIs**: `nirs4all.retrain()`, `TransferPreprocessingSelector`, retrain modes

---

### ⚙️ Advanced Features (3 examples)

Metadata branching, concat transform, repetition transform.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: Metadata Branching](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/05_advanced_features/D01_metadata_branching.py) | ★★★★☆ | Partition by metadata with value mapping |
| [D02: Concat Transform](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/05_advanced_features/D02_concat_transform.py) | ★★★☆☆ | Concatenate multiple transformer outputs |
| [D03: Repetition Transform](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/05_advanced_features/D03_repetition_transform.py) | ★★★★☆ | Convert repetitions to sources or preprocessing variants |

**Key APIs**: `by_metadata`, `concat_transform`, `rep_to_sources`, `rep_to_pp`

---

### 🔧 Internals (3 examples)

Session API, custom controllers, cache optimization.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [D01: Session Workflow](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D01_session_workflow.py) | ★★★★☆ | Stateful workflows with save/load/persist |
| [D02: Custom Controllers](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D02_custom_controllers.py) | ★★★★★ | Extend nirs4all with custom operator controllers |
| [D03: Cache Performance](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D03_cache_performance.py) | ★★★★☆ | Step caching, CoW snapshots, memory optimization |

**Key APIs**: `nirs4all.session()`, `@register_controller`, `CacheConfig`, controller architecture

---

## Reference Examples (7 examples)

Comprehensive reference examples for advanced features.

| Example | Difficulty | Description |
|---------|------------|-------------|
| [R01: Pipeline Syntax](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R01_pipeline_syntax.py) | ★★☆☆☆ | Complete pipeline syntax reference |
| [R02: Generator Reference](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R02_generator_reference.py) | ★★☆☆☆ | All generator keywords and patterns |
| [R03: All Keywords](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R03_all_keywords.py) | ★★★★★ | Integration test for all pipeline keywords |
| [R04: Legacy API](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R04_legacy_api.py) | ★★☆☆☆ | Deprecated API patterns (for migration) |
| [R05: Environmental Effects](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R05_synthetic_environmental.py) | ★★☆☆☆ | Matrix effects simulation in synthetic data |
| [R06: Synthetic Validation](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R06_synthetic_validation.py) | ★★☆☆☆ | Quality assessment for synthetic spectra |
| [R07: Synthetic Fitter](https://github.com/nirs4all/nirs4all/blob/main/examples/reference/R07_synthetic_fitter.py) | ★★★★★ | Fit synthetic data to match real spectra |

---

## Find Examples by Feature

### By Feature Type

- **Caching**: [D03: Cache Performance](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D03_cache_performance.py)
- **Branching**: [D01: Branching Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D01_branching_basics.py), [D06: Separation Branches](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D06_separation_branches.py)
- **Merging**: [D03: Merge Basics](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D03_merge_basics.py), [D04: Merge Sources](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py)
- **Stacking**: [U03: Stacking Ensembles](https://github.com/nirs4all/nirs4all/blob/main/examples/user/04_models/U03_stacking_ensembles.py), [D05: Meta Stacking](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D05_meta_stacking.py)
- **Generators**: [D01: Generator Syntax](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py), [D02: Generator Advanced](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D02_generator_advanced.py)
- **Session API**: [U03: Workspace Management](https://github.com/nirs4all/nirs4all/blob/main/examples/user/06_deployment/U03_workspace_management.py), [D01: Session Workflow](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D01_session_workflow.py)
- **Synthetic Data**: [U05: Synthetic Data](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U05_synthetic_data.py), [U06: Advanced Synthetic](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U06_synthetic_advanced.py), [D05-D09: Synthetic Generators](https://github.com/nirs4all/nirs4all/tree/main/examples/developer/02_generators)
- **Multi-Source**: [U03: Multi-Source](https://github.com/nirs4all/nirs4all/blob/main/examples/user/02_data_handling/U03_multi_source.py), [D04: Merge Sources](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/01_advanced_pipelines/D04_merge_sources.py)
- **Aggregation**: [U04: Repetition Aggregation](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U04_aggregation.py), [D03: Repetition Transform](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/05_advanced_features/D03_repetition_transform.py)
- **Signal Types**: [U04: Signal Conversion](https://github.com/nirs4all/nirs4all/blob/main/examples/user/03_preprocessing/U04_signal_conversion.py)
- **Outlier Filtering**: [U03: Sample Exclusion](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U03_sample_filtering.py), [U06: Exclusion Strategies](https://github.com/nirs4all/nirs4all/blob/main/examples/user/05_cross_validation/U06_exclusion_strategies.py)
- **Deep Learning**: [D01-D04: Deep Learning Examples](https://github.com/nirs4all/nirs4all/tree/main/examples/developer/03_deep_learning)
- **Custom Controllers**: [D02: Custom Controllers](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/06_internals/D02_custom_controllers.py)

---

## Next Steps

After exploring examples:

- **User Guide**: {doc}`/user_guide/index` - Complete documentation with detailed explanations
- **API Reference**: {doc}`/reference/api_reference` - Full API documentation
- **Onboarding**: {doc}`/onboarding/index` - Deep dive into architecture and design patterns
- **Getting Started**: {doc}`/getting_started/quickstart` - Installation and first steps

**Questions?** Check the {doc}`/user_guide/troubleshooting/faq` or raise an issue on [GitHub](https://github.com/nirs4all/nirs4all/issues).

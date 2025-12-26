# nirs4all Examples

Welcome to the nirs4all examples! This collection provides a progressive learning path from basic usage to advanced features.

## 📁 Structure

Examples are organized into two main learning paths:

```
examples/
├── user/                    # 👤 User Path (U01-U27)
│   ├── 01_getting_started/  # Your first pipelines
│   ├── 02_data_handling/    # Loading and managing data
│   ├── 03_preprocessing/    # NIRS preprocessing techniques
│   ├── 04_models/           # Model training and comparison
│   ├── 05_cross_validation/ # Validation strategies
│   ├── 06_deployment/       # Saving, loading, and deploying
│   └── 07_explainability/   # SHAP and feature importance
│
├── developer/               # 🔧 Developer Path (D01-D22)
│   ├── 01_advanced_pipelines/  # Branching and merging
│   ├── 02_generators/          # Pipeline generation syntax
│   ├── 03_deep_learning/       # TensorFlow, PyTorch, JAX
│   ├── 04_transfer_learning/   # Domain adaptation
│   ├── 05_advanced_features/   # Metadata, outliers, etc.
│   └── 06_internals/           # Custom controllers, sessions
│
├── reference/               # 📚 Reference Examples (R01-R04)
│   └── Complete syntax documentation
│
├── benchmarks/              # 📊 Benchmarks
│   └── Performance comparisons
│
└── sample_data/             # 📂 Sample datasets
```

---

## 👤 User Path

The User Path provides a complete introduction to nirs4all, from your first pipeline to production deployment.

### 01_getting_started/ - Getting Started

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_hello_world.py | Your first pipeline in 20 lines | ★☆☆☆☆ |
| U02_basic_regression.py | Regression with NIRS preprocessing | ★★☆☆☆ |
| U03_basic_classification.py | Classification with RF and XGBoost | ★★☆☆☆ |
| U04_visualization.py | Tour of visualization tools | ★★☆☆☆ |

### 02_data_handling/ - Data Handling

| Example | Description | Complexity |
|---------|-------------|------------|
| U05_flexible_inputs.py | Different input formats (numpy, path, dict) | ★☆☆☆☆ |
| U06_multi_datasets.py | Analyze multiple datasets | ★★☆☆☆ |
| U07_multi_source.py | Multi-source data (NIR + other sensors) | ★★★☆☆ |
| U08_wavelength_handling.py | Wavelength interpolation and units | ★★☆☆☆ |

### 03_preprocessing/ - Preprocessing

| Example | Description | Complexity |
|---------|-------------|------------|
| U09_preprocessing_basics.py | SNV, MSC, derivatives, smoothing | ★★☆☆☆ |
| U10_feature_augmentation.py | Feature augmentation modes | ★★★☆☆ |
| U11_sample_augmentation.py | Sample augmentation (noise, drift) | ★★★☆☆ |
| U12_signal_conversion.py | Absorbance, reflectance, Kubelka-Munk | ★★☆☆☆ |

### 04_models/ - Models

| Example | Description | Complexity |
|---------|-------------|------------|
| U13_multi_model.py | Compare PLS, RF, Ridge, XGBoost | ★★☆☆☆ |
| U14_hyperparameter_tuning.py | Optuna optimization | ★★★☆☆ |
| U15_stacking_ensembles.py | Stacking and Voting | ★★★☆☆ |
| U16_pls_variants.py | PLSR, OPLS, SparsePLS, iPLS | ★★★☆☆ |

### 05_cross_validation/ - Cross-Validation

| Example | Description | Complexity |
|---------|-------------|------------|
| U17_cv_strategies.py | KFold, ShuffleSplit, RepeatedKFold | ★★☆☆☆ |
| U18_group_splitting.py | GroupKFold, force_group | ★★★☆☆ |
| U19_sample_filtering.py | Outlier filtering (IQR, Z-score) | ★★★☆☆ |
| U20_aggregation.py | Aggregation of repetitions | ★★☆☆☆ |

### 06_deployment/ - Deployment

| Example | Description | Complexity |
|---------|-------------|------------|
| U21_save_load_predict.py | Save, load, and predict | ★★☆☆☆ |
| U22_export_bundle.py | Export .n4a bundles | ★★☆☆☆ |
| U23_workspace_management.py | Workspace and artifacts | ★★☆☆☆ |
| U24_sklearn_integration.py | NIRSPipeline sklearn wrapper | ★★★☆☆ |

### 07_explainability/ - Explainability

| Example | Description | Complexity |
|---------|-------------|------------|
| U25_shap_basics.py | SHAP for NIRS | ★★★☆☆ |
| U26_shap_sklearn.py | SHAP with sklearn wrapper | ★★★☆☆ |
| U27_feature_selection.py | CARS, MC-UVE, wavelength selection | ★★★☆☆ |

---

## 🔧 Developer Path

The Developer Path covers advanced features for power users and contributors.

### 01_advanced_pipelines/ - Advanced Pipelines

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_branching_basics.py | Introduction to branching | ★★★☆☆ |
| D02_branching_advanced.py | BranchAnalyzer, statistics | ★★★★☆ |
| D03_merge_strategies.py | Merge features/predictions | ★★★★☆ |
| D04_source_branching.py | Per-source preprocessing | ★★★★☆ |
| D05_meta_stacking.py | Multi-level stacking | ★★★★★ |

### 02_generators/ - Generators

| Example | Description | Complexity |
|---------|-------------|------------|
| D06_generator_basics.py | _or_, pick, count, _range_ | ★★★☆☆ |
| D07_generator_advanced.py | _grid_, _zip_, _chain_, _sample_ | ★★★★☆ |
| D08_generator_nested.py | Nested generators, _cartesian_ | ★★★★★ |
| D09_constraints_presets.py | Constraints and presets | ★★★★★ |

### 03_deep_learning/ - Deep Learning

| Example | Description | Complexity |
|---------|-------------|------------|
| D10_nicon_tensorflow.py | TensorFlow nicon models | ★★★★☆ |
| D11_pytorch_models.py | Custom PyTorch models | ★★★★☆ |
| D12_jax_models.py | JAX/Flax models | ★★★★☆ |
| D13_framework_comparison.py | TF vs PyTorch vs JAX | ★★★★★ |

### 04_transfer_learning/ - Transfer Learning

| Example | Description | Complexity |
|---------|-------------|------------|
| D14_retrain_modes.py | Full, transfer, finetune modes | ★★★★☆ |
| D15_transfer_analysis.py | TransferPreprocessingSelector | ★★★★☆ |
| D16_domain_adaptation.py | PCA analysis for transfer | ★★★★★ |

### 05_advanced_features/ - Advanced Features

| Example | Description | Complexity |
|---------|-------------|------------|
| D17_outlier_partitioning.py | Sample partitioning | ★★★★☆ |
| D18_metadata_branching.py | Metadata-based branching | ★★★★☆ |
| D19_repetition_transform.py | Repetition to sources/preprocessing | ★★★★☆ |
| D20_concat_transform.py | Concatenated transformers | ★★★☆☆ |

### 06_internals/ - Internals

| Example | Description | Complexity |
|---------|-------------|------------|
| D21_session_workflow.py | Session-based workflows | ★★★☆☆ |
| D22_custom_controllers.py | Custom controller development | ★★★★★ |

---

## 📚 Reference

| Example | Description |
|---------|-------------|
| R01_pipeline_syntax.py | Complete pipeline syntax reference |
| R02_generator_reference.py | Generator syntax documentation |
| R03_all_keywords.py | Test all pipeline keywords |
| R04_legacy_api.py | Legacy PipelineRunner API |

---

## 🚀 Running Examples

### Run all examples
```bash
./run.sh                    # Linux/Mac
./run.ps1                   # Windows
```

### Run by category
```bash
./run.sh -c user            # User path only
./run.sh -c developer       # Developer path only
./run.sh -c reference       # Reference examples
./run.sh -c legacy          # Old Q* examples (during migration)
```

### Run specific example
```bash
./run.sh -i 1               # By index
./run.sh -n "U01*.py"       # By name pattern
```

### Quick mode (skip deep learning)
```bash
./run.sh -q                 # Skip TensorFlow/PyTorch/JAX examples
```

### Options
```bash
./run.sh -l                 # Enable logging to log.txt
./run.sh -p                 # Generate plots
./run.sh -s                 # Show plots interactively
./run.sh -p -s              # Generate and show plots
```

---

## 📋 Migration Status

This directory is being reorganized. During the transition period:

- **New examples** are in `user/`, `developer/`, `reference/`
- **Legacy examples** (Q*.py, X*.py) remain at the root level
- Use `-c legacy` to run only legacy examples

See [EXAMPLES_REORGANIZATION.md](../docs/EXAMPLES_REORGANIZATION.md) for the full migration plan.

---

## 💡 Tips

1. **Start with U01**: If you're new to nirs4all, start with `U01_hello_world.py`
2. **Follow the path**: Examples are numbered for sequential learning
3. **Check prerequisites**: Some examples require specific packages (TensorFlow, PyTorch, etc.)
4. **Use `--plots`**: Add `--plots --show` to visualize results
5. **Read docstrings**: Each example has detailed documentation at the top

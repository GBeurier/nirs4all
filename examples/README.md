# nirs4all Examples

Welcome to the nirs4all examples! This collection provides a progressive learning path from basic usage to advanced features.

## 📁 Structure

Examples are organized into two main learning paths, with **per-folder numbering** (each folder starts at 01):

```
examples/
├── user/                    # 👤 User Path
│   ├── 01_getting_started/  # U01-U04: Your first pipelines
│   ├── 02_data_handling/    # U01-U06: Loading and managing data
│   ├── 03_preprocessing/    # U01-U04: NIRS preprocessing techniques
│   ├── 04_models/           # U01-U04: Model training and comparison
│   ├── 05_cross_validation/ # U01-U04: Validation strategies
│   ├── 06_deployment/       # U01-U04: Saving, loading, and deploying
│   └── 07_explainability/   # U01-U03: SHAP and feature importance
│
├── developer/               # 🔧 Developer Path
│   ├── 01_advanced_pipelines/  # D01-D05: Branching and merging
│   ├── 02_generators/          # D01-D04: Pipeline generation, D05-D09: Synthetic data
│   ├── 03_deep_learning/       # D01-D04: TensorFlow, PyTorch, JAX
│   ├── 04_transfer_learning/   # D01-D03: Domain adaptation
│   ├── 05_advanced_features/   # D01-D03: Metadata, outliers, etc.
│   └── 06_internals/           # D01-D02: Custom controllers, sessions
│
├── reference/               # 📚 Reference Examples (R01-R07)
│   ├── R01-R03                  # Pipeline syntax documentation
│   └── R05-R07                  # Advanced synthetic data (Phase 3-4)
│
└── sample_data/             # 📂 Sample datasets
```

**Output Directory**: Examples save generated files to `workspace/examples_output/`

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
| U01_flexible_inputs.py | Different input formats (numpy, path, dict) | ★☆☆☆☆ |
| U02_multi_datasets.py | Analyze multiple datasets | ★★☆☆☆ |
| U03_multi_source.py | Multi-source data (NIR + other sensors) | ★★★☆☆ |
| U04_wavelength_handling.py | Wavelength interpolation and units | ★★☆☆☆ |
| U05_synthetic_data.py | Generate synthetic NIRS data | ★★☆☆☆ |
| U06_synthetic_advanced.py | Builder API for synthetic data | ★★★☆☆ |

### 03_preprocessing/ - Preprocessing

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_preprocessing_basics.py | SNV, MSC, derivatives, smoothing | ★★☆☆☆ |
| U02_feature_augmentation.py | Feature augmentation modes | ★★★☆☆ |
| U03_sample_augmentation.py | Sample augmentation (noise, drift) | ★★★☆☆ |
| U04_signal_conversion.py | Absorbance, reflectance, Kubelka-Munk | ★★☆☆☆ |

### 04_models/ - Models

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_multi_model.py | Compare PLS, RF, Ridge, XGBoost | ★★☆☆☆ |
| U02_hyperparameter_tuning.py | Optuna optimization | ★★★☆☆ |
| U03_stacking_ensembles.py | Stacking and Voting | ★★★☆☆ |
| U04_pls_variants.py | PLSR, OPLS, SparsePLS, iPLS | ★★★☆☆ |

### 05_cross_validation/ - Cross-Validation

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_cv_strategies.py | KFold, ShuffleSplit, RepeatedKFold | ★★☆☆☆ |
| U02_group_splitting.py | GroupKFold, repetition | ★★★☆☆ |
| U03_sample_filtering.py | Outlier filtering (IQR, Z-score) | ★★★☆☆ |
| U04_aggregation.py | Aggregation of repetitions | ★★☆☆☆ |

### 06_deployment/ - Deployment

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_save_load_predict.py | Save, load, and predict | ★★☆☆☆ |
| U02_export_bundle.py | Export .n4a bundles | ★★☆☆☆ |
| U03_workspace_management.py | Workspace and artifacts | ★★☆☆☆ |
| U04_sklearn_integration.py | NIRSPipeline sklearn wrapper | ★★★☆☆ |

### 07_explainability/ - Explainability

| Example | Description | Complexity |
|---------|-------------|------------|
| U01_shap_basics.py | SHAP for NIRS | ★★★☆☆ |
| U02_shap_sklearn.py | SHAP with sklearn wrapper | ★★★☆☆ |
| U03_feature_selection.py | CARS, MC-UVE, wavelength selection | ★★★☆☆ |

---

## 🔧 Developer Path

The Developer Path covers advanced features for power users and contributors.

### 01_advanced_pipelines/ - Advanced Pipelines

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_branching_basics.py | Introduction to branching | ★★★☆☆ |
| D02_branching_advanced.py | BranchAnalyzer, statistics | ★★★★☆ |
| D03_merge_basics.py | Merge features/predictions | ★★★★☆ |
| D04_merge_sources.py | Per-source preprocessing | ★★★★☆ |
| D05_meta_stacking.py | Multi-level stacking | ★★★★★ |

### 02_generators/ - Generators & Synthetic Data

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_generator_syntax.py | _or_, pick, count, _range_ | ★★★☆☆ |
| D02_generator_advanced.py | _grid_, _zip_, _chain_, _sample_ | ★★★★☆ |
| D03_generator_iterators.py | Generator iterators | ★★★★★ |
| D04_nested_generators.py | Nested generators, _cartesian_ | ★★★★★ |
| D05_synthetic_custom_components.py | Custom NIR bands & components | ★★★★☆ |
| D06_synthetic_testing.py | Testing with synthetic data | ★★★★☆ |
| D07_synthetic_wavenumber_procedural.py | Wavenumber utilities & procedural generation | ★★★★☆ |
| D08_synthetic_application_domains.py | Application domains & domain-aware generation | ★★★★☆ |
| D09_synthetic_instruments.py | Simulate instrument-specific characteristics | ★★★★☆ |

> **Note**: Advanced synthetic data features (environmental effects, validation, fitting) are in Reference examples R05-R07.

### 03_deep_learning/ - Deep Learning

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_pytorch_models.py | Custom PyTorch models | ★★★★☆ |
| D02_jax_models.py | JAX/Flax models | ★★★★☆ |
| D03_tensorflow_models.py | TensorFlow nicon models | ★★★★☆ |
| D04_framework_comparison.py | TF vs PyTorch vs JAX | ★★★★★ |

### 04_transfer_learning/ - Transfer Learning

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_transfer_analysis.py | TransferPreprocessingSelector | ★★★★☆ |
| D02_retrain_modes.py | Full, transfer, finetune modes | ★★★★☆ |
| D03_pca_geometry.py | PCA analysis for transfer | ★★★★★ |

### 05_advanced_features/ - Advanced Features

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_metadata_branching.py | Metadata-based branching | ★★★★☆ |
| D02_concat_transform.py | Concatenated transformers | ★★★☆☆ |
| D03_repetition_transform.py | Repetition to sources/preprocessing | ★★★★☆ |

### 06_internals/ - Internals

| Example | Description | Complexity |
|---------|-------------|------------|
| D01_session_workflow.py | Session-based workflows | ★★★☆☆ |
| D02_custom_controllers.py | Custom controller development | ★★★★★ |
| D03_cache_performance.py | Cache performance comparison (step cache + CoW) | ★★★★☆ |

---

## 📚 Reference

| Example | Description |
|---------|-------------|
| R01_pipeline_syntax.py | Complete pipeline syntax reference |
| R02_generator_reference.py | Generator syntax documentation |
| R03_all_keywords.py | Test all pipeline keywords |
| R05_synthetic_environmental.py | Environmental & matrix effects (Phase 3) |
| R06_synthetic_validation.py | Validation & quality assessment (Phase 4) |
| R07_synthetic_fitter.py | Fitting generators to real data (Phase 4) |

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
```

### Run specific example
```bash
./run.sh -i 1               # By index
./run.sh -n "U01*.py"       # By name pattern
```

### Options
```bash
./run.sh -l                 # Enable logging to log.txt
./run.sh -p                 # Generate plots
./run.sh -s                 # Show plots interactively
./run.sh -p -s              # Generate and show plots
```

---

## 🧪 CI Validation Scripts

For stricter testing that catches silent errors (N/A values, NaN metrics, warnings), use the CI validation scripts. These mirror the GitHub Actions workflow locally.

### Linux/Mac
```bash
./run_ci_examples.sh                    # Run all with strict validation
./run_ci_examples.sh -c user            # User examples only
./run_ci_examples.sh -c developer       # Developer examples only
./run_ci_examples.sh -c reference       # Reference examples only
./run_ci_examples.sh -k                 # Keep going on failures
./run_ci_examples.sh -v                 # Verbose output
./run_ci_examples.sh -c user -k -v      # Combine options
```

### Windows PowerShell
```powershell
.\run_ci_examples.ps1                           # Run all with strict validation
.\run_ci_examples.ps1 -Category user            # User examples only
.\run_ci_examples.ps1 -Category developer       # Developer examples only
.\run_ci_examples.ps1 -KeepGoing                # Don't stop on first failure
.\run_ci_examples.ps1 -VerboseOutput            # Show all output
```

### What CI Scripts Check

The CI scripts detect:
- **Critical errors**: Tracebacks, ValueError, TypeError, ModuleNotFoundError, etc.
- **Warning patterns**: N/A values, NaN metrics, `[!]` warnings, deprecation warnings
- **Invalid results**: 0 samples, 0 predictions, empty results

### Output

Results are saved to `workspace/ci_output/run_YYYYMMDD_HHMMSS/`:
- `summary.txt` - Overall pass/fail summary
- `errors.txt` - Detailed error information
- `*.log` - Individual example output logs

---

## 💡 Tips

1. **Start with U01**: If you're new to nirs4all, start with `user/01_getting_started/U01_hello_world.py`
2. **Per-folder numbering**: Each folder has its own U01, U02, etc. - this makes it easier to add new examples
3. **Follow the path**: Examples within each folder are numbered for sequential learning
4. **Check prerequisites**: Some examples require specific packages (TensorFlow, PyTorch, etc.)
5. **Use `--plots`**: Add `--plots --show` to visualize results
6. **Read docstrings**: Each example has detailed documentation at the top
7. **Check outputs**: Generated files are saved to `workspace/examples_output/`

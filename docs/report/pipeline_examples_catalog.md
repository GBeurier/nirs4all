# Pipeline Examples Catalog: Branching, Merging & Multi-Source

**Version**: 2.0.0
**Date**: December 2025
**Purpose**: 40 progressive examples illustrating nirs4all's branching, merging, concat-transform, and multi-source capabilities.

---

## Table of Contents

1. [Basic Pipelines (1-5)](#basic-pipelines-1-5)
2. [Concat-Transform Pipelines (6-10)](#concat-transform-pipelines-6-10)
3. [Simple Branching Pipelines (11-15)](#simple-branching-pipelines-11-15)
4. [Advanced Branching with Merge (16-20)](#advanced-branching-with-merge-16-20)
5. [Complex Multi-Source & Stacking (21-25)](#complex-multi-source--stacking-21-25)
6. [Merge Output Targets (26-28)](#merge-output-targets-26-30)
7. [Source Branching (29-31)](#source-branching-29-33)
8. [Specialized Branching Scenarios (32-35)](#specialized-branching-scenarios-32-36)
9. [Advanced Stacking Patterns (36-40)](#advanced-stacking-patterns-36-40)

---

## Basic Pipelines (1-5)

### Example 1: Minimal Linear Pipeline

**Use Case**: Single preprocessing step followed by a model. The simplest possible pipeline.

```python
pipeline = [
    SNV(),
    PLSRegression(n_components=10)
]
```

**Explanation**: Data flows linearly through SNV normalization, then trains a PLS model. No branching, no merging—just sequential execution.

---

### Example 2: Multiple Preprocessing Steps

**Use Case**: Chain multiple preprocessing transforms before modeling.

```python
pipeline = [
    Detrend(),
    SNV(),
    SavitzkyGolay(window_length=11, polyorder=2),
    PLSRegression(n_components=15)
]
```

**Explanation**: Preprocessing is applied sequentially (Detrend → SNV → SG smoothing). Each transform sees the output of the previous one. Single execution path.

---

### Example 3: Cross-Validation with Single Model

**Use Case**: Add cross-validation to evaluate model performance.

```python
pipeline = [
    SNV(),
    ShuffleSplit(n_splits=5, test_size=0.2),
    PLSRegression(n_components=10)
]
```

**Explanation**: ShuffleSplit creates 5 train/validation splits. The model trains on each fold, producing out-of-fold predictions for all samples.

---

### Example 4: Target Preprocessing (Y-Transform)

**Use Case**: Apply scaling to both features and target variable.

```python
pipeline = [
    MinMaxScaler(),
    {"y_processing": StandardScaler()},
    KFold(n_splits=5),
    PLSRegression(n_components=10)
]
```

**Explanation**: Features are scaled with MinMax, targets with StandardScaler. The `y_processing` keyword applies transforms to y independently. Predictions are inverse-transformed automatically.

---

### Example 5: Multiple Models (Sequential)

**Use Case**: Train multiple models on the same preprocessed data.

```python
pipeline = [
    SNV(),
    KFold(n_splits=5),
    PLSRegression(n_components=10),
    RandomForestRegressor(n_estimators=100),
    Ridge(alpha=1.0)
]
```

**Explanation**: Three models train sequentially on the same features. Each model sees the same SNV-transformed data. All predictions are stored and can be compared.

---

## Concat-Transform Pipelines (6-10)

### Example 6: Basic Concat-Transform

**Use Case**: Apply multiple transforms in parallel and concatenate results.

```python
pipeline = [
    {"concat_transform": [SNV(), MSC(), Detrend()]},
    PLSRegression(n_components=20)
]
```

**Explanation**: Three transforms run **in parallel** on the same input data. Results are horizontally concatenated: `[SNV_features | MSC_features | Detrend_features]`. If original has 100 wavelengths, output has 300 features.

---

### Example 7: Concat-Transform with Original Features

**Use Case**: Keep original features alongside transformed versions.

```python
pipeline = [
    {"concat_transform": [None, SNV(), D1()]},
    PLSRegression(n_components=25)
]
```

**Explanation**: `None` means "include original features unchanged". Output is `[original | SNV | first_derivative]`. Useful when raw spectra contain complementary information.

---

### Example 8: Nested Concat-Transform

**Use Case**: Create hierarchical feature combinations.

```python
pipeline = [
    {"concat_transform": [
        SNV(),
        {"concat_transform": [D1(), D2()]}
    ]},
    Ridge()
]
```

**Explanation**: Inner concat produces `[D1 | D2]`, outer concat produces `[SNV | D1 | D2]`. Nesting allows complex feature engineering in a single step.

---

### Example 9: Concat-Transform with Dimensionality Reduction

**Use Case**: Apply different transforms, then reduce combined dimensionality.

```python
pipeline = [
    {"concat_transform": [SNV(), MSC(), Detrend()]},
    PCA(n_components=50),
    PLSRegression(n_components=15)
]
```

**Explanation**: Concat creates 300 features (3 × 100), PCA reduces to 50 principal components. This captures the most important variance across all transform variants.

---

### Example 10: Concat-Transform in Feature Augmentation Mode

**Use Case**: Add features without replacing existing processing.

```python
pipeline = [
    SNV(),
    {"feature_augmentation": [D1(), D2()]},
    PLSRegression(n_components=20)
]
```

**Explanation**: First, SNV is applied. Then `feature_augmentation` **adds** derivative features to the existing SNV processing: `[SNV_features | D1_features | D2_features]`. Unlike top-level concat_transform which replaces, this mode extends.

---

## Simple Branching Pipelines (11-15)

### Example 11: Basic Two-Branch Pipeline

**Use Case**: Explore two preprocessing strategies independently.

```python
pipeline = [
    {"branch": [
        [SNV()],
        [MSC()]
    ]},
    PCA(n_components=20),
    {"model": PLSRegression(n_components=10)}
]
```

**Explanation**: Creates 2 parallel execution paths. PCA and PLS run **twice** (once per branch). Results: 2 trained models, 2 sets of predictions. Compare which preprocessing works better.

---

### Example 12: Branching with Different Preprocessing Chains

**Use Case**: Each branch has a different preprocessing pipeline.

```python
pipeline = [
    {"branch": [
        [SNV(), SavitzkyGolay(window_length=11)],
        [Detrend(), D1()],
        [MSC(), PCA(n_components=30)]
    ]},
    {"model": PLSRegression(n_components=10)}
]
```

**Explanation**: Three branches with distinct preprocessing chains. The PLS model trains 3 times on different feature representations. Useful for systematic preprocessing comparison.

---

### Example 13: Branching with Per-Branch Models

**Use Case**: Different models for different preprocessing strategies.

```python
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=10)}],
        [MSC(), {"model": RandomForestRegressor(n_estimators=100)}],
        [Detrend(), {"model": Ridge(alpha=1.0)}]
    ]}
]
```

**Explanation**: Each branch has its own model. No shared steps after branching. Results in 3 independent preprocessing+model combinations. Compare end-to-end pipeline variants.

---

### Example 14: Branching with Shared Model After

**Use Case**: Multiple preprocessing paths converge to the same model type.

```python
pipeline = [
    {"branch": [
        [SNV()],
        [MSC()],
        [Detrend()]
    ]},
    PCA(n_components=20),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
```

**Explanation**: Three branches, but PCA, CV splitter, and PLS are shared steps that run 3 times. Produces 3 PLS models trained on different preprocessing. Systematic comparison with controlled model.

---

### Example 15: Nested Branching

**Use Case**: Combinatorial exploration of preprocessing options.

```python
pipeline = [
    {"branch": [
        [SNV()],
        [MSC()]
    ]},
    {"branch": [
        [PCA(n_components=20)],
        [PCA(n_components=50)]
    ]},
    {"model": PLSRegression(n_components=10)}
]
```

**Explanation**: First branch: 2 options (SNV, MSC). Second branch: 2 options (PCA-20, PCA-50). Result: 2 × 2 = 4 pipeline variants. Useful for grid search over preprocessing parameters.

---

## Advanced Branching with Merge (16-20)

### Example 16: Basic Feature Merge

**Use Case**: Combine features from all branches into a single feature set.

```python
pipeline = [
    {"branch": [
        [SNV()],
        [MSC()],
        [Detrend()]
    ]},
    {"merge": "features"},
    {"model": PLSRegression(n_components=30)}
]
```

**Explanation**: Three branches run independently, then merge **exits branch mode** and concatenates features: `[SNV_features | MSC_features | Detrend_features]`. PLS trains once on combined features. Similar to concat_transform but allows intermediate branch-specific steps.

---

### Example 17: Merge with Branch-Specific Processing

**Use Case**: Each branch has different intermediate steps before merge.

```python
pipeline = [
    {"branch": [
        [SNV(), PCA(n_components=20)],
        [MSC(), SavitzkyGolay(window_length=15)],
        [Detrend(), D1(), PCA(n_components=10)]
    ]},
    {"merge": "features"},
    {"model": Ridge(alpha=1.0)}
]
```

**Explanation**: Branch 0: SNV → PCA(20) = 20 features. Branch 1: MSC → SG = 100 features. Branch 2: Detrend → D1 → PCA(10) = 10 features. Merge: 20 + 100 + 10 = 130 features. Different dimensionality per branch is fine.

---

### Example 18: Prediction Merge (Stacking Foundation)

**Use Case**: Merge model predictions from branches for stacking.

```python
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=10)}],
        [MSC(), {"model": RandomForestRegressor(n_estimators=100)}],
        [Detrend(), {"model": Ridge(alpha=1.0)}]
    ]},
    KFold(n_splits=5),
    {"merge": "predictions"},
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: Three branches each train a model. Merge collects **OOF predictions** (out-of-fold, to prevent data leakage). Result: 3 features (one prediction per branch). Final Ridge is a meta-learner trained on stacked predictions.

---

### Example 19: Mixed Merge (Features + Predictions)

**Use Case**: Combine features from one branch with predictions from another.

```python
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=10)}],
        [PCA(n_components=30)]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": [0],
        "features": [1]
    }},
    {"model": Ridge(alpha=1.0)}
]
```

**Explanation**: Branch 0: SNV → PLS (produces predictions). Branch 1: PCA (produces features only). Merge takes OOF predictions from branch 0 (1 feature) and PCA features from branch 1 (30 features). Ridge trains on 31 combined features.

---

### Example 20: Selective Branch Merge

**Use Case**: Only merge specific branches, ignore others.

```python
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=10)}],
        [MSC(), {"model": RandomForestRegressor()}],
        [Detrend(), {"model": XGBRegressor()}],
        [D1(), {"model": Ridge()}]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": [0, 2]
    }},
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: Four branches, but merge only includes predictions from branches 0 and 2 (PLS and XGB). Branches 1 and 3 are ignored. Useful when you know some models perform better and want to exclude weak contributors.

---

## Complex Multi-Source & Stacking (21-25)

### Example 21: Multi-Source Dataset

**Use Case**: Data from multiple instruments/sensors processed together.

```python
# Dataset configuration with multiple sources
dataset_config = DatasetConfigs([
    {"path": "nir_sensor1.csv", "source_name": "NIR"},
    {"path": "mir_sensor2.csv", "source_name": "MIR"},
    {"path": "raman.csv", "source_name": "Raman"}
])

pipeline = [
    SNV(),  # Applied to each source independently
    {"model": PLSRegression(n_components=15)}
]
```

**Explanation**: Three data sources (NIR, MIR, Raman) from different instruments. SNV applies to each source. PLS sees concatenated features from all sources. Multi-source preserves data provenance.

---

### Example 22: Multi-Source with Source-Specific Processing

**Use Case**: Different preprocessing per data source.

```python
dataset_config = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "raman.csv", "source_name": "Raman"}
])

pipeline = [
    {"source": "NIR", "steps": [SNV(), D1()]},
    {"source": "Raman", "steps": [BaselineCorrection(), Normalize()]},
    {"model": PLSRegression(n_components=20)}
]
```

**Explanation**: NIR data gets SNV + first derivative. Raman data gets baseline correction + normalization. Each source has appropriate preprocessing. Model sees combined features.

---

### Example 23: MetaModel Stacking (Convenience Syntax)

**Use Case**: High-level stacking with automatic OOF handling.

```python
pipeline = [
    SNV(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)},
    {"model": RandomForestRegressor(n_estimators=100)},
    {"model": Ridge(alpha=1.0)},
    {"model": MetaModel(
        model=Ridge(alpha=0.5),
        source_models=["PLSRegression", "RandomForestRegressor", "Ridge"]
    )}
]
```

**Explanation**: Three base models train, then MetaModel collects their OOF predictions and trains a Ridge meta-learner. MetaModel is a convenience wrapper that handles OOF reconstruction automatically. Equivalent to merge + model.

---

### Example 24: Advanced Stacking with Per-Branch Model Selection

**Use Case**: Control which models contribute from each branch.

```python
pipeline = [
    {"branch": [
        [SNV(),
         {"model": PLSRegression(n_components=5)},
         {"model": PLSRegression(n_components=10)},
         {"model": PLSRegression(n_components=15)},
         {"model": PLSRegression(n_components=20)}],
        [MSC(),
         {"model": RandomForestRegressor(n_estimators=50)},
         {"model": RandomForestRegressor(n_estimators=100)}]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "aggregate": "mean"}
        ]
    }},
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: Branch 0 has 4 PLS variants, branch 1 has 2 RF variants. Merge selects **best** PLS from branch 0 (1 feature) and **averages** RF predictions from branch 1 (1 feature). Final meta-learner trains on 2 stacked features.

---

### Example 25: Full Pipeline - Multi-Source, Branching, Merge, Stacking

**Use Case**: Complete real-world pipeline combining all features.

```python
# Multi-source dataset
dataset_config = DatasetConfigs([
    {"path": "nir_lab.csv", "source_name": "NIR_Lab"},
    {"path": "nir_portable.csv", "source_name": "NIR_Portable"}
])

pipeline = [
    # Source-specific preprocessing
    {"source": "NIR_Lab", "steps": [SNV()]},
    {"source": "NIR_Portable", "steps": [SNV(), TransferCorrection()]},

    # Branch for different modeling strategies
    {"branch": [
        # Branch 0: Traditional chemometrics
        [PCA(n_components=30),
         {"model": PLSRegression(n_components=10)},
         {"model": PLSRegression(n_components=15)}],

        # Branch 1: Machine learning ensemble
        [{"concat_transform": [D1(), D2()]},
         {"model": RandomForestRegressor(n_estimators=100)},
         {"model": XGBRegressor(n_estimators=100)}],

        # Branch 2: Feature selection path
        [SelectKBest(k=50),
         {"model": Ridge(alpha=1.0)}]
    ]},

    # Cross-validation
    KFold(n_splits=5),

    # Merge: best from branches 0&1, features from branch 2
    {"merge": {
        "predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},
            {"branch": 1, "select": {"top_k": 2}, "aggregate": "weighted_mean"}
        ],
        "features": [2]
    }},

    # Meta-learner on combined features + predictions
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**:
1. **Multi-source**: NIR from lab and portable instruments, with transfer correction for portable.
2. **Branching**: Three strategies - chemometrics (PLS), ML ensemble (RF+XGB), and feature selection (SelectKBest+Ridge).
3. **Per-branch selection**: Best PLS from branch 0, weighted average of top-2 from branch 1.
4. **Mixed merge**: Predictions from branches 0&1, features from branch 2.
5. **Final stacking**: Ridge meta-learner combines all inputs.

This represents a production-grade pipeline combining sensor fusion, transfer learning, multiple modeling paradigms, and ensemble stacking.

---

## Merge Output Targets (26-30)

### Example 26: Merge to Sources

**Use Case**: Convert branches to separate sources for different downstream processing.

```python
pipeline = [
    {"branch": [
        [SNV(), PCA(n_components=20)],
        [MSC(), PCA(n_components=30)]
    ]},
    {"merge": {
        "features": "all",
        "output_as": "sources",
        "source_names": ["snv_features", "msc_features"]
    }},
    # Now we have 2 sources instead of branches
    # Each source can be processed differently downstream
    {"source": "snv_features", "steps": [StandardScaler()]},
    {"source": "msc_features", "steps": [MinMaxScaler()]},
    {"model": PLSRegression(n_components=15)}
]
```

**Explanation**: After merge with `output_as: "sources"`, the branch outputs become independent data sources. This allows source-specific processing after branch merging. Useful when branches represent different data modalities that need separate downstream handling.

---

### Example 27: Predictions to Sources for Multi-Head Models

**Use Case**: Keep branch predictions as separate sources for multi-head neural networks.

```python
pipeline = [
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=10)}],
        [MSC(), {"model": RandomForestRegressor()}],
        [Detrend(), {"model": XGBRegressor()}]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": "all",
        "output_as": "sources"  # Each branch's predictions become a source
    }},
    # Multi-head model receives predictions as separate inputs
    {"model": MultiHeadNN(heads_per_source=True)}
]
```

**Explanation**: Instead of concatenating predictions into a single feature vector, each branch's predictions become a separate source. Multi-head architectures can then process each prediction stream independently before fusion.

---

### Example 28: Merge to Dict for Custom Processing

**Use Case**: Keep branch outputs as a structured dictionary for programmatic access.

```python
pipeline = [
    {"branch": [
        [SNV(), PCA(n_components=50)],
        [MSC(), SelectKBest(k=30)],
        [D1(), VarianceThreshold()]
    ]},
    {"merge": {
        "features": "all",
        "output_as": "dict"  # Returns {"branch_0": array, "branch_1": array, ...}
    }},
    # Custom layer can access branches by key
    {"model": CustomFusionModel(
        branch_weights={"branch_0": 0.5, "branch_1": 0.3, "branch_2": 0.2}
    )}
]
```

**Explanation**: With `output_as: "dict"`, merged features remain structured. Custom models can access each branch's output by key for weighted fusion, attention mechanisms, or other branch-aware processing.

---

## Source Branching (29-33)

### Example 29: Source-Specific Branching

**Use Case**: Create branches that process different data sources independently.

```python
dataset_config = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "raman.csv", "source_name": "Raman"},
    {"path": "markers.csv", "source_name": "Markers"}
])

pipeline = [
    # Each source gets its own processing branch
    {"source_branch": {
        "NIR": [SNV(), SavitzkyGolay(window_length=11)],
        "Raman": [BaselineCorrection(), Normalize()],
        "Markers": [VarianceThreshold(), StandardScaler()]
    }},
    {"merge_sources": "concat"},  # Combine processed sources
    {"model": PLSRegression(n_components=20)}
]
```

**Explanation**: `source_branch` creates per-source pipelines automatically. Each source is processed by its dedicated branch, then merged. Unlike regular branching (which duplicates the full dataset per branch), source branching keeps data separate by origin.

---

### Example 30: Multi-Source Merge Strategies

**Use Case**: Control how multiple data sources are combined.

```python
dataset_config = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},          # 500 features
    {"path": "markers.csv", "source_name": "Markers"}   # 1000 features
])

# Strategy 1: Horizontal concatenation (default)
pipeline_concat = [
    SNV(),
    {"merge_sources": "concat"},  # → 1500 features
    {"model": PLSRegression(n_components=15)}
]

# Strategy 2: Stack as 3D array (requires compatible shapes)
pipeline_stack = [
    SNV(),
    PCA(n_components=100),  # Make both sources same dim
    {"merge_sources": "stack"},  # → (samples, 2, 100)
    {"model": Conv1DRegressor()}
]

# Strategy 3: Keep as dict for multi-input models
pipeline_dict = [
    SNV(),
    {"merge_sources": "dict"},  # → {"NIR": array, "Markers": array}
    {"model": MultiInputNN()}
]
```

**Explanation**: `merge_sources` controls how multi-source data is combined. `"concat"` creates a wide feature matrix. `"stack"` creates a 3D tensor (for CNNs). `"dict"` preserves source structure for multi-input architectures.

---

### Example 31: Multi-Source with Asymmetric Processing Counts

**Use Case**: Handle sources with different processing histories.

```python
dataset_config = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "markers.csv", "source_name": "Markers"}
])

pipeline = [
    # NIR gets 3 processings, Markers gets 1
    {"source": "NIR", "steps": [
        {"feature_augmentation": [SNV(), MSC(), D1()]}  # 3 processings
    ]},
    {"source": "Markers", "steps": [StandardScaler()]},  # 1 processing

    # Merge with incompatibility handling
    {"merge_sources": {
        "strategy": "concat",
        "on_incompatible": "flatten"  # 2D concat despite different processing counts
    }},
    {"model": PLSRegression(n_components=20)}
]
```

**Explanation**: When sources have different processing counts (different numbers of feature variants), 3D stacking fails. `on_incompatible: "flatten"` forces 2D concatenation, summing all features regardless of processing structure.

---

## Specialized Branching Scenarios (32-36)

### Example 32: Branch per Wavelength Region

**Use Case**: Apply different processing to different spectral regions.

```python
pipeline = [
    # Split spectrum into regions and process differently
    {"branch": [
        # UV region (200-400 nm): high noise, needs heavy smoothing
        [WavelengthSelect(200, 400), SavitzkyGolay(window_length=21), SNV()],

        # VIS region (400-800 nm): stable, light processing
        [WavelengthSelect(400, 800), SNV()],

        # NIR region (800-2500 nm): baseline issues
        [WavelengthSelect(800, 2500), Detrend(), MSC()]
    ]},
    {"merge": "features"},  # Combine processed regions
    {"model": PLSRegression(n_components=15)}
]
```

**Explanation**: Different spectral regions often require different preprocessing. Branching by wavelength region allows region-specific treatment before merging into a unified feature set.

---

### Example 33: Branch for Outlier Handling

**Use Case**: Create parallel paths with and without outlier removal.

```python
pipeline = [
    SNV(),  # Common preprocessing
    {"branch": [
        # Path 1: Keep all samples (robust to outliers)
        [{"model": PLSRegression(n_components=10)}],

        # Path 2: Remove Mahalanobis outliers before modeling
        [OutlierRemoval(method="mahalanobis", threshold=3.0),
         {"model": PLSRegression(n_components=10)}],

        # Path 3: Remove high-leverage samples
        [OutlierRemoval(method="leverage", threshold=0.5),
         {"model": PLSRegression(n_components=10)}]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": "all",
        "on_missing": "skip"  # Handle if some folds lose all samples
    }},
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: Different outlier strategies can dramatically affect model performance. Branching allows comparing strategies, and stacking predictions from different outlier-handling approaches can improve robustness.

---

### Example 34: Branch for Data Quality Scenarios

**Use Case**: Handle samples with different quality levels differently.

```python
pipeline = [
    # Branch based on sample metadata (e.g., signal quality score)
    {"branch_on_metadata": {
        "quality_score > 0.8": [SNV(), {"model": PLSRegression(n_components=15)}],
        "0.5 <= quality_score <= 0.8": [SNV(), SavitzkyGolay(21), {"model": PLSRegression(n_components=10)}],
        "quality_score < 0.5": [HeavySmoothing(), {"model": RobustRegressor()}]
    }},
    KFold(n_splits=5),
    {"merge": "predictions"},
    {"model": Ridge()}
]
```

**Explanation**: When samples have varying quality, one-size-fits-all preprocessing may not work. Metadata-based branching routes samples to appropriate processing paths based on quality indicators.

---

### Example 35: Branch for Cross-Validation Strategy Comparison

**Use Case**: Compare different CV strategies in parallel.

```python
pipeline = [
    SNV(),
    {"branch": [
        # Random splitting
        [ShuffleSplit(n_splits=10, test_size=0.2),
         {"model": PLSRegression(n_components=10)}],

        # Kennard-Stone based splitting
        [KennardStoneSplit(n_splits=10),
         {"model": PLSRegression(n_components=10)}],

        # Group-based splitting (by farm/batch)
        [GroupKFold(n_splits=10, groups="farm_id"),
         {"model": PLSRegression(n_components=10)}]
    ]},
    # Each branch produces predictions with different CV strategies
    # Compare results after pipeline runs
]
```

**Explanation**: Different CV strategies suit different data characteristics. Branching allows parallel evaluation of CV methods, helping choose the most appropriate for deployment.

---

## Advanced Stacking Patterns (36-40)

### Example 36: Complex Model Selection with Metrics

**Use Case**: Select models based on validation performance metrics.

```python
pipeline = [
    {"branch": [
        [SNV(),
         {"model": PLSRegression(n_components=5)},
         {"model": PLSRegression(n_components=10)},
         {"model": PLSRegression(n_components=15)},
         {"model": PLSRegression(n_components=20)},
         {"model": PLSRegression(n_components=25)}],
        [MSC(),
         {"model": RandomForestRegressor(n_estimators=50)},
         {"model": RandomForestRegressor(n_estimators=100)},
         {"model": RandomForestRegressor(n_estimators=200)}]
    ]},
    KFold(n_splits=5),
    {"merge": {
        "predictions": [
            # Branch 0: Select top 2 PLS models by RMSE
            {"branch": 0, "select": {"top_k": 2}, "metric": "rmse", "aggregate": "separate"},
            # Branch 1: Select best RF by R², then average
            {"branch": 1, "select": "best", "metric": "r2"}
        ]
    }},
    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: With many model variants per branch, selecting only the best performers reduces noise in the stacking layer. Different metrics can be used per branch depending on what matters most for that preprocessing path.

---

### Example 37: Weighted Ensemble by Validation Score

**Use Case**: Create a weighted average of predictions based on model performance.

```python
pipeline = [
    SNV(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)},
    {"model": PLSRegression(n_components=15)},
    {"model": RandomForestRegressor(n_estimators=100)},
    {"model": XGBRegressor(n_estimators=100)},
    {"model": SVR(kernel='rbf')},
    {"merge": {
        "predictions": [
            {"models": "all", "aggregate": "weighted_mean", "weight_metric": "r2"}
        ]
    }}
    # No meta-model needed - weighted average is the final prediction
]
```

**Explanation**: Instead of training a meta-model, predictions are combined using a weighted average where weights come from validation R² scores. Models with better validation performance contribute more to the final prediction.

---

### Example 38: Classification Stacking with Probability Averaging

**Use Case**: Stack classifiers using averaged class probabilities.

```python
pipeline = [
    SNV(),
    StratifiedKFold(n_splits=5),
    {"model": LogisticRegression()},
    {"model": RandomForestClassifier(n_estimators=100)},
    {"model": XGBClassifier()},
    {"model": GaussianNB()},
    {"merge": {
        "predictions": [
            {"models": "all", "proba": True, "aggregate": "proba_mean"}
        ]
    }},
    # Meta-classifier trains on averaged probabilities
    {"model": LogisticRegression()}
]
```

**Explanation**: For classification, using probability outputs (not class labels) preserves uncertainty information. `proba_mean` averages class probabilities across models, giving K features (one per class) to the meta-classifier.

---

### Example 39: Hierarchical Stacking (Two-Level)

**Use Case**: Build a two-level stacking ensemble.

```python
pipeline = [
    SNV(),
    KFold(n_splits=5),

    # Level 1: Base learners grouped by type
    {"branch": [
        # Linear models
        [{"model": PLSRegression(n_components=10)},
         {"model": Ridge(alpha=1.0)},
         {"model": Lasso(alpha=0.1)}],

        # Tree-based models
        [{"model": RandomForestRegressor(n_estimators=100)},
         {"model": XGBRegressor()},
         {"model": LGBMRegressor()}],

        # Kernel methods
        [{"model": SVR(kernel='rbf')},
         {"model": KernelRidge(kernel='rbf')}]
    ]},

    # Level 1.5: Average within each group
    {"merge": {
        "predictions": [
            {"branch": 0, "aggregate": "mean"},  # 1 linear prediction
            {"branch": 1, "aggregate": "mean"},  # 1 tree prediction
            {"branch": 2, "aggregate": "mean"}   # 1 kernel prediction
        ]
    }},

    # Level 2: Stack the group averages
    {"model": ElasticNet(alpha=0.5, l1_ratio=0.5)}
]
```

**Explanation**: Two-level stacking first averages predictions within model families (linear, tree, kernel), then combines family predictions. This reduces overfitting compared to stacking all models directly.

---

### Example 40: Multi-Source Stacking with Source-Aware Selection

**Use Case**: Stack models trained on different data sources with source-aware selection.

```python
dataset_config = DatasetConfigs([
    {"path": "nir.csv", "source_name": "NIR"},
    {"path": "markers.csv", "source_name": "Markers"}
])

pipeline = [
    # Source-specific preprocessing
    {"source": "NIR", "steps": [SNV(), D1()]},
    {"source": "Markers", "steps": [VarianceThreshold(), StandardScaler()]},

    KFold(n_splits=5),

    # Branch for different modeling strategies
    {"branch": [
        # Spectral specialists
        [SourceSelect("NIR"),
         {"model": PLSRegression(n_components=10)},
         {"model": PLSRegression(n_components=20)}],

        # Marker specialists
        [SourceSelect("Markers"),
         {"model": RandomForestRegressor(n_estimators=100)}],

        # Fusion: use both sources
        [{"model": PLSRegression(n_components=15)},
         {"model": XGBRegressor()}]
    ]},

    # Selective merge: best from each branch
    {"merge": {
        "predictions": [
            {"branch": 0, "select": "best", "metric": "rmse"},  # Best NIR model
            {"branch": 1, "select": "all"},                      # Markers model
            {"branch": 2, "select": "best", "metric": "rmse"}   # Best fusion model
        ]
    }},

    {"model": Ridge(alpha=0.5)}
]
```

**Explanation**: Multi-source stacking combines specialists (models trained on single sources) with fusers (models trained on all sources). Source-aware selection ensures each data modality contributes its best model to the ensemble.

---

## Summary Table

| Example | Features Used | Complexity |
|---------|---------------|------------|
| 1-5 | Linear pipeline, CV, y-transform, multi-model | ⭐ Basic |
| 6-10 | concat_transform, nesting, augmentation | ⭐⭐ Intermediate |
| 11-15 | branch, nested branching, per-branch models | ⭐⭐⭐ Advanced |
| 16-20 | merge (features/predictions), mixed merge | ⭐⭐⭐⭐ Expert |
| 21-25 | multi-source, MetaModel, full stacking | ⭐⭐⭐⭐⭐ Production |
| 26-28 | merge output_as (sources, dict) | ⭐⭐⭐⭐ Expert |
| 29-31 | source_branch, merge_sources | ⭐⭐⭐⭐ Expert |
| 32-35 | wavelength regions, outliers, quality routing | ⭐⭐⭐⭐⭐ Domain |
| 36-40 | complex selection, weighted ensembles, hierarchical stacking | ⭐⭐⭐⭐⭐ Production |

---

## Key Concepts Reference

| Concept | Keyword | Purpose |
|---------|---------|---------|
| **concat_transform** | `{"concat_transform": [...]}` | Parallel transforms, single path |
| **branch** | `{"branch": [[...], [...]]}` | Multiple execution paths |
| **merge** | `{"merge": ...}` | Exit branch mode, combine outputs |
| **merge output_as** | `{"merge": {..., "output_as": "sources"}}` | Control merge output format |
| **source_branch** | `{"source_branch": {...}}` | Per-source processing branches |
| **merge_sources** | `{"merge_sources": ...}` | Combine multi-source data |
| **source** | Multi-source dataset | Data from different origins |
| **MetaModel** | `{"model": MetaModel(...)}` | Convenience stacking wrapper |
| **select/aggregate** | Per-branch prediction control | Fine-grained model selection |

---

## Design Principles Illustrated

1. **Branches ≠ Sources**: Branches are computation paths; sources are data origins.
2. **Merge is explicit**: Always required to exit branch mode.
3. **OOF by default**: Prediction merging uses out-of-fold reconstruction.
4. **Composability**: Complex pipelines built from simple primitives.
5. **Flexibility over rigidity**: Multiple ways to achieve the same result.
6. **Output control**: `output_as` controls whether merge produces features, sources, or dicts.
7. **Source awareness**: Multi-source datasets maintain provenance through source branching.
8. **Asymmetric support**: Branches can have different structures (models, transforms, dimensions).
9. **Per-branch control**: Selection and aggregation strategies can differ across branches.

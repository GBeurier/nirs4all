# Meta-Model Stacking User Guide

## Overview

Meta-model stacking (stacked generalization) is an ensemble technique that combines predictions from multiple base models using a second-level "meta-learner". This approach often improves prediction accuracy by leveraging the complementary strengths of different models.

In nirs4all, the `MetaModel` operator provides a flexible, robust implementation of stacking that:

- **Prevents data leakage** through out-of-fold (OOF) predictions
- **Supports flexible source selection** (all, explicit, top-K, diversity)
- **Handles edge cases** with configurable coverage strategies
- **Integrates with branches** for multi-preprocessing pipelines
- **Persists and reloads** seamlessly for production use

## Quick Start

### Basic Stacking Pipeline

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineRunner, PipelineConfigs
from nirs4all.operators.models import MetaModel

# Load dataset
dataset = DatasetConfigs("path/to/data/")

# Pipeline with base models and meta-learner
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),  # Required for OOF
    PLSRegression(n_components=5),                      # Base model 1
    RandomForestRegressor(n_estimators=50),             # Base model 2
    {"model": MetaModel(model=Ridge(alpha=1.0))},       # Meta-learner
]

runner = PipelineRunner()
predictions, _ = runner.run(PipelineConfigs(pipeline, "Stacking"), dataset)
```

## Core Concepts

### Out-of-Fold (OOF) Predictions

Stacking requires predictions that were **not** made on the training data to avoid leakage. During cross-validation:

1. Each fold's model predicts on its validation set
2. These validation predictions become training features for the meta-model
3. The meta-model sees the same samples as the base models, but through their predictions

```
Fold 1: Train on [2,3,4,5], Predict on [1] → OOF for samples in fold 1
Fold 2: Train on [1,3,4,5], Predict on [2] → OOF for samples in fold 2
...
Result: Complete OOF predictions for all training samples
```

### Source Model Selection

The `source_models` parameter controls which base models contribute to the meta-learner:

| Mode | Syntax | Description |
|------|--------|-------------|
| All Previous | `source_models="all"` (default) | Use all models before the MetaModel |
| Explicit | `source_models=["Model1", "Model2"]` | Use specific named models |
| Top-K | `source_models={"top_k": 3, "metric": "r2"}` | Best N models by metric |
| Diversity | `source_models={"diversity": True, "max_models": 5}` | Diverse model selection |

### Coverage Strategies

When some samples lack OOF predictions (e.g., excluded samples), coverage strategies determine behavior:

| Strategy | Enum | Behavior |
|----------|------|----------|
| Strict | `CoverageStrategy.STRICT` | Error if any sample missing (default) |
| Drop | `CoverageStrategy.DROP_INCOMPLETE` | Mask incomplete samples |
| Impute Zero | `CoverageStrategy.IMPUTE_ZERO` | Fill missing with 0 |
| Impute Mean | `CoverageStrategy.IMPUTE_MEAN` | Fill missing with column mean |

### Test Aggregation

Multiple folds produce multiple test predictions. Aggregation strategies combine them:

| Strategy | Enum | Behavior |
|----------|------|----------|
| Mean | `TestAggregation.MEAN` | Simple average (default) |
| Weighted | `TestAggregation.WEIGHTED_MEAN` | Weight by validation scores |
| Best | `TestAggregation.BEST_FOLD` | Use only best-scoring fold |

## Configuration Reference

### MetaModel Parameters

```python
MetaModel(
    model,                    # Required: sklearn-compatible meta-learner
    source_models="all",      # Source selection mode
    use_proba=False,          # Use probabilities (classification)
    stacking_config=None,     # StackingConfig instance
)
```

### StackingConfig Parameters

```python
from nirs4all.operators.models import StackingConfig, CoverageStrategy, TestAggregation, BranchScope

config = StackingConfig(
    coverage_strategy=CoverageStrategy.STRICT,    # How to handle missing OOF
    test_aggregation=TestAggregation.MEAN,        # How to aggregate test preds
    branch_scope=BranchScope.CURRENT_ONLY,        # Which branches to use
    min_coverage_ratio=1.0,                       # Minimum required coverage
    allow_no_cv=False,                            # Allow non-CV pipelines
)
```

## Usage Patterns

### Pattern 1: Named Source Selection

Select specific models by name:

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=3), "name": "PLS_3"},
    {"model": PLSRegression(n_components=5), "name": "PLS_5"},
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
    RandomForestRegressor(n_estimators=100),  # Not selected

    # Only use PLS models
    {"model": MetaModel(
        model=Ridge(),
        source_models=["PLS_3", "PLS_5", "PLS_10"],
    )},
]
```

### Pattern 2: Top-K Selection

Automatically select best models:

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    # Many base models...
    PLSRegression(n_components=3),
    PLSRegression(n_components=5),
    PLSRegression(n_components=10),
    RandomForestRegressor(n_estimators=50),
    GradientBoostingRegressor(n_estimators=50),

    # Select top 3 by validation R²
    {"model": MetaModel(
        model=Ridge(),
        source_models={"top_k": 3, "metric": "r2"},
    )},
]
```

### Pattern 3: Robust Configuration

Handle missing predictions gracefully:

```python
from nirs4all.operators.models import StackingConfig, CoverageStrategy, TestAggregation

config = StackingConfig(
    coverage_strategy=CoverageStrategy.IMPUTE_MEAN,   # Fill gaps
    test_aggregation=TestAggregation.WEIGHTED_MEAN,   # Weight by performance
    min_coverage_ratio=0.8,                           # Allow up to 20% missing
)

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),
    PLSRegression(n_components=5),
    {"model": MetaModel(model=Ridge(), stacking_config=config)},
]
```

### Pattern 4: Branch Stacking

Stack models from preprocessing branches:

```python
from nirs4all.operators.transforms import FirstDerivative, SecondDerivative
from nirs4all.operators.models import MetaModel, StackingConfig, BranchScope

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    {"branch": [
        [PLSRegression(n_components=5)],                     # Raw
        [FirstDerivative(), PLSRegression(n_components=5)],  # D1
        [SecondDerivative(), PLSRegression(n_components=5)], # D2
    ]},

    {"merge": "predictions"},

    # Stack all branch models
    {"model": MetaModel(
        model=Ridge(),
        stacking_config=StackingConfig(
            branch_scope=BranchScope.ALL_BRANCHES,
        ),
    )},
]
```

### Pattern 5: Multi-Level Stacking

Create hierarchical stacking:

```python
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Level 0: Base models
    {"model": PLSRegression(n_components=3), "name": "PLS_L0"},
    {"model": PLSRegression(n_components=10), "name": "PLS10_L0"},
    RandomForestRegressor(n_estimators=50),

    # Level 1: Stack PLS models only
    {"model": MetaModel(
        model=Ridge(),
        source_models=["PLS_L0", "PLS10_L0"],
    ), "name": "Meta_L1"},

    # Level 2: Final meta-model
    {"model": MetaModel(
        model=Lasso(alpha=0.1),
    ), "name": "Meta_L2"},
]
```

### Pattern 6: Classification Stacking

Stack classification models:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    RandomForestClassifier(n_estimators=50),
    LinearDiscriminantAnalysis(),

    # Stack with probabilities
    {"model": MetaModel(
        model=LogisticRegression(),
        use_proba=True,  # Use class probabilities as features
    )},
]
```

## Best Practices

### ✅ DO

- **Always use cross-validation** - Stacking requires OOF predictions
- **Set random_state** - Ensure reproducibility
- **Start simple** - Begin with default settings, then tune
- **Use diverse base models** - Mix linear and non-linear models
- **Name your models** - Makes source selection clearer
- **Test on held-out data** - Validate improvement on unseen data

### ❌ DON'T

- **Stack too many models** - Diminishing returns, consider top-K
- **Ignore base model quality** - Bad base models hurt stacking
- **Use complex meta-learner** - Simple models (Ridge, Linear) often best
- **Forget to check coverage** - Ensure OOF predictions are complete
- **Over-engineer** - Sometimes a single good model is enough

## Troubleshooting

### Common Errors

**"No source models found"**
```
Solution: Ensure base models are defined before MetaModel in pipeline
```

**"Incomplete OOF coverage"**
```
Solution:
1. Check that KFold or similar is in pipeline
2. Use CoverageStrategy.DROP_INCOMPLETE or IMPUTE_MEAN
```

**"Source model not found: ModelName"**
```
Solution: Verify model names match exactly (case-sensitive)
```

**"No fold data found"**
```
Solution: Ensure cross-validation splitter is before base models
```

### Debugging Tips

1. **Check predictions store**:
   ```python
   predictions = runner.predictions
   for pred in predictions.filter(partition="val"):
       print(f"{pred['model_name']}: fold={pred.get('fold_id')}")
   ```

2. **Verify source models exist**:
   ```python
   # After run
   all_models = predictions.filter(partition="val")
   model_names = set(p['model_name'] for p in all_models)
   print(f"Available models: {model_names}")
   ```

3. **Check coverage**:
   ```python
   meta_preds = predictions.filter(model_name_contains="MetaModel")
   if meta_preds:
       print(f"Coverage: {meta_preds[0].get('coverage_ratio', 'N/A')}")
   ```

## API Reference

### MetaModel

```python
class MetaModel:
    """
    Meta-model operator for stacked generalization.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible meta-learner (e.g., Ridge, LogisticRegression)
    source_models : str, list, or dict
        Source model selection:
        - "all": Use all previous models (default)
        - ["name1", "name2"]: Use specific named models
        - {"top_k": N, "metric": "r2"}: Use top N by metric
        - {"diversity": True}: Use diverse selection
    use_proba : bool
        For classification: use probabilities instead of predictions
    stacking_config : StackingConfig
        Configuration for coverage and aggregation strategies
    """
```

### StackingConfig

```python
@dataclass
class StackingConfig:
    """
    Configuration for meta-model stacking behavior.

    Attributes
    ----------
    coverage_strategy : CoverageStrategy
        How to handle missing OOF predictions
    test_aggregation : TestAggregation
        How to combine fold predictions for test set
    branch_scope : BranchScope
        Which branches contribute source models
    min_coverage_ratio : float
        Minimum required sample coverage (0.0-1.0)
    allow_no_cv : bool
        Allow stacking without cross-validation
    """
```

### Enums

```python
class CoverageStrategy(Enum):
    STRICT = "strict"              # Error if incomplete
    DROP_INCOMPLETE = "drop"       # Mask incomplete samples
    IMPUTE_ZERO = "impute_zero"    # Fill with 0
    IMPUTE_MEAN = "impute_mean"    # Fill with column mean

class TestAggregation(Enum):
    MEAN = "mean"                  # Simple average
    WEIGHTED_MEAN = "weighted"     # Weight by val scores
    BEST_FOLD = "best"             # Use best fold only

class BranchScope(Enum):
    CURRENT_ONLY = "current"       # Only current branch
    ALL_BRANCHES = "all"           # All branches
    SPECIFIED = "specified"        # Explicitly listed
```

## See Also

- {doc}`branching` - Pipeline branching for preprocessing variations
- {doc}`writing_pipelines` - General pipeline usage
- {doc}`/reference/pipeline_syntax` - Complete pipeline syntax reference

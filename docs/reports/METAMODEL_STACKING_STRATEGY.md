# Meta-Model Stacking Strategy

## Executive Summary

This document specifies the implementation of **meta-model stacking** in nirs4all: a feature enabling users to train models that use predictions from previously trained pipeline models as input features. This represents a key advancement in building sophisticated ensemble architectures where a meta-learner combines outputs from multiple base models.

---

## 1. Objective Definition

### 1.1 Problem Statement

In machine learning, **stacking** (or stacked generalization) is an ensemble technique where:
1. **Level-0 models** (base models) are trained on the original features
2. **Level-1 model** (meta-model) is trained on the predictions of the base models

The primary challenge is **preventing data leakage**: if the meta-model trains on predictions made by base models on the same samples they were trained on, overfitting occurs. The solution is to use **out-of-fold (OOF) predictions**.

### 1.2 Core Objective

**Enable users to train a meta-model on the predictions of other models already trained in the current pipeline, using a reconstructed training set built from fold predictions, without data leakage.**

Specifically:
- Base models train normally, storing predictions in `prediction_store`
- Meta-model constructs its training features from **validation partition predictions across all folds**
- The reconstruction must be **sample-aligned**: each sample's feature comes from a fold where it was NOT used for training
- The meta-model can then be serialized and used for prediction on new data

### 1.3 Key Requirements

| Requirement | Description |
|-------------|-------------|
| **No Leakage** | Meta-model training uses ONLY out-of-fold predictions |
| **Sample Alignment** | Training set reconstructed with correct sample indices |
| **Branch Compatibility** | Works with branching, including sample partitioning |
| **Flexible Source Selection** | User can select which models to include in the stack |
| **Serialization** | Complete save/load cycle with dependency tracking |
| **Cross-Framework** | Works with sklearn, TF, PyTorch, JAX base models |

### 1.4 Non-Goals (Explicitly Out of Scope for v1)

- Multi-level stacking (stacking on stacking) - future enhancement
- Automatic model selection via optimization - future enhancement
- Parallel execution of meta-model with base models - not possible by design

---

## 2. State of the Art (État des Lieux)

### Current Architecture

#### Pipeline Execution Flow
```
PipelineRunner → PipelineOrchestrator → PipelineExecutor → StepRunner → Controller
```

1. **PipelineRunner**: Entry point, delegates to orchestrator
2. **PipelineOrchestrator**: Manages multi-dataset/multi-pipeline execution
3. **PipelineExecutor**: Executes single pipeline, manages `prediction_store` (Predictions instance)
4. **StepRunner**: Parses steps and routes to appropriate controllers
5. **Controllers**: Execute operators (models, transforms, splitters, etc.)

#### Model Controller Pattern

Controllers follow an **operator-controller pattern**:
- **Operators** (`nirs4all/operators/models/`): Define WHAT models to use (sklearn, tensorflow, pytorch, jax, autogluon)
- **Controllers** (`nirs4all/controllers/models/`): Define HOW to execute them

Controllers are selected via **ControllerRouter** using:
- `matches()` class method (checks step type, operator, keyword)
- Priority system (lower number = higher priority)
- Registry pattern (`@register_controller` decorator)

#### Data Flow in Model Controllers

```python
# BaseModelController.execute()
X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled = self.get_xy(dataset, context)
folds = dataset.folds

# For each fold: train → predict → store predictions
for train_idx, val_idx in folds:
    X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
    # ... train and predict
    prediction_store.add_prediction(
        partition="train"|"val"|"test",
        y_true=y_true, y_pred=y_pred,
        sample_indices=indices,
        fold_id=fold_idx,
        ...
    )
```

#### Predictions Storage Structure

The `Predictions` class stores predictions with the following key fields:
- `id`: Unique prediction identifier
- `dataset_name`, `config_name`, `pipeline_uid`
- `model_name`, `model_classname`, `step_idx`, `op_counter`
- `fold_id`: Cross-validation fold identifier
- `partition`: "train", "val", or "test"
- `sample_indices`: Original sample indices from dataset
- `y_true`, `y_pred`: Actual and predicted values (stored in ArrayRegistry)
- `scores`: Evaluation metrics per partition

#### Cross-Validation and Folds

Folds are managed at the dataset level (`dataset.folds`) and passed to controllers. Each fold generates:
- **Train partition predictions**: In-fold training data predictions
- **Validation partition predictions**: Out-of-fold (OOF) predictions
- **Test partition predictions**: Holdout test set predictions

### Current Stacking Support (sklearn Native)

Q18 example demonstrates sklearn's `StackingRegressor`/`StackingClassifier`:
- Uses sklearn's internal stacking mechanism
- Base estimators and meta-learner defined upfront
- No access to nirs4all's cross-validation or prediction storage
- Cannot leverage predictions from previous pipeline steps

### Gap Analysis

| Feature | Current State | Desired State |
|---------|---------------|---------------|
| Stacking base models | sklearn-internal only | Pipeline-level stacking |
| Meta-model inputs | Raw features (X) | Previous predictions |
| Fold alignment | sklearn manages internally | Respects pipeline CV folds |
| Prediction access | Not exposed | Available via prediction_store |
| Heterogeneous models | Same-framework only | Cross-framework (sklearn + TF + PyTorch) |

### 2.2 Proposed Logic (Logique Proposée)

#### Core Concept: MetaModelController

Introduce a new controller type that:
1. Uses `y` from the dataset (unchanged)
2. Constructs `X` from predictions of previous models in the current pipeline
3. Respects the fold structure for proper out-of-fold stacking
4. Works with any sklearn-compatible meta-learner

#### Architecture Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pipeline Execution                          │
├─────────────────────────────────────────────────────────────────────┤
│  Step 1: Splitter (KFold, GroupKFold, etc.)                        │
│  Step 2: Model A (PLS)           → predictions stored               │
│  Step 3: Model B (RandomForest)  → predictions stored               │
│  Step 4: Model C (XGBoost)       → predictions stored               │
│  Step 5: MetaModel (Ridge)       ← uses predictions from A, B, C    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key Design Decisions

##### 2.2.1 Meta-Model Operator

Create a lightweight operator wrapper to signal meta-model intent:

```python
# nirs4all/operators/models/meta.py
class MetaModel:
    """Wrapper for meta-model stacking using pipeline predictions.

    Args:
        model: Any sklearn-compatible model (or TF/PyTorch/JAX model)
        source_models: List of model names or step indices to use as features
                      If None, uses ALL previous models in the pipeline
        use_proba: For classification, use probabilities instead of class predictions
        include_original_features: Whether to include original X features (passthrough)
    """
    def __init__(
        self,
        model,
        source_models: Optional[List[Union[str, int]]] = None,
        use_proba: bool = True,
        include_original_features: bool = False
    ):
        self.model = model
        self.source_models = source_models
        self.use_proba = use_proba
        self.include_original_features = include_original_features
```

##### 2.2.2 MetaModelController

Create a dedicated controller inheriting from BaseModelController:

```python
# nirs4all/controllers/models/meta_model.py
@register_controller
class MetaModelController(BaseModelController):
    """Controller for meta-model stacking using pipeline predictions."""

    priority = 5  # Higher priority than SklearnModelController (6)

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Match MetaModel operators or dict with 'meta_model' key."""
        if isinstance(operator, MetaModel):
            return True
        if isinstance(step, dict) and 'meta_model' in step:
            return True
        return False

    def get_xy(self, dataset, context) -> Tuple:
        """Override to construct X from predictions instead of dataset features."""
        # Get y from dataset (unchanged)
        y_train, y_test, y_train_unscaled, y_test_unscaled = self._get_y_from_dataset(dataset, context)

        # Construct X from predictions
        X_train, X_test = self._build_features_from_predictions(
            prediction_store=self.prediction_store,
            dataset=dataset,
            context=context
        )

        return X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled
```

##### 2.2.3 Feature Construction from Predictions

The critical method that builds meta-features:

```python
def _build_features_from_predictions(
    self,
    prediction_store: Predictions,
    dataset: SpectroDataset,
    context: ExecutionContext
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct feature matrix from previous model predictions.

    For each source model:
    - Collect OOF (out-of-fold) predictions for training samples
    - Collect test predictions for test samples
    - Stack horizontally to form meta-features

    Returns:
        Tuple of (X_train_meta, X_test_meta)
    """
    folds = dataset.folds
    n_train = len(dataset.get_indices('train'))
    n_test = len(dataset.get_indices('test'))

    # Identify source models
    source_models = self._get_source_models(prediction_store)

    # Initialize feature arrays
    X_train_meta = np.zeros((n_train, len(source_models)))
    X_test_meta = np.zeros((n_test, len(source_models)))

    for col_idx, model_name in enumerate(source_models):
        # Get OOF predictions for training (from val partitions across folds)
        train_preds = self._collect_oof_predictions(
            prediction_store, model_name, folds, n_train
        )
        X_train_meta[:, col_idx] = train_preds

        # Get test predictions (averaged across folds if multiple)
        test_preds = self._collect_test_predictions(
            prediction_store, model_name, n_test
        )
        X_test_meta[:, col_idx] = test_preds

    # Optionally include original features
    if self.operator.include_original_features:
        X_orig_train = dataset.x(context.with_partition('train').selector)
        X_orig_test = dataset.x(context.with_partition('test').selector)
        X_train_meta = np.hstack([X_train_meta, X_orig_train])
        X_test_meta = np.hstack([X_test_meta, X_orig_test])

    return X_train_meta, X_test_meta
```

##### 2.2.4 Out-of-Fold (OOF) Prediction Collection

Proper stacking requires using **out-of-fold predictions** for training to prevent data leakage:

```python
def _collect_oof_predictions(
    self,
    prediction_store: Predictions,
    model_name: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_samples: int
) -> np.ndarray:
    """Collect out-of-fold predictions for a single model.

    For each fold, the validation predictions become the training features
    for those samples, ensuring no leakage.
    """
    oof_preds = np.zeros(n_samples)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Get validation partition predictions for this fold
        val_preds = prediction_store.filter_predictions(
            model_name=model_name,
            partition='val',
            fold_id=fold_idx,
            load_arrays=True
        )

        if val_preds:
            pred = val_preds[0]
            indices = pred.get('sample_indices', val_idx)
            oof_preds[indices] = pred['y_pred']

    return oof_preds
```

##### 2.2.5 Pipeline Syntax

Users would declare meta-models with a clean syntax:

```python
from nirs4all.operators.models import MetaModel
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),

    # Base models
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF", "model": RandomForestRegressor(n_estimators=100)},

    # Meta-model using all previous predictions
    {"name": "Stacked-Ridge", "meta_model": MetaModel(
        model=Ridge(alpha=1.0),
        source_models=None,  # Use all previous models
        include_original_features=False
    )},
]
```

Alternative syntax using explicit source selection:

```python
pipeline = [
    # ...base models...

    # Meta-model using specific models only
    {"name": "Stacked-Ridge", "meta_model": MetaModel(
        model=Ridge(alpha=1.0),
        source_models=["PLS-10", "RF"],  # Explicit selection
    )},
]
```

#### Classification Support

For classification tasks, additional considerations:

```python
class MetaModel:
    def __init__(
        self,
        model,
        source_models=None,
        use_proba: bool = True,  # Use probabilities for classification
        proba_class: int = None,  # Which class probability to use (None = all)
        include_original_features: bool = False
    ):
        ...
```

For binary classification with `use_proba=True`:
- Uses probability of positive class as feature (1 feature per model)

For multiclass with `use_proba=True`:
- Uses all class probabilities (n_classes features per model)

---

## 3. Branching Complexity Analysis

### 3.1 The Branching Challenge

The branching feature (`{"branch": [...]}`) introduces significant complexity for meta-model stacking because:

1. **Independent Contexts**: Each branch operates with its own preprocessing, potentially different X transformations
2. **Different Sample Sizes**: Sample partitioner (`outlier_excluder`, `sample_partitioner`) creates branches with **disjoint sample sets**
3. **Different Splits**: In theory, branches could use different splitters (though not recommended)
4. **Multiplied Predictions**: N branches × M models × K folds = N×M×K prediction entries

### 3.2 Branching Scenarios

#### Scenario A: Preprocessing Branches (Same Samples, Different Features)

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
    }},
    PLSRegression(n_components=10),  # Trained twice
    MetaModel(Ridge()),  # Stack on which branch?
]
```

**Problem**: PLS produces predictions in 2 branches. The meta-model needs to know:
- Stack within each branch separately? (2 meta-models)
- Stack across branches? (1 meta-model using predictions from all branches)

**Design Decision**: Meta-model **inherits branch context** and stacks only within its current branch.
- If declared after branch, it runs on each branch with branch-filtered sources
- If user wants cross-branch stacking, they must exit branching first (future feature)

#### Scenario B: Sample Partitioner Branches (Different Samples)

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "by": "sample_partitioner",
        "filter": {"method": "y_outlier", "threshold": 1.5},
    }},  # Creates "outliers" and "inliers" branches
    PLSRegression(n_components=10),  # Trained on different sample sets
    MetaModel(Ridge()),  # Problem: samples don't align!
]
```

**Problem**: "outliers" branch has N₁ samples, "inliers" has N₂ samples. OOF reconstruction cannot merge them.

**Design Decision**: Meta-model **respects sample partitioning** and stacks within its partition.
- "outliers" meta-model uses only outliers' predictions
- "inliers" meta-model uses only inliers' predictions
- Cross-partition stacking is explicitly forbidden (would cause leakage)

#### Scenario C: Outlier Excluder Branches (Same Samples, Different Training Sets)

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [None, {"method": "isolation_forest"}],
    }},  # "baseline" and "outlier_excluded" branches
    PLSRegression(n_components=10),
    MetaModel(Ridge()),
]
```

**Problem**: Both branches predict on ALL samples, but one trained without outliers.

**Design Decision**: This scenario is **compatible** with stacking since all samples have predictions.
The meta-model in each branch uses predictions from models in that branch only.

### 3.3 Branch Compatibility Matrix

| Branching Type | Same Samples? | Stacking Compatible? | Notes |
|----------------|---------------|----------------------|-------|
| Preprocessing (`{"branch": [...]}`) | ✅ Yes | ✅ Yes | Stack within branch |
| Generator (`{"_or_": [...]}`) | ✅ Yes | ✅ Yes | Stack within branch |
| Outlier Excluder | ✅ Yes | ✅ Yes | All samples have predictions |
| Sample Partitioner | ❌ No | ⚠️ Partial | Stack within partition only |
| Nested Branches | ✅ Depends | ⚠️ Complex | Flatten before stacking |

### 3.4 Branch-Aware Source Selection

The meta-model must filter source predictions by branch context:

```python
def _get_source_models_for_branch(
    self,
    prediction_store: Predictions,
    context: ExecutionContext
) -> List[str]:
    """Get source models filtered by current branch context."""
    branch_id = context.selector.branch_id
    branch_name = context.selector.branch_name

    # Filter predictions by branch
    branch_preds = prediction_store.filter_predictions(
        branch_id=branch_id,
        branch_name=branch_name
    )

    # Extract unique model names from this branch
    model_names = set()
    for pred in branch_preds.to_dicts():
        if pred.get('step_idx', 0) < context.step_idx:
            model_names.add(pred['model_name'])

    return list(model_names)
```

---

## 4. Training Set Reconstruction Logic

### 4.1 Core Principle: Out-of-Fold (OOF) Reconstruction

The **fundamental invariant** for stacking is: **No sample sees predictions from a model trained on that sample**.

For K-fold cross-validation:
- Fold k trains on samples NOT in validation set k
- Validation predictions for fold k are "out-of-fold" for those samples
- OOF reconstruction collects each sample's prediction from the fold where it was in validation

**Visual representation**:

```
Fold 1: Train [20-100], Val [0-19]   → OOF predictions for samples 0-19
Fold 2: Train [0-19,40-100], Val [20-39] → OOF predictions for samples 20-39
Fold 3: Train [0-39,60-100], Val [40-59] → OOF predictions for samples 40-59
Fold 4: Train [0-59,80-100], Val [60-79] → OOF predictions for samples 60-79
Fold 5: Train [0-79], Val [80-100]   → OOF predictions for samples 80-100

Result: Each sample has exactly ONE OOF prediction (from its val fold)
```

### 4.2 Training Set Reconstruction Algorithm

```python
class TrainingSetReconstructor:
    """Reconstructs meta-model training set from OOF predictions."""

    def __init__(
        self,
        prediction_store: Predictions,
        source_models: List[str],
        config: StackingConfig
    ):
        self.prediction_store = prediction_store
        self.source_models = source_models
        self.config = config

    def reconstruct(
        self,
        dataset: SpectroDataset,
        context: ExecutionContext
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct X_train, y_train, and sample_mask for meta-model.

        Returns:
            X_meta_train: (n_valid_samples, n_features) feature matrix
            y_meta_train: (n_valid_samples,) target vector
            valid_mask: (n_total_samples,) boolean mask of valid samples
        """
        folds = dataset.folds
        n_samples = len(dataset.get_indices('train'))
        n_features = len(self.source_models)

        # Initialize with NaN to detect missing predictions
        X_meta = np.full((n_samples, n_features), np.nan)
        sample_coverage = np.zeros(n_samples, dtype=int)

        for col_idx, model_name in enumerate(self.source_models):
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                # Get validation predictions for this fold
                val_preds = self._get_fold_predictions(
                    model_name=model_name,
                    fold_id=fold_idx,
                    partition='val',
                    context=context
                )

                if val_preds is None:
                    self._handle_missing_fold(model_name, fold_idx)
                    continue

                # Place predictions at correct sample positions
                sample_indices = val_preds.get('sample_indices')
                y_pred = val_preds['y_pred'].flatten()

                X_meta[sample_indices, col_idx] = y_pred
                sample_coverage[sample_indices] += 1

        # Validate coverage
        valid_mask = self._compute_valid_mask(sample_coverage, n_features)

        # Apply configured strategy for partial coverage
        X_meta_clean, y_meta_clean = self._apply_coverage_strategy(
            X_meta, dataset.y_train, valid_mask
        )

        return X_meta_clean, y_meta_clean, valid_mask
```

### 4.3 Coverage Validation and Strategies

Different scenarios require different handling strategies:

#### 4.3.1 Full Coverage (Normal Case)

All samples have predictions from all source models in exactly one fold.

```python
# Expected: sample_coverage[i] == n_source_models for all i
valid_mask = (sample_coverage == len(self.source_models))
assert valid_mask.all(), "Full coverage expected"
```

#### 4.3.2 Partial Coverage (Sample Partitioning)

Some samples are missing from certain branches/models.

```python
class CoverageStrategy(Enum):
    STRICT = "strict"          # Raise error if any sample missing
    DROP_INCOMPLETE = "drop"   # Exclude samples with missing predictions
    IMPUTE_ZERO = "impute_zero"  # Fill missing with 0
    IMPUTE_MEAN = "impute_mean"  # Fill missing with mean prediction
    IMPUTE_FOLD_MEAN = "impute_fold_mean"  # Fill with fold's mean
```

**Configuration example**:

```python
MetaModel(
    model=Ridge(),
    stacking_config=StackingConfig(
        coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
        min_coverage_ratio=0.8,  # At least 80% coverage required
        warn_on_partial=True
    )
)
```

### 4.4 Stacking Configuration Class

```python
@dataclass
class StackingConfig:
    """Configuration for meta-model training set reconstruction.

    Attributes:
        coverage_strategy: How to handle samples with missing predictions
        min_coverage_ratio: Minimum fraction of samples required (0.0-1.0)
        require_all_folds: Whether all folds must have predictions
        require_all_models: Whether all source models must have predictions
        warn_on_partial: Log warning when coverage < 100%
        test_aggregation: How to aggregate test predictions across folds
        branch_scope: How to handle branch boundaries
    """
    coverage_strategy: CoverageStrategy = CoverageStrategy.STRICT
    min_coverage_ratio: float = 1.0
    require_all_folds: bool = True
    require_all_models: bool = True
    warn_on_partial: bool = True
    test_aggregation: TestAggregation = TestAggregation.MEAN
    branch_scope: BranchScope = BranchScope.CURRENT_ONLY


class TestAggregation(Enum):
    """How to aggregate test predictions from multiple folds."""
    MEAN = "mean"              # Simple average
    WEIGHTED_MEAN = "weighted"  # Weighted by fold validation score
    MEDIAN = "median"          # Median prediction
    BEST_FOLD = "best"         # Use prediction from best-scoring fold


class BranchScope(Enum):
    """Which branches to include as source models."""
    CURRENT_ONLY = "current"   # Only models from current branch
    ALL_BRANCHES = "all"       # Models from all branches (future)
    SPECIFIED = "specified"    # Explicit list via source_models
```

### 4.5 Fold Alignment Validation

Critical safety checks before reconstruction:

```python
def _validate_fold_alignment(self) -> ValidationResult:
    """Validate that fold structure is consistent across source models.

    Checks:
    1. All models have same number of folds
    2. Fold indices match (0, 1, 2, ... K-1)
    3. Sample indices within folds are consistent
    4. No overlapping validation sets (would indicate data leak)
    """
    results = ValidationResult()

    # Check 1: Same fold count
    fold_counts = {}
    for model in self.source_models:
        folds = self._get_model_folds(model)
        fold_counts[model] = len(folds)

    if len(set(fold_counts.values())) > 1:
        results.add_error(
            "FOLD_COUNT_MISMATCH",
            f"Models have different fold counts: {fold_counts}"
        )

    # Check 2: Fold indices sequential
    for model in self.source_models:
        fold_ids = sorted(self._get_model_folds(model))
        expected = list(range(len(fold_ids)))
        if fold_ids != expected:
            results.add_error(
                "FOLD_INDEX_GAP",
                f"Model {model} has non-sequential folds: {fold_ids}"
            )

    # Check 3: No sample appears in multiple validation sets
    all_val_indices = []
    for model in self.source_models:
        for fold_id in self._get_model_folds(model):
            val_preds = self._get_fold_predictions(model, fold_id, 'val')
            if val_preds:
                all_val_indices.extend(val_preds['sample_indices'])

    duplicates = self._find_duplicates(all_val_indices)
    if duplicates:
        results.add_warning(
            "DUPLICATE_VAL_SAMPLES",
            f"Samples appear in multiple validation sets: {duplicates[:10]}..."
        )

    return results
```

### 4.6 Handling Edge Cases

#### Edge Case 1: Different Splits per Branch

```python
# INVALID PIPELINE - should raise error
pipeline = [
    {"branch": {
        "5fold": [KFold(n_splits=5), PLSRegression()],
        "3fold": [KFold(n_splits=3), RandomForestRegressor()],  # Different!
    }},
    MetaModel(Ridge()),  # Cannot align folds!
]
```

**Solution**: Validate at meta-model step that all source models share the same fold structure.

#### Edge Case 2: Sample Partitioner with Stacking

```python
pipeline = [
    ShuffleSplit(n_splits=5),
    {"branch": {
        "by": "sample_partitioner",
        "filter": {"method": "y_outlier"},
    }},
    PLSRegression(n_components=10),
    MetaModel(Ridge()),  # Stacks within partition
]
```

**Behavior**:
- "outliers" branch: Meta-model trains on outlier samples' OOF predictions
- "inliers" branch: Meta-model trains on inlier samples' OOF predictions
- Sample indices are different per branch, reconstruction respects this

#### Edge Case 3: No Folds (Single Split)

```python
pipeline = [
    ShuffleSplit(n_splits=1, test_size=0.2),  # No CV, just train/test
    PLSRegression(n_components=10),
    MetaModel(Ridge()),  # No OOF possible!
]
```

**Behavior**:
- Warning: "Stacking without cross-validation risks overfitting"
- Use train partition predictions (with leakage warning)
- Or: require `config.allow_no_cv=True` to proceed

---

## 5. Source Model Selection Strategies

### 5.1 Abstract Base Class

To support flexible and extensible model selection, we introduce an abstract strategy pattern:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelCandidate:
    """Information about a candidate source model."""
    model_name: str
    model_classname: str
    step_idx: int
    fold_id: Optional[int]
    branch_id: Optional[int]
    branch_name: Optional[str]
    val_score: float
    test_score: float
    metric: str
    n_samples: int
    predictions: Optional[Dict[str, np.ndarray]] = None


class SourceModelSelector(ABC):
    """Abstract base class for source model selection strategies.

    Subclass this to implement custom selection logic for choosing
    which models to include in the meta-model stack.

    Example usage:
        >>> selector = TopKByMetricSelector(k=3, metric='rmse', lower_is_better=True)
        >>> selected = selector.select(candidates, context)
        >>> meta_model = MetaModel(Ridge(), source_selector=selector)
    """

    @abstractmethod
    def select(
        self,
        candidates: List[ModelCandidate],
        context: 'ExecutionContext'
    ) -> List[ModelCandidate]:
        """Select source models from available candidates.

        Args:
            candidates: All available model candidates from prediction_store
            context: Current execution context for branch/step info

        Returns:
            Filtered list of selected candidates
        """
        pass

    @abstractmethod
    def validate(
        self,
        selected: List[ModelCandidate],
        config: StackingConfig
    ) -> ValidationResult:
        """Validate that selection meets requirements.

        Args:
            selected: Models selected by this strategy
            config: Stacking configuration

        Returns:
            Validation result with errors/warnings
        """
        pass

    def get_config_dict(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "params": self._get_params()
        }

    @abstractmethod
    def _get_params(self) -> Dict[str, Any]:
        """Get parameters for serialization."""
        pass
```

### 5.2 Built-in Selection Strategies

#### AllPreviousModelsSelector (Default)

```python
class AllPreviousModelsSelector(SourceModelSelector):
    """Select all models from previous steps in current branch."""

    def __init__(self, same_branch_only: bool = True):
        self.same_branch_only = same_branch_only

    def select(
        self,
        candidates: List[ModelCandidate],
        context: ExecutionContext
    ) -> List[ModelCandidate]:
        selected = []
        current_step = context.step_idx
        current_branch = context.selector.branch_id

        for c in candidates:
            # Only include models from earlier steps
            if c.step_idx >= current_step:
                continue

            # Branch filtering
            if self.same_branch_only and c.branch_id != current_branch:
                continue

            selected.append(c)

        return selected

    def validate(self, selected, config):
        result = ValidationResult()
        if not selected:
            result.add_error("NO_SOURCE_MODELS", "No source models found for stacking")
        return result

    def _get_params(self):
        return {"same_branch_only": self.same_branch_only}
```

#### ExplicitModelSelector

```python
class ExplicitModelSelector(SourceModelSelector):
    """Select explicitly named models."""

    def __init__(self, model_names: List[str]):
        self.model_names = model_names

    def select(
        self,
        candidates: List[ModelCandidate],
        context: ExecutionContext
    ) -> List[ModelCandidate]:
        return [c for c in candidates if c.model_name in self.model_names]

    def validate(self, selected, config):
        result = ValidationResult()
        found_names = {c.model_name for c in selected}
        missing = set(self.model_names) - found_names
        if missing:
            result.add_error(
                "MISSING_SOURCE_MODELS",
                f"Specified models not found: {missing}"
            )
        return result

    def _get_params(self):
        return {"model_names": self.model_names}
```

#### TopKByMetricSelector

```python
class TopKByMetricSelector(SourceModelSelector):
    """Select top K models by a validation metric."""

    def __init__(
        self,
        k: int,
        metric: str = 'rmse',
        partition: str = 'val',
        lower_is_better: bool = True
    ):
        self.k = k
        self.metric = metric
        self.partition = partition
        self.lower_is_better = lower_is_better

    def select(
        self,
        candidates: List[ModelCandidate],
        context: ExecutionContext
    ) -> List[ModelCandidate]:
        # Filter to relevant candidates (earlier steps, same branch)
        filtered = [c for c in candidates if c.step_idx < context.step_idx]

        # Get score for ranking
        def get_score(c):
            return c.val_score if self.partition == 'val' else c.test_score

        # Sort by metric
        sorted_candidates = sorted(
            filtered,
            key=get_score,
            reverse=not self.lower_is_better
        )

        return sorted_candidates[:self.k]

    def validate(self, selected, config):
        result = ValidationResult()
        if len(selected) < self.k:
            result.add_warning(
                "FEWER_THAN_K_MODELS",
                f"Only {len(selected)} models available, requested {self.k}"
            )
        return result

    def _get_params(self):
        return {
            "k": self.k,
            "metric": self.metric,
            "partition": self.partition,
            "lower_is_better": self.lower_is_better
        }
```

#### DiversitySelector

```python
class DiversitySelector(SourceModelSelector):
    """Select diverse models by class type to maximize ensemble diversity."""

    def __init__(
        self,
        max_per_class: int = 1,
        prefer_metric: str = 'rmse',
        lower_is_better: bool = True
    ):
        self.max_per_class = max_per_class
        self.prefer_metric = prefer_metric
        self.lower_is_better = lower_is_better

    def select(
        self,
        candidates: List[ModelCandidate],
        context: ExecutionContext
    ) -> List[ModelCandidate]:
        # Group by model class
        by_class = {}
        for c in candidates:
            if c.step_idx >= context.step_idx:
                continue
            if c.model_classname not in by_class:
                by_class[c.model_classname] = []
            by_class[c.model_classname].append(c)

        # Take top max_per_class from each class
        selected = []
        for classname, class_candidates in by_class.items():
            sorted_class = sorted(
                class_candidates,
                key=lambda c: c.val_score,
                reverse=not self.lower_is_better
            )
            selected.extend(sorted_class[:self.max_per_class])

        return selected

    def validate(self, selected, config):
        result = ValidationResult()
        if len(selected) < 2:
            result.add_warning(
                "LOW_DIVERSITY",
                "Fewer than 2 model classes available for diversity selection"
            )
        return result

    def _get_params(self):
        return {
            "max_per_class": self.max_per_class,
            "prefer_metric": self.prefer_metric,
            "lower_is_better": self.lower_is_better
        }
```

### 5.3 Selector Factory

```python
class SelectorFactory:
    """Factory for creating source model selectors."""

    _registry: Dict[str, Type[SourceModelSelector]] = {
        "all": AllPreviousModelsSelector,
        "explicit": ExplicitModelSelector,
        "top_k": TopKByMetricSelector,
        "diversity": DiversitySelector,
    }

    @classmethod
    def register(cls, name: str, selector_class: Type[SourceModelSelector]):
        """Register a custom selector class."""
        cls._registry[name] = selector_class

    @classmethod
    def create(cls, config: Union[str, Dict, SourceModelSelector]) -> SourceModelSelector:
        """Create selector from config.

        Args:
            config: Either:
                - String name of built-in selector
                - Dict with 'type' and params
                - Already instantiated selector

        Returns:
            SourceModelSelector instance
        """
        if isinstance(config, SourceModelSelector):
            return config

        if isinstance(config, str):
            return cls._registry[config]()

        if isinstance(config, dict):
            selector_type = config.pop('type', 'all')
            return cls._registry[selector_type](**config)

        raise ValueError(f"Invalid selector config: {config}")
```

### 5.4 Updated MetaModel Operator

```python
class MetaModel:
    """Wrapper for meta-model stacking using pipeline predictions.

    Args:
        model: Any sklearn-compatible model
        source_selector: Strategy for selecting source models
            - None: Use all previous models (default)
            - List[str]: Explicit model names
            - SourceModelSelector: Custom selector instance
        stacking_config: Configuration for training set reconstruction
        use_proba: For classification, use probabilities instead of class predictions
        include_original_features: Whether to include original X features
    """

    def __init__(
        self,
        model,
        source_selector: Optional[Union[List[str], SourceModelSelector]] = None,
        stacking_config: Optional[StackingConfig] = None,
        use_proba: bool = True,
        include_original_features: bool = False
    ):
        self.model = model
        self.stacking_config = stacking_config or StackingConfig()
        self.use_proba = use_proba
        self.include_original_features = include_original_features

        # Convert source_selector to selector object
        if source_selector is None:
            self.source_selector = AllPreviousModelsSelector()
        elif isinstance(source_selector, list):
            self.source_selector = ExplicitModelSelector(source_selector)
        else:
            self.source_selector = source_selector
```

---

## 6. Serialization & Prediction Mode

### 6.1 Overview

Serialization for meta-models is **significantly more complex** than for regular models because:

1. The meta-model depends on **multiple source models** that must also be serialized
2. The **feature column order** must be preserved exactly
3. In prediction mode, the **entire pipeline** must execute up to the meta-model step
4. Branch context must be preserved and matched during reload

### 6.2 Serialization Requirements

When a meta-model is trained, the following must be persisted:

| Component | Description | Storage Location |
|-----------|-------------|------------------|
| Meta-model binary | Trained meta-learner (pickle/joblib) | `{model_id}.pkl` |
| Meta-model config | Source refs, feature mapping, settings | `{model_id}_meta_config.json` |
| Source model refs | Ordered list of source model IDs | In meta_config |
| Stacking config | Coverage strategy, aggregation settings | In meta_config |
| Selector config | Model selection strategy used | In meta_config |
| Branch context | branch_id, branch_name at training time | In meta_config |

### 6.3 MetaModel Artifact Structure

```python
@dataclass
class MetaModelArtifact:
    """Complete artifact for meta-model persistence."""

    # Model identification
    model_id: str
    model_name: str
    model_classname: str

    # Source model references (ORDERED - critical for feature reconstruction)
    source_models: List[SourceModelReference]

    # Configuration
    stacking_config: StackingConfigDict
    selector_config: SelectorConfigDict

    # Feature mapping
    feature_columns: List[str]  # ["PLS-10_pred", "RF_pred", ...]
    n_features: int

    # Task info
    task_type: str  # "regression" or "classification"
    use_proba: bool
    include_original_features: bool
    original_feature_count: Optional[int]  # If include_original_features=True

    # Branch context
    branch_id: Optional[int]
    branch_name: Optional[str]

    # Training metadata
    n_samples_trained: int
    fold_count: int
    training_timestamp: str


@dataclass
class SourceModelReference:
    """Reference to a source model used in stacking."""
    model_name: str
    model_id: str
    step_idx: int
    model_classname: str
    feature_index: int  # Column index in meta-features
```

**Persisted JSON example**:

```json
{
    "meta_model_type": "MetaModel",
    "model_id": "stacked_ridge_20241212_143022_abc123",
    "model_name": "Stacked-Ridge",
    "model_classname": "Ridge",

    "source_models": [
        {
            "model_name": "PLS-10",
            "model_id": "pls10_20241212_143020_def456",
            "step_idx": 3,
            "model_classname": "PLSRegression",
            "feature_index": 0
        },
        {
            "model_name": "RF",
            "model_id": "rf_20241212_143021_ghi789",
            "step_idx": 4,
            "model_classname": "RandomForestRegressor",
            "feature_index": 1
        }
    ],

    "stacking_config": {
        "coverage_strategy": "strict",
        "min_coverage_ratio": 1.0,
        "test_aggregation": "mean"
    },

    "selector_config": {
        "class": "AllPreviousModelsSelector",
        "params": {"same_branch_only": true}
    },

    "feature_columns": ["PLS-10_pred", "RF_pred"],
    "n_features": 2,

    "task_type": "regression",
    "use_proba": false,
    "include_original_features": false,

    "branch_id": 0,
    "branch_name": "snv_branch",

    "n_samples_trained": 80,
    "fold_count": 5,
    "training_timestamp": "2024-12-12T14:30:22Z"
}
```

### 6.4 Prediction Mode Flow

#### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION MODE FLOW                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. Load meta-model artifact (config + binary)                               │
│                                                                               │
│  2. Parse source model references                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐  │
│     │ For each source model in order:                                     │  │
│     │   - Find binary in artifacts                                        │  │
│     │   - Register step dependency                                         │  │
│     └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  3. Execute pipeline up to meta-model step                                    │
│     ┌─────────────────────────────────────────────────────────────────────┐  │
│     │ Step 1: Preprocessing (apply transforms)                            │  │
│     │ Step 2: Source Model A (predict → store in prediction_store)        │  │
│     │ Step 3: Source Model B (predict → store in prediction_store)        │  │
│     │ ...                                                                  │  │
│     └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  4. Construct meta-features from prediction_store                             │
│     ┌─────────────────────────────────────────────────────────────────────┐  │
│     │ X_meta = []                                                         │  │
│     │ for ref in source_models (in feature_index order):                  │  │
│     │     preds = prediction_store.get(ref.model_name, partition='test')  │  │
│     │     X_meta.append(preds['y_pred'])                                  │  │
│     │ X_meta = np.column_stack(X_meta)                                    │  │
│     └─────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  5. Load meta-model binary and predict                                        │
│     y_pred = meta_model.predict(X_meta)                                       │
│                                                                               │
│  6. Store/return predictions                                                  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Key Insight: Automatic Dependency Resolution

The `BinaryLoader` must understand that loading a meta-model artifact implicitly requires loading all source model artifacts:

```python
class BinaryLoader:
    """Enhanced binary loader with meta-model support."""

    def load_meta_model(self, model_id: str) -> Tuple[Any, MetaModelArtifact]:
        """Load meta-model and resolve all dependencies."""

        # Load meta-model config
        config = self._load_meta_config(model_id)
        artifact = MetaModelArtifact.from_dict(config)

        # Ensure all source models are loadable
        for ref in artifact.source_models:
            if not self._model_exists(ref.model_id):
                raise MissingSourceModelError(
                    f"Meta-model {model_id} depends on source model "
                    f"{ref.model_name} ({ref.model_id}) which is not found."
                )

        # Load the actual meta-model binary
        model = self._load_binary(model_id)

        return model, artifact

    def get_required_steps(self, meta_artifact: MetaModelArtifact) -> List[int]:
        """Get all step indices that must execute before meta-model."""
        steps = set()
        for ref in meta_artifact.source_models:
            steps.add(ref.step_idx)
        return sorted(steps)
```

### 6.5 Branch Context Matching

In prediction mode, the branch context must match training:

```python
def _validate_branch_context(
    self,
    meta_artifact: MetaModelArtifact,
    current_context: ExecutionContext
) -> None:
    """Ensure prediction branch matches training branch."""

    expected_branch = meta_artifact.branch_id
    current_branch = current_context.selector.branch_id

    if expected_branch is not None and current_branch != expected_branch:
        raise BranchMismatchError(
            f"Meta-model was trained on branch {expected_branch} "
            f"({meta_artifact.branch_name}), but prediction is running "
            f"on branch {current_branch} ({current_context.selector.branch_name})"
        )
```

### 6.6 Error Handling and Recovery

```python
class MetaModelPredictionError(Exception):
    """Base exception for meta-model prediction errors."""
    pass


class MissingSourceModelError(MetaModelPredictionError):
    """Raised when a source model binary is not found."""
    pass


class SourcePredictionError(MetaModelPredictionError):
    """Raised when a source model fails to produce predictions."""
    pass


class FeatureOrderMismatchError(MetaModelPredictionError):
    """Raised when feature columns don't match expected order."""
    pass


class BranchMismatchError(MetaModelPredictionError):
    """Raised when prediction branch doesn't match training branch."""
    pass


def _build_predict_features_safe(
    self,
    prediction_store: Predictions,
    meta_artifact: MetaModelArtifact
) -> np.ndarray:
    """Build meta-features with comprehensive error handling."""

    features = []
    errors = []

    for ref in meta_artifact.source_models:
        try:
            preds = prediction_store.filter_predictions(
                model_name=ref.model_name,
                partition='test',
                load_arrays=True
            )

            if not preds:
                errors.append(SourcePredictionError(
                    f"No predictions found for source model: {ref.model_name}"
                ))
                continue

            y_pred = preds[0]['y_pred']
            features.append((ref.feature_index, y_pred))

        except Exception as e:
            errors.append(SourcePredictionError(
                f"Failed to get predictions from {ref.model_name}: {e}"
            ))

    # Check for errors
    if errors:
        if self.stacking_config.coverage_strategy == CoverageStrategy.STRICT:
            raise MetaModelPredictionError(
                f"Failed to build meta-features: {errors}"
            )
        else:
            for err in errors:
                print(f"⚠️ {err}")

    # Sort by feature index and stack
    features.sort(key=lambda x: x[0])
    return np.column_stack([f[1] for f in features])
```

---

## 7. Testing Strategy

### 7.1 Testing Philosophy

Given the complexity of meta-model stacking, testing must be **exhaustive and multi-layered**:

1. **Unit tests**: Individual components (selector, reconstructor, validator)
2. **Integration tests**: Full pipeline execution with stacking
3. **Roundtrip tests**: Save → Load → Predict consistency
4. **Edge case tests**: Branching, partial coverage, missing models
5. **Regression tests**: Ensure no performance degradation

### 7.2 Test Categories

#### 7.2.1 Unit Tests

```python
class TestTrainingSetReconstructor:
    """Unit tests for OOF reconstruction logic."""

    def test_basic_oof_reconstruction(self):
        """Test basic 5-fold OOF reconstruction."""
        # Setup mock prediction_store with known predictions
        # Verify each sample gets correct fold's prediction

    def test_sample_indices_alignment(self):
        """Test that sample_indices are correctly used."""
        # Create predictions with non-contiguous indices
        # Verify reconstruction places values correctly

    def test_coverage_strategy_strict(self):
        """Test STRICT coverage raises on missing samples."""
        # Create predictions with missing fold
        # Expect error

    def test_coverage_strategy_drop(self):
        """Test DROP_INCOMPLETE excludes partial samples."""
        # Create predictions with partial coverage
        # Verify dropped samples

    def test_coverage_strategy_impute(self):
        """Test IMPUTE_MEAN fills missing values."""
        # Verify imputation logic


class TestSourceModelSelector:
    """Unit tests for selection strategies."""

    def test_all_previous_models_same_branch(self):
        """Test default selector filters by branch."""

    def test_explicit_selector_missing_model(self):
        """Test explicit selector error on missing model."""

    def test_topk_selector_ranking(self):
        """Test top-k correctly ranks by metric."""

    def test_diversity_selector_class_distribution(self):
        """Test diversity selector picks one per class."""
```

#### 7.2.2 Integration Tests

```python
class TestMetaModelIntegration:
    """Integration tests for full stacking pipelines."""

    def test_basic_stacking_pipeline(self):
        """Test simplest stacking pipeline works end-to-end."""
        pipeline = [
            KFold(n_splits=5),
            PLSRegression(n_components=10),
            RandomForestRegressor(),
            MetaModel(Ridge()),
        ]
        predictions = run_pipeline(pipeline, sample_dataset)

        # Verify meta-model predictions exist
        meta_preds = predictions.filter(model_name__contains="MetaModel")
        assert len(meta_preds) > 0

        # Verify no data leakage (OOF used correctly)
        # Compare meta-model performance to random baseline

    def test_stacking_with_preprocessing_branch(self):
        """Test stacking within preprocessing branches."""
        pipeline = [
            KFold(n_splits=3),
            {"branch": {"snv": [SNV()], "msc": [MSC()]}},
            PLSRegression(n_components=5),
            MetaModel(Ridge()),
        ]
        predictions = run_pipeline(pipeline, sample_dataset)

        # Should have 2 meta-models (one per branch)
        meta_preds = predictions.filter(model_name__contains="MetaModel")
        branches = set(p['branch_name'] for p in meta_preds)
        assert len(branches) == 2

    def test_stacking_with_sample_partitioner(self):
        """Test stacking respects sample partitioning."""
        pipeline = [
            KFold(n_splits=3),
            {"branch": {"by": "sample_partitioner", "filter": {...}}},
            PLSRegression(n_components=5),
            MetaModel(Ridge()),
        ]
        # Each partition should have its own meta-model
        # Sample indices should not overlap
```

#### 7.2.3 Roundtrip Tests

```python
class TestSerializationRoundtrip:
    """Tests for save/load/predict consistency."""

    def test_basic_roundtrip(self):
        """Test predict(save(train())) == original predictions."""
        # Train pipeline
        preds_train, _ = runner.run(pipeline, dataset)
        meta_pred = get_best_meta_prediction(preds_train)
        original_y_pred = meta_pred['y_pred']

        # Save and reload
        model_id = meta_pred['id']

        # Predict on same test data
        preds_reload, _ = runner.predict(model_id, dataset)
        reloaded_y_pred = preds_reload[0]['y_pred']

        # Should be identical
        np.testing.assert_allclose(original_y_pred, reloaded_y_pred, rtol=1e-5)

    def test_roundtrip_different_dataset_size(self):
        """Test predict works with different sample counts."""
        # Train on N samples
        # Predict on M samples (M != N)
        # Should work correctly

    def test_roundtrip_preserves_feature_order(self):
        """Test that feature column order is preserved."""
        # Train with models A, B, C
        # Reload and verify features are [A_pred, B_pred, C_pred]

    def test_roundtrip_branch_matching(self):
        """Test that branch context is validated on reload."""
        # Train on branch 0
        # Attempt reload with branch 1 context
        # Should raise error

    def test_roundtrip_missing_source_model(self):
        """Test error when source model binary is missing."""
        # Train meta-model
        # Delete one source model binary
        # Attempt reload
        # Should raise MissingSourceModelError


class TestDeterminism:
    """Tests for prediction determinism."""

    def test_multiple_predict_calls_identical(self):
        """Test that multiple predict() calls return same result."""
        for _ in range(5):
            result = runner.predict(model_id, dataset)
            results.append(result)

        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)

    def test_predict_after_restart(self):
        """Test that predictions survive process restart."""
        # This would be a subprocess test
```

#### 7.2.4 Edge Case Tests

```python
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_no_source_models(self):
        """Test error when no source models available."""
        pipeline = [
            KFold(n_splits=3),
            MetaModel(Ridge()),  # No models before!
        ]
        with pytest.raises(ValueError, match="No source models"):
            run_pipeline(pipeline, dataset)

    def test_single_fold(self):
        """Test warning for single-fold (no OOF possible)."""
        pipeline = [
            ShuffleSplit(n_splits=1),
            PLSRegression(),
            MetaModel(Ridge()),
        ]
        with pytest.warns(UserWarning, match="without cross-validation"):
            run_pipeline(pipeline, dataset)

    def test_different_fold_counts_error(self):
        """Test error when source models have different fold counts."""
        # Simulate by manually creating mismatched predictions

    def test_include_original_features(self):
        """Test passthrough of original features works."""
        pipeline = [
            KFold(n_splits=3),
            PLSRegression(),
            MetaModel(Ridge(), include_original_features=True),
        ]
        predictions = run_pipeline(pipeline, dataset)
        # Verify feature count = n_source_models + n_original_features
```

### 7.3 Test Data Requirements

```python
# Fixtures for testing
@pytest.fixture
def sample_regression_dataset():
    """Small regression dataset for quick tests."""
    return DatasetConfigs("tests/fixtures/regression_small")

@pytest.fixture
def mock_prediction_store():
    """Pre-populated prediction store for unit tests."""
    store = Predictions()
    # Add mock predictions for 3 models, 5 folds each
    for model in ["PLS", "RF", "XGB"]:
        for fold in range(5):
            store.add_prediction(
                model_name=model,
                fold_id=fold,
                partition="val",
                sample_indices=get_val_indices(fold),
                y_pred=np.random.randn(20),
                y_true=np.random.randn(20),
                ...
            )
    return store
```

### 7.4 Test Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| TrainingSetReconstructor | 95% |
| SourceModelSelector (all implementations) | 90% |
| MetaModelController | 85% |
| Serialization/Deserialization | 95% |
| Prediction Mode Flow | 90% |
| Error Handling | 80% |

---

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Create MetaModel Operator
- [ ] Create `nirs4all/operators/models/meta.py`
- [ ] Define `MetaModel` class with full configuration parameters
- [ ] Define `StackingConfig` dataclass for coverage strategies
- [ ] Export in `nirs4all/operators/models/__init__.py`

#### 1.2 Create MetaModelController
- [ ] Create `nirs4all/controllers/models/meta_model.py`
- [ ] Implement `matches()` for MetaModel detection
- [ ] Override `get_xy()` for prediction-based feature construction
- [ ] Register controller with appropriate priority (before SklearnModelController)
- [ ] Export in `nirs4all/controllers/models/__init__.py`

#### 1.3 Prediction Store Enhancements
- [ ] Add `get_predictions_by_step()` method to Predictions
- [ ] Add `get_oof_predictions()` convenience method
- [ ] Add `filter_by_branch()` for branch-aware querying
- [ ] Ensure sample_indices are consistently stored for all partitions

#### 1.4 Source Model Selection Framework
- [ ] Create `nirs4all/operators/models/selection.py`
- [ ] Implement `SourceModelSelector` abstract base class
- [ ] Implement `AllPreviousModelsSelector` (default)
- [ ] Implement `ExplicitModelSelector`
- [ ] Implement `TopKByMetricSelector`
- [ ] Implement `DiversitySelector`
- [ ] Create `SelectorFactory` for easy instantiation

### Phase 2: Training Set Reconstruction (Week 2-3)

#### 2.1 TrainingSetReconstructor Class
- [ ] Create `nirs4all/controllers/models/stacking/reconstructor.py`
- [ ] Implement core OOF prediction collection logic
- [ ] Implement `_validate_fold_alignment()` method
- [ ] Implement coverage validation and strategies

#### 2.2 Coverage Strategies
- [ ] Implement `CoverageStrategy.STRICT` (default)
- [ ] Implement `CoverageStrategy.DROP_INCOMPLETE`
- [ ] Implement `CoverageStrategy.IMPUTE_ZERO`
- [ ] Implement `CoverageStrategy.IMPUTE_MEAN`
- [ ] Add clear warning/error messages for each strategy

#### 2.3 Test Prediction Aggregation
- [ ] Implement `_collect_test_predictions()` method
- [ ] Implement `TestAggregation.MEAN` (default)
- [ ] Implement `TestAggregation.WEIGHTED_MEAN`
- [ ] Implement `TestAggregation.BEST_FOLD`

#### 2.4 Branch-Aware Reconstruction
- [ ] Implement `BranchScope.CURRENT_ONLY` filtering
- [ ] Handle sample partitioner with disjoint sample sets
- [ ] Validate fold structure consistency within branch
- [ ] Add clear errors for incompatible branching scenarios

### Phase 3: Serialization & Prediction Mode (Week 3-4)

#### 3.1 MetaModel Artifact Persistence
- [ ] Create `MetaModelArtifact` dataclass for serialization
- [ ] Create `SourceModelReference` dataclass
- [ ] Persist meta-model configuration as JSON alongside binary
- [ ] Store source model references with feature column mapping
- [ ] Store stacking config and selector config
- [ ] Integrate with existing `ArtifactManager`

#### 3.2 Prediction Mode Implementation
- [ ] Implement `_execute_predict()` for MetaModelController
- [ ] Load meta-model configuration from artifacts
- [ ] Implement dependency resolution for source models
- [ ] Collect source model predictions from prediction_store
- [ ] Build meta-features with correct column order
- [ ] Validate branch context matches training

#### 3.3 BinaryLoader Enhancements
- [ ] Add `load_meta_model()` method
- [ ] Implement `get_required_steps()` for dependency tracking
- [ ] Add validation for source model availability
- [ ] Implement `MissingSourceModelError` exception

### Phase 4: Branching Integration (Week 4-5)

#### 4.1 Preprocessing Branch Support
- [ ] Test stacking within preprocessing branches
- [ ] Verify independent meta-models per branch
- [ ] Test branch context propagation to artifacts

#### 4.2 Sample Partitioner Support
- [ ] Test stacking with sample partitioner
- [ ] Verify sample indices are correctly partitioned
- [ ] Add validation for non-overlapping sample sets
- [ ] Document limitations (cross-partition stacking not supported)

#### 4.3 Outlier Excluder Support
- [ ] Test stacking with outlier excluder
- [ ] Verify all samples have predictions
- [ ] Test different exclusion strategies with stacking

#### 4.4 Edge Cases
- [ ] Test nested branching with stacking
- [ ] Test generator syntax with stacking
- [ ] Add clear error messages for unsupported scenarios

### Phase 5: Classification Support (Week 5-6)

#### 5.1 Probability Features
- [ ] Detect classification task type from predictions
- [ ] Extract `y_proba` from predictions when available
- [ ] Handle binary classification (single probability column)
- [ ] Handle multiclass (n_classes probability columns)

#### 5.2 Feature Naming
- [ ] Generate meaningful feature names for meta-features
- [ ] Support feature importance tracking in meta-model
- [ ] Include probability class in feature names for classification

### Phase 6: Testing & Documentation (Week 6-7)

#### 6.1 Unit Tests
- [ ] Test `TrainingSetReconstructor` with all coverage strategies
- [ ] Test all `SourceModelSelector` implementations
- [ ] Test fold alignment validation
- [ ] Test coverage edge cases

#### 6.2 Integration Tests
- [ ] Test basic stacking pipeline end-to-end
- [ ] Test stacking with preprocessing branches
- [ ] Test stacking with sample partitioner
- [ ] Test stacking with outlier excluder
- [ ] Test mixed frameworks (sklearn + TF base models)

#### 6.3 Roundtrip Tests
- [ ] Test save/reload/predict consistency
- [ ] Test with different dataset sizes
- [ ] Test feature order preservation
- [ ] Test branch context validation
- [ ] Test missing source model error handling

#### 6.4 Documentation
- [ ] Create `examples/Q_meta_stacking.py` demonstration
- [ ] Add comprehensive docstrings to all new classes
- [ ] Create `docs/user_guide/stacking.md`
- [ ] Update README with stacking examples
- [ ] Add branching + stacking examples

### Phase 7: Advanced Features (Future)

#### 7.1 Multi-Level Stacking
- [ ] Support stacking meta-models on meta-models
- [ ] Implement `level` parameter for layer management
- [ ] Validate no circular dependencies

#### 7.2 Cross-Branch Stacking
- [ ] Implement `BranchScope.ALL_BRANCHES`
- [ ] Handle feature alignment across branches
- [ ] Add proper validation for compatible branches

#### 7.3 Finetune Integration
- [ ] Support Optuna finetuning of meta-model hyperparameters
- [ ] Support automatic source model selection via optimization
- [ ] Support stacking architecture search

---

## 9. File Structure

```
nirs4all/
├── operators/
│   └── models/
│       ├── __init__.py          # Add MetaModel, StackingConfig exports
│       ├── meta.py              # NEW: MetaModel operator
│       └── selection.py         # NEW: Source model selectors
├── controllers/
│   └── models/
│       ├── __init__.py          # Add MetaModelController export
│       ├── meta_model.py        # NEW: MetaModelController
│       └── stacking/            # NEW: Stacking subpackage
│           ├── __init__.py
│           ├── reconstructor.py # TrainingSetReconstructor
│           ├── config.py        # StackingConfig, enums
│           └── artifacts.py     # MetaModelArtifact, SourceModelReference
├── data/
│   └── predictions.py           # Add get_oof_predictions(), filter_by_branch()
└── ...

examples/
├── Q_meta_stacking.py           # NEW: Basic stacking example
├── Q_meta_stacking_branch.py    # NEW: Stacking with branches
└── Q_meta_stacking_advanced.py  # NEW: Advanced stacking scenarios

tests/
├── unit/
│   └── controllers/
│       └── models/
│           └── stacking/        # NEW: Stacking unit tests
│               ├── test_reconstructor.py
│               ├── test_selectors.py
│               └── test_config.py
└── integration/
    └── test_meta_stacking.py    # NEW: Integration tests

docs/
└── user_guide/
    └── stacking.md              # NEW: User documentation
```

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Fold misalignment** | Medium | High | Rigorous validation checks, clear error messages, automated tests |
| **Data leakage from incorrect OOF** | Low | Critical | Comprehensive unit tests, visual verification, documentation |
| **Branch context mismatch** | Medium | High | Explicit branch validation in serialization, clear errors |
| **Sample partitioner incompatibility** | Medium | Medium | Clear documentation, explicit `BranchScope` configuration |
| **Memory usage (large prediction arrays)** | Medium | Medium | Lazy loading via ArrayRegistry, chunked processing |
| **Serialization format changes** | Low | Medium | Version field in artifacts, migration utilities |
| **Framework incompatibility** | Low | Low | Abstract interfaces, duck typing, integration tests per framework |
| **Performance regression** | Low | Medium | Benchmarking suite, profiling before merge |
| **Complex debugging** | Medium | Low | Detailed logging, prediction provenance tracking |

---

## 11. Success Metrics

### Functional Metrics
1. **Correct OOF Reconstruction**: 100% of samples get prediction from correct fold
2. **No Data Leakage**: Statistical tests confirm meta-model doesn't overfit
3. **Branch Compatibility**: All branching scenarios work as documented

### Quality Metrics
4. **Test Coverage**: >90% coverage on stacking components
5. **Roundtrip Consistency**: Save/load/predict produces identical results
6. **Documentation Completeness**: All public APIs documented with examples

### Usability Metrics
7. **Clean API**: Minimal configuration needed for basic stacking
8. **Clear Errors**: All error messages actionable and informative
9. **Example Coverage**: Working examples for all major scenarios

### Performance Metrics
10. **Negligible Overhead**: <5% execution time increase vs. base models only
11. **Memory Efficiency**: No OOM on datasets up to 10,000 samples

---

## 12. References

- [Stacking (Wikipedia)](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)
- [sklearn StackingClassifier/Regressor](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
- [Kaggle Stacking Guide](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
- [Wolpert (1992) - Stacked Generalization](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
- nirs4all Roadmap: `> [Stacking] Use prediction as X for stacking.`
- nirs4all Branching Documentation: `docs/reference/branching.md`

---

## 13. Example Implementation: Q_meta_stacking.py

This example demonstrates the complete workflow including training, saving, reloading, and prediction.

```python
"""
Q_meta_stacking.py - Meta-Model Stacking Example
=================================================
Demonstrates pipeline-level stacking where a meta-learner uses predictions
from previous models as input features.

This example covers:
- Training base models (PLS, RF, XGBoost)
- Training a meta-model using base model predictions
- Saving and reloading the stacked pipeline
- Prediction on new data using the reloaded stack
- Verifying prediction roundtrip consistency

Key concept: The meta-model trains on OUT-OF-FOLD predictions to prevent leakage.
"""

# Standard library imports
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models import MetaModel
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.utils.emoji import CHECK, CROSS, ROCKET, MEDAL_GOLD


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Meta-Model Stacking Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()


print("\n" + "=" * 80)
print(f"{ROCKET} META-MODEL STACKING EXAMPLE")
print("=" * 80 + "\n")


# =============================================================================
# PART 1: TRAINING THE STACKED PIPELINE
# =============================================================================
print("\n" + "-" * 80)
print("PART 1: Training Base Models + Meta-Model")
print("-" * 80 + "\n")

# Define the stacking pipeline
stacking_pipeline = [
    # Preprocessing
    MinMaxScaler(),
    {"y_processing": MinMaxScaler()},
    {"feature_augmentation": [FirstDerivative, StandardNormalVariate]},

    # Cross-validation splitter
    KFold(n_splits=5, shuffle=True, random_state=42),

    # =========================================================================
    # BASE MODELS - These generate predictions stored in prediction_store
    # =========================================================================
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF", "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)},
    {"name": "GBR", "model": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)},

    # =========================================================================
    # META-MODEL - Uses OOF predictions from base models as features
    # =========================================================================
    {"name": "Stacked-Ridge", "meta_model": MetaModel(
        model=Ridge(alpha=1.0),
        source_models=None,  # Use ALL previous models
        use_proba=False,     # Regression task
        include_original_features=False  # Only use predictions as features
    )},
]

# Configure pipeline and dataset
pipeline_config = PipelineConfigs(stacking_pipeline, "meta_stacking")
dataset_config = DatasetConfigs('sample_data/regression')

# Run training
runner = PipelineRunner(save_files=True, verbose=1, plots_visible=args.plots)
predictions, per_dataset = runner.run(pipeline_config, dataset_config)


# =============================================================================
# PART 2: ANALYZE RESULTS
# =============================================================================
print("\n" + "-" * 80)
print("PART 2: Analyzing Stacking Results")
print("-" * 80 + "\n")

# Display top models
print("--- Top 5 Models by RMSE ---")
top_models = predictions.top(5, 'rmse')
for idx, pred in enumerate(top_models):
    is_meta = "META" if "Stacked" in pred.get('model_name', '') else "BASE"
    print(f"{idx+1}. [{is_meta}] {Predictions.pred_short_string(pred, metrics=['rmse', 'r2'])}")

# Check if meta-model improved over base models
meta_preds = [p for p in predictions.to_dicts() if 'Stacked' in p.get('model_name', '')]
base_preds = [p for p in predictions.to_dicts() if 'Stacked' not in p.get('model_name', '')]

if meta_preds:
    best_meta = min(meta_preds, key=lambda p: p.get('test_rmse', float('inf')))
    best_base = min(base_preds, key=lambda p: p.get('test_rmse', float('inf')))

    meta_rmse = best_meta.get('test_rmse', 0)
    base_rmse = best_base.get('test_rmse', 0)

    improvement = (base_rmse - meta_rmse) / base_rmse * 100

    print(f"\n{MEDAL_GOLD} Best Meta-Model RMSE: {meta_rmse:.4f}")
    print(f"   Best Base Model RMSE: {base_rmse:.4f}")
    print(f"   Improvement: {improvement:.2f}%")


# =============================================================================
# PART 3: SAVE & RELOAD ROUNDTRIP TEST
# =============================================================================
print("\n" + "-" * 80)
print("PART 3: Save/Reload/Predict Roundtrip Test")
print("-" * 80 + "\n")

# Find the best stacked model prediction entry
stacked_predictions = [p for p in predictions.to_dicts() if 'Stacked' in p['model_name']]

if stacked_predictions:
    # Get best stacking prediction by validation score
    best_stacked = min(stacked_predictions, key=lambda p: p.get('val_rmse', float('inf')))
    model_id = best_stacked['id']
    model_name = best_stacked['model_name']
    fold_id = best_stacked['fold_id']

    print(f"Source model: {model_name}")
    print(f"  - ID: {model_id}")
    print(f"  - Fold: {fold_id}")
    print(f"  - Val RMSE: {best_stacked.get('val_rmse', 'N/A')}")
    print(f"  - Test RMSE: {best_stacked.get('test_rmse', 'N/A')}")

    # =========================================================================
    # METHOD 1: Predict using prediction entry (dict)
    # =========================================================================
    print("\n--- Method 1: Predict with prediction entry ---")

    predictor1 = PipelineRunner(save_files=False, verbose=0)
    prediction_dataset = DatasetConfigs({'X_test': 'sample_data/regression/Xval.csv.gz'})

    method1_predictions, method1_preds = predictor1.predict(
        best_stacked,  # Pass the full prediction dict
        prediction_dataset,
        verbose=0
    )

    print(f"Method 1 predictions shape: {method1_predictions.shape}")
    print(f"Method 1 predictions (first 5): {method1_predictions[:5].flatten()}")

    # =========================================================================
    # METHOD 2: Predict using model ID only
    # =========================================================================
    print("\n--- Method 2: Predict with model ID ---")

    predictor2 = PipelineRunner(save_files=False, verbose=0)

    method2_predictions, method2_preds = predictor2.predict(
        model_id,  # Pass just the ID string
        prediction_dataset,
        verbose=0
    )

    print(f"Method 2 predictions shape: {method2_predictions.shape}")
    print(f"Method 2 predictions (first 5): {method2_predictions[:5].flatten()}")

    # =========================================================================
    # VERIFY PREDICTIONS MATCH
    # =========================================================================
    print("\n--- Verification ---")

    is_identical = np.allclose(method1_predictions, method2_predictions, rtol=1e-5)

    if is_identical:
        print(f"{CHECK} Method 1 and Method 2 predictions are IDENTICAL")
    else:
        print(f"{CROSS} Method 1 and Method 2 predictions DIFFER")
        max_diff = np.max(np.abs(method1_predictions - method2_predictions))
        print(f"   Max difference: {max_diff}")

    assert is_identical, "Prediction roundtrip failed: methods produce different results!"

    # =========================================================================
    # VERIFY AGAINST ORIGINAL TEST PREDICTIONS
    # =========================================================================
    print("\n--- Compare to Original Training Run ---")

    # Get original test predictions from training run
    original_test = predictions.filter_predictions(
        id=model_id,
        partition='test',
        load_arrays=True
    )

    if original_test:
        original_y_pred = original_test[0].get('y_pred', np.array([]))

        # Note: Predictions might differ slightly due to:
        # - Different samples in prediction dataset vs training test set
        # - Floating point precision
        # Here we just verify the model produces valid outputs

        print(f"Original test predictions (first 5): {original_y_pred[:5]}")
        print(f"Reload predictions (first 5): {method1_predictions[:5].flatten()}")
        print(f"{CHECK} Meta-model successfully reloaded and produced predictions")

    print(f"\n{CHECK} STACKED MODEL SAVE/RELOAD/PREDICT ROUNDTRIP PASSED!")

else:
    print(f"{CROSS} No stacked model predictions found to test.")


# =============================================================================
# PART 4: ADVANCED - VERIFY META-FEATURES RECONSTRUCTION
# =============================================================================
print("\n" + "-" * 80)
print("PART 4: Verify Meta-Features Reconstruction (Debug)")
print("-" * 80 + "\n")

# This section demonstrates that in prediction mode, the meta-model correctly:
# 1. Runs all source models on new data
# 2. Collects their predictions
# 3. Constructs meta-features
# 4. Runs the meta-learner

# Check that all base models also produced predictions during the predict call
if method1_preds:
    print("Models that produced predictions during predict():")
    for pred in method1_preds.to_dicts():
        print(f"  - {pred.get('model_name', 'unknown')}: "
              f"{pred.get('n_samples', 0)} samples")

    # Verify base models were run (required for meta-model)
    base_model_names = ["PLS-10", "RF", "GBR"]
    predict_models = [p.get('model_name', '') for p in method1_preds.to_dicts()]

    all_base_ran = all(any(bm in pm for pm in predict_models) for bm in base_model_names)

    if all_base_ran:
        print(f"\n{CHECK} All base models ran during prediction (required for meta-features)")
    else:
        print(f"\n{CROSS} Some base models missing from prediction run")


# =============================================================================
# VISUALIZATION (optional)
# =============================================================================
if args.plots or args.show:
    analyzer = PredictionAnalyzer(predictions)

    # Plot top models
    fig_topk = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='test')

    # Candlestick comparison
    fig_candle = analyzer.plot_candlestick(variable="model_name", partition="test")

if args.show:
    plt.show()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("META-MODEL STACKING EXAMPLE COMPLETE")
print("=" * 80)
print("""
Key takeaways:
1. MetaModel wrapper signals pipeline-level stacking intent
2. Base models train normally, storing predictions in prediction_store
3. Meta-model uses OUT-OF-FOLD predictions for training (no leakage)
4. Serialization preserves both meta-model and source model references
5. Prediction mode runs source models first, then meta-model
6. Save/reload roundtrip produces identical predictions
""")
```

---

## 14. Example: Stacking with Branches

This example demonstrates stacking within preprocessing branches.

```python
"""
Q_meta_stacking_branch.py - Stacking with Branches Example
===========================================================
Demonstrates stacking within preprocessing branches.
Each branch gets its own meta-model trained on predictions from that branch only.
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.operators.models import MetaModel
from nirs4all.operators.models.selection import TopKByMetricSelector
from nirs4all.operators.models.config import StackingConfig, CoverageStrategy
from nirs4all.operators.transforms import SNV, MSC, FirstDerivative
from nirs4all.pipeline import PipelineConfigs, PipelineRunner


# =============================================================================
# EXAMPLE 1: Preprocessing Branches with Stacking
# =============================================================================
print("Example 1: Preprocessing branches with per-branch stacking")

pipeline_preprocessing_branches = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Create preprocessing branches
    {"branch": {
        "snv": [SNV()],
        "msc": [MSC()],
        "derivative": [FirstDerivative()],
    }},

    # Base models run on EACH branch
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "RF", "model": RandomForestRegressor(n_estimators=50, random_state=42)},

    # Meta-model runs on EACH branch, stacking that branch's models
    {"name": "Stacked-Ridge", "meta_model": MetaModel(
        model=Ridge(alpha=1.0),
        # Default: uses all previous models in current branch only
    )},
]

runner = PipelineRunner(workspace_path="workspace")
predictions, _ = runner.run(
    PipelineConfigs(pipeline_preprocessing_branches),
    DatasetConfigs('sample_data/regression')
)

# Check: should have 3 meta-models (one per branch)
meta_preds = predictions.filter_predictions(model_name__contains="Stacked")
branches = set(p.get('branch_name', '') for p in meta_preds.to_dicts())
print(f"Meta-models per branch: {branches}")
assert len(branches) == 3, "Expected one meta-model per preprocessing branch"


# =============================================================================
# EXAMPLE 2: Advanced Stacking Configuration
# =============================================================================
print("\nExample 2: Advanced stacking with TopK selector and coverage config")

pipeline_advanced = [
    MinMaxScaler(),
    KFold(n_splits=5, shuffle=True, random_state=42),

    # Many base models
    {"name": "PLS-5", "model": PLSRegression(n_components=5)},
    {"name": "PLS-10", "model": PLSRegression(n_components=10)},
    {"name": "PLS-15", "model": PLSRegression(n_components=15)},
    {"name": "RF-50", "model": RandomForestRegressor(n_estimators=50, random_state=42)},
    {"name": "RF-100", "model": RandomForestRegressor(n_estimators=100, random_state=42)},

    # Meta-model with advanced configuration
    {"name": "Stacked-Best3", "meta_model": MetaModel(
        model=Ridge(alpha=1.0),
        source_selector=TopKByMetricSelector(
            k=3,
            metric='rmse',
            partition='val',
            lower_is_better=True
        ),
        stacking_config=StackingConfig(
            coverage_strategy=CoverageStrategy.STRICT,
            require_all_folds=True,
            warn_on_partial=True,
        ),
    )},
]

predictions, _ = runner.run(
    PipelineConfigs(pipeline_advanced),
    DatasetConfigs('sample_data/regression')
)

print("Advanced stacking completed successfully")


# =============================================================================
# EXAMPLE 3: Stacking with Outlier Excluder (Compatible Scenario)
# =============================================================================
print("\nExample 3: Stacking with outlier excluder branches")

pipeline_outlier = [
    KFold(n_splits=3, shuffle=True, random_state=42),
    MinMaxScaler(),

    # Outlier excluder: all samples get predictions, only training differs
    {"branch": {
        "by": "outlier_excluder",
        "strategies": [
            None,  # Baseline - no exclusion
            {"method": "isolation_forest", "contamination": 0.05},
        ],
    }},

    {"name": "PLS-10", "model": PLSRegression(n_components=10)},

    # Works because all samples have predictions in both branches
    {"name": "Stacked-Ridge", "meta_model": MetaModel(model=Ridge())},
]

predictions, _ = runner.run(
    PipelineConfigs(pipeline_outlier),
    DatasetConfigs('sample_data/regression')
)

# Should have 2 meta-models
meta_preds = predictions.filter_predictions(model_name__contains="Stacked")
assert len(meta_preds) >= 2, "Expected meta-models for each outlier strategy"

print("Outlier excluder stacking completed successfully")
```

---

## 15. Serialization Test Checklist

When implementing the feature, ensure these test cases pass:

### Training Phase Tests
- [ ] Base models store predictions with correct `sample_indices` and `fold_id`
- [ ] Meta-model receives correct OOF predictions (no leakage)
- [ ] Meta-model artifact includes source model references with feature order
- [ ] All binaries (base + meta) are persisted correctly
- [ ] Stacking config and selector config are serialized
- [ ] Branch context (branch_id, branch_name) is persisted

### Reload Phase Tests
- [ ] `BinaryLoader` correctly loads meta-model configuration
- [ ] `BinaryLoader` resolves all source model dependencies
- [ ] Error raised if any source model binary is missing
- [ ] Pipeline step order matches original training
- [ ] Branch context is validated on reload

### Prediction Phase Tests
- [ ] Source models execute and store predictions in prediction_store
- [ ] Meta-model collects source predictions in correct order
- [ ] Meta-features are constructed in correct column order
- [ ] Final predictions match expected values
- [ ] Different dataset sizes work correctly

### Roundtrip Consistency Tests
- [ ] `predict(model_id, X)` == `predict(prediction_entry, X)`
- [ ] Predictions are deterministic (same input → same output)
- [ ] Works with different dataset sizes than training
- [ ] Works across process restarts (persistence test)

### Branch-Specific Tests
- [ ] Preprocessing branches: one meta-model per branch
- [ ] Outlier excluder: all samples have predictions
- [ ] Sample partitioner: meta-model respects partition boundaries
- [ ] Nested branches: correct branch context propagation

---

## 16. Appendix: Visual Diagrams

### A. OOF Reconstruction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OOF RECONSTRUCTION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRAINING: 5-Fold Cross-Validation                                          │
│  ════════════════════════════════════                                        │
│                                                                              │
│  Fold 0: Train [████████████████████] Val [░░░░░]  →  OOF for samples 0-19  │
│  Fold 1: Train [░░░░░████████████████] Val [░░░░░]  →  OOF for samples 20-39│
│  Fold 2: Train [░░░░░░░░░░████████████] Val [░░░░░]  →  OOF for samples 40-59│
│  Fold 3: Train [░░░░░░░░░░░░░░░████████] Val [░░░░░]  →  OOF for samples 60-79│
│  Fold 4: Train [░░░░░░░░░░░░░░░░░░░░████] Val [░░░░░]  →  OOF for samples 80-99│
│                                                                              │
│  META-MODEL TRAINING SET                                                     │
│  ════════════════════════                                                    │
│                                                                              │
│  Sample 0:  Prediction from Fold 0 val (model never saw this sample)        │
│  Sample 20: Prediction from Fold 1 val (model never saw this sample)        │
│  Sample 40: Prediction from Fold 2 val (model never saw this sample)        │
│  ...                                                                         │
│                                                                              │
│  → NO DATA LEAKAGE: Each sample's prediction comes from a model              │
│    that was trained WITHOUT that sample                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B. Branching with Stacking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BRANCHING WITH STACKING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Pipeline:                                                                   │
│  ─────────                                                                   │
│                                                                              │
│  Step 1: KFold(n_splits=5)                                                  │
│  Step 2: MinMaxScaler()                                                     │
│  Step 3: {"branch": {"snv": [SNV()], "msc": [MSC()]}}                       │
│  Step 4: PLSRegression()                                                    │
│  Step 5: RandomForestRegressor()                                            │
│  Step 6: MetaModel(Ridge())                                                 │
│                                                                              │
│  Execution:                                                                  │
│  ──────────                                                                  │
│                                                                              │
│  ┌─────────────────────────────────────┐                                    │
│  │ Shared: KFold, MinMaxScaler        │                                    │
│  └──────────────┬──────────────────────┘                                    │
│                 │                                                            │
│         ┌───────┴───────┐                                                   │
│         ▼               ▼                                                   │
│  ┌─────────────┐ ┌─────────────┐                                           │
│  │ Branch: snv │ │ Branch: msc │                                           │
│  │  SNV()      │ │  MSC()      │                                           │
│  │  PLS → pred │ │  PLS → pred │                                           │
│  │  RF → pred  │ │  RF → pred  │                                           │
│  │  META(snv)  │ │  META(msc)  │  ← Each branch has its own meta-model     │
│  └─────────────┘ └─────────────┘                                           │
│                                                                              │
│  Result: 2 meta-models, each trained on its branch's predictions only       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### C. Prediction Mode Dependency Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREDICTION MODE DEPENDENCY CHAIN                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  To predict with meta-model, the entire dependency chain must execute:       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  1. Load meta-model artifact (includes source model references)        │ │
│  │                               │                                        │ │
│  │                               ▼                                        │ │
│  │  2. Resolve dependencies: [PLS-10, RF, GBR]                            │ │
│  │                               │                                        │ │
│  │                               ▼                                        │ │
│  │  3. Execute pipeline from start:                                       │ │
│  │     ┌─────────────────────────────────────────────────────────────┐   │ │
│  │     │ Step 1: Load preprocessing → Apply to new X                 │   │ │
│  │     │ Step 2: Load PLS-10 → Predict → Store in prediction_store  │   │ │
│  │     │ Step 3: Load RF → Predict → Store in prediction_store      │   │ │
│  │     │ Step 4: Load GBR → Predict → Store in prediction_store     │   │ │
│  │     │ Step 5: Load MetaModel → Collect predictions → Predict     │   │ │
│  │     └─────────────────────────────────────────────────────────────┘   │ │
│  │                               │                                        │ │
│  │                               ▼                                        │ │
│  │  4. Return meta-model predictions                                      │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Key insight: prediction_store acts as inter-step communication channel     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

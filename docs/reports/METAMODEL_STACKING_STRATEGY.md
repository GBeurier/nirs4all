# Meta-Model Stacking Strategy

## Overview

This document outlines the strategy for implementing **meta-model stacking** in nirs4all, enabling users to train models that use predictions from previous pipeline models as input features. This feature allows building sophisticated ensemble architectures where a meta-learner combines outputs from multiple base models.

---

## 1. State of the Art (État des Lieux)

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

---

## 2. Proposed Logic (Logique Proposée)

### Core Concept: MetaModelController

Introduce a new controller type that:
1. Uses `y` from the dataset (unchanged)
2. Constructs `X` from predictions of previous models in the current pipeline
3. Respects the fold structure for proper out-of-fold stacking
4. Works with any sklearn-compatible meta-learner

### Architecture Design

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

### Key Design Decisions

#### 2.1 Meta-Model Operator

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

#### 2.2 MetaModelController

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

#### 2.3 Feature Construction from Predictions

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

#### 2.4 Out-of-Fold (OOF) Prediction Collection

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

#### 2.5 Pipeline Syntax

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

### Classification Support

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

### Fold Alignment

The meta-model must respect the same fold structure as base models:

```python
def _validate_fold_alignment(
    self,
    prediction_store: Predictions,
    source_models: List[str],
    expected_folds: List[Tuple]
) -> bool:
    """Ensure all source models have predictions for all folds."""
    for model_name in source_models:
        model_folds = prediction_store.get_unique_values(
            column='fold_id',
            model_name=model_name
        )
        if len(model_folds) != len(expected_folds):
            raise ValueError(
                f"Model {model_name} has {len(model_folds)} folds, "
                f"expected {len(expected_folds)}"
            )
    return True
```

### 2.6 Serialization & Prediction Mode

A critical consideration is how meta-models are saved and reloaded for prediction on new data.

#### Serialization Requirements

When a meta-model is trained, the following must be persisted:

1. **Meta-model binary**: The trained meta-learner (e.g., Ridge, XGBoost)
2. **Source model references**: List of model names/IDs used as features
3. **Feature configuration**:
   - `use_proba` setting
   - `include_original_features` setting
   - Feature order (column mapping)
4. **Source model binaries**: All base models must also be serialized (already handled by existing controllers)

#### MetaModel Artifact Structure

```python
# Persisted as meta_model_config.json alongside the model binary
{
    "meta_model_type": "MetaModel",
    "source_models": ["PLS-10", "RF"],  # Ordered list
    "source_model_steps": [3, 4],        # Step indices for binary loading
    "use_proba": false,
    "include_original_features": false,
    "feature_columns": ["PLS-10_pred", "RF_pred"],  # For reconstruction
    "task_type": "regression"
}
```

#### Prediction Mode Flow

In prediction mode, the meta-model controller must:

1. **Load source model binaries** from previous steps
2. **Run source models** on new data to generate base predictions
3. **Construct meta-features** from source model predictions
4. **Load meta-model binary** and predict

```python
# MetaModelController._execute_predict()
def _execute_predict(
    self,
    dataset, model_config, context, runtime_context, prediction_store,
    X_train, y_train, X_test, y_test, y_train_unscaled, y_test_unscaled, folds,
    loaded_binaries, mode
):
    """Execute prediction mode for meta-model.

    Unlike training mode where we use OOF predictions from prediction_store,
    in predict mode we must:
    1. Wait for source models to generate predictions on new data
    2. Collect those predictions from prediction_store
    3. Build meta-features and predict
    """
    # Load meta-model configuration
    meta_config = self._load_meta_config(loaded_binaries)
    source_models = meta_config['source_models']

    # Verify source model predictions exist in prediction_store
    # (they were generated by previous steps in this predict run)
    self._wait_for_source_predictions(prediction_store, source_models)

    # Build meta-features from source predictions (test partition only)
    X_meta = self._build_predict_features(
        prediction_store, source_models, meta_config
    )

    # Load trained meta-model and predict
    meta_model = self._load_model_from_binaries(loaded_binaries)
    y_pred = meta_model.predict(X_meta)

    # Store prediction
    self._store_prediction(prediction_store, y_pred, ...)
```

#### Key Insight: Prediction Store as Inter-Step Communication

In prediction mode, the `prediction_store` serves as a communication channel:
- Source models (steps N, N+1, ...) generate predictions and store them
- Meta-model (step M) reads those predictions to construct its features

This is already the pattern in training mode, and it naturally extends to prediction mode.

#### Handling Missing Source Predictions

Edge case: What if a source model fails or is skipped in prediction mode?

```python
def _build_predict_features(self, prediction_store, source_models, meta_config):
    """Build meta-features with fallback for missing sources."""
    features = []
    for model_name in source_models:
        preds = prediction_store.filter_predictions(
            model_name=model_name,
            partition='test',
            load_arrays=True
        )
        if not preds:
            if meta_config.get('strict', True):
                raise ValueError(f"Missing predictions from source model: {model_name}")
            else:
                # Use NaN or zero as placeholder
                features.append(np.zeros(n_samples))
        else:
            features.append(preds[0]['y_pred'])

    return np.column_stack(features)
```

---

## 3. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Create MetaModel Operator
- [ ] Create `nirs4all/operators/models/meta.py`
- [ ] Define `MetaModel` class with configuration parameters
- [ ] Export in `nirs4all/operators/models/__init__.py`

#### 1.2 Create MetaModelController
- [ ] Create `nirs4all/controllers/models/meta_model.py`
- [ ] Implement `matches()` for MetaModel detection
- [ ] Override `get_xy()` for prediction-based feature construction
- [ ] Register controller with appropriate priority
- [ ] Export in `nirs4all/controllers/models/__init__.py`

#### 1.3 Prediction Store Enhancements
- [ ] Add `get_predictions_by_step()` method to Predictions
- [ ] Add `get_oof_predictions()` convenience method
- [ ] Ensure sample_indices are consistently stored for all partitions

### Phase 2: Feature Construction (Week 2-3)

#### 2.1 OOF Prediction Collection
- [ ] Implement `_collect_oof_predictions()` method
- [ ] Handle edge cases (missing folds, partial predictions)
- [ ] Support both regression and classification predictions

#### 2.2 Test Prediction Aggregation
- [ ] Implement `_collect_test_predictions()` method
- [ ] Support fold-averaging for test predictions
- [ ] Handle weighted averaging option

#### 2.3 Feature Matrix Assembly
- [ ] Implement `_build_features_from_predictions()`
- [ ] Support horizontal stacking of multiple models
- [ ] Support passthrough of original features

### Phase 3: Serialization & Prediction Mode (Week 3-4)

#### 3.1 MetaModel Artifact Persistence
- [ ] Create `MetaModelArtifact` dataclass for serialization
- [ ] Persist meta-model configuration alongside binary
- [ ] Store source model references and feature mapping
- [ ] Integrate with existing `ArtifactManager`

#### 3.2 Prediction Mode Implementation
- [ ] Implement `_execute_predict()` for MetaModelController
- [ ] Load meta-model configuration from artifacts
- [ ] Collect source model predictions from prediction_store
- [ ] Build meta-features in prediction mode (no OOF, use test only)

#### 3.3 Binary Loading Integration
- [ ] Ensure `BinaryLoader` handles meta-model artifacts
- [ ] Load both meta-model and its configuration
- [ ] Validate source model availability

### Phase 4: Classification Support (Week 4-5)

#### 4.1 Probability Features
- [ ] Detect classification task type
- [ ] Extract `y_proba` from predictions when available
- [ ] Handle binary vs multiclass probability shapes

#### 4.2 Feature Naming
- [ ] Generate meaningful feature names for meta-features
- [ ] Support feature importance tracking in meta-model

### Phase 5: Integration & Testing (Week 5-6)

#### 5.1 Pipeline Integration
- [ ] Ensure compatibility with existing pipeline syntax
- [ ] Handle edge cases (no previous models, empty predictions)
- [ ] Support mixed frameworks (sklearn meta on TF base models)

#### 5.2 Example and Tests
- [ ] Create `examples/Q_meta_stacking.py` demonstration
- [ ] Write unit tests for MetaModelController
- [ ] Write integration tests for full stacking pipelines
- [ ] **Test save/reload/predict roundtrip**

#### 5.3 Documentation
- [ ] Add docstrings to all new classes/methods
- [ ] Update user guide with stacking documentation
- [ ] Add examples to README

### Phase 5: Advanced Features (Future)

#### 5.1 Multi-Level Stacking
- [ ] Support stacking meta-models on meta-models
- [ ] Implement `level` parameter for layer management

#### 5.2 Prediction Export
- [ ] Allow meta-model to export its constructed features
- [ ] Support feature importance analysis

#### 5.3 Finetune Integration
- [ ] Support Optuna finetuning of meta-model hyperparameters
- [ ] Support automatic source model selection via optimization

---

## 4. File Structure

```
nirs4all/
├── operators/
│   └── models/
│       ├── __init__.py          # Add MetaModel export
│       └── meta.py              # NEW: MetaModel operator
├── controllers/
│   └── models/
│       ├── __init__.py          # Add MetaModelController export
│       └── meta_model.py        # NEW: MetaModelController
├── data/
│   └── predictions.py           # Add get_oof_predictions() etc.
└── ...

examples/
└── Q_meta_stacking.py           # NEW: Stacking example

tests/
└── unit/
    └── controllers/
        └── test_meta_model.py   # NEW: Unit tests

docs/
└── user_guide/
    └── stacking.md              # NEW: User documentation
```

---

## 5. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Fold misalignment | Medium | High | Validation checks, clear error messages |
| Memory usage (large predictions) | Medium | Medium | Lazy loading, array registry optimization |
| Leakage from incorrect OOF | Low | Critical | Comprehensive tests, documentation |
| Performance regression | Low | Medium | Benchmarking before merge |
| Framework incompatibility | Low | Low | Abstract interface, duck typing |

---

## 6. Success Metrics

1. **Functional**: Meta-model produces valid predictions using base model outputs
2. **Correctness**: No data leakage (OOF predictions properly aligned)
3. **Compatibility**: Works with all model controllers (sklearn, TF, PyTorch, JAX)
4. **Usability**: Clean API requiring minimal user configuration
5. **Performance**: Negligible overhead compared to running base models separately

---

## 7. References

- [Stacking (Wikipedia)](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)
- [sklearn StackingClassifier/Regressor](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
- [Kaggle Stacking Guide](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
- nirs4all Roadmap: `> [Stacking] Use prediction as X for stacking.`

---

## 8. Example Implementation: Q_meta_stacking.py

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

## 9. Serialization Test Checklist

When implementing the feature, ensure these test cases pass:

### Training Phase Tests
- [ ] Base models store predictions with correct `sample_indices` and `fold_id`
- [ ] Meta-model receives correct OOF predictions (no leakage)
- [ ] Meta-model artifact includes source model references
- [ ] All binaries (base + meta) are persisted correctly

### Reload Phase Tests
- [ ] `BinaryLoader` correctly loads meta-model configuration
- [ ] `BinaryLoader` correctly loads all source model binaries
- [ ] Pipeline step order matches original training

### Prediction Phase Tests
- [ ] Source models execute and store predictions in prediction_store
- [ ] Meta-model waits for/collects source predictions
- [ ] Meta-features are constructed in correct column order
- [ ] Final predictions match expected values

### Roundtrip Consistency Tests
- [ ] `predict(model_id, X)` == `predict(prediction_entry, X)`
- [ ] Predictions are deterministic (same input → same output)
- [ ] Works with different dataset sizes than training

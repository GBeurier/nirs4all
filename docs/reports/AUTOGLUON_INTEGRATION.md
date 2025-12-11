# AutoGluon Integration Analysis & Roadmap

## Executive Summary

This document analyzes the integration of AutoGluon TabularPredictor into the nirs4all pipeline and provides a roadmap for full feature support.

**Status**: Initial implementation complete
**Priority**: Medium (valuable for automated model selection)
**Effort**: ~4-6 hours for full integration

---

## 1. What is AutoGluon?

AutoGluon is an open-source AutoML library developed by Amazon that automatically trains and tunes machine learning models. Key features:

- **Automatic Model Selection**: Trains multiple models (LightGBM, CatBoost, XGBoost, Random Forest, Neural Networks, etc.)
- **Automatic Ensembling**: Creates weighted ensembles of trained models
- **Zero Configuration**: Works out-of-the-box with minimal configuration
- **Built-in Cross-Validation**: Handles data splitting internally
- **Hyperparameter Tuning**: Internal HPO without needing external tools like Optuna

### AutoGluon vs Traditional Approach

| Aspect | Traditional (sklearn/xgboost) | AutoGluon |
|--------|------------------------------|-----------|
| Model Selection | Manual | Automatic |
| Hyperparameter Tuning | Requires Optuna | Built-in |
| Ensembling | Manual stacking | Automatic weighted ensemble |
| Configuration | Model-specific params | Presets (best_quality, medium_quality, etc.) |
| Training API | `model.fit(X, y)` | `predictor.fit(DataFrame)` |
| Persistence | joblib/pickle per model | Directory-based with multiple models |

---

## 2. Architecture Analysis

### 2.1 Controller Design

The `AutoGluonModelController` follows the same pattern as other model controllers:

```
BaseModelController (abstract)
    ├── SklearnModelController
    ├── TensorFlowModelController
    ├── PyTorchModelController
    ├── JaxModelController
    └── AutoGluonModelController (new)
```

### 2.2 Key Differences from sklearn Controller

| Method | sklearn | AutoGluon |
|--------|---------|-----------|
| `_get_model_instance` | Returns sklearn estimator | Returns config dict (predictor created in train) |
| `_train_model` | Calls `model.fit(X, y)` | Creates DataFrame, calls `predictor.fit(df)` |
| `_predict_model` | `model.predict(X)` | Converts to DataFrame, `predictor.predict(df)` |
| `_clone_model` | `sklearn.base.clone()` | `predictor.clone()` or deepcopy config |
| Persistence | Single pickle file | Directory with multiple model files |

### 2.3 Data Flow

```
nirs4all Pipeline
    │
    ▼
AutoGluonModelController.execute()
    │
    ├── X (np.ndarray) → pd.DataFrame
    ├── y (np.ndarray) → DataFrame column
    │
    ▼
TabularPredictor.fit(DataFrame)
    │
    ├── Trains LightGBM, CatBoost, XGBoost, NN, etc.
    ├── Cross-validates each model
    ├── Creates weighted ensemble
    │
    ▼
TabularPredictor.predict(DataFrame)
    │
    ▼
np.ndarray → nirs4all predictions storage
```

---

## 3. Current Implementation

### 3.1 Completed Features

- [x] `AutoGluonModelController` class with proper registration
- [x] `matches()` method for framework detection
- [x] `_get_model_instance()` - Creates config for TabularPredictor
- [x] `_train_model()` - Trains with DataFrame API
- [x] `_predict_model()` - Predictions with DataFrame conversion
- [x] `_predict_proba_model()` - Classification probabilities
- [x] `_prepare_data()` - Data format preparation
- [x] `_evaluate_model()` - Uses AutoGluon's evaluate method
- [x] `_clone_model()` - Config/predictor cloning
- [x] Framework detection in `ModelFactory.detect_framework()`

### 3.2 Configuration Example

```python
from autogluon.tabular import TabularPredictor

pipeline = [
    # ... preprocessing steps ...
    {
        'model': TabularPredictor,
        'framework': 'autogluon',
        'params': {
            'time_limit': 300,  # 5 minutes training
            'presets': 'medium_quality',
            'verbosity': 1,
        }
    }
]
```

Or using dict config:

```python
pipeline = [
    {
        'model': {
            'framework': 'autogluon',
            'params': {
                'presets': 'best_quality',
                'time_limit': 600,
                'num_bag_folds': 5,
            }
        }
    }
]
```

---

## 4. Known Limitations & Issues

### 4.1 Current Limitations

1. **No Custom Label Column**: Currently uses `__target__` placeholder
2. **Directory-based Persistence**: Doesn't integrate with nirs4all's artifact binary system
3. **No Feature Importance**: AutoGluon's `feature_importance()` not exposed
4. **No Leaderboard Export**: Model comparison not saved to predictions
5. **Limited Finetune Integration**: AutoGluon has internal HPO, Optuna less useful

### 4.2 Compatibility Concerns

- **Memory Usage**: AutoGluon trains many models simultaneously
- **Disk Space**: Model directories can be large (100MB+)
- **Training Time**: Full ensemble training takes longer than single models

---

## 5. Roadmap

### Phase 1: Core Functionality (Done ✓)
- [x] Basic controller implementation
- [x] Training and prediction support
- [x] Framework detection
- [x] Registration with controller registry

### Phase 2: Persistence & Integration (~2 hours)

| Task | Description | Priority |
|------|-------------|----------|
| Directory persistence | Save AutoGluon model directories with artifacts | High |
| Binary conversion | Zip model directory for artifact system | High |
| Load from artifacts | Unzip and load predictor in predict mode | High |
| Cleanup temp dirs | Remove temp directories after saving | Medium |

Implementation:

```python
def save_model(self, model, filepath):
    # Save to temp dir, zip it, store as binary artifact
    import zipfile
    model.save(temp_path)
    with zipfile.ZipFile(filepath + '.zip', 'w') as zf:
        for root, dirs, files in os.walk(temp_path):
            for file in files:
                zf.write(os.path.join(root, file))
```

### Phase 3: Feature Enhancements (~2 hours)

| Task | Description | Priority |
|------|-------------|----------|
| Leaderboard integration | Save model leaderboard to predictions metadata | Medium |
| Feature importance | Expose AutoGluon's feature_importance() | Medium |
| Custom metrics | Support nirs4all custom eval metrics | Low |
| Model selection | Allow specifying which models to train | Low |

### Phase 4: Advanced Features (~2 hours)

| Task | Description | Priority |
|------|-------------|----------|
| Distillation | Support `predictor.distill()` for smaller models | Low |
| Refit full | Support `predictor.refit_full()` for final training | Low |
| HPO integration | Better Optuna integration for preset selection | Low |
| GPU support | Ensure GPU models work correctly | Medium |

---

## 6. Usage Examples

### 6.1 Quick Start (Regression)

```python
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.data.dataset import SpectroDataset

dataset = SpectroDataset("my_data.csv", target="property")

pipeline = [
    {"SavGol": {"window": 11, "polyorder": 2}},
    {"SNV": {}},
    {
        "model": {"framework": "autogluon"},
        "params": {
            "time_limit": 120,
            "presets": "medium_quality"
        }
    }
]

runner = PipelineRunner()
runner.run(dataset, pipeline)
```

### 6.2 Classification with Quality Preset

```python
pipeline = [
    {"StandardScaler": {}},
    {
        "model": {"framework": "autogluon"},
        "params": {
            "presets": "best_quality",
            "time_limit": 600,
            "eval_metric": "balanced_accuracy"
        }
    }
]
```

### 6.3 With Custom Hyperparameters

```python
pipeline = [
    {
        "model": {"framework": "autogluon"},
        "params": {
            "time_limit": 300,
            "hyperparameters": {
                "GBM": {"num_boost_round": 1000},
                "CAT": {"iterations": 500},
                "NN_TORCH": {"epochs": 50}
            }
        }
    }
]
```

---

## 7. Comparison with Existing Controllers

### When to Use AutoGluon vs Others

| Use Case | Recommended Controller |
|----------|----------------------|
| Quick baseline | AutoGluon (best_quality) |
| Production single model | sklearn/XGBoost |
| Deep learning | TensorFlow/PyTorch |
| Maximum control | sklearn with Optuna |
| Time-constrained AutoML | AutoGluon (time_limit) |
| Ensemble from scratch | sklearn StackingRegressor |

### Performance Expectations

Based on typical NIRS datasets:

| Model | Training Time | Prediction Speed | Memory |
|-------|--------------|------------------|--------|
| PLS | ~1s | Very fast | Low |
| XGBoost | ~10s | Fast | Medium |
| AutoGluon (medium) | ~2min | Medium | High |
| AutoGluon (best) | ~10min | Medium | Very High |

---

## 8. Testing Checklist

- [ ] Basic training with regression dataset
- [ ] Basic training with classification dataset
- [ ] Prediction mode with saved model
- [ ] Cross-validation fold handling
- [ ] Multiple time_limit values
- [ ] Different presets (medium_quality, best_quality)
- [ ] Memory cleanup after training
- [ ] Error handling for missing autogluon

---

## 9. Dependencies

Add to `requirements.txt` (optional dependency):

```
# AutoML support (optional)
autogluon>=1.0.0  # Large dependency (~1GB+)
```

Recommend installing separately:

```bash
pip install autogluon.tabular
```

---

## 10. References

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [TabularPredictor API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html)
- [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- [AutoGluon Paper](https://arxiv.org/abs/2003.06505)

---

## Appendix A: AutoGluon Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `best_quality` | Maximum accuracy, slow | Final model, competitions |
| `high_quality` | High accuracy, moderate speed | Production with time budget |
| `good_quality` | Balanced | General use |
| `medium_quality` | Fast, good accuracy | Development, iteration |
| `optimize_for_deployment` | Small, fast models | Edge deployment |
| `interpretable` | Simple models only | Explainability needed |

## Appendix B: Supported Models in AutoGluon

AutoGluon trains these models automatically:

- **LightGBM** (multiple configurations)
- **CatBoost**
- **XGBoost**
- **Random Forest** (Gini and Entropy)
- **Extra Trees**
- **Neural Network** (FastAI and PyTorch)
- **K-Nearest Neighbors**
- **Linear Models**
- **Weighted Ensemble** (final stacking)

# Roadmap: Customizable Phase Models

## Goal
Make Phase 1 (finetuning) and Phase 2 (training) models customizable via `study_runner.py`, enabling users to add models like Ridge finetuning in Phase 1 or multiple XGBoost variants in Phase 2.

---

## Current Architecture

| Phase | Pipeline | Purpose | Hardcoded Models |
|-------|----------|---------|-----------------|
| **Phase 1** | `run_pipeline_1` | Preprocessing selection + finetuning | PLS, OPLS |
| **Phase 2** | `run_pipeline_2` | Ensemble training with top preprocessings | Ridge, RandomForest, CatBoost×3, Nicon×3, LWPLS |
| **Phase 3** | `run_pipeline_3` | TabPFN with expanded preprocessings | TabPFN (already customizable via config) |

---

## Implementation Tasks

### 1. Define Model Configuration Schema (~30 min)
**File:** `study_base_runner.py`

Add new configuration attributes:
```python
# Phase 1 models (finetuned)
self.phase1_models: Optional[List[Dict[str, Any]]] = None

# Phase 2 models (trained with top preprocessings)
self.phase2_models: Optional[List[Dict[str, Any]]] = None
```

Expected model config format:
```python
{
    "model": Ridge(),  # or model class
    "name": "Ridge-Finetuned",
    "finetune_params": {...}  # optional
}
```

---

### 2. Update `study_runner.py` Example Config (~20 min)
**File:** `study_runner.py`

Add commented examples in `MyStudy.__init__`:
```python
# Phase 1: Finetuning models (default: PLS, OPLS)
# self.phase1_models = [
#     {"model": PLSRegression(), "name": "PLS-Finetuned", "finetune_params": {...}},
#     {"model": OPLS(), "name": "OPLS-Finetuned", "finetune_params": {...}},
#     {"model": Ridge(), "name": "Ridge-Finetuned", "finetune_params": {...}},
# ]

# Phase 2: Training models (default: Ridge, RF, CatBoost×3, Nicon×3)
# self.phase2_models = [
#     {"model": XGBRegressor(n_estimators=200), "name": "XGBoost-v1"},
#     {"model": XGBRegressor(n_estimators=400, max_depth=8), "name": "XGBoost-v2"},
#     {"model": XGBRegressor(n_estimators=300, learning_rate=0.05), "name": "XGBoost-v3"},
# ]
```

---

### 3. Update `study_base_runner.py` Config Passing (~15 min)
**File:** `study_base_runner.py`

In `_run_training_advanced()`, add to config dict:
```python
config = {
    ...
    'phase1_models': self.phase1_models,
    'phase2_models': self.phase2_models,
}
```

---

### 4. Refactor `run_pipeline_1` (~30 min)
**File:** `study_training.py`

Change signature:
```python
def run_pipeline_1(dataset_config, filtered_pp_list, aggregation_key, custom_models=None):
```

Logic:
- If `custom_models` is `None`, use default PLS/OPLS
- Otherwise, iterate over `custom_models` and build pipeline steps
- Keep existing finetuning infrastructure intact

---

### 5. Refactor `run_pipeline_2` (~30 min)
**File:** `study_training.py`

Change signature:
```python
def run_pipeline_2(dataset_config, top3_pp, best_n_components, aggregation_key, custom_models=None):
```

Logic:
- If `custom_models` is `None`, use default ensemble (Ridge, RF, CatBoost, Nicon)
- Otherwise, build pipeline from `custom_models` list
- Each entry can be:
  - A simple `{"model": ..., "name": ...}` dict
  - A dict with `"finetune_params"` for finetuned models

---

### 6. Create Default Model Factories (~20 min)
**File:** `study_training.py` (or new `study_model_defaults.py`)

Extract current hardcoded configs into functions:
```python
def get_default_phase1_models(pls_trials, opls_trials):
    """Return default Phase 1 models: PLS, OPLS with finetuning."""
    return [...]

def get_default_phase2_models(ridge_trials, test_lwpls, best_n_components):
    """Return default Phase 2 models: Ridge, RF, CatBoost, Nicon, LWPLS."""
    return [...]
```

---

### 7. Update `main()` to Pass Custom Models (~15 min)
**File:** `study_training.py`

In `main()`, retrieve custom models from module-level vars or config:
```python
phase1_models = getattr(sys.modules[__name__], 'PHASE1_MODELS', None)
phase2_models = getattr(sys.modules[__name__], 'PHASE2_MODELS', None)

# Pass to pipeline functions
preds_1, top_pp_list, best_n_components = run_pipeline_1(
    dataset_config, filtered_pp_list, aggregation_key,
    custom_models=phase1_models
)

preds_2 = run_pipeline_2(
    dataset_config, top_pp_list, best_n_components, aggregation_key,
    custom_models=phase2_models
)
```

---

## Validation Checklist

- [ ] Running with no custom models uses existing defaults (backward compatible)
- [ ] Phase 1 with Ridge finetuning works
- [ ] Phase 2 with 3× XGBoost variants works
- [ ] Test mode respects custom models
- [ ] CLI mode falls back to defaults (no advanced objects via CLI)

---

## Example Usage (after implementation)

```python
# In study_runner.py

from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from nirs4all.operators.models.sklearn import OPLS

class MyStudy(StudyRunner):
    def __init__(self):
        super().__init__()

        # Custom Phase 1: Add Ridge finetuning
        self.phase1_models = [
            {
                "model": PLSRegression(),
                "name": "PLS-Finetuned",
                "finetune_params": {
                    "n_trials": 20,
                    "model_params": {"n_components": ("int", 1, 40)},
                },
            },
            {
                "model": Ridge(),
                "name": "Ridge-Finetuned",
                "finetune_params": {
                    "n_trials": 20,
                    "model_params": {"alpha": ("log_float", 0.001, 100)},
                },
            },
        ]

        # Custom Phase 2: 3× XGBoost variants
        self.phase2_models = [
            {"model": XGBRegressor(n_estimators=200, max_depth=6), "name": "XGBoost-shallow"},
            {"model": XGBRegressor(n_estimators=400, max_depth=10), "name": "XGBoost-deep"},
            {"model": XGBRegressor(n_estimators=300, learning_rate=0.05), "name": "XGBoost-slow"},
        ]
```

---

## Estimated Time: ~2.5 hours

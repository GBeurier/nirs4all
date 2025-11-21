# Model Handling & Fallback Strategy Roadmap

## Executive Summary

This report outlines a roadmap to enhance the model handling capabilities of `nirs4all`. The primary goal is to establish `SklearnModelController` as the universal fallback controller for all standard machine learning models (including XGBoost, LightGBM, CatBoost, and unidentified models), while ensuring that specialized frameworks (TensorFlow, PyTorch, JAX) retain their specific handling logic.

## 1. Current State Analysis

### 1.1 The Routing Mechanism
The `nirs4all` pipeline uses a `ControllerRouter` to assign execution steps to controllers. This assignment is based on a **priority system** where controllers with lower priority values are checked first.

**Current Priorities:**
| Controller | Priority | Matching Logic |
| :--- | :--- | :--- |
| **SklearnModelController** | **6** | Checks for `sklearn.base.BaseEstimator` or dicts with `class` containing "sklearn". |
| **TransformerMixin** | 10 | Checks for `fit` + `transform`. |
| **TensorFlowModelController** | 20 | Checks for `keras.Model` or `framework='tensorflow'`. |
| **PyTorchModelController** | 20 | Checks for `nn.Module` or `framework='pytorch'`. |
| **JaxModelController** | 20 | Checks for `nn.Module` (Flax) or `framework='jax'`. |
| **DummyController** | 1000 | Matches everything else. |

### 1.2 The Problem
1.  **Priority Inversion**: `SklearnModelController` (Priority 6) runs *before* the specific framework controllers (Priority 20). Currently, this doesn't cause conflicts only because `SklearnModelController` is very restrictive in what it accepts (strictly `BaseEstimator`).
2.  **Missing Support**: Libraries like `xgboost`, `lightgbm`, and `catboost` often do not inherit from `sklearn.base.BaseEstimator`. Consequently, they are rejected by the Sklearn controller and, since no other controller matches them, they fall through to the `DummyController`.
3.  **No Generic Fallback**: If a user provides a generic model configuration (e.g., `{"model": SomeCustomClass}`), it is not caught by any controller unless it explicitly mimics Scikit-learn's inheritance structure.

## 2. Proposed Solution

To achieve the desired behavior—where specialized frameworks handle their own models and everything else falls back to the Sklearn controller—we must invert the priority relationship and broaden the Sklearn controller's scope.

### 2.1 Strategy
1.  **Elevate Specific Frameworks**: Move TensorFlow, PyTorch, and JAX controllers to a higher priority (lower number) than Sklearn. This ensures they get "first dibs" on models they recognize.
2.  **Broaden Sklearn Matching**: Relax the matching logic of `SklearnModelController` to accept:
    *   Explicitly supported libraries (`xgboost`, `lightgbm`, `catboost`).
    *   Any object that looks like a model (has `fit`/`predict`).
    *   Any generic model configuration dictionary.

## 3. Implementation Roadmap

### Step 1: Adjust Controller Priorities
**Objective**: Ensure specific frameworks claim their models before the generic fallback.

We need to change the priority of the deep learning framework controllers to be **higher** (lower value) than `SklearnModelController` (which is at 6).

*   **Target Files**:
    *   `nirs4all/controllers/models/tensorflow_model.py`
    *   `nirs4all/controllers/models/torch/torch_model.py`
    *   `nirs4all/controllers/models/jax/jax_model.py`

*   **Change**:
    ```python
    class TensorFlowModelController(BaseModelController):
        priority = 4  # Changed from 20 to 4
    ```
    *(Repeat for PyTorch and JAX)*

### Step 2: Enhance `SklearnModelController.matches`
**Objective**: Make Sklearn controller the universal handler for "unknown" and library-specific models.

We will update the `matches` method to use `ModelFactory` for detection and implement a "catch-all" logic for models.

*   **Target File**: `nirs4all/controllers/models/sklearn_model.py`

*   **New Logic Implementation**:
    ```python
    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        # Extract the model object/config
        model = step.get('model') if isinstance(step, dict) and 'model' in step else step

        # Use Factory to detect framework
        framework = ModelFactory.detect_framework(model)

        # 1. Safety Net: Explicitly reject other specific frameworks
        # This prevents the controller from "stealing" models if priorities are messed up later
        if framework in ['tensorflow', 'pytorch', 'jax']:
            return False

        # 2. Explicitly accept known libraries that follow sklearn API
        if framework in ['sklearn', 'xgboost', 'lightgbm', 'catboost']:
            return True

        # 3. Accept generic objects with fit/predict methods (Duck Typing)
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return True

        # 4. Aggressive Fallback: Accept any dict with 'model' key
        # Since we've already rejected TF/Torch/JAX, this assumes it's a generic model
        if isinstance(step, dict) and 'model' in step:
            return True

        return False
    ```

### Step 3: Verify `ModelFactory` Robustness
**Objective**: Ensure `detect_framework` correctly identifies libraries.

*   **Target File**: `nirs4all/controllers/models/factory.py`
*   **Action**: Verify that `detect_framework` checks for `xgboost`, `lightgbm`, and `catboost` in the module path or MRO (Method Resolution Order). The current implementation appears to support this, but it should be double-checked during implementation.

## 4. Recommendations & Best Practices

### 4.1 Rename to `GenericModelController`
Since `SklearnModelController` is evolving to handle `xgboost`, `catboost`, and custom classes, the name "Sklearn" is becoming a misnomer.
*   **Recommendation**: In a future refactoring, rename `SklearnModelController` to `GenericModelController` or `StandardModelController`. This accurately reflects its role as the handler for all standard, non-deep-learning models.

### 4.2 Explicit Framework Configuration
Auto-detection is powerful but can be fragile (e.g., if a user wraps a model in a custom class that hides the module name).
*   **Recommendation**: Encourage users to add a `framework` key in their model configuration for ambiguous cases:
    ```json
    {
        "model": "my_custom_library.MyModel",
        "framework": "xgboost"
    }
    ```
    This allows `ModelFactory` to bypass detection logic and ensures the correct controller is selected immediately.

### 4.3 Robust Training Loops
By accepting "unknown" models, we assume they follow the Scikit-learn API (`fit(X, y)`, `predict(X)`). However, some libraries might have slight variations (e.g., requiring `eval_set` instead of `validation_data`).
*   **Recommendation**: Ensure `_train_model` in the controller uses `inspect` to check valid arguments before calling `fit()`. This prevents crashes when passing unsupported arguments like `sample_weight` or `eval_set` to simple custom models.

### 4.4 Safety Nets
The "Safety Net" logic in Step 2 (explicitly returning `False` for TF/Torch/JAX) is crucial. Even though we are adjusting priorities, this defensive programming ensures that `SklearnModelController` never accidentally attempts to train a Neural Network using `.fit(X, y)` instead of the specialized training loops required for those frameworks.

## 5. Future Considerations: Custom & Advanced Models

### 5.1 Data Dimensionality (2D vs 1D Samples)
Currently, the `SklearnModelController` (and by extension, the proposed fallback mechanism) implicitly assumes that input data `X` is 2D `(n_samples, n_features)`, where each sample is a 1D vector. The controller's `_prepare_data` method actively flattens any input with more than 2 dimensions.

*   **Insight**: Emerging models or custom deep learning wrappers might expect 2D samples (e.g., images, spectrograms) resulting in 3D input tensors `(n_samples, height, width)` or `(n_samples, time, channels)`.
*   **Implication**: If a custom model requires 3D input but falls back to the `SklearnModelController`, the current data preparation logic will flatten the input, potentially destroying spatial or temporal structure required by the model. Future iterations of the `GenericModelController` may need a mechanism to respect the input dimensionality requested by the model (e.g., via a `_get_input_layout` method or config flag) rather than enforcing a flat 2D structure.

### 5.2 Custom Training Loops
The fallback mechanism relies on the standard `.fit()` API.

*   **Insight**: Complex custom models might require bespoke training loops (e.g., GANs, Reinforcement Learning agents) that cannot be encapsulated in a simple `fit/predict` call.
*   **Implication**: These models will likely require their own dedicated controllers or a more flexible `GenericModelController` that accepts a `trainer` callable in the configuration to override the default `.fit()` behavior.

# 🎯 NIRS4All Prediction Mode Implementation Plan

## Overview

This document outlines the detailed implementation plan for adding prediction capabilities to the NIRS4All pipeline system. The solution leverages deterministic pipeline replay with metadata-based binary loading to enable prediction-only mode execution.

## 🎯 **REFINED IMPLEMENTATION PLAN**

Based on requirements clarification, here's the **detailed implementation strategy** for the predict implementation:

### **Core Architecture Approach**
- **Deterministic Pipeline Replay**: Load pipeline.json and replay each step
- **Metadata-Based Binary Loading**: Use existing metadata.json to find required binaries
- **Prediction Object Input**: Single signature accepting prediction object + dataset

---

## 📋 **DETAILED IMPLEMENTATION PLAN**

### **1. PipelineRunner.predict() - Main Entry Point**

**🔧 Function:** `PipelineRunner.predict(prediction_obj: Dict, dataset_config: DatasetConfigs) -> Dict[str, Any]`

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
@staticmethod
def predict(prediction_obj: Dict[str, Any], dataset_config: DatasetConfigs,
           verbose: int = 0, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute pipeline in prediction mode using a prediction object.

    Args:
        prediction_obj: Prediction entry containing all metadata:
            - config_path: Path to pipeline configuration
            - model_path: Path to specific model (if single model prediction)
            - All other metadata fields for context
        dataset_config: Dataset to make predictions on
        verbose: Verbosity level
        output_path: Optional path to save prediction CSV

    Returns:
        Dict containing predictions, CSV path, and metadata
    """
    # 1. Extract paths and load pipeline configuration
    config_path = prediction_obj['config_path']
    pipeline_config_file = Path(f"results/{config_path}/pipeline.json")
    metadata_file = Path(f"results/{config_path}/metadata.json")

    # 2. Load pipeline steps and metadata
    pipeline_steps = json.load(open(pipeline_config_file))

    # Try enhanced metadata first, fallback to standard metadata
    enhanced_metadata_file = Path(f"results/{config_path}/metadata_enhanced.json")
    if enhanced_metadata_file.exists():
        metadata = json.load(open(enhanced_metadata_file))
    else:
        metadata = json.load(open(metadata_file))

    # 3. Create prediction runner with binary resolution capability
    runner = PipelineRunner(mode="predict", verbose=verbose, save_files=False)
    runner.config_path = config_path
    runner.prediction_metadata = metadata
    runner.target_model_info = prediction_obj  # For model-specific execution

    # 4. Load dataset and execute pipeline
    dataset = dataset_config.get_dataset_at(0)
    prediction_store = Predictions()

    # 5. Run pipeline in prediction mode
    final_dataset = runner._run_single(pipeline_steps, "prediction", dataset, prediction_store)

    # 6. Extract predictions from prediction store
    return runner._extract_prediction_results(prediction_store, prediction_obj)
```

---

### **2. Enhanced Metadata-Based Binary Resolution**

**🔧 Function:** `PipelineRunner._resolve_binaries_for_step(step_number: int) -> List[Tuple[str, Any]]`

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
def _resolve_binaries_for_step(self, step_number: int) -> List[Tuple[str, Any]]:
    """
    Resolve and load binary files for a specific step using enhanced metadata.

    Uses step-to-binary mapping stored during training to eliminate counter guessing.
    """
    step_key = str(step_number)

    # Get binary filenames for this step from enhanced metadata
    binary_filenames = self.prediction_metadata.get('step_binaries', {}).get(step_key, [])

    if not binary_filenames:
        return []  # No binaries for this step

    # Load all binaries for this step
    loaded_binaries = []
    for filename in binary_filenames:
        binary_path = Path(f"results/{self.config_path}/{filename}")
        if binary_path.exists():
            with open(binary_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            loaded_binaries.append((filename, loaded_obj))
        else:
            print(f"⚠️ Binary file not found: {filename}")

    return loaded_binaries

def _resolve_target_model_binary(self) -> Optional[Tuple[str, Any]]:
    """
    Load specific target model binary for model-specific prediction (Q4 use case).
    """
    if not hasattr(self, 'target_model_info'):
        return None

    model_path = self.target_model_info.get('model_path', '')
    if model_path and Path(model_path).exists():
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        return (Path(model_path).name, loaded_obj)

    return None
```

---

### **3. Enhanced Pipeline Execution in Predict Mode**

**🔧 Function:** Modify `PipelineRunner.run_step()` and `PipelineRunner._execute_controller()`

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
def run_step(self, step: Any, dataset: SpectroDataset, context: Dict[str, Any],
            prediction_store: Optional['Predictions'] = None, *, is_substep: bool = False):
    """Enhanced to support predict mode with simplified binary loading"""

    # ... existing step setup code ...

    if self.mode == "predict":
        # Check if controller supports prediction mode
        if not controller.supports_prediction_mode():
            if self.verbose > 0:
                print(f"🔄 Skipping step {self.step_number} ({operator.__class__.__name__}) in prediction mode")
            return context

        # Resolve binaries for this step (no counter guessing needed)
        loaded_binaries = self._resolve_binaries_for_step(self.step_number)

        # For model-specific prediction, also try direct model loading
        if hasattr(self, 'target_model_info') and self.target_model_info.get('step_idx') == self.step_number:
            target_binary = self._resolve_target_model_binary()
            if target_binary:
                loaded_binaries = [target_binary]  # Use specific model

        if self.verbose > 1:
            print(f"🔍 Step {self.step_number}: Loading {len(loaded_binaries)} binaries")
    else:
        loaded_binaries = None

    # Execute with loaded binaries
    return self._execute_controller(
        controller, step, operator, dataset, context,
        prediction_store=prediction_store, source=source,
        loaded_binaries=loaded_binaries
    )

# Enhanced training mode binary tracking
def _execute_controller(self, controller, step, operator, dataset, context,
                       prediction_store=None, source=-1, loaded_binaries=None):
    """Execute controller with enhanced binary tracking for training mode."""

    # ... existing execution logic ...

    # Enhanced binary tracking during training
    if self.mode == "train" and self.save_files and binaries:
        step_key = str(self.step_number)

        # Initialize step_binaries tracking
        if not hasattr(self, 'step_binaries'):
            self.step_binaries = {}

        if step_key not in self.step_binaries:
            self.step_binaries[step_key] = []

        # Track actual saved filenames
        for binary_name, _ in binaries:
            actual_filename = f"{self.step_number}_{binary_name}"
            self.step_binaries[step_key].append(actual_filename)

        # Save binaries and enhanced metadata
        self.saver.save_files(self.step_number, self.substep_number, binaries, self.save_files)

        # Save enhanced metadata with step-to-binary mapping and model performance
        enhanced_metadata = {
            "step_binaries": self.step_binaries,
            "model_performance": getattr(self, 'model_performance', {}),
            "execution_info": {
                "created_at": datetime.now().isoformat(),
                "mode": self.mode,
                "pipeline_version": "2.0"
            }
        }
        self.saver.save_json("metadata_enhanced.json", enhanced_metadata)

    return context
```

---

### **4.1. Enhanced Metadata Tracking for Model Selection**

**🔧 Function:** Track model performance during training for prediction selection

**📁 File:** `nirs4all/controllers/models/base_model_controller.py`

**💻 Implementation:**
```python
# In BaseModelController.launch_training() method
def launch_training(self, dataset, model_config, context, runner, prediction_store, ...):
    # ... existing training logic ...

    # After calculating scores, track model performance
    model_performance_key = f"step_{step_id}_model_{model_name}_fold_{fold_idx}"

    # Initialize model_performance tracking in runner if not exists
    if not hasattr(runner, 'model_performance'):
        runner.model_performance = {}

    # Store model performance for prediction selection
    runner.model_performance[model_performance_key] = {
        "step_idx": step_id,
        "model_name": model_name,
        "model_classname": model_classname,
        "fold_id": fold_idx,
        "val_score": score_val,
        "test_score": score_test,
        "metric": metric,
        "model_path": f"{dataset_name}/{pipeline_name}/{step_id}_{model_name}_{operation_counter}.pkl",
        "binary_filename": f"{step_id}_{model_name}_{operation_counter}.pkl"
    }

    # ... rest of existing logic ...
```

**🔧 Function:** Best model selection for config-based prediction

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
def _select_best_model_from_metadata(self) -> Optional[Dict[str, Any]]:
    """Select best model from enhanced metadata for config-based prediction."""

    model_performance = self.prediction_metadata.get('model_performance', {})
    if not model_performance:
        return None

    # Find best model based on val_score (assuming lower is better for most metrics)
    best_model = None
    best_score = float('inf')

    for model_key, model_info in model_performance.items():
        val_score = model_info.get('val_score')
        if val_score is not None and val_score < best_score:
            best_score = val_score
            best_model = model_info

    return best_model
```

---

### **5. Prediction Result Extraction & Filtering**

**🔧 Function:** Extract predictions from prediction store

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
def _extract_prediction_results(self, prediction_store: 'Predictions', prediction_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and filter prediction results from the prediction store.
    """
    # For model-specific prediction, filter by specific model info
    if hasattr(self, 'target_model_info'):
        model_info = self.target_model_info
        filtered_predictions = prediction_store.filter_predictions(
            dataset_name=model_info.get('dataset_name'),
            config_name=model_info.get('config_name'),
            model_name=model_info.get('model_name'),
            fold_id=model_info.get('fold_id'),
            step_idx=model_info.get('step_idx'),
            partition="prediction"
        )
    else:
        # For config-based prediction, get all predictions
        filtered_predictions = prediction_store.filter_predictions(
            partition="prediction"
        )

    if not filtered_predictions:
        return {"predictions": np.array([]), "metadata": {}}

    # Get the best prediction (or specific one)
    best_prediction = filtered_predictions[0]  # Take first (or implement selection logic)

    return {
        "predictions": best_prediction["y_pred"],
        "y_true": best_prediction.get("y_true", np.array([])),
        "metadata": {
            "model_name": best_prediction["model_name"],
            "model_path": best_prediction["model_path"],
            "fold_id": best_prediction["fold_id"],
            "n_samples": best_prediction["n_samples"],
            "n_features": best_prediction["n_features"]
        }
    }
```

---

### **5. Prediction Result Extraction & Filtering**

**🔧 Function:** Extract predictions from prediction store

**📁 File:** `nirs4all/pipeline/runner.py`

**💻 Implementation:**
```python
def _extract_prediction_results(self, prediction_store: 'Predictions', prediction_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and filter prediction results from the prediction store.
    """
    # For model-specific prediction, filter by specific model info
    if hasattr(self, 'target_model_info'):
        model_info = self.target_model_info
        filtered_predictions = prediction_store.filter_predictions(
            dataset_name=model_info.get('dataset_name'),
            config_name=model_info.get('config_name'),
            model_name=model_info.get('model_name'),
            fold_id=model_info.get('fold_id'),
            step_idx=model_info.get('step_idx'),
            partition="prediction"
        )
    else:
        # For config-based prediction, get all predictions
        filtered_predictions = prediction_store.filter_predictions(
            partition="prediction"
        )

    if not filtered_predictions:
        return {"predictions": np.array([]), "metadata": {}}

    # Get the best prediction (or specific one)
    best_prediction = filtered_predictions[0]  # Take first (or implement selection logic)

    return {
        "predictions": best_prediction["y_pred"],
        "y_true": best_prediction.get("y_true", np.array([])),
        "metadata": {
            "model_name": best_prediction["model_name"],
            "model_path": best_prediction["model_path"],
            "fold_id": best_prediction["fold_id"],
            "n_samples": best_prediction["n_samples"],
            "n_features": best_prediction["n_features"]
        }
    }
```

---

## 🔄 **WORKFLOW SUMMARY**

1. **`PipelineRunner.predict(prediction_obj, dataset_config)`** is called
2. **Load pipeline.json and enhanced metadata** with step-to-binary mapping and model performance
3. **Create runner in predict mode** with prediction metadata and target model info
4. **Execute pipeline steps** with binary loading, skipping non-prediction controllers
5. **Store predictions** in prediction store (like training mode) via controller execution
6. **Extract and filter predictions** from prediction store based on model/fold specifications
7. **Return structured results** with y_pred and metadata using Predictions class capabilities

## 🎯 **ADVANTAGES OF THIS APPROACH**

- ✅ **Simple API**: Single function signature with prediction object
- ✅ **Deterministic**: Replays exact same pipeline using existing metadata
- ✅ **No Counter Guessing**: Direct step-to-binary mapping eliminates complex counter logic
- ✅ **Controller-Based**: Controllers handle prediction logic naturally (like training mode)
- ✅ **Prediction Store Integration**: Leverages existing Predictions class for result management
- ✅ **Model Performance Tracking**: Enhanced metadata tracks val_scores for best model selection
- ✅ **Flexible Filtering**: Can filter predictions by model, fold, or config specifications
- ✅ **Maintainable**: Uses established patterns with minimal new infrastructure

## 📋 **Implementation Checklist**

### Todo List

- [ ] **Enhance PipelineRunner.predict() method**
  - Modify the static predict method to accept prediction object and dataset, load pipeline config, create runner in predict mode, and execute pipeline

- [ ] **Add enhanced metadata tracking**
  - Modify training pipeline to track step-to-binary mapping and save enhanced metadata during execution

- [ ] **Add simplified binary file resolution logic**
  - Create method to resolve binary files using step-to-binary mapping from enhanced metadata, eliminating counter guessing

- [ ] **Implement pipeline replay in predict mode**
  - Modify run/run_steps to support predict mode execution, skipping non-prediction controllers and loading binaries

- [ ] **Update controllers for prediction mode**
  - Modify BaseModelController._execute_prediction_mode() to store predictions in prediction store
  - Update TransformerMixinController to ensure proper binary loading
  - Add supports_prediction_mode() = False to feature augmentation controllers
  - Add supports_prediction_mode() = False to chart and splitter controllers

- [ ] **Add enhanced metadata tracking**
  - Track model performance (val_scores) during training in BaseModelController
  - Store model performance data in enhanced metadata for best model selection
  - Implement best model selection logic for config-based predictions

- [ ] **Add prediction result extraction**
  - Extract predictions from prediction store after pipeline execution
  - Filter predictions based on model/fold specifications
  - Return predictions with metadata (leveraging existing Predictions class capabilities)

## 🔧 **Key Implementation Notes**

### Enhanced Metadata Structure
The system uses enhanced metadata to track step-to-binary mapping and model performance:
```json
{
  "step_binaries": {
    "1": ["1_MinMaxScaler_1.pkl"],
    "4": ["4_PLSRegression_1.pkl", "4_PLSRegression_2.pkl", "4_PLSRegression_3.pkl"]
  },
  "model_performance": {
    "step_4_model_PLSRegression_fold_0": {
      "step_idx": 4,
      "model_name": "PLSRegression",
      "fold_id": 0,
      "val_score": 20.088,
      "test_score": 15.019,
      "metric": "rmse",
      "model_path": "regression/config_7dbfba/4_PLSRegression_1.pkl"
    }
  },
  "execution_info": {
    "created_at": "2025-10-02T...",
    "mode": "train",
    "pipeline_version": "2.0"
  }
}
```

### Binary File Resolution
- **Training**: Tracks step-to-binary mapping automatically during execution
- **Prediction**: Direct lookup using step number - no counter guessing needed
- **Model-Specific**: Direct binary loading from model_path for Q4 scenarios

### Pipeline Execution Flow
1. **Load enhanced metadata** with step-to-binary mapping and model performance
2. **Skip non-prediction controllers** (charts, splitters, augmentation, etc.)
3. **Load step binaries** using direct mapping lookup
4. **Execute model controllers** in prediction mode, storing results in prediction store
5. **Extract predictions** from prediction store with filtering by model/fold
6. **Return structured results** with y_pred and metadata (using Predictions class capabilities)

### Prediction Object Structure
Expected prediction object contains:
```json
{
    "dataset_name": "regression",
    "dataset_path": "regression",
    "config_name": "config_7dbfba",
    "config_path": "regression/config_7dbfba",
    "step_idx": 4,
    "op_counter": 1,
    "model_name": "PLSRegression",
    "model_classname": "PLSRegression",
    "model_path": "regression/config_7dbfba/4_PLSRegression_1.pkl",
    "fold_id": "0",
    "sample_indices": "...",
    "weights": "[]",
    "metadata": "{}",
    "partition": "train",
    "y_true": "...",
    "y_pred": "...",
    "val_score": 20.088551184412292,
    "test_score": 15.018898780399995,
    "metric": "rmse",
    "task_type": "regression",
    "n_samples": 97,
    "n_features": 2151,
    "preprocessings": "MinMax"
}
```

### Pipeline Configuration
Uses existing pipeline.json format:
```json
[
  {
    "class": "sklearn.preprocessing._data.MinMaxScaler"
  },
  {
    "y_processing": {
      "class": "sklearn.preprocessing._data.MinMaxScaler"
    }
  },
  {
    "class": "sklearn.model_selection._split.ShuffleSplit",
    "params": {
      "n_splits": 3,
      "test_size": 0.25
    }
  },
  {
    "model": {
      "class": "sklearn.cross_decomposition._pls.PLSRegression",
      "params": {
        "n_components": 10
      }
    }
  }
]
```
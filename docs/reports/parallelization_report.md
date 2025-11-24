# Report: Reintroducing Parallel Execution in nirs4all

## 1. Current Architecture Analysis

The current pipeline execution flow is strictly sequential:
- **`PipelineOrchestrator`**: Iterates over datasets and pipeline configurations.
- **`PipelineExecutor`**: Iterates over the main steps of a pipeline.
- **`StepRunner`**: Executes individual steps. It handles nested lists (`StepType.SUBPIPELINE`) by iterating through them sequentially.

The previous `run_steps` method with `execution="parallel"` logic has been removed during refactoring. To reintroduce this capability, we need to integrate it into the new `StepRunner` architecture.

## 2. Strategy for Reintroduction

We will introduce a dedicated **Parallel Step** type. This allows users to explicitly define which parts of the pipeline should run in parallel, offering granular control.

### 2.1 Configuration Syntax
We will support a new dictionary-based syntax in the pipeline configuration:

```json
{
    "parallel": [
        {"model": "SVC", "params": {"C": 1.0}},
        {"model": "SVC", "params": {"C": 10.0}}
    ],
    "n_jobs": -1
}
```

- **`parallel`**: A list of steps (or sub-pipelines) to execute in parallel.
- **`n_jobs`**: (Optional) Number of parallel workers (default: -1 for all CPUs).

### 2.2 Parsing Logic (`StepParser`)
The `StepParser` will be updated to:
1.  Detect the `parallel` keyword in dictionary steps.
2.  Parse the list of steps within `parallel`.
3.  Return a `ParsedStep` with `StepType.PARALLEL`.

### 2.3 Execution Logic (`StepRunner`)
The `StepRunner` will be updated to handle `StepType.PARALLEL`:
1.  **Context Propagation**: All parallel branches will receive the **same initial context**. This ensures they start from the same state (same dataset, same metadata).
2.  **Parallel Execution**: Use `joblib.Parallel` and `delayed` to execute the branches concurrently.
3.  **Result Aggregation**:
    - **Predictions**: Each branch will produce its own `Predictions` object. These will be merged into the main `prediction_store`.
    - **Artifacts**: Artifacts from all branches will be collected and returned.
    - **Context**: The execution will return the **original input context** (or potentially the context from the last branch, but returning the original context is safer for independent branches like model training). *Note: If parallel steps are intended to modify the dataset (e.g., parallel data augmentation), a subsequent "Merge" step might be required to combine the results, which is outside the scope of this immediate change.*

## 3. Implementation Plan

### Step 1: Update `StepParser`
**File:** `nirs4all/pipeline/steps/parser.py`
- Add `PARALLEL` to `StepType` enum.
- In `_parse_dict_step`, check for the `parallel` keyword.
- Create a `ParsedStep` with `StepType.PARALLEL` and extract `n_jobs`.

### Step 2: Update `StepRunner`
**File:** `nirs4all/pipeline/steps/step_runner.py`
- Import `Parallel`, `delayed` from `joblib`.
- Add a helper method `_execute_branch` (static or instance) to run a single branch, capturing its predictions and artifacts.
- In `execute`, add a handler for `StepType.PARALLEL`:
    - Configure `Parallel` with `n_jobs`.
    - Run branches.
    - Merge `Predictions` into `prediction_store`.
    - Aggregate artifacts.
    - Return `StepResult` with the original context (or updated if a merge strategy is defined).

## 4. Implications & Limitations

- **Thread Safety**: `joblib` with `n_jobs > 1` typically uses multiprocessing (or threading).
    - **Multiprocessing**: Objects (Context, Dataset) are pickled. Changes to the dataset in a branch **will not** be reflected in the main process unless explicitly returned and merged. This is fine for "models on CPU" or "independent reporting".
    - **Threading**: Shared memory. `SpectroDataset` and `Predictions` need to be thread-safe if modified concurrently. `Predictions` has a lock? No. But `joblib` usually pickles for safety with `loky` backend.
- **Use Cases Supported**:
    - Grid Search / Multiple Models (Perfect fit).
    - Independent Analyses (Perfect fit).
    - Parallel Preprocessing (Works if side-effects are artifacts; if modifying dataset, requires careful context management).

## 5. Example Usage

```python
pipeline = [
    {"preprocessing": "StandardScaler"},
    {
        "parallel": [
            {"model": "RandomForestClassifier", "params": {"n_estimators": 100}},
            {"model": "SVC", "params": {"kernel": "rbf"}},
            {"model": "LogisticRegression"}
        ],
        "n_jobs": 3
    }
]
```

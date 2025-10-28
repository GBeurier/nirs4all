# Group-Based Splitting with Metadata Integration

**Date**: October 14, 2025
**Status**: 📋 Design Document - Ready for Implementation

---

## Executive Summary

This document outlines the implementation plan for adding group-based cross-validation support to nirs4all. The feature will allow splitters like `GroupKFold` and `GroupShuffleSplit` to use metadata columns (e.g., sample IDs, batch numbers) for group-aware splitting, preventing data leakage from related samples.

**New Syntax**:
```python
{"split": GroupKFold(n_splits=5), "group": "batch_id"}
```

**Key Benefits**:
- ✅ Prevent data leakage from related samples (same batch, patient, location, etc.)
- ✅ Seamless integration with existing metadata system
- ✅ Backward compatible with existing pipelines
- ✅ Consistent with existing special operator patterns (`y_processing`, `feature_augmentation`)
- ✅ Proper serialization and prediction mode support

---

## Current State Analysis

### Metadata System (✅ Complete)

The metadata system is already implemented and tested:

**Dataset Methods Available**:
```python
# From d:\Workspace\ML\NIRS\nirs4all\nirs4all\dataset\dataset.py

def metadata_column(self, column: str, selector: Optional[Selector] = None) -> np.ndarray:
    """Get single metadata column as array."""
    indices = self._indexer.x_indices(selector) if selector else None
    return self._metadata.get_column(column, indices)
```

**Features**:
- ✅ Stores sample-level auxiliary information
- ✅ Supports filtering by partition (train/test)
- ✅ Returns numpy arrays compatible with sklearn
- ✅ Handles missing values and data types
- ✅ Column-level access with selector support

### Cross-Validator Controller (⚠️ Needs Enhancement)

**Current Implementation**: `d:\Workspace\ML\NIRS\nirs4all\nirs4all\controllers\sklearn\op_split.py`

**What Works**:
- ✅ Automatic detection of splitter requirements (`_needs()` function)
- ✅ Introspects `split()` signature to detect `y` and `groups` parameters
- ✅ Generates and saves fold indices
- ✅ Supports prediction mode

**What's Missing**:
- ❌ No mechanism to specify which metadata column to use as groups
- ❌ Groups parameter hardcoded to `None` (line 105)
- ❌ No support for `{"split": ..., "group": ...}` syntax
- ❌ No serialization handling for the new format

**Current Limitation (Line 105)**:
```python
groups = dataset.groups(local_context) if needs_g else None
groups = None  # ← Currently hardcoded!
```

The TODO comment on line 85 acknowledges this: `##TODO manage groups`

### Runner and Serialization (⚠️ Needs Enhancement)

**WORKFLOW_OPERATORS** (line 50 in runner.py):
```python
WORKFLOW_OPERATORS = ["sample_augmentation", "feature_augmentation", "branch",
                      "dispatch", "model", "stack", "scope", "cluster",
                      "merge", "uncluster", "unscope", "chart_2d", "chart_3d",
                      "fold_chart", "model", "y_processing", "y_chart"]
```

**Missing**: `"split"` is NOT in the list!

**Impact**: The runner won't recognize `{"split": ...}` as a special workflow operator and will fail to route it correctly.

---

## Proposed Solution

### 1. New Pipeline Syntax

Users will specify group-based splitting using the `split` keyword:

```python
# Basic usage - group column name
{"split": GroupKFold(n_splits=5), "group": "batch_id"}

# With parameters
{"split": {"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}, "group": "sample_id"}

# Optional: default to first metadata column if not specified
{"split": GroupKFold(n_splits=5)}  # Uses metadata column 0 if available
```

**Backward Compatibility**: Existing pipelines without the `split` keyword continue to work:
```python
# Still valid - matches directly as operator
GroupKFold(n_splits=5)

# Still valid - uses existing dict format
{"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}
```

### 2. Serialization Format

**Input (Python)**:
```python
{"split": GroupKFold(n_splits=5), "group": "batch_id"}
```

**Serialized (JSON)**:
```json
{
  "split": {
    "class": "sklearn.model_selection._split.GroupKFold",
    "params": {
      "n_splits": 5
    }
  },
  "group": "batch_id"
}
```

**With Defaults (Minimal)**:
```python
{"split": GroupKFold(), "group": "sample"}
```

**Serialized (Minimal)**:
```json
{
  "split": "sklearn.model_selection._split.GroupKFold",
  "group": "sample"
}
```

**Normalization Rule**: The `split` key follows the same serialization rules as `model`, `y_processing`, etc.:
- Classes with default params → serialize to string
- Instances with non-default params → serialize to dict with `class` and `params`

### 3. Matching Logic Enhancement

**Enhanced `matches()` method in `CrossValidatorController`**:

```python
@classmethod
def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
    """Return True if operator behaves like a splitter OR step contains 'split' keyword."""

    # NEW: Match on 'split' keyword (higher priority)
    if keyword == "split":
        return True

    # NEW: Match dict with 'split' key
    if isinstance(step, dict) and "split" in step:
        return True

    # EXISTING: Match objects with split() method
    if operator is not None:
        split_fn = getattr(operator, "split", None)
        if not callable(split_fn):
            return False
        try:
            sig = inspect.signature(split_fn)
        except (TypeError, ValueError):
            return True
        params: List[inspect.Parameter] = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return bool(params) and params[0].name == "X"

    return False
```

**Matching Priority**:
1. `keyword == "split"` (highest - explicit intent)
2. `"split" in step` (dict structure)
3. `operator.split()` (object introspection - existing behavior)

### 4. Execution Logic Enhancement

**Enhanced `execute()` method in `CrossValidatorController`**:

```python
def execute(
    self,
    step: Any,
    operator: Any,
    dataset: "SpectroDataset",
    context: Dict[str, Any],
    runner: "PipelineRunner",
    source: int = -1,
    mode: str = "train",
    loaded_binaries: Any = None,
    prediction_store: Any = None
):
    """Run operator.split and store folds, with optional metadata group support."""

    local_context = copy.deepcopy(context)
    local_context["partition"] = "train"

    # NEW: Extract group column specification
    group_column = None
    if isinstance(step, dict) and "group" in step:
        group_column = step["group"]

    # Detect requirements
    needs_y, needs_g = _needs(operator)

    # Get data
    X = dataset.x(local_context, layout="2d", concat_source=True)
    y = dataset.y(local_context) if needs_y else None

    # NEW: Get groups from metadata if needed
    groups = None
    if needs_g:
        if group_column is None:
            # Default behavior: use first metadata column if available
            if hasattr(dataset, 'metadata_columns') and dataset.metadata_columns:
                group_column = dataset.metadata_columns[0]
                print(f"⚠️ No group column specified, using default: '{group_column}'")
            else:
                raise ValueError(
                    f"{operator.__class__.__name__} requires groups but no metadata available. "
                    f"Specify group column using: {{'split': {operator.__class__.__name__}(), 'group': 'column_name'}}"
                )

        # Extract groups from metadata
        try:
            groups = dataset.metadata_column(group_column, local_context)
        except Exception as e:
            raise ValueError(
                f"Failed to extract groups from metadata column '{group_column}': {e}\n"
                f"Available columns: {dataset.metadata_columns}"
            ) from e

    # Build kwargs
    kwargs: Dict[str, Any] = {}
    if needs_y:
        if y is None:
            raise ValueError(f"{operator.__class__.__name__} requires y but dataset.y returned None")
        kwargs["y"] = y
    if needs_g:
        if groups is None:
            raise ValueError(f"{operator.__class__.__name__} requires groups but groups is None")
        kwargs["groups"] = groups

    # Generate folds (existing logic continues...)
    if mode != "predict" and mode != "explain":
        folds = list(operator.split(X, **kwargs))
        # ... rest of fold generation and saving

    return context, binaries
```

### 5. Runner Integration

**Add `"split"` to `WORKFLOW_OPERATORS`** in `runner.py`:

```python
WORKFLOW_OPERATORS = [
    "sample_augmentation",
    "feature_augmentation",
    "branch",
    "dispatch",
    "model",
    "stack",
    "scope",
    "cluster",
    "merge",
    "uncluster",
    "unscope",
    "chart_2d",
    "chart_3d",
    "fold_chart",
    "y_processing",
    "y_chart",
    "split"  # ← NEW
]
```

**Why This Matters**:
- Enables the runner to recognize `{"split": ...}` as a special workflow operator
- Routes to controller with `keyword="split"` for proper matching
- Extracts operator from `step["split"]` (similar to how `model`, `y_processing` work)

**Runner Logic Flow** (already implemented pattern):

```python
# In run_step() method (line ~507)
if isinstance(step, dict):
    if key := next((k for k in step if k in self.WORKFLOW_OPERATORS), None):
        # ✅ This will now match key="split"
        if 'class' in step[key]:
            if '_runtime_instance' in step[key]:
                operator = step[key]['_runtime_instance']
            else:
                operator = deserialize_component(step[key])
            controller = self._select_controller(step, keyword=key, operator=operator)
        else:
            controller = self._select_controller(step, keyword=key)
```

### 6. Serialization Enhancement

**Preprocessing** (`_preprocess_steps()` in `config.py`):

The existing logic already handles special operators like `y_processing`:

```python
# Existing code that handles {"y_processing": MinMaxScaler} → {"y_processing": {"class": ...}}
for key, value in list(result.items()):
    if (inspect.isclass(value) and
        key not in ["class", "params"] and
        not key.endswith("_params")):
        result[key] = {"class": value}
```

**Enhancement Needed**: None! The existing logic already handles:
- `{"split": GroupKFold}` → `{"split": {"class": "sklearn.model_selection._split.GroupKFold"}}`
- `{"split": GroupKFold(), "group": "batch"}` → Preserved with serialization

**Serialization** (`serialize_component()` in `serialization.py`):

No changes needed - already handles nested dicts and special keys.

---

## Detailed Implementation Roadmap

### Phase 1: Core Functionality (Priority: High)

**Estimated Time**: 2-3 hours

#### Step 1.1: Update WORKFLOW_OPERATORS (15 min)
**File**: `nirs4all/pipeline/runner.py`

```python
# Line 50 - Add "split" to the list
WORKFLOW_OPERATORS = [
    "sample_augmentation", "feature_augmentation", "branch", "dispatch",
    "model", "stack", "scope", "cluster", "merge", "uncluster", "unscope",
    "chart_2d", "chart_3d", "fold_chart", "y_processing", "y_chart",
    "split"  # ← ADD THIS
]
```

**Testing**:
```python
# Verify runner recognizes the keyword
assert "split" in PipelineRunner.WORKFLOW_OPERATORS
```

#### Step 1.2: Enhance CrossValidatorController.matches() (30 min)
**File**: `nirs4all/controllers/sklearn/op_split.py`

```python
@classmethod
def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
    """Return True if operator behaves like a splitter OR step contains 'split' keyword."""

    # Priority 1: Match on 'split' keyword (explicit workflow operator)
    if keyword == "split":
        return True

    # Priority 2: Match dict with 'split' key
    if isinstance(step, dict) and "split" in step:
        return True

    # Priority 3: Match objects with split() method (existing behavior)
    if operator is not None:
        split_fn = getattr(operator, "split", None)
        if not callable(split_fn):
            return False
        try:
            sig = inspect.signature(split_fn)
        except (TypeError, ValueError):
            return True
        params: List[inspect.Parameter] = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return bool(params) and params[0].name == "X"

    return False
```

**Testing**:
```python
from sklearn.model_selection import GroupKFold

controller = CrossValidatorController()

# Test keyword matching
assert controller.matches({"split": GroupKFold()}, None, "split")

# Test dict structure matching
assert controller.matches({"split": GroupKFold(), "group": "batch"}, None, "")

# Test backward compatibility - direct operator
assert controller.matches(GroupKFold(), GroupKFold(), "")
```

#### Step 1.3: Enhance CrossValidatorController.execute() (1.5 hours)
**File**: `nirs4all/controllers/sklearn/op_split.py`

```python
def execute(
    self,
    step: Any,
    operator: Any,
    dataset: "SpectroDataset",
    context: Dict[str, Any],
    runner: "PipelineRunner",
    source: int = -1,
    mode: str = "train",
    loaded_binaries: Any = None,
    prediction_store: Any = None
):
    """Run operator.split and store folds, with optional metadata group support."""

    local_context = copy.deepcopy(context)
    local_context["partition"] = "train"

    # Extract group column specification
    group_column = None
    if isinstance(step, dict) and "group" in step:
        group_column = step["group"]
        if not isinstance(group_column, str):
            raise TypeError(f"Group column must be a string, got {type(group_column)}")

    # Detect requirements
    needs_y, needs_g = _needs(operator)

    # Get data
    X = dataset.x(local_context, layout="2d", concat_source=True)
    y = dataset.y(local_context) if needs_y else None

    # Get groups from metadata if needed
    groups = None
    if needs_g:
        if group_column is None:
            # Default behavior: use first metadata column if available
            if hasattr(dataset, 'metadata_columns') and dataset.metadata_columns:
                group_column = dataset.metadata_columns[0]
                print(f"⚠️ GroupKFold requires groups but no 'group' specified. Using default: '{group_column}'")
            else:
                raise ValueError(
                    f"{operator.__class__.__name__} requires groups parameter.\n"
                    f"Dataset has no metadata columns.\n"
                    f"Please add metadata or use a non-grouped splitter.\n"
                    f"Syntax: {{'split': {operator.__class__.__name__}(n_splits=5), 'group': 'column_name'}}"
                )

        # Validate column exists
        if hasattr(dataset, 'metadata_columns'):
            if group_column not in dataset.metadata_columns:
                raise ValueError(
                    f"Group column '{group_column}' not found in metadata.\n"
                    f"Available columns: {dataset.metadata_columns}"
                )

        # Extract groups from metadata
        try:
            groups = dataset.metadata_column(group_column, local_context)
            if len(groups) != X.shape[0]:
                raise ValueError(
                    f"Group array length ({len(groups)}) doesn't match X rows ({X.shape[0]})"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to extract groups from metadata column '{group_column}': {e}"
            ) from e

    n_samples = X.shape[0]

    # Build kwargs
    kwargs: Dict[str, Any] = {}
    if needs_y:
        if y is None:
            raise ValueError(f"{operator.__class__.__name__} requires y but dataset.y returned None")
        kwargs["y"] = y
    if needs_g:
        if groups is None:
            raise ValueError(f"{operator.__class__.__name__} requires groups but groups is None")
        kwargs["groups"] = groups

    # Execute splitting (existing logic)
    if mode != "predict" and mode != "explain":
        folds = list(operator.split(X, **kwargs))

        if dataset.x({"partition": "test"}).shape[0] == 0:
            print("⚠️ No test partition found; using first fold as test set.")
            fold_1 = folds[0]
            dataset._indexer.update_by_indices(fold_1[1], {"partition": "test"})
            return context, []
        else:
            dataset.set_folds(folds)

            # Generate CSV binary
            headers = [f"fold_{i}" for i in range(len(folds))]
            binary = ",".join(headers).encode("utf-8") + b"\n"
            max_train_samples = max(len(train_idx) for train_idx, _ in folds)

            for row_idx in range(max_train_samples):
                row_values = []
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    if row_idx < len(train_idx):
                        row_values.append(str(train_idx[row_idx]))
                    else:
                        row_values.append("")
                binary += ",".join(row_values).encode("utf-8") + b"\n"

            # Filename includes group column if used
            folds_name = f"folds_{operator.__class__.__name__}"
            if group_column:
                folds_name += f"_group-{group_column}"
            if hasattr(operator, "random_state"):
                seed = getattr(operator, "random_state")
                if seed is not None:
                    folds_name += f"_seed{seed}"
            folds_name += ".csv"

            return context, [(folds_name, binary)]
    else:
        # Prediction mode
        n_folds = operator.get_n_splits(**kwargs) if hasattr(operator, "get_n_splits") else 1
        dataset.set_folds([(list(range(n_samples)), [])] * n_folds)
        return context, []
```

**Key Enhancements**:
1. ✅ Extract `group_column` from step dict
2. ✅ Validate column name (string type)
3. ✅ Default to first metadata column with warning
4. ✅ Validate column exists in metadata
5. ✅ Extract groups using `dataset.metadata_column()`
6. ✅ Validate groups array length matches X
7. ✅ Include group column in filename for tracking
8. ✅ Comprehensive error messages

**Testing**:
```python
from sklearn.model_selection import GroupKFold
from nirs4all.data.dataset import SpectroDataset
import numpy as np
import pandas as pd

# Create test dataset with metadata
dataset = SpectroDataset(name="test")
X = np.random.rand(100, 10)
y = np.random.rand(100)
metadata = pd.DataFrame({
    'batch': [1]*25 + [2]*25 + [3]*25 + [4]*25,
    'location': ['A']*50 + ['B']*50
})

dataset.add_samples(X, {"partition": "train"})
dataset.add_targets(y)
dataset.add_metadata(metadata)

# Test with group specification
step = {"split": GroupKFold(n_splits=4), "group": "batch"}
controller = CrossValidatorController()
context = {"processing": [["raw"]]}
runner = MockRunner()

context, binaries = controller.execute(step, step["split"], dataset, context, runner)

# Verify folds were created
assert dataset._folds is not None
assert len(dataset._folds) == 4

# Verify binary filename includes group
assert binaries[0][0].startswith("folds_GroupKFold_group-batch")
```

#### Step 1.4: Unit Tests (45 min)
**File**: `tests/test_group_split.py`

```python
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from nirs4all.data.dataset import SpectroDataset
from nirs4all.controllers.splitters.split import CrossValidatorController
from nirs4all.pipeline.runner import PipelineRunner

class TestGroupSplitSyntax:
    """Test new split syntax with group parameter."""

    def test_matches_split_keyword(self):
        """Test controller matches on 'split' keyword."""
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group": "batch"}
        assert controller.matches(step, None, "split")

    def test_matches_split_in_dict(self):
        """Test controller matches dict with 'split' key."""
        controller = CrossValidatorController()
        step = {"split": GroupKFold(), "group": "batch"}
        assert controller.matches(step, None, "")

    def test_backward_compatible_matching(self):
        """Test backward compatibility with direct operator."""
        controller = CrossValidatorController()
        splitter = GroupKFold()
        assert controller.matches(splitter, splitter, "")

class TestGroupSplitExecution:
    """Test execution with metadata groups."""

    @pytest.fixture
    def dataset_with_metadata(self):
        """Create dataset with metadata for testing."""
        dataset = SpectroDataset(name="test")
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        metadata = pd.DataFrame({
            'batch': [1]*25 + [2]*25 + [3]*25 + [4]*25,
            'location': ['A']*50 + ['B']*50,
            'sample_id': range(100)
        })

        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)
        dataset.add_metadata(metadata)
        return dataset

    def test_group_split_with_batch(self, dataset_with_metadata):
        """Test GroupKFold with batch column."""
        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()
        context = {"processing": [["raw"]]}

        context, binaries = controller.execute(
            step, step["split"], dataset_with_metadata, context, None, mode="train"
        )

        # Verify folds created
        assert dataset_with_metadata._folds is not None
        assert len(dataset_with_metadata._folds) == 4

        # Verify no batch leakage between train/val
        for train_idx, val_idx in dataset_with_metadata._folds:
            train_batches = dataset_with_metadata.metadata_column("batch")[train_idx]
            val_batches = dataset_with_metadata.metadata_column("batch")[val_idx]
            # No overlap in batches
            assert len(set(train_batches) & set(val_batches)) == 0

    def test_default_group_column(self, dataset_with_metadata):
        """Test default to first metadata column."""
        step = {"split": GroupKFold(n_splits=4)}  # No group specified
        controller = CrossValidatorController()
        context = {"processing": [["raw"]]}

        # Should use first column (batch) by default
        context, binaries = controller.execute(
            step, step["split"], dataset_with_metadata, context, None, mode="train"
        )

        assert dataset_with_metadata._folds is not None
        assert binaries[0][0] == "folds_GroupKFold_group-batch.csv"

    def test_invalid_group_column(self, dataset_with_metadata):
        """Test error on invalid group column."""
        step = {"split": GroupKFold(n_splits=4), "group": "nonexistent"}
        controller = CrossValidatorController()
        context = {"processing": [["raw"]]}

        with pytest.raises(ValueError, match="not found in metadata"):
            controller.execute(
                step, step["split"], dataset_with_metadata, context, None, mode="train"
            )

    def test_no_metadata_error(self):
        """Test error when no metadata available."""
        dataset = SpectroDataset(name="test")
        dataset.add_samples(np.random.rand(100, 10), {"partition": "train"})
        dataset.add_targets(np.random.rand(100))

        step = {"split": GroupKFold(n_splits=4), "group": "batch"}
        controller = CrossValidatorController()
        context = {"processing": [["raw"]]}

        with pytest.raises(ValueError, match="no metadata"):
            controller.execute(step, step["split"], dataset, context, None, mode="train")

    def test_non_grouped_splitter(self, dataset_with_metadata):
        """Test non-grouped splitter still works."""
        step = KFold(n_splits=5)  # No groups needed
        controller = CrossValidatorController()
        context = {"processing": [["raw"]]}

        context, binaries = controller.execute(
            step, step, dataset_with_metadata, context, None, mode="train"
        )

        assert dataset_with_metadata._folds is not None
        assert len(dataset_with_metadata._folds) == 5

class TestSerialization:
    """Test serialization of new syntax."""

    def test_serialize_split_with_group(self):
        """Test serialization preserves group parameter."""
        from nirs4all.pipeline.config import PipelineConfigs

        pipeline = [
            {"split": GroupKFold(n_splits=5), "group": "batch_id"}
        ]

        config = PipelineConfigs(pipeline)
        serialized = config.serializable_steps(config.steps[0])

        # Verify structure preserved
        assert "split" in serialized[0]
        assert "group" in serialized[0]
        assert serialized[0]["group"] == "batch_id"

    def test_roundtrip_serialization(self):
        """Test save/load roundtrip."""
        from nirs4all.pipeline.config import PipelineConfigs
        import json

        original = [
            {"split": {"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}, "group": "sample"}
        ]

        config = PipelineConfigs(original)
        serialized = json.dumps(config.serializable_steps(config.steps[0]))
        deserialized = json.loads(serialized)

        assert deserialized[0]["group"] == "sample"
        assert "GroupKFold" in deserialized[0]["split"]["class"]
```

**Run Tests**:
```bash
pytest tests/test_group_split.py -v
```

---

### Phase 2: Documentation (Priority: High)

**Estimated Time**: 1-2 hours

#### Step 2.1: Update WRITING_A_PIPELINE.md (45 min)
**File**: `docs/WRITING_A_PIPELINE.md`

Add new section after "7. Dictionary - Special Operators":

```markdown
### 8. Dictionary - Split with Group

**Syntax**: Dict with `split` and optional `group` keys.

```python
pipeline = [
    {"split": GroupKFold(n_splits=5), "group": "batch_id"}
]
```

**Purpose**: Cross-validation with group awareness using metadata columns.

**Use Cases**:
- Prevent data leakage from related samples
- Stratify by batch, patient, location, instrument, etc.
- Ensure temporal or spatial independence

**Serializes to**:
```json
{
    "split": {
        "class": "sklearn.model_selection._split.GroupKFold",
        "params": {
            "n_splits": 5
        }
    },
    "group": "batch_id"
}
```

**With defaults**:
```python
pipeline = [
    {"split": GroupKFold()}  # Uses first metadata column by default
]
```

Serializes to:
```json
{
    "split": "sklearn.model_selection._split.GroupKFold"
}
```

**Group Column Options**:
- String: Name of metadata column (e.g., `"batch"`, `"patient_id"`)
- Omitted: Uses first metadata column with warning

**Supported Splitters**:
- `GroupKFold`: K-fold with group awareness
- `GroupShuffleSplit`: Random shuffle with group awareness
- `LeaveOneGroupOut`: Leave one group out
- `LeavePGroupsOut`: Leave P groups out
- Custom splitters with `groups` parameter

**Example**:
```python
from sklearn.model_selection import GroupKFold

pipeline = [
    MinMaxScaler(),
    {"split": GroupKFold(n_splits=5), "group": "batch_number"},
    PLSRegression(n_components=10)
]
```
```

#### Step 2.2: Update METADATA_USAGE.md (30 min)
**File**: `docs/METADATA_USAGE.md`

Add new section "Using Metadata for Group-Aware Cross-Validation":

```markdown
## Using Metadata for Group-Aware Cross-Validation

### Preventing Data Leakage

When samples are related (same batch, patient, time period), standard cross-validation can leak information between folds. Group-aware splitting ensures related samples stay together.

**Problem Example**:
```python
# Bad: Same batch in both train and validation
KFold(n_splits=5)  # May split batch 1 across multiple folds
```

**Solution**:
```python
# Good: Each batch entirely in train OR validation
{"split": GroupKFold(n_splits=5), "group": "batch"}
```

### Syntax

```python
{"split": <Splitter>, "group": <metadata_column_name>}
```

**Parameters**:
- `split`: Any sklearn splitter that accepts `groups` parameter
- `group`: Name of metadata column containing group identifiers

### Examples

#### By Batch Number
```python
pipeline = [
    MinMaxScaler(),
    {"split": GroupKFold(n_splits=5), "group": "batch_number"},
    RandomForestRegressor()
]
```

#### By Patient ID (Medical Data)
```python
pipeline = [
    StandardScaler(),
    {"split": GroupShuffleSplit(n_splits=10, test_size=0.2), "group": "patient_id"},
    SVC()
]
```

#### By Collection Location
```python
pipeline = [
    SNV(),
    {"split": LeaveOneGroupOut(), "group": "site"},
    PLSRegression(n_components=15)
]
```

### Verifying No Leakage

After splitting, verify groups don't overlap:

```python
from nirs4all.data.dataset import SpectroDataset

dataset = SpectroDataset(...)
# ... add data and metadata ...

# Check folds
for fold_idx, (train_idx, val_idx) in enumerate(dataset._folds):
    train_groups = dataset.metadata_column("batch")[train_idx]
    val_groups = dataset.metadata_column("batch")[val_idx]

    overlap = set(train_groups) & set(val_groups)
    assert len(overlap) == 0, f"Fold {fold_idx} has group leakage: {overlap}"
    print(f"✅ Fold {fold_idx}: No group leakage")
```

### Default Behavior

If `group` is omitted, the first metadata column is used:

```python
# Uses first metadata column (e.g., column 0)
{"split": GroupKFold(n_splits=5)}

# Equivalent to:
{"split": GroupKFold(n_splits=5), "group": dataset.metadata_columns[0]}
```

**Warning**: Always shown when defaulting:
```
⚠️ GroupKFold requires groups but no 'group' specified. Using default: 'batch_id'
```

### Supported Splitters

Any sklearn splitter with `groups` parameter:

| Splitter | Description | Use Case |
|----------|-------------|----------|
| `GroupKFold` | K-fold with groups | General group-aware CV |
| `GroupShuffleSplit` | Random shuffle with groups | Fast approximate CV |
| `LeaveOneGroupOut` | Hold out each group once | Extreme group testing |
| `LeavePGroupsOut` | Hold out P groups | Robust group testing |
| `StratifiedGroupKFold` | Stratified + grouped | Imbalanced classification |

### Error Handling

**Missing Column**:
```python
{"split": GroupKFold(), "group": "nonexistent"}
# ❌ ValueError: Group column 'nonexistent' not found in metadata.
#    Available columns: ['batch', 'location', 'instrument']
```

**No Metadata**:
```python
# Dataset has no metadata
{"split": GroupKFold(), "group": "batch"}
# ❌ ValueError: Dataset has no metadata columns.
#    Please add metadata or use a non-grouped splitter.
```

**Wrong Type**:
```python
{"split": GroupKFold(), "group": 123}  # Not a string
# ❌ TypeError: Group column must be a string, got <class 'int'>
```
```

#### Step 2.3: Update PIPELINE_SYNTAX_COMPLETE_GUIDE.md (15 min)

Add to "All Supported Step Syntaxes" section:

```markdown
# 18. Split with Group (Group-Aware Cross-Validation)
{"split": GroupKFold(n_splits=5), "group": "batch_id"},

# 19. Split with default group
{"split": GroupShuffleSplit(n_splits=10), "group": "patient"},
```

---

### Phase 3: Integration Testing (Priority: Medium)

**Estimated Time**: 1 hour

#### Step 3.1: End-to-End Pipeline Test (30 min)
**File**: `examples/test_group_split_e2e.py`

```python
"""
End-to-end test for group-based cross-validation.
Demonstrates full pipeline with metadata and grouped splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from nirs4all.data.dataset import SpectroDataset
from nirs4all.data.dataset_config import DatasetConfigs
from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner

def test_full_pipeline_with_groups():
    """Test complete pipeline with grouped cross-validation."""

    # 1. Create dataset with metadata
    dataset = SpectroDataset(name="batch_test")

    # Simulate 100 samples across 5 batches
    X = np.random.rand(100, 50)
    y = np.random.rand(100)
    metadata = pd.DataFrame({
        'batch': [1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20,
        'location': ['A']*50 + ['B']*50
    })

    dataset.add_samples(X, {"partition": "train"})
    dataset.add_targets(y)
    dataset.add_metadata(metadata)

    # Add test set
    X_test = np.random.rand(20, 50)
    y_test = np.random.rand(20)
    dataset.add_samples(X_test, {"partition": "test"})
    dataset.add_targets(y_test)

    # 2. Define pipeline with group-aware splitting
    pipeline = [
        StandardScaler(),
        {"split": GroupKFold(n_splits=5), "group": "batch"},
        PLSRegression(n_components=10)
    ]

    # 3. Run pipeline
    pipeline_config = PipelineConfigs(pipeline, name="group_split_test")
    dataset_config = DatasetConfigs.from_dataset(dataset)

    runner = PipelineRunner(verbose=1, save_files=False)
    predictions, _ = runner.run(pipeline_config, dataset_config)

    # 4. Verify results
    assert predictions.num_predictions > 0
    print(f"✅ Pipeline completed with {predictions.num_predictions} predictions")

    # 5. Verify no group leakage
    for fold_idx, (train_idx, val_idx) in enumerate(dataset._folds):
        train_batches = set(dataset.metadata_column("batch")[train_idx])
        val_batches = set(dataset.metadata_column("batch")[val_idx])

        overlap = train_batches & val_batches
        assert len(overlap) == 0, f"Group leakage in fold {fold_idx}: {overlap}"
        print(f"✅ Fold {fold_idx}: train_batches={train_batches}, val_batches={val_batches}")

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_full_pipeline_with_groups()
```

**Run Test**:
```bash
python examples/test_group_split_e2e.py
```

#### Step 3.2: Serialization Roundtrip Test (30 min)
**File**: `tests/test_group_split_serialization.py`

```python
"""Test serialization/deserialization of group split configurations."""

import json
import tempfile
from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nirs4all.pipeline.config import PipelineConfigs
from nirs4all.pipeline.serialization import serialize_component, deserialize_component

def test_serialize_group_split():
    """Test serialization of group split step."""
    step = {"split": GroupKFold(n_splits=5), "group": "batch"}

    serialized = serialize_component(step, include_runtime=False)

    # Verify structure
    assert "split" in serialized
    assert "group" in serialized
    assert serialized["group"] == "batch"
    assert "GroupKFold" in serialized["split"]["class"]
    assert serialized["split"]["params"]["n_splits"] == 5

def test_roundtrip_json():
    """Test JSON save/load roundtrip."""
    pipeline = [
        {"split": GroupKFold(n_splits=5), "group": "batch_id"},
        {"split": GroupShuffleSplit(n_splits=10, test_size=0.2), "group": "location"}
    ]

    config = PipelineConfigs(pipeline, name="test_group")

    # Save to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "pipeline.json"
        with open(json_path, 'w') as f:
            json.dump(config.serializable_steps(config.steps[0]), f)

        # Load from JSON
        with open(json_path, 'r') as f:
            loaded = json.load(f)

        # Verify
        assert loaded[0]["group"] == "batch_id"
        assert loaded[1]["group"] == "location"
        assert "GroupKFold" in loaded[0]["split"]["class"]
        assert "GroupShuffleSplit" in loaded[1]["split"]["class"]

def test_backward_compatible_serialization():
    """Test that old format still works."""
    old_format = GroupKFold(n_splits=5)

    serialized = serialize_component(old_format, include_runtime=False)

    # Should serialize without error
    assert "GroupKFold" in serialized["class"]
    assert serialized["params"]["n_splits"] == 5

if __name__ == "__main__":
    test_serialize_group_split()
    test_roundtrip_json()
    test_backward_compatible_serialization()
    print("✅ All serialization tests passed!")
```

---

### Phase 4: Edge Cases & Polish (Priority: Low)

**Estimated Time**: 1 hour

#### Step 4.1: Handle Edge Cases (30 min)

**Scenarios to Test**:

1. **Empty metadata**: Dataset has no metadata
2. **Column doesn't exist**: Invalid column name
3. **Mismatched lengths**: Groups array doesn't match X
4. **Non-string group parameter**: `group` is not a string
5. **Multiple datasets**: Different metadata columns per dataset

**Test File**: `tests/test_group_split_edge_cases.py`

```python
def test_empty_metadata():
    """Test error when dataset has no metadata."""
    dataset = SpectroDataset(name="no_meta")
    dataset.add_samples(np.random.rand(50, 10), {"partition": "train"})
    dataset.add_targets(np.random.rand(50))

    step = {"split": GroupKFold(), "group": "batch"}
    controller = CrossValidatorController()

    with pytest.raises(ValueError, match="no metadata"):
        controller.execute(step, step["split"], dataset, {}, None)

def test_invalid_column_name():
    """Test error on non-existent column."""
    # ... setup dataset with metadata ...

    step = {"split": GroupKFold(), "group": "INVALID"}

    with pytest.raises(ValueError, match="not found in metadata"):
        controller.execute(step, step["split"], dataset, {}, None)

def test_non_string_group():
    """Test error on non-string group parameter."""
    step = {"split": GroupKFold(), "group": 123}

    with pytest.raises(TypeError, match="must be a string"):
        controller.execute(step, step["split"], dataset, {}, None)
```

#### Step 4.2: UI Integration (30 min)

**File**: `nirs4all_ui/public/component-library.json`

Update splitter components to support group parameter:

```json
{
  "subcategory": "splitting_strategies",
  "id": "group_kfold",
  "label": "Group K-Fold",
  "short": "GroupKFold",
  "description": "Group-aware K-fold cross-validation (GroupKFold). Requires metadata column for groups.",
  "defaults": {
    "n_splits": 5
  },
  "editable": [
    {
      "name": "n_splits",
      "type": "integer",
      "description": "Number of folds",
      "default": 5
    },
    {
      "name": "group",
      "type": "string",
      "description": "Metadata column name for groups (required)",
      "default": null,
      "placeholder": "e.g., batch, patient_id"
    }
  ]
}
```

---

## Testing Strategy

### Unit Tests (✅ Phase 1)
- Controller matching logic
- Execution with various group columns
- Error handling (missing column, no metadata, wrong type)
- Backward compatibility

### Integration Tests (✅ Phase 3)
- Full pipeline execution
- Serialization roundtrip
- Multiple datasets
- Prediction mode

### Edge Case Tests (✅ Phase 4)
- Empty metadata
- Invalid columns
- Type errors
- Multiple group strategies

### Manual Tests
1. Run example pipeline with real dataset
2. Verify fold CSV files include group info
3. Check prediction mode doesn't break
4. Test UI component (if applicable)

---

## Migration Guide

### For Existing Users

**No action required!** Existing pipelines continue to work:

```python
# Old syntax - still works
GroupKFold(n_splits=5)

# Old dict format - still works
{"class": "sklearn.model_selection.GroupKFold", "params": {"n_splits": 5}}
```

### For New Features

To use group-aware splitting:

```python
# New syntax
{"split": GroupKFold(n_splits=5), "group": "batch_id"}
```

### Updating Existing Pipelines

**Before**:
```python
pipeline = [
    StandardScaler(),
    GroupKFold(n_splits=5),  # ❌ Groups not specified
    PLSRegression(10)
]
```

**After**:
```python
pipeline = [
    StandardScaler(),
    {"split": GroupKFold(n_splits=5), "group": "batch"},  # ✅ Explicit groups
    PLSRegression(10)
]
```

---

## Future Enhancements

### Phase 5: Advanced Features (Optional)

1. **Multiple Group Columns**: Support hierarchical grouping
   ```python
   {"split": GroupKFold(), "group": ["batch", "location"]}
   ```

2. **Group Transformations**: Encode groups on-the-fly
   ```python
   {"split": GroupKFold(), "group": "date", "group_transform": "year"}
   ```

3. **Auto-Detect Groups**: Analyze metadata to suggest group columns
   ```python
   {"split": GroupKFold(), "group": "auto"}  # Uses best candidate
   ```

4. **Visualization**: Show group distribution across folds
   ```python
   {"split": GroupKFold(), "group": "batch", "visualize": True}
   ```

---

## Summary

### What's Being Added
✅ Support for `{"split": <Splitter>, "group": <column>}` syntax
✅ Automatic group extraction from metadata
✅ Proper serialization and prediction mode handling
✅ Comprehensive error messages and validation
✅ Full backward compatibility

### Changes Required
1. **Add** `"split"` to `WORKFLOW_OPERATORS` in `runner.py`
2. **Enhance** `CrossValidatorController.matches()` for keyword matching
3. **Enhance** `CrossValidatorController.execute()` for group extraction
4. **Add** unit tests, integration tests, edge case tests
5. **Update** documentation

### Estimated Total Time
- **Core Implementation**: 2-3 hours
- **Testing**: 1.5 hours
- **Documentation**: 1-2 hours
- **Polish**: 1 hour
- **Total**: 5.5-7.5 hours

### Risk Assessment
- **Low Risk**: Backward compatible, well-isolated changes
- **High Value**: Prevents data leakage, essential for proper CV
- **Well-Defined**: Clear requirements and existing patterns to follow

---

## Checklist for Implementation

### Phase 1: Core (Must Have)
- [ ] Add `"split"` to `WORKFLOW_OPERATORS`
- [ ] Update `CrossValidatorController.matches()`
- [ ] Update `CrossValidatorController.execute()`
- [ ] Add unit tests
- [ ] Verify backward compatibility

### Phase 2: Documentation (Must Have)
- [ ] Update `WRITING_A_PIPELINE.md`
- [ ] Update `METADATA_USAGE.md`
- [ ] Update `PIPELINE_SYNTAX_COMPLETE_GUIDE.md`
- [ ] Add code examples

### Phase 3: Integration (Should Have)
- [ ] End-to-end pipeline test
- [ ] Serialization roundtrip test
- [ ] Prediction mode test

### Phase 4: Polish (Nice to Have)
- [ ] Edge case tests
- [ ] UI component update
- [ ] Performance benchmarks

---

**Ready to implement!** All design decisions are made, patterns are established, and the roadmap is clear. 🚀


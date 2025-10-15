# Sample Augmentation Design Document

**Version:** 1.0
**Date:** October 15, 2025
**Author:** Architecture Analysis
**Status:** Awaiting Validation

---

## Executive Summary

This document proposes a comprehensive architecture for implementing **sample augmentation** in the nirs4all library. Unlike feature augmentation (which adds new processings to existing samples), sample augmentation creates new synthetic samples through data transformation, with support for both **standard** and **balanced** augmentation modes.

### Key Features
- **Dual Modes**: Standard (count-based) and Balanced (class-aware) augmentation
- **Leak Prevention**: Augmented samples follow their origin through splits
- **Seamless Integration**: Transparent behavior in x(), y(), metadata() calls
- **Flexible Configuration**: Support for random transformer selection and intelligent balancing

---

## 1. Current Architecture Analysis

### 1.1 Existing Data Flow

```
Pipeline Step → Controller → Dataset API → Indexer + Features + Targets
                                              ↓
                                        Context-based Selection
                                              ↓
                                    x(selector) → filtered data
```

### 1.2 Key Components

#### Indexer (nirs4all/dataset/indexer.py)
- **Current Fields**: `row`, `sample`, `origin`, `partition`, `group`, `branch`, `processings`, `augmentation`
- **Selection Methods**:
  - `x_indices(selector)` → returns `sample` IDs
  - `y_indices(selector)` → returns `origin` if not null, else `sample`
- **Augmentation Support**: `augment_rows()` method exists but incomplete

#### Features (nirs4all/dataset/features.py + feature_source.py)
- **Structure**: 3D array (samples, processings, features)
- **Augmentation Method**: `augment_samples()` duplicates samples and adds processing data
- **Multi-Source**: Handles list of feature sources

#### Dataset (nirs4all/dataset/dataset.py)
- **Facade Methods**: `x()`, `y()`, `add_samples()`, `augment_samples()`
- **Current `augment_samples()`**: Takes `data`, `processings`, `augmentation_id`, `selector`, `count`

### 1.3 Existing Sample Augmentation Controller

The current `SampleAugmentationController` (op_sample_augmentation.py):
- Iterates through transformer list
- Calls `dataset.augment_samples()` with count=1
- Applies transformers sequentially via `runner.run_steps()`
- **Limitation**: No balancing, fixed count, no random selection

---

## 2. Proposed Architecture

### 2.1 Core Principle: Origin Tracking

**Every augmented sample maintains a reference to its origin:**

```
Original Sample (sample=10, origin=null)
  ├─ Augmented #1 (sample=100, origin=10, augmentation="aug_0")
  ├─ Augmented #2 (sample=101, origin=10, augmentation="aug_0")
  └─ Augmented #3 (sample=102, origin=10, augmentation="aug_1")
```

### 2.2 Selection Behavior (Leak Prevention)

#### Two-Phase Selection for x(), y(), metadata()

**Phase 1: Base Selection**
```python
# Filter with selector + implicit origin=null filter
base_selector = {**selector, "origin": None}
base_samples = indexer.x_indices(base_selector)
```

**Phase 2: Augmented Aggregation**
```python
# Find all augmented versions of base samples
augmented_samples = indexer.get_augmented_for_origins(base_samples)
final_samples = np.concatenate([base_samples, augmented_samples])
```

**Example:**
```python
# User calls: dataset.x({"partition": "train"})
# Internally:
# 1. Get base samples: [1, 2, 3, 4, 5]
# 2. Find augmented: [100, 101] (origin=1), [102, 103] (origin=2), ...
# 3. Return: [1, 2, 3, 4, 5, 100, 101, 102, 103, ...]
```

#### Split Behavior
- Splits (KFold, ShuffleSplit, etc.) operate **only on base samples** (origin=null)
- Augmented samples automatically follow their origin's fold assignment
- **Implementation**: Split controller filters with `origin=None` before calling `splitter.split()`

### 2.3 Augmentation Modes

#### Mode 1: Standard Augmentation (Count-Based)

**Configuration:**
```python
{
    "sample_augmentation": {
        "transformers": [SavGol(), Gaussian(), SNV()],
        "count": 5,  # Generate 5 augmented samples per original sample
        "selection": "random"  # or "all" to apply all transformers
    }
}
```

**Behavior:**
- For each sample in train partition
- Generate `count` augmented versions
- If `selection="random"`: randomly pick transformer for each augmentation
- If `selection="all"`: cycle through all transformers

#### Mode 2: Balanced Augmentation (Class-Aware)

**Configuration:**
```python
{
    "sample_augmentation": {
        "transformers": [SavGol(), Gaussian(), SNV()],
        "balance": "y",  # or metadata column name like "group_id"
        "max_factor": 0.8,  # Target 80% of majority class size
        "selection": "random"
    }
}
```

**Balancing Algorithm:**
```python
# Example: Class distribution
majority_class_count = 100
minority_class_count = 20
max_factor = 0.8

# Target count for minority class
target_count = int(majority_class_count * max_factor)  # 80

# Augmentations needed
augmentations_needed = max(0, target_count - minority_class_count)  # 60

# Per-sample augmentation count (distributed evenly)
aug_per_sample = augmentations_needed // minority_class_count  # 3
remainder = augmentations_needed % minority_class_count  # 0
```

**Handling Multiple Underrepresented Classes:**
- Calculate target for each class relative to majority
- Generate augmentations independently per class
- Support both classification (y) and regression (metadata groups)

### 2.4 Delegation Pattern (Like Feature Augmentation)

**Key Principle:** `SampleAugmentationController` ONLY orchestrates, `TransformerMixinController` does ALL the work.

**Workflow:**
1. **SampleAugmentationController** (orchestration only):
   - Calculates augmentation plan (standard or balanced)
   - Creates transformer → indices mapping (e.g., {SavGol: [1,3,7], Gaussian: [2,5,8]})
   - For each transformer, emits ONE context via `runner.run_step()` with the target indices list
   - **Does NOT pre-allocate samples** - that's TransformerMixin's job

2. **TransformerMixinController** (execution):
   - Detects `augment_sample` action in context (like `add_feature` for feature augmentation)
   - For each sample index in the provided list:
     - Retrieves origin sample data (all sources)
     - Applies transformer to all sources
     - Calls `dataset.add_samples()` with transformed data (equivalent to `augment_samples` with count=1)
   - Handles binary serialization

**Benefits:**
- ✅ **Separation of Concerns**: Controller = plan distribution, TransformerMixin = execute augmentation
- ✅ **Multi-Source**: Automatically handles all sources via TransformerMixin
- ✅ **Serialization**: Binary saving/loading handled by existing logic
- ✅ **Parallelization Ready**: Can use `run_steps()` in future for parallel execution
- ✅ **Consistency**: Follows EXACT same pattern as `FeatureAugmentationController`
- ✅ **No Pre-allocation**: Samples created only when transformed (cleaner, more efficient)

---

## 3. Implementation Plan

### 3.1 Phase 1: Indexer Enhancements

**File:** `nirs4all/dataset/indexer.py`

#### New Method: `get_augmented_for_origins()`

```python
def get_augmented_for_origins(self, origin_samples: List[int]) -> np.ndarray:
    """
    Get all augmented samples for given origin sample IDs.

    Args:
        origin_samples: List of origin sample IDs

    Returns:
        Array of augmented sample IDs
    """
    if not origin_samples:
        return np.array([], dtype=np.int32)

    filter_condition = pl.col("origin").is_in(origin_samples)
    augmented_df = self.df.filter(filter_condition)
    return augmented_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)
```

#### Modified Method: `x_indices()` - Add Two-Phase Logic

```python
def x_indices(self, selector: Selector, include_augmented: bool = True) -> np.ndarray:
    """
    Get sample indices with optional augmented sample aggregation.

    Args:
        selector: Filter criteria
        include_augmented: If True, include augmented versions of selected samples

    Returns:
        Array of sample indices
    """
    # Phase 1: Get base samples (origin=null)
    base_selector = selector.copy() if selector else {}
    if include_augmented and "origin" not in base_selector:
        base_selector["origin"] = None  # Force origin filter

    filtered_df = self._apply_filters(base_selector) if base_selector else self.df.filter(pl.col("origin").is_null())
    base_indices = filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)

    # Phase 2: Get augmented samples if requested
    if include_augmented and len(base_indices) > 0:
        augmented_indices = self.get_augmented_for_origins(base_indices.tolist())
        return np.concatenate([base_indices, augmented_indices])

    return base_indices
```

#### Keep Existing: `y_indices()`
- Already handles origin → sample mapping correctly
- No changes needed

### 3.2 Phase 2: Dataset API Enhancement

**File:** `nirs4all/dataset/dataset.py`

#### Modified Method: `x()`

```python
def x(self, selector: Selector, layout: Layout = "2d", concat_source: bool = True, include_augmented: bool = True) -> OutputData:
    """Get feature data with automatic augmented sample aggregation."""
    indices = self._indexer.x_indices(selector, include_augmented=include_augmented)
    return self._features.x(indices, layout, concat_source)
```

#### Modified Method: `y()`

```python
def y(self, selector: Selector, include_augmented: bool = True) -> np.ndarray:
    """Get target data - automatically maps augmented → origin for y values."""
    if include_augmented:
        # Get base + augmented samples
        x_indices = self._indexer.x_indices(selector, include_augmented=True)
        # Map to y indices (augmented → origin)
        y_indices = []
        for sample_id in x_indices:
            origin_id = self._indexer.get_origin_for_sample(sample_id)
            y_indices.append(origin_id if origin_id is not None else sample_id)
        y_indices = np.array(y_indices, dtype=np.int32)
    else:
        y_indices = self._indexer.y_indices(selector)

    processing = selector.get("y") if selector and "y" in selector else "numeric"
    return self._targets.y(y_indices, processing)
```

#### New Helper in Indexer:

```python
def get_origin_for_sample(self, sample_id: int) -> Optional[int]:
    """Get origin sample ID for a given sample, or None if it's not augmented."""
    row = self.df.filter(pl.col("sample") == sample_id)
    if len(row) == 0:
        return None
    origin = row.select(pl.col("origin")).item()
    return origin if origin is not None else sample_id
```

### 3.3 Phase 3: Balancing Utility

**New File:** `nirs4all/utils/balancing.py`

```python
from typing import List, Dict, Union, Tuple
import numpy as np
from collections import Counter

class BalancingCalculator:
    """Calculate augmentation counts for balanced datasets."""

    @staticmethod
    def calculate_balanced_counts(
        labels: np.ndarray,
        sample_indices: np.ndarray,
        max_factor: float = 1.0
    ) -> Dict[int, int]:
        """
        Calculate how many augmentations needed per sample for balancing.

        Args:
            labels: Class labels or group IDs (1D array)
            sample_indices: Corresponding sample IDs
            max_factor: Target fraction of majority class (0.0-1.0)

        Returns:
            Dictionary mapping sample_id → augmentation_count
        """
        # Count samples per class
        label_to_samples = {}
        for sample_id, label in zip(sample_indices, labels):
            label_to_samples.setdefault(label, []).append(sample_id)

        # Find majority class size
        class_sizes = {label: len(samples) for label, samples in label_to_samples.items()}
        majority_size = max(class_sizes.values())

        # Calculate augmentations per class
        augmentation_map = {}

        for label, samples in label_to_samples.items():
            current_size = len(samples)
            target_size = int(majority_size * max_factor)

            if current_size >= target_size:
                # Already balanced or majority class
                for sample_id in samples:
                    augmentation_map[sample_id] = 0
            else:
                # Needs augmentation
                total_needed = target_size - current_size
                base_count = total_needed // current_size
                remainder = total_needed % current_size

                for i, sample_id in enumerate(samples):
                    # Distribute remainder to first samples
                    augmentation_map[sample_id] = base_count + (1 if i < remainder else 0)

        return augmentation_map

    @staticmethod
    def apply_random_transformer_selection(
        transformers: List,
        augmentation_counts: Dict[int, int],
        random_state: Optional[int] = None
    ) -> Dict[int, List[int]]:
        """
        Randomly select transformers for each augmentation.

        Args:
            transformers: List of transformer instances
            augmentation_counts: sample_id → number of augmentations
            random_state: Random seed

        Returns:
            Dictionary mapping sample_id → list of transformer indices
        """
        rng = np.random.default_rng(random_state)
        transformer_selection = {}

        for sample_id, count in augmentation_counts.items():
            if count > 0:
                # Randomly select transformer indices
                selected = rng.integers(0, len(transformers), size=count).tolist()
                transformer_selection[sample_id] = selected
            else:
                transformer_selection[sample_id] = []

        return transformer_selection
```

### 3.4 Phase 4: Controller Rewrite (Delegation Pattern)

**File:** `nirs4all/controllers/dataset/op_sample_augmentation.py`

**Key Principle:** The controller ONLY creates the distribution map and emits contexts. TransformerMixinController does the actual augmentation via `add_samples`.

```python
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import copy

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller
from nirs4all.utils.balancing import BalancingCalculator

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


@register_controller
class SampleAugmentationController(OperatorController):
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "sample_augmentation"

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Sample augmentation only runs during training."""
        return False

    def execute(
        self,
        step: Any,
        operator: Any,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[Any] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List]:
        """
        Execute sample augmentation with standard or balanced mode.

        Step format:
            {
                "sample_augmentation": {
                    "transformers": [transformer1, transformer2, ...],
                    "count": int  # Standard mode
                    # OR
                    "balance": "y" or "metadata_column",  # Balanced mode
                    "max_factor": float,  # Optional, default 1.0
                    "selection": "random" or "all",  # Default "random"
                    "random_state": int  # Optional
                }
            }
        """
        config = step["sample_augmentation"]
        transformers = config.get("transformers", [])

        if not transformers:
            raise ValueError("sample_augmentation requires at least one transformer")

        # Determine mode
        is_balanced = "balance" in config

        if is_balanced:
            return self._execute_balanced(config, transformers, dataset, context, runner, loaded_binaries)
        else:
            return self._execute_standard(config, transformers, dataset, context, runner, loaded_binaries)

    def _execute_standard(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ) -> Tuple[Dict[str, Any], List]:
        """Execute standard count-based augmentation."""
        count = config.get("count", 1)
        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)

        # Get train samples (origin=null implied)
        train_context = copy.deepcopy(context)
        train_context["partition"] = "train"

        # Get base samples only (no augmented)
        base_samples = dataset._indexer.x_indices(train_context, include_augmented=False).tolist()

        if not base_samples:
            return context, []

        # Create augmentation plan: sample_id → number of augmentations
        augmentation_counts = {sample_id: count for sample_id in base_samples}

        # Build transformer distribution: sample_id → list of transformer indices
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:  # "all"
            transformer_map = self._cycle_transformers(transformers, augmentation_counts)

        # Invert map: transformer_idx → list of sample_ids
        transformer_to_samples = self._invert_transformer_map(transformer_map, len(transformers))

        # Emit ONE run_step per transformer
        self._emit_augmentation_steps(
            transformer_to_samples, transformers, context, runner, loaded_binaries
        )

        return context, []

    def _execute_balanced(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ) -> Tuple[Dict[str, Any], List]:
        """Execute balanced class-aware augmentation."""
        balance_source = config.get("balance")
        max_factor = config.get("max_factor", 1.0)
        selection = config.get("selection", "random")
        random_state = config.get("random_state", None)

        # Get train samples
        train_context = copy.deepcopy(context)
        train_context["partition"] = "train"

        base_samples = dataset._indexer.x_indices(train_context, include_augmented=False)

        if len(base_samples) == 0:
            return context, []

        # Get labels for balancing
        if balance_source == "y":
            labels = dataset.y(train_context, include_augmented=False)
        else:
            # Metadata column
            labels = dataset.metadata_column(balance_source, train_context)

        # Calculate augmentation counts per sample
        augmentation_counts = BalancingCalculator.calculate_balanced_counts(
            labels, base_samples, max_factor
        )

        # Build transformer distribution
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:
            transformer_map = self._cycle_transformers(transformers, augmentation_counts)

        # Invert map: transformer_idx → list of sample_ids
        transformer_to_samples = self._invert_transformer_map(transformer_map, len(transformers))

        # Emit ONE run_step per transformer
        self._emit_augmentation_steps(
            transformer_to_samples, transformers, context, runner, loaded_binaries
        )

        return context, []

    def _invert_transformer_map(
        self,
        transformer_map: Dict[int, List[int]],
        n_transformers: int
    ) -> Dict[int, List[int]]:
        """
        Invert sample→transformer map to transformer→samples map.

        Args:
            transformer_map: {sample_id: [trans_idx1, trans_idx2, ...]}
            n_transformers: Total number of transformers

        Returns:
            {trans_idx: [sample_id1, sample_id2, ...]}
        """
        inverted = {i: [] for i in range(n_transformers)}

        for sample_id, trans_indices in transformer_map.items():
            for trans_idx in trans_indices:
                inverted[trans_idx].append(sample_id)

        return inverted

    def _emit_augmentation_steps(
        self,
        transformer_to_samples: Dict[int, List[int]],
        transformers: List,
        context: Dict[str, Any],
        runner: 'PipelineRunner',
        loaded_binaries: Optional[Any]
    ):
        """
        Emit ONE run_step per transformer with the list of target sample indices.

        TransformerMixinController will:
        1. Detect augment_sample action
        2. For each sample in the list:
           - Get origin data (all sources)
           - Transform it
           - Call dataset.add_samples() (augment_samples with count=1)
        """
        for trans_idx, sample_ids in transformer_to_samples.items():
            if not sample_ids:
                continue

            transformer = transformers[trans_idx]

            # Create context for this transformer's augmentation
            local_context = copy.deepcopy(context)
            local_context["augment_sample"] = True  # Signal action (like add_feature)
            local_context["target_samples"] = sample_ids  # Indices to augment
            local_context["partition"] = "train"

            # ONE run_step per transformer - it handles all target samples
            runner.run_step(
                transformer,
                dataset,
                local_context,
                prediction_store=None,
                is_substep=True,
                propagated_binaries=loaded_binaries
            )

    def _cycle_transformers(
        self,
        transformers: List,
        augmentation_counts: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """Cycle through transformers for 'all' selection mode."""
        transformer_map = {}
        for sample_id, count in augmentation_counts.items():
            if count > 0:
                transformer_map[sample_id] = [i % len(transformers) for i in range(count)]
            else:
                transformer_map[sample_id] = []
        return transformer_map
```### 3.5 Phase 5: TransformerMixinController Enhancement

**File:** `nirs4all/controllers/sklearn/op_transformermixin.py`

**Add sample augmentation detection (like add_feature detection):**

```python
def execute(
    self,
    step: Any,
    operator: Any,
    dataset: 'SpectroDataset',
    context: Dict[str, Any],
    runner: 'PipelineRunner',
    source: int = -1,
    mode: str = "train",
    loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
    prediction_store: Optional[Any] = None
):
    """Execute transformer - handles normal, feature augmentation, and sample augmentation modes."""

    # Check if we're in sample augmentation mode
    if context.get("augment_sample", False):
        return self._execute_for_sample_augmentation(
            operator, dataset, context, runner, mode, loaded_binaries, prediction_store
        )

    # Normal or feature augmentation execution (existing code)
    # ... existing code ...

def _execute_for_sample_augmentation(
    self,
    operator: Any,
    dataset: 'SpectroDataset',
    context: Dict[str, Any],
    runner: 'PipelineRunner',
    mode: str,
    loaded_binaries: Optional[List[Tuple[str, Any]]],
    prediction_store: Optional[Any]
) -> Tuple[Dict[str, Any], List]:
    """
    Apply transformer to origin samples and add augmented samples.

    Context contains:
        - augment_sample: True flag (like add_feature)
        - target_samples: list of sample_ids to augment
        - partition: "train" (filtering context)
    """
    target_sample_ids = context.get("target_samples", [])
    if not target_sample_ids:
        return context, []

    operator_name = operator.__class__.__name__
    fitted_transformers = []

    # Process each target sample
    for sample_id in target_sample_ids:
        # Get origin sample data (all sources)
        origin_selector = {"sample": [sample_id]}
        origin_data = dataset.x(origin_selector, "3d", concat_source=False, include_augmented=False)

        # Ensure list format for multi-source
        if not isinstance(origin_data, list):
            origin_data = [origin_data]

        # Transform each source
        transformed_sources = []

        for source_idx, source_data in enumerate(origin_data):
            # source_data shape: (1, n_processings, n_features) for single sample
            source_2d_list = []

            for proc_idx in range(source_data.shape[1]):
                proc_data = source_data[:, proc_idx, :]  # (1, n_features)

                # Apply transformer
                if loaded_binaries and mode in ["predict", "explain"]:
                    transformer = dict(loaded_binaries).get(f"{operator_name}_{source_idx}_{proc_idx}_{sample_id}")
                    if transformer is None:
                        raise ValueError(f"Binary for {operator_name} not found")
                else:
                    transformer = clone(operator)
                    transformer.fit(proc_data)

                transformed_data = transformer.transform(proc_data)
                source_2d_list.append(transformed_data)

                # Save transformer binary (per sample, per source, per processing)
                import pickle
                transformer_binary = pickle.dumps(transformer)
                fitted_transformers.append(
                    (f"{operator_name}_{source_idx}_{proc_idx}_{sample_id}.pkl", transformer_binary)
                )

            # Stack back to 3D (1, n_processings, n_features)
            source_3d = np.stack(source_2d_list, axis=1)
            transformed_sources.append(source_3d)

        # Add augmented sample to dataset (equivalent to augment_samples with count=1)
        # dataset.add_samples handles:
        # - Indexer: creates row with origin=sample_id
        # - Features: adds sample to all sources
        # - Metadata: duplicates from origin
        augmentation_id = f"aug_{operator_name}_{sample_id}"

        dataset.add_samples(
            data=transformed_sources,  # List of 3D arrays (one per source)
            origin=sample_id,
            augmentation=augmentation_id,
            partition="train",
            # Metadata automatically copied from origin sample
        )

    return context, fitted_transformers
```

**Note:** Assumes `dataset.add_samples()` method exists with signature:
```python
def add_samples(
    self,
    data: List[np.ndarray],  # List of 3D arrays (samples, processings, features)
    origin: int,  # Origin sample ID
    augmentation: str,  # Augmentation identifier
    partition: str = "train",
    **metadata_overrides
) -> None:
    """Add new samples to dataset with proper indexer and features updates."""
```

If `add_samples` doesn't exist, can use existing `augment_samples()` method:
```python
dataset.augment_samples(
    data=transformed_sources,
    processings=dataset.features_processings(0),  # Get existing processings
    augmentation_id=augmentation_id,
    selector={"sample": [sample_id]},
    count=1
)
```

### 3.6 Phase 6: Split Controller Modification

**File:** `nirs4all/controllers/sklearn/op_split.py`

**Modification:**

```python
# In execute() method, before calling operator.split():

# Filter to base samples only (origin=null) for splitting
local_context = copy.deepcopy(context)
local_context["partition"] = "train"

# IMPORTANT: Only split on base samples
X = dataset.x(local_context, layout="2d", concat_source=True, include_augmented=False)
y = dataset.y(local_context, include_augmented=False) if needs_y else None
groups = # ... (same logic, but with include_augmented=False if applicable)
```

This ensures splits operate only on original samples, preventing leakage.

---

## 4. Data Flow Examples

### 4.1 Standard Augmentation Flow (with Delegation)

```python
# Pipeline
pipeline = [
    MinMaxScaler(),
    {
        "sample_augmentation": {
            "transformers": [SavGol(), Gaussian(), SNV()],
            "count": 3,
            "selection": "random",
            "random_state": 42
        }
    },
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": PLSRegression(n_components=10)}
]

# Execution:
# 1. MinMaxScaler applies to all samples (via TransformerMixinController)

# 2. SampleAugmentationController:
#    a. Get train samples: base_samples = [1, 2, 3, ..., 100]
#    b. Calculate plan: each sample gets 3 augmentations
#    c. Random transformer selection:
#       sample_1 → [SavGol_idx, Gaussian_idx, SavGol_idx]  (indices: 0, 1, 0)
#       sample_2 → [SNV_idx, SavGol_idx, Gaussian_idx]     (indices: 2, 0, 1)
#       ...
#    d. Invert to transformer→samples map:
#       SavGol    → [1, 1, 2, 5, 7, ...]  (150 samples total)
#       Gaussian  → [1, 2, 3, 8, ...]     (100 samples)
#       SNV       → [2, 4, 6, ...]        (50 samples)
#    e. Emit 3 run_step calls (ONE per transformer):
#       runner.run_step(SavGol, dataset, context={augment_sample=True, target_samples=[1,1,2,5,7,...]})
#       runner.run_step(Gaussian, dataset, context={augment_sample=True, target_samples=[1,2,3,8,...]})
#       runner.run_step(SNV, dataset, context={augment_sample=True, target_samples=[2,4,6,...]})

# 3. TransformerMixinController (per run_step):
#    - Detect augment_sample=True
#    - For each sample in target_samples list:
#      * Get origin data (all sources)
#      * Apply transformer to all sources
#      * Call dataset.add_samples(data, origin=sample_id, augmentation=aug_id)
#    - Result: 150 augmented samples from SavGol, 100 from Gaussian, 50 from SNV

# 4. ShuffleSplit:
#    - Operates on base samples only: [1-100]
#    - Creates folds: fold1: train=[1,2,5,...], val=[3,4,6,...]

# 5. Model training (fold 1):
#    - dataset.x({"partition": "train"}) returns:
#      base samples [1,2,5,...] + their augmented versions (auto-aggregated)
#    - Prevents leakage: val samples' augmented versions stay out
```

### 4.2 Balanced Augmentation Flow (with Delegation)

```python
# Dataset: 100 samples, class 0: 80 samples, class 1: 20 samples

pipeline = [
    {
        "sample_augmentation": {
            "transformers": [SavGol(), Gaussian()],
            "balance": "y",
            "max_factor": 0.9,  # Target 90% of majority
            "selection": "random"
        }
    },
    ShuffleSplit(n_splits=3, test_size=0.25),
    {"model": RandomForestClassifier()}
]

# Execution:
# 1. SampleAugmentationController:
#    a. Get labels: [0]*80 + [1]*20
#    b. Calculate:
#       - Majority size: 80
#       - Target for class 1: int(80 * 0.9) = 72
#       - Augmentations needed: 72 - 20 = 52
#       - Per sample (class 1): 52 // 20 = 2, remainder = 12
#    c. Augmentation plan:
#       - First 12 class-1 samples: 3 augmentations each
#       - Remaining 8 class-1 samples: 2 augmentations each
#       - Class 0 samples: 0 augmentations (already majority)
#    d. Random transformer selection → build transformer→samples map:
#       SavGol   → [class_1_sample_ids] (26 augmentations)
#       Gaussian → [class_1_sample_ids] (26 augmentations)
#    e. Emit 2 run_step calls:
#       runner.run_step(SavGol, dataset, context={augment_sample=True, target_samples=[...]})
#       runner.run_step(Gaussian, dataset, context={augment_sample=True, target_samples=[...]})

# 2. TransformerMixinController:
#    - Process each run_step call
#    - For each sample: transform origin + add augmented sample
#    - Total 52 new samples created

# 3. Final distribution:
#    - Class 0: 80 samples (unchanged)
#    - Class 1: 20 + 52 = 72 samples
#    - Ratio: 80:72 (balanced to ~90%)

# 4. ShuffleSplit operates on base samples only (leak-free)
```

### 4.3 Multi-Source Augmentation Flow

```python
# Dataset: 2 sources, 50 samples

pipeline = [
    {
        "sample_augmentation": {
            "transformers": [SavGol(), Gaussian()],
            "count": 2
        }
    },
    # ...
]

# Execution:
# 1. SampleAugmentationController:
#    - Plan: 50 samples × 2 augmentations = 100 augmentations
#    - Random selection → transformer→samples map:
#      SavGol   → [1, 2, 5, 7, ...] (50 samples)
#      Gaussian → [3, 4, 6, 8, ...] (50 samples)
#    - Emit 2 run_step calls

# 2. TransformerMixinController (per run_step):
#    - Detect multi-source: origin_data = [source0_data, source1_data]
#    - For each sample in target_samples:
#      * Transform source0 → transformed_source0
#      * Transform source1 → transformed_source1
#      * add_samples([transformed_source0, transformed_source1], origin=sample_id)

# 3. Result: 100 augmented samples with transformed data in BOTH sources
```

---

## 5. API Changes Summary

### 5.1 User-Facing API (Backward Compatible)

**Existing methods maintain default behavior (include augmented):**

```python
# All return base + augmented samples by default
dataset.x({"partition": "train"})
dataset.y({"partition": "train"})
dataset.metadata({"partition": "train"})
```

**New optional parameter for advanced use:**

```python
# Explicitly exclude augmented samples
dataset.x({"partition": "train"}, include_augmented=False)
```

### 5.2 Indexer API

**New Methods:**
- `get_augmented_for_origins(origin_samples) -> np.ndarray`
- `get_origin_for_sample(sample_id) -> Optional[int]`

**Modified Methods:**
- `x_indices(selector, include_augmented=True) -> np.ndarray`

### 5.3 Controller Configuration

**Sample Augmentation Step:**

```python
# Standard Mode
{
    "sample_augmentation": {
        "transformers": [transformer1, transformer2],
        "count": int,
        "selection": "random" | "all",  # optional
        "random_state": int  # optional
    }
}

# Balanced Mode
{
    "sample_augmentation": {
        "transformers": [transformer1, transformer2],
        "balance": "y" | "metadata_column",
        "max_factor": float,  # optional, default 1.0
        "selection": "random" | "all",  # optional
        "random_state": int  # optional
    }
}
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Indexer Tests** (`tests/unit/test_indexer.py`)
- `test_get_augmented_for_origins()`: Verify augmented sample retrieval
- `test_x_indices_with_augmented()`: Two-phase selection
- `test_x_indices_without_augmented()`: Base-only selection
- `test_get_origin_for_sample()`: Origin lookup

**Balancing Tests** (`tests/unit/test_balancing.py`)
- `test_calculate_balanced_counts_binary()`: 2-class balancing
- `test_calculate_balanced_counts_multiclass()`: 3+ classes
- `test_max_factor_behavior()`: Different max_factor values
- `test_already_balanced()`: No augmentation needed
- `test_random_transformer_selection()`: Randomness and seed

**Controller Tests** (`tests/unit/test_sample_augmentation_controller.py`)
- `test_standard_mode()`: Count-based augmentation
- `test_balanced_mode()`: Class-aware augmentation
- `test_random_selection()`: Random transformer picking
- `test_all_selection()`: Cycle through all transformers

### 6.2 Integration Tests

**End-to-End Pipeline** (`tests/integration/test_sample_augmentation.py`)
- `test_full_pipeline_with_splits()`: Verify no leakage across folds
- `test_augmented_samples_in_x()`: Data retrieval includes augmented
- `test_balanced_pipeline()`: Complete balanced workflow
- `test_multi_source_augmentation()`: Multi-source compatibility

### 6.3 Validation Scenarios

1. **Leak Prevention**: Augmented samples never appear in opposite fold
2. **Balancing Correctness**: Class distribution matches max_factor target
3. **Multi-Source**: Works with datasets having multiple sources
4. **Metadata Groups**: Balancing on metadata columns (not just y)
5. **Backward Compatibility**: Existing pipelines without augmentation unchanged

---

## 7. Implementation Roadmap

### Phase 1: Foundation - BalancingCalculator (Week 1, Days 1-2)
**Goal:** Create the utility class for balancing calculations and transformer distribution.

**Tasks:**
- [ ] Create `nirs4all/utils/balancing.py`
- [ ] Implement `BalancingCalculator.calculate_balanced_counts(labels, sample_indices, max_factor)`
  - Binary classification logic
  - Multi-class support
  - Handle edge cases (already balanced, impossible targets)
- [ ] Implement `BalancingCalculator.apply_random_transformer_selection(transformers, augmentation_counts, random_state)`
  - Random transformer assignment per augmentation
  - Seed support for reproducibility
- [ ] Write comprehensive unit tests:
  - `test_binary_balancing()` - 2-class scenarios
  - `test_multiclass_balancing()` - 3+ classes
  - `test_max_factor_variants()` - Different max_factor values (0.5, 0.8, 1.0)
  - `test_already_balanced()` - No augmentation needed cases
  - `test_random_selection_deterministic()` - Seed reproducibility
  - `test_random_selection_distribution()` - Fair transformer distribution

**Deliverables:**
- `nirs4all/utils/balancing.py` (~150 lines)
- `tests/unit/test_balancing.py` (~300 lines)

---

### Phase 2: Indexer Enhancement (Week 1, Days 3-4)
**Goal:** Add methods for augmented sample management and two-phase selection.

**Tasks:**
- [ ] Add `Indexer.get_augmented_for_origins(origin_samples)` method
  - Query augmented samples by origin field
  - Return numpy array of sample IDs
  - Handle empty cases gracefully
- [ ] Add `Indexer.get_origin_for_sample(sample_id)` method
  - Lookup origin for a given sample
  - Return None if sample is not augmented (is itself an origin)
- [ ] Modify `Indexer.x_indices(selector, include_augmented=True)`
  - Add `include_augmented` parameter (default=True for backward compatibility)
  - Implement two-phase selection:
    1. Filter base samples (origin=null or explicit origin filter)
    2. If include_augmented, aggregate augmented versions
  - Optimize for performance (minimize DataFrame operations)
- [ ] Write unit tests:
  - `test_get_augmented_for_origins()` - Various origin lists
  - `test_get_augmented_for_empty_origins()` - Empty input handling
  - `test_get_origin_for_sample()` - Origin lookup
  - `test_x_indices_with_augmented()` - Two-phase behavior
  - `test_x_indices_without_augmented()` - Base-only selection
  - `test_x_indices_backward_compatibility()` - Existing selectors unchanged

**Deliverables:**
- Modified `nirs4all/dataset/indexer.py` (+80 lines)
- `tests/unit/test_indexer_augmentation.py` (~250 lines)

---

### Phase 3: Dataset API Enhancement (Week 1-2, Days 5-7)
**Goal:** Propagate `include_augmented` parameter through Dataset API.

**Tasks:**
- [ ] Modify `Dataset.x(selector, layout, concat_source, include_augmented=True)`
  - Pass `include_augmented` to indexer.x_indices()
  - Maintain existing behavior by default
  - Document parameter in docstring
- [ ] Modify `Dataset.y(selector, include_augmented=True)`
  - Implement augmented→origin mapping for y values
  - Use `get_origin_for_sample()` for mapping
  - Ensure y values match x sample order
- [ ] Modify `Dataset.metadata(selector, include_augmented=True)`
  - Similar behavior to x() and y()
  - Propagate parameter to indexer
- [ ] Add helper method `Dataset.add_samples(data, origin, augmentation, partition, **metadata)`
  - Add samples to indexer (with origin field)
  - Add features to all sources
  - Duplicate metadata from origin sample
  - Alternative: verify existing `augment_samples()` can be used
- [ ] Write integration tests:
  - `test_x_includes_augmented_by_default()` - Backward compatibility
  - `test_x_excludes_augmented_when_false()` - Explicit filtering
  - `test_y_maps_augmented_to_origin()` - Y value consistency
  - `test_metadata_consistency()` - Metadata propagation
  - `test_add_samples_multi_source()` - Multi-source sample addition

**Deliverables:**
- Modified `nirs4all/dataset/dataset.py` (+100 lines)
- `tests/integration/test_dataset_augmentation.py` (~300 lines)

---

### Phase 4: Sample Augmentation Controller (Week 2, Days 8-10)
**Goal:** Rewrite controller with delegation pattern (NO pre-allocation, only distribution map + run_step).

**Tasks:**
- [ ] Rewrite `SampleAugmentationController.execute()`
  - Detect standard vs. balanced mode
  - Route to `_execute_standard()` or `_execute_balanced()`
- [ ] Implement `_execute_standard()`
  - Get base train samples
  - Build augmentation counts map (sample_id → count)
  - Apply random or cycle transformer selection
  - Invert to transformer→samples map
  - Emit ONE run_step per transformer with target_samples list
- [ ] Implement `_execute_balanced()`
  - Get base train samples + labels
  - Use BalancingCalculator to compute augmentation counts
  - Apply transformer selection
  - Invert to transformer→samples map
  - Emit ONE run_step per transformer with target_samples list
- [ ] Implement `_invert_transformer_map(sample_to_trans, n_transformers)`
  - Convert {sample_id: [trans_idx]} to {trans_idx: [sample_id]}
- [ ] Implement `_emit_augmentation_steps(transformer_to_samples, transformers, context, runner, loaded_binaries)`
  - For each transformer with target samples:
    * Create local context with augment_sample=True, target_samples=list
    * Call runner.run_step(transformer, dataset, context, is_substep=True, propagated_binaries=...)
- [ ] Implement `_cycle_transformers()` helper for "all" selection mode
- [ ] Write unit tests:
  - `test_standard_mode_count()` - Count-based augmentation plan
  - `test_balanced_mode_calculation()` - Balancing logic integration
  - `test_random_selection()` - Random transformer assignment
  - `test_all_selection()` - Cycle through all transformers
  - `test_invert_transformer_map()` - Map inversion correctness
  - `test_emit_calls_run_step()` - Verify run_step calls

**Deliverables:**
- Rewritten `nirs4all/controllers/dataset/op_sample_augmentation.py` (~250 lines)
- `tests/unit/test_sample_augmentation_controller.py` (~400 lines)

---

### Phase 5: TransformerMixin Enhancement (Week 2-3, Days 11-14)
**Goal:** Add augment_sample action detection and execution (like add_feature).

**Tasks:**
- [ ] Modify `TransformerMixinController.execute()`
  - Add detection: `if context.get("augment_sample", False):`
  - Route to `_execute_for_sample_augmentation()`
- [ ] Implement `_execute_for_sample_augmentation()`
  - Extract target_samples from context
  - For each sample_id in target_samples:
    * Get origin data: `dataset.x({"sample": [sample_id]}, "3d", concat_source=False, include_augmented=False)`
    * Ensure multi-source list format
    * For each source:
      - For each processing:
        * Fit transformer on origin data
        * Transform origin data
        * Save binary (per sample, per source, per processing)
      - Stack back to 3D
    * Call `dataset.add_samples(data=transformed_sources, origin=sample_id, augmentation=aug_id, partition="train")`
- [ ] Handle serialization:
  - Binary naming: `{operator_name}_{source_idx}_{proc_idx}_{sample_id}.pkl`
  - Load from propagated_binaries if in predict/explain mode
- [ ] Write unit tests:
  - `test_detect_augment_sample_action()` - Action detection
  - `test_single_sample_augmentation()` - Single sample transformation
  - `test_multi_sample_augmentation()` - Multiple samples
  - `test_multi_source_augmentation()` - All sources transformed
  - `test_binary_serialization()` - Save/load transformers
  - `test_add_samples_called()` - Verify dataset.add_samples() calls

**Deliverables:**
- Modified `nirs4all/controllers/sklearn/op_transformermixin.py` (+120 lines)
- `tests/unit/test_transformermixin_augmentation.py` (~350 lines)

---

### Phase 6: Split Controller Integration (Week 3, Days 15-16)
**Goal:** Ensure splits operate only on base samples (leak prevention).

**Tasks:**
- [ ] Modify `CrossValidatorController.execute()`
  - Before calling `splitter.split()`, set `include_augmented=False`
  - Filter data retrieval:
    * `X = dataset.x(context, include_augmented=False)`
    * `y = dataset.y(context, include_augmented=False)`
    * `groups = dataset.metadata_column(..., include_augmented=False)` if applicable
- [ ] Test with multiple splitters:
  - KFold, StratifiedKFold, GroupKFold
  - ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit
  - TimeSeriesSplit, LeaveOneOut, LeavePOut
- [ ] Write integration tests:
  - `test_kfold_with_augmentation()` - Verify no leakage
  - `test_stratified_split_with_augmentation()` - Class distribution preserved
  - `test_group_split_with_augmentation()` - Group integrity maintained
  - `test_augmented_samples_follow_origin()` - Fold assignment correctness

**Deliverables:**
- Modified `nirs4all/controllers/sklearn/op_split.py` (+20 lines)
- `tests/integration/test_split_augmentation.py` (~300 lines)

---

### Phase 7: End-to-End Testing & Documentation (Week 3-4, Days 17-21)
**Goal:** Comprehensive validation and user-facing documentation.

**Tasks:**
- [ ] End-to-end pipeline tests:
  - `test_full_pipeline_standard_mode()` - Complete pipeline with count-based augmentation
  - `test_full_pipeline_balanced_mode()` - Complete pipeline with balancing
  - `test_multi_source_pipeline()` - Multi-source compatibility
  - `test_pipeline_serialization()` - Save/load with augmented samples
  - `test_prediction_mode()` - Augmentation doesn't run in predict
- [ ] Leak prevention validation:
  - `test_no_leakage_across_folds()` - Augmented samples in correct fold only
  - `test_validation_fold_isolation()` - Val samples' augmented versions excluded
- [ ] Performance benchmarks:
  - Memory usage profiling (with/without augmentation)
  - Execution time measurements
  - Scalability tests (1K, 10K, 100K samples)
- [ ] Update documentation:
  - Add `docs/SAMPLE_AUGMENTATION.md` user guide
  - Update API reference for Dataset methods
  - Add pipeline examples to `examples/` directory
  - Update CHANGELOG and release notes
- [ ] Code quality:
  - Type hints verification
  - Docstring completeness
  - Code style consistency (black, isort)
  - Remove debug code and print statements

**Deliverables:**
- `tests/integration/test_sample_augmentation_e2e.py` (~500 lines)
- `docs/SAMPLE_AUGMENTATION.md` (~200 lines)
- `examples/sample_augmentation_standard.py` (~80 lines)
- `examples/sample_augmentation_balanced.py` (~100 lines)
- `examples/sample_augmentation_multi_source.py` (~120 lines)

---

### Phase 8: Polish & Finalization (Week 4, Days 22-28)
**Goal:** Error handling, logging, edge cases, and production readiness.

**Tasks:**
- [ ] Error handling:
  - Empty transformer lists
  - Invalid balance sources (non-existent metadata columns)
  - Impossible balancing targets (max_factor too high)
  - Memory overflow protection (warn for large augmentations)
  - Graceful degradation
- [ ] Logging and diagnostics:
  - Progress indicators for long operations (tqdm integration?)
  - Verbose mode: log augmentation plan, transformer distribution
  - Debug mode: detailed execution trace
  - Warning messages for suboptimal configs
- [ ] Edge cases:
  - Single-sample datasets
  - All samples already balanced
  - Transformers that fail on certain samples
  - Very large augmentation counts (memory management)
- [ ] Pipeline serialization:
  - Ensure augmented samples persist across save/load
  - Binary compatibility checks
  - Version tracking for augmentation metadata
- [ ] Final code review:
  - Security review (pickle safety)
  - Performance optimization passes
  - Memory leak checks
  - Thread safety considerations (if parallelization planned)

**Deliverables:**
- Hardened production code across all modules
- Logging infrastructure in place
- Edge case handling and tests
- Final release candidate

---

### Estimated Timeline: 4 Weeks

| **Week** | **Focus**                              | **Phases**       | **Deliverables**                                  |
|----------|----------------------------------------|------------------|---------------------------------------------------|
| **1**    | Foundation + Dataset API               | 1, 2, 3          | BalancingCalculator, Indexer, Dataset enhancements|
| **2**    | Controllers (SampleAug + TransformerMixin) | 4, 5        | Controller logic with delegation pattern          |
| **3**    | Integration + Testing                  | 6, 7             | Split integration, E2E tests, documentation       |
| **4**    | Polish + Production Readiness          | 8                | Error handling, logging, edge cases               |

**Key Milestones:**
- ✅ **End of Week 1:** Core infrastructure ready (can manually test augmentation)
- ✅ **End of Week 2:** Full controller delegation working (automated tests passing)
- ✅ **End of Week 3:** Production-ready with documentation and examples
- ✅ **End of Week 4:** Polished, tested, and ready for release

---

## 8. Potential Challenges & Mitigation

### Challenge 1: Performance with Many Augmentations
**Issue**: Generating 1000s of augmented samples could be slow.

**Mitigation:**
- Batch operations by transformer
- Use vectorized operations where possible
- Add progress indicators for long operations
- Consider lazy augmentation (generate on-the-fly during training)

### Challenge 2: Memory Footprint
**Issue**: Storing augmented samples doubles/triples memory usage.

**Mitigation:**
- Add configuration option for in-memory vs. on-disk storage
- Implement lazy evaluation for augmentation
- Monitor and warn about memory usage
- Provide documentation on memory management

### Challenge 3: Metadata Consistency
**Issue**: Metadata must align with augmented samples.

**Mitigation:**
- Metadata automatically duplicated from origin sample
- Document metadata behavior clearly
- Add validation checks
- Provide `update_metadata` for post-augmentation adjustments

### Challenge 4: Serialization/Deserialization
**Issue**: Saving/loading datasets with augmented samples.

**Mitigation:**
- Ensure indexer DataFrame serializes correctly
- Store augmentation lineage in metadata
- Add tests for save/load cycles
- Document persistence behavior

---

## 9. Future Enhancements

### 9.1 Advanced Balancing
- **Multi-Objective Balancing**: Balance by multiple metadata columns simultaneously
- **Hierarchical Balancing**: Balance within groups (e.g., balance classes within each batch)
- **Custom Balancing Functions**: User-defined balancing logic

### 9.2 Conditional Augmentation
- **Difficulty-Based**: Augment only samples near decision boundary
- **Confidence-Based**: Augment samples with low model confidence
- **Error-Based**: Augment samples that were misclassified in previous runs

### 9.3 Augmentation Strategies
- **Mixup/CutMix**: Blend samples from different classes
- **SMOTE-like**: Synthetic samples between neighbors
- **Adversarial**: Generate adversarial examples for robustness

### 9.4 Pipeline Integration
- **Augmentation Chains**: Multiple augmentation steps with different configs
- **Fold-Specific Augmentation**: Different augmentation per CV fold
- **Augmentation Pruning**: Remove low-quality augmented samples

---

## 10. Open Questions for Validation

1. **Default Behavior**: Should `include_augmented=True` be the default, or should users opt-in?
   - **Recommendation**: True by default (transparent augmentation)

2. **Augmentation Timing**: Should augmentation happen before or after preprocessing?
   - **Recommendation**: After preprocessing (apply to already-scaled data)

3. **Multi-Source Behavior**: Should balancing be per-source or global?
   - **Recommendation**: Global (balance across all sources combined)

4. **Transformer Fit Behavior**: Should transformers be fit on each sample or on all samples?
   - **Recommendation**: Fit per sample (independent augmentations)

5. **Metadata Propagation**: Should metadata be deep-copied or shared reference?
   - **Recommendation**: Deep-copied (augmented samples are independent)

---

## 11. Conclusion

This design provides a **comprehensive, maintainable, and extensible** solution for sample augmentation in nirs4all. Key strengths:

✅ **Leak Prevention**: Robust two-phase selection ensures augmented samples follow origins
✅ **Flexibility**: Supports both standard and balanced modes
✅ **Backward Compatible**: Existing pipelines work unchanged
✅ **Clean Architecture**: Minimal changes to core components, clear separation of concerns
✅ **Delegation Pattern**: Follows feature_augmentation architecture exactly:
   - **SampleAugmentationController**: Plans distribution, emits contexts (NO pre-allocation)
   - **TransformerMixinController**: Executes augmentation, adds samples (all the work)
✅ **Multi-Source Support**: Automatically handles all sources via TransformerMixinController
✅ **Serialization Ready**: Binary saving/loading through existing mechanisms
✅ **Parallelization Ready**: Can use `run_steps()` for parallel execution in future
✅ **Extensible**: Easy to add new augmentation strategies
✅ **Well-Tested**: Comprehensive 8-phase test coverage planned

**Architecture Highlights:**
- **SampleAugmentationController**:
  - Calculates augmentation plan (standard or balanced)
  - Builds transformer→indices distribution map
  - Emits ONE `run_step()` per transformer with `target_samples` list
  - **Does NOT pre-allocate** - TransformerMixin handles that

- **TransformerMixinController**:
  - Detects `augment_sample=True` action (like `add_feature`)
  - For each sample in `target_samples`:
    * Gets origin data (all sources)
    * Applies transformer to all sources
    * Calls `dataset.add_samples()` to create augmented sample
  - Handles binary serialization per sample

- **Indexer**: Two-phase selection (base + augmented aggregation)
- **Dataset**: Transparent `include_augmented` parameter for backward compatibility

**Key Difference from Previous Design:**
- ❌ **OLD**: Controller pre-allocates samples, TransformerMixin replaces features
- ✅ **NEW**: Controller only plans, TransformerMixin creates samples during transformation
- **Benefit**: Cleaner separation, follows feature_augmentation pattern exactly, no wasted pre-allocation

**Next Steps:**
1. ✅ Review this corrected design document
2. ⏳ Address open questions (Section 10)
3. ⏳ Approve architecture
4. ⏳ Begin Phase 1 implementation (BalancingCalculator)

---

**Document Prepared By:** AI Architecture Analysis
**Review Requested From:** @GBeurier
**Status:** ⏳ Awaiting Final Validation (Corrected Delegation Pattern)
**Last Updated:** October 15, 2025 - Corrected delegation pattern: controller distributes, TransformerMixin executes + adds samples
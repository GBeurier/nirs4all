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

### 3.4 Phase 4: Controller Rewrite

**File:** `nirs4all/controllers/dataset/op_sample_augmentation.py`

```python
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING, Union
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
            return self._execute_balanced(config, transformers, dataset, context, runner)
        else:
            return self._execute_standard(config, transformers, dataset, context, runner)
    
    def _execute_standard(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
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
        
        # Create augmentation plan
        augmentation_counts = {sample_id: count for sample_id in base_samples}
        
        # Select transformers
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:  # "all"
            transformer_map = self._apply_all_transformers(transformers, base_samples, count)
        
        # Apply transformations
        self._apply_augmentations(
            dataset, transformer_map, transformers, context, runner
        )
        
        return context, []
    
    def _execute_balanced(
        self,
        config: Dict,
        transformers: List,
        dataset: 'SpectroDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
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
        
        # Calculate augmentation counts
        augmentation_counts = BalancingCalculator.calculate_balanced_counts(
            labels, base_samples, max_factor
        )
        
        # Select transformers
        if selection == "random":
            transformer_map = BalancingCalculator.apply_random_transformer_selection(
                transformers, augmentation_counts, random_state
            )
        else:
            transformer_map = self._apply_all_transformers_balanced(
                transformers, augmentation_counts
            )
        
        # Apply transformations
        self._apply_augmentations(
            dataset, transformer_map, transformers, context, runner
        )
        
        return context, []
    
    def _apply_augmentations(
        self,
        dataset: 'SpectroDataset',
        transformer_map: Dict[int, List[int]],
        transformers: List,
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ):
        """
        Apply transformations to generate augmented samples.
        
        Args:
            transformer_map: sample_id → list of transformer indices to apply
            transformers: List of transformer instances
            dataset, context, runner: Standard execution context
        """
        # Group by transformer to batch operations
        transformer_to_samples = {}
        for sample_id, transformer_indices in transformer_map.items():
            for trans_idx in transformer_indices:
                transformer_to_samples.setdefault(trans_idx, []).append(sample_id)
        
        # Apply each transformer to its samples
        for trans_idx, sample_ids in transformer_to_samples.items():
            transformer = transformers[trans_idx]
            augmentation_id = f"aug_{trans_idx}_{transformer.__class__.__name__}"
            
            # Get original data for these samples
            sample_selector = {"sample": sample_ids}
            X_original = dataset.x(sample_selector, include_augmented=False)
            
            # Transform data
            X_augmented = transformer.fit_transform(X_original)
            
            # Add augmented samples to dataset
            count_per_sample = [transformer_map[sid].count(trans_idx) for sid in sample_ids]
            
            # Replicate data if count > 1 per sample
            if sum(count_per_sample) > len(sample_ids):
                X_augmented = np.repeat(X_augmented, count_per_sample, axis=0)
            
            # Add to dataset
            dataset.augment_samples(
                data=X_augmented,
                processings=["raw"],  # Use existing processings from parent
                augmentation_id=augmentation_id,
                selector={"sample": sample_ids},
                count=count_per_sample
            )
    
    def _apply_all_transformers(self, transformers, samples, count):
        """Cycle through all transformers for 'all' selection mode."""
        transformer_map = {}
        for sample_id in samples:
            # Cycle through transformers
            transformer_map[sample_id] = [i % len(transformers) for i in range(count)]
        return transformer_map
    
    def _apply_all_transformers_balanced(self, transformers, augmentation_counts):
        """Apply all transformers in balanced mode."""
        transformer_map = {}
        for sample_id, count in augmentation_counts.items():
            if count > 0:
                transformer_map[sample_id] = [i % len(transformers) for i in range(count)]
            else:
                transformer_map[sample_id] = []
        return transformer_map
```

### 3.5 Phase 5: Split Controller Modification

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

### 4.1 Standard Augmentation Flow

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
# 1. MinMaxScaler applies to all samples
# 2. Sample augmentation:
#    - Get train samples: [1, 2, 3, ..., 100]
#    - For each sample, generate 3 augmented versions
#    - Randomly pick transformer for each
#    - Result: 300 new samples (sample=101-400, origin=1-100)
# 3. ShuffleSplit:
#    - Operates on base samples only: [1-100]
#    - Creates folds: fold1: train=[1,2,5,...], val=[3,4,6,...]
# 4. Model training (fold 1):
#    - dataset.x({"partition": "train"}) returns:
#      base samples [1,2,5,...] + their augmented versions
#    - Prevents leakage: val samples' augmented versions stay out
```

### 4.2 Balanced Augmentation Flow

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
    # ...
]

# Execution:
# 1. Get labels: [0]*80 + [1]*20
# 2. Calculate:
#    - Majority size: 80
#    - Target for class 1: int(80 * 0.9) = 72
#    - Augmentations needed: 72 - 20 = 52
#    - Per sample (class 1): 52 // 20 = 2, remainder = 12
# 3. Apply:
#    - 12 class-1 samples get 3 augmentations each
#    - 8 class-1 samples get 2 augmentations each
#    - Total: 12*3 + 8*2 = 52 augmented samples
# 4. Final distribution:
#    - Class 0: 80 samples (unchanged)
#    - Class 1: 20 + 52 = 72 samples
#    - Ratio: 80:72 (balanced to ~90%)
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

### Phase 1: Foundation (Week 1)
- [ ] Implement `BalancingCalculator` utility class
- [ ] Add new indexer methods (`get_augmented_for_origins`, `get_origin_for_sample`)
- [ ] Modify `Indexer.x_indices()` with two-phase logic
- [ ] Unit tests for indexer and balancing

### Phase 2: Dataset API (Week 1-2)
- [ ] Update `Dataset.x()` with `include_augmented` parameter
- [ ] Update `Dataset.y()` with augmented mapping
- [ ] Update `Dataset.metadata()` similarly
- [ ] Integration tests for dataset methods

### Phase 3: Controller (Week 2)
- [ ] Rewrite `SampleAugmentationController`
- [ ] Implement `_execute_standard()` and `_execute_balanced()`
- [ ] Add `_apply_augmentations()` logic
- [ ] Controller unit tests

### Phase 4: Split Integration (Week 2-3)
- [ ] Modify `CrossValidatorController.execute()` to filter base samples
- [ ] Test leak prevention across different splitters
- [ ] Integration tests with various CV strategies

### Phase 5: Validation & Documentation (Week 3)
- [ ] End-to-end pipeline tests
- [ ] Performance benchmarks
- [ ] Update user documentation
- [ ] Add examples to `examples/` directory

### Phase 6: Polish (Week 3-4)
- [ ] Error handling and edge cases
- [ ] Logging and verbose output
- [ ] Pipeline serialization support
- [ ] Code review and cleanup

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
✅ **Clean Architecture**: Minimal changes to core components  
✅ **Extensible**: Easy to add new augmentation strategies  
✅ **Well-Tested**: Comprehensive test coverage planned  

**Next Steps:**
1. Review this design document
2. Address open questions
3. Approve/modify architecture
4. Begin Phase 1 implementation

---

**Document Prepared By:** AI Architecture Analysis  
**Review Requested From:** @GBeurier  
**Status:** ⏳ Awaiting Validation

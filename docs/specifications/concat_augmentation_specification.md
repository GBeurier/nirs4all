# Concat Augmentation Controller - Technical Specification

**Date:** December 2024
**Author:** nirs4all Development Team
**Status:** Draft Specification v2

---

## Table of Contents

1. [Part 1: Feature Augmentation Logic Overview](#part-1-feature-augmentation-logic-overview)
   - [Data Model Architecture](#data-model-architecture)
   - [Feature Augmentation Controller](#feature-augmentation-controller)
   - [Processing Dimension Management](#processing-dimension-management)
   - [Feature Dimension Change Handling](#feature-dimension-change-handling)
2. [Part 2: Concat Augmentation Controller Proposal](#part-2-concat-augmentation-controller-proposal)
   - [Use Case and Motivation](#use-case-and-motivation)
   - [Core Semantics](#core-semantics)
   - [Proposed API](#proposed-api)
   - [Chained Transformers](#chained-transformers)
   - [Generator Support](#generator-support)
   - [Implementation Design](#implementation-design)
   - [Serialization Strategy](#serialization-strategy)
3. [Part 3: Implementation Roadmap](#part-3-implementation-roadmap)

---

## Part 1: Feature Augmentation Logic Overview

### Data Model Architecture

The nirs4all feature storage uses a **3D array architecture**:

```
Array Shape: (n_samples, n_processings, n_features)
             ↑           ↑              ↑
             │           │              └── Feature/wavelength dimension (max across all processings)
             │           └── Processing dimension (raw, SNV, SG, etc.)
             └── Sample dimension
```

#### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `SpectroDataset` | `data/dataset.py` | Main facade coordinating all data operations |
| `Features` | `data/features.py` | Manages N aligned FeatureSource objects (multi-source) |
| `FeatureSource` | `data/_features/feature_source.py` | Single source: 3D array + processing management |
| `ArrayStorage` | `data/_features/array_storage.py` | Low-level 3D NumPy array operations with padding |
| `ProcessingManager` | `data/_features/processing_manager.py` | Tracks processing IDs and their indices |

### Feature Augmentation Controller

The `FeatureAugmentationController` orchestrates **adding new processing dimensions** while preserving existing ones.

#### Key Behavior

1. **Sets `add_feature=True`** in the context metadata
2. **Iterates through sub-steps** (each preprocessing operation)
3. **Each sub-step ADDS a new processing dimension** to the feature array

```python
# Pipeline configuration
pipeline = [
    {
        "feature_augmentation": [
            SNV(),           # Adds processing "SNV"
            SavGol(11, 2),   # Adds processing "SavGol_11_2"
        ]
    }
]
# Before: (500, 1, 500) - raw only
# After:  (500, 3, 500) - raw, SNV, SavGol
```

### Processing Dimension Management

#### add_feature Flag Semantics

The `add_feature` flag in `StepMetadata` controls transformer behavior:

| Context | `add_feature` | Transformer Behavior |
|---------|---------------|---------------------|
| Top-level transform | `False` | **REPLACES** source processing(s) |
| Inside `feature_augmentation` | `True` | **ADDS** new processing dimension |

```python
# Top-level: REPLACES raw with PCA output
pipeline = [PCA(50)]
# Before: (500, 1, 500)
# After:  (500, 1, 50)  ← Feature dimension changed, processing renamed

# Inside feature_augmentation: ADDS PCA as new processing
pipeline = [{"feature_augmentation": [PCA(50)]}]
# Before: (500, 1, 500)
# After:  (500, 2, 500)  ← PCA output padded to 500, new processing added
```

### Feature Dimension Change Handling

#### Current Behavior

When a transformer outputs different feature count:

1. **REPLACE mode** (`add_feature=False`):
   - Uses `resize_features()` if ALL processings replaced with same dimension
   - Array shape changes: `(n, p, 500) → (n, p, 50)`

2. **ADD mode** (`add_feature=True`):
   - Uses `_prepare_data_for_storage()` which **pads** smaller outputs
   - Array shape preserved: `(n, p, 500) → (n, p+1, 500)`
   - New processing has 50 real features + 450 zeros

```python
# In ArrayStorage._prepare_data_for_storage()
if self.padding and data.shape[1] < self.num_features:
    padded_data = np.full((data.shape[0], self.num_features), self.pad_value)
    padded_data[:, :data.shape[1]] = data
    return padded_data
```

---

## Part 2: Concat Augmentation Controller Proposal

### Use Case and Motivation

#### Manual Approach (current)

In `bench/spectral_latent_features.py`, concatenation is done manually:

```python
class SpectralLatentFeatures(BaseEstimator, TransformerMixin):
    def transform(self, X: np.ndarray) -> np.ndarray:
        blocks = []
        blocks.append(self._pca.transform(X))      # 50 features
        blocks.append(self._wavelet.transform(X))  # 40 features
        blocks.append(self._fft.transform(X))      # 30 features

        return np.hstack(blocks)  # 120 features total
```

**Problems:**
- Manual concatenation logic
- No pipeline serialization integration
- Not composable with nirs4all generators

### Core Semantics

**`concat_transform` is a REPLACEMENT operation that applies to EACH existing processing.**

Unlike `feature_augmentation` which ADDS new processings, `concat_transform`:
1. Runs multiple transformers on the input
2. Concatenates their outputs horizontally
3. **REPLACES** the source processing with the concatenated result

#### Behavior Matrix

| Location | Behavior | Feature Dimension |
|----------|----------|-------------------|
| **Top-level** | Replaces ALL processings with concatenated versions | Changes to sum of outputs |
| **Inside `feature_augmentation`** | ADDS one new processing (inherits `add_feature=True`) | Padded to current max |

### Proposed API

#### Example 1: Top-Level Replacement

```python
# Initial: (500, 3, 500) with processings ["raw", "snv", "savgol"]

pipeline = [
    {"concat_transform": [PCA(50), SVD(50)]}
]

# Result: (500, 3, 100)
# Processings: ["raw_concat_PCA_SVD", "snv_concat_PCA_SVD", "savgol_concat_PCA_SVD"]
# Each processing now has 100 features (50 from PCA + 50 from SVD)
```

**Flow:**
```
For each processing in [raw, snv, savgol]:
    1. Extract 2D data for this processing
    2. Apply PCA(50) → 50 features
    3. Apply SVD(50) → 50 features
    4. Concatenate → 100 features
    5. Replace original processing with concatenated result
```

#### Example 2: Nested Inside feature_augmentation

```python
# Initial: (500, 1, 500) with processing ["raw"]

pipeline = [
    {
        "feature_augmentation": [
            SNV(),                                    # Adds "snv" processing
            SavGol(11, 2),                           # Adds "savgol" processing
            {"concat_transform": [PCA(50), SVD(50)]}  # Adds "concat_PCA_SVD" processing
        ]
    }
]

# Result: (500, 4, 500)
# Processings: ["raw", "snv", "savgol", "concat_PCA_SVD"]
#                                        └── 100 real features + 400 padding
```

**Flow:**
```
feature_augmentation sets add_feature=True:
    1. SNV() → adds processing "snv"
    2. SavGol() → adds processing "savgol"
    3. concat_transform (inherits add_feature=True):
        - Apply to current active processing (or raw)
        - PCA(50) + SVD(50) → 100 features
        - ADD as new processing (padded to 500)
```

#### Example 3: Extended Configuration Syntax

```python
{
    "concat_transform": {
        "name": "latent_features",           # Custom processing name
        "operations": [PCA(50), SVD(50)],    # Transformers to concatenate
        "source_processing": "snv",          # Apply only to this processing (optional)
    }
}
```

### Chained Transformers

Operations can be **single transformers** or **chains** (sequential pipelines):

```python
{
    "concat_transform": [
        PCA(100),                    # Single: 100 features
        [Wavelet("haar"), PCA(50)],  # Chain: Wavelet → PCA = 50 features
        SVD(30),                     # Single: 30 features
    ]
}
# Total: 100 + 50 + 30 = 180 features
```

**Chain semantics:** `[A, B, C]` means `C(B(A(X)))` - sequential application.

### Generator Support

The pipeline generator should support `concat_transform` with the same syntax as `feature_augmentation`:

#### Current feature_augmentation Generator

```python
{
    "feature_augmentation": {
        "_or_": [Detrend, FstDer, SndDer, Gauss, SNV, SavGol, Haar, MSC],
        "size": [(1, 3), (1, 2)],  # Pick 1-3 or 1-2 items
        "count": 50                 # Generate 50 pipeline variants
    }
}
```

#### Proposed concat_transform Generator

```python
{
    "concat_transform": {
        "_or_": [
            PCA(100),
            PCA(50),
            SVD(50),
            [Wavelet("haar"), PCA(30)],  # Chain as single option
            [FFT(), PCA(40)],            # Another chain
            LocalStats(n_bands=10),
        ],
        "size": [(2, 4)],  # Concatenate 2-4 items
        "count": 50        # Generate 50 variants
    }
}
```

**Generated examples:**
```python
# Variant 1
{"concat_transform": [PCA(100), SVD(50)]}  # 150 features

# Variant 2
{"concat_transform": [PCA(50), [Wavelet("haar"), PCA(30)], LocalStats(n_bands=10)]}  # 50+30+X features

# Variant 3
{"concat_transform": [[FFT(), PCA(40)], SVD(50), PCA(100)]}  # 40+50+100 features
```

#### Nested Generator (inside feature_augmentation)

```python
{
    "feature_augmentation": {
        "_or_": [
            SNV,
            SavGol,
            Detrend,
            {
                "concat_transform": {
                    "_or_": [PCA(50), SVD(30), [Wavelet, PCA(20)]],
                    "size": [(2, 3)]
                }
            }
        ],
        "size": [(2, 4)],
        "count": 100
    }
}
```

**Generated example:**
```python
{
    "feature_augmentation": [
        SNV(),
        {"concat_transform": [PCA(50), SVD(30)]},  # 80 features, padded
        Detrend()
    ]
}
# Result: 4 processings ["raw", "snv", "concat_PCA_SVD", "detrend"]
```

### Implementation Design

#### Controller Class

```python
# nirs4all/controllers/data/concat_transform.py

from typing import Any, Dict, List, Tuple, Optional, Union, TYPE_CHECKING
import numpy as np
from sklearn.base import clone, TransformerMixin

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.config.context import ExecutionContext, RuntimeContext
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.pipeline.steps.parser import ParsedStep


@register_controller
class ConcatAugmentationController(OperatorController):
    """
    Controller that concatenates multiple transformer outputs.

    Semantics:
    - Top-level (add_feature=False): REPLACES each processing with concatenated version
    - Inside feature_augmentation (add_feature=True): ADDS one new processing

    Supports:
    - Single transformers: PCA(50)
    - Chained transformers: [Wavelet(), PCA(50)] → sequential application
    - Mixed: [PCA(50), [Wavelet(), SVD(30)], LocalStats()]
    """
    priority = 10

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        return keyword == "concat_transform"

    @classmethod
    def use_multi_source(cls) -> bool:
        return True

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        return True

    def execute(
        self,
        step_info: 'ParsedStep',
        dataset: 'SpectroDataset',
        context: 'ExecutionContext',
        runtime_context: 'RuntimeContext',
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Optional[List[Tuple[str, Any]]] = None,
        prediction_store: Optional[Any] = None
    ) -> Tuple['ExecutionContext', List[Tuple[str, bytes]]]:
        """Execute concat augmentation."""

        config = self._parse_config(step_info.original_step["concat_transform"])
        operations = config["operations"]
        output_name_base = config.get("name")
        source_processing_filter = config.get("source_processing")

        if not operations:
            return context, []

        # Determine mode: replace or add
        is_add_mode = context.metadata.add_feature

        all_artifacts = []
        n_sources = dataset.features_sources()

        for sd_idx in range(n_sources):
            processing_ids = dataset.features_processings(sd_idx)

            # Determine which processings to operate on
            if source_processing_filter:
                target_processings = [p for p in processing_ids if p == source_processing_filter]
            else:
                target_processings = list(processing_ids) if not is_add_mode else [processing_ids[0]]

            # Get data
            train_context = context.with_partition("train")
            train_data = dataset.x(train_context.selector, "3d", concat_source=False)
            all_data = dataset.x(context.selector, "3d", concat_source=False)

            if not isinstance(train_data, list):
                train_data = [train_data]
            if not isinstance(all_data, list):
                all_data = [all_data]

            source_train = train_data[sd_idx]
            source_all = all_data[sd_idx]

            # Process each target processing
            for proc_name in target_processings:
                proc_idx = processing_ids.index(proc_name)
                train_2d = source_train[:, proc_idx, :]
                all_2d = source_all[:, proc_idx, :]

                # Apply all operations and concatenate
                concat_blocks = []
                for op_idx, operation in enumerate(operations):
                    op_name_base = self._get_operation_name(operation, op_idx)
                    binary_key = f"{proc_name}_{op_name_base}"

                    # Handle chain vs single transformer
                    if isinstance(operation, list):
                        # Chain: [A, B, C] → C(B(A(X)))
                        transformed, chain_artifacts = self._execute_chain(
                            operation, train_2d, all_2d, binary_key,
                            mode, loaded_binaries, runtime_context
                        )
                        all_artifacts.extend(chain_artifacts)
                    else:
                        # Single transformer
                        transformed, artifact = self._execute_single(
                            operation, train_2d, all_2d, binary_key,
                            mode, loaded_binaries, runtime_context
                        )
                        if artifact:
                            all_artifacts.append(artifact)

                    concat_blocks.append(transformed)

                # Concatenate all blocks
                concatenated = np.hstack(concat_blocks)

                # Generate output processing name
                if output_name_base:
                    output_name = f"{proc_name}_{output_name_base}" if not is_add_mode else output_name_base
                else:
                    op_names = [self._get_operation_name(op, i) for i, op in enumerate(operations)]
                    suffix = "concat_" + "_".join(op_names[:3])
                    if len(op_names) > 3:
                        suffix += f"_+{len(op_names) - 3}"
                    output_name = f"{proc_name}_{suffix}" if not is_add_mode else suffix

                # Apply result
                if is_add_mode:
                    # ADD mode: add as new processing (padded)
                    dataset.update_features(
                        source_processings=[""],
                        features=[concatenated],
                        processings=[output_name],
                        source=sd_idx
                    )
                    break  # Only add once in add mode
                else:
                    # REPLACE mode: replace this processing
                    dataset.replace_features(
                        source_processings=[proc_name],
                        features=[concatenated],
                        processings=[output_name],
                        source=sd_idx
                    )

        # Update context with new processing names
        new_processing = []
        for sd_idx in range(n_sources):
            src_processing = list(dataset.features_processings(sd_idx))
            new_processing.append(src_processing)
        context = context.with_processing(new_processing)

        return context, all_artifacts

    def _parse_config(self, config) -> Dict:
        """Parse concat_transform configuration."""
        if isinstance(config, list):
            return {"operations": config, "name": None, "source_processing": None}
        elif isinstance(config, dict):
            if "operations" in config:
                return config
            elif "_or_" in config:
                # Generator syntax - should be expanded by generator before reaching here
                raise ValueError("Generator syntax not expanded - should be handled by pipeline generator")
            else:
                return {"operations": list(config.values()), "name": None, "source_processing": None}
        else:
            raise ValueError(f"Invalid concat_transform config: {config}")

    def _get_operation_name(self, operation, index: int) -> str:
        """Get name for an operation (single or chain)."""
        if isinstance(operation, list):
            # Chain: use last transformer's name
            return f"chain_{operation[-1].__class__.__name__}_{index}"
        else:
            return f"{operation.__class__.__name__}_{index}"

    def _execute_single(
        self,
        transformer: TransformerMixin,
        train_data: np.ndarray,
        all_data: np.ndarray,
        binary_key: str,
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        runtime_context: 'RuntimeContext'
    ) -> Tuple[np.ndarray, Optional[Tuple[str, bytes]]]:
        """Execute a single transformer."""
        if loaded_binaries and mode in ["predict", "explain"]:
            fitted = dict(loaded_binaries).get(binary_key)
            if fitted is None:
                raise ValueError(f"Binary for {binary_key} not found")
        else:
            fitted = clone(transformer)
            fitted.fit(train_data)

        transformed = fitted.transform(all_data)

        artifact = None
        if mode == "train":
            artifact = runtime_context.saver.persist_artifact(
                step_number=runtime_context.step_number,
                name=binary_key,
                obj=fitted,
                format_hint='sklearn'
            )

        return transformed, artifact

    def _execute_chain(
        self,
        chain: List[TransformerMixin],
        train_data: np.ndarray,
        all_data: np.ndarray,
        binary_key_base: str,
        mode: str,
        loaded_binaries: Optional[List[Tuple[str, Any]]],
        runtime_context: 'RuntimeContext'
    ) -> Tuple[np.ndarray, List[Tuple[str, bytes]]]:
        """Execute a chain of transformers sequentially."""
        artifacts = []
        current_train = train_data
        current_all = all_data

        for i, transformer in enumerate(chain):
            binary_key = f"{binary_key_base}_chain{i}"

            if loaded_binaries and mode in ["predict", "explain"]:
                fitted = dict(loaded_binaries).get(binary_key)
                if fitted is None:
                    raise ValueError(f"Binary for {binary_key} not found")
            else:
                fitted = clone(transformer)
                fitted.fit(current_train)

            current_train = fitted.transform(current_train)
            current_all = fitted.transform(current_all)

            if mode == "train":
                artifact = runtime_context.saver.persist_artifact(
                    step_number=runtime_context.step_number,
                    name=binary_key,
                    obj=fitted,
                    format_hint='sklearn'
                )
                artifacts.append(artifact)

        return current_all, artifacts
```

### Serialization Strategy

#### Binary Naming Convention

```
step{N}_{processing}_{operation}_{index}[_chain{M}]
```

**Examples:**

| Scenario | Binary Keys |
|----------|-------------|
| Top-level on 3 processings with 2 ops | `step5_raw_PCA_0`, `step5_raw_SVD_1`, `step5_snv_PCA_0`, `step5_snv_SVD_1`, `step5_savgol_PCA_0`, `step5_savgol_SVD_1` |
| Nested (add mode) with 2 ops | `step3_concat_PCA_0`, `step3_concat_SVD_1` |
| Chain `[Wavelet, PCA]` | `step5_raw_chain_PCA_0_chain0` (Wavelet), `step5_raw_chain_PCA_0_chain1` (PCA) |

#### Prediction Mode

1. Load all binaries matching pattern `step{N}_*`
2. For each processing × operation combination, retrieve fitted transformer
3. Apply in same order as training
4. Concatenate and replace/add

---

## Part 3: Implementation Roadmap

### Phase 1: Core Implementation (3-4 days)

| Task | Duration | Description |
|------|----------|-------------|
| 1.1 Controller skeleton | 0.5 day | Basic `ConcatAugmentationController` with replace mode |
| 1.2 Add mode support | 0.5 day | Handle `add_feature=True` context (padding) |
| 1.3 Chain support | 0.5 day | Execute `[A, B, C]` as sequential pipeline |
| 1.4 Multi-processing replace | 1 day | Apply to all processings, resize features |
| 1.5 Serialization | 0.5 day | Binary naming, prediction mode loading |
| 1.6 Parser integration | 0.5 day | Recognize `concat_transform` keyword |

### Phase 2: Generator Support (2-3 days)

| Task | Duration | Description |
|------|----------|-------------|
| 2.1 Generator syntax | 1 day | Support `_or_`, `size`, `count` in generator |
| 2.2 Chain generation | 0.5 day | Generate chains from nested lists |
| 2.3 Nested generation | 1 day | Support inside `feature_augmentation` generator |
| 2.4 Combination expansion | 0.5 day | Cartesian product of options |

### Phase 3: Testing (2-3 days)

| Task | Duration | Description |
|------|----------|-------------|
| 3.1 Unit tests | 1 day | Replace mode, add mode, chains |
| 3.2 Integration tests | 1 day | Full pipeline train/predict cycle |
| 3.3 Generator tests | 0.5 day | Expansion correctness |
| 3.4 Serialization tests | 0.5 day | Save/load with binaries |

### Phase 4: Documentation (1-2 days)

| Task | Duration | Description |
|------|----------|-------------|
| 4.1 User guide | 0.5 day | Conceptual docs with examples |
| 4.2 Example script | 0.5 day | `Q22_concat_transform.py` |
| 4.3 Generator docs | 0.5 day | Generator syntax examples |

### Timeline Summary

**Total: 8-12 days**

```
Week 1: Core implementation + basic testing
Week 2: Generator support + comprehensive testing + docs
```

### Dependencies

```
1.1 Controller ──┬──► 1.2 Add mode ──┬──► 1.4 Multi-proc ──► 1.5 Serialization ──► 1.6 Parser
                 │                   │
                 └──► 1.3 Chains ────┘
                                                                    │
                                                                    ▼
                                                            2.1 Generator syntax
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                            2.2 Chain gen    2.3 Nested gen    3.1 Unit tests
                                                    │               │               │
                                                    └───────┬───────┘               │
                                                            ▼                       ▼
                                                    3.3 Generator tests     3.2 Integration tests
                                                                                    │
                                                                                    ▼
                                                                            3.4 Serialization tests
                                                                                    │
                                                                                    ▼
                                                                            4.x Documentation
```

---

## Appendix A: Complete Example Walkthrough

### Scenario: TabPFN Feature Engineering

**Goal:** Convert 500 wavelengths to ~150 tabular-like features for TabPFN.

```python
from nirs4all.pipeline import PipelineRunner
from nirs4all.preprocessing import SNV, SavGol
from sklearn.decomposition import PCA, TruncatedSVD
from nirs4all.operators import WaveletFeatures, LocalStats

# Load dataset: (200 samples, 1 processing "raw", 500 features)
dataset = load_nirs_dataset()

pipeline = [
    # Step 1: Add preprocessing variants
    {
        "feature_augmentation": [
            SNV(),
            SavGol(window=11, polyorder=2),
        ]
    },
    # After: (200, 3, 500) - ["raw", "snv", "savgol"]

    # Step 2: Replace each processing with concatenated features
    {
        "concat_transform": [
            PCA(n_components=50),          # 50 features
            TruncatedSVD(n_components=30), # 30 features
            [WaveletFeatures(), PCA(40)],  # Chain: Wavelet→PCA = 40 features
            LocalStats(n_bands=10),        # ~30 features
        ]
    },
    # After: (200, 3, 150) - ["raw_concat_...", "snv_concat_...", "savgol_concat_..."]

    # Step 3: Model selection per processing
    {
        "model": TabPFNClassifier(),
        "processing": "snv_concat_PCA_SVD_chain_PCA_LocalStats"
    }
]

runner = PipelineRunner(dataset)
results = runner.run(pipeline)
```

---

## Appendix B: Comparison Table

| Aspect | `feature_augmentation` | `concat_transform` |
|--------|------------------------|----------------------|
| **Primary action** | ADDS processing dimensions | REPLACES/transforms processing content |
| **Output count** | N processings (one per op) | Same count, different features |
| **Feature dimension** | Preserved (padding if needed) | Changed (sum of outputs) |
| **Inside other** | N/A | Inherits `add_feature` flag |
| **Use case** | Multiple preprocessing views | Feature extraction/engineering |
| **Generator support** | `_or_`, `size`, `count` | Same syntax |
| **Chains** | Not applicable | `[A, B, C]` → sequential |

---

## Appendix C: Edge Cases and Considerations

### Case 1: Empty Operations List

```python
{"concat_transform": []}
# → No-op, returns unchanged context
```

### Case 2: Single Operation (Degenerate Case)

```python
{"concat_transform": [PCA(50)]}
# → Equivalent to just PCA(50), but follows concat naming convention
# → Processing becomes "raw_concat_PCA_0"
```

### Case 3: Mixed Feature Dimensions in Replace Mode

When replacing multiple processings, each gets the **same** concatenated feature dimension:

```python
# Before: (500, 3, 500) - raw, snv, savgol all have 500 features
{"concat_transform": [PCA(50), SVD(30)]}
# After: (500, 3, 80) - all processings now have 80 features
```

### Case 4: Chained Transformer Failure

If any transformer in a chain fails, the entire chain fails:

```python
[[BadTransformer(), PCA(50)]]  # If BadTransformer raises, PCA never runs
```

### Case 5: Generator with Weighted Probabilities (Future)

```python
{
    "concat_transform": {
        "_or_": [
            {"op": PCA(100), "weight": 2},  # 2x more likely
            {"op": SVD(50), "weight": 1},
        ],
        "size": [(2, 3)],
        "count": 50
    }
}
```

---

*Document generated for nirs4all project - December 2024 - v2*

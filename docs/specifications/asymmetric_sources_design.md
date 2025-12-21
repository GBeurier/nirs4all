# Asymmetric Multi-Source Dataset Design

**Version**: 1.0.0-draft
**Status**: Draft Design Document
**Date**: December 2025
**Related Spec**: [dataset_config_specification.md](dataset_config_specification.md)

---

## Table of Contents

1. [Objective](#objective)
2. [Current State](#current-state)
3. [Problem Statement](#problem-statement)
4. [Proposition](#proposition)
5. [Implementation Roadmap](#implementation-roadmap)

---

## Objective

Enable nirs4all to load and process **asymmetric multi-source datasets** where each source can have:

- Different feature dimensions (e.g., 500 vs 300 vs 50,000 features)
- Different numbers of preprocessing variations per source
- Different data modalities (spectral, time series, markers)

### Example Use Case

```
Source 1: NIRS spectra     (500 samples × 500 features)  - 3 preprocessings (raw, SNV, derivative)
Source 2: Time series      (500 samples × 300 features)  - 1 preprocessing (raw only)
Source 3: Genetic markers  (500 samples × 50,000 features) - 2 preprocessings (raw, scaled)
```

All sources share the same samples (rows aligned) but differ in their feature space and preprocessing history.

---

## Current State

### Data Structure

The current architecture supports multi-source datasets via the `Features` class:

```
Features
├── sources: List[FeatureSource]
│   ├── FeatureSource[0]  # Source 1
│   │   └── _storage: 3D array (samples × processings × features)
│   ├── FeatureSource[1]  # Source 2
│   │   └── _storage: 3D array (samples × processings × features)
│   └── ...
```

### Current Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Multiple sources | ✅ Supported | Via list of paths in `train_x` |
| Different feature dims per source | ✅ Supported | Each `FeatureSource` has independent shape |
| Same processings across sources | ✅ Supported | All sources get same preprocessing pipeline |
| Different processings per source | ❌ Not supported | All sources share processing list |
| Asymmetric source concatenation | ⚠️ Partial | Works for horizontal concat only |

### Execution Behaviors

When operators are applied to multi-source datasets, there are two dimensions of variation:

#### 1. Processing Dimension (within a source)

| Mode | Behavior | Use Case |
|------|----------|----------|
| **2D (concat)** | Flatten processings: `(samples, processings × features)` | sklearn models, most regressors |
| **3D (separate)** | Keep shape: `(samples, processings, features)` | Apply transformer to each processing separately |

#### 2. Source Dimension (across sources)

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Concatenate** | Merge horizontally: `(samples, Σ features)` | Single-input models |
| **Separate** | List of arrays: `[source1, source2, ...]` | Multi-head models, per-source transforms |

### Current Controller Behaviors

```python
# TransformerController: Iterates over sources, then processings
for sd_idx, (fit_x, all_x) in enumerate(zip(fit_data, all_data)):
    for processing_idx in range(fit_x.shape[1]):
        # Apply transformer to each (source, processing) pair
        transformer.fit(fit_2d)
        transformed = transformer.transform(all_2d)

# ModelController: Gets concatenated data
X_train = dataset.x(train_context.selector, layout="2d")  # Concatenates all
```

---

## Problem Statement

### Core Issue: Asymmetric Source Concatenation

When a model requests all sources concatenated (`concat_source=True`), the system attempts to merge them horizontally. This works when:

- All sources have compatible row counts (samples) ✅
- The model accepts variable feature dimensions ✅

**But fails when:**

1. **NN models with fixed input shapes**: A neural network expecting input shape `(batch, 500)` cannot accept concatenated asymmetric sources `(batch, 500+300+50000)`

2. **3D models with uniform expectations**: A CNN expecting `(batch, channels, features)` where all channels have the same feature count cannot handle asymmetric sources

3. **Preprocessing asymmetry**: If source 1 has 3 preprocessings and source 2 has 1, concatenating in 3D layout is impossible:
   ```
   Source 1: (500, 3, 500)   # 3 processings
   Source 2: (500, 1, 300)   # 1 processing
   # Cannot concatenate along feature axis when processing counts differ
   ```

### Failure Scenarios

| Scenario | Request | Sources | Result |
|----------|---------|---------|--------|
| A | `layout="2d", concat_source=True` | (500,500), (500,300) | ✅ Works: (500, 800) |
| B | `layout="3d", concat_source=True` | (500,2,500), (500,2,300) | ✅ Works: (500, 2, 800) |
| C | `layout="3d", concat_source=True` | (500,3,500), (500,1,300) | ❌ Fails: incompatible shapes |
| D | NN with input_shape=(500,) | (500,500), (500,300) | ❌ Fails: shape mismatch |
| E | Multi-head NN | (500,500), (500,300) | ✅ Works: separate inputs |

### Current Error Handling

Currently, these failures result in numpy broadcasting errors or silent shape mismatches, which are confusing for users.

---

## Proposition

### Design Principles

1. **Explicit configuration**: Users declare source characteristics and expected behaviors
2. **Early validation**: Detect incompatible configurations at load time, not execution time
3. **Clear error messages**: Explain why a configuration fails and suggest alternatives
4. **Flexible execution**: Support multiple strategies for handling asymmetric sources

### Proposed Configuration Schema

#### Extended Source Configuration

```yaml
sources:
  - name: "NIR"
    train_x: data/nir_train.csv
    test_x: data/nir_test.csv
    params:
      header_unit: cm-1
    # NEW: Source-specific metadata
    source_type: spectral          # spectral | timeseries | tabular | image
    shape_hint: [500]              # Expected feature shape per sample
    allow_preprocessing: true      # Whether in-pipeline preprocessing is allowed

  - name: "timeseries"
    train_x: data/ts_train.csv
    shape_hint: [15, 300]          # 15 timesteps × 300 features (2D per sample)
    source_type: timeseries
    allow_preprocessing: false     # Pre-processed externally

  - name: "markers"
    train_x: data/markers_train.csv
    shape_hint: [50000]
    source_type: tabular
    allow_preprocessing: true

# NEW: Source fusion strategy
source_fusion:
  mode: separate | concat | multi_head | custom
  # separate: Each source processed independently, merged at model level
  # concat: Concatenate for compatible sources (error if incompatible)
  # multi_head: Route to multi-input model architecture
  # custom: User-defined fusion function
```

### Proposed Execution Strategies

#### Strategy 1: Separate Processing (Default for Asymmetric)

Each source is processed independently through the pipeline, then merged at the model level:

```
Pipeline:
  [MinMaxScaler] → Applied to each source independently
  [SNV]          → Applied only to sources with allow_preprocessing=True
  [Model]        → Receives list of source arrays
```

#### Strategy 2: Source-Aware Transformers

Transformers can declare which sources they apply to:

```python
pipeline = [
    {"preprocessing": SNV(), "sources": ["NIR"]},           # Only NIR
    {"preprocessing": MinMaxScaler(), "sources": "all"},    # All sources
    {"model": MultiHeadNN(), "source_inputs": ["NIR", "timeseries", "markers"]}
]
```

#### Strategy 3: Branching by Source

Use existing branching mechanism to create per-source pipelines:

```python
pipeline = [
    {
        "source_branch": {
            "NIR": [SNV(), SavitzkyGolay()],
            "timeseries": [Standardize()],
            "markers": [VarianceThreshold()]
        }
    },
    {"merge": "concat"},  # or "stack" for multi-head
    {"model": ...}
]
```

### Validation Rules

The system should validate at configuration/load time:

| Check | When | Error Message |
|-------|------|---------------|
| Sample count alignment | Load | "Source 'X' has N samples but source 'Y' has M samples. All sources must have the same sample count." |
| Concat compatibility | Operator requests concat | "Cannot concatenate sources with different processing counts: NIR has 3, timeseries has 1. Use `concat_source=False` or align preprocessings." |
| Shape compatibility | Model input validation | "Model expects input shape (500,) but concatenated sources produce (50800,). Consider using `source_fusion: separate` or a multi-head architecture." |
| Preprocessing applicability | Pipeline step | "Cannot apply SNV to source 'markers' (type: tabular). SNV requires spectral data." |

### API Changes

#### Dataset Query API

```python
# Current
X = dataset.x(selector, layout="2d", concat_source=True)

# Extended
X = dataset.x(
    selector,
    layout="2d",
    concat_source=True,
    sources=["NIR", "markers"],      # NEW: Select specific sources
    on_incompatible="error"          # NEW: "error" | "warn" | "separate"
)
```

#### Source Introspection

```python
# NEW: Query source metadata
dataset.source_info("NIR")
# Returns: {"name": "NIR", "shape": (500, 3, 500), "type": "spectral", ...}

dataset.sources_compatible(layout="3d", concat=True)
# Returns: False, "Processing count mismatch: NIR=3, timeseries=1"
```

### Error Message Examples

```
ConfigurationError: Incompatible source concatenation requested.

Sources have different processing counts:
  - NIR: 3 processings (raw, SNV, derivative)
  - timeseries: 1 processing (raw)
  - markers: 2 processings (raw, scaled)

To resolve:
  1. Use `concat_source=False` to process sources separately
  2. Align processing counts across sources
  3. Use `source_fusion: multi_head` for models that accept multiple inputs
  4. Use source branching to apply different pipelines per source

See: https://nirs4all.readthedocs.io/asymmetric-sources
```

---

## Implementation Roadmap

### Phase 8A: Source Metadata & Validation (Foundation)

**Goal**: Enable source-level metadata and early validation of compatibility.

#### Tasks

- [ ] Add `source_type`, `shape_hint`, `allow_preprocessing` to `SourceConfig` schema
- [ ] Implement `SourceCompatibilityChecker` validation class
- [ ] Add `sources_compatible()` method to `SpectroDataset`
- [ ] Add `source_info(name)` introspection method
- [ ] Implement clear error messages for incompatible configurations
- [ ] Update `FeatureAccessor.x()` with `sources` parameter for selective retrieval
- [ ] Add `on_incompatible` parameter to control error/warn/separate behavior

#### Validation Points

| Point | Check |
|-------|-------|
| Config parsing | Source schema validity |
| Data loading | Sample count alignment |
| Operator execution | Shape compatibility for requested operation |
| Model input | Validate expected vs actual shapes |

### Phase 8B: Source-Aware Pipeline Execution

**Goal**: Enable per-source pipeline execution and source-specific operators.

#### Tasks

- [ ] Extend `ExecutionContext` with current source information
- [ ] Add `sources` parameter to controller execution
- [ ] Implement source filtering in `TransformerMixinController`
- [ ] Support `{"preprocessing": op, "sources": [...]}` syntax
- [ ] Track preprocessing history per source independently
- [ ] Update artifact storage to include source index

### Phase 8C: Source Branching Syntax

**Goal**: Enable declarative per-source pipeline branches.

#### Tasks

- [ ] Implement `source_branch` keyword in pipeline syntax
- [ ] Create `SourceBranchController` for routing
- [ ] Implement branch merge strategies (concat, stack, dict)
- [ ] Support nested source branches
- [ ] Integration with existing branching system

### Phase 8D: Multi-Head Model Support

**Goal**: Enable models that accept multiple source inputs.

#### Tasks

- [ ] Define `MultiInputModel` protocol/interface
- [ ] Implement `source_fusion: multi_head` mode
- [ ] Create adapters for common multi-head architectures
- [ ] Support `source_inputs` parameter in model configuration
- [ ] Integration with TensorFlow/PyTorch multi-input models

### Dependencies

```
Phase 8A (Foundation)
    ↓
Phase 8B (Execution) ←── Phase 8C (Branching)
    ↓
Phase 8D (Multi-Head)
```

### Testing Strategy

| Test Category | Coverage |
|---------------|----------|
| Unit: Schema | Source config validation, compatibility checks |
| Unit: Accessor | Selective source retrieval, error cases |
| Integration: Pipeline | Per-source execution, branching |
| Integration: Models | Multi-head models, shape validation |
| E2E: Examples | Real asymmetric datasets |

### Documentation Updates

- [ ] Update dataset_config_specification.md with source metadata
- [ ] Add asymmetric sources example to examples/
- [ ] Document error messages and resolutions
- [ ] Add troubleshooting guide for shape mismatches

---

## Appendix: Use Case Examples

### Use Case 1: Spectral + Genetic Markers

```yaml
name: spectral_genomic_fusion

sources:
  - name: "NIR"
    train_x: data/nir_spectra.csv
    source_type: spectral
    shape_hint: [2000]

  - name: "SNPs"
    train_x: data/snp_markers.csv
    source_type: tabular
    shape_hint: [50000]
    allow_preprocessing: false

source_fusion: multi_head

targets:
  path: data/phenotypes.csv

task_type: regression
```

### Use Case 2: Multi-Sensor Fusion

```yaml
name: sensor_fusion

sources:
  - name: "NIR"
    train_x: data/nir.csv
    shape_hint: [500]

  - name: "MIR"
    train_x: data/mir.csv
    shape_hint: [800]

  - name: "Raman"
    train_x: data/raman.csv
    shape_hint: [1200]

# All spectral, can be concatenated
source_fusion: concat

task_type: classification
```

### Use Case 3: Time Series + Metadata

```yaml
name: multimodal_forecast

sources:
  - name: "sensor_readings"
    train_x: data/sensors.csv
    source_type: timeseries
    shape_hint: [24, 10]  # 24 hours × 10 sensors

  - name: "static_features"
    train_x: data/metadata.csv
    source_type: tabular
    shape_hint: [50]

source_fusion: separate  # Must be handled by model architecture

task_type: regression
```

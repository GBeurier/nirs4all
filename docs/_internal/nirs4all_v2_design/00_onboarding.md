# NIRS4ALL v2.0: Onboarding Document

**Author**: GitHub Copilot (Claude Opus 4.5)
**Date**: December 26, 2025
**Status**: Design Overview
**Audience**: Technical stakeholders, new contributors, architects

---

## Table of Contents

1. [What is NIRS4ALL v2.0?](#what-is-nirs4all-v20)
2. [Why a Full Rewrite?](#why-a-full-rewrite)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Layer 1: Data Layer](#layer-1-data-layer)
5. [Layer 2: DAG Execution Engine](#layer-2-dag-execution-engine)
6. [Layer 3: API Layer](#layer-3-api-layer)
7. [Key Concepts](#key-concepts)
8. [User Experience](#user-experience)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Glossary](#glossary)

---

## What is NIRS4ALL v2.0?

NIRS4ALL is a Python library for **Near-Infrared Spectroscopy (NIRS) data analysis**. It provides machine learning pipelines for classification and regression, supporting multiple backends (scikit-learn, TensorFlow, PyTorch, JAX).

**v2.0** is a ground-up redesign that:
- Separates concerns into **three distinct layers**
- Introduces a **DAG-based execution model** for pipelines
- Provides **full reproducibility** through lineage tracking
- Maintains **backward compatibility** with v1 user syntax

---

## Why a Full Rewrite?

| Current Problem | Root Cause | v2.0 Solution |
|-----------------|------------|---------------|
| Data mutations are hard to track | In-place updates | **Immutable blocks** with Copy-on-Write |
| Branching/merging is complex | Context copying everywhere | **DAG with explicit fork/join** nodes |
| Cross-validation leaks are possible | Manual partition management | **Views enforce partition boundaries** |
| Caching is ad-hoc | No lineage tracking | **Hash-based block identification** |
| sklearn integration is awkward | Different execution models | **NIRSEstimator wraps DAG** execution |

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                 │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Static API  │  │ sklearn Compat  │  │   CLI / Config      │  │
│  │ run/predict │  │ NIRSEstimator   │  │   YAML pipelines    │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     DAG EXECUTION ENGINE                         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ DAG Builder  │  │ Node Executor │  │  Artifact Manager    │  │
│  │ from syntax  │  │  parallel/seq │  │  cache & lineage     │  │
│  └──────────────┘  └───────────────┘  └──────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  FeatureBlockStore                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ FeatureBlock│  │ FeatureBlock│  │ FeatureBlock    │   │   │
│  │  │ (NIR specs) │  │ (markers)   │  │ (predictions)   │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ SampleRegistry│  │ ViewResolver  │  │  TargetStore        │  │
│  └───────────────┘  └───────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flows upward** for reads, **commands flow downward** for execution.

---

## Layer 1: Data Layer

The Data Layer provides **unified data storage** with Copy-on-Write semantics and lazy evaluation.

### Components Overview

| Component | Purpose | Key Capability |
|-----------|---------|----------------|
| **FeatureBlock** | Stores feature arrays | Immutable, 3D (samples × processings × features), lineage-tracked |
| **FeatureBlockStore** | Central registry of blocks | Append-only, hash-based lookup, garbage collection |
| **SampleRegistry** | Tracks sample identity | Polars-backed, manages partitions, folds, groups, repetitions |
| **TargetStore** | Manages target variables (y) | Multi-version, bidirectional transform chains |
| **ViewSpec** | Specifies data subsets | Declarative, lazy evaluation |
| **ViewResolver** | Materializes views | Converts ViewSpec → actual arrays |
| **DatasetContext** | Bundles everything | Single object for pipeline execution |

### FeatureBlock

A **FeatureBlock** is an immutable container for feature data.

```
┌─────────────────────────────────────────────┐
│              FeatureBlock                    │
├─────────────────────────────────────────────┤
│  data: 3D array (samples × processings × features) │
│  headers: feature names (e.g., wavelengths)  │
│  source_name: "NIR", "markers", etc.         │
│  lineage_hash: unique identifier from history │
│  parent_hash: link to parent block           │
│  transform_info: what created this block     │
└─────────────────────────────────────────────┘
```

**Key properties:**
- **Logically immutable**: transformations create new blocks
- **Copy-on-Write**: memory is shared until modification
- **Lineage-tracked**: every block knows its complete history

### SampleRegistry

The **SampleRegistry** is a Polars DataFrame tracking sample metadata.

| Column | Type | Purpose |
|--------|------|---------|
| `sample_id` | String | Unique identifier |
| `partition` | Enum | train / validation / test |
| `fold_id` | Int | Cross-validation fold assignment |
| `group` | String | Group for grouped CV |
| `aggregation_key` | String | For sample repetitions |
| `origin` | Enum | original / augmented / pseudo |
| `excluded` | Bool | Soft-delete flag |

### TargetStore

The **TargetStore** manages target variables with transformation tracking.

```
TargetStore
├── versions: {"raw", "scaled", "encoded", ...}
├── transformers: [fitted scaler, encoder, ...]
└── inverse_transform(): restore original units
```

**Why it matters**: When you scale y with `MinMaxScaler`, predictions come out scaled. `TargetStore` automatically inverse-transforms to original units.

### ViewSpec and ViewResolver

**ViewSpec** declaratively describes what data you want:

```
ViewSpec
├── block_ids: which blocks to include
├── partition: train / validation / test
├── fold_id: which CV fold
├── branch_id: which pipeline branch
├── sample_filter: custom Polars expression
└── aggregation: mean / median / none
```

**ViewResolver** materializes views lazily—arrays are only created when accessed.

### DatasetContext

**DatasetContext** bundles everything needed for pipeline execution:

```
DatasetContext
├── store: FeatureBlockStore
├── registry: SampleRegistry
├── targets: TargetStore
├── resolver: ViewResolver
└── metadata: dict
```

---

## Layer 2: DAG Execution Engine

The DAG Engine compiles pipeline syntax into a **Directed Acyclic Graph** and executes it.

### Execution Flow

```
[Pipeline Syntax] → [DAG Builder] → [Executable DAG] → [Execution Engine] → [Results]
```

1. **Parse**: Convert user syntax to normalized form
2. **Expand**: Generators (`_or_`, `_range_`) create multiple DAG variants
3. **Build**: Create nodes and edges
4. **Execute**: Topological sort → run nodes → collect results

### Node Types

| Node Type | Purpose | Example |
|-----------|---------|---------|
| **SOURCE** | Entry point | Load data |
| **TRANSFORM** | Apply transformer | `MinMaxScaler`, `SNV` |
| **MODEL** | Train/predict | `PLSRegression`, `RandomForest` |
| **SPLITTER** | Assign CV folds | `KFold`, `ShuffleSplit` |
| **FORK** | Create branches | Start parallel paths |
| **JOIN** | Merge branches | Combine branch outputs |
| **FUSION** | Combine NN towers | Mid-level fusion |
| **SINK** | Terminal | Collect results |

### DAG Node Structure

```
DAGNode
├── id: unique identifier
├── type: NodeType enum
├── operator: the actual transformer/model
├── parents: list of parent node IDs
├── children: list of child node IDs
├── branch_id: which branch this belongs to
└── config: node-specific configuration
```

### Branching and Merging

**Fork/Join pattern** for parallel preprocessing:

```
              ┌─── [SNV] ───┐
[MinMax] ─── FORK           JOIN ─── [Model]
              └─── [MSC] ───┘
```

- **FORK**: Creates parallel execution paths
- **JOIN**: Combines outputs (features or predictions)

**Merge modes**:
- `features`: Concatenate features horizontally
- `predictions`: Use out-of-fold predictions (stacking)
- `mixed`: Combine features from some branches, predictions from others

### Generator Expansion

**Generators** create multiple pipeline variants before execution:

| Syntax | Meaning | Result |
|--------|---------|--------|
| `{"_or_": [A, B, C]}` | Try each option | 3 separate pipelines |
| `{"_range_": [1, 10, 2]}` | Parameter sweep | 5 pipelines (1, 3, 5, 7, 9) |
| `{"count": 5}` | Limit variants | Max 5 pipelines |

### VirtualModel

After cross-validation, a **VirtualModel** wraps all fold models:

```
VirtualModel
├── fold_models: [model_fold0, model_fold1, ...]
├── weights: [1.0, 1.0, ...]  # or validation-score-based
├── aggregation: mean / median / vote
├── predict(X): aggregate predictions
└── predict_with_std(X): include uncertainty
```

**Why not just one model?**
- Uncertainty estimation from fold disagreement
- Stacking/ensemble support
- Per-fold interpretability

### Artifact Management

**ArtifactManager** handles:
- Serialization of trained models and transformers
- Hash-based caching (same inputs → skip computation)
- Bundle export/import (`.n4a` format)
- Lineage tracking for reproducibility

---

## Layer 3: API Layer

The API Layer provides **three access levels**:

### 1. Static Functions (Simple)

```python
import nirs4all

# Training
result = nirs4all.run(pipeline, data)

# Prediction
predictions = nirs4all.predict(result, new_data)

# Explanation
explanations = nirs4all.explain(result, data)
```

### 2. sklearn Estimators (Integration)

```python
from nirs4all import NIRSRegressor

reg = NIRSRegressor(pipeline)
reg.fit(X, y)
predictions = reg.predict(X_new)

# Works with sklearn ecosystem
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(reg, param_grid, cv=5)
```

### 3. CLI / Config (Automation)

```bash
nirs4all run --config pipeline.yaml data/
nirs4all predict --model bundle.n4a --input new_data.csv
```

### Result Objects

#### RunResult (from training)

```
RunResult
├── best: best pipeline result
├── y_pred: predictions (in original units)
├── y_true: actual values
├── metrics: {"r2": 0.95, "rmse": 0.12, ...}
├── rankings: all pipelines ranked by score
├── artifacts: trained models and transformers
└── dag: the executed DAG for debugging
```

#### PredictResult (from inference)

```
PredictResult
├── y_pred: predictions
├── y_std: uncertainty (if available)
└── metadata: sample information
```

#### ExplainResult (from SHAP)

```
ExplainResult
├── shap_values: SHAP explanations
├── feature_names: what each column means
└── base_value: expected model output
```

### Pipeline Syntax Flexibility

All these formats are equivalent and valid:

```python
# Class reference
MinMaxScaler

# Instance
MinMaxScaler()

# Dict with keyword
{"preprocessing": MinMaxScaler()}

# Explicit class path
{"class": "sklearn.preprocessing.MinMaxScaler"}

# YAML (from file)
- class: sklearn.preprocessing.MinMaxScaler
```

---

## Key Concepts

### Copy-on-Write (CoW)

Blocks share memory until modified. When a transformation runs:
- If the block has one owner → reuse memory (in-place)
- If shared → copy then modify

**Result**: Memory efficiency without mutation bugs.

### Lineage Hashing

Every block has a **lineage hash** computed from:
- Parent block hash
- Transformation applied
- Parameters used
- Random seeds

**Result**: Two blocks with the same hash are guaranteed identical.

### Views vs. Copies

- **View**: Lightweight reference (block IDs + row/column spec)
- **Materialization**: Convert view to actual arrays

Views are cheap; materialization happens only when needed.

### Partition Boundaries

Views enforce strict separation:
- Training data never leaks into validation/test
- Fold boundaries are respected automatically
- No accidental data leakage

---

## User Experience

### Preserved from v1

| Feature | Status |
|---------|--------|
| Multiple syntax forms | ✅ All supported |
| Branching pipelines | ✅ Improved with DAG |
| Multi-source fusion | ✅ Native support |
| Optuna tuning | ✅ Integrated |
| SHAP explanations | ✅ Via explain() |
| Bundle export | ✅ .n4a format |

### New in v2

| Feature | Benefit |
|---------|---------|
| Immutable data | No mutation bugs |
| DAG execution | Clear execution model |
| Lineage tracking | Full reproducibility |
| Hash-based caching | Skip redundant work |
| VirtualModel | Proper ensemble handling |
| sklearn estimators | Native ecosystem integration |

### Simple Example

```python
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Define pipeline
pipeline = [
    MinMaxScaler(),
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]

# Run
result = nirs4all.run(pipeline, "data/spectra.csv")

# Access results
print(f"Best R²: {result.metrics['r2']:.3f}")
print(f"Predictions: {result.y_pred}")
```

### Advanced Example (Branching)

```python
pipeline = [
    MinMaxScaler(),
    {"branch": [
        [SNV(), FirstDerivative()],
        [MSC(), SecondDerivative()],
    ]},
    {"merge": "predictions"},  # Stack with OOF predictions
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=10)}
]
```

---

## Implementation Roadmap

### Phase 0: Foundation (Weeks 1-5)

- Protocol interfaces
- FeatureBlock core
- SampleRegistry
- TargetStore
- **Risk mitigation spikes** (CoW, GC, threading)

### Phase 1: Data Layer (Weeks 3-5)

- FeatureBlockStore
- ViewSpec / ViewResolver
- DatasetContext
- Multi-source support

### Phase 2: DAG Engine (Weeks 4-7)

- DAG model and node types
- DAG Builder (syntax → graph)
- Execution engine
- Generator expansion

### Phase 3: Model Management (Weeks 6-8)

- VirtualModel
- Aggregation strategies
- ArtifactManager
- Bundle format

### Phase 4: API Layer (Weeks 8-10)

- Static API (run, predict, explain)
- Result objects
- sklearn estimators

### Phase 5: Operators (Weeks 9-11)

- Port NIRS transforms (SNV, MSC, etc.)
- Port splitters
- Port models

### Success Criteria

- [ ] All v1 examples (Q1-Q36) work unchanged
- [ ] Performance equal or better than v1
- [ ] sklearn integration passes compatibility tests
- [ ] Numerical equivalence within tolerance (r² ± 0.001)

---

## Glossary

| Term | Definition |
|------|------------|
| **Block** | Immutable container for feature data |
| **CoW** | Copy-on-Write: share memory until mutation |
| **DAG** | Directed Acyclic Graph: nodes connected by edges, no cycles |
| **Fork** | Create parallel execution branches |
| **Generator** | Syntax that expands to multiple pipeline variants |
| **Join** | Merge parallel branches back together |
| **Lineage** | Complete transformation history of a block |
| **Materialize** | Convert a lazy view into actual arrays |
| **OOF** | Out-of-Fold: predictions on validation data during CV |
| **Partition** | Data split: train / validation / test |
| **View** | Lightweight reference to a data subset |
| **VirtualModel** | Wrapper aggregating multiple fold models |

---

## Document References

For implementation details, see:

| Document | Content |
|----------|---------|
| [01_architecture_overview.md](01_architecture_overview.md) | Full architecture, design philosophy, error handling |
| [02_data_layer.md](02_data_layer.md) | FeatureBlock, Store, Registry, Views implementation |
| [03_dag_engine.md](03_dag_engine.md) | DAG model, node types, execution, generators |
| [04_api_layer.md](04_api_layer.md) | Static API, sklearn estimators, CLI |
| [05_implementation_plan.md](05_implementation_plan.md) | Phases, testing, migration, risks |

---

*This document provides a complete overview of NIRS4ALL v2.0 architecture. For code-level details, consult the referenced design documents.*

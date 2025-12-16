# Meta-Model Stacking Strategy

> **Note**: This document defines the meta-model stacking feature design. The artifact management aspects of stacking (serialization, dependency tracking) should use the V3 artifact system.
> 
> See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for:
> - Section 4.1: `ArtifactRecordV3.depends_on` for source model dependencies
> - Section 9.3: `MetaModelControllerV3` for chain-based persistence
> - Section 8.2: `MinimalPredictorV3` for prediction replay with meta-models

---

## Executive Summary

This document specifies the implementation of **meta-model stacking** in nirs4all: a feature enabling users to train models that use predictions from previously trained pipeline models as input features. This represents a key advancement in building sophisticated ensemble architectures where a meta-learner combines outputs from multiple base models.

---

## V3 Integration Points

The stacking strategy integrates with the V3 artifact system in these ways:

### 1. Source Model Chains

In V3, each source model has an `OperatorChain` that captures its full execution path:

```python
# Source chains for meta-model
source_chains = [
    OperatorChain(nodes=[...]),  # PLS chain: s1.Scaler > s3.PLS[br=0]
    OperatorChain(nodes=[...]),  # RF chain: s1.Scaler > s4.RF[br=0]
]

# Meta-model chain includes source chains
meta_chain_path = "s1.Scaler>s3.PLS[br=0]+s1.Scaler>s4.RF[br=0]>s5.Ridge"
```

### 2. Dependency Tracking

V3's `DependencyGraphV3` tracks meta-model dependencies:

```python
# In MetaModelControllerV3._persist_meta_model()
depends_on = []
for source_chain in source_chains:
    source_record = registry.get_by_chain(source_chain)
    depends_on.append(source_record.artifact_id)

record = registry.register(
    obj=meta_model,
    chain=meta_chain,
    artifact_type=ArtifactType.META_MODEL,
    depends_on=depends_on  # Explicit dependencies
)
```

### 3. Prediction Access for Stacking

V3's `PredictionDBV3` provides chain-based prediction access:

```python
def get_predictions_for_meta_model(
    self,
    source_chains: List[OperatorChain]
) -> Dict[str, List[PredictionRecordV3]]:
    """Get OOF predictions from all source models for stacking."""
    result = {}
    for chain in source_chains:
        result[chain.to_path()] = self.get_predictions_for_chain(chain)
    return result
```

---

## Original Strategy Document

*The following is the core stacking strategy document, which remains valid. The artifact management aspects should be implemented using V3 conventions.*

---

## 1. Objective Definition

### 1.1 Problem Statement

In machine learning, **stacking** (or stacked generalization) is an ensemble technique where:
1. **Level-0 models** (base models) are trained on the original features
2. **Level-1 model** (meta-model) is trained on the predictions of the base models

The primary challenge is **preventing data leakage**: if the meta-model trains on predictions made by base models on the same samples they were trained on, overfitting occurs. The solution is to use **out-of-fold (OOF) predictions**.

### 1.2 Core Objective

**Enable users to train a meta-model on the predictions of other models already trained in the current pipeline, using a reconstructed training set built from fold predictions, without data leakage.**

Specifically:
- Base models train normally, storing predictions in `prediction_store`
- Meta-model constructs its training features from **validation partition predictions across all folds**
- The reconstruction must be **sample-aligned**: each sample's feature comes from a fold where it was NOT used for training
- The meta-model can then be serialized and used for prediction on new data

### 1.3 Key Requirements

| Requirement | Description |
|-------------|-------------|
| **No Leakage** | Meta-model training uses ONLY out-of-fold predictions |
| **Sample Alignment** | Training set reconstructed with correct sample indices |
| **Branch Compatibility** | Works with branching, including sample partitioning |
| **Flexible Source Selection** | User can select which models to include in the stack |
| **Serialization** | Complete save/load cycle with dependency tracking (V3 artifact system) |
| **Cross-Framework** | Works with sklearn, TF, PyTorch, JAX base models |

---

## 2. Branch Compatibility with V3

V3's chain-based approach simplifies branch handling for meta-models:

### 2.1 Within-Branch Stacking

When a meta-model stacks within its branch, the source chains are filtered:

```python
# Current branch: [0] (SNV branch)
target_chain = OperatorChain(nodes=[...])  # Includes branch_path=[0]

# Source models are filtered by branch
source_chains = [
    chain for chain in all_chains
    if chain.filter_branch([0]).to_path() == chain.to_path()
]
```

### 2.2 Cross-Branch Stacking (Future)

V3 enables cross-branch stacking through explicit chain references:

```python
# Meta-model stacking predictions from both branches
source_chains = [
    OperatorChain(...),  # s3.PLS[br=0] - Branch 0 predictions
    OperatorChain(...),  # s3.RF[br=1] - Branch 1 predictions
]

# V3 chain path captures this:
# "s3.PLS[br=0]+s3.RF[br=1]>s4.Ridge"
```

---

## 3. Implementation Notes

See the original strategy sections below for:
- Source model selection strategies
- Training set reconstruction logic
- Coverage strategies for partial data
- Serialization format

These remain valid and should be implemented using V3's chain-based artifact system.

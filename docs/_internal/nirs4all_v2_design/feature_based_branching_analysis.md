# Feature-Based Branching Analysis

## Executive Summary

This document analyzes the current state of feature-based branching (splitting spectra into wavelength regions like VIS, NIR1, NIR2, NIR3) in nirs4all and evaluates how this functionality integrates with the new workflow operators design (v2).

**Key Findings:**
1. **Feature-based branching does NOT currently exist** as a native concept in nirs4all
2. The current "source" concept refers to **different data types** (e.g., NIR + markers), not wavelength regions
3. The new v2 design focuses entirely on **sample-based** operations (tags, exclude, branch by samples)
4. **Minimal impact on roadmap** - feature-based branching can be implemented as a new operator that converts one source to multiple sources, working with existing infrastructure

---

## 1. Current State Analysis

### 1.1 What is a "Source" in nirs4all?

A **source** is a **completely separate feature matrix** from a different data type or instrument, NOT a wavelength region:

```
Source concept in nirs4all:
┌─────────────────────────────────────────────────┐
│ SpectroDataset                                  │
├─────────────────────────────────────────────────┤
│ Source 0: NIR spectra    (100 samples × 2048)   │
│ Source 1: Lab markers    (100 samples × 10)     │
│ Source 2: Raman spectra  (100 samples × 1024)   │
└─────────────────────────────────────────────────┘
                    ↓
        Concatenated: (100 samples × 3082)
```

**Key files:**
- [dataset.py](nirs4all/data/dataset.py) - SpectroDataset manages multiple FeatureSource objects
- [features.py](nirs4all/data/features.py) - Features class coordinates N aligned sources
- [feature_source.py](nirs4all/data/_features/feature_source.py) - Single 3D feature matrix

### 1.2 What is a "Branch" in nirs4all?

Branches in nirs4all have two modes, both operating on **samples**, not features:

#### A. Duplication Branches (All samples → all branches)
Every sample goes through every branch with different preprocessing:

```python
{"branch": [
    [SNV()],              # Branch 0: all 100 samples
    [MSC()],              # Branch 1: all 100 samples
    [FirstDerivative()],  # Branch 2: all 100 samples
]}
```

#### B. Separation Branches (Samples partitioned → different branches)
Samples are split into non-overlapping partitions:

```python
{"branch": {"by_tag": "quality"}}      # By tag value
{"branch": {"by_metadata": "site"}}    # By metadata column
{"branch": {"by_filter": Filter()}}    # By filter pass/fail
{"branch": {"by_source": True}}        # Per-source preprocessing
```

### 1.3 Existing Feature-Related Operators

| Operator | Purpose | Creates Branches? |
|----------|---------|------------------|
| **CARS** | Wavelength selection via PLS weights | No (reduces features) |
| **MCUVE** | Uninformative variable elimination | No (reduces features) |
| **iPLS** | Evaluates spectral intervals | No (identifies optimal region) |
| **sklearn VarianceThreshold** | Removes low-variance features | No (reduces features) |
| **sklearn SelectKBest** | Selects top K features | No (reduces features) |

**Key observation:** All existing operators **reduce** features rather than **branch** by feature regions.

### 1.4 What Feature-Based Branching Would Mean

The requested functionality would split a single spectrum into wavelength regions and process each region through its own branch:

```
Requested feature-based branching:
┌─────────────────────────────────────────────────────────────┐
│ Single spectrum: 400-2500 nm (2100 features)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ VIS: 400-700nm  │ │ NIR1: 700-1100  │ │ NIR2: 1100-2500 nm  │
│ (300 features)  │ │ (400 features)  │ │ (1400 features)     │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
       ↓                   ↓                      ↓
    Branch 0            Branch 1              Branch 2
    (VIS preproc)       (NIR1 preproc)        (NIR2 preproc)
```

---

## 2. Why Tags Cannot Be Used for Feature-Based Branching

### 2.1 Tags Are Sample-Centric

The v2 design defines tags as **sample metadata**:

```
Tag storage in IndexStore:
| sample | partition | excluded | tag:y_outlier | tag:cluster_id |
|--------|-----------|----------|---------------|----------------|
| 0      | train     | False    | False         | "A"            |
| 1      | train     | False    | True          | "B"            |
| 2      | test      | False    | False         | "A"            |
```

Tags answer: "Which samples have property X?"
- `tag:y_outlier = True` → samples 1, 5, 23 are y-outliers
- `tag:cluster_id = "A"` → samples 0, 2, 7 belong to cluster A

### 2.2 Features Are Not Tagged

There is no equivalent system for features:

```
Features are stored as arrays, not indexed:
X[sample_idx, feature_idx] = spectral_value

There is no:
| feature | wavelength | region    | selected |
|---------|------------|-----------|----------|
| 0       | 400 nm     | VIS       | True     |
| 1       | 401 nm     | VIS       | True     |
| ...     | ...        | ...       | ...      |
| 2099    | 2500 nm    | NIR2      | False    |
```

### 2.3 Why This Design Choice Makes Sense

Sample-based operations are the core of machine learning pipelines:
- **Training/validation/test splits** - sample partitions
- **Outlier removal** - sample exclusion
- **Cross-validation** - sample folds
- **Stratification** - sample grouping

Feature operations are typically:
- **Reduction** (PCA, feature selection)
- **Transformation** (derivatives, scaling)
- **Selection** (wavelength bands)

These are fundamentally different operations.

---

## 3. Constraints and Challenges

### 3.1 Preprocessing That Destroys Feature Correspondence

Some preprocessing steps **lose** the original feature structure:

| Operation | Feature Correspondence |
|-----------|----------------------|
| SNV, MSC | **Preserved** - output features map 1:1 to input |
| Derivatives | **Preserved** - slight shift but same dimension |
| **PCA** | **Lost** - n_components features, no wavelength meaning |
| **CARS** | **Lost** - selected subset, non-contiguous |
| **Autoencoders** | **Lost** - latent space, no wavelength meaning |

**Problem:** If we apply PCA before feature-based branching, we cannot split by wavelength regions because features no longer correspond to wavelengths.

### 3.2 Multi-Source Multiplication

Current branch behavior multiplies with sources:

```python
# 2 sources × 3 branches = 6 paths
pipeline = [
    {"branch": [[SNV()], [MSC()], [Detrend()]]},  # 3 branches
    PLSRegression(10),  # Applied 6 times!
]
```

Feature-based branching would need to **not multiply** but instead operate on the feature dimension within each source.

### 3.3 Merge Strategy

After feature-based branching, the merge strategy depends on the use case:

| Use Case | Merge Strategy |
|----------|---------------|
| Separate models per region | `concat` (predictions) or no merge |
| Feature-level fusion | `features` (horizontal concat) |
| Model selection | `best` (select optimal region) |
| Ensemble | `average` or `voting` |

The current merge strategies assume sample-based operations.

---

## 4. Proposed Solutions

### Solution A: Wavelength Region Extractor Operator (Recommended)

Create a **transformer operator** that converts one source into multiple sources based on wavelength regions:

```python
class WavelengthRegionExtractor(TransformerMixin, BaseEstimator):
    """Splits a single spectrum into multiple sources by wavelength region."""

    def __init__(self, regions: Dict[str, Tuple[float, float]]):
        """
        Parameters
        ----------
        regions : dict
            Mapping of region name to (start_wavelength, end_wavelength).
            Example: {"VIS": (400, 700), "NIR1": (700, 1100), "NIR2": (1100, 2500)}
        """
        self.regions = regions

    def fit(self, X, y=None):
        # Get wavelength headers from context
        # Calculate feature indices for each region
        return self

    def transform(self, X):
        # Returns list of X matrices, one per region
        # Or: modifies dataset to have multiple sources
        return [X[:, region_indices] for region_indices in self.region_indices_]
```

**Usage in pipeline:**

```python
pipeline = [
    WavelengthRegionExtractor({
        "VIS": (400, 700),
        "NIR1": (700, 1100),
        "NIR2": (1100, 2500),
    }),
    # Now dataset has 3 sources
    {"branch": {
        "by_source": True,
        "steps": {
            "VIS": [SNV(), PCA(5)],
            "NIR1": [MSC(), SavitzkyGolay()],
            "NIR2": [Detrend(), FirstDerivative()],
        }
    }},
    {"merge": {"sources": "concat"}},
    PLSRegression(10),
]
```

**Advantages:**
- Works with existing `by_source` infrastructure
- No changes to branch/merge controllers
- Explicit operator with clear semantics
- Compatible with current roadmap

**Disadvantages:**
- Requires dataset modification (converting features to sources)
- Must be placed before any feature-destroying preprocessing

### Solution B: Native `by_feature_region` Branch Mode

Add a new separation branch mode for feature regions:

```python
{"branch": {
    "by_feature_region": {
        "VIS": (400, 700),
        "NIR1": (700, 1100),
        "NIR2": (1100, 2500),
    },
    "steps": {
        "VIS": [SNV()],
        "NIR1": [MSC()],
        "NIR2": [Detrend()],
    }
}}
```

**Implementation in BranchController:**

```python
def _execute_by_feature_region(self, branch_def, dataset, context):
    """Branch by feature regions (wavelength bands)."""
    regions = branch_def.get("by_feature_region", {})
    steps = branch_def.get("steps", {})

    # Get wavelength headers
    headers = dataset.get_headers(source_idx=0)

    # For each region:
    for name, (start_wl, end_wl) in regions.items():
        # 1. Find feature indices
        mask = (headers >= start_wl) & (headers <= end_wl)
        feature_indices = np.where(mask)[0]

        # 2. Create temporary dataset with subset features
        # 3. Execute branch steps
        # 4. Store results with feature region metadata
```

**Advantages:**
- Native syntax, clearer intent
- No need to convert to sources
- Could support dynamic feature grouping

**Disadvantages:**
- Requires changes to BranchController
- Needs new merge handling for feature-based branches
- More complex implementation

### Solution C: Feature Tagging System

Create a parallel tagging system for features (not recommended due to complexity):

```python
# Hypothetical feature tags
dataset.add_feature_tag("region", ["VIS"] * 300 + ["NIR1"] * 400 + ["NIR2"] * 1400)

{"branch": {"by_feature_tag": "region"}}
```

**Why this is NOT recommended:**
- Doubles the complexity of the indexing system
- Features don't have individual identities like samples
- Most feature operations are array-based, not index-based

---

## 5. Impact on Current Roadmap

### 5.1 If Using Solution A (Operator Approach)

**Impact: MINIMAL - No roadmap changes required**

| Phase | Status | Impact |
|-------|--------|--------|
| Phase 1: Indexer Tags | Complete | No impact |
| Phase 2: Tag Controller | Complete | No impact |
| Phase 3: Exclude Controller | Complete | No impact |
| Phase 4: Branch Controller | Complete | No impact - `by_source` already exists |
| Phase 5: Merge Controller | Complete | No impact - `sources` merge already exists |
| Phase 6-8: Tests/Docs | Pending | Add examples for wavelength region extraction |

**Additional work:**
- Create `WavelengthRegionExtractor` operator in `nirs4all/operators/transforms/`
- Add controller support if operator needs special handling
- Write documentation and examples

### 5.2 If Using Solution B (Native Branch Mode)

**Impact: MODERATE - Requires Phase 4 additions**

| Phase | Status | Impact |
|-------|--------|--------|
| Phase 1-3 | Complete | No impact |
| Phase 4: Branch Controller | Complete | **Add `_execute_by_feature_region()` method** |
| Phase 5: Merge Controller | Complete | **May need `feature_concat` strategy** |
| Phase 6-8: Tests/Docs | Pending | Add tests for feature-based branching |

**Additional work:**
- Add `by_feature_region` to `SEPARATION_KEYWORDS` in branch.py
- Implement `_execute_by_feature_region()` handler
- Update merge controller for feature-based branch outputs
- Add comprehensive tests

---

## 6. Recommendation

### Primary Recommendation: Solution A (Operator Approach)

**Rationale:**
1. **Minimal disruption** - Works with existing infrastructure
2. **Clear semantics** - Explicit operator in pipeline
3. **Flexible** - Can be combined with any downstream branching
4. **Already have the pattern** - Multi-source handling is mature

**Proposed implementation:**

```python
# nirs4all/operators/transforms/region_extractor.py

class WavelengthRegionExtractor(TransformerMixin, BaseEstimator):
    """
    Splits a single source into multiple sources by wavelength regions.

    This operator converts contiguous wavelength bands into separate sources,
    enabling per-region preprocessing via `by_source` branching.

    Parameters
    ----------
    regions : dict
        Mapping of region name to wavelength range.
        Format: {"name": (start_nm, end_nm)} or {"name": slice}
        Example: {"VIS": (400, 700), "NIR1": (700, 1100)}
    source_idx : int, default=0
        Index of source to split.
    keep_original : bool, default=False
        If True, keep original source alongside split regions.

    Examples
    --------
    >>> from nirs4all.operators.transforms import WavelengthRegionExtractor
    >>>
    >>> pipeline = [
    ...     WavelengthRegionExtractor({
    ...         "VIS": (400, 700),
    ...         "NIR": (700, 2500),
    ...     }),
    ...     {"branch": {
    ...         "by_source": True,
    ...         "steps": {
    ...             "VIS": [SNV()],
    ...             "NIR": [MSC(), FirstDerivative()],
    ...         }
    ...     }},
    ...     {"merge": {"sources": "concat"}},
    ...     PLSRegression(10),
    ... ]
    """
```

### Secondary Recommendation: Document the Constraint

If a user needs wavelength-region branching but has already applied PCA:

```python
# THIS WILL NOT WORK - PCA destroys wavelength correspondence
pipeline = [
    PCA(n_components=50),  # Features are now PC scores, not wavelengths!
    WavelengthRegionExtractor(...)  # Cannot split by wavelength anymore
]
```

**Documentation should clearly state:**
> WavelengthRegionExtractor must be applied BEFORE any transformation that destroys feature correspondence (PCA, autoencoders, non-contiguous feature selection).

---

## 7. Implementation Plan

### Phase 1: Create WavelengthRegionExtractor Operator

**Files to create:**
- `nirs4all/operators/transforms/region_extractor.py`

**API:**
```python
class WavelengthRegionExtractor:
    def __init__(
        self,
        regions: Dict[str, Tuple[float, float]],
        source_idx: int = 0,
        unit: str = "nm",
        tolerance: float = 0.5,
    ): ...

    def fit(self, X, y=None, context=None): ...
    def transform(self, X, context=None): ...  # Returns multi-source dataset
```

### Phase 2: Create Controller (If Needed)

If the operator needs special context handling (accessing wavelength headers), create:
- `nirs4all/controllers/transforms/region_extractor.py`

### Phase 3: Update Exports and Tests

**Files to update:**
- `nirs4all/operators/transforms/__init__.py`
- `nirs4all/operators/__init__.py`

**Files to create:**
- `tests/unit/operators/transforms/test_region_extractor.py`
- `tests/integration/pipeline/test_wavelength_region_branching.py`

### Phase 4: Documentation and Examples

**Files to create:**
- `examples/developer/01_advanced_pipelines/D08_wavelength_region_branching.py`
- `docs/source/tutorials/wavelength_regions.rst`

---

## 8. Future Considerations

### 8.1 Dynamic Region Detection

Instead of fixed regions, detect optimal regions automatically:

```python
class OptimalRegionDetector:
    """Uses iPLS-style analysis to identify optimal wavelength regions."""

    def fit(self, X, y):
        # Evaluate all possible region combinations
        # Select regions with best CV performance
        pass
```

### 8.2 Overlapping Regions

For techniques like sliding window analysis:

```python
WavelengthRegionExtractor({
    "window_1": (400, 800),
    "window_2": (600, 1000),  # Overlaps with window_1
    "window_3": (800, 1200),
})
```

This requires different merge handling (samples would have multiple values per feature).

### 8.3 Non-Contiguous Regions

For methods like CARS that select non-contiguous wavelengths:

```python
WavelengthRegionExtractor({
    "selected": [401, 405, 423, 567, 890, 1234],  # Specific wavelengths
})
```

---

## Conclusion

Feature-based branching is **not currently supported** in nirs4all but can be implemented with **minimal impact on the v2 roadmap** using the operator approach. The recommended `WavelengthRegionExtractor` operator converts wavelength regions into separate sources, enabling the existing `by_source` branching infrastructure to handle per-region preprocessing.

**Key takeaways:**
1. Tags are for samples, not features - this is the correct design choice
2. Solution A (operator) is recommended over native branch support
3. No changes required to Phases 1-5 of the current roadmap
4. Document the constraint that region extraction must precede feature-destroying transformations

---

*Document version: 1.0*
*Created: 2026-01-20*
*Related documents:*
- `workflows_operator_design_v2.md`
- `ROADMAP_workflow_v2.md`

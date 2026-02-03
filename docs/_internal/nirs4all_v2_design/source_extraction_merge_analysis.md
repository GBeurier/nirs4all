# Source Extraction and Merge: Feature-Level Operations Analysis

## Executive Summary

This document analyzes the current implementation of feature sources in nirs4all, how merge operations work at the feature level, and proposes solutions for wavelength region extraction (splitting a single spectrum into VIS, NIR1, NIR2, etc.).

**Key Findings:**
1. **Current merge is "hard" splitting** - Sources are real separate arrays, merge physically creates/destroys them
2. **No virtual view system exists** - All operations work on actual numpy arrays
3. **X() virtualization already exists** - When calling `dataset.x()` with `concat_source=True`, sources are concatenated on-the-fly
4. **Region extraction is missing** - No operator to split one source into multiple by wavelength ranges

---

## 1. Current Source Architecture

### 1.1 Data Structure Hierarchy

```
SpectroDataset
├── _feature_accessor: FeatureAccessor
│   └── _block: Features
│       └── sources: List[FeatureSource]
│           └── _storage: ArrayStorage
│               └── _array: np.ndarray (n_samples, n_processings, n_features)
│           └── _processing_mgr: ProcessingManager
│           └── _header_mgr: HeaderManager (wavelengths, units)
└── _indexer: Indexer (sample metadata, tags)
```

### 1.2 Source Storage Model

Each `FeatureSource` contains:
- **3D numpy array**: `(n_samples, n_processings, n_features)`
- **Processing IDs**: e.g., `["raw", "SNV", "MSC"]`
- **Headers**: Wavelength values or feature names
- **Header unit**: `"nm"`, `"cm-1"`, `"text"`, `"index"`, `"none"`

```python
# Example: Single source with 100 samples, 2 preprocessings, 2048 wavelengths
FeatureSource:
  _array: (100, 2, 2048)
  processing_ids: ["raw", "SNV"]
  headers: ["400.0", "400.5", ..., "2500.0"]  # 2048 values
  header_unit: "nm"
```

### 1.3 Multi-Source Dataset

Multiple sources are stored as separate `FeatureSource` objects:

```python
# Multi-source dataset: NIR + Lab markers
Features:
  sources[0]: FeatureSource  # NIR spectra
    _array: (100, 2, 2048)   # 100 samples, 2 preprocessings, 2048 wavelengths
    headers: ["400.0", ..., "2500.0"]
    header_unit: "nm"

  sources[1]: FeatureSource  # Lab markers
    _array: (100, 1, 10)     # 100 samples, 1 preprocessing, 10 markers
    headers: ["pH", "temp", ...]
    header_unit: "text"
```

---

## 2. How X() Works: Virtual Concatenation

### 2.1 Default Behavior (Concatenation)

When you call `dataset.x()`, sources are **concatenated on-the-fly**:

```python
# In features.py:240-296
def x(self, indices, layout="2d", concat_source=True):
    res = []
    for src in self.sources:
        res.append(src.x(indices, layout))

    if concat_source and len(res) > 1:
        return np.concatenate(res, axis=res[0].ndim - 1)  # Concat on last axis

    if len(res) == 1:
        return res[0]
    return res  # List of arrays
```

**Key insight:** This is already a form of "virtualization" - the concatenation is computed dynamically, not stored.

### 2.2 Per-Source Access

```python
# Get all sources concatenated (default)
X = dataset.x(concat_source=True)   # (100, 4106) = 2048*2 + 10

# Get sources separately
X_list = dataset.x(concat_source=False)  # [array(100, 4096), array(100, 10)]
```

---

## 3. How Merge Works: Hard Operations

### 3.1 Source Merge (`merge_sources`)

The `merge_sources` keyword **physically combines** multiple sources:

**File:** [merge.py:4009-4178](nirs4all/controllers/data/merge.py#L4009-L4178)

```python
# Three strategies
SourceMergeStrategy.CONCAT   # Horizontal concatenation -> 2D
SourceMergeStrategy.STACK    # Stack along new axis -> 3D
SourceMergeStrategy.DICT     # Keep as dict for multi-input models
```

#### Concat Strategy (Default)

```python
# Before merge_sources
sources[0]: (100, 2048)  # NIR
sources[1]: (100, 10)    # Markers

# After {"merge_sources": "concat"}
# dataset.add_merged_features() is called:
sources[0]: (100, 2058)  # Concatenated
# sources[1] is REMOVED (keep_sources is called)
```

### 3.2 Branch Feature Merge

When branches with different preprocessing are merged:

```python
# Pipeline
{"branch": [[SNV()], [MSC()]]}
{"merge": "features"}
PLS()

# Before merge
Branch 0: sources[0] = (100, 1, 2048)  # SNV processed
Branch 1: sources[0] = (100, 1, 2048)  # MSC processed

# After merge (features concatenated)
sources[0]: (100, 1, 4096)  # SNV + MSC features side by side
```

### 3.3 add_merged_features: The Core Operation

**File:** [dataset.py:225-295](nirs4all/data/dataset.py#L225-L295)

This method **REPLACES** all existing features:

```python
def add_merged_features(self, features, processing_name="merged", source=0):
    # Replace ALL existing processings with merged output
    self._feature_accessor.reset_features(
        features=features,
        processings=processings,
        source=source
    )
```

**Key observation:** Merge is **destructive** - it replaces the original feature structure.

### 3.4 keep_sources: Source Removal

**File:** [features.py:240-266](nirs4all/data/features.py#L240-L266)

```python
def keep_sources(self, source_indices):
    """Keep only specified sources, removing all others."""
    self.sources = [self.sources[i] for i in source_indices]
```

This physically removes FeatureSource objects from the list.

---

## 4. Hard Splitting vs Virtual Splitting

### 4.1 Option A: Hard Splitting (Current Pattern)

**Approach:** Create new `FeatureSource` objects, remove the original.

```
Before split:
sources[0]: (100, 2, 2100)  # Full spectrum 400-2500nm

After split (VIS, NIR1, NIR2):
sources[0]: (100, 2, 300)   # VIS   400-700nm
sources[1]: (100, 2, 400)   # NIR1  700-1100nm
sources[2]: (100, 2, 1400)  # NIR2  1100-2500nm
```

**Pros:**
- Consistent with existing multi-source architecture
- Each region has its own headers, processing chain
- Works with `by_source` branching immediately

**Cons:**
- Destructive - original source structure lost
- Memory duplication during split
- Can't easily "unsplit" or see original view
- Merge must physically recombine

**Implementation:**

```python
class WavelengthRegionSplitter:
    def transform(self, dataset):
        # 1. Get source features and headers
        X = dataset.x(concat_source=False)[0]  # Original source
        headers = dataset.headers(0)

        # 2. Create new FeatureSource for each region
        for name, (start, end) in self.regions.items():
            mask = (headers >= start) & (headers <= end)
            X_region = X[:, mask]
            headers_region = headers[mask]

            # 3. Add as new source
            new_source = FeatureSource()
            new_source.add_samples(X_region, headers_region)
            dataset._feature_accessor._block.sources.append(new_source)

        # 4. Remove original source
        dataset.keep_sources(list(range(1, len(self.regions) + 1)))
```

### 4.2 Option B: Virtual Splitting (View-Based)

**Approach:** Keep original data, provide slice views on access.

```
Storage (unchanged):
sources[0]: (100, 2, 2100)  # Full spectrum

Virtual regions (computed on x() call):
region_views:
  "VIS":  slice(0, 300)    -> view into sources[0][:, :, 0:300]
  "NIR1": slice(300, 700)  -> view into sources[0][:, :, 300:700]
  "NIR2": slice(700, 2100) -> view into sources[0][:, :, 700:2100]
```

**Pros:**
- Non-destructive - original structure preserved
- Memory efficient - no data duplication
- Can toggle between split and full view
- Undo is trivial (remove views)

**Cons:**
- Requires new infrastructure (view registry)
- Headers/processing tracking becomes complex
- `by_source` branching would need adaptation
- More complex implementation

**Proposed API:**

```python
# Virtual regions don't create new sources
dataset.add_virtual_regions({
    "VIS": (400, 700),
    "NIR1": (700, 1100),
    "NIR2": (1100, 2500),
}, source=0)

# Access by virtual region name
X_vis = dataset.x(region="VIS")           # (100, 300)
X_nir1 = dataset.x(region="NIR1")         # (100, 400)

# Or get all regions as dict
X_dict = dataset.x(by_region=True)        # {"VIS": array, "NIR1": array, ...}

# Original still accessible
X_full = dataset.x()                       # (100, 2100)

# Remove virtual regions
dataset.remove_virtual_regions(source=0)
```

**Implementation sketch:**

```python
class Features:
    def __init__(self):
        self.sources: List[FeatureSource] = []
        self._virtual_regions: Dict[int, Dict[str, slice]] = {}  # source_idx -> {name: slice}

    def add_virtual_regions(self, regions: Dict[str, Tuple[float, float]], source: int = 0):
        headers = self.headers(source)
        header_values = np.array([float(h) for h in headers])

        self._virtual_regions[source] = {}
        for name, (start, end) in regions.items():
            mask = (header_values >= start) & (header_values <= end)
            indices = np.where(mask)[0]
            self._virtual_regions[source][name] = slice(indices[0], indices[-1] + 1)

    def x(self, indices, layout="2d", region=None, by_region=False, **kwargs):
        if region is not None:
            # Return specific region
            src_regions = self._virtual_regions.get(0, {})
            if region in src_regions:
                X = self.sources[0].x(indices, layout)
                return X[..., src_regions[region]]

        if by_region:
            # Return dict of all regions
            return {name: self.x(indices, layout, region=name)
                    for name in self._virtual_regions.get(0, {})}

        # Default: full data
        return self._default_x(indices, layout, **kwargs)
```

---

## 5. Presets and Index-Based Splitting

### 5.1 Wavelength Presets

Common spectral region definitions:

```python
SPECTRAL_PRESETS = {
    # Visible
    "VIS": (400, 700),

    # Near-Infrared
    "NIR": (700, 2500),
    "NIR1": (700, 1100),   # Short-wave NIR
    "NIR2": (1100, 1800),  # Mid-wave NIR
    "NIR3": (1800, 2500),  # Long-wave NIR

    # Combined
    "VISNIR": (400, 2500),

    # Water absorption bands
    "H2O_1": (1400, 1500),  # First overtone
    "H2O_2": (1900, 2100),  # Combination band

    # Organic bands
    "CH": (2200, 2400),     # C-H stretching region
    "OH": (1400, 1500),     # O-H first overtone
}
```

**Usage:**

```python
# By preset name
WavelengthRegionExtractor(regions=["VIS", "NIR1", "NIR2"])

# By custom ranges
WavelengthRegionExtractor(regions={
    "low": (400, 1200),
    "high": (1200, 2500),
})

# Mix of preset and custom
WavelengthRegionExtractor(regions={
    "VIS": "preset",           # Use preset definition
    "custom_nir": (800, 1500), # Custom range
})
```

### 5.2 Index-Based Splitting

For non-spectroscopic data or when wavelength headers are unavailable:

```python
# By feature indices
WavelengthRegionExtractor(
    regions={
        "first_half": (0, 1024),    # Indices 0-1023
        "second_half": (1024, 2048) # Indices 1024-2047
    },
    mode="index"  # Use indices instead of wavelength values
)

# By percentage
WavelengthRegionExtractor(
    regions={
        "first_third": "0%..33%",
        "middle_third": "33%..66%",
        "last_third": "66%..100%",
    },
    mode="percent"
)
```

---

## 6. Cross-Source Splitting Edge Cases

### 6.1 Problem: Splitting Across Source Boundaries

When you have multiple sources with different wavelength ranges:

```
Source 0: NIR (700-2500 nm)   - 1800 features
Source 1: VIS (400-700 nm)    - 300 features
```

A split request for `(400, 1200)` spans both sources.

### 6.2 Solution: Source-Aware Splitting

**Option A: Error on cross-source split**

```python
WavelengthRegionExtractor(
    regions={"combined": (400, 1200)},
    cross_source="error"  # Raise error
)
# Raises: "Region (400, 1200) spans sources [0, 1]. Use cross_source='merge' to combine."
```

**Option B: Automatic merge before split**

```python
WavelengthRegionExtractor(
    regions={"combined": (400, 1200)},
    cross_source="merge"  # Auto-merge sources first
)
# 1. Concatenates Source 0 + Source 1 based on wavelength order
# 2. Then extracts region (400, 1200)
```

**Option C: Per-source partial extraction**

```python
WavelengthRegionExtractor(
    regions={"combined": (400, 1200)},
    cross_source="partial"  # Extract from each source where available
)
# Returns: {"combined": concat(VIS[400-700], NIR[700-1200])}
```

### 6.3 Wavelength Ordering

When sources have overlapping or non-ordered wavelengths:

```python
# Source 0: 1000-2500 nm
# Source 1: 400-1200 nm (overlaps with source 0)

WavelengthRegionExtractor(
    cross_source="merge",
    overlap_handling="average"  # Average overlapping regions
    # or "first", "last", "error"
)
```

---

## 7. Merge After Split: Round-Trip Behavior

### 7.1 Current Merge Behavior

```python
# After split into VIS, NIR1, NIR2
sources[0]: (100, 300)   # VIS
sources[1]: (100, 400)   # NIR1
sources[2]: (100, 1400)  # NIR2

# After {"merge_sources": "concat"}
sources[0]: (100, 2100)  # Back to full
# But: headers may be lost or recomputed
# But: processing history is lost
```

### 7.2 Virtual Merge (Proposal)

With virtual regions, merge could be a simple view removal:

```python
# Virtual split
dataset.add_virtual_regions({"VIS": ..., "NIR1": ..., "NIR2": ...})

# "Merge" = just remove virtual regions
dataset.remove_virtual_regions()
# Original structure preserved, no data loss
```

---

## 8. Recommendation

### 8.1 Short-Term: Hard Splitting Operator

For immediate use with existing infrastructure:

```python
class WavelengthRegionExtractor(TransformerMixin, BaseEstimator):
    """
    Splits a single source into multiple sources by wavelength regions.

    Uses HARD splitting - creates new FeatureSource objects.
    Works with existing by_source branching.
    """

    def __init__(
        self,
        regions: Union[Dict[str, Tuple[float, float]], List[str]],
        source: int = 0,
        mode: str = "wavelength",  # "wavelength", "index", "percent"
        presets: bool = True,      # Allow preset names
        keep_original: bool = False,
        cross_source: str = "error",
    ):
        self.regions = regions
        self.source = source
        self.mode = mode
        self.presets = presets
        self.keep_original = keep_original
        self.cross_source = cross_source
```

**Pipeline usage:**

```python
pipeline = [
    WavelengthRegionExtractor({
        "VIS": (400, 700),
        "NIR1": (700, 1100),
        "NIR2": (1100, 2500),
    }),
    {"branch": {
        "by_source": True,
        "steps": {
            "VIS": [SNV()],
            "NIR1": [MSC(), SavitzkyGolay()],
            "NIR2": [Detrend(), FirstDerivative()],
        }
    }},
    {"merge": {"sources": "concat"}},
    PLSRegression(10),
]
```

### 8.2 Medium-Term: Virtual Region System

Add view-based region access to Features class:

1. Add `_virtual_regions: Dict[int, Dict[str, RegionSpec]]` to `Features`
2. Add `add_virtual_regions()` and `remove_virtual_regions()` methods
3. Extend `x()` to support `region=` and `by_region=` parameters
4. Modify `by_source` branching to optionally use virtual regions

**Benefits:**
- Non-destructive
- Memory efficient
- Preserves original structure for analysis
- Easy undo

### 8.3 Integration with v2 Design

**No changes to v2 roadmap required** - wavelength region operations are orthogonal to sample-based tagging/branching:

| Concept | Operates On | v2 Design |
|---------|-------------|-----------|
| Tags | Samples | Complete |
| Exclude | Samples | Complete |
| Branch (duplication) | Samples | Complete |
| Branch (separation) | Samples | Complete |
| **Region extraction** | **Features** | **New operator** |
| Merge (features) | Branches | Complete |
| Merge (sources) | Sources | Complete |

---

## 9. Implementation Roadmap

### Phase 1: Hard Splitting Operator

**Files to create:**
- `nirs4all/operators/transforms/region_extractor.py`

**Files to modify:**
- `nirs4all/operators/transforms/__init__.py` (export)
- `nirs4all/operators/__init__.py` (export)

**Tasks:**
1. Implement `WavelengthRegionExtractor` class
2. Support wavelength, index, and percent modes
3. Add preset definitions
4. Handle cross-source edge cases
5. Write unit tests
6. Write integration tests with `by_source` branching

### Phase 2: Virtual Region System (Optional)

**Files to modify:**
- `nirs4all/data/features.py` (add virtual region tracking)
- `nirs4all/data/_dataset/feature_accessor.py` (region access API)
- `nirs4all/data/dataset.py` (expose region API)

**Tasks:**
1. Add `_virtual_regions` dict to Features class
2. Implement `add_virtual_regions()` / `remove_virtual_regions()`
3. Extend `x()` method with region parameters
4. Optional: Modify `by_source` to use virtual regions

### Phase 3: Controller Integration (Optional)

If region extraction needs special pipeline handling:

**Files to create:**
- `nirs4all/controllers/transforms/region_extractor.py`

---

## Appendix A: Current Merge Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MERGE_SOURCES WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Parse config: strategy, sources, on_incompatible            │
│                                                                  │
│  2. Validate multi-source dataset (n_sources >= 2)              │
│                                                                  │
│  3. Collect features from each source                           │
│     X = dataset.x(concat_source=False)  # List[ndarray]         │
│                                                                  │
│  4. Apply merge strategy:                                        │
│     CONCAT: np.concatenate(source_features, axis=-1)            │
│     STACK:  np.stack(source_features, axis=1)                   │
│     DICT:   {name: features for name, features in ...}          │
│                                                                  │
│  5. Store merged features:                                       │
│     dataset.add_merged_features(merged)  # REPLACES all!        │
│                                                                  │
│  6. Remove old sources (implicit via reset_features)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: Feature Access Patterns

```python
# Pattern 1: Default (concatenated)
X = dataset.x()                    # All samples, all sources concat
# Shape: (n_samples, sum(n_features_per_source))

# Pattern 2: Per-source list
X_list = dataset.x(concat_source=False)
# Returns: [source_0_features, source_1_features, ...]

# Pattern 3: With selector
X_train = dataset.x({"partition": "train"})

# Pattern 4: 3D layout
X_3d = dataset.x(layout="3d")      # (n_samples, n_processings, n_features)

# Pattern 5: Specific source (proposed)
X_nir = dataset.x(source=0)        # Only first source
X_markers = dataset.x(source=1)    # Only second source

# Pattern 6: Virtual region (proposed)
X_vis = dataset.x(region="VIS")    # Virtual region view
```

---

*Document version: 1.0*
*Created: 2026-01-20*
*Related documents:*
- `feature_based_branching_analysis.md`
- `workflows_operator_design_v2.md`
- `ROADMAP_workflow_v2.md`

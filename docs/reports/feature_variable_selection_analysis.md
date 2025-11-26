# Feature Management & Variable Selection Analysis Report

## Executive Summary

This report analyzes the nirs4all feature management system and proposes design changes to support variable selection methods (VIP, MCUVE, CARS, SPA from `auswahl`) that reduce feature dimensionality.

---

## 1. Current Status and Architecture Overview

### 1.1 Core Components

The feature management system is built on a **modular, component-based architecture**:

```
Features (Facade)
  └── List[FeatureSource]
        ├── ArrayStorage         # 3D numpy array (samples, processings, features)
        ├── ProcessingManager    # Tracks processing IDs and indices
        ├── HeaderManager        # Feature headers and units
        ├── LayoutTransformer    # 2D/3D layout conversions
        ├── UpdateStrategy       # Categorizes add/replace operations
        └── AugmentationHandler  # Sample augmentation logic
```

### 1.2 Data Structure

Features are stored in a **3D array**: `(n_samples, n_processings, n_features)`

- **Axis 0 (samples)**: Each row is a sample (spectrum)
- **Axis 1 (processings)**: Different preprocessing versions (raw, SNV, SG, etc.)
- **Axis 2 (features)**: Feature values (wavelengths/channels)

**Key constraint**: All processings within a source MUST have the same number of features.

### 1.3 Layout Transformations

The `LayoutTransformer` supports four output formats:

| Layout | Shape | Use Case |
|--------|-------|----------|
| `2d` | (samples, processings × features) | ML models (flattened, concatenated) |
| `2d_interleaved` | (samples, features × processings) | ML models (interleaved) |
| `3d` | (samples, processings, features) | DL models (CNN, LSTM) |
| `3d_transpose` | (samples, features, processings) | DL models (channels-last) |

### 1.4 Multi-Source Support

nirs4all supports multiple data sources (e.g., multiple sensors, multi-block PLS):

```python
Features
  └── sources[0]: FeatureSource  # NIR sensor 1 (n_samples, n_processings_0, n_features_0)
  └── sources[1]: FeatureSource  # NIR sensor 2 (n_samples, n_processings_1, n_features_1)
```

Each source can have **independent feature dimensions and processing chains**.

### 1.5 Preprocessing Pipeline (TransformerMixinController)

When a `TransformerMixin` operator is applied in the pipeline:

1. Data is extracted in 3D format: `(samples, processings, features)`
2. Each processing slice `(samples, features)` is transformed **independently**
3. Results are stored as new processings (feature augmentation) or replace existing ones
4. The transformer is cloned for each processing to allow different fits

**Current flow**:
```
Train 3D → For each processing_idx:
              train_2d = X[:, processing_idx, :]
              transformer.fit(train_2d)
              transformed = transformer.transform(all_2d)
           → Store results
```

### 1.6 Padding Mechanism (Existing)

`ArrayStorage` already supports **padding for smaller features**:

```python
class ArrayStorage:
    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        self.padding = padding
        self.pad_value = pad_value

    def _prepare_data_for_storage(self, data: np.ndarray) -> np.ndarray:
        if self.padding and data.shape[1] < self.num_features:
            padded_data = np.full((data.shape[0], self.num_features), self.pad_value, ...)
            padded_data[:, :data.shape[1]] = data
            return padded_data
```

However, this padding is **only applied when adding new samples**, not when updating processings with different feature counts.

### 1.7 Feature Resize Mechanism (Existing but Limited)

`UpdateStrategy.should_resize_features()` checks if all processings are being replaced with a new dimension:

```python
def should_resize_features(replacements, additions, current_num_features):
    if replacements and not additions:
        new_feature_dims = [op.new_data.shape[1] for op in replacements]
        if len(set(new_feature_dims)) == 1 and new_feature_dims[0] != current_num_features:
            return True, new_feature_dims[0]
    return False, current_num_features
```

This resize **clears headers** and is only used when ALL processings are replaced with the same new dimension.

---

## 2. Impact of Variable Selection Methods

### 2.1 What Variable Selection Does

Methods like VIP, MCUVE, CARS, SPA are `TransformerMixin` that:
1. **Fit**: Analyze feature importance using training data
2. **Transform**: Return a **reduced-dimension** subset of features

Example:
```python
from auswahl import VIP, CARS, SPA, MCUVE

# Input: X shape (100, 500)  → 500 wavelengths
selector = VIP(pls_kwargs=dict(n_components=8)).fit(X_train, y_train)
X_selected = selector.transform(X)
# Output: X_selected shape (100, 50)  → 50 selected wavelengths
```

### 2.2 Core Problem

**Variable selection produces outputs with different feature dimensions** than inputs:

```
Processing "raw":        (samples, 500)  ← Original wavelengths
Processing "VIP_sel":    (samples, 50)   ← Selected wavelengths
```

This violates the current constraint that all processings must have the same feature count.

### 2.3 Cascade Effects

| Component | Current Behavior | Issue with Variable Selection |
|-----------|------------------|------------------------------|
| `ArrayStorage` | Fixed feature dimension for all processings | Cannot store mixed-dimension processings |
| `LayoutTransformer` | Assumes uniform `num_features` | `2d` concatenation produces wrong shapes |
| `HeaderManager` | Single header list | Selected features have subset of headers |
| `TransformerMixinController` | Stores results with same shape | Would crash or produce wrong results |
| `3d` layouts | Stacked along axis 1 | Requires all slices same shape |

### 2.4 ML vs DL Layout Implications

| Layout | Variable Selection Impact |
|--------|---------------------------|
| **2D (ML)** | Works if we handle concatenation of different sizes or use only selected features |
| **2D Interleaved** | Same as above, different order |
| **3D (DL)** | **Cannot stack** different-sized processings without padding |
| **3D Transpose** | Same issue as 3D |

---

## 3. Design Proposals

### 3.1 Approach A: Per-Processing Feature Dimensions (Recommended)

**Concept**: Allow each processing to have its own feature dimension, stored as a **list of 2D arrays** or **dictionary of arrays** instead of a single 3D array.

**Implementation**:

```python
class FlexibleArrayStorage:
    """New storage that supports variable feature dimensions per processing."""
    
    def __init__(self):
        # Dict: processing_name -> 2D array (samples, features_for_this_processing)
        self._arrays: Dict[str, np.ndarray] = {}
        self._processing_order: List[str] = []
    
    @property
    def feature_dims(self) -> Dict[str, int]:
        """Return feature dimension for each processing."""
        return {name: arr.shape[1] for name, arr in self._arrays.items()}
    
    def get_2d(self, processing_name: str, sample_indices: np.ndarray) -> np.ndarray:
        return self._arrays[processing_name][sample_indices]
    
    def get_3d_padded(self, sample_indices: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
        """For DL: pad to max feature dim and stack."""
        max_features = max(arr.shape[1] for arr in self._arrays.values())
        stacked = []
        for name in self._processing_order:
            arr = self._arrays[name][sample_indices]
            if arr.shape[1] < max_features:
                padded = np.full((len(sample_indices), max_features), pad_value)
                padded[:, :arr.shape[1]] = arr
                arr = padded
            stacked.append(arr)
        return np.stack(stacked, axis=1)
```

**Pros**:
- Most flexible, no data loss
- Each processing can have natural dimension
- Headers can be per-processing
- Backward compatible for uniform dimensions

**Cons**:
- More complex implementation
- Memory overhead for metadata
- Need to handle padding masks for DL

### 3.2 Approach B: Preprocessing Groups with Uniform Dimensions

**Concept**: Group processings by feature dimension. Each group is a 3D array. The layout selector chooses which group(s) to use.

```python
class GroupedArrayStorage:
    """Groups processings by feature dimension."""
    
    def __init__(self):
        # Dict: feature_dim -> (3D array, processing_names)
        self._groups: Dict[int, Tuple[np.ndarray, List[str]]] = {}
```

**Pros**:
- Maintains 3D structure for DL
- Clear separation of dimension groups

**Cons**:
- Less flexible
- Complexity in managing groups
- Which group to use for model input?

### 3.3 Approach C: Variable Selection as Terminal Transformation

**Concept**: Variable selection is applied **only for model input**, not stored as a processing. The selection mask is stored separately.

```python
class FeatureSelector:
    """Wraps variable selection for model input only."""
    
    def __init__(self, selector: TransformerMixin):
        self.selector = selector
        self.selected_indices_: np.ndarray = None
    
    def fit(self, X, y):
        self.selector.fit(X, y)
        self.selected_indices_ = self.selector.get_support(indices=True)
        return self
    
    def transform(self, X):
        # Returns reduced X, but NOT stored in dataset
        return X[:, self.selected_indices_]
```

The `TransformerMixinController` would detect this type and:
1. Store the selector but NOT add to dataset processings
2. Apply selection at model training/prediction time

**Pros**:
- Minimal architecture changes
- No storage of reduced data
- Headers/features remain intact

**Cons**:
- Variable selection not visible in dataset preprocessing
- Selection applied dynamically each time
- More complex controller logic

### 3.4 Approach D: Hybrid - Tracked Selection with Optional Storage

**Concept**: Combine A and C. Store selection masks/indices, optionally store reduced data.

```python
class FeatureSource:
    def __init__(self):
        self._storage = ArrayStorage()  # Main storage (uniform dim)
        self._selections: Dict[str, SelectionInfo] = {}  # Selection masks
        
@dataclass
class SelectionInfo:
    source_processing: str           # Which processing was selected from
    selected_indices: np.ndarray     # Feature indices that were selected
    selected_headers: List[str]      # Subset of headers
    stored_data: Optional[np.ndarray] = None  # Optional: actual reduced data
```

**Pros**:
- Flexible storage decisions
- Maintains selection provenance
- Can reconstruct full data if needed

**Cons**:
- Most complex implementation
- Two parallel tracking systems

---

## 4. Recommended Design: Approach A with Enhancements

### 4.1 Core Changes

#### 4.1.1 New `FlexibleArrayStorage`

Replace or extend `ArrayStorage` to support per-processing dimensions:

```python
class FlexibleArrayStorage:
    """Supports variable feature dimensions per processing."""
    
    def __init__(self, padding: bool = True, pad_value: float = 0.0):
        self.padding = padding
        self.pad_value = pad_value
        self._arrays: Dict[str, np.ndarray] = {}  # name -> (samples, features)
        self._processing_order: List[str] = ["raw"]
        self._arrays["raw"] = np.empty((0, 0), dtype=np.float32)
    
    @property
    def num_samples(self) -> int:
        if not self._arrays:
            return 0
        return next(iter(self._arrays.values())).shape[0]
    
    @property
    def max_features(self) -> int:
        if not self._arrays:
            return 0
        return max(arr.shape[1] for arr in self._arrays.values())
    
    @property
    def feature_dims(self) -> Dict[str, int]:
        return {name: arr.shape[1] for name, arr in self._arrays.items()}
    
    def add_processing(self, name: str, data: np.ndarray) -> None:
        """Add processing with any feature dimension."""
        if name in self._arrays:
            raise ValueError(f"Processing '{name}' already exists")
        self._arrays[name] = data.astype(np.float32)
        self._processing_order.append(name)
    
    def get_data(self, sample_indices: np.ndarray, 
                 processings: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get data for specific processings."""
        if processings is None:
            processings = self._processing_order
        return {name: self._arrays[name][sample_indices] for name in processings}
    
    def get_3d_padded(self, sample_indices: np.ndarray,
                      processings: Optional[List[str]] = None) -> np.ndarray:
        """Get 3D array with padding for DL models."""
        data = self.get_data(sample_indices, processings)
        max_feat = max(arr.shape[1] for arr in data.values())
        
        result = []
        for name in (processings or self._processing_order):
            arr = data[name]
            if arr.shape[1] < max_feat:
                padded = np.full((arr.shape[0], max_feat), self.pad_value)
                padded[:, :arr.shape[1]] = arr
                result.append(padded)
            else:
                result.append(arr)
        
        return np.stack(result, axis=1)  # (samples, processings, max_features)
    
    def get_2d_concat(self, sample_indices: np.ndarray,
                      processings: Optional[List[str]] = None) -> np.ndarray:
        """Get 2D concatenated array for ML models."""
        data = self.get_data(sample_indices, processings)
        return np.concatenate([data[name] for name in (processings or self._processing_order)], axis=1)
```

#### 4.1.2 Per-Processing Headers

Extend `HeaderManager` to support per-processing headers:

```python
class FlexibleHeaderManager:
    """Manages headers per processing."""
    
    def __init__(self):
        self._headers: Dict[str, List[str]] = {}  # processing_name -> headers
        self._header_units: Dict[str, HeaderUnit] = {}
    
    def set_headers(self, processing: str, headers: List[str], unit: str = "cm-1"):
        self._headers[processing] = headers
        self._header_units[processing] = normalize_header_unit(unit)
    
    def get_headers(self, processing: str) -> Optional[List[str]]:
        return self._headers.get(processing)
    
    def derive_headers(self, source_processing: str, 
                       target_processing: str, 
                       selected_indices: np.ndarray) -> None:
        """Create headers for selected features from source processing."""
        source_headers = self._headers.get(source_processing)
        if source_headers:
            self._headers[target_processing] = [source_headers[i] for i in selected_indices]
            self._header_units[target_processing] = self._header_units.get(
                source_processing, HeaderUnit.INDEX
            )
```

#### 4.1.3 Updated LayoutTransformer

```python
class FlexibleLayoutTransformer:
    """Transforms with support for variable feature dimensions."""
    
    @staticmethod
    def transform(storage: FlexibleArrayStorage, 
                  sample_indices: np.ndarray,
                  layout: LayoutType,
                  processings: Optional[List[str]] = None,
                  pad_value: float = 0.0) -> np.ndarray:
        
        layout_enum = normalize_layout(layout)
        
        if layout_enum == FeatureLayout.FLAT_2D:
            # Concatenate all processings (different sizes OK)
            return storage.get_2d_concat(sample_indices, processings)
        
        elif layout_enum == FeatureLayout.VOLUME_3D:
            # Pad to uniform size for 3D stacking
            return storage.get_3d_padded(sample_indices, processings)
        
        elif layout_enum == FeatureLayout.VOLUME_3D_TRANSPOSE:
            data_3d = storage.get_3d_padded(sample_indices, processings)
            return np.transpose(data_3d, (0, 2, 1))
        
        # ... other layouts
```

### 4.2 TransformerMixinController Updates

```python
class TransformerMixinController(OperatorController):
    
    def execute(self, step_info, dataset, context, runtime_context, ...):
        op = step_info.operator
        
        # Detect if this is a variable selector
        is_selector = hasattr(op, 'get_support') or hasattr(op, 'selected_indices_')
        
        for sd_idx, (train_x, all_x) in enumerate(zip(train_data, all_data)):
            for processing_idx in range(train_x.shape[1]):
                # ... existing fit/transform logic ...
                transformed_2d = transformer.transform(all_2d)
                
                # If dimensions differ, handle appropriately
                if is_selector and transformed_2d.shape[1] != train_2d.shape[1]:
                    # Store selected indices for header derivation
                    selected_indices = transformer.get_support(indices=True)
                    
                    # Derive headers from source processing
                    dataset._features.sources[sd_idx]._header_mgr.derive_headers(
                        source_processing=processing_name,
                        target_processing=new_processing_name,
                        selected_indices=selected_indices
                    )
```

### 4.3 Sparsity and Padding Masks

For DL models using 3D layouts with padded data, add mask support:

```python
class FlexibleArrayStorage:
    
    def get_3d_with_mask(self, sample_indices: np.ndarray,
                         processings: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D padded data with validity mask."""
        data = self.get_data(sample_indices, processings)
        max_feat = max(arr.shape[1] for arr in data.values())
        
        result = []
        mask = []  # 1 = valid, 0 = padding
        
        for name in (processings or self._processing_order):
            arr = data[name]
            n_feat = arr.shape[1]
            
            if n_feat < max_feat:
                padded = np.full((arr.shape[0], max_feat), self.pad_value)
                padded[:, :n_feat] = arr
                result.append(padded)
                
                m = np.zeros((arr.shape[0], max_feat))
                m[:, :n_feat] = 1
                mask.append(m)
            else:
                result.append(arr)
                mask.append(np.ones((arr.shape[0], max_feat)))
        
        return np.stack(result, axis=1), np.stack(mask, axis=1)
```

### 4.4 Backward Compatibility

To maintain backward compatibility:

1. Keep `ArrayStorage` as legacy class
2. `FeatureSource` detects mixed dimensions and switches to flexible storage
3. Default behavior unchanged for uniform dimensions
4. Add deprecation warning for legacy mode

```python
class FeatureSource:
    def __init__(self, flexible: bool = False, ...):
        if flexible:
            self._storage = FlexibleArrayStorage(...)
        else:
            self._storage = ArrayStorage(...)
        self._is_flexible = flexible
    
    def update_features(self, ...):
        # Check if we need to upgrade to flexible storage
        new_dim = features[0].shape[1]
        if not self._is_flexible and new_dim != self._storage.num_features:
            self._upgrade_to_flexible_storage()
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (MVP)
1. Implement `FlexibleArrayStorage`
2. Extend `HeaderManager` to per-processing headers
3. Update `LayoutTransformer` for mixed dimensions
4. Add backward-compatible `FeatureSource` with auto-upgrade

### Phase 2: Controller Integration
1. Update `TransformerMixinController` for variable selectors
2. Add `get_support()` detection
3. Implement header derivation for selected features
4. Add padding mask support for 3D layouts

### Phase 3: Variable Selection Operators
1. Create `nirs4all/operators/transforms/feature_selection.py`
2. Wrap `auswahl` selectors (VIP, MCUVE, CARS, SPA)
3. Add unit tests
4. Add to Q19 example

### Phase 4: Testing & Documentation
1. Comprehensive unit tests for flexible storage
2. Integration tests with variable selection in pipelines
3. Update documentation
4. Add examples showing variable selection usage

---

## 6. Open Questions

1. **Padding strategy for DL**: Should padding be at start, end, or center? Should it use zeros or other values (mean, edge)?

2. **Header preservation**: When variable selection is applied, should we keep original headers as metadata for visualization/interpretation?

3. **Processing chain**: Should variable selection outputs be treated as derived processings (can be replaced) or final outputs?

4. **Multi-source variable selection**: If selecting features from multiple sources, should selections be synchronized or independent?

5. **Serialization**: How to serialize flexible storage with per-processing dimensions efficiently?

---

## 7. Conclusion

The nirs4all feature management system is well-architected but constrained by the uniform feature dimension assumption. Variable selection methods violate this constraint.

**Recommended approach**: Implement `FlexibleArrayStorage` with per-processing dimensions, supporting:
- Per-processing headers derived from selections
- Automatic padding for 3D DL layouts with mask support
- Concatenation for 2D ML layouts
- Backward compatibility for uniform dimensions

This approach provides the most flexibility while maintaining the existing API and supporting both ML (2D) and DL (3D) layouts.

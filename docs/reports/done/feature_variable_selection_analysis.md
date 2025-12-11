# Feature Management & Variable Selection Analysis Report

## Executive Summary

This report analyzes the nirs4all feature management system and proposes design changes to support variable selection methods (VIP, MCUVE, CARS, SPA from `auswahl`) that reduce feature dimensionality.

**Key Finding**: The system already supports dimension-changing transformations through:
1. `CropTransformer` / `ResampleTransformer` - change feature count
2. `Resampler` with `ResamplerController` - specialized handling for wavelength resampling
3. `resize_features()` mechanism - works when ALL processings are replaced with same new dimension

**Recommended Approach**: Simple "Resize or Pad" strategy:
- **Sequential application** → resize all processings to new dimension
- **Feature augmentation** → pad smaller processings with zeros to match largest

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

### 1.4 Existing Dimension-Changing Operators

nirs4all already has operators that change feature dimensions:

#### 1.4.1 CropTransformer
```python
class CropTransformer(BaseEstimator, TransformerMixin):
    """Crops features to a range [start:end]."""
    def transform(self, X):
        return X[:, self.start:self.end]  # Output has fewer features
```

#### 1.4.2 ResampleTransformer
```python
class ResampleTransformer(BaseEstimator, TransformerMixin):
    """Resamples features to a target count using interpolation."""
    def transform(self, X):
        # Uses scipy.interpolate.interp1d to resample
        # Output: (n_samples, num_samples) - different from input!
```

#### 1.4.3 Resampler with ResamplerController
The `Resampler` operator has a **dedicated controller** that:
1. Extracts wavelengths from dataset headers
2. Transforms to target wavelength grid
3. **Calls `replace_features()` which triggers `resize_features()`**
4. **Updates headers** after dimension change

### 1.5 How Dimension Change Currently Works

When `update_features()` is called, `should_resize_features()` checks:
- ✅ Replacing ALL processings with same new dimension → works (triggers resize)
- ❌ Adding a new processing with different dimension → fails
- ❌ Replacing only SOME processings with different dimension → fails
- ❌ Mixed dimensions in same update → fails

---

## 2. Impact of Variable Selection Methods

### 2.1 What Variable Selection Does

Methods like VIP, MCUVE, CARS, SPA are `TransformerMixin` that:
1. **Fit**: Analyze feature importance using training data
2. **Transform**: Return a **reduced-dimension** subset of features

```python
from auswahl import VIP
selector = VIP(pls_kwargs=dict(n_components=8)).fit(X_train, y_train)
X_selected = selector.transform(X)  # (100, 500) → (100, 50)
```

### 2.2 The Challenge

**Sequential application** (like Resampler) already works - all processings get transformed.

**Feature augmentation** is the problem:
```python
pipeline = [
    {"feature_augmentation": [SNV(), CropTransformer(start=0, end=50)]}
]
# Result: raw(500), SNV(500), Crop(50) → mixed dimensions!
```

---

## 3. Recommended Design: Resize or Pad Strategy

### 3.0 Overview

Keep the existing 3D array structure with uniform feature dimension. Handle dimension-changing transformations with two simple rules:

| Context | Behavior |
|---------|----------|
| **Sequential** (replace all) | Resize entire 3D array to new dimension |
| **Feature Augmentation** (add new) | Pad smaller processings to match largest |

### 3.1 Rule 1: Sequential Application → Resize All

When a dimension-changing preprocessing is applied **sequentially**, it transforms ALL existing processings and the entire 3D array resizes.

```python
# Example: Dataset with Raw(500), Haar(500), SNV(500)
pipeline = [
    Haar(),                    # raw(500) → raw(500), Haar(500)
    StandardNormalVariate(),   # → raw(500), Haar(500), SNV(500)
    VIPSelector(n_features=20) # Apply to ALL → VIP_raw(20), VIP_Haar(20), VIP_SNV(20)
]
# Result: 3D array shape (samples, 3, 20) - all resized to 20
```

**This already works** via existing `resize_features()` mechanism.

### 3.2 Rule 2: Feature Augmentation → Pad Smaller to Largest

When a dimension-changing preprocessing is added as **feature augmentation**, the 3D array keeps the size of the **largest** processing, and smaller ones are **padded with zeros**.

```python
# Example: Feature augmentation with mixed dimensions
pipeline = [
    {"feature_augmentation": [SNV(), MSC(), CropTransformer(start=0, end=50)]}
]
# Processings: raw(500), SNV(500), MSC(500), Crop(50)
# 3D array shape: (samples, 4, 500)
# - raw:  [values...500]
# - SNV:  [values...500]
# - MSC:  [values...500]
# - Crop: [values...50, 0, 0, 0, ...450 zeros]  ← padded
```

**Padding configuration**:
- **Position**: Configurable (left/right/center), default = **left** (values at start, zeros at end)
- **Value**: 0.0 (configurable)

### 3.3 Implementation Changes Required

#### 3.3.1 Update `ArrayStorage` for Padding on Processing Updates

```python
class ArrayStorage:
    def __init__(self, padding: bool = True, pad_value: float = 0.0, pad_position: str = 'left'):
        self.padding = padding
        self.pad_value = pad_value
        self.pad_position = pad_position  # 'left', 'right', 'center'

    def _prepare_data_for_storage(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for storage, padding if smaller than current feature dimension."""
        if self.num_samples == 0:
            return data

        if data.shape[1] < self.num_features:
            # Pad smaller data to match current size
            if not self.padding:
                raise ValueError(f"Feature dimension mismatch: expected {self.num_features}, got {data.shape[1]}")

            padded = np.full((data.shape[0], self.num_features), self.pad_value, dtype=self._array.dtype)

            if self.pad_position == 'left':
                padded[:, :data.shape[1]] = data  # Values at start, zeros at end
            elif self.pad_position == 'right':
                padded[:, -data.shape[1]:] = data  # Zeros at start, values at end
            elif self.pad_position == 'center':
                start = (self.num_features - data.shape[1]) // 2
                padded[:, start:start + data.shape[1]] = data

            return padded

        elif data.shape[1] > self.num_features:
            # New data is LARGER - expand array and pad existing processings
            self._expand_features(data.shape[1])

        return data.astype(self._array.dtype)

    def _expand_features(self, new_num_features: int) -> None:
        """Expand feature dimension to accommodate larger processing."""
        if new_num_features <= self.num_features:
            return

        new_shape = (self.num_samples, self.num_processings, new_num_features)
        new_array = np.full(new_shape, self.pad_value, dtype=self._array.dtype)

        # Copy existing data based on pad position
        if self.pad_position == 'left':
            new_array[:, :, :self.num_features] = self._array
        elif self.pad_position == 'right':
            new_array[:, :, -self.num_features:] = self._array
        elif self.pad_position == 'center':
            start = (new_num_features - self.num_features) // 2
            new_array[:, :, start:start + self.num_features] = self._array

        self._array = new_array
```

#### 3.3.2 Update `UpdateStrategy` to Handle Mixed Dimensions

```python
def should_resize_features(replacements, additions, current_num_features):
    """Determine resize behavior for feature updates."""

    all_new_dims = []
    if replacements:
        all_new_dims.extend([op.new_data.shape[1] for op in replacements])
    if additions:
        all_new_dims.extend([op.new_data.shape[1] for op in additions])

    if not all_new_dims:
        return False, current_num_features

    max_new_dim = max(all_new_dims)

    # Case 1: Pure replacement with uniform dimensions → resize to new dim
    if replacements and not additions:
        if len(set(all_new_dims)) == 1 and all_new_dims[0] != current_num_features:
            return True, all_new_dims[0]

    # Case 2: Additions or mixed → expand to max if larger
    if max_new_dim > current_num_features:
        return True, max_new_dim

    # Case 3: All new dims <= current → no resize, pad smaller ones
    return False, current_num_features
```

#### 3.3.3 Optional: Track Actual Feature Counts Per Processing

For future 2D trimming optimization:

```python
class ProcessingManager:
    def __init__(self):
        self._processing_ids: List[str] = [DEFAULT_PROCESSING]
        self._processing_id_to_index: Dict[str, int] = {DEFAULT_PROCESSING: 0}
        self._feature_counts: Dict[str, int] = {}  # Actual feature count per processing

    def set_feature_count(self, processing_id: str, count: int) -> None:
        self._feature_counts[processing_id] = count

    def get_feature_count(self, processing_id: str) -> Optional[int]:
        return self._feature_counts.get(processing_id)
```

### 3.4 Example Scenarios

**Scenario A: Sequential VIP (works today)**
```python
pipeline = [
    StandardNormalVariate(),     # raw(500), SNV(500)
    VIPSelector(n_features=20),  # Both → 20 features
]
# Result: 3D shape (samples, 2, 20)
# Headers: updated to selected wavelengths
```

**Scenario B: Feature Augmentation with Crop (NEW - with padding)**
```python
pipeline = [
    {"feature_augmentation": [SNV(), CropTransformer(start=100, end=200)]}
]
# Result: 3D shape (samples, 3, 500)
# - raw:  [v0, v1, ..., v499]          (500 real values)
# - SNV:  [v0, v1, ..., v499]          (500 real values)
# - Crop: [v100..v199, 0, 0, ..., 0]   (100 real + 400 zeros)
```

**Scenario C: Feature Augmentation with VIP (NEW - with padding)**
```python
pipeline = [
    StandardNormalVariate(),  # raw(500), SNV(500)
    {"feature_augmentation": VIPSelector(n_features=50)}
]
# Result: 3D shape (samples, 4, 500)
# - raw:      [v0..v499]
# - SNV:      [v0..v499]
# - VIP_raw:  [50 selected values, 450 zeros]
# - VIP_SNV:  [50 selected values, 450 zeros]
```

### 3.5 Advantages

- **Simple**: Works within existing 3D array structure
- **No new storage classes**: Modify existing `ArrayStorage` and `UpdateStrategy`
- **Backward compatible**: Existing pipelines work unchanged
- **DL-friendly**: 3D arrays remain stackable (padded with zeros)
- **ML-friendly**: 2D concatenation works (includes padding, but models handle zeros)

### 3.6 Limitations

- **Memory**: Padded arrays use more memory than strictly necessary
- **Sparsity**: Zero-padded values may slightly affect some models
- **2D optimization**: Could trim padding when extracting 2D, but adds complexity

### 3.7 Future Enhancement: 2D Trimming (Optional)

For 2D extraction, could optionally trim padding by tracking actual feature counts:

```python
def get_2d_trimmed(self, sample_indices, processings):
    """Get 2D data with padding trimmed per processing."""
    result = []
    for proc in processings:
        data = self._array[sample_indices, self._get_proc_idx(proc), :]
        actual_count = self._processing_mgr.get_feature_count(proc)
        if actual_count:
            data = data[:, :actual_count]  # Trim to actual size
        result.append(data)
    return np.concatenate(result, axis=1)
```

This is optional - zeros in ML models are usually not problematic.

---

## 4. Alternative Approaches (For Reference)

### 4.1 Follow Resampler Pattern (Simpler but No Feature Augmentation)

Create `VariableSelectionController` that replaces ALL processings:
- Uses existing `resize_features()` mechanism
- **Limitation**: Does not support feature augmentation with mixed dimensions

### 4.2 Per-Processing Feature Dimensions (Overkill)

Store each processing as separate 2D array with its own dimension:
- Most flexible, but complex implementation
- Unnecessary given the padding approach

### 4.3 Variable Selection as Terminal Transformation

Apply selection only at model input time, don't store in dataset:
- Selection not visible in dataset preprocessing
- More complex controller logic

---

## 5. Implementation Roadmap

### Phase 1: Core Padding Support
1. Update `ArrayStorage._prepare_data_for_storage()` to pad on processing updates
2. Add `_expand_features()` method for when new data is larger
3. Add `pad_position` configuration parameter
4. Update `UpdateStrategy.should_resize_features()` for mixed dimensions
5. Add unit tests for padding scenarios

**Effort**: Medium
**Result**: Feature augmentation with mixed dimensions works

### Phase 2: Variable Selection Operators
1. Create wrapper operators for `auswahl` (VIP, MCUVE, CARS, SPA)
2. Add to `nirs4all/operators/transforms/feature_selection.py`
3. Update `__init__.py` exports
4. Add to Q19 example

**Effort**: Low
**Result**: Working variable selection in pipelines

### Phase 3: Enhanced Features (Optional)
1. Track actual feature counts per processing
2. Add 2D trimming option
3. Per-processing header tracking
4. Visualization support for selected wavelengths

**Effort**: Medium
**Result**: Better memory efficiency and interpretability

---

## 6. Open Questions

1. **Default padding position**: Left (values at start) seems most intuitive for spectral data. Confirm?

2. **Header handling for padded processings**: Should headers reflect padded size or actual size?

3. **Sparsity masks**: For DL models, should we provide a mask indicating valid vs padded values?

4. **Memory optimization**: Is 2D trimming worth the complexity?

---

## 7. Conclusion

The recommended **"Resize or Pad"** strategy is:
- **Simple**: Minimal changes to existing architecture
- **Pragmatic**: Uses padding which ML/DL models handle well
- **Backward compatible**: Existing pipelines unchanged
- **Extensible**: Can add trimming/masks later if needed

Implementation requires modifying `ArrayStorage` and `UpdateStrategy` - no new classes needed.

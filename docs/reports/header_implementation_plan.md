# Header Generalization Implementation Plan - Scenario 2

## Overview
Implement metadata-aware header system with unit tracking and automatic conversion.

---

## Implementation Steps

### **Step 1: Add Unit Metadata to FeatureSource** ✅
**Goal**: Store header unit information alongside headers

**Files**: `nirs4all/dataset/feature_source.py`

**Changes**:
```python
class FeatureSource:
    def __init__(self, ...):
        self._headers: Optional[List[str]] = None
        self._header_unit: str = "cm-1"  # NEW: track unit type

    def set_headers(self, headers: List[str], unit: str = "cm-1"):
        """Set headers with unit metadata"""
        self._headers = headers
        self._header_unit = unit

    @property
    def header_unit(self) -> str:
        return self._header_unit
```

**Tests**:
- Test setting headers with default unit
- Test setting headers with custom unit
- Test that unit persists correctly

---

### **Step 2: Add Conversion Methods to Dataset** ✅
**Goal**: Provide clean API for wavelength access with automatic conversion

**Files**: `nirs4all/dataset/dataset.py`

**Changes**:
```python
def header_unit(self, src: int) -> str:
    """Get header unit for a source"""
    return self._features.sources[src].header_unit

def wavelengths_cm1(self, src: int) -> np.ndarray:
    """Get wavelengths in cm⁻¹, converting from nm if needed"""
    headers = self.headers(src)
    unit = self.header_unit(src)

    if unit == "cm-1":
        return np.array([float(h) for h in headers])
    elif unit == "nm":
        nm_values = np.array([float(h) for h in headers])
        return 10_000_000 / nm_values
    elif unit in ["none", "index"]:
        return np.arange(len(headers), dtype=float)
    else:
        raise ValueError(f"Cannot convert unit '{unit}' to wavelengths")

def wavelengths_nm(self, src: int) -> np.ndarray:
    """Get wavelengths in nm, converting from cm⁻¹ if needed"""
    headers = self.headers(src)
    unit = self.header_unit(src)

    if unit == "nm":
        return np.array([float(h) for h in headers])
    elif unit == "cm-1":
        cm1_values = np.array([float(h) for h in headers])
        return 10_000_000 / cm1_values
    elif unit in ["none", "index"]:
        return np.arange(len(headers), dtype=float)
    else:
        raise ValueError(f"Cannot convert unit '{unit}' to wavelengths")
```

**Tests**:
- Test cm⁻¹ headers (no conversion)
- Test nm → cm⁻¹ conversion
- Test cm⁻¹ → nm conversion
- Test "none" fallback to indices
- Test invalid unit raises error
- Test conversion math accuracy (780 nm = 12820.5 cm⁻¹)

---

### **Step 3: Update CSV Loader to Handle Header Units** ✅
**Goal**: Accept and pass through header unit metadata

**Files**: `nirs4all/dataset/csv_loader.py`

**Changes**:
```python
def load_csv(path, na_policy='auto', data_type='x',
             categorical_mode='auto', header_unit='cm-1', **user_params):
    """
    Loads a CSV file with header unit metadata.

    Args:
        ...existing args...
        header_unit: Unit of the headers - "cm-1", "nm", "none", "text", "index"

    Returns:
        (data, report, na_mask, headers, header_unit)
    """
    # ... existing loading logic ...

    # Extract headers
    headers = data.columns.tolist() if not data.empty else []

    # Return with unit metadata
    return data, report, na_mask_after_conversions, headers, header_unit
```

**Tests**:
- Test that header_unit is returned correctly
- Test default header_unit="cm-1"
- Test custom header_unit values
- Test that existing functionality still works

---

### **Step 4: Update load_XY to Pass Header Unit** ✅
**Goal**: Thread header unit through the loading pipeline

**Files**: `nirs4all/dataset/loader.py`

**Changes**:
```python
def load_XY(...):
    # Extract header_unit from params
    x_header_unit = x_params.pop('header_unit', 'cm-1')

    # Load X with unit
    x_df, x_report, x_na_mask, x_headers, x_unit = load_csv(
        x_path, header_unit=x_header_unit, **x_params
    )

    # Return unit alongside headers
    return (x, y, m, x_headers, m_headers, x_unit)
```

**Tests**:
- Test header unit passed from params
- Test default when not specified
- Test unit returned correctly

---

### **Step 5: Update Dataset Configuration to Store Unit** ✅
**Goal**: Persist header unit in dataset configs

**Files**:
- `nirs4all/dataset/dataset_config.py`
- `nirs4all/dataset/loader.py` (handle_data function)

**Changes**:
```python
# In handle_data():
x_params_with_unit = x_params.copy()
if 'header_unit' not in x_params_with_unit:
    x_params_with_unit['header_unit'] = config.get('header_unit', 'cm-1')

# Pass to load_XY and store in dataset
dataset.add_samples(x_train, headers=train_headers, header_unit=train_unit)
```

**Tests**:
- Test loading dataset with header_unit in config
- Test default when not specified
- Test unit stored in dataset correctly

---

### **Step 6: Update add_samples to Accept Unit** ✅
**Goal**: Allow setting unit when adding samples to dataset

**Files**: `nirs4all/dataset/dataset.py`, `nirs4all/dataset/features.py`

**Changes**:
```python
# Dataset
def add_samples(self, samples, metadata=None, headers=None,
                header_unit='cm-1', source=-1):
    self._features.add_samples(samples, headers, header_unit, source)

# Features
def add_samples(self, samples, headers=None, header_unit='cm-1', source=-1):
    if source == -1:
        source = len(self.sources) - 1
    self.sources[source].add_samples(samples, headers)
    self.sources[source].set_headers(headers, header_unit)
```

**Tests**:
- Test adding samples with unit
- Test default unit
- Test retrieving unit after adding

---

### **Step 7: Update Resampler to Use New API** ✅
**Goal**: Make resampler work with unit metadata

**Files**: `nirs4all/controllers/dataset/op_resampler.py`

**Changes**:
```python
def _extract_wavelengths(self, dataset, source_idx):
    """Extract wavelengths in cm⁻¹, converting if needed"""
    unit = dataset.header_unit(source_idx)

    if unit not in ["cm-1", "nm"]:
        raise ValueError(
            f"Resampler requires wavelength headers (cm-1 or nm), "
            f"but dataset source {source_idx} has '{unit}' headers. "
            f"Cannot resample non-wavelength data."
        )

    # Use unified method that handles conversion
    return dataset.wavelengths_cm1(source_idx)
```

**Tests**:
- Test resampler with cm⁻¹ headers
- Test resampler with nm headers (auto-converts)
- Test resampler rejects "none" headers
- Test resampler rejects "text" headers
- Test error messages are clear

---

### **Step 8: Update Visualization to Use Units** ✅
**Goal**: Show correct axis labels based on unit

**Files**: `nirs4all/controllers/chart/op_spectra_charts.py`

**Changes**:
```python
def _plot_2d_spectra(self, ax, x_sorted, y_sorted, processing_name,
                     headers=None, header_unit="cm-1"):
    if headers and len(headers) == n_features:
        try:
            x_values = np.array([float(h) for h in headers])

            # Set label based on unit
            if header_unit == "cm-1":
                x_label = 'Wavenumber (cm⁻¹)'
            elif header_unit == "nm":
                x_label = 'Wavelength (nm)'
            else:
                x_label = 'Features'
        except (ValueError, TypeError):
            x_values = np.arange(n_features)
            x_label = 'Features'
```

**Tests**:
- Test plot with cm⁻¹ shows correct label
- Test plot with nm shows correct label
- Test plot falls back gracefully

---

### **Step 9: Add Comprehensive Integration Tests** ✅
**Goal**: Ensure entire pipeline works end-to-end

**Files**: `nirs4all/tests/dataset/test_header_units.py` (new)

**Tests**:
1. Load CSV with nm headers → check stored correctly
2. Load CSV with cm⁻¹ headers → check stored correctly
3. Load CSV with no headers → check "none" unit
4. Convert nm to cm⁻¹ → verify math
5. Convert cm⁻¹ to nm → verify math
6. Run resampler with nm input → verify works
7. Run resampler with cm⁻¹ input → verify works
8. Run resampler with "none" → verify error
9. Multi-source with mixed units → verify each source independent
10. Full pipeline with preprocessing → verify units preserved

---

### **Step 10: Add Real-World Example** ✅
**Goal**: Demonstrate usage in practical scenario

**Files**: `nirs4all/examples/example_header_units.py` (new)

**Example**:
```python
# Example 1: Load nm dataset
config_nm = {
    'train_x': 'data_nm.csv',
    'train_y': 'y.csv',
    'header_unit': 'nm'
}

# Example 2: Resample nm data
pipeline = [
    Resampler(target_wavelengths=np.linspace(1000, 2500, 100)),
    # Works automatically - converts nm to cm⁻¹ internally
    PLSRegression(n_components=10)
]
```

---

## Testing Strategy

### Unit Tests (After Each Step)
- Test individual functions in isolation
- Mock dependencies
- Fast execution (<1s per test)

### Integration Tests (After Step 9)
- Test complete workflows
- Use synthetic datasets
- Cover edge cases

### Manual Validation (Final)
- Test with real NIR datasets in both units
- Verify visualizations look correct
- Check error messages are helpful

---

## Implementation Order

**Day 1**: Steps 1-3 (Core data structures)
- ✅ FeatureSource unit metadata
- ✅ Dataset conversion methods
- ✅ CSV loader updates
- 🧪 Run tests after each

**Day 2**: Steps 4-6 (Loading pipeline)
- ✅ load_XY threading
- ✅ Config persistence
- ✅ add_samples unit handling
- 🧪 Run integration tests

**Day 3**: Steps 7-8 (Consumers)
- ✅ Resampler adaptation
- ✅ Visualization updates
- 🧪 Test with real data

**Day 4**: Steps 9-10 (Validation)
- ✅ Comprehensive test suite
- ✅ Example code
- 📝 Documentation updates

---

## Success Criteria

### Must Have:
- ✅ Headers stored with unit metadata
- ✅ Conversion methods work correctly
- ✅ Resampler works with nm and cm⁻¹
- ✅ No breaking changes to existing code
- ✅ All tests pass

### Nice to Have:
- ✅ Clear error messages
- ✅ Example code
- ✅ Visualization labels adapt
- 📝 Updated documentation

---

## Rollback Plan

If issues arise:
1. All changes are in feature branch
2. Each step is independent
3. Can revert to any previous step
4. Existing tests provide safety net

---

## Next Steps After Completion

1. Update documentation (RESAMPLER.md, etc.)
2. Add migration guide for users
3. Consider auto-detection (Scenario 3 feature)
4. Gather feedback from usage

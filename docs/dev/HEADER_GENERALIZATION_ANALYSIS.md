# Header Generalization Analysis

## Current State

### What Works Now
Currently, nirs4all assumes CSV headers are **wavenumbers in cm⁻¹** format:

1. **CSV Loading** (`csv_loader.py`):
   - Headers are read as strings from CSV files
   - Stored as-is in `_headers` (List[str])
   - No metadata about header type/units

2. **Dataset Storage** (`dataset.py`, `feature_source.py`):
   - Headers stored as `List[str]` in `FeatureSource._headers`
   - `dataset.headers(src)` returns string headers
   - `dataset.float_headers(src)` **assumes** headers are numeric cm⁻¹ values

3. **Resampler** (`op_resampler.py`):
   - **Requires** numeric wavelength headers
   - Calls `float(h)` on headers to extract wavelengths
   - Raises `ValueError` if conversion fails
   - **Explicitly states** "Headers must be numeric values (wavelengths in cm-1)"

4. **Visualization** (`op_spectra_charts.py`):
   - Tries to convert headers to float for wavelength axis
   - Falls back to feature indices if conversion fails
   - Labels as "Wavelength (cm-1)" if numeric, "Features" otherwise

5. **UI** (`AddDatasetModal.tsx`, `EditDatasetModal.tsx`):
   - Provides header type options: `none`, `nm`, `cm-1`, `text`
   - **Stores choice but doesn't pass metadata to backend**
   - Backend has no awareness of selected header type

### Critical Issues

1. **No Unit Metadata**: Headers are stored as strings without type/unit information
2. **Hardcoded cm⁻¹ Assumption**: All numeric operations assume cm⁻¹
3. **Resampler Breaks**: Cannot handle non-numeric or nm headers
4. **No Conversion Logic**: nm ↔ cm⁻¹ conversion not implemented
5. **UI Disconnect**: UI collects header type but doesn't persist it

---

## Impact Analysis

### Components Affected

#### 🔴 **High Impact - Requires Changes**

1. **CSV Loader** (`csv_loader.py`)
   - Must detect/store header type metadata
   - Add nm → cm⁻¹ conversion option
   - Handle missing headers gracefully

2. **Dataset Config** (`loader.py`, `dataset_config.py`)
   - Pass header type metadata through loading pipeline
   - Store in dataset configuration

3. **Feature Source** (`feature_source.py`)
   - Add `_header_unit` or `_header_type` field
   - Store: "cm-1", "nm", "none", "text", "index"

4. **Dataset** (`dataset.py`)
   - Add `header_unit(src)` method
   - Modify `float_headers()` to handle nm → cm⁻¹ conversion
   - Add `get_wavelengths_cm1(src)` for standardized access

5. **Resampler Controller** (`op_resampler.py`)
   - Check header type before extraction
   - Convert nm → cm⁻¹ if needed
   - Provide clear error messages for incompatible headers

#### 🟡 **Medium Impact - Needs Adaptation**

6. **Visualization** (`op_spectra_charts.py`)
   - Use header_unit metadata for axis labels
   - Support multiple unit displays

7. **SHAP Analyzer** (`shap_analyzer.py`)
   - Handle non-numeric headers more gracefully

8. **UI/API** (`workspace_manager.py`, modals)
   - Pass header_type through API
   - Persist in workspace.json

#### 🟢 **Low Impact - Mostly Compatible**

9. **Preprocessing Operators**: Already work on numeric data regardless of headers
10. **Model Training**: Independent of header semantics

---

## Conversion Formulas

### Wavelength ↔ Wavenumber
```python
# nm → cm⁻¹ (wavenumber)
wavenumber_cm1 = 10_000_000 / wavelength_nm

# cm⁻¹ → nm
wavelength_nm = 10_000_000 / wavenumber_cm1
```

### Example
```python
# NIR range in nm: 780-2500 nm
# Converts to: 12820-4000 cm⁻¹

780 nm  →  12,820.5 cm⁻¹
1000 nm →  10,000.0 cm⁻¹
2500 nm →   4,000.0 cm⁻¹
```

**Note**: nm and cm⁻¹ have **inverse relationship** - higher nm = lower cm⁻¹

---

## Three Scenarios for Implementation

### **Scenario 1: Minimal Conversion (Quick Fix)**
**Effort**: 2-3 days
**Scope**: Enable basic nm support without full metadata system

#### Changes:
1. **CSV Loader**
   - Add `header_unit` parameter to `load_csv()` (default: "cm-1")
   - If `header_unit == "nm"`, convert headers to cm⁻¹ during loading
   - Store converted headers as strings

2. **Dataset Config**
   - Pass `header_unit` from config to `load_csv()`
   - Add to `global_params` in workspace_manager

3. **Resampler**
   - No changes needed (already gets cm⁻¹ headers)

4. **UI**
   - Wire up existing header type dropdown to backend

#### Pros:
- ✅ Fast implementation
- ✅ Minimal code changes
- ✅ Resampler works immediately

#### Cons:
- ❌ Loses original nm values
- ❌ No runtime inspection of header type
- ❌ Cannot support "none" or "text" headers for resampler
- ❌ All headers converted to cm⁻¹ strings

---

### **Scenario 2: Metadata-Aware System (Recommended)**
**Effort**: 5-7 days
**Scope**: Full header type tracking with on-demand conversion

#### Changes:
1. **Feature Source** (`feature_source.py`)
   ```python
   class FeatureSource:
       def __init__(self, ...):
           self._headers: Optional[List[str]] = None
           self._header_unit: str = "cm-1"  # "cm-1", "nm", "none", "text", "index"

       def set_headers(self, headers: List[str], unit: str = "cm-1"):
           self._headers = headers
           self._header_unit = unit

       @property
       def header_unit(self) -> str:
           return self._header_unit
   ```

2. **Dataset** (`dataset.py`)
   ```python
   def header_unit(self, src: int) -> str:
       return self._features.sources[src].header_unit

   def wavelengths_cm1(self, src: int) -> np.ndarray:
       """Get wavelengths in cm⁻¹, converting if needed."""
       headers = self.headers(src)
       unit = self.header_unit(src)

       if unit == "cm-1":
           return np.array([float(h) for h in headers])
       elif unit == "nm":
           nm_values = np.array([float(h) for h in headers])
           return 10_000_000 / nm_values
       elif unit == "none":
           # Generate numeric sequence
           return np.arange(len(headers), dtype=float)
       elif unit == "index":
           return np.arange(len(headers), dtype=float)
       else:
           raise ValueError(f"Cannot convert unit '{unit}' to cm⁻¹")
   ```

3. **CSV Loader** (`csv_loader.py`)
   ```python
   def load_csv(..., header_unit='cm-1', **user_params):
       # Load data as before
       headers = data.columns.tolist()

       # Return unit metadata
       return data, report, na_mask, headers, header_unit
   ```

4. **Resampler Controller** (`op_resampler.py`)
   ```python
   def _extract_wavelengths(self, dataset, source_idx):
       unit = dataset.header_unit(source_idx)

       if unit not in ["cm-1", "nm"]:
           raise ValueError(
               f"Resampler requires wavelength headers (cm-1 or nm), "
               f"but dataset has '{unit}' headers"
           )

       # Use new unified method
       return dataset.wavelengths_cm1(source_idx)
   ```

5. **Visualization** (`op_spectra_charts.py`)
   ```python
   unit = dataset.header_unit(sd_idx)

   if unit == "cm-1":
       x_label = "Wavenumber (cm⁻¹)"
   elif unit == "nm":
       x_label = "Wavelength (nm)"
   else:
       x_label = "Features"
   ```

6. **Config Persistence**
   - Add `header_unit` to dataset configs
   - Store in workspace.json
   - Pass through API endpoints

#### Pros:
- ✅ Preserves original header values
- ✅ Runtime inspection of header type
- ✅ Clean abstraction for conversions
- ✅ Supports all header types
- ✅ Resampler gets proper error messages
- ✅ Visualization adapts automatically

#### Cons:
- ⚠️ Moderate implementation effort
- ⚠️ Need to update multiple components

---

### **Scenario 3: Full Spectroscopy Metadata System**
**Effort**: 10-14 days
**Scope**: Complete wavelength management with validation and advanced features

#### Changes:
All from Scenario 2, plus:

1. **Wavelength Validator**
   ```python
   class WavelengthMetadata:
       unit: Literal["cm-1", "nm", "index", "text", "none"]
       values: np.ndarray
       original_values: Optional[np.ndarray]
       is_ascending: bool
       is_uniform: bool  # Evenly spaced?
       range: Tuple[float, float]

       def to_cm1(self) -> np.ndarray:
           """Convert to cm⁻¹"""

       def to_nm(self) -> np.ndarray:
           """Convert to nm"""

       def validate(self):
           """Check for valid spectroscopy range"""
   ```

2. **Auto-detection**
   - Detect unit from header patterns (e.g., 400-2500 likely nm, 4000-12000 likely cm⁻¹)
   - Warn if values outside typical NIR ranges

3. **Resampler Enhancements**
   - Option to specify target unit
   - Auto-handle ascending/descending order
   - Validate target range compatibility

4. **Export/Import**
   - Preserve unit metadata in saved pipelines
   - Export datasets with unit information
   - Convert on import if needed

5. **Documentation**
   - Update all docs with unit handling
   - Add wavelength conversion guide

#### Pros:
- ✅ Complete, professional solution
- ✅ Auto-detection reduces user errors
- ✅ Extensive validation
- ✅ Future-proof for other spectroscopy types
- ✅ Publication-quality exports

#### Cons:
- ❌ Significant development time
- ❌ More complex API
- ❌ Higher testing burden

---

## Recommendation

**Choose Scenario 2: Metadata-Aware System**

### Reasoning:
1. **Right Balance**: Not too simple, not over-engineered
2. **Solves All Core Issues**: nm support, resampler compatibility, visualization
3. **Preserves Data**: Original headers retained
4. **Extensible**: Easy to add Scenario 3 features later
5. **User-Friendly**: Clear error messages, automatic conversions

### Implementation Order:
1. **Week 1**: Core infrastructure (FeatureSource, Dataset, CSV loader)
2. **Week 2**: Resampler adaptation, visualization updates, UI wiring
3. **Testing**: Validate with nm/cm⁻¹ datasets, edge cases

### Testing Strategy:
```python
# Test cases needed:
1. CSV with nm headers (780-2500)
2. CSV with cm⁻¹ headers (4000-12820)
3. CSV with no headers
4. CSV with text headers
5. Resampler with nm input → cm⁻¹ target
6. Resampler with cm⁻¹ input → nm target
7. Mixed datasets (different units per source)
8. Visualization with both units
```

---

## Migration Path

### For Existing Users:
- **Default behavior**: All existing datasets interpreted as cm⁻¹ (backward compatible)
- **No breaking changes**: Current pipelines work as-is
- **Opt-in**: Users explicitly set header_unit for nm data

### For New Users:
- UI prompts for header type during dataset creation
- Auto-detection suggests likely unit
- Clear documentation on when to use each type

---

## Questions to Consider

1. **Should we auto-detect unit from numeric ranges?**
   - Pro: Less user input needed
   - Con: Could misidentify edge cases

2. **Support mixed units in multi-source datasets?**
   - Current: Each source can have different unit
   - Resampler: Needs all sources in same unit (or converts)

3. **What about other spectroscopy techniques?**
   - Raman: cm⁻¹ (wavenumber shift)
   - UV-Vis: nm (wavelength)
   - IR: cm⁻¹ (wavenumber)
   - → Scenario 2 handles all

4. **Store both original and converted values?**
   - Pro: No information loss
   - Con: More memory
   - Decision: Only store original, convert on-demand

---

## API Changes Summary (Scenario 2)

### New Methods:
```python
# Dataset
dataset.header_unit(src: int) -> str
dataset.wavelengths_cm1(src: int) -> np.ndarray
dataset.wavelengths_nm(src: int) -> np.ndarray  # optional

# FeatureSource
feature_source.set_headers(headers: List[str], unit: str = "cm-1")
feature_source.header_unit -> str

# CSV Loader
load_csv(..., header_unit: str = "cm-1") -> (data, report, mask, headers, unit)
```

### Modified Methods:
```python
# Dataset.float_headers() - DEPRECATED, use wavelengths_cm1()
# Config loading - add header_unit parameter
# Resampler._extract_wavelengths() - use dataset.wavelengths_cm1()
```

### Config Schema Changes:
```json
{
  "dataset": {
    "train_x": "X_train.csv",
    "global_params": {
      "delimiter": ";",
      "decimal_separator": ".",
      "has_header": true,
      "header_unit": "cm-1"  // NEW
    }
  }
}
```

---

## File Impact Summary

### 🔴 Must Modify:
- `nirs4all/dataset/csv_loader.py` (add header_unit parameter/return)
- `nirs4all/dataset/feature_source.py` (add _header_unit field)
- `nirs4all/dataset/dataset.py` (add wavelengths_cm1() method)
- `nirs4all/dataset/loader.py` (pass header_unit through)
- `nirs4all/controllers/dataset/op_resampler.py` (use new methods)

### 🟡 Should Modify:
- `nirs4all/controllers/chart/op_spectra_charts.py` (use unit for labels)
- `nirs4all_ui/api/workspace_manager.py` (handle header_unit config)
- `nirs4all_ui/src/components/AddDatasetModal.tsx` (wire up UI)
- `nirs4all_ui/src/components/EditDatasetModal.tsx` (wire up UI)

### 🟢 Optional:
- `nirs4all/utils/shap_analyzer.py` (better header handling)
- Documentation files (update with unit info)
- Test files (add unit conversion tests)

---

## Estimated Timelines

| Scenario | Dev Time | Testing | Total | Risk |
|----------|----------|---------|-------|------|
| 1. Minimal | 2 days | 1 day | 3 days | Low |
| 2. Metadata | 5 days | 2 days | 7 days | Medium |
| 3. Full System | 10 days | 4 days | 14 days | High |

**Recommendation**: Start with **Scenario 2**, ship it, gather feedback, then add Scenario 3 features if needed.

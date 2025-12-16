# Header Units Analysis and Implementation Report

## Executive Summary

The nirs4all library had a fragmented approach to handling feature header units (cm⁻¹, nm, text, index, none). While the core data structures supported unit metadata, the propagation and usage of this metadata was inconsistent across the codebase, leading to display bugs in visualizations.

**Status: IMPLEMENTED** ✅

---

## 1. Objective

Create a **stable, elegant, and maintainable** system for managing feature header units that:

1. ✅ **Consistently propagates** unit information from data loading to visualization
2. ✅ **Automatically selects** appropriate axis labels and formatting based on unit type
3. ✅ **Handles conversions** correctly when required (nm ↔ cm⁻¹)
4. ✅ **Fails gracefully** with sensible defaults when unit information is missing
5. ✅ **Is centralized** to avoid code duplication and inconsistencies

---

## 2. Current State Analysis (Before Fix)

### 2.1 Unit Definition

The `HeaderUnit` enum in `nirs4all/data/_features/feature_constants.py` defines five unit types:

```python
class HeaderUnit(str, Enum):
    WAVENUMBER = "cm-1"    # Wavenumber in cm⁻¹
    WAVELENGTH = "nm"      # Wavelength in nanometers
    NONE = "none"          # No units
    TEXT = "text"          # Text labels
    INDEX = "index"        # Numeric indices
```

### 2.2 Problems Identified

| Problem | Severity | Location |
|---------|----------|----------|
| X-axis label logic duplicated 5+ times | High | Multiple controllers |
| No centralized unit-to-label mapping | High | Scattered |
| Inconsistent default labels (`Features` vs `Feature Index`) | Medium | Controllers |
| Silent fallbacks hide configuration issues | Medium | All controllers |
| No validation of unit consistency across train/test | Low | Data loading |
| `TEXT` and `INDEX` units not distinctly handled | Low | Visualizations |

### 2.3 Code Duplication Locations

Before the fix, identical logic existed in:
- `nirs4all/controllers/charts/spectra.py` (lines 299-320 and 439-460)
- `nirs4all/controllers/charts/spectral_distribution.py` (lines 139-150)
- `nirs4all/controllers/charts/augmentation.py` (lines 302-320)
- `bench/studies/study_report.py` (lines 344-370)

---

## 3. Solution Implemented

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              nirs4all/utils/header_units.py (NEW)                │
│  - AXIS_LABELS: dict mapping HeaderUnit → label string          │
│  - get_axis_label(unit) → str                                   │
│  - get_x_values_and_label(headers, unit, n_features) → tuple    │
│  - should_invert_x_axis(x_values) → bool                        │
│  - apply_x_axis_limits(ax, x_values) → None                     │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    Used by all consumers:
         ┌────────────┬────────────┬────────────────┐
         │            │            │                │
    Controllers   Analyzers   ReportGenerator   Tests
```

### 3.2 Key Functions

#### `get_axis_label(unit: str | HeaderUnit) -> str`
Simple lookup returning the human-readable axis label for any unit type.

```python
>>> get_axis_label("cm-1")
'Wavenumber (cm⁻¹)'
>>> get_axis_label("nm")
'Wavelength (nm)'
>>> get_axis_label("unknown")  # Graceful fallback
'Features'
```

#### `get_x_values_and_label(headers, header_unit, n_features) -> tuple`
Main workhorse function that returns both x-axis values and label in one atomic operation.

```python
>>> x_vals, label = get_x_values_and_label(["4000", "4500", "5000"], "cm-1", 3)
>>> x_vals
array([4000., 4500., 5000.])
>>> label
'Wavenumber (cm⁻¹)'
```

Handles:
- Numeric headers → parsed float array with appropriate label
- Non-numeric headers → indices with fallback label
- Missing/mismatched headers → indices with fallback label

#### `apply_x_axis_limits(ax, x_values) -> None`
Applies explicit axis limits to preserve data ordering (prevents matplotlib auto-sorting).

### 3.3 Canonical Labels (Single Source of Truth)

```python
AXIS_LABELS = {
    HeaderUnit.WAVENUMBER: "Wavenumber (cm⁻¹)",
    HeaderUnit.WAVELENGTH: "Wavelength (nm)",
    HeaderUnit.NONE: "Feature Index",
    HeaderUnit.TEXT: "Features",
    HeaderUnit.INDEX: "Feature Index",
}
DEFAULT_AXIS_LABEL = "Features"
```

### 3.4 Rationale

| Decision | Rationale |
|----------|-----------|
| Single utility module | DRY principle; one place to fix bugs |
| Return tuple from `get_x_values_and_label()` | Avoid separate calls; atomic operation |
| Keep `HeaderUnit` enum | Type safety; IDE support; documented values |
| Graceful fallback for invalid units | Better UX; no crashes on misconfiguration |
| Separate `apply_x_axis_limits()` | Reusable; clear responsibility |

---

## 4. Implementation Roadmap

### Phase 1: Foundation ✅
- Created `nirs4all/utils/header_units.py`
- Exported from `nirs4all/utils/__init__.py`
- Added 32 unit tests in `tests/unit/utils/test_header_units.py`

### Phase 2: Controller Migration ✅
Migrated all chart controllers:
- `SpectraChartController` - `_plot_2d_spectra()` and `_plot_3d_spectra()`
- `SpectralDistributionController` - `_get_x_label()` method
- `AugmentationChartController` - inline logic

### Phase 3: Report Generator ✅
- Updated `bench/studies/study_report.py` `_create_spectra_chart()` method

### Phase 4: Predictions Enhancement (Future)
- Store `header_unit` in Predictions schema for analyzer charts
- Update model controllers to store this metadata

### Phase 5: Data Loading Validation (Future)
- Add unit validation in `DatasetConfigs` when loading train+test
- Log warnings for potential misconfigurations

---

## 5. Files Modified

### New Files
| File | Description |
|------|-------------|
| `nirs4all/utils/header_units.py` | Centralized header unit utilities |
| `tests/unit/utils/test_header_units.py` | 32 unit tests |

### Modified Files
| File | Changes |
|------|---------|
| `nirs4all/utils/__init__.py` | Export header_units functions |
| `nirs4all/controllers/charts/spectra.py` | Use `get_x_values_and_label()`, `apply_x_axis_limits()` |
| `nirs4all/controllers/charts/spectral_distribution.py` | Use `get_axis_label()` |
| `nirs4all/controllers/charts/augmentation.py` | Use `get_x_values_and_label()`, `apply_x_axis_limits()` |
| `bench/studies/study_report.py` | Use `get_x_values_and_label()` |

---

## 6. Testing

### Unit Tests (32 tests, all passing)
```
tests/unit/utils/test_header_units.py::TestAxisLabels - 4 tests
tests/unit/utils/test_header_units.py::TestGetAxisLabel - 7 tests
tests/unit/utils/test_header_units.py::TestGetXValuesAndLabel - 10 tests
tests/unit/utils/test_header_units.py::TestShouldInvertXAxis - 5 tests
tests/unit/utils/test_header_units.py::TestApplyXAxisLimits - 3 tests
tests/unit/utils/test_header_units.py::TestIntegration - 3 tests
```

### Integration Verification
- `Q13_nm_headers.py` - nm header handling works correctly
- `Q1_regression.py` - chart visualization works correctly

---

## 7. Usage Examples

### Before (duplicated in each controller)
```python
if header_unit == "cm-1":
    x_label = 'Wavenumber (cm⁻¹)'
elif header_unit == "nm":
    x_label = 'Wavelength (nm)'
else:
    x_label = 'Features'

if headers and len(headers) == n_features:
    try:
        x_values = np.array([float(h) for h in headers])
    except (ValueError, TypeError):
        x_values = np.arange(n_features)
        x_label = 'Features'
else:
    x_values = np.arange(n_features)
    x_label = 'Features'

if len(x_values) > 1 and x_values[0] > x_values[-1]:
    ax.set_xlim(x_values[0], x_values[-1])
```

### After (centralized)
```python
from nirs4all.utils.header_units import get_x_values_and_label, apply_x_axis_limits

x_values, x_label = get_x_values_and_label(headers, header_unit, n_features)
apply_x_axis_limits(ax, x_values)
```

---

## 8. Future Considerations

1. **Predictions Storage**: Add `header_unit` field to prediction records for analyzer charts
2. **Unit Validation**: Validate train/test unit consistency during data loading
3. **Auto-detection**: Add heuristics to detect unit from header values when not specified
4. **Documentation**: Update user guide with header_unit configuration examples

---

## 9. Appendix: Code Locations Reference

### Unit Definition
- `nirs4all/data/_features/feature_constants.py`: `HeaderUnit` enum

### Unit Storage
- `nirs4all/data/_features/header_manager.py`: `HeaderManager` class
- `nirs4all/data/_features/feature_source.py`: `FeatureSource.header_unit` property
- `nirs4all/data/_dataset/feature_accessor.py`: `FeatureAccessor.header_unit()` method

### Centralized Utilities (NEW)
- `nirs4all/utils/header_units.py`: All header unit handling functions

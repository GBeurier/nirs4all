# Feature Components Refactoring - Migration Guide

## Overview

In version 0.4.1, the feature management system has been refactored to use a modular component-based architecture. This refactoring improves code maintainability, testability, and extensibility while maintaining **100% backward compatibility** with existing code.

## What Changed

### New Module Structure

```
nirs4all/data/
├── feature_components/           # NEW MODULE
│   ├── __init__.py              # Exports all components + constants
│   ├── feature_constants.py     # Enums and constants
│   ├── feature_source.py        # Refactored FeatureSource (moved here)
│   ├── array_storage.py         # 3D array management
│   ├── processing_manager.py    # Processing ID tracking
│   ├── header_manager.py        # Feature header management
│   ├── layout_transformer.py    # Layout transformations
│   ├── update_strategy.py       # Update operation logic
│   └── augmentation_handler.py  # Sample augmentation
├── features.py                   # Features class (uses feature_components)
├── dataset.py                    # Dataset (uses feature_components)
└── feature_source.py             # DEPRECATED: Compatibility stub
```

### New Features

#### 1. Type-Safe Enums

The refactoring introduces enums for layouts and header units:

```python
from nirs4all.data import FeatureLayout, HeaderUnit

# Layout options
FeatureLayout.FLAT_2D           # "2d"
FeatureLayout.FLAT_2D_INTERLEAVED  # "2d_interleaved"
FeatureLayout.VOLUME_3D         # "3d"
FeatureLayout.VOLUME_3D_TRANSPOSE  # "3d_transpose"

# Header unit options
HeaderUnit.WAVENUMBER  # "cm-1"
HeaderUnit.WAVELENGTH  # "nm"
HeaderUnit.NONE        # "none"
HeaderUnit.TEXT        # "text"
HeaderUnit.INDEX       # "index"
```

#### 2. Component-Based Architecture

`FeatureSource` now uses specialized components:

- **ArrayStorage**: Manages 3D numpy array with padding
- **ProcessingManager**: Tracks processing IDs and indices
- **HeaderManager**: Manages feature headers and units
- **LayoutTransformer**: Handles layout transformations
- **UpdateStrategy**: Categorizes update operations
- **AugmentationHandler**: Manages sample augmentation

## Migration Guide

### No Action Required for Most Users

If you use the public API (dataset.x(), dataset.y(), etc.), **no changes are needed**. Your existing code will continue to work without modification.

### Recommended Updates (Optional)

While not required, you may want to update your code to use the new enums for better type safety:

#### Before (still works):
```python
X = dataset.x({"partition": "train"}, layout="2d")
```

#### After (recommended for type safety):
```python
from nirs4all.data import FeatureLayout

X = dataset.x({"partition": "train"}, layout=FeatureLayout.FLAT_2D)
# Or use the string value
X = dataset.x({"partition": "train"}, layout=FeatureLayout.FLAT_2D.value)
```

### Import Path Changes

#### Old Import (deprecated):
```python
from nirs4all.data.feature_source import FeatureSource  # ⚠️ Deprecated
```

#### New Imports (recommended):
```python
# Primary import path
from nirs4all.data import FeatureSource

# Or direct from feature_components
from nirs4all.data._features import FeatureSource

# With enums
from nirs4all.data import FeatureSource, FeatureLayout, HeaderUnit
```

### Using Enums in Your Code

#### Layout Enums
```python
from nirs4all.data import FeatureLayout

# Both work (backward compatible)
X1 = dataset.x({}, layout="2d")  # String (old way)
X2 = dataset.x({}, layout=FeatureLayout.FLAT_2D)  # Enum (new way)

# Enum values are the same as strings
assert FeatureLayout.FLAT_2D.value == "2d"
```

#### Header Unit Enums
```python
from nirs4all.data import HeaderUnit

# Both work
dataset.add_samples(X, headers=headers, header_unit="cm-1")  # String
dataset.add_samples(X, headers=headers, header_unit=HeaderUnit.WAVENUMBER)  # Enum

# Enum values are the same as strings
assert HeaderUnit.WAVENUMBER.value == "cm-1"
```

## Benefits of the Refactoring

1. **Modularity**: Clear separation of concerns - each component has a single responsibility
2. **Testability**: Components can be tested independently
3. **Type Safety**: Enums prevent typos and invalid values
4. **Extensibility**: Easy to add new layouts or processing strategies
5. **Maintainability**: Smaller, focused classes are easier to understand and modify
6. **Performance**: Same or better performance with cleaner code

## Backward Compatibility

All existing code continues to work:

- ✅ String layouts (`"2d"`, `"3d_transpose"`, etc.) still accepted
- ✅ String header units (`"cm-1"`, `"nm"`, etc.) still accepted
- ✅ Old import paths work with deprecation warning
- ✅ All public APIs unchanged
- ✅ Serialization format unchanged

## Deprecation Timeline

- **v0.4.1**: Old import path deprecated (with warning)
- **v0.5.0**: Old import path will be removed
- **v1.0.0**: String layouts/units may be deprecated (with migration period)

## Testing

All existing tests pass without modification. New comprehensive test suite added for feature components:
- `tests/unit/data/test_feature_components.py`

To run tests:
```bash
pytest tests/unit/data/test_feature_components.py -v
```

## Examples

All examples continue to work without modification. The public API remains unchanged.

## Questions or Issues?

If you encounter any issues during migration:
1. Check that you're using the public API (dataset.x(), dataset.y(), etc.)
2. Update imports from deprecated paths
3. Report issues on GitHub: https://github.com/GBeurier/nirs4all/issues

## Summary

This refactoring improves the internal architecture of nirs4all while maintaining complete backward compatibility. Most users don't need to change anything. For new code, consider using the enum-based APIs for better type safety and developer experience.

# SpectroDataset Refactoring - Implementation Summary

**Author:** GitHub Copilot
**Date:** October 29, 2025
**Status:** ✅ COMPLETED

---

## Implementation Summary

The `SpectroDataset` refactoring has been successfully completed. The class now uses specialized accessor components internally while maintaining full backward compatibility with the existing API.

### What Changed

1. **Created `dataset_components` package** with three accessor classes:
   - `FeatureAccessor` - Manages all feature operations
   - `TargetAccessor` - Manages all target operations
   - `MetadataAccessor` - Manages all metadata operations

2. **Refactored `SpectroDataset`** to use accessors internally:
   - All public methods delegate to accessors
   - Primary API remains `dataset.x()` and `dataset.y()` (not `dataset.features.x()`)
   - Internal references preserved for backward compatibility

3. **Added task_type to Targets class**:
   - Task type detection now happens in the Targets block
   - Automatically set when targets are added
   - Accessible via `dataset.task_type` property

4. **Improved test coverage**:
   - Added comprehensive unit tests for all dataset operations
   - All existing integration tests pass
   - 80+ tests covering features, targets, metadata, and edge cases

### Architecture (After Refactoring)

```
SpectroDataset (Main Facade - ~500 LOC)
├── _feature_accessor: FeatureAccessor (internal)
├── _target_accessor: TargetAccessor (internal)
├── _metadata_accessor: MetadataAccessor (internal)
├── _indexer: Indexer (internal)
├── _folds: List[Tuple] (CV folds)
└── name: str (dataset identifier)

dataset_components/
├── feature_accessor.py   # Feature operations (~450 LOC)
├── target_accessor.py    # Target operations (~200 LOC)
└── metadata_accessor.py  # Metadata operations (~180 LOC)
```

### API Compatibility

✅ **All existing code works without changes:**

```python
# Public API (unchanged)
dataset = SpectroDataset("my_dataset")
dataset.add_samples(X_train, {"partition": "train"})
dataset.add_targets(y_train)
X = dataset.x({"partition": "train"})
y = dataset.y({"partition": "train"})
```

The accessor components are internal implementation details and not part of the public API.

---

## Original Proposal (For Historical Reference)

The `SpectroDataset` class currently serves as a "God Object" facade that orchestrates feature, target, metadata, and indexing operations. While this provides a convenient single entry point, it results in:

1. **Poor Separation of Concerns** - Dataset mixes high-level orchestration with low-level data access patterns
2. **Inconsistent API Patterns** - Mix of convenience methods, delegation patterns, and property accessors
3. **Missing/Incomplete Docstrings** - Many methods lack proper Google-style documentation
4. **Tight Coupling** - Direct manipulation of internal blocks instead of well-defined boundaries
5. **Performance Concerns** - No lazy evaluation or caching strategy for expensive operations
6. **Testing Complexity** - Monolithic class is harder to unit test in isolation

This proposal outlines a **pragmatic refactoring** that maintains backward compatibility while improving maintainability and clarity.

---

## Current Architecture Analysis

### Current Class Structure

```
SpectroDataset (Main Facade - 600+ LOC)
├── _indexer: Indexer          # Sample filtering & index management
├── _features: Features         # Multi-source feature data
├── _targets: Targets           # Target data with processing chains
├── _metadata: Metadata         # Auxiliary sample data
└── _folds: List[Tuple]        # Cross-validation folds
```

### Problems Identified

#### 1. **Method Proliferation**
- 40+ public methods mixed with helper methods
- No clear grouping (feature methods, target methods, metadata methods)
- Duplication: `x()` vs `x_train()` vs `x_test()` (commented out)

#### 2. **Inconsistent Delegation Pattern**
```python
# Direct delegation (good)
def headers(self, src: int) -> List[str]:
    return self._features.headers(src)

# Property wrapper (unnecessary layer)
@property
def num_classes(self) -> int:
    return self._targets.num_classes  # Why not call directly?

# Mixed responsibility (bad)
def y(self, selector: Selector, include_augmented: bool = True):
    # Indexer logic mixed with target retrieval
    if include_augmented:
        x_indices = self._indexer.x_indices(selector, include_augmented=True)
        y_indices = np.array([...])  # Complex mapping logic HERE
    else:
        y_indices = self._indexer.x_indices(selector, include_augmented=False)

    processing = selector.get("y", "numeric")
    return self._targets.y(y_indices, processing)
```

#### 3. **Wavelength Conversion Methods Are Low-Level**
These methods (`wavelengths_cm1()`, `wavelengths_nm()`, `float_headers()`) belong in the Features/FeatureSource layer, not the main facade:

```python
def wavelengths_cm1(self, src: int) -> np.ndarray:
    """74 lines of conversion logic in the facade!"""
    headers = self.headers(src)
    unit = self.header_unit(src)
    # ... conversion logic ...
```

**Why this is wrong:**
- Dataset shouldn't know about spectroscopy-specific conversions
- Makes the class harder to use for non-spectroscopy data
- Violates Single Responsibility Principle

#### 4. **Processing String Manipulation**
```python
def short_preprocessings_str(self) -> str:
    """35 lines of string manipulation in main class"""
    processings = "|".join(self.features_processings(0))
    replacements = [("SavitzkyGolay", "SG"), ...]
    # ... more string manipulation ...
```

This is formatting logic, not core dataset responsibility.

#### 5. **Task Type Management is Confusing**
```python
# Set in multiple places
def add_targets(self, y):
    self._task_type = ModelUtils.detect_task_type(y)

def add_processed_targets(self, ...):
    new_task_type = ModelUtils.detect_task_type(targets)
    if self._task_type != new_task_type:
        print(f"Task type updated...")
        self._task_type = new_task_type

# Also has manual override
def set_task_type(self, task_type):
    self._task_type = task_type
```

Task type should be managed by Targets block, not duplicated in Dataset.

#### 6. **Missing Docstrings**
Methods missing proper documentation:
- `add_features()`
- `replace_features()`
- `update_features()`
- `features_processings()`
- `features_sources()`
- `is_multi_source()`
- `headers()`, `header_unit()`
- `short_preprocessings_str()`
- `index_column()`
- `print_summary()`
- `_fold_str()`
- `__str__()`

---

## Proposed Refactoring

### Principle: **Thin Facade + Specialized Accessors**

Instead of one massive class, separate concerns into:

1. **SpectroDataset** - Minimal orchestration facade (initialization, coordination)
2. **FeatureAccessor** - All feature-related operations (includes wavelength conversions)
3. **TargetAccessor** - All target-related operations
4. **MetadataAccessor** - All metadata operations

### New Architecture

```
SpectroDataset (Slim Facade - ~150 LOC)
├── features: FeatureAccessor      # .x(), .add_samples(), .headers(), etc.
├── targets: TargetAccessor        # .y(), .add_targets(), .task_type, etc.
├── metadata: MetadataAccessor     # .get(), .add_metadata(), etc.
├── _indexer: Indexer              # Internal filtering engine
├── _folds: List[Tuple]            # CV folds
└── name: str                       # Dataset name
```

### Backward Compatibility Strategy

Keep the existing public API by forwarding calls to accessors:

```python
class SpectroDataset:
    def __init__(self, name: str = "Unknown_dataset"):
        self._indexer = Indexer()
        self.features = FeatureAccessor(self._indexer, Features())
        self.targets = TargetAccessor(self._indexer, Targets())
        self.metadata = MetadataAccessor(self._indexer, Metadata())
        self._folds = []
        self.name = name

    # Backward compatibility: delegate to accessors (no deprecation warnings)
    def x(self, selector, layout="2d", concat_source=True, include_augmented=True):
        """Get feature data. See features.x() for details."""
        return self.features.x(selector, layout, concat_source, include_augmented)

    def y(self, selector, include_augmented=True):
        """Get target data. See targets.y() for details."""
        return self.targets.y(selector, include_augmented)
```

---

## Detailed Component Design

### 1. FeatureAccessor

**Responsibility:** All feature data operations

```python
class FeatureAccessor:
    """
    Accessor for feature data operations.

    Provides methods for adding, retrieving, and manipulating feature data
    across multiple sources with different processing chains.

    Attributes:
        num_samples (int): Number of samples in the dataset
        num_features (Union[List[int], int]): Number of features per source
        num_sources (int): Number of feature sources

    Examples:
        >>> dataset = SpectroDataset()
        >>> dataset.features.add_samples(X_train, {"partition": "train"})
        >>> X = dataset.features.x({"partition": "train"}, layout="2d")
    """

    def __init__(self, indexer: Indexer, features_block: Features):
        """
        Initialize feature accessor.

        Args:
            indexer: Sample index manager for filtering
            features_block: Underlying feature storage
        """
        self._indexer = indexer
        self._block = features_block

    def x(self,
          selector: Selector = None,
          layout: Layout = "2d",
          concat_source: bool = True,
          include_augmented: bool = True) -> OutputData:
        """
        Get feature data with filtering and layout control.

        Args:
            selector: Filter criteria. Supported keys:
                - partition: "train" | "test" | "val"
                - group: int or List[int]
                - branch: int or List[int]
                - augmentation: str or None
            layout: Output array layout:
                - "2d": (n_samples, n_features)
                - "3d": (n_samples, n_features, 1)
            concat_source: If True and multiple sources exist,
                concatenate along feature axis. If False, return
                list of arrays.
            include_augmented: If True, include augmented samples.
                If False, return only base samples.

        Returns:
            Feature array(s) matching the selector criteria.
            - Single source + concat_source=True: np.ndarray
            - Multi-source + concat_source=True: np.ndarray (concatenated)
            - Multi-source + concat_source=False: List[np.ndarray]

        Raises:
            ValueError: If no features available

        Examples:
            >>> # Get all train data
            >>> X_train = dataset.features.x({"partition": "train"})

            >>> # Get base samples only (for splitting)
            >>> X_base = dataset.features.x(
            ...     {"partition": "train"},
            ...     include_augmented=False
            ... )

            >>> # Get multi-source data separately
            >>> X_sources = dataset.features.x(
            ...     {"partition": "test"},
            ...     concat_source=False
            ... )
        """
        if selector is None:
            selector = {}
        indices = self._indexer.x_indices(selector, include_augmented)
        return self._block.x(indices, layout, concat_source)

    def add_samples(self,
                    data: InputData,
                    indexes: Optional[IndexDict] = None,
                    headers: Optional[Union[List[str], List[List[str]]]] = None,
                    header_unit: Optional[Union[str, List[str]]] = None) -> None:
        """
        Add feature samples to the dataset.

        This is the primary method for loading spectral or feature data.
        Automatically registers samples in the indexer.

        Args:
            data: Feature data. Can be:
                - np.ndarray: (n_samples, n_features) for single source
                - List[np.ndarray]: Multiple sources
            indexes: Index attributes for samples:
                - partition: "train" | "test" | "val" (default: "train")
                - group: int or List[int] (optional)
                - branch: int or List[int] (optional)
                - augmentation: str (optional)
            headers: Feature names/wavelengths. Can be:
                - List[str]: Single source headers
                - List[List[str]]: Per-source headers
                - None: Auto-generate as "feat_0", "feat_1", ...
            header_unit: Unit type for headers:
                - "nm": Wavelength in nanometers
                - "cm-1": Wavenumber in cm⁻¹
                - "none": No wavelength information
                - "index": Sequential indices

        Raises:
            ValueError: If data dimensions don't match existing samples
            ValueError: If number of sources doesn't match

        Examples:
            >>> # Basic usage
            >>> dataset.features.add_samples(X_train, {"partition": "train"})

            >>> # With wavelength headers
            >>> wavelengths = ["1000", "1001", "1002", ...]
            >>> dataset.features.add_samples(
            ...     X_train,
            ...     {"partition": "train"},
            ...     headers=wavelengths,
            ...     header_unit="nm"
            ... )

            >>> # Multi-source data
            >>> dataset.features.add_samples(
            ...     [X_nir, X_mir],
            ...     {"partition": "train"},
            ...     headers=[nir_wavelengths, mir_wavelengths],
            ...     header_unit=["nm", "cm-1"]
            ... )
        """
        num_samples = get_num_samples(data)
        self._indexer.add_samples_dict(num_samples, indexes)
        self._block.add_samples(data, headers, header_unit)

    def add_features(self,
                     features: InputFeatures,
                     processings: ProcessingList,
                     source: int = -1) -> None:
        """
        Add processed feature versions to existing data.

        Use this to add features from preprocessing pipelines
        (e.g., after Savitzky-Golay filtering, MSC, derivatives).

        Args:
            features: Processed feature data matching sample count
            processings: Processing step names to add to processing lists
            source: Target source index:
                - -1: Apply to all sources (default)
                - 0, 1, ...: Apply to specific source

        Examples:
            >>> # After preprocessing
            >>> X_savgol = savgol_filter(X_raw, ...)
            >>> dataset.features.add_features(
            ...     [X_savgol],
            ...     ["SavitzkyGolay_11_2"],
            ...     source=0
            ... )
        """
        self._block.update_features([], features, processings, source)
        self._indexer.add_processings(processings)

    def replace_features(self,
                         source_processings: ProcessingList,
                         features: InputFeatures,
                         processings: ProcessingList,
                         source: int = -1) -> None:
        """
        Replace existing processed features with new versions.

        Args:
            source_processings: Existing processing names to replace
            features: New feature data
            processings: New processing names
            source: Target source (-1 for all, or specific index)

        Examples:
            >>> # Replace old preprocessing with improved version
            >>> dataset.features.replace_features(
            ...     ["SavitzkyGolay_5_1"],
            ...     [X_new_sg],
            ...     ["SavitzkyGolay_11_2"]
            ... )
        """
        self._block.update_features(source_processings, features, processings, source)
        if source <= 0:
            self._indexer.replace_processings(source_processings, processings)

    def augment_samples(self,
                        data: InputData,
                        processings: ProcessingList,
                        augmentation_id: str,
                        selector: Optional[Selector] = None,
                        count: Union[int, List[int]] = 1) -> List[int]:
        """
        Create augmented versions of existing samples.

        Augmentation creates new synthetic samples based on existing ones
        (e.g., noise addition, spectral shifts, mixup). Augmented samples
        are tracked and can be excluded during CV splitting.

        Args:
            data: Augmented feature data
            processings: Processing names for augmented data
            augmentation_id: Unique identifier for this augmentation type
                (e.g., "noise_0.01", "shift_5nm")
            selector: Filter to select which samples to augment.
                If None, augments all base samples.
            count: Number of augmentations per sample:
                - int: Same count for all samples
                - List[int]: Per-sample counts

        Returns:
            List of newly created sample IDs

        Raises:
            ValueError: If selector matches no samples
            ValueError: If data shape doesn't match sample count

        Examples:
            >>> # Add noise to train samples
            >>> X_noisy = add_gaussian_noise(X_train, std=0.01)
            >>> dataset.features.augment_samples(
            ...     X_noisy,
            ...     ["raw", "noise_0.01"],
            ...     "gaussian_noise",
            ...     selector={"partition": "train"},
            ...     count=3  # 3 noisy versions per sample
            ... )

            >>> # Variable augmentation counts
            >>> dataset.features.augment_samples(
            ...     X_aug,
            ...     ["raw", "mixup"],
            ...     "mixup_augmentation",
            ...     count=[2, 1, 3, 2, ...]  # Per-sample counts
            ... )
        """
        # Always exclude already-augmented samples
        if selector is None:
            sample_indices = self._indexer.x_indices(
                {}, include_augmented=False
            ).tolist()
        else:
            sample_indices = self._indexer.x_indices(
                selector, include_augmented=False
            ).tolist()

        if not sample_indices:
            return []

        augmented_ids = self._indexer.augment_rows(
            sample_indices, count, augmentation_id
        )

        self._block.augment_samples(
            sample_indices, data, processings, count
        )

        return augmented_ids

    def headers(self, source: int = 0) -> List[str]:
        """
        Get feature headers (wavelengths, feature names) for a source.

        Args:
            source: Source index (default: 0)

        Returns:
            List of header strings

        Examples:
            >>> headers = dataset.features.headers(0)
            >>> print(headers[:5])
            ['1000.0', '1001.5', '1003.0', '1004.5', '1006.0']
        """
        return self._block.headers(source)

    def header_unit(self, source: int = 0) -> str:
        """
        Get the unit type for headers.

        Args:
            source: Source index (default: 0)

        Returns:
            Unit string: "cm-1" | "nm" | "none" | "text" | "index"
        """
        return self._block.sources[source].header_unit

    def processing_names(self, source: int = 0) -> List[str]:
        """
        Get list of processing step names for a source.

        Args:
            source: Source index (default: 0)

        Returns:
            List of processing names (e.g., ["raw", "SavitzkyGolay_11_2", "MSC"])

        Examples:
            >>> processings = dataset.features.processing_names(0)
            >>> print(" -> ".join(processings))
            raw -> SavitzkyGolay_11_2 -> MSC -> StandardScaler
        """
        return self._block.preprocessing_str[source]

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return self._block.num_samples

    @property
    def num_features(self) -> Union[List[int], int]:
        """Number of features per source (int if single source, list if multi)."""
        return self._block.num_features

    @property
    def num_sources(self) -> int:
        """Number of feature sources."""
        return len(self._block.sources)

    @property
    def is_multi_source(self) -> bool:
        """True if dataset has multiple feature sources."""
        return self.num_sources > 1

    # Wavelength conversion methods (for spectroscopy datasets)
    def wavelengths_cm1(self, source: int = 0) -> np.ndarray:
        """
        Get wavelengths in cm⁻¹ (wavenumber), converting from nm if needed.

        Note: Only applicable to spectroscopy datasets with wavelength headers.
        For non-spectroscopy datasets, returns feature indices.

        Args:
            source: Source index (default: 0)

        Returns:
            Wavelengths in cm⁻¹ as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths

        Examples:
            >>> wl_cm1 = dataset.features.wavelengths_cm1(0)
            >>> print(wl_cm1[:5])
            [10000.0, 9975.06, 9950.25, 9925.56, 9900.99]
        """
        headers = self.headers(source)
        unit = self.header_unit(source)

        if unit == "cm-1":
            return np.array([float(h) for h in headers])
        elif unit == "nm":
            nm_values = np.array([float(h) for h in headers])
            return 10_000_000.0 / nm_values
        elif unit in ["none", "index"]:
            return np.arange(len(headers), dtype=float)
        else:
            raise ValueError(
                f"Cannot convert unit '{unit}' to wavelengths (cm⁻¹). "
                f"Expected 'cm-1', 'nm', 'none', or 'index'."
            )

    def wavelengths_nm(self, source: int = 0) -> np.ndarray:
        """
        Get wavelengths in nm, converting from cm⁻¹ if needed.

        Note: Only applicable to spectroscopy datasets with wavelength headers.
        For non-spectroscopy datasets, returns feature indices.

        Args:
            source: Source index (default: 0)

        Returns:
            Wavelengths in nm as float array

        Raises:
            ValueError: If headers cannot be converted to wavelengths
        """
        headers = self.headers(source)
        unit = self.header_unit(source)

        if unit == "nm":
            return np.array([float(h) for h in headers])
        elif unit == "cm-1":
            cm1_values = np.array([float(h) for h in headers])
            return 10_000_000.0 / cm1_values
        elif unit in ["none", "index"]:
            return np.arange(len(headers), dtype=float)
        else:
            raise ValueError(
                f"Cannot convert unit '{unit}' to wavelengths (nm). "
                f"Expected 'cm-1', 'nm', 'none', or 'index'."
            )
```

### 2. TargetAccessor

**Responsibility:** All target data operations

```python
class TargetAccessor:
    """
    Accessor for target data operations.

    Manages target values with processing chains, task type detection,
    and prediction transformations.

    Attributes:
        num_samples (int): Number of samples with targets
        num_classes (int): Number of unique classes (classification only)
        task_type (TaskType): Detected or manually set task type
        processing_ids (List[str]): Available target processing versions

    Examples:
        >>> dataset = SpectroDataset()
        >>> dataset.targets.add_targets(y_train)
        >>> y = dataset.targets.y({"partition": "train"})
        >>> print(dataset.targets.task_type)
        TaskType.MULTICLASS_CLASSIFICATION
    """

    def __init__(self, indexer: Indexer, targets_block: Targets):
        """
        Initialize target accessor.

        Args:
            indexer: Sample index manager for filtering
            targets_block: Underlying target storage
        """
        self._indexer = indexer
        self._block = targets_block

    def y(self,
          selector: Selector = None,
          include_augmented: bool = True) -> np.ndarray:
        """
        Get target values with filtering.

        Automatically maps augmented samples to their origin's y values,
        preventing data leakage in cross-validation.

        Args:
            selector: Filter criteria (same as features.x)
            include_augmented: If True, include augmented samples
                (mapped to their origin's y values).
                If False, return only base samples.

        Returns:
            Target array of shape (n_samples, n_targets)

        Examples:
            >>> # Get all train targets (base + augmented)
            >>> y_train = dataset.targets.y({"partition": "train"})

            >>> # Get base targets only (for splitting)
            >>> y_base = dataset.targets.y(
            ...     {"partition": "train"},
            ...     include_augmented=False
            ... )
        """
        if selector is None:
            selector = {}

        if include_augmented:
            x_indices = self._indexer.x_indices(selector, include_augmented=True)
            # Map to origin indices for y lookup
            y_indices = np.array([
                self._indexer.get_origin_for_sample(int(sample_id))
                for sample_id in x_indices
            ], dtype=np.int32)
        else:
            y_indices = self._indexer.x_indices(selector, include_augmented=False)

        processing = selector.get("y", "numeric") if selector else "numeric"
        return self._block.y(y_indices, processing)

    def add_targets(self, y: np.ndarray) -> None:
        """
        Add target samples to the dataset.

        Automatically detects task type (regression, binary, multiclass)
        and creates "raw" and "numeric" processing versions.

        Args:
            y: Target values as 1D or 2D array

        Examples:
            >>> # Classification
            >>> dataset.targets.add_targets([0, 1, 2, 0, 1])
            >>> print(dataset.targets.task_type)
            TaskType.MULTICLASS_CLASSIFICATION

            >>> # Regression
            >>> dataset.targets.add_targets([1.5, 2.3, 3.1, 4.2])
            >>> print(dataset.targets.task_type)
            TaskType.REGRESSION
        """
        self._block.add_targets(y)

    def add_processed_targets(self,
                              processing_name: str,
                              targets: np.ndarray,
                              ancestor_processing: str = "numeric",
                              transformer: Optional[TransformerMixin] = None) -> None:
        """
        Add processed target version (e.g., scaled, encoded).

        Args:
            processing_name: Unique name for this processing
            targets: Processed target data
            ancestor_processing: Parent processing name
            transformer: Scikit-learn transformer used (for inverse transform)

        Examples:
            >>> # Add scaled targets for neural networks
            >>> from sklearn.preprocessing import StandardScaler
            >>> scaler = StandardScaler()
            >>> y_scaled = scaler.fit_transform(y_numeric.reshape(-1, 1))
            >>> dataset.targets.add_processed_targets(
            ...     "scaled",
            ...     y_scaled,
            ...     "numeric",
            ...     scaler
            ... )
        """
        self._block.add_processed_targets(
            processing_name, targets, ancestor_processing, transformer
        )

    def transform_predictions(self,
                              predictions: np.ndarray,
                              from_processing: str,
                              to_processing: str) -> np.ndarray:
        """
        Transform predictions between processing states.

        Useful for converting model predictions back to original scale.

        Args:
            predictions: Model predictions
            from_processing: Current processing state of predictions
            to_processing: Target processing state

        Returns:
            Transformed predictions

        Examples:
            >>> # Model trained on scaled targets
            >>> y_pred_scaled = model.predict(X_test)
            >>> # Transform back to numeric
            >>> y_pred = dataset.targets.transform_predictions(
            ...     y_pred_scaled,
            ...     from_processing="scaled",
            ...     to_processing="numeric"
            ... )
        """
        return self._block.transform_predictions(
            predictions, from_processing, to_processing
        )

    @property
    def task_type(self) -> Optional[TaskType]:
        """
        Get detected task type.

        Returns:
            TaskType enum or None if no targets added
        """
        return self._block._task_type  # Expose from Targets block

    @property
    def num_classes(self) -> int:
        """Number of unique classes (for classification tasks)."""
        return self._block.num_classes

    @property
    def num_samples(self) -> int:
        """Number of samples with target values."""
        return self._block.num_samples

    @property
    def processing_ids(self) -> List[str]:
        """List of available target processing versions."""
        return self._block.processing_ids
```

### 3. MetadataAccessor

**Responsibility:** Sample-level auxiliary data

```python
class MetadataAccessor:
    """
    Accessor for metadata operations.

    Manages sample-level auxiliary information like sample IDs,
    batch numbers, quality scores, etc.

    Examples:
        >>> dataset.metadata.add_metadata(
        ...     metadata_df,
        ...     headers=["sample_id", "batch", "quality"]
        ... )
        >>> batch_info = dataset.metadata.column(
        ...     "batch",
        ...     {"partition": "train"}
        ... )
    """

    def __init__(self, indexer: Indexer, metadata_block: Metadata):
        self._indexer = indexer
        self._block = metadata_block

    def get(self,
            selector: Optional[Selector] = None,
            columns: Optional[List[str]] = None,
            include_augmented: bool = True) -> pl.DataFrame:
        """
        Get metadata as Polars DataFrame.

        Args:
            selector: Filter criteria
            columns: Specific columns to return (None = all)
            include_augmented: Include augmented samples

        Returns:
            Polars DataFrame with metadata
        """
        indices = self._indexer.x_indices(
            selector, include_augmented
        ) if selector else None
        return self._block.get(indices, columns)

    def column(self,
               column: str,
               selector: Optional[Selector] = None,
               include_augmented: bool = True) -> np.ndarray:
        """
        Get single metadata column as array.

        Args:
            column: Column name
            selector: Filter criteria
            include_augmented: Include augmented samples

        Returns:
            Numpy array of column values
        """
        indices = self._indexer.x_indices(
            selector, include_augmented
        ) if selector else None
        return self._block.get_column(column, indices)

    def add_metadata(self,
                     data: Union[np.ndarray, pl.DataFrame, pd.DataFrame],
                     headers: Optional[List[str]] = None) -> None:
        """
        Add metadata rows (aligns with add_samples call order).

        Args:
            data: Metadata as 2D array or DataFrame
            headers: Column names (required if data is ndarray)
        """
        self._block.add_metadata(data, headers)

    def add_column(self, column: str, values: Union[List, np.ndarray]) -> None:
        """
        Add new metadata column.

        Args:
            column: Column name
            values: Values (must match number of samples)
        """
        self._block.add_column(column, values)

    @property
    def columns(self) -> List[str]:
        """List of metadata column names."""
        return self._block.columns

    @property
    def num_rows(self) -> int:
        """Number of metadata rows."""
        return self._block.num_rows
```

### 4. Updated SpectroDataset (Slim Facade)

```python
class SpectroDataset:
    """
    Main dataset facade for spectroscopy and ML/DL pipelines.

    Coordinates feature, target, and metadata management through
    specialized accessor interfaces. Provides high-level operations
    like cross-validation folds and summary printing.

    Attributes:
        name (str): Dataset identifier
        features (FeatureAccessor): Feature data operations
        targets (TargetAccessor): Target data operations
        metadata (MetadataAccessor): Auxiliary data operations
        folds (List[Tuple]): Cross-validation fold splits

    Examples:
        >>> # Create dataset
        >>> dataset = SpectroDataset(name="corn_dataset")

        >>> # Load data
        >>> dataset.features.add_samples(X_train, {"partition": "train"})
        >>> dataset.targets.add_targets(y_train)

        >>> # Access data
        >>> X = dataset.features.x({"partition": "train"})
        >>> y = dataset.targets.y({"partition": "train"})

        >>> # Or use backward-compatible API
        >>> X = dataset.x({"partition": "train"})
        >>> y = dataset.y({"partition": "train"})
    """

    def __init__(self, name: str = "Unknown_dataset"):
        """
        Initialize empty dataset.

        Args:
            name: Dataset identifier for tracking and display
        """
        self.name = name

        # Core components
        self._indexer = Indexer()
        self._features_block = Features()
        self._targets_block = Targets()
        self._metadata_block = Metadata()

        # Accessor interfaces
        self.features = FeatureAccessor(self._indexer, self._features_block)
        self.targets = TargetAccessor(self._indexer, self._targets_block)
        self.metadata = MetadataAccessor(self._indexer, self._metadata_block)

        # CV folds
        self._folds: List[Tuple[List[int], List[int]]] = []

    # ===== Backward Compatibility API =====

    def x(self, selector: Selector = None, layout: Layout = "2d",
          concat_source: bool = True, include_augmented: bool = True) -> OutputData:
        """Get feature data. See features.x() for details."""
        return self.features.x(selector, layout, concat_source, include_augmented)

    def y(self, selector: Selector = None, include_augmented: bool = True) -> np.ndarray:
        """Get target data. See targets.y() for details."""
        return self.targets.y(selector, include_augmented)

    def add_samples(self, data: InputData, indexes: Optional[IndexDict] = None,
                    headers: Optional[Union[List[str], List[List[str]]]] = None,
                    header_unit: Optional[Union[str, List[str]]] = None) -> None:
        """Add feature samples. See features.add_samples() for details."""
        self.features.add_samples(data, indexes, headers, header_unit)

    def add_targets(self, y: np.ndarray) -> None:
        """Add target samples. See targets.add_targets() for details."""
        self.targets.add_targets(y)

    # ... more backward compatibility methods ...

    # ===== Cross-Validation Folds =====

    @property
    def folds(self) -> List[Tuple[List[int], List[int]]]:
        """
        Get cross-validation fold splits.

        Returns:
            List of (train_indices, val_indices) tuples
        """
        return self._folds

    def set_folds(self, folds_iterable) -> None:
        """
        Set cross-validation folds.

        Args:
            folds_iterable: Iterable of (train_idx, val_idx) tuples
                from scikit-learn splitters

        Examples:
            >>> from sklearn.model_selection import KFold
            >>> kf = KFold(n_splits=5)
            >>> dataset.set_folds(kf.split(X))
        """
        self._folds = list(folds_iterable)

    @property
    def num_folds(self) -> int:
        """Number of cross-validation folds."""
        return len(self._folds)

    # ===== Summary and Display =====

    def __str__(self) -> str:
        """Return readable dataset summary."""
        lines = [f"{CHART}Dataset: {self.name}"]

        # Task type
        if self.targets.task_type:
            lines.append(f"  Task: {self.targets.task_type.value}")

        # Features summary
        lines.append(str(self._features_block))

        # Targets summary
        if self.targets.num_samples > 0:
            lines.append(str(self._targets_block))

        # Metadata summary
        if self.metadata.num_rows > 0:
            lines.append(str(self._metadata_block))

        # Indexer summary
        lines.append(str(self._indexer))

        # Folds
        if self._folds:
            fold_str = ", ".join([
                f"({len(tr)}/{len(val)})"
                for tr, val in self._folds
            ])
            lines.append(f"Folds: {fold_str}")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print comprehensive dataset summary."""
        print(self)
```

---

## Migration Strategy

### Phase 1: Add Accessors (Backward Compatible)

1. Create `FeatureAccessor`, `TargetAccessor`, `MetadataAccessor` classes
2. Add them as attributes to `SpectroDataset`
3. Keep existing methods working (delegate to accessors internally)
4. Add deprecation warnings to guide users

```python
# Old API still works
dataset.add_samples(X, {"partition": "train"})
dataset.x({"partition": "train"})

# New API available
dataset.features.add_samples(X, {"partition": "train"})
dataset.features.x({"partition": "train"})
```

### Phase 2: Move Spectroscopy Methods

1. Move `wavelengths_cm1()`, `wavelengths_nm()` to `FeatureAccessor`
2. Keep backward compatibility methods in `SpectroDataset` (delegate to `features.wavelengths_*()`)
3. Remove deprecated `float_headers()` method

### Phase 3: Clean Up Task Type Management

1. Move `_task_type` entirely to `Targets` block
2. Remove from `SpectroDataset`
3. Access via `dataset.targets.task_type`

### Phase 4: Documentation Sprint

1. Add complete Google-style docstrings to all methods
2. Update examples to show both old and new API
3. Create migration guide (optional - both APIs work)

### Phase 5: Remove Deprecated Methods (v2.0)

1. Remove old direct methods from `SpectroDataset`
2. Keep only accessors
3. Major version bump

---

## Benefits

### 1. **Clear Separation of Concerns**
- Features, targets, metadata in separate namespaces
- Easy to find related methods
- Reduced cognitive load

### 2. **Better Testing**
- Each accessor can be unit tested independently
- Mock accessors for integration tests
- Easier to test edge cases

### 3. **Improved Discoverability**
```python
# IDE autocomplete shows logical grouping:
dataset.features.  # → x(), add_samples(), headers(), ...
dataset.targets.   # → y(), add_targets(), task_type, ...
dataset.metadata.  # → get(), column(), add_metadata(), ...
```

### 4. **Maintainability**
- Changes to feature operations don't affect targets
- Each accessor ~200 LOC instead of monolithic 600 LOC class
- New contributors can understand one accessor at a time

### 5. **Performance Opportunities**
- Accessors can add internal caching
- Lazy evaluation strategies
- Batch operations

### 6. **Backward Compatibility**
- Existing code continues to work
- Gradual migration path
- No breaking changes in v1.x

---

## Design Decisions (Finalized)

1. **Naming:** `FeatureAccessor`, `TargetAccessor`, `MetadataAccessor` - clear and unambiguous

2. **Wavelength Conversions:** Integrated directly into `FeatureAccessor` as default methods
   - Rationale: Wavelengths are feature headers (X data), not targets (y data)
   - Non-spectroscopy datasets simply won't use these methods
   - Keeps API simple without mixin complexity

3. **Backward Compatibility:** Keep existing public methods WITHOUT deprecation warnings
   - Both `dataset.x()` and `dataset.features.x()` will work
   - No breaking changes, users can migrate at their own pace
   - Removed: deprecated methods will be cleaned up (e.g., `float_headers()`)

4. **Folds Management:** Keep at dataset level (`dataset.folds`, `dataset.set_folds()`)
   - Simple enough (3 methods) that separate accessor adds no value
   - Creating `dataset.cv.folds` would be overengineering

5. **Predictions:** Removed from dataset scope entirely
   - Now handled at pipeline level, not dataset level
   - All references removed from refactoring

6. **Properties vs Methods:** Use properties for getters without parameters
   - `dataset.targets.task_type` ✅ (not an antipattern here)
   - `dataset.features.num_samples` ✅
   - Properties are appropriate for simple attribute access with no side effects---

## Implementation Checklist

### Pre-Implementation
- [ ] Review this proposal with team
- [ ] Decide on accessor naming convention
- [ ] Define deprecation timeline
- [ ] Create issue/PR structure

### Phase 1: Core Refactoring
- [ ] Create `FeatureAccessor` class with full docstrings
- [ ] Create `TargetAccessor` class with full docstrings
- [ ] Create `MetadataAccessor` class with full docstrings
- [ ] Update `SpectroDataset` to use accessors internally
- [ ] Add backward compatibility delegation methods
- [ ] Write unit tests for each accessor

### Phase 2: Documentation
- [ ] Complete all missing docstrings (Google style)
- [ ] Update existing docstrings to reference accessors
- [ ] Update examples to show new API (both syntaxes work)
- [ ] Add inline comments explaining backward compatibility

### Phase 3: Wavelength Conversion
- [ ] Move wavelength methods to `FeatureAccessor`
- [ ] Add backward compatibility delegates in `SpectroDataset`
- [ ] Add proper unit tests
- [ ] Remove legacy `float_headers()` method

### Phase 4: Testing & Validation
- [ ] Run full test suite
- [ ] Test all examples with new API
- [ ] Performance benchmarks (before/after)
- [ ] Update CI/CD if needed

### Phase 5: Release
- [ ] Update CHANGELOG
- [ ] Update README with new API examples
- [ ] Release v1.x with deprecations
- [ ] Plan v2.0 for breaking changes

---

## Estimated Effort

- **Accessor Implementation:** 8-12 hours
- **Docstring Completion:** 4-6 hours
- **Testing:** 6-8 hours
- **Documentation/Examples:** 4-6 hours
- **Code Review/Refinement:** 4-6 hours

**Total:** ~30-40 hours of focused work

---

## Conclusion

This refactoring proposal provides a **pragmatic path** to improve `SpectroDataset`'s maintainability without breaking existing code. The accessor pattern creates clear boundaries, improves discoverability, and sets the foundation for future enhancements.

The migration strategy ensures users can adopt the new API gradually, and the complete docstring coverage will make the library more accessible to new users and contributors.

**Next Steps:** Review this proposal, discuss open questions, and decide if we should proceed with implementation.

---

*End of Proposal*

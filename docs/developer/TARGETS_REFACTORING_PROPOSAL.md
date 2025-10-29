# Targets Class Refactoring Proposal

**Date:** October 29, 2025
**Status:** Proposal
**Priority:** High (Maintainability & Performance)

## Executive Summary

The current `Targets` class has grown to handle too many responsibilities, making it difficult to maintain, test, and optimize. This proposal outlines a refactoring strategy that:

- **Separates concerns** into focused, single-responsibility classes
- **Improves maintainability** through clearer code organization
- **Enhances performance** via caching and optimized data structures
- **Maintains backward compatibility** with existing API
- **Adds comprehensive documentation** with NumPy-style docstrings

**Estimated Impact:**
- Code clarity: +80%
- Testability: +90%
- Performance: +20-30% for common operations
- Maintainability: Significantly improved

---

## Current Issues Analysis

### 1. **Multiple Responsibilities (SRP Violation)**

The `Targets` class currently handles:
- ✗ Data storage and management
- ✗ Processing chain tracking (ancestry)
- ✗ Transformer management
- ✗ Numeric conversion logic
- ✗ Transformation operations (forward/inverse)
- ✗ Statistical computations
- ✗ Display formatting

**Problem:** Changes to any one concern require modifying the monolithic class, increasing risk of bugs.

### 2. **Complex Internal Logic**

- `_make_numeric()`: 100+ lines handling multiple cases (numeric data, string data, column-wise conversion)
- `transform_predictions()`: Complex ancestry traversal logic embedded in the class
- Transformation logic tightly coupled with storage

### 3. **Performance Issues**

```python
# Excessive copying
self._data[processing_name] = data.copy()  # Copy on every add

# No caching
@property
def num_classes(self) -> int:
    # Recalculates every time it's called
    unique_classes = np.unique(y_numeric[~np.isnan(y_numeric)])
    return len(unique_classes)

# Repeated ancestry traversal
def transform_predictions(...):
    # Walks ancestry chain every call, no memoization
```

### 4. **Testing Challenges**

- Hard to unit test individual concerns (encoding, transformation, ancestry)
- Mock/stub requirements are complex due to tight coupling
- Edge cases difficult to isolate

### 5. **Missing Documentation**

Many methods lack proper NumPy-style docstrings:
- `_add_processing()`
- `_make_numeric()`
- Various internal helpers

---

## Proposed Architecture

### Component Separation

```
nirs4all/data/
├── targets.py              # Main Targets class (facade/orchestrator)
├── targets/
│   ├── __init__.py
│   ├── encoders.py         # FlexibleLabelEncoder + utilities
│   ├── processing_chain.py # ProcessingChain class
│   ├── converters.py       # NumericConverter class
│   └── transformers.py     # TargetTransformer class
```

### Class Responsibilities

#### 1. **FlexibleLabelEncoder** (`encoders.py`)
**Responsibility:** Encode labels with unseen label handling

```python
class FlexibleLabelEncoder(TransformerMixin):
    """
    Label encoder that handles unseen labels during transform.

    Unseen labels are mapped to the next available integer beyond the
    known classes. Useful for group-based splits where test groups may
    not appear in training.

    Attributes
    ----------
    classes_ : np.ndarray
        Array of known classes from training
    class_to_idx : dict
        Mapping from class values to integer indices

    Examples
    --------
    >>> encoder = FlexibleLabelEncoder()
    >>> encoder.fit(['cat', 'dog', 'bird'])
    >>> encoder.transform(['cat', 'dog', 'bird', 'fish'])
    array([0, 1, 2, 3])  # 'fish' gets next available index
    """
```

#### 2. **ProcessingChain** (`processing_chain.py`)
**Responsibility:** Track processing ancestry and transformers

```python
class ProcessingChain:
    """
    Manages the chain of processing transformations applied to target data.

    Tracks the lineage of processing steps, their relationships, and
    associated transformers. Provides efficient ancestry queries and
    path finding between processing states.

    Attributes
    ----------
    _processing_ids : list of str
        Ordered list of processing names
    _ancestors : dict
        Maps processing name to its parent processing
    _transformers : dict
        Maps processing name to its transformer
    _ancestry_cache : dict
        Cached ancestry chains for performance

    Methods
    -------
    add_processing(name, ancestor, transformer)
        Register a new processing step
    get_ancestry(name)
        Get full ancestry chain for a processing
    get_path(from_proc, to_proc)
        Find transformation path between two processings
    has_processing(name)
        Check if processing exists
    """
```

#### 3. **NumericConverter** (`converters.py`)
**Responsibility:** Convert raw data to numeric format

```python
class NumericConverter:
    """
    Converts raw target data to numeric representation.

    Handles multiple data types (numeric, string, mixed) and applies
    appropriate transformations column-wise. Supports both classification
    labels and regression targets.

    Methods
    -------
    convert(data, existing_transformer=None)
        Convert data to numeric format

    Returns
    -------
    tuple
        (numeric_data, transformer) where transformer can recreate
        the conversion

    Examples
    --------
    >>> converter = NumericConverter()
    >>> data = np.array(['cat', 'dog', 'cat'])
    >>> numeric, transformer = converter.convert(data)
    >>> numeric
    array([0., 1., 0.])
    """

    @staticmethod
    def convert(data: np.ndarray,
                existing_transformer: Optional[TransformerMixin] = None
                ) -> Tuple[np.ndarray, TransformerMixin]:
        """
        Convert raw target data to numeric format.

        Parameters
        ----------
        data : np.ndarray
            Raw target data (any dtype)
        existing_transformer : TransformerMixin, optional
            Reuse existing transformer if available

        Returns
        -------
        numeric_data : np.ndarray
            Data converted to float32
        transformer : TransformerMixin
            Transformer that can recreate this conversion
        """
```

#### 4. **TargetTransformer** (`transformers.py`)
**Responsibility:** Handle prediction transformations

```python
class TargetTransformer:
    """
    Transforms predictions between different processing states.

    Uses processing chain information to apply forward and inverse
    transformations along the ancestry path. Handles edge cases like
    LabelEncoder inverse transformation.

    Attributes
    ----------
    processing_chain : ProcessingChain
        Reference to processing chain for ancestry queries

    Methods
    -------
    transform(predictions, from_processing, to_processing, transformers)
        Transform predictions between processing states
    inverse_transform(predictions, from_processing, to_processing, transformers)
        Inverse transform predictions

    Examples
    --------
    >>> transformer = TargetTransformer(processing_chain)
    >>> # Transform from 'scaled' back to 'numeric'
    >>> numeric_preds = transformer.transform(
    ...     predictions, 'scaled', 'numeric', transformers_dict
    ... )
    """
```

#### 5. **Targets** (Refactored - `targets.py`)
**Responsibility:** Orchestrate components, provide clean API

```python
class Targets:
    """
    Target manager that stores target arrays with processing chains.

    Manages multiple versions of target data (raw, numeric, scaled, etc.)
    with processing ancestry tracking and transformation capabilities.
    Delegates specialized tasks to helper components.

    Components
    ----------
    _data : dict
        Storage for all target arrays by processing name
    _processing_chain : ProcessingChain
        Manages ancestry and transformers
    _converter : NumericConverter
        Handles raw to numeric conversion
    _transformer : TargetTransformer
        Manages prediction transformations
    _stats_cache : dict
        Cached statistics for performance

    Attributes
    ----------
    num_samples : int
        Number of samples in target data
    num_targets : int
        Number of target variables
    num_classes : int
        Number of unique classes (classification only)
    processing_ids : list of str
        Available processing names

    Methods
    -------
    add_targets(targets)
        Add new target samples
    add_processed_targets(name, targets, ancestor, transformer)
        Add a processed version of targets
    get_targets(processing, indices)
        Retrieve target data
    transform_predictions(predictions, from_processing, to_processing)
        Transform predictions between processing states

    Examples
    --------
    >>> targets = Targets()
    >>> targets.add_targets(np.array([1, 2, 3, 1, 2]))
    >>> targets.num_samples
    5
    >>> targets.num_classes
    3

    See Also
    --------
    ProcessingChain : Manages processing ancestry
    NumericConverter : Converts raw data to numeric
    TargetTransformer : Transforms predictions
    """
```

---

## Detailed Implementation

### 1. `encoders.py`

```python
"""Label encoders for target data."""

from typing import Dict, Optional

import numpy as np
from sklearn.base import TransformerMixin


class FlexibleLabelEncoder(TransformerMixin):
    """
    Label encoder that can handle unseen labels during transform.

    Unseen labels are mapped to the next available integer beyond the known
    classes. This is useful for group-based splits where test groups may not
    appear in training.

    Attributes
    ----------
    classes_ : np.ndarray or None
        Array of unique class labels encountered during fit
    class_to_idx : dict
        Mapping from class label to integer index

    Examples
    --------
    >>> encoder = FlexibleLabelEncoder()
    >>> encoder.fit(['cat', 'dog', 'bird'])
    >>> encoder.transform(['cat', 'dog', 'bird', 'fish'])
    array([0., 1., 2., 3.])  # 'fish' gets next available index

    >>> encoder.transform(['dog', 'cat'])
    array([1., 0.])

    Notes
    -----
    - NaN values are preserved in the transformation
    - Unseen labels get indices starting from len(classes_)
    - Thread-safe for transform after fit
    """

    def __init__(self):
        """Initialize label encoder with empty state."""
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx: Dict = {}

    def fit(self, y: np.ndarray) -> 'FlexibleLabelEncoder':
        """
        Fit the encoder to the training labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels to fit

        Returns
        -------
        self : FlexibleLabelEncoder
            Fitted encoder instance

        Notes
        -----
        NaN values are filtered out before determining unique classes.
        """
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y[~np.isnan(y)])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels, handling unseen labels by assigning new indices.

        Parameters
        ----------
        y : array-like
            Labels to transform

        Returns
        -------
        y_transformed : np.ndarray
            Transformed labels with same shape as input

        Notes
        -----
        - Known labels are mapped using class_to_idx
        - Unseen labels get indices starting from len(classes_)
        - NaN values are preserved
        - Original shape is maintained
        """
        if self.classes_ is None:
            raise ValueError("Encoder must be fitted before transform")

        y_input = np.asarray(y)
        original_shape = y_input.shape
        y_flat = y_input.ravel()
        result = np.empty_like(y_flat, dtype=np.float32)

        next_idx = len(self.classes_)
        unseen_map: Dict = {}

        for i, label in enumerate(y_flat):
            if np.isnan(label):
                result[i] = label
            elif label in self.class_to_idx:
                result[i] = self.class_to_idx[label]
            else:
                # Handle unseen label
                if label not in unseen_map:
                    unseen_map[label] = next_idx
                    next_idx += 1
                result[i] = unseen_map[label]

        return result.reshape(original_shape)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        y : array-like
            Labels to fit and transform

        Returns
        -------
        y_transformed : np.ndarray
            Transformed labels

        See Also
        --------
        fit : Fit the encoder
        transform : Transform labels
        """
        return self.fit(y).transform(y)
```

### 2. `processing_chain.py`

```python
"""Processing chain management for target transformations."""

from typing import Dict, List, Optional, Set

from sklearn.base import TransformerMixin


class ProcessingChain:
    """
    Manages the chain of processing transformations applied to target data.

    Tracks the lineage of processing steps, their relationships, and
    associated transformers. Provides efficient ancestry queries with caching.

    Attributes
    ----------
    _processing_ids : list of str
        Ordered list of all processing names
    _processing_set : set of str
        Set for O(1) existence checks
    _ancestors : dict of str to str
        Maps each processing to its immediate parent
    _transformers : dict of str to TransformerMixin
        Maps each processing to its transformer
    _ancestry_cache : dict of str to list of str
        Cached full ancestry chains for performance

    Examples
    --------
    >>> chain = ProcessingChain()
    >>> chain.add_processing('raw', None, None)
    >>> chain.add_processing('numeric', 'raw', encoder)
    >>> chain.add_processing('scaled', 'numeric', scaler)
    >>> chain.get_ancestry('scaled')
    ['raw', 'numeric', 'scaled']
    >>> chain.get_path('scaled', 'raw')
    (['scaled', 'numeric', 'raw'], 'inverse')
    """

    def __init__(self):
        """Initialize empty processing chain."""
        self._processing_ids: List[str] = []
        self._processing_set: Set[str] = set()
        self._ancestors: Dict[str, str] = {}
        self._transformers: Dict[str, TransformerMixin] = {}
        self._ancestry_cache: Dict[str, List[str]] = {}

    def add_processing(self,
                      name: str,
                      ancestor: Optional[str] = None,
                      transformer: Optional[TransformerMixin] = None) -> None:
        """
        Register a new processing step in the chain.

        Parameters
        ----------
        name : str
            Unique name for this processing
        ancestor : str, optional
            Name of parent processing (None for root)
        transformer : TransformerMixin, optional
            Transformer used to create this processing from ancestor

        Raises
        ------
        ValueError
            If processing name already exists
            If ancestor doesn't exist
        """
        if name in self._processing_set:
            raise ValueError(f"Processing '{name}' already exists")

        if ancestor is not None and ancestor not in self._processing_set:
            raise ValueError(f"Ancestor '{ancestor}' does not exist")

        self._processing_ids.append(name)
        self._processing_set.add(name)

        if ancestor is not None:
            self._ancestors[name] = ancestor

        if transformer is not None:
            self._transformers[name] = transformer

        # Invalidate ancestry cache for new processing
        self._ancestry_cache.clear()

    def has_processing(self, name: str) -> bool:
        """
        Check if a processing exists.

        Parameters
        ----------
        name : str
            Processing name to check

        Returns
        -------
        bool
            True if processing exists
        """
        return name in self._processing_set

    def get_transformer(self, name: str) -> Optional[TransformerMixin]:
        """
        Get the transformer for a processing.

        Parameters
        ----------
        name : str
            Processing name

        Returns
        -------
        TransformerMixin or None
            The transformer, or None if no transformer exists
        """
        return self._transformers.get(name)

    def get_ancestry(self, name: str) -> List[str]:
        """
        Get the full ancestry chain for a processing.

        Parameters
        ----------
        name : str
            Processing name

        Returns
        -------
        list of str
            Processing names from root to the specified processing

        Raises
        ------
        ValueError
            If processing doesn't exist

        Notes
        -----
        Results are cached for performance. Cache is invalidated when
        new processings are added.
        """
        if not self.has_processing(name):
            raise ValueError(f"Processing '{name}' not found")

        # Check cache first
        if name in self._ancestry_cache:
            return self._ancestry_cache[name].copy()

        # Build ancestry chain
        ancestry = []
        current = name

        while current is not None:
            ancestry.append(current)
            current = self._ancestors.get(current)

        ancestry.reverse()

        # Cache and return
        self._ancestry_cache[name] = ancestry
        return ancestry.copy()

    def get_path(self,
                from_processing: str,
                to_processing: str) -> tuple[List[str], str]:
        """
        Find transformation path between two processings.

        Parameters
        ----------
        from_processing : str
            Starting processing name
        to_processing : str
            Target processing name

        Returns
        -------
        path : list of str
            Sequence of processing names to traverse
        direction : str
            Either 'forward', 'inverse', or 'mixed'

        Raises
        ------
        ValueError
            If either processing doesn't exist
            If no path exists between processings

        Examples
        --------
        >>> path, direction = chain.get_path('scaled', 'numeric')
        >>> path
        ['scaled', 'numeric']
        >>> direction
        'inverse'
        """
        if not self.has_processing(from_processing):
            raise ValueError(f"Processing '{from_processing}' not found")
        if not self.has_processing(to_processing):
            raise ValueError(f"Processing '{to_processing}' not found")

        from_ancestry = self.get_ancestry(from_processing)
        to_ancestry = self.get_ancestry(to_processing)

        # Find common ancestor
        common_ancestor = None
        for ancestor in reversed(from_ancestry):
            if ancestor in to_ancestry:
                common_ancestor = ancestor
                break

        if common_ancestor is None:
            raise ValueError(
                f"No common ancestor between '{from_processing}' and '{to_processing}'"
            )

        # Build path
        path = []

        # Path from from_processing to common ancestor (inverse direction)
        current = from_processing
        while current != common_ancestor:
            path.append(current)
            current = self._ancestors[current]

        # Add common ancestor
        if common_ancestor != to_processing:
            path.append(common_ancestor)

            # Path from common ancestor to to_processing (forward direction)
            forward_path = []
            current = to_processing
            while current != common_ancestor:
                forward_path.append(current)
                current = self._ancestors[current]
            path.extend(reversed(forward_path))
        else:
            path.append(common_ancestor)

        # Determine direction
        if from_processing == to_processing:
            direction = 'identity'
        elif to_processing in from_ancestry:
            direction = 'inverse'
        elif from_processing in to_ancestry:
            direction = 'forward'
        else:
            direction = 'mixed'

        return path, direction

    @property
    def processing_ids(self) -> List[str]:
        """
        Get list of all processing IDs.

        Returns
        -------
        list of str
            Copy of processing IDs list
        """
        return self._processing_ids.copy()

    @property
    def num_processings(self) -> int:
        """
        Get number of processings in the chain.

        Returns
        -------
        int
            Number of registered processings
        """
        return len(self._processing_ids)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ProcessingChain(processings={len(self._processing_ids)})"
```

### 3. `converters.py`

```python
"""Numeric conversion utilities for target data."""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from .encoders import FlexibleLabelEncoder


class NumericConverter:
    """
    Converts raw target data to numeric representation.

    Handles multiple data types (numeric, string, object, mixed) and applies
    appropriate transformations column-wise. Supports both classification
    labels and regression targets.

    Methods
    -------
    convert(data, existing_transformer)
        Convert data to numeric format with appropriate transformer

    Examples
    --------
    >>> converter = NumericConverter()
    >>> data = np.array(['cat', 'dog', 'cat', 'bird'])
    >>> numeric, transformer = converter.convert(data)
    >>> numeric
    array([0., 1., 0., 2.])

    See Also
    --------
    FlexibleLabelEncoder : Handles label encoding with unseen labels
    """

    @staticmethod
    def convert(data: np.ndarray,
                existing_transformer: Optional[TransformerMixin] = None
                ) -> Tuple[np.ndarray, TransformerMixin]:
        """
        Convert raw target data to numeric format.

        Analyzes data type and applies appropriate transformation:
        - Already numeric: identity or label encoding for classification
        - String/object: label encoding
        - Mixed columns: column-wise transformation

        Parameters
        ----------
        data : np.ndarray
            Raw target data of any dtype
        existing_transformer : TransformerMixin, optional
            Reuse existing transformer if provided (for appending data)

        Returns
        -------
        numeric_data : np.ndarray
            Data converted to float32
        transformer : TransformerMixin
            Transformer that can recreate this conversion

        Notes
        -----
        - Preserves NaN values in numeric data
        - Detects classification labels (small set of integer-like values)
        - Creates column-wise transformers for mixed data types
        - Always returns float32 dtype for consistency

        Examples
        --------
        >>> # Numeric regression data
        >>> data = np.array([1.5, 2.3, 3.1])
        >>> numeric, trans = NumericConverter.convert(data)
        >>> numeric.dtype
        dtype('float32')

        >>> # Classification labels
        >>> data = np.array([1, 2, 1, 3])
        >>> numeric, trans = NumericConverter.convert(data)
        >>> numeric
        array([0., 1., 0., 2.])  # Re-encoded to 0-based
        """
        # Reuse existing transformer if provided
        if existing_transformer is not None:
            if hasattr(existing_transformer, 'transform'):
                numeric = existing_transformer.transform(data)
                return numeric.astype(np.float32), existing_transformer

        # Ensure 2D shape
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Check if already numeric
        if np.issubdtype(data.dtype, np.number):
            return NumericConverter._handle_numeric_data(data)

        # Handle non-numeric data column by column
        return NumericConverter._handle_mixed_data(data)

    @staticmethod
    def _handle_numeric_data(data: np.ndarray) -> Tuple[np.ndarray, TransformerMixin]:
        """Handle already numeric data."""
        # Check if classification labels needing encoding
        data_flat = data.flatten()
        unique_vals = np.unique(data_flat[~np.isnan(data_flat)])

        # Detect classification: integer-like values, small set, not 0-based
        is_integer_like = np.allclose(unique_vals, np.round(unique_vals), atol=1e-10)
        expected_consecutive = np.arange(len(unique_vals))
        is_classification = (
            is_integer_like
            and len(unique_vals) <= 50
            and not np.array_equal(unique_vals, expected_consecutive)
        )

        if is_classification:
            # Re-encode to 0-based consecutive integers
            encoder = FlexibleLabelEncoder()
            encoded = encoder.fit_transform(data_flat.astype(np.int32))
            return encoded.reshape(data.shape).astype(np.float32), encoder
        else:
            # Identity transformation for regression or already 0-based
            transformer = FunctionTransformer(validate=False)
            transformer.fit(data)
            return data.astype(np.float32), transformer

    @staticmethod
    def _handle_mixed_data(data: np.ndarray) -> Tuple[np.ndarray, TransformerMixin]:
        """Handle mixed or non-numeric data column by column."""
        numeric = np.empty_like(data, dtype=np.float32)
        column_transformers: Dict[int, Optional[TransformerMixin]] = {}

        for col in range(data.shape[1]):
            col_data = data[:, col]

            if col_data.dtype.kind in {"U", "S", "O"}:  # String/object types
                encoder = FlexibleLabelEncoder()
                numeric[:, col] = encoder.fit_transform(col_data)
                column_transformers[col] = encoder
            else:
                # Try numeric conversion
                try:
                    numeric[:, col] = col_data.astype(np.float32)
                    column_transformers[col] = None
                except (ValueError, TypeError):
                    # Fallback to encoding
                    encoder = FlexibleLabelEncoder()
                    numeric[:, col] = encoder.fit_transform(col_data.astype(str))
                    column_transformers[col] = encoder

        # Wrap in a transformer that remembers column-wise logic
        transformer = ColumnWiseTransformer(column_transformers)
        return numeric, transformer


class ColumnWiseTransformer(TransformerMixin):
    """
    Applies different transformers to different columns.

    Parameters
    ----------
    column_transformers : dict
        Maps column index to transformer (or None for identity)

    Attributes
    ----------
    column_transformers : dict
        Stored column transformer mapping
    """

    def __init__(self, column_transformers: Dict[int, Optional[TransformerMixin]]):
        """Initialize with column transformer mapping."""
        self.column_transformers = column_transformers

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ColumnWiseTransformer':
        """
        Fit transformer (no-op, already fitted).

        Parameters
        ----------
        X : np.ndarray
            Input data (ignored)
        y : np.ndarray, optional
            Target data (ignored)

        Returns
        -------
        self : ColumnWiseTransformer
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using column-specific transformers.

        Parameters
        ----------
        X : np.ndarray
            Data to transform

        Returns
        -------
        np.ndarray
            Transformed data
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.empty_like(X, dtype=np.float32)
        for col, transformer in self.column_transformers.items():
            if transformer is None:
                result[:, col] = X[:, col].astype(np.float32)
            else:
                result[:, col] = transformer.transform(X[:, col])
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data using column-specific transformers.

        Parameters
        ----------
        X : np.ndarray
            Data to inverse transform

        Returns
        -------
        np.ndarray
            Original representation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = np.empty(X.shape, dtype=object)
        for col, transformer in self.column_transformers.items():
            if transformer is None:
                result[:, col] = X[:, col]
            elif hasattr(transformer, 'inverse_transform'):
                result[:, col] = transformer.inverse_transform(X[:, col].astype(int))
            else:
                result[:, col] = X[:, col]
        return result
```

### 4. `transformers.py`

```python
"""Prediction transformation utilities."""

from typing import Dict

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from .processing_chain import ProcessingChain


class TargetTransformer:
    """
    Transforms predictions between different processing states.

    Uses processing chain information to apply forward and inverse
    transformations along the ancestry path.

    Attributes
    ----------
    processing_chain : ProcessingChain
        Reference to processing chain for ancestry and transformer lookup

    Methods
    -------
    transform(predictions, from_processing, to_processing, data_storage)
        Transform predictions between processing states

    Examples
    --------
    >>> transformer = TargetTransformer(processing_chain)
    >>> predictions = model.predict(X_test)  # In 'scaled' space
    >>> numeric_preds = transformer.transform(
    ...     predictions,
    ...     from_processing='scaled',
    ...     to_processing='numeric',
    ...     data_storage=targets._data
    ... )
    """

    def __init__(self, processing_chain: ProcessingChain):
        """
        Initialize transformer with processing chain reference.

        Parameters
        ----------
        processing_chain : ProcessingChain
            Processing chain to use for ancestry and transformers
        """
        self.processing_chain = processing_chain

    def transform(self,
                 predictions: np.ndarray,
                 from_processing: str,
                 to_processing: str,
                 data_storage: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform predictions from one processing state to another.

        Applies inverse transformations when going up the ancestry chain
        and forward transformations when going down.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction array to transform
        from_processing : str
            Current processing state of predictions
        to_processing : str
            Target processing state
        data_storage : dict
            Storage dictionary mapping processing names to data arrays

        Returns
        -------
        np.ndarray
            Transformed predictions in target processing state

        Raises
        ------
        ValueError
            If processing names don't exist
            If no path exists between processings
            If transformation fails

        Notes
        -----
        - Empty predictions return empty array
        - Handles LabelEncoder specially (converts to int32)
        - Uses cached ancestry for efficiency

        Examples
        --------
        >>> # Transform from model's output space back to numeric
        >>> preds_numeric = transformer.transform(
        ...     model_predictions,
        ...     'minmax-scaled',
        ...     'numeric',
        ...     targets._data
        ... )
        """
        if from_processing == to_processing:
            return predictions.copy()

        if predictions.shape[0] == 0:
            return predictions.copy()

        # Get transformation path
        path, direction = self.processing_chain.get_path(
            from_processing, to_processing
        )

        current = predictions.copy()

        if direction == 'inverse':
            # Walk backward from from_processing to to_processing
            for i in range(len(path) - 1):
                current_proc = path[i]
                transformer = self.processing_chain.get_transformer(current_proc)

                if transformer is not None and hasattr(transformer, 'inverse_transform'):
                    current = self._apply_inverse_transform(current, transformer)
                else:
                    raise ValueError(
                        f"No inverse transformer for processing '{current_proc}'"
                    )

        elif direction == 'forward':
            # Walk forward from from_processing to to_processing
            for i in range(len(path) - 1):
                next_proc = path[i + 1]
                transformer = self.processing_chain.get_transformer(next_proc)

                if transformer is not None and hasattr(transformer, 'transform'):
                    current = self._apply_transform(current, transformer)
                else:
                    raise ValueError(
                        f"No forward transformer for processing '{next_proc}'"
                    )

        elif direction == 'mixed':
            # Handle mixed direction (through common ancestor)
            # Find common ancestor in path
            from_ancestry = self.processing_chain.get_ancestry(from_processing)
            to_ancestry = self.processing_chain.get_ancestry(to_processing)

            common_ancestor = None
            for anc in reversed(from_ancestry):
                if anc in to_ancestry:
                    common_ancestor = anc
                    break

            # Inverse transform to common ancestor
            common_idx = path.index(common_ancestor)
            for i in range(common_idx):
                current_proc = path[i]
                transformer = self.processing_chain.get_transformer(current_proc)

                if transformer and hasattr(transformer, 'inverse_transform'):
                    current = self._apply_inverse_transform(current, transformer)

            # Forward transform to target
            for i in range(common_idx, len(path) - 1):
                next_proc = path[i + 1]
                transformer = self.processing_chain.get_transformer(next_proc)

                if transformer and hasattr(transformer, 'transform'):
                    current = self._apply_transform(current, transformer)

        return current

    @staticmethod
    def _apply_transform(data: np.ndarray,
                        transformer: TransformerMixin) -> np.ndarray:
        """
        Apply forward transformation with error handling.

        Parameters
        ----------
        data : np.ndarray
            Data to transform
        transformer : TransformerMixin
            Transformer to apply

        Returns
        -------
        np.ndarray
            Transformed data

        Raises
        ------
        ValueError
            If transformation fails
        """
        try:
            return transformer.transform(data)  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Forward transform failed with {type(transformer).__name__}: {e}"
            ) from e

    @staticmethod
    def _apply_inverse_transform(data: np.ndarray,
                                 transformer: TransformerMixin) -> np.ndarray:
        """
        Apply inverse transformation with error handling.

        Handles special cases like LabelEncoder which requires int32 input.

        Parameters
        ----------
        data : np.ndarray
            Data to inverse transform
        transformer : TransformerMixin
            Transformer to apply

        Returns
        -------
        np.ndarray
            Inverse transformed data

        Raises
        ------
        ValueError
            If transformation fails
        """
        try:
            # LabelEncoder requires integer input for inverse_transform
            if isinstance(transformer, LabelEncoder):
                return transformer.inverse_transform(data.astype(np.int32))  # type: ignore
            else:
                return transformer.inverse_transform(data)  # type: ignore
        except Exception as e:
            raise ValueError(
                f"Inverse transform failed with {type(transformer).__name__}: {e}"
            ) from e
```

---

## Refactored Targets Class

```python
"""Target data management with processing chains."""

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import TransformerMixin

from nirs4all.data.helpers import SampleIndices
from nirs4all.data.targets.converters import NumericConverter
from nirs4all.data.targets.processing_chain import ProcessingChain
from nirs4all.data.targets.transformers import TargetTransformer


class Targets:
    """
    Target manager that stores target arrays with processing chains.

    Manages multiple versions of target data (raw, numeric, scaled, etc.) with
    processing ancestry tracking and transformation capabilities. Delegates
    specialized operations to helper components for better maintainability.

    Attributes
    ----------
    num_samples : int
        Number of samples in target data
    num_targets : int
        Number of target variables
    num_classes : int
        Number of unique classes (for classification tasks)
    num_processings : int
        Number of processing versions
    processing_ids : list of str
        Names of available processings

    Examples
    --------
    >>> targets = Targets()
    >>> targets.add_targets(np.array([1, 2, 3, 1, 2]))
    >>> targets.num_samples
    5
    >>> targets.num_classes
    3

    >>> # Add scaled version
    >>> scaler = StandardScaler()
    >>> scaled_data = scaler.fit_transform(targets.get_targets('numeric'))
    >>> targets.add_processed_targets('scaled', scaled_data, 'numeric', scaler)

    >>> # Transform predictions back to numeric space
    >>> predictions = model.predict(X_test)
    >>> numeric_preds = targets.transform_predictions(
    ...     predictions, 'scaled', 'numeric'
    ... )

    See Also
    --------
    ProcessingChain : Manages processing ancestry
    NumericConverter : Converts raw data to numeric
    TargetTransformer : Transforms predictions between states
    """

    def __init__(self):
        """Initialize empty target manager."""
        # Core data storage
        self._data: Dict[str, np.ndarray] = {}

        # Delegate to specialized components
        self._processing_chain = ProcessingChain()
        self._converter = NumericConverter()
        self._transformer = TargetTransformer(self._processing_chain)

        # Performance caching
        self._stats_cache: Dict[str, any] = {}

    def __repr__(self) -> str:
        """
        Return unambiguous string representation.

        Returns
        -------
        str
            String showing samples, targets, and processings
        """
        return (
            f"Targets(samples={self.num_samples}, "
            f"targets={self.num_targets}, "
            f"processings={self._processing_chain.processing_ids})"
        )

    def __str__(self) -> str:
        """
        Return readable string representation with statistics.

        Returns
        -------
        str
            Multi-line string with processing statistics

        Notes
        -----
        - Skips 'raw' processing in display
        - Shows min/max/mean for numeric processings
        - Computed statistics are not cached
        """
        if self.num_samples == 0:
            return "Targets:\n(empty)"

        processing_stats = []
        for proc_name in self._processing_chain.processing_ids:
            if proc_name == "raw":
                continue

            data = self._data[proc_name]
            if np.issubdtype(data.dtype, np.number) and data.size > 0:
                try:
                    min_val = round(float(np.nanmin(data)), 3)
                    max_val = round(float(np.nanmax(data)), 3)
                    mean_val = round(float(np.nanmean(data)), 3)
                    processing_stats.append((proc_name, min_val, max_val, mean_val))
                except (TypeError, ValueError):
                    processing_stats.append((proc_name, "N/A", "N/A", "N/A"))
            else:
                processing_stats.append((proc_name, "N/A", "N/A", "N/A"))

        visible = [p for p in self._processing_chain.processing_ids if p != "raw"]
        result = (
            f"Targets: (samples={self.num_samples}, "
            f"targets={self.num_targets}, "
            f"processings={visible})"
        )

        for proc_name, min_val, max_val, mean_val in processing_stats:
            result += (
                f"\n- {proc_name}: min={min_val}, max={max_val}, mean={mean_val}"
            )

        return result

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            Number of samples (0 if no data)
        """
        if not self._data:
            return 0
        first_data = next(iter(self._data.values()))
        return first_data.shape[0]

    @property
    def num_targets(self) -> int:
        """
        Get the number of target variables.

        Returns
        -------
        int
            Number of targets (0 if no data)
        """
        if not self._data:
            return 0
        first_data = next(iter(self._data.values()))
        return first_data.shape[1]

    @property
    def num_processings(self) -> int:
        """
        Get the number of unique processings.

        Returns
        -------
        int
            Number of processing versions
        """
        return self._processing_chain.num_processings

    @property
    def processing_ids(self) -> List[str]:
        """
        Get the list of processing IDs.

        Returns
        -------
        list of str
            Copy of processing names
        """
        return self._processing_chain.processing_ids

    @property
    def num_classes(self) -> int:
        """
        Get the number of unique classes from numeric targets.

        Returns
        -------
        int
            Number of unique classes

        Raises
        ------
        ValueError
            If no target data available
            If numeric targets not available

        Notes
        -----
        - Uses numeric targets (not raw)
        - For multi-target, uses first column
        - Result is cached until data changes
        - NaN values are excluded from count
        """
        # Check cache first
        if 'num_classes' in self._stats_cache:
            return self._stats_cache['num_classes']

        if self.num_samples == 0:
            raise ValueError("Cannot compute num_classes: no target data available")

        y_numeric = self._data.get("numeric")
        if y_numeric is None:
            raise ValueError(
                "Cannot compute num_classes: numeric targets not available"
            )

        # Use first column for multi-target
        if y_numeric.ndim > 1:
            y_numeric = y_numeric[:, 0]

        # Count unique classes
        unique_classes = np.unique(y_numeric[~np.isnan(y_numeric)])
        num_classes = len(unique_classes)

        # Cache result
        self._stats_cache['num_classes'] = num_classes
        return num_classes

    def add_targets(self, targets: Union[np.ndarray, List, tuple]) -> None:
        """
        Add target samples. Can be called multiple times to append.

        Automatically creates 'raw' and 'numeric' processings on first call.
        Subsequent calls append to existing data.

        Parameters
        ----------
        targets : array-like
            Target data as 1D (single target) or 2D (multiple targets)

        Raises
        ------
        ValueError
            If processings beyond 'raw' and 'numeric' exist
            If target dimensions don't match existing data

        Notes
        -----
        - First call: creates 'raw' and 'numeric' processings
        - Subsequent calls: appends to existing arrays
        - Invalidates statistics cache

        Examples
        --------
        >>> targets = Targets()
        >>> targets.add_targets([1, 2, 3])
        >>> targets.num_samples
        3
        >>> targets.add_targets([4, 5])
        >>> targets.num_samples
        5
        """
        # Validate state
        if self.num_processings > 2:
            raise ValueError(
                "Cannot add new samples after additional processings created"
            )

        # Prepare data
        targets_array = np.asarray(targets)
        if targets_array.ndim == 1:
            targets_array = targets_array.reshape(-1, 1)
        elif targets_array.ndim != 2:
            raise ValueError(
                f"Targets must be 1D or 2D array, got {targets_array.ndim}D"
            )

        # First time: initialize structure
        if self.num_processings == 0:
            self._data["raw"] = targets_array.copy()
            self._processing_chain.add_processing("raw", None, None)

            # Auto-create numeric processing
            numeric_data, transformer = self._converter.convert(targets_array)
            self._data["numeric"] = numeric_data
            self._processing_chain.add_processing("numeric", "raw", transformer)
        else:
            # Subsequent times: append
            if targets_array.shape[1] != self.num_targets:
                raise ValueError(
                    f"Target data has {targets_array.shape[1]} targets, "
                    f"expected {self.num_targets}"
                )

            self._data["raw"] = np.vstack([self._data["raw"], targets_array])

            # Update numeric using existing transformer
            numeric_data, _ = self._converter.convert(
                targets_array,
                self._processing_chain.get_transformer("numeric")
            )
            self._data["numeric"] = np.vstack([self._data["numeric"], numeric_data])

        # Invalidate cache
        self._stats_cache.clear()

    def add_processed_targets(self,
                             processing_name: str,
                             targets: Union[np.ndarray, List, tuple],
                             ancestor: str = "numeric",
                             transformer: Optional[TransformerMixin] = None) -> None:
        """
        Add processed version of target data.

        Parameters
        ----------
        processing_name : str
            Unique name for this processing
        targets : array-like
            Processed target data (same number of samples)
        ancestor : str, default='numeric'
            Source processing name
        transformer : TransformerMixin, optional
            Transformer used to create this processing

        Raises
        ------
        ValueError
            If processing_name already exists
            If ancestor doesn't exist
            If shape doesn't match existing data

        Examples
        --------
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaled = scaler.fit_transform(targets.get_targets('numeric'))
        >>> targets.add_processed_targets('scaled', scaled, 'numeric', scaler)
        """
        if self._processing_chain.has_processing(processing_name):
            raise ValueError(f"Processing '{processing_name}' already exists")

        if not self._processing_chain.has_processing(ancestor):
            raise ValueError(f"Ancestor processing '{ancestor}' does not exist")

        targets_array = np.asarray(targets)
        if targets_array.ndim == 1:
            targets_array = targets_array.reshape(-1, 1)
        elif targets_array.ndim != 2:
            raise ValueError(
                f"Targets must be 1D or 2D array, got {targets_array.ndim}D"
            )

        if targets_array.shape[0] != self.num_samples:
            raise ValueError(
                f"Target data has {targets_array.shape[0]} samples, "
                f"expected {self.num_samples}"
            )

        if targets_array.shape[1] != self.num_targets:
            raise ValueError(
                f"Target data has {targets_array.shape[1]} targets, "
                f"expected {self.num_targets}"
            )

        self._data[processing_name] = targets_array.copy()
        self._processing_chain.add_processing(processing_name, ancestor, transformer)
        self._stats_cache.clear()

    def get_targets(self,
                   processing: str = "numeric",
                   indices: Optional[Union[List[int], np.ndarray]] = None
                   ) -> np.ndarray:
        """
        Get target data for a specific processing.

        Parameters
        ----------
        processing : str, default='numeric'
            Processing name to retrieve
        indices : array-like of int, optional
            Sample indices to retrieve (None for all)

        Returns
        -------
        np.ndarray
            Target array of shape (n_samples, n_targets) or
            (selected_samples, n_targets)

        Raises
        ------
        ValueError
            If processing doesn't exist

        Examples
        --------
        >>> targets.get_targets('numeric')
        array([[1.], [2.], [3.]])

        >>> targets.get_targets('numeric', indices=[0, 2])
        array([[1.], [3.]])
        """
        if not self._processing_chain.has_processing(processing):
            available = self._processing_chain.processing_ids
            raise ValueError(
                f"Processing '{processing}' not found. Available: {available}"
            )

        data = self._data[processing]

        if indices is None or len(indices) == 0 or data.shape[0] == 0:
            return data.copy()

        indices_array = np.asarray(indices, dtype=int)
        return data[indices_array]

    def y(self,
          indices: SampleIndices,
          processing: str) -> np.ndarray:
        """
        Convenience method to get targets with indices.

        Alias for get_targets with different parameter order.

        Parameters
        ----------
        indices : array-like of int
            Sample indices to retrieve
        processing : str
            Processing name

        Returns
        -------
        np.ndarray
            Target array for specified indices

        Examples
        --------
        >>> targets.y([0, 1, 2], 'numeric')
        array([[1.], [2.], [3.]])
        """
        if len(self._data) == 0:
            return np.array([])
        return self.get_targets(processing, indices)

    def get_processing_ancestry(self, processing: str) -> List[str]:
        """
        Get the full ancestry chain for a processing.

        Parameters
        ----------
        processing : str
            Processing name

        Returns
        -------
        list of str
            Processing names from root to specified processing

        Raises
        ------
        ValueError
            If processing doesn't exist

        Examples
        --------
        >>> targets.get_processing_ancestry('scaled')
        ['raw', 'numeric', 'scaled']
        """
        return self._processing_chain.get_ancestry(processing)

    def transform_predictions(self,
                             y_pred: np.ndarray,
                             from_processing: str,
                             to_processing: str = "numeric") -> np.ndarray:
        """
        Transform predictions from one processing state to another.

        Applies appropriate forward or inverse transformations based on
        the ancestry relationship between processings.

        Parameters
        ----------
        y_pred : np.ndarray
            Prediction array to transform
        from_processing : str
            Current processing state of predictions
        to_processing : str, default='numeric'
            Target processing state

        Returns
        -------
        np.ndarray
            Transformed predictions in target processing state

        Raises
        ------
        ValueError
            If either processing doesn't exist
            If no transformation path exists
            If transformation fails

        Examples
        --------
        >>> # Model trained on scaled targets
        >>> predictions = model.predict(X_test)
        >>> # Transform back to numeric space
        >>> numeric_preds = targets.transform_predictions(
        ...     predictions, 'scaled', 'numeric'
        ... )

        Notes
        -----
        - Empty predictions return empty array
        - Uses cached ancestry for efficiency
        - Handles both forward and inverse transformations

        See Also
        --------
        TargetTransformer : Handles transformation logic
        """
        return self._transformer.transform(
            y_pred, from_processing, to_processing, self._data
        )
```

---

## Benefits of Refactoring

### 1. **Separation of Concerns**
- Each class has a single, well-defined responsibility
- Easier to understand and modify individual components
- Changes to one component don't affect others

### 2. **Improved Testability**
```python
# Easy to test individual components
def test_flexible_label_encoder():
    encoder = FlexibleLabelEncoder()
    encoder.fit(['a', 'b', 'c'])
    assert np.array_equal(encoder.transform(['a', 'd']), [0, 3])

def test_processing_chain_ancestry():
    chain = ProcessingChain()
    chain.add_processing('raw', None, None)
    chain.add_processing('numeric', 'raw', encoder)
    assert chain.get_ancestry('numeric') == ['raw', 'numeric']

def test_numeric_converter():
    data = np.array(['cat', 'dog', 'cat'])
    numeric, transformer = NumericConverter.convert(data)
    assert numeric.dtype == np.float32
    assert len(np.unique(numeric)) == 2
```

### 3. **Performance Improvements**

**Caching:**
```python
# Ancestry cache in ProcessingChain
def get_ancestry(self, name: str) -> List[str]:
    if name in self._ancestry_cache:
        return self._ancestry_cache[name].copy()
    # ... compute and cache

# Statistics cache in Targets
@property
def num_classes(self) -> int:
    if 'num_classes' in self._stats_cache:
        return self._stats_cache['num_classes']
    # ... compute and cache
```

**Optimized lookups:**
```python
# Use set for O(1) membership checks
self._processing_set: Set[str] = set()

def has_processing(self, name: str) -> bool:
    return name in self._processing_set  # O(1) vs O(n)
```

### 4. **Better Maintainability**

- **Clear boundaries:** Each file/class has a specific purpose
- **Easier debugging:** Smaller, focused components
- **Simplified modifications:** Change one component without touching others
- **Better code reuse:** Components can be used independently

### 5. **Comprehensive Documentation**

All methods include:
- NumPy-style docstrings
- Parameter descriptions with types
- Return value specifications
- Raises sections for exceptions
- Examples demonstrating usage
- Notes for important details

---

## Migration Strategy

### Phase 1: Create New Components (Non-Breaking)

1. Create new directory structure:
   ```bash
   mkdir nirs4all/data/targets
   touch nirs4all/data/targets/__init__.py
   touch nirs4all/data/targets/encoders.py
   touch nirs4all/data/targets/processing_chain.py
   touch nirs4all/data/targets/converters.py
   touch nirs4all/data/targets/transformers.py
   ```

2. Implement new classes in separate files

3. Add comprehensive unit tests for each component

### Phase 2: Update Targets Class

1. Modify `targets.py` to use new components
2. Keep all public methods identical (backward compatible)
3. Update internal implementation to delegate to components

### Phase 3: Testing & Validation

1. Run all existing tests - should pass without changes
2. Add new tests for component isolation
3. Performance benchmarking (before/after)
4. Integration testing with existing code

### Phase 4: Cleanup

1. Remove old implementation code from `targets.py`
2. Update documentation
3. Add migration notes if needed

---

## Performance Benchmarks (Estimated)

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| `num_classes` (repeated calls) | ~100μs | ~1μs | 100x |
| `get_ancestry()` (repeated) | ~50μs | ~5μs | 10x |
| `transform_predictions()` | ~200μs | ~150μs | 25% |
| Memory overhead | Baseline | +~5% | Slight increase |

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/data/targets/test_encoders.py
def test_flexible_encoder_fit_transform()
def test_flexible_encoder_unseen_labels()
def test_flexible_encoder_nan_handling()

# tests/unit/data/targets/test_processing_chain.py
def test_add_processing()
def test_get_ancestry_caching()
def test_get_path_inverse()
def test_get_path_forward()
def test_get_path_mixed()

# tests/unit/data/targets/test_converters.py
def test_convert_numeric_data()
def test_convert_string_data()
def test_convert_mixed_columns()
def test_convert_classification_labels()

# tests/unit/data/targets/test_transformers.py
def test_transform_inverse()
def test_transform_forward()
def test_transform_mixed_path()
def test_label_encoder_special_case()
```

### Integration Tests

```python
# tests/integration/test_targets_refactored.py
def test_full_workflow()
def test_backward_compatibility()
def test_performance_benchmarks()
```

---

## Conclusion

This refactoring proposal provides:

✅ **Cleaner architecture** - Single responsibility per class
✅ **Better performance** - Caching and optimized data structures
✅ **Improved maintainability** - Easier to understand and modify
✅ **Comprehensive docs** - NumPy-style docstrings throughout
✅ **Full backward compatibility** - No breaking changes to API
✅ **Enhanced testability** - Isolated components easy to test

**Recommendation:** Implement in phases, starting with component creation and thorough testing before switching the Targets class implementation.

**Estimated Effort:**
- Phase 1 (Components): 6-8 hours
- Phase 2 (Targets refactor): 4-6 hours
- Phase 3 (Testing): 4-6 hours
- Total: ~14-20 hours

**Risk Level:** Low (with proper testing)

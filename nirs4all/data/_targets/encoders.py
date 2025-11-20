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

    Attributes:
        classes_ (np.ndarray or None): Array of unique class labels encountered during fit
        class_to_idx (dict): Mapping from class label to integer index

    Examples:
    >>> encoder = FlexibleLabelEncoder()
    >>> encoder.fit(['cat', 'dog', 'bird'])
    >>> encoder.transform(['cat', 'dog', 'bird', 'fish'])
    array([0., 1., 2., 3.])  # 'fish' gets next available index

    >>> encoder.transform(['dog', 'cat'])
    array([1., 0.])

    Notes:
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

        Args:
            y (array-like of shape (n_samples,)): Target labels to fit

        Returns:
            FlexibleLabelEncoder: Fitted encoder instance

        Notes:
        NaN values are filtered out before determining unique classes.
        """
        y = np.asarray(y).ravel()
        # Filter NaN only for numeric types
        if np.issubdtype(y.dtype, np.number):
            mask = ~np.isnan(y)
        else:
            mask = np.ones(len(y), dtype=bool)
        self.classes_ = np.unique(y[mask])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform labels, handling unseen labels by assigning new indices.

        Args:
            y (array-like): Labels to transform

        Returns:
            y_transformed (np.ndarray): Transformed labels with same shape as input

        Notes:
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
            # Check for NaN only on numeric types
            is_nan = False
            if np.issubdtype(y_flat.dtype, np.number):
                is_nan = np.isnan(label)

            if is_nan:
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

        Args:
            y (array-like): Labels to fit and transform

        Returns:
            y_transformed (np.ndarray): Transformed labels

        See Also:
        fit: Fit the encoder
        transform: Transform labels
        """
        return self.fit(y).transform(y)

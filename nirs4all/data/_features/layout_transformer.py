"""Layout transformation for feature arrays."""

import numpy as np
from nirs4all.data._features.feature_constants import FeatureLayout, normalize_layout, LayoutType


class LayoutTransformer:
    """Transforms 3D feature arrays to different layouts.

    Handles conversion between different data layouts for compatibility with
    various machine learning frameworks.
    """

    @staticmethod
    def transform(
        data: np.ndarray,
        layout: LayoutType,
        num_processings: int,
        num_features: int
    ) -> np.ndarray:
        """Transform 3D array to requested layout.

        Args:
            data: 3D array of shape (samples, processings, features).
            layout: Desired output layout.
            num_processings: Number of processing dimensions.
            num_features: Number of features.

        Returns:
            Transformed array in requested layout.

        Raises:
            ValueError: If layout is unknown.
        """
        layout_enum = normalize_layout(layout)

        if layout_enum == FeatureLayout.FLAT_2D:
            # Flatten to (samples, processings * features)
            return data.reshape(data.shape[0], -1)

        elif layout_enum == FeatureLayout.FLAT_2D_INTERLEAVED:
            # Transpose then flatten to (samples, features * processings)
            return np.transpose(data, (0, 2, 1)).reshape(data.shape[0], -1)

        elif layout_enum == FeatureLayout.VOLUME_3D:
            # Keep as (samples, processings, features)
            return data

        elif layout_enum == FeatureLayout.VOLUME_3D_TRANSPOSE:
            # Transpose to (samples, features, processings)
            return np.transpose(data, (0, 2, 1))

        else:
            raise ValueError(f"Unknown layout: {layout}")

    @staticmethod
    def get_empty_array(
        layout: LayoutType,
        num_processings: int,
        num_features: int,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """Create an empty array with the correct shape for the layout.

        Args:
            layout: Desired layout.
            num_processings: Number of processing dimensions.
            num_features: Number of features.
            dtype: Data type for the array.

        Returns:
            Empty array with shape matching the layout.
        """
        layout_enum = normalize_layout(layout)

        if layout_enum in [FeatureLayout.FLAT_2D, FeatureLayout.FLAT_2D_INTERLEAVED]:
            return np.empty((0, num_processings * num_features), dtype=dtype)
        else:
            return np.empty((0, num_processings, num_features), dtype=dtype)

    def __repr__(self) -> str:
        return "LayoutTransformer()"

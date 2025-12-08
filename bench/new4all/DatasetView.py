import polars as pl
import numpy as np
from typing import Any, Dict, List, Optional, Union


class DatasetView:
    """Efficient view of dataset with applied filters - avoids copying data."""

    def __init__(self, dataset: 'SpectraDataset', filters: Dict[str, Any]):
        # Import here to avoid circular import
        try:
            from SpectraDataset import SpectraDataset
        except ImportError:
            from SpectraDataset import SpectraDataset
        self.dataset = dataset
        self.filters = filters
        self._cached_selection = None
        self._cache_valid = True

    def _get_selection(self) -> pl.DataFrame:
        """Get filtered indices with caching."""
        if not self._cache_valid or self._cached_selection is None:
            selection = self.dataset.indices

            for key, value in self.filters.items():
                if isinstance(value, (list, tuple, set)):
                    selection = selection.filter(pl.col(key).is_in(list(value)))
                else:
                    selection = selection.filter(pl.col(key) == value)

            self._cached_selection = selection
            self._cache_valid = True

        return self._cached_selection

    def __len__(self) -> int:
        return len(self._get_selection())

    @property
    def sample_ids(self) -> List[int]:
        return self._get_selection()["sample"].to_list()

    @property
    def row_indices(self) -> np.ndarray:
        return self._get_selection()["row"].to_numpy()

    def get_features(self, source_indices: Optional[Union[int, List[int]]] = None,
                     concatenate: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Get features for this view."""
        return self.dataset.get_features(self.row_indices, source_indices, concatenate)

    def get_targets(self, representation: str = "auto",
                   transformer_key: Optional[str] = None) -> np.ndarray:
        """Get targets for this view using the new target management system."""
        return self.dataset.get_targets(self.sample_ids, representation, transformer_key)

    def split_by(self, column: str) -> Dict[Any, 'DatasetView']:
        """Split view by a column value."""
        selection = self._get_selection()
        unique_values = selection[column].unique().to_list()

        views = {}
        for value in unique_values:
            new_filters = {**self.filters, column: value}
            views[value] = DatasetView(self.dataset, new_filters)

        return views

    def invalidate_cache(self):
        """Invalidate cached selection."""
        self._cache_valid = False

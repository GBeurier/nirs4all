from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import polars as pl

from nirs4all.dataset.feature_source import FeatureSource
from nirs4all.dataset.indexer import SampleIndexManager


class Features:
    """Manages N aligned NumPy sources + a Polars index."""

    def __init__(self):
        """Initialize empty feature block."""
        self.sources: List[FeatureSource] = []
        self.index: Optional[pl.DataFrame] = None
        self.indexer: Optional[SampleIndexManager] = None

    def add_features(
        self,
        filter_dict: Dict[str, Any],
        x_list: np.ndarray | List[np.ndarray],
    ) -> None:

        if not x_list:
            raise ValueError("x_list cannot be empty")

        if not isinstance(x_list, list):
            if isinstance(x_list, np.ndarray):
                x_list = [x_list]
            else:
                raise TypeError(
                    f"x_list must be a list or a numpy array, got {type(x_list)}"
                )

        if not all(isinstance(arr, np.ndarray) for arr in x_list):
            raise TypeError("All elements in x_list must be numpy arrays")

        if not all(arr.ndim == 2 for arr in x_list):
            raise ValueError("All arrays in x_list must be 2-D numpy arrays")

        n_rows = x_list[0].shape[0]
        for i, arr in enumerate(x_list):
            if arr.ndim != 2:
                raise ValueError(f"Array {i} must be 2-D, got shape {arr.shape}")
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Array {i} has {arr.shape[0]} rows, expected {n_rows}"
                )

        # -----------------------------------------------------------------
        #  CAS 1 : première insertion  → création des sources + indexer
        # -----------------------------------------------------------------
        if not self.sources:
            # 1) création des sources
            for arr in x_list:
                self.sources.append(FeatureSource(arr))

            # 2) création de l’index initial
            self._build_initial_index(n_rows, overrides=filter_dict)

            # 3) indexer avec valeurs par défaut de base
            base_defaults = {
                "partition": "train",
                "group": 0,
                "branch": 0,
                "processing": "raw",
            }
            self.indexer = SampleIndexManager(
                self.index, default_values=base_defaults
            )

        # -----------------------------------------------------------------
        #  CAS 2 : sources déjà présentes  → on AJOUTE des échantillons
        # -----------------------------------------------------------------
        else:
            if len(x_list) != len(self.sources):
                raise ValueError(
                    f"Expected {len(self.sources)} arrays (one per source), "
                    f"got {len(x_list)}"
                )

            # 1) ajout des échantillons dans chaque source
            for src, new_arr in zip(self.sources, x_list):
                src.add_samples(new_arr)

            # 2) ajout des lignes dans l’index (avec overrides éventuels)
            self.indexer.add_rows(n_rows, overrides=filter_dict)

        # -----------------------------------------------------------------
        #  Synchronisation de la copie locale de l’index
        # -----------------------------------------------------------------
        self.index = self.indexer.df

    def _build_initial_index(
        self,
        n_rows: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Construit l'index de base + injection d'overrides éventuels.
        """
        data = {
            "row":       pl.Series(range(n_rows), dtype=pl.Int32),
            "sample":    pl.Series(range(n_rows), dtype=pl.Int32),
            "origin":    pl.Series(range(n_rows), dtype=pl.Int32),
            "partition": pl.Series(["train"] * n_rows, dtype=pl.Categorical),
            "group":     pl.Series([0] * n_rows, dtype=pl.Int8),
            "branch":    pl.Series([0] * n_rows, dtype=pl.Int8),
            "processing": pl.Series(["raw"] * n_rows, dtype=pl.Categorical),
        }

        overrides = overrides or {}
        for col, val in overrides.items():
            if isinstance(val, list):
                if len(val) != n_rows:
                    raise ValueError(
                        f"Override list for '{col}' must have {n_rows} elements"
                    )
                data[col] = pl.Series(val)
            else:
                data[col] = pl.Series([val] * n_rows)

        self.index = pl.DataFrame(data)

    @property
    def num_samples(self) -> int:
        """Get the number of samples (rows) across all sources."""
        if not self.sources:
            return 0
        return self.sources[0].num_samples

    def __repr__(self):
        n_sources = len(self.sources)
        n_samples = self.num_samples
        return f"FeatureBlock(sources={n_sources}, samples={n_samples})"


    #####################################################################################

    def x(self, filter_dict: Dict[str, Any], layout: str = "2d", src_concat: bool = False) -> np.ndarray | Tuple[np.ndarray, ...]:
        """
        Get feature arrays with specified layout and filtering.

        Args:
            filter_dict: Dictionary of column: value pairs for filtering
            layout: Layout type ("2d", "2d_interleaved", "3d", "3d_transpose")
            src_concat: Whether to concatenate sources along axis=1

        Returns:
            Tuple of numpy arrays (zero-copy views)
        """
        if not self.sources:
            raise ValueError("No features available")

        row_ranges = self.indexer.get_contiguous_ranges(filter_dict)
        res = []
        for source in self.sources:
            if layout == "2d":
                res.append(source.layout_2d(row_ranges))
            elif layout == "2d_interleaved":
                res.append(source.layout_2d_interleaved(row_ranges))
            elif layout == "3d":
                res.append(source.layout_3d(row_ranges))
            elif layout == "3d_transpose":
                res.append(source.layout_3d_transpose(row_ranges))
            else:
                raise ValueError(f"Unknown layout: {layout}")

        if src_concat:
            return np.concatenate(res, axis=1)

        return tuple(res)


    # def update_processing(self, processing_id: str | int, new_data: np.ndarray, new_processing: str) -> None:
    #     pass

    # def augment_samples(self, count: Union[int, List[int]], indices: List[int] | None = None) -> None:
    #     pass

    # def add_samples(self, new_samples: np.ndarray) -> None:
    #     pass

    # def add_processings(self, processing_id: Union[str, List[str]]) -> None:
    #     pass

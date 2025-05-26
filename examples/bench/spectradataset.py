# spectradataset.py
"""A module for handling spectral datasets with features, indices, targets, and results."""

from typing import Any, Sequence, Mapping, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl


indices_schema = {
    "row": pl.Int64,
    "sample": pl.Int64,
    "origin": pl.Int64,
    "spectra_type": pl.Utf8,
    "partition": pl.Utf8,
    "processing": pl.Utf8,
    "group": pl.Int32,
    "branch": pl.Int32,
    "seed": pl.Int64,
}

targets_schema = {
    "sample": pl.Int64,
    "target": pl.List(pl.Float64),
    "metadata": pl.List(pl.Object),
}

results_schema = {
    "sample": pl.Int64,
    "seed": pl.Int64,
    "branch": pl.Int32,
    "model": pl.Utf8,
    "fold": pl.Int32,
    "stack_index": pl.Int32,
    "prediction": pl.Float64,
    "datetime": pl.Datetime,
}


class SpectraFeatures:
    """Class to handle spectral features, allowing for multiple sources."""
    def __init__(self, sources: Union[np.ndarray, List[np.ndarray]]) -> None:
        if isinstance(sources, np.ndarray):
            sources = [sources]
        self.sources = sources
        self.total_width = sum(source.shape[1] for source in sources)
        self.ranges = []
        start = 0
        for source in sources:
            end = start + source.shape[1]
            self.ranges.append((start, end))
            start = end

    def __len__(self) -> int:
        return len(self.sources[0]) if self.sources else 0

    def __repr__(self) -> str:
        text = "num_sources: " + str(self.num_sources) + "\n"
        for i, spectra in enumerate(self.sources):
            text += f"Source {i}: shape={spectra.shape}, dtype={spectra.dtype}\n"
        return text

    @property
    def num_sources(self) -> int:
        """Return the number of sources."""
        return len(self.sources)

    @property
    def max_width(self) -> int:
        """Return the maximum width of the sources."""
        return max(source.shape[1] for source in self.sources) if self.sources else 0

    def concatenate(self, sources: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Concatenate new sources to the existing ones."""
        if isinstance(sources, np.ndarray):
            sources = [sources]

        for i, source in enumerate(sources):
            if i < len(self.sources):
                self.sources[i] = np.concatenate((self.sources[i], source), axis=0)
            else:
                self.sources.append(source)

    def to_numpy(self, indices: Optional[np.ndarray] = None, sources: Optional[Union[int, Sequence[int]]] = None,
                 merge: bool = True, pad: bool = False, pad_value: float = np.nan) -> Union[np.ndarray, List[np.ndarray]]:
        """Convert sources to a NumPy array or list of arrays based on indices."""
        if indices is None:
            indices = np.arange(len(self))

        if self.num_sources == 0:
            return np.array([])

        from_sources = np.arange(self.num_sources)
        if isinstance(sources, int):
            from_sources = [sources]
        elif isinstance(sources, Sequence):
            from_sources = list(sources)

        if len(from_sources) == 0:
            return np.array([])

        if len(from_sources) == 1:
            return self.sources[from_sources[0]][indices]

        if merge:
            return np.concatenate([self.sources[i][indices] for i in from_sources], axis=1)

        return [self.sources[i][indices] for i in from_sources]


class SpectraDataset:
    """A class to handle spectral datasets with features, indices, targets, and results."""
    def __init__(self, float64: bool = True) -> None:

        self.float64 = float64

        # Internal counters
        self._next_row_id = 0
        self._next_sample_id = 0

        # Data structures
        self.features = None
        self.indices = pl.DataFrame(schema=indices_schema)
        self.targets = pl.DataFrame(schema=targets_schema)
        self.results = pl.DataFrame(schema=results_schema)
        self.folds = []

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        txt = f"SpectraDataset with {len(self)} samples\n"
        txt += f"Features: {self.features}\n"
        txt += f"Indices: {self.indices.shape[0]} rows\n"
        txt += f"Targets: {self.targets.shape[0]} rows\n"
        txt += f"Results: {self.results.shape[0]} rows\n"
        txt += f"Folds: {len(self.folds)}\n"
        return txt

    def add_spectra(
        self,
        sources: Union[np.ndarray, List[np.ndarray]],
        targets: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None,
        sample: Optional[Union[int, Sequence[int], None]] = None,
        origin: Optional[Union[int, Sequence[int], None, Sequence[None]]] = None,
        spectra_type: Optional[Union[str, Sequence[str]]] = "nirs",
        partition: Optional[Union[str, Sequence[str]]] = "train",
        processing: Optional[Union[str, Sequence[str]]] = None,
        group: Optional[Union[int, Sequence[int]]] = 0,
        branch: Optional[Union[int, Sequence[int]]] = 0,
        seed: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        """Add spectra to the dataset with associated metadata."""

        if isinstance(sources, np.ndarray):
            sources = [sources]

        num_new_samples = len(sources[0])
        if num_new_samples == 0:
            raise ValueError("Sources must not be empty.")

        for i, source in enumerate(sources):
            if source.shape[0] != num_new_samples:
                raise ValueError("All sources must have the same number of samples.")

        if self.features is None:
            self.features = SpectraFeatures(sources)
        else:
            if len(sources) != self.features.num_sources:
                raise ValueError("Length of sources must match length of existing features.")
            for i, source in enumerate(sources):
                if source.shape[1] != self.features.sources[i].shape[1]:
                    raise ValueError(f"Source {i} width does not match existing features.")

            self.features.concatenate(sources)

        if sample is None:
            sample = np.arange(self._next_sample_id, self._next_sample_id + num_new_samples).tolist()
            self._next_sample_id += num_new_samples
        elif isinstance(sample, Sequence):
            if len(sample) != num_new_samples:
                raise ValueError("Length of sample must match length of features.")
            self._next_sample_id = max(self._next_sample_id, max(sample) + 1)
        else:
            raise ValueError("Sample must be a sequence of integers or None.")

        indices_data = {
            "row": np.arange(self._next_row_id, self._next_row_id + num_new_samples),
            "sample": sample,
            "origin": origin,
            "spectra_type": spectra_type,
            "partition": partition,
            "processing": processing,
            "group": group,
            "branch": branch,
            "seed": seed,
        }
        self._next_row_id += num_new_samples
        self.indices = pl.concat([self.indices, pl.DataFrame(indices_data)])


    def x(self, selector: Mapping[str, Any] | None = None, *, sources: Optional[Union[int, Sequence[int]]] = None,
          source_merge: bool = True, pad: bool = False, pad_value: float = np.nan) -> Union[np.ndarray, List[np.ndarray]]:
        """Get features based on the provided selector."""

        filtered = self._select_indices(selector)

        if filtered.is_empty():
            return np.array([])

        rows_filtered = filtered["row"].to_numpy()

        if self.features is None:
            return np.array([])

        return self.features.to_numpy(indices=rows_filtered, sources=sources, merge=source_merge, pad=pad, pad_value=pad_value)

    def _mask(self, **selector) -> pl.Expr:
        """Create boolean mask from filters."""
        if not selector:
            return pl.lit(True)

        exprs = []
        for k, v in selector.items():
            if isinstance(v, (list, tuple, set, np.ndarray)):
                exprs.append(pl.col(k).is_in(list(v)))
            else:
                exprs.append(pl.col(k) == v)

        return exprs[0] if len(exprs) == 1 else pl.all_horizontal(exprs)

    def _select_indices(self, selector: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """Select indices based on the provided criteria."""
        filtered = self.indices

        if selector:
            filtered = filtered.filter(self._mask(**selector))

        return filtered

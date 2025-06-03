# spectradataset.py OLD VERSION


"""A module for handling spectral datasets with features, indices, targets, and results."""

from typing import Any, Sequence, Mapping, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from sklearn.base import TransformerMixin

from spectra_features import SpectraFeatures

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

        if self.features is None:
            self.features = SpectraFeatures(sources)
        else:
            self.features.concatenate(sources)

        if sample is None:
            sample = np.arange(self._next_sample_id, self._next_sample_id + num_new_samples).tolist()
            self._next_sample_id += num_new_samples
        elif isinstance(sample, Sequence):
            self._next_sample_id = max(self._next_sample_id, max(sample) + 1)

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
          merge: bool = True, pad: bool = False, pad_value: float = np.nan) -> Union[np.ndarray, List[np.ndarray]]:
        """Get features based on the provided selector."""

        filtered = self._select_indices(selector)
        rows_filtered = filtered["row"].to_numpy()
        return self.features.to_numpy(indices=rows_filtered, sources=sources, merge=merge, pad=pad, pad_value=pad_value)

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

    def copy_x(self, selector: Mapping[str, Any] | None = None, *, sources: Optional[Union[int, Sequence[int]]] = None,
                  merge: bool = True, pad: bool = False, pad_value: float = np.nan) -> Union[np.ndarray, List[np.ndarray]]:
        """Return a copy of the features based on the provided selector."""
        pass
        # return self.x(selector=selector, sources=sources, merge=merge, pad=pad, pad_value=pad_value).copy()


    def _run_transformation(self, selector: Mapping[str, Any], transformer: TransformerMixin) -> None:
        """Applique une transformation simple - remplace les données transformées."""
        selector["partition"] = "train"
        x_train = self.x(selector, merge=False)
        x_all = self.x(selector, merge=False)
        if isinstance(x_train, list):
            for i, x_train_i in enumerate(x_train):
                print(f"Transforming source {i} with shape {x_train_i.shape}")
                transformer.fit(x_train_i)
                x_all[i] = transformer.transform(x_all[i])
        else:
            print(f"Transforming with shape {x_train.shape}")
            transformer.fit(x_train)
            x_all = transformer.transform(x_all)
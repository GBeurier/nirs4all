import numpy as np
import polars as pl
from typing import Any, Sequence, Union

class SpectraDataset:
    """
    3-table dataset: features, labels, predictions for pipeline processing.
    fold management
    """
    
    ## X => sample_id, original_sample_id (for augmentation), type (nirs, raman, mir, etc.), set (train/test), processing (hash of transformation), branch (pipeline branch), experiment_id (for reuse in cache (none when exp is finished)), group (custom cluster option), optional(feature_augmentation_id), optional(sample_augmentation_id)
    ## y => sample_id, targets, metadata
    ## results => sample_id, model, fold, seed, prediction, stack_index, branch, source, type, experiment_id
    ## folds a list of list of sample_id

## Transformation
#  take all samples X, or y, fit on train, transform on train and test, update processing with hash (transformer + old_processing)

## Sample augmentation for list of samples / or balance groups - 
#  Warning if features are already augmented.
#  take all samples from the X train set or the list of samples that are not augmented, copy, fit_transform and append to the train set. update processing with hash (transformer + old_processing), original_sample_id and sample_augmentation_id

##  Feature augmentation for list of samples
#  take all samples from the X, copy, fit on train, transform on train and test, update processing with hash (transformer + old_processing), original_sample_id and feature_augmentation_id, append to X


    DEFAULT_STATIC_INDEX = ("origin", "sample", "type", "set", "processing", "branch")

    _DEFAULT_VALUES = {
        "set": "train",
        "processing": "raw",
        "branch": 0,
        "augmentation": "raw",
    }

    def __init__(self) -> None:
        self.features: pl.DataFrame | None = None
        self.labels: pl.DataFrame | None = None
        self.results: pl.DataFrame | None = None
        self._next_row_id: int = 0
        self._next_source_id: int = 0

    def _mask(self, **filters: Any) -> pl.Expr:
        exprs = []
        for k, v in filters.items():
            if isinstance(v, (list, tuple, set, np.ndarray)):
                exprs.append(pl.col(k).is_in(v))
            else:
                exprs.append(pl.col(k) == v)
        return exprs[0] if len(exprs) == 1 else pl.all_horizontal(exprs)
    
    def _select(self, **filters: Any) -> pl.DataFrame:
        if self.features is None:
            return pl.DataFrame()
        return self.features.filter(self._mask(**filters)) if filters else self.features

    def add_spectra(
        self,
        spectra: Sequence[Sequence[float]],
        target: Any,
        **metadata: Any,
    ) -> None:
        """
        Add a group of augmented spectra for a single source.
        `spectra`: list of n spectral vectors.
        `target`: label for this source.
        metadata: static metadata keys must be scalar:
           origin, sample, type, set, processing, branch.
        augmentation: scalar or sequence of length n.
        """
        n = len(spectra)
        # Validate static metadata
        for key in self.DEFAULT_STATIC_INDEX:
            v = metadata.get(key, self._DEFAULT_VALUES.get(key))
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                raise ValueError(f"Static metadata '{key}' must be scalar, got sequence")
        # Handle augmentation
        aug = metadata.get(self.DEFAULT_AUGMENTATION_KEY, self._DEFAULT_VALUES[self.DEFAULT_AUGMENTATION_KEY])
        if isinstance(aug, Sequence) and not isinstance(aug, (str, bytes)):
            if len(aug) != n:
                raise ValueError(f"Augmentation sequence length {len(aug)} != spectra count {n}")
            aug_list = list(aug)
        else:
            aug_list = [aug] * n

        # Assign new source_id
        source_id = self._next_source_id
        self._next_source_id += 1

        # Build features rows
        rows = []
        for i, spec in enumerate(spectra):
            row_id = self._next_row_id
            self._next_row_id += 1
            rows.append({
                "row_id": row_id,
                "source_id": source_id,
                "spectrum": list(spec),
                self.DEFAULT_AUGMENTATION_KEY: aug_list[i],
            })

        new_feats = pl.DataFrame(rows)
        self.features = new_feats if self.features is None else pl.concat([self.features, new_feats], how="vertical")

        # Build labels row (one per source)
        static_row = {"source_id": source_id, "target": target}
        for key in self.DEFAULT_STATIC_INDEX:
            static_row[key] = metadata.get(key, self._DEFAULT_VALUES.get(key))
        new_label = pl.DataFrame([static_row])
        self.labels = new_label if self.labels is None else pl.concat([self.labels, new_label], how="vertical")


    def change_spectra(
        self,
        new_spectra: Sequence[Sequence[float]],
        **filters: Any,
    ) -> None:
        if self.df is None:
            return
        mask = self._mask(**filters)
        n = self.df.filter(mask).height
        if n != len(new_spectra):
            raise ValueError("new_spectra count mismatch")
        self.df = self.df.with_columns(
            pl.when(mask)
              .then(pl.Series(new_spectra, dtype=pl.List(pl.Float64)))
              .otherwise(pl.col("spectrum"))
              .alias("spectrum")
        )

    def add_tag(
        self,
        tag_name: str,
        tag_values: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """
        Ajoute une nouvelle colonne `tag_name`.
        - Si `tag_values` scalaire → broadcast sur toutes les lignes filtrées.
        - Si `tag_values` séquence de longueur m → assigné séquentiellement aux m lignes filtrées.
        """
        if self.df is None:
            raise ValueError("Dataset vide.")
        if tag_name in self.df.columns:
            raise ValueError(f"Le tag '{tag_name}' existe déjà ; utilisez set_tag.")

        df = self.df
        mask_series = df.select(self._mask(**filters).alias("_mask"))
        mask = mask_series.to_series()
        total = len(mask)
        # Construire liste complète
        full: list[Any] = [None] * total
        if isinstance(tag_values, Sequence) and not isinstance(tag_values, (str, bytes)):
            values = list(tag_values)
            if sum(mask) != len(values):
                raise ValueError("Le nombre de 'tag_values' ne correspond pas aux lignes filtrées")
            it = iter(values)
            for i, m in enumerate(mask):
                if m:
                    full[i] = next(it)
        else:
            for i, m in enumerate(mask):
                if m:
                    full[i] = tag_values
        self.df = df.with_columns(pl.Series(tag_name, full))

    def set_tag(
        self,
        tag_name: str,
        new_value: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """
        Change les valeurs d'un tag existant.
        - Si `new_value` scalaire → broadcast.
        - Si `new_value` séquence de longueur m → assigné séquentiellement aux m lignes filtrées.
        """
        if self.df is None:
            raise ValueError("Dataset vide.")
        if tag_name not in self.df.columns:
            raise KeyError(f"Tag '{tag_name}' inexistant.")

        df = self.df
        mask_series = df.select(self._mask(**filters).alias("_mask")).to_series()
        total = len(mask_series)
        full = df.get_column(tag_name).to_list()
        if isinstance(new_value, Sequence) and not isinstance(new_value, (str, bytes)):
            vals = list(new_value)
            if sum(mask_series) != len(vals):
                raise ValueError("Le nombre de 'new_value' ne correspond pas aux lignes filtrées")
            it = iter(vals)
            for i, m in enumerate(mask_series):
                if m:
                    full[i] = next(it)
        else:
            for i, m in enumerate(mask_series):
                if m:
                    full[i] = new_value
        self.df = df.with_columns(pl.Series(tag_name, full))

    def X(
        self,
        pad: bool = False,
        pad_value: float = np.nan,
        as_arrow: bool = False,
        return_ids: bool = False,
        **filters: Any,
    ) -> Union[np.ndarray, tuple[Union[np.ndarray, "pyarrow.ListArray"], np.ndarray]]:
        sub = self._select(**filters)
        spectra = sub["spectrum"].to_list()

        if as_arrow and not pad:
            import pyarrow as pa
            arr = pa.array(spectra, type=pa.list_(pa.float64()))
            ids = sub["row_id"].to_numpy() if return_ids else None
            return (arr, ids) if return_ids else arr

        if pad:
            max_len = max((len(v) for v in spectra), default=0)
            out = np.full((len(spectra), max_len), pad_value, dtype=np.float64)
            for i, vec in enumerate(spectra):
                out[i, : len(vec)] = vec
        else:
            out = np.array(spectra, dtype=object)

        if return_ids:
            ids = sub["row_id"].to_numpy()
            return out, ids
        return out

    def y(self, **filters: Any) -> np.ndarray:
        """
        Return target array for features matching filters.
        Filters on features-level and label-level columns are supported.
        """
        if self.features is None or self.labels is None:
            return np.array([])

        feat_cols = set(self.features.columns)
        label_cols = set(self.labels.columns)

        feat_filters = {k: v for k, v in filters.items() if k in feat_cols}
        label_filters = {k: v for k, v in filters.items() if k in label_cols}

        # Validate filters
        unknown = [k for k in filters if k not in feat_cols and k not in label_cols]
        if unknown:
            raise KeyError(f"Unknown filter key(s): {unknown}")

        feats = (
            self.features.filter(self._mask(**feat_filters))
            if feat_filters else self.features
        )
        labs = (
            self.labels.filter(self._mask(**label_filters))
            if label_filters else self.labels
        )

        joined = feats.join(labs, on="source_id", how="inner")
        return joined["target"].to_numpy()

    def add_prediction(self, model: str, fold: int, seed: int, preds: Sequence[float], **filters: Any) -> None:
        """
        Add predictions for features matching optional filters.
        """
        if self.features is None:
            raise ValueError("No features to predict on.")
        feats = self.features.filter(self._mask(**filters)) if filters else self.features
        row_ids = feats["row_id"].to_list()
        if len(row_ids) != len(preds):
            raise ValueError("Number of predictions does not match number of selected features.")
        rows = [
            {"row_id": rid, "model": model, "fold": fold, "seed": seed, "prediction": p}
            for rid, p in zip(row_ids, preds)
        ]
        new_preds = pl.DataFrame(rows)
        self.results = new_preds if self.results is None else pl.concat([self.results, new_preds], how="vertical")


    def to_arrow_table(self) -> "pyarrow.Table":
        return self.df.to_arrow()  # type: ignore

    def __len__(self) -> int:
        return self.df.height if self.df is not None else 0

    def __repr__(self) -> str:
        cols = self.df.columns if self.df is not None else []
        return f"SpectraDataset(n={len(self)}, cols={cols})"



### X _ courant > ori + transform
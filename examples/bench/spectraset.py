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

    _DEFAULT_INDEXES_VALUES = {
        "origin": None,
        "sample": None,
        "type": "nirs",
        "set": "train",
        "processing": None,
        "branch": 0,
        "augmentation": None,
    }
    
    DEFAULT_AUGMENTATION_KEY = "augmentation"

    def __init__(self) -> None:
        self.features: pl.DataFrame | None = None
        self.labels: pl.DataFrame | None = None
        self.results: pl.DataFrame | None = None
        self.folds: list[list[int]] = []
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
        """Select features with optional join to labels for filtering."""
        if self.features is None:
            return pl.DataFrame()
        
        if not filters:
            return self.features
        
        # Separate feature-level and label-level filters
        feat_cols = set(self.features.columns)
        label_cols = set(self.labels.columns) if self.labels is not None else set()
        
        feat_filters = {k: v for k, v in filters.items() if k in feat_cols}
        label_filters = {k: v for k, v in filters.items() if k in label_cols}
        
        # Start with features
        result = self.features
        
        # Apply feature-level filters
        if feat_filters:
            feat_mask = self._mask(**feat_filters)
            result = result.filter(feat_mask)
        
        # Apply label-level filters by joining
        if label_filters and self.labels is not None:
            label_mask = self._mask(**label_filters)
            filtered_labels = self.labels.filter(label_mask)
            result = result.join(filtered_labels, on="source_id", how="inner")
            # Keep only original feature columns
            result = result.select(self.features.columns)
        
        return result

    def add_spectra(
        self,
        spectra: Sequence[Sequence[float]],
        target: Any,
        **metadata: Any,
    ) -> None:
        ## Add spectra to the dataset
        
    def swap_spectra(
        self,
        new_spectra: Sequence[Sequence[float]],
        old_spectra_ids: Sequence[int],
        **filters: Any,
    ) -> None:
        # replace old spectra with new ones based on row_id

    def add_tag(
        self,
        tag_name: str,
        tag_values: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """Add a new tag column to features or labels table."""
        if tag_name in ["row_id", "source_id", "spectrum"]:
            raise ValueError(f"Cannot add reserved tag '{tag_name}'")
        
        # Determine target table based on tag context
        if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
            # Label-level tags
            if self.labels is None:
                raise ValueError("No labels table to add tag to")
            target_df = self.labels
            id_col = "source_id"
        else:
            # Feature-level tags
            if self.features is None:
                raise ValueError("No features table to add tag to")
            target_df = self.features
            id_col = "row_id"
        
        if tag_name in target_df.columns:
            raise ValueError(f"Tag '{tag_name}' already exists; use set_tag")
        
        # Filter and apply values
        if filters:
            if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
                # For label-level tags, filter labels table
                filtered = target_df.filter(self._mask(**filters))
            else:
                # For feature-level tags, need to join with labels for filtering
                if self.labels is not None:
                    joined = self.features.join(self.labels, on="source_id", how="inner")
                    filtered = joined.filter(self._mask(**filters))
                    filtered = filtered.select(self.features.columns)
                else:
                    filtered = self.features.filter(self._mask(**filters))
        else:
            filtered = target_df
        
        n_filtered = filtered.height
        
        # Handle tag values
        if isinstance(tag_values, Sequence) and not isinstance(tag_values, (str, bytes)):
            if len(tag_values) != n_filtered:
                raise ValueError(f"Tag values count ({len(tag_values)}) != filtered rows ({n_filtered})")
            values_list = list(tag_values)
        else:
            values_list = [tag_values] * n_filtered
        
        # Add the tag column
        if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
            # Update labels table
            target_ids = filtered["source_id"].to_list()
            for i, (target_id, value) in enumerate(zip(target_ids, values_list)):
                self.labels = self.labels.with_columns(
                    pl.when(pl.col("source_id") == target_id)
                      .then(pl.lit(value))
                      .otherwise(pl.col(tag_name) if tag_name in self.labels.columns else pl.lit(None))
                      .alias(tag_name)
                )
        else:
            # Update features table
            target_ids = filtered["row_id"].to_list()
            full_values = [None] * self.features.height
            for target_id, value in zip(target_ids, values_list):
                idx = self.features["row_id"].to_list().index(target_id)
                full_values[idx] = value
            
            self.features = self.features.with_columns(pl.Series(tag_name, full_values))

    def set_tag(
        self,
        tag_name: str,
        new_value: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """Update an existing tag."""
        # Determine target table
        if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
            target_df = self.labels
        elif self.features is not None and tag_name in self.features.columns:
            target_df = self.features
        else:
            raise KeyError(f"Tag '{tag_name}' not found")
        
        if target_df is None:
            raise ValueError("Target table is None")
        
        # Filter and update
        if filters:
            if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
                filtered = target_df.filter(self._mask(**filters))
                target_ids = filtered["source_id"].to_list()
                id_col = "source_id"
            else:
                if self.labels is not None:
                    joined = self.features.join(self.labels, on="source_id", how="inner")
                    filtered = joined.filter(self._mask(**filters))
                    target_ids = filtered["row_id"].to_list()
                    id_col = "row_id"
                else:
                    filtered = self.features.filter(self._mask(**filters))
                    target_ids = filtered["row_id"].to_list()
                    id_col = "row_id"
        else:
            if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
                target_ids = target_df["source_id"].to_list()
                id_col = "source_id"
            else:
                target_ids = target_df["row_id"].to_list()
                id_col = "row_id"
        
        # Handle new values
        if isinstance(new_value, Sequence) and not isinstance(new_value, (str, bytes)):
            if len(new_value) != len(target_ids):
                raise ValueError(f"New values count ({len(new_value)}) != filtered rows ({len(target_ids)})")
            values_list = list(new_value)
        else:
            values_list = [new_value] * len(target_ids)
        
        # Update the appropriate table
        if tag_name in ["target", "origin", "sample", "type", "set", "processing", "branch"]:
            for target_id, value in zip(target_ids, values_list):
                self.labels = self.labels.with_columns(
                    pl.when(pl.col("source_id") == target_id)
                      .then(pl.lit(value))
                      .otherwise(pl.col(tag_name))
                      .alias(tag_name)
                )
        else:
            for target_id, value in zip(target_ids, values_list):
                self.features = self.features.with_columns(
                    pl.when(pl.col("row_id") == target_id)
                      .then(pl.lit(value))
                      .otherwise(pl.col(tag_name))
                      .alias(tag_name)
                )

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

    def to_arrow_table(self):
        if self.features is not None:
            return self.features.to_arrow()
        return None

    def __len__(self) -> int:
        return self.features.height if self.features is not None else 0

    def __repr__(self) -> str:
        cols = self.features.columns if self.features is not None else []
        return f"SpectraDataset(n={len(self)}, cols={cols})"

    def get_fold_data(self, fold: int, set_type: str = "train"):
        """Get data for a specific fold and set type."""
        return self._select(fold=fold, set=set_type)

    def get_unique_values(self, column: str, table: str = "features"):
        """Get unique values from a column in features or labels table."""
        if table == "features" and self.features is not None:
            if column in self.features.columns:
                return self.features[column].unique().to_list()
        elif table == "labels" and self.labels is not None:
            if column in self.labels.columns:
                return self.labels[column].unique().to_list()
        return []

    def get_predictions(self, **filters):
        """Get predictions matching filters."""
        if self.results is None:
            return pl.DataFrame()
        return self.results.filter(self._mask(**filters)) if filters else self.results


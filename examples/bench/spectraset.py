import numpy as np
import polars as pl
import hashlib
from typing import Any, Sequence, Mapping, Literal
from datetime import datetime
import json

class SpectraDataset:
    """
    Light-weight, in-memory spectroscopic data container for ML pipelines.
    
    Manages four logically separate artifacts:
    - Features (X): spectra with canonical multi-index
    - Labels & metadata (Y): sample-level information
    - Predictions: model outputs keyed by row_id
    - Folds: train/validation splits
    """
    
    def __init__(self,
                 features: pl.DataFrame | None = None,
                 labels: pl.DataFrame | None = None,
                 folds: list[list[int]] | pl.DataFrame | None = None,
                 *,
                 float64: bool = True,
                 validate: bool = True) -> None:
        """Initialize SpectraDataset with optional initial data."""
        self.features = features
        self.labels = labels
        self.results: pl.DataFrame | None = None
        self.folds = folds or []
        
        # Internal counters
        self._next_row_id = self._compute_next_row_id()
        self._next_sample_id = self._compute_next_sample_id()
        self._autovalidate = True
        
        self.spectrum_dtype = pl.Float64 if float64 else pl.Float32
        
        # self._features_dtypes = {
        #     "row_id":           pl.Int64,
        #     "sample_id":        pl.Int64,
        #     "origin_id":        pl.Int64,
        #     "type":             pl.Utf8,
        #     "transformation_id":pl.Utf8,
        #     "set_id":           pl.Utf8,
        #     "branch":           pl.Int32,
        #     "group":            pl.Int32,
        #     "spectrum":         self.spectrum_dtype,
        # }
        # self._labels_dtypes = {
        #     "sample_id": pl.Int64,
        #     "target":    pl.Float64,    # or Any-typed if you mix types
        #     # … any other metadata columns you want to type …
        # }
        # self._results_dtypes = {
        #     "row_id":       pl.Int64,
        #     "model_id":     pl.Utf8,
        #     "fold_id":      pl.Int32,
        #     "seed":         pl.Int32,
        #     "stack_index":  pl.Int32,
        #     "branch":       pl.Int32,
        #     "prediction":   pl.Float64,
        #     "predicted_at": pl.Datetime,
        #     # + any extra_cols you record
        # }
        
        # if validate:
            # self.validate()
            
            
    def _compute_next(self, col: str, df: pl.DataFrame | None) -> int:
        if df is None or df.height == 0:
            return 0
        
        return df[col].max() + 1
    
    def _compute_next_row_id(self) -> int:
        return self._compute_next('row_id', self.features)
    
    def _compute_next_sample_id(self) -> int:
        return self._compute_next('sample_id', self.labels)
    
    # def validate(self) -> None:
    #     """Validate dataset invariants and constraints."""
    #     if self.features is not None:
    #         # Check required columns exist
    #         required_cols = ['row_id', 'sample_id', 'origin_id', 'type', 
    #                        'transformation_id', 'set_id', 'branch', 'group', 'spectrum']
    #         missing = set(required_cols) - set(self.features.columns)
    #         if missing:
    #             raise ValueError(f"Features table missing required columns: {missing}")
            
    #         # Check uniqueness constraint
    #         multi_index_cols = ['sample_id', 'origin_id', 'type', 'transformation_id', 'branch']
    #         duplicates = self.features.group_by(multi_index_cols).agg(pl.len().alias('count')).filter(pl.col('count') > 1)
    #         if duplicates.height > 0:
    #             raise ValueError("Features table violates uniqueness constraint on multi-index")
            
    #         # Check row_id uniqueness
    #         if self.features['row_id'].n_unique() != self.features.height:
    #             raise ValueError("row_id values must be unique")
        
    #     if self.labels is not None:
    #         # Check sample_id uniqueness
    #         if self.labels['sample_id'].n_unique() != self.labels.height:
    #             raise ValueError("sample_id values must be unique in labels table")
        
    #     if self.results is not None and self.features is not None:
    #         # Check foreign key constraint
    #         valid_row_ids = set(self.features['row_id'].to_list())
    #         result_row_ids = set(self.results['row_id'].to_list())
    #         invalid = result_row_ids - valid_row_ids
    #         if invalid:
    #             raise ValueError(f"Results table contains invalid row_ids: {invalid}")
    
    def add_samples(self,
                    spectra: np.ndarray,
                    *,
                    type: str = "nirs",
                    set_id: str = "train",
                    branch: int = 0,
                    group: str | int | None = None,
                    targets: Sequence[Any] | None = None,
                    metadata: Mapping[str, Sequence[Any]] | None = None) -> list[int]:
        """Add new samples to the dataset."""
        n_samples = len(spectra)
        
        sample_ids = [self._next_sample_id + i for i in range(n_samples)]
        self._next_sample_id += n_samples
        
        row_ids = [self._next_row_id + i for i in range(n_samples)]
        self._next_row_id += n_samples
        
        # Create features rows
        features_data = {
            'row_id': row_ids,
            'sample_id': sample_ids,
            'origin_id': [None] * n_samples,
            'type': [type] * n_samples,
            'transformation_id': [None] * n_samples,
            'set_id': [set_id] * n_samples,
            'branch': [branch] * n_samples,
            'group': [group] * n_samples,
            'spectrum': spectra
        }
        
        new_features = pl.DataFrame(features_data)
        
        # Add to features table
        if self.features is None:
            self.features = new_features
        else:
            self.features = pl.concat([self.features, new_features], how="vertical")
        
        # Handle labels and metadata
        if targets is not None or metadata is not None:
            labels_data = {'sample_id': sample_ids}
            
            if targets is not None:
                if len(targets) != n_samples:
                    raise ValueError("Length of targets must match length of spectra")
                labels_data['target'] = list(targets)
            
            if metadata is not None:
                for key, values in metadata.items():
                    if len(values) != n_samples:
                        raise ValueError(f"Length of metadata[{key}] must match length of spectra")
                    labels_data[key] = list(values)
            
            new_labels = pl.DataFrame(labels_data)
            
            if self.labels is None:
                self.labels = new_labels
            else:
                # Update existing labels or add new ones
                existing_sample_ids = set(self.labels['sample_id'].to_list())
                new_sample_ids = set(sample_ids)
                
                # Split into updates and inserts
                to_update = new_sample_ids & existing_sample_ids
                to_insert = new_sample_ids - existing_sample_ids
                
                if to_insert:
                    insert_mask = pl.col('sample_id').is_in(list(to_insert))
                    insert_labels = new_labels.filter(insert_mask)
                    self.labels = pl.concat([self.labels, insert_labels], how="vertical")
                
                # For updates, we'd need more complex logic - for now, raise error
                if to_update:
                    raise ValueError(f"Sample IDs already exist in labels: {to_update}")
        
        # self._enforce_schema()
        
        # if self._autovalidate:
            # self.validate()
        
        return row_ids
    
    def fork_branch(self,
                    new_branch: pl.Int64 | None = None,
                    source_branch: pl.Int64 = 0,
                    copy_predictions: bool = False) -> int:
        """Fork a branch by copying all rows from source_branch."""
        if self.features is None:
            raise ValueError("No features to fork")
        
        # Determine new branch number
        if new_branch is None:
            new_branch = self._compute_next('branch', self.features)
        
        # Filter source branch
        source_rows = self.features.filter(pl.col('branch') == source_branch)
        if source_rows.height == 0:
            raise ValueError(f"No rows found in source branch {source_branch}")
        
        # Create new row_ids
        n_rows = source_rows.height
        new_row_ids = [self._next_row_id + i for i in range(n_rows)]
        self._next_row_id += n_rows
        
        print(f"new_row_ids type: {type(new_row_ids[0])}")
        # Copy and modify branch
        forked_rows = source_rows.with_columns([
            pl.Series('row_id', new_row_ids),
            pl.lit(new_branch, dtype=pl.Int64).alias('branch')
        ])
        
        # Deep copy spectra
        spectra_copy = []
        for spectrum in forked_rows['spectrum'].to_list():
            spectra_copy.append(spectrum.copy())
        
        forked_rows = forked_rows.with_columns(
            pl.Series('spectrum', spectra_copy)
        )
        
        # Add to features
        print(">>>>", forked_rows)
        print(self.features)
        self.features = pl.concat([self.features, forked_rows], how="vertical")
        
        # Copy predictions if requested
        if copy_predictions and self.results is not None:
            old_row_ids = source_rows['row_id'].to_list()
            predictions_to_copy = self.results.filter(pl.col('row_id').is_in(old_row_ids))
            
            if predictions_to_copy.height > 0:
                # Map old row_ids to new ones
                row_id_mapping = dict(zip(old_row_ids, new_row_ids))
                
                copied_predictions = predictions_to_copy.with_columns([
                    pl.col('row_id').map_elements(lambda x: row_id_mapping[x], return_dtype=pl.Int64),
                    pl.lit(new_branch).alias('branch')
                ])
                
                self.results = pl.concat([self.results, copied_predictions], how="vertical")
        
        self._enforce_schema()
        
        if self._autovalidate:
            self.validate()
        
        return new_branch
    
    def augment_samples(self,
                        selector,
                        transformer,
                        *,
                        new_type: str | None = None,
                        new_group: str | int | None = None) -> list[int]:
        """Augment samples by creating new samples with new sample_ids."""
        if self.features is None:
            raise ValueError("No features to augment")
        
        # Select rows to augment
        if selector:
            selected_rows = self._select(**selector)
        else:
            selected_rows = self.features
        
        if selected_rows.height == 0:
            return []
        
        n_new = selected_rows.height
        new_row_ids = [self._next_row_id + i for i in range(n_new)]
        self._next_row_id += n_new
        
        # Generate new sample_ids
        new_sample_ids = [self._next_sample_id + i for i in range(n_new)]
        self._next_sample_id += n_new
        
        # Transform spectra
        augmented_spectra = []
        for spectrum in selected_rows['spectrum'].to_list():
            spectrum_array = np.array(spectrum)
            augmented_array = transformer(spectrum_array)
            augmented_spectra.append(augmented_array.tolist())
        
        # Compute transformation hash
        old_transformation_ids = selected_rows['transformation_id'].to_list()
        new_transformation_ids = [
            self.compute_hash(old_id, transformer) 
            for old_id in old_transformation_ids
        ]
        
        # Create augmented rows
        augmented_data = {
            'row_id': new_row_ids,
            'sample_id': new_sample_ids,
            'origin_id': selected_rows['sample_id'].to_list(),  # Parent sample_ids
            'type': [new_type or t for t in selected_rows['type'].to_list()],
            'transformation_id': new_transformation_ids,
            'set_id': selected_rows['set_id'].to_list(),
            'branch': selected_rows['branch'].to_list(),
            'group': [new_group if new_group is not None else g for g in selected_rows['group'].to_list()],
            'spectrum': augmented_spectra
        }
        
        augmented_features = pl.DataFrame(augmented_data)
        self.features = pl.concat([self.features, augmented_features], how="vertical")
        
        if self._autovalidate:
            self.validate()
        
        return new_row_ids
    
    def augment_features(self, selector, transformer) -> list[int]:
        """Augment features preserving sample_ids."""
        if self.features is None:
            raise ValueError("No features to augment")
        
        # Select rows to augment
        if selector:
            selected_rows = self._select(**selector)
        else:
            selected_rows = self.features
        
        if selected_rows.height == 0:
            return []
        
        n_new = selected_rows.height
        new_row_ids = [self._next_row_id + i for i in range(n_new)]
        self._next_row_id += n_new
        
        # Transform spectra
        augmented_spectra = []
        for spectrum in selected_rows['spectrum'].to_list():
            spectrum_array = np.array(spectrum)
            augmented_array = transformer(spectrum_array)
            augmented_spectra.append(augmented_array.tolist())
        
        # Compute transformation hash
        old_transformation_ids = selected_rows['transformation_id'].to_list()
        new_transformation_ids = [
            self.compute_hash(old_id, transformer) 
            for old_id in old_transformation_ids
        ]
        
        # Create augmented rows (preserving sample_id)
        augmented_data = {
            'row_id': new_row_ids,
            'sample_id': selected_rows['sample_id'].to_list(),  # Preserve sample_ids
            'origin_id': selected_rows['origin_id'].to_list(),
            'type': selected_rows['type'].to_list(),
            'transformation_id': new_transformation_ids,
            'set_id': selected_rows['set_id'].to_list(),
            'branch': selected_rows['branch'].to_list(),
            'group': selected_rows['group'].to_list(),
            'spectrum': augmented_spectra
        }
        
        augmented_features = pl.DataFrame(augmented_data)
        
        # Check uniqueness constraint before adding
        test_features = pl.concat([self.features, augmented_features], how="vertical")
        multi_index_cols = ['sample_id', 'origin_id', 'type', 'transformation_id', 'branch']
        duplicates = test_features.group_by(multi_index_cols).agg(pl.len().alias('count')).filter(pl.col('count') > 1)
        if duplicates.height > 0:
            raise ValueError("Feature augmentation would violate uniqueness constraint")
        
        self.features = test_features
        
        if self._autovalidate:
            self.validate()
        
        return new_row_ids
    
    @staticmethod
    def compute_hash(prev_hash: str | None, transformer: Any) -> str:
        """Compute deterministic hash for transformation pipeline."""
        # Start with previous hash or "RAW"
        base = prev_hash or "RAW"
        
        # Get transformer info
        transformer_name = f"{transformer.__class__.__module__}.{transformer.__class__.__name__}"
        
        # Try to get transformer parameters
        try:
            if hasattr(transformer, 'get_params'):
                params = transformer.get_params()
            elif hasattr(transformer, '__dict__'):
                params = {k: v for k, v in transformer.__dict__.items() 
                         if not k.startswith('_') and not callable(v)}
            else:
                params = {}
            
            params_str = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            params_str = repr(transformer)
        
        # Create hash input
        hash_input = f"{base}|{transformer_name}|{params_str}"
        
        # Return 128-bit hex hash
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_tag(self, name: str, default: Any = None, *, table: str = "features") -> None:
        """Add a new tag column with default value."""
        reserved_names = {
            'row_id', 'sample_id', 'origin_id', 'type', 'transformation_id', 
            'set_id', 'branch', 'group', 'model_id', 'fold_id', 'seed', 
            'stack_index', 'predicted_at', 'spectrum', 'target'
        }
        
        if name in reserved_names:
            raise ValueError(f"Cannot add reserved tag '{name}'")
        
        if table == "features":
            if self.features is None:
                raise ValueError("No features table to add tag to")
            if name in self.features.columns:
                raise ValueError(f"Tag '{name}' already exists in features table")
            
            self.features = self.features.with_columns(pl.lit(default).alias(name))
            
        elif table == "labels":
            if self.labels is None:
                raise ValueError("No labels table to add tag to")
            if name in self.labels.columns:
                raise ValueError(f"Tag '{name}' already exists in labels table")
            
            self.labels = self.labels.with_columns(pl.lit(default).alias(name))
            
        else:
            raise ValueError("table must be 'features' or 'labels'")
    
    def set_tag(self, name: str, value: Any | Sequence[Any], 
                selector: Mapping[str, Any] | None = None, *, table: str = "features") -> None:
        """Update an existing tag for rows matching selector."""
        if table == "features":
            if self.features is None or name not in self.features.columns:
                raise KeyError(f"Tag '{name}' not found in features table")
            
            target_df = self.features
            if selector:
                filtered_df = self._select(**selector)
                target_row_ids = filtered_df['row_id'].to_list()
            else:
                target_row_ids = target_df['row_id'].to_list()
            
            # Handle value assignment
            if isinstance(value, (list, tuple)) and not isinstance(value, str):
                if len(value) != len(target_row_ids):
                    raise ValueError("Length of values must match number of selected rows")
                value_list = list(value)
            else:
                value_list = [value] * len(target_row_ids)
            
            # Update values
            for row_id, new_val in zip(target_row_ids, value_list):
                self.features = self.features.with_columns(
                    pl.when(pl.col('row_id') == row_id)
                      .then(pl.lit(new_val))
                      .otherwise(pl.col(name))
                      .alias(name)
                )
        
        elif table == "labels":
            if self.labels is None or name not in self.labels.columns:
                raise KeyError(f"Tag '{name}' not found in labels table")
            
            if selector:
                # For labels, we need to handle cross-table filtering
                filtered_sample_ids = self._get_sample_ids_for_selector(selector)
            else:
                filtered_sample_ids = self.labels['sample_id'].to_list()
            
            # Handle value assignment
            if isinstance(value, (list, tuple)) and not isinstance(value, str):
                if len(value) != len(filtered_sample_ids):
                    raise ValueError("Length of values must match number of selected samples")
                value_list = list(value)
            else:
                value_list = [value] * len(filtered_sample_ids)
            
            # Update values
            for sample_id, new_val in zip(filtered_sample_ids, value_list):
                self.labels = self.labels.with_columns(
                    pl.when(pl.col('sample_id') == sample_id)
                      .then(pl.lit(new_val))
                      .otherwise(pl.col(name))
                      .alias(name)
                )
        
        else:
            raise ValueError("table must be 'features' or 'labels'")
    
    def _select(self, **selector) -> pl.DataFrame:
        """Select features with optional join to labels for filtering."""
        if self.features is None:
            return pl.DataFrame()
        
        if not selector:
            return self.features
        
        # Separate feature-level and label-level filters
        feat_cols = set(self.features.columns)
        label_cols = set(self.labels.columns) if self.labels is not None else set()
        
        feat_filters = {k: v for k, v in selector.items() if k in feat_cols}
        label_filters = {k: v for k, v in selector.items() if k in label_cols}
        
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
            result = result.join(filtered_labels.select(['sample_id']), on="sample_id", how="inner")
        
        return result
    
    def _get_sample_ids_for_selector(self, selector: Mapping[str, Any]) -> list:
        """Get sample_ids that match the selector (handles cross-table filtering)."""
        if self.features is None:
            return []
        
        filtered_features = self._select(**selector)
        return filtered_features['sample_id'].unique().to_list()
    
    def _mask(self, **filters) -> pl.Expr:
        """Create boolean mask from filters."""
        if not filters:
            return pl.lit(True)
        
        exprs = []
        for k, v in filters.items():  
            if isinstance(v, (list, tuple, set, np.ndarray)):
                exprs.append(pl.col(k).is_in(list(v)))
            else:
                exprs.append(pl.col(k) == v)
        
        return exprs[0] if len(exprs) == 1 else pl.all_horizontal(exprs)
    
    def select(self, selector: Mapping[str, Any] | None = None, *, 
               columns: Sequence[str] | None = None) -> pl.DataFrame:
        """Select and return features with optional column filtering."""
        result = self._select(**(selector or {}))
        
        if columns is not None:
            available_cols = [col for col in columns if col in result.columns]
            result = result.select(available_cols)
        
        return result
    
    def X(self, selector: Mapping[str, Any] | None = None, *, 
          pad: bool = False, pad_value: float = np.nan) -> np.ndarray:
        """Return feature matrix (spectra) as numpy array."""
        filtered = self._select(**(selector or {}))
        spectra = filtered['spectrum'].to_list()
        
        if not spectra:
            return np.array([])
        
        if pad:
            max_len = max(len(spec) for spec in spectra)
            result = np.full((len(spectra), max_len), pad_value, dtype=np.float64)
            for i, spec in enumerate(spectra):
                result[i, :len(spec)] = spec
            return result
        else:
            return np.array(spectra, dtype=object)
    
    def Y(self, selector: Mapping[str, Any] | None = None) -> np.ndarray:
        """Return target array for selected features."""
        if self.features is None or self.labels is None:
            return np.array([])
        
        filtered_features = self._select(**(selector or {}))
        if filtered_features.height == 0:
            return np.array([])
        
        # Join with labels to get targets
        with_labels = filtered_features.join(self.labels, on='sample_id', how='inner')
        
        if 'target' not in with_labels.columns:
            raise ValueError("No 'target' column found in labels")
        
        return with_labels['target'].to_numpy()
    
    def add_predictions(self,
                        model_id: str,
                        fold_id: int,
                        seed: int,
                        preds: Sequence[float],
                        *,
                        selector: Mapping[str, Any] | None = None,
                        stack_index: int = 0,
                        extra_cols: Mapping[str, Sequence[Any]] | None = None) -> None:
        """Add predictions for selected features."""
        if self.features is None:
            raise ValueError("No features to add predictions for")
        
        filtered = self._select(**(selector or {}))
        if filtered.height == 0:
            raise ValueError("No features match selector")
        
        if len(preds) != filtered.height:
            raise ValueError("Number of predictions must match number of selected features")
        
        # Check for duplicate predictions
        row_ids = filtered['row_id'].to_list()
        if self.results is not None:
            existing_key = (
                pl.col('row_id').is_in(row_ids) &
                (pl.col('model_id') == model_id) &
                (pl.col('fold_id') == fold_id) &
                (pl.col('seed') == seed) &
                (pl.col('stack_index') == stack_index)
            )
            duplicates = self.results.filter(existing_key)
            if duplicates.height > 0:
                raise ValueError("Duplicate predictions detected")
        
        # Create predictions data
        pred_data = {
            'row_id': row_ids,
            'model_id': [model_id] * len(row_ids),
            'fold_id': [fold_id] * len(row_ids),
            'seed': [seed] * len(row_ids),
            'stack_index': [stack_index] * len(row_ids),
            'branch': filtered['branch'].to_list(),
            'prediction': list(preds),
            'predicted_at': [datetime.now()] * len(row_ids)
        }
        
        # Add extra columns
        if extra_cols:
            for col_name, col_values in extra_cols.items():
                if len(col_values) != len(row_ids):
                    raise ValueError(f"Length of extra_cols[{col_name}] must match number of predictions")
                pred_data[col_name] = list(col_values)
        
        new_results = pl.DataFrame(pred_data)
        
        if self.results is None:
            self.results = new_results
        else:
            self.results = pl.concat([self.results, new_results], how="vertical")
            
        self._enforce_schema()
    
    def get_predictions(self, selector: Mapping[str, Any] | None = None) -> pl.DataFrame:
        """Get predictions matching selector."""
        if self.results is None:
            return pl.DataFrame()
        
        if selector is None:
            return self.results
        
        return self.results.filter(self._mask(**selector))
    
    def set_folds(self, splitter: Any, *, n_splits: int, random_state: int | None = None) -> None:
        """Set cross-validation folds using scikit-learn splitter."""
        if self.labels is None:
            raise ValueError("No labels available for fold splitting")
        
        sample_ids = self.labels['sample_id'].to_list()
        targets = self.labels['target'].to_list() if 'target' in self.labels.columns else None
        
        # Configure splitter
        if hasattr(splitter, 'n_splits'):
            splitter.n_splits = n_splits
        if hasattr(splitter, 'random_state') and random_state is not None:
            splitter.random_state = random_state
        
        # Generate splits
        if targets is not None:
            splits = list(splitter.split(sample_ids, targets))
        else:
            splits = list(splitter.split(sample_ids))
        
        # Convert to list of validation sample_ids
        self.folds = []
        for train_idx, val_idx in splits:
            val_sample_ids = [sample_ids[i] for i in val_idx]
            self.folds.append(val_sample_ids)
    
    def get_fold(self, fold_id: int, *, split: str = "train") -> pl.DataFrame:
        """Get features for a specific fold and split."""
        if not self.folds or fold_id >= len(self.folds):
            raise ValueError(f"Invalid fold_id {fold_id}")
        
        val_sample_ids = self.folds[fold_id]
        
        if split == "train":
            # Return samples NOT in validation set
            return self._select().filter(~pl.col('sample_id').is_in(val_sample_ids))
        elif split == "val" or split == "validation":
            # Return samples in validation set
            return self._select().filter(pl.col('sample_id').is_in(val_sample_ids))
        else:
            raise ValueError("split must be 'train' or 'val'/'validation'")
    
    def save(self, path: str, *, fmt: Literal["arrow", "parquet"] = "arrow") -> None:
        """Save dataset to disk."""
        import os
        
        base_path = path.rstrip('/')
        os.makedirs(base_path, exist_ok=True)
        
        if fmt == "arrow":
            if self.features is not None:
                self.features.write_ipc(f"{base_path}/features.arrow")
            if self.labels is not None:
                self.labels.write_ipc(f"{base_path}/labels.arrow")
            if self.results is not None:
                self.results.write_ipc(f"{base_path}/results.arrow")
        elif fmt == "parquet":
            if self.features is not None:
                self.features.write_parquet(f"{base_path}/features.parquet")
            if self.labels is not None:
                self.labels.write_parquet(f"{base_path}/labels.parquet")
            if self.results is not None:
                self.results.write_parquet(f"{base_path}/results.parquet")
        else:
            raise ValueError("fmt must be 'arrow' or 'parquet'")
        
        # Save folds as JSON
        folds_data = {
            'folds': self.folds,
            'next_row_id': self._next_row_id,
            'next_sample_id': self._next_sample_id
        }
        import json
        with open(f"{base_path}/folds.json", 'w') as f:
            json.dump(folds_data, f, default=str)
    
    @classmethod
    def load(cls, path: str, *, fmt: Literal["arrow", "parquet"] = "arrow") -> "SpectraDataset":
        """Load dataset from disk."""
        import os
        import json
        
        base_path = path.rstrip('/')
        
        # Load tables
        features = None
        labels = None
        results = None
        
        if fmt == "arrow":
            if os.path.exists(f"{base_path}/features.arrow"):
                features = pl.read_ipc(f"{base_path}/features.arrow")
            if os.path.exists(f"{base_path}/labels.arrow"):
                labels = pl.read_ipc(f"{base_path}/labels.arrow")
            if os.path.exists(f"{base_path}/results.arrow"):
                results = pl.read_ipc(f"{base_path}/results.arrow")
        elif fmt == "parquet":
            if os.path.exists(f"{base_path}/features.parquet"):
                features = pl.read_parquet(f"{base_path}/features.parquet")
            if os.path.exists(f"{base_path}/labels.parquet"):
                labels = pl.read_parquet(f"{base_path}/labels.parquet")
            if os.path.exists(f"{base_path}/results.parquet"):
                results = pl.read_parquet(f"{base_path}/results.parquet")
        else:
            raise ValueError("fmt must be 'arrow' or 'parquet'")
        
        # Create dataset
        dataset = cls(features=features, labels=labels, validate=False)
        dataset.results = results
        
        # Load folds and metadata
        if os.path.exists(f"{base_path}/folds.json"):
            with open(f"{base_path}/folds.json", 'r') as f:
                folds_data = json.load(f)
            dataset.folds = folds_data.get('folds', [])
            dataset._next_row_id = folds_data.get('next_row_id', dataset._compute_next_row_id())
            dataset._next_sample_id = folds_data.get('next_sample_id', dataset._compute_next_sample_id())
        
        dataset.validate()
        return dataset
    
    def __len__(self) -> int:
        """Return number of feature rows."""
        return self.features.height if self.features is not None else 0
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        n_rows = len(self)
        n_samples = self.labels.height if self.labels is not None else 0
        n_branches = self.features['branch'].n_unique() if self.features is not None else 0
        
        return f"SpectraDataset(n_rows={n_rows}, n_samples={n_samples}, n_branches={n_branches})"

    def _enforce_schema(self) -> None:
        """Cast features, labels & results to the canonical schema in one go."""
        # — FEATURES —
        if self.features is not None:
            casts: list[pl.Expr] = []
            for col, target_dtype in self._features_dtypes.items():
                if col not in self.features.columns:
                    continue
                # detect a List type
                if isinstance(target_dtype, PlList):
                    inner = target_dtype.inner
                    casts.append(
                        pl.col(col)
                          .list
                          .cast(inner)
                          .alias(col)
                    )
                else:
                    casts.append(
                        pl.col(col)
                          .cast(target_dtype)
                          .alias(col)
                    )
            if casts:
                self.features = self.features.with_columns(casts)

        # — LABELS —
        if self.labels is not None:
            casts = [
                pl.col(col).cast(dt).alias(col)
                for col, dt in self._labels_dtypes.items()
                if col in self.labels.columns
            ]
            if casts:
                self.labels = self.labels.with_columns(casts)

        # — RESULTS —
        if self.results is not None:
            casts = [
                pl.col(col).cast(dt).alias(col)
                for col, dt in self._results_dtypes.items()
                if col in self.results.columns
            ]
            if casts:
                self.results = self.results.with_columns(casts)

            
            

# Example usage and testing functions
def _test_basic_functionality():
    """Basic test of SpectraDataset functionality."""
    # Create sample data
    spectra = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0]
    ]
    targets = [10.5, 20.3, 15.7]
    metadata = {'instrument': ['A', 'B', 'A']}
    
    # Create dataset
    dataset = SpectraDataset()
    
    # Add samples
    row_ids = dataset.add_samples(
        spectra=spectra,
        targets=targets,
        metadata=metadata,
        type="nirs"
    )
    
    print(f"Added {len(row_ids)} samples")
    print(f"Dataset: {dataset}")
    
    # Test X and Y methods
    X = dataset.X()
    Y = dataset.Y()
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Test selection
    nirs_data = dataset.select({'type': 'nirs'})
    print(f"NIRS data: {nirs_data.height} rows")
    
    # Test fork
    new_branch = dataset.fork_branch()
    print(f"Forked to branch {new_branch}")
    print(f"Dataset after fork: {dataset}")
    
    # Test augmentation
    def dummy_transformer(x):
        return x * 1.1  # Simple scaling
    
    aug_rows = dataset.augment_samples(
        selector={'branch': 0},
        transformer=dummy_transformer
    )
    print(f"Augmented {len(aug_rows)} samples")
    print(f"Dataset after augmentation: {dataset}")
    
    return dataset

if __name__ == "__main__":
    # Run basic test
    test_dataset = _test_basic_functionality()
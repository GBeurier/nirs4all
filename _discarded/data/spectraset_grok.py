import numpy as np
import xarray as xr
from typing import Optional, Dict, List, Sequence, Any, Union, Literal, Tuple, Hashable
from sklearn.preprocessing import LabelEncoder

def _ensure_1d_float(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim == 0:
        return arr[None]
    elif arr.ndim > 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D array.")
    return arr

class SpectraSetCore:
    def __init__(self, ds: Optional[xr.Dataset] = None):
        if ds is None:
            ds = xr.Dataset(coords={"obs": np.array([], dtype=int)})
        self.ds = ds
        self._next_obs_id = int(self.ds.sizes.get("obs", 0))

    def build(self, spectra: Dict[str, List[List[np.ndarray]]], target: Optional[np.ndarray] = None, metadata: Optional[Dict[str, np.ndarray]] = None, sample_ids: Optional[List[Any]] = None, augmentation_ids: Optional[List[Any]] = None, feature_names: Optional[Dict[str, List[str]]] = None, target_names: Optional[List[str]] = None, splits: Optional[np.ndarray] = None, groups: Optional[Dict[str, np.ndarray]] = None, fold_ids: Optional[np.ndarray] = None, branches: Optional[np.ndarray] = None):
        n_samples, total_obs, final_sample_ids, sample_id_to_index, sample_aug_counts, obs_indices_list, final_augmentation_ids, obs_to_sample_id_list = self._prepare_ids_and_counts(spectra, sample_ids, augmentation_ids)
        coords = {"obs": obs_indices_list, "sample": ("obs", obs_to_sample_id_list), "augmentation": ("obs", final_augmentation_ids)}
        data_vars = self._build_spectra_data_vars(spectra, obs_indices_list, feature_names)
        target_da = self._build_target_data_var(target, n_samples, obs_indices_list, obs_to_sample_id_list, sample_id_to_index, target_names)
        if target_da is not None:
            data_vars["target"] = target_da
        data_vars.update(self._build_metadata_data_vars(metadata, n_samples, obs_indices_list, obs_to_sample_id_list, sample_id_to_index))
        self.ds = xr.Dataset(data_vars, coords=coords)
        self._add_sample_level_coords(self.ds, n_samples, obs_to_sample_id_list, sample_id_to_index, splits, groups, fold_ids, branches)
        self._next_obs_id = int(self.ds.sizes["obs"])

    def append(self, other_ds: xr.Dataset):
        if self.ds.sizes.get("obs", 0) == 0:
            self.ds = other_ds
        else:
            for var in other_ds.data_vars:
                if var.startswith("spectra_"):
                    if var not in self.ds.data_vars:
                        raise ValueError(f"New dataset contains unknown source '{var}'.")
                    feat_dim_other = other_ds[var].dims[1]
                    feat_dim_self = self.ds[var].dims[1]
                    if feat_dim_other != feat_dim_self:
                        raise ValueError(f"Feature dimension mismatch for '{var}' (expected '{feat_dim_self}', got '{feat_dim_other}').")
                    if not np.array_equal(self.ds[var].coords[feat_dim_self].values, other_ds[var].coords[feat_dim_other].values):
                        raise ValueError(f"Feature names mismatch for '{var}'.")
            self.ds = xr.concat([self.ds, other_ds], dim="obs")
        self._next_obs_id = int(self.ds.sizes["obs"])

    def _generate_next_augmentation_ids(self, n_new: int) -> np.ndarray:
        current_max = int(self.ds.coords["augmentation"].max().item()) if "augmentation" in self.ds.coords and self.ds.sizes["obs"] > 0 else -1
        return np.arange(current_max + 1, current_max + 1 + n_new, dtype=int)

    @staticmethod
    def _prepare_ids_and_counts(spectra, sample_ids_input, augmentation_ids_input):
        if not spectra:
            raise ValueError("'spectra' must contain at least one source.")
        first_src_name = next(iter(spectra))
        n_samples = len(spectra[first_src_name])
        final_sample_ids = list(range(n_samples)) if sample_ids_input is None else sample_ids_input
        if len(final_sample_ids) != n_samples:
            raise ValueError("sample_ids length mismatch with samples.")
        sample_id_to_index = {sid: i for i, sid in enumerate(final_sample_ids)}
        sample_aug_counts = [len(s_augs) for s_augs in spectra[first_src_name]]
        for src_name, sample_lists in spectra.items():
            if len(sample_lists) != n_samples:
                raise ValueError(f"Source '{src_name}' has {len(sample_lists)} samples, but '{first_src_name}' has {n_samples}.")
            current_aug_counts = [len(s_augs) for s_augs in sample_lists]
            if current_aug_counts != sample_aug_counts:
                raise ValueError(f"Augmentation counts for source '{src_name}' are not consistent across samples with other sources.")
        total_obs = sum(sample_aug_counts)
        obs_indices_list = np.arange(total_obs).tolist()
        obs_to_sample_id_list = [final_sample_ids[i] for i, count in enumerate(sample_aug_counts) for _ in range(count)]
        final_augmentation_ids = [j for count in sample_aug_counts for j in range(count)] if augmentation_ids_input is None else augmentation_ids_input
        if len(final_augmentation_ids) != total_obs:
            raise ValueError("augmentation_ids length must equal total observations.")
        return n_samples, total_obs, final_sample_ids, sample_id_to_index, sample_aug_counts, obs_indices_list, final_augmentation_ids, obs_to_sample_id_list

    @staticmethod
    def _build_spectra_data_vars(spectra, obs_ids, feature_names_input):
        data_vars = {}
        for src_name, sample_lists in spectra.items():
            all_spectra_rows = [aug_vec for sample_augs in sample_lists for aug_vec in sample_augs]
            matrix = np.vstack([_ensure_1d_float(vec) for vec in all_spectra_rows])
            n_feat = matrix.shape[1]
            feat_dim = f"feature_{src_name}"
            current_feature_names = feature_names_input.get(src_name, [f"{src_name}_f{i}" for i in range(n_feat)]) if feature_names_input else [f"{src_name}_f{i}" for i in range(n_feat)]
            if len(current_feature_names) != n_feat:
                raise ValueError(f"feature_names length mismatch for '{src_name}'.")
            data_vars[f"spectra_{src_name}"] = xr.DataArray(matrix, dims=("obs", feat_dim), coords={"obs": obs_ids, feat_dim: current_feature_names})
        return data_vars

    @staticmethod
    def _build_target_data_var(target, n_samples, obs_ids, obs_sample_ids, sample_id_to_index, target_names_input):
        if target is None:
            return None
        targ_arr = np.asarray(target)
        if targ_arr.ndim == 1:
            targ_arr = targ_arr[:, None]
        if targ_arr.shape[0] != n_samples:
            raise ValueError("Target array length mismatch with samples.")
        expanded_target = np.vstack([targ_arr[sample_id_to_index[sid]] for sid in obs_sample_ids])
        n_vars = expanded_target.shape[1]
        final_target_names = target_names_input if target_names_input else [str(i) for i in range(n_vars)]
        if len(final_target_names) != n_vars:
            raise ValueError(f"target_names length ({len(final_target_names)}) mismatch with number of target variables ({n_vars}).")
        return xr.DataArray(expanded_target, dims=("obs", "variable"), coords={"obs": obs_ids, "variable": final_target_names})

    @staticmethod
    def _build_metadata_data_vars(metadata, n_samples, obs_ids, obs_sample_ids, sample_id_to_index):
        data_vars = {}
        if metadata:
            for key, arr_val in metadata.items():
                arr_np = np.asarray(arr_val)
                if len(arr_np) != n_samples:
                    raise ValueError(f"Metadata '{key}' length mismatch with samples.")
                expanded = [arr_np[sample_id_to_index[sid]] for sid in obs_sample_ids]
                data_vars[f"metadata_{key}"] = xr.DataArray(expanded, dims=("obs",), coords={"obs": obs_ids})
        return data_vars

    @staticmethod
    def _add_sample_level_coords(ds, n_samples, obs_sample_ids, sample_id_to_index, splits, groups, fold_ids, branches):
        def _add_coord_to_ds_helper(name, values):
            if values is None:
                return
            if len(values) != n_samples:
                raise ValueError(f"'{name}' array length mismatch with samples.")
            expanded_values = [values[sample_id_to_index[sid]] for sid in obs_sample_ids]
            ds.coords[name] = ("obs", np.asarray(expanded_values))
        _add_coord_to_ds_helper("split", splits)
        _add_coord_to_ds_helper("fold_id", fold_ids)
        _add_coord_to_ds_helper("branch", branches)
        if groups:
            for gname, arr_val in groups.items():
                _add_coord_to_ds_helper(f"group_id_{gname}", arr_val)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping source names to their feature names.
        """
        out = {}
        for var in self.ds.data_vars:
            if var.startswith("spectra_"):
                src_name = var.replace("spectra_", "")
                feat_dim = self.ds[var].dims[1]
                out[src_name] = list(self.ds[var].coords[feat_dim].astype(str).values)
        return out

    
class AugmentationManager:
    def __init__(self, core: SpectraSetCore):
        self.core = core

    def augment_samples(self, new_spectra_by_source: Dict[str, Sequence[np.ndarray]], original_obs_indices: Optional[Sequence[Union[int, Hashable]]] = None):
        if not new_spectra_by_source:
            raise ValueError("new_spectra_by_source cannot be empty.")
        first_src_name = next(iter(new_spectra_by_source))
        n_new_augmentations = len(new_spectra_by_source[first_src_name])
        for src_name, new_data_list in new_spectra_by_source.items():
            spectra_var_name = f"spectra_{src_name}"
            if spectra_var_name not in self.core.ds.data_vars:
                raise KeyError(f"Unknown source '{src_name}' in new_spectra_by_source.")
            expected_n_feat = self.core.ds[spectra_var_name].shape[1]
            if len(new_data_list) != n_new_augmentations:
                raise ValueError(f"Number of augmentations for source '{src_name}' ({len(new_data_list)}) does not match others ({n_new_augmentations}).")
            for vec in new_data_list:
                if _ensure_1d_float(vec).shape[0] != expected_n_feat:
                    raise ValueError(f"Feature count mismatch for source '{src_name}'. Expected {expected_n_feat}, got {_ensure_1d_float(vec).shape[0]}.")
        if original_obs_indices is None:
            if self.core.ds.sizes["obs"] == 0:
                raise ValueError("Cannot augment an empty SpectraSet without original_obs_indices.")
            first_obs_label = self.core.ds.coords["obs"].values[0]
            original_obs_indices = [first_obs_label] * n_new_augmentations
        if len(original_obs_indices) != n_new_augmentations:
            raise ValueError("original_obs_indices must match the number of new augmentations.")
        current_obs_labels = self.core.ds.coords["obs"].values
        obs_labels_to_copy = []
        for obs_idx in original_obs_indices:
            if obs_idx in current_obs_labels:
                obs_labels_to_copy.append(obs_idx)
            elif isinstance(obs_idx, int) and 0 <= obs_idx < len(current_obs_labels):
                obs_labels_to_copy.append(current_obs_labels[obs_idx])
            else:
                raise IndexError(f"Observation label/position {obs_idx!r} not found in dataset.")
        if "augmentation" not in self.core.ds.coords:
            raise LookupError("Coordinate 'augmentation' not found. Cannot determine original samples for augmentation.")
        original_parents_mask = (self.core.ds.coords["augmentation"] == 0)
        original_parent_candidates_ds = self.core.ds.isel(obs=original_parents_mask)
        parent_ds = original_parent_candidates_ds.reindex(obs=obs_labels_to_copy)
        if parent_ds.obs.size != len(obs_labels_to_copy) or parent_ds.obs.isnull().any():
            missing_parents = [label for i, label in enumerate(obs_labels_to_copy) if parent_ds.obs.isel(obs=i).isnull().item()]
            if missing_parents:
                raise LookupError(f"Could not find original (augmentation_id==0) parent observations for labels: {missing_parents}.")
        new_sample_ids = parent_ds.coords["sample"].values.tolist()
        new_target_values = parent_ds["target"].values if "target" in parent_ds else None
        new_metadata = {k.replace("metadata_", ""): parent_ds[k].values for k in parent_ds.data_vars if k.startswith("metadata_")}
        new_splits = parent_ds.coords["split"].values if "split" in parent_ds.coords else None
        new_fold_ids = parent_ds.coords["fold_id"].values if "fold_id" in parent_ds.coords else None
        new_branches = parent_ds.coords["branch"].values if "branch" in parent_ds.coords else None
        new_groups = {k.replace("group_id_", ""): parent_ds.coords[k].values for k in parent_ds.coords if k.startswith("group_id_")}
        max_existing_aug_id = int(self.core.ds.coords["augmentation"].max().item()) if "augmentation" in self.core.ds.coords and self.core.ds.sizes["obs"] > 0 else -1
        new_augmentation_ids = np.arange(max_existing_aug_id + 1, max_existing_aug_id + 1 + n_new_augmentations, dtype=int)
        spectra_for_build = {src: [[aug_list[i]] for i in range(n_new_augmentations)] for src, aug_list in new_spectra_by_source.items()}
        metadata_for_build = new_metadata if new_metadata else None
        target_for_build = new_target_values
        splits_for_build = new_splits
        fold_ids_for_build = new_fold_ids
        branches_for_build = new_branches
        groups_for_build = new_groups if new_groups else None
        _target_names_for_build = list(self.core.ds["target"].coords["variable"].astype(str).values) if "target" in self.core.ds and "variable" in self.core.ds["target"].coords else None
        new_augmentations_sset = SpectraSet.build(spectra=spectra_for_build, target=target_for_build, metadata=metadata_for_build, sample_ids=new_sample_ids, augmentation_ids=new_augmentation_ids.tolist(), feature_names=self.core.feature_names, target_names=_target_names_for_build, splits=splits_for_build, groups=groups_for_build, fold_ids=fold_ids_for_build, branches=branches_for_build)
        self.core.append(new_augmentations_sset.core.ds)

    def add_features(self,
                 source_name: str,
                 new_features: np.ndarray,
                 feature_names: Optional[Sequence[str]] = None,
                 feature_dim_name: Optional[str] = None):
        spectra_var = f"spectra_{source_name}"
        if spectra_var not in self.core.ds.data_vars:
            raise KeyError(f"Source '{source_name}' not found in the dataset.")

        da = self.core.ds[spectra_var]
        n_obs, n_old = da.shape

        # 1) Validate shapes
        if new_features.ndim != 2 or new_features.shape[0] != n_obs:
            raise ValueError(f"'new_features' must be 2D with {n_obs} rows, got {new_features.shape!r}")
        n_new = new_features.shape[1]

        # 2) Build the new feature‐name list
        if feature_names is None:
            feature_names = [f"derived_f{i}" for i in range(n_new)]
        if len(feature_names) != n_new:
            raise ValueError(f"Expected {n_new} feature names, got {len(feature_names)}")

        # 3) Horizontally stack the old + new data
        combined = np.hstack([da.values, new_features])  # shape (n_obs, n_old + n_new)

        # 4) Prepare the full coordinate for feature_nirs
        feat_dim = da.dims[1]  # e.g. "feature_nirs"
        old_labels = list(da.coords[feat_dim].values)
        all_labels = old_labels + list(feature_names)

        # 5) Drop the old dimension (and its variable) entirely
        ds = self.core.ds.drop_dims(feat_dim)

        # 6) Re-create the new data‐variable with the expanded shape
        ds = ds.assign({spectra_var: (("obs", feat_dim), combined)})

        # 7) Assign the new coordinate for that feature‐dimension
        ds = ds.assign_coords({feat_dim: all_labels})

        # 8) Put it back on the core
        self.core.ds = ds




class CoordinateManager:
    def __init__(self, core: SpectraSetCore):
        self.core = core

    def set_coord(self, name: str, values: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        if by == "obs":
            if len(values) != len(self.core.ds["obs"]):
                raise ValueError(f"Length of values ({len(values)}) must equal n_obs ({len(self.core.ds['obs'])}).")
            self.core.ds.coords[name] = ("obs", np.asarray(values))
        elif by == "sample":
            unique_sample_ids = np.unique(self.core.ds.coords["sample"].values)
            if len(values) != len(unique_sample_ids):
                raise ValueError(f"Length of values ({len(values)}) must equal n_unique_samples ({len(unique_sample_ids)}) when 'by' is 'sample'.")
            sample_to_val = dict(zip(unique_sample_ids, values))
            mapped_values = [sample_to_val[sid] for sid in self.core.ds.coords["sample"].values]
            self.core.ds.coords[name] = ("obs", np.asarray(mapped_values))
        else:
            raise ValueError(f"Invalid 'by' argument: {by}. Must be 'obs' or 'sample'.")

    def add_split(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.set_coord("split", labels, by=by)

    def add_fold(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.set_coord("fold_id", labels, by=by)

    def add_branch(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.set_coord("branch", labels, by=by)

    def add_group(self, name: str, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.set_coord(f"group_id_{name}", labels, by=by)

class DataRetriever:
    def __init__(self, core: SpectraSetCore):
        self.core = core
        self._label_encoder: Optional[LabelEncoder] = None

    def _filter_ds(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all"):
        ds = self.core.ds
        if ds.sizes.get("obs", 0) == 0:
            return ds.copy()
        mask = np.ones(ds.sizes["obs"], dtype=bool)
        def _apply_filter_to_mask(coord_name: str, values_to_match: Any):
            nonlocal mask
            if coord_name in ds.coords and values_to_match is not None:
                mask &= np.isin(ds.coords[coord_name].values, np.atleast_1d(values_to_match))
        _apply_filter_to_mask("split", split)
        _apply_filter_to_mask("fold_id", fold_id)
        _apply_filter_to_mask("branch", branch)
        if groups:
            for gname, v in groups.items():
                _apply_filter_to_mask(f"group_id_{gname}", v)
        if augment == "original":
            if "augmentation" in ds.coords:
                mask &= ds.coords["augmentation"].values == 0
        elif augment == "augmented":
            if "augmentation" in ds.coords:
                mask &= ds.coords["augmentation"].values > 0
            else:
                mask[:] = False
        return ds.isel(obs=mask)

    def X(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, sources: Optional[Union[bool, str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all", feature_shape: Literal["concatenate", "interlace", "2d", "transpose2d"] = "concatenate"):
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)
        if ds_filtered.sizes.get("obs", 0) == 0:
            return np.empty((0, 0))
        all_src_vars = [v for v in ds_filtered.data_vars if v.startswith("spectra_")]
        chosen_src_vars = []
        if sources is None or sources is True:
            chosen_src_vars = all_src_vars
        elif sources is False:
            chosen_src_vars = []
        elif isinstance(sources, str):
            spectra_var_name = f"spectra_{sources}"
            if spectra_var_name in ds_filtered.data_vars:
                chosen_src_vars = [spectra_var_name]
            else:
                return np.empty((ds_filtered.sizes["obs"], 0))
        elif isinstance(sources, Sequence):
            chosen_src_vars = [f"spectra_{s}" for s in sources if f"spectra_{s}" in ds_filtered.data_vars]
        if not chosen_src_vars:
            return np.empty((ds_filtered.sizes["obs"], 0))
        if feature_shape == "concatenate":
            return np.hstack([ds_filtered[v].values for v in chosen_src_vars])
        source_data_arrays = [ds_filtered[v] for v in chosen_src_vars]
        max_n_feat = max(da.shape[1] for da in source_data_arrays) if source_data_arrays else 0
        if max_n_feat == 0:
            return np.empty((ds_filtered.sizes["obs"], 0))
        padded_matrices = []
        for da in source_data_arrays:
            n_feat = da.shape[1]
            if n_feat < max_n_feat:
                padding = np.full((da.shape[0], max_n_feat - n_feat), np.nan)
                padded_matrices.append(np.hstack([da.values, padding]))
            else:
                padded_matrices.append(da.values)
        stacked_arrays = np.stack(padded_matrices, axis=1)
        if feature_shape == "2d":
            return stacked_arrays
        elif feature_shape == "transpose2d":
            return np.transpose(stacked_arrays, (0, 2, 1))
        elif feature_shape == "interlace":
            return stacked_arrays.reshape(ds_filtered.sizes["obs"], -1)
        else:
            raise ValueError(f"Unsupported 'feature_shape': {feature_shape}.")

    def X_with_labels(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        sources: Optional[Union[bool, str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
        feature_shape: Literal["concatenate", "interlace", "2d", "transpose2d"] = "concatenate",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Like `X(...)` but also returns the observation labels (from `ds.coords["obs"]`)
        row-by-row, which can be useful for linking back to the full dataset.
        """
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)
        X_data = self.X(
            split=split, fold_id=fold_id, groups=groups,
            branch=branch, sources=sources, augment=augment,
            feature_shape=feature_shape
        )
        obs_labels = ds_filtered.coords["obs"].values
        return X_data, obs_labels

    def y(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all", encode_labels: bool = False):
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)
        if "target" not in ds_filtered.data_vars:
            return np.array([])
        arr = ds_filtered["target"].values
        if encode_labels:
            if arr.dtype.kind in ("U", "S", "O"):
                le = LabelEncoder()
                self._label_encoder = le.fit(arr.ravel())
                arr = self._label_encoder.transform(arr.ravel()).reshape(arr.shape)
        return arr

    def inverse_transform_y(self, encoded_labels: np.ndarray):
        if self._label_encoder is None:
            raise RuntimeError("LabelEncoder not fitted. Call .y(encode_labels=True) first.")
        return self._label_encoder.inverse_transform(encoded_labels)

class PredictionManager:
    def __init__(self, core: SpectraSetCore):
        self.core = core

    def add_prediction(self, model_id: str, y_pred: np.ndarray, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None):
        pred_var_name = f"pred_{model_id}"
        ds_filtered = DataRetriever(self.core)._filter_ds(split, fold_id, groups, branch, augment="all")
        if ds_filtered.sizes.get("obs", 0) == 0 and len(y_pred) > 0:
            raise ValueError("No observations in filtered subset, but y_pred is not empty.")
        elif ds_filtered.sizes.get("obs", 0) != len(y_pred):
            raise ValueError(f"Length of y_pred ({len(y_pred)}) does not match the number of observations in the filtered subset ({ds_filtered.sizes.get('obs', 0)}).")
        full_predictions = np.full(self.core.ds.sizes["obs"], np.nan, dtype=float)
        obs_labels_filtered = ds_filtered.coords["obs"].values
        obs_labels_full = self.core.ds.coords["obs"].values
        original_indices_mask = np.isin(obs_labels_full, obs_labels_filtered)
        full_predictions[original_indices_mask] = y_pred
        self.core.ds[pred_var_name] = xr.DataArray(full_predictions, dims=("obs",), coords={"obs": self.core.ds.coords["obs"].values}, name=pred_var_name)

    def get_prediction(self, model_id: str, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None):
        pred_var_name = f"pred_{model_id}"
        if pred_var_name not in self.core.ds.data_vars:
            raise KeyError(f"No prediction stored for model_id '{model_id}'.")
        ds_filtered = DataRetriever(self.core)._filter_ds(split, fold_id, groups, branch, augment="all")
        if ds_filtered.sizes.get("obs", 0) == 0:
            return np.array([])
        return self.core.ds[pred_var_name].sel(obs=ds_filtered.coords["obs"].values).values

class SubsetManager:
    def __init__(self, core: SpectraSetCore):
        self.core = core

    def subset_by_samples(self, sample_ids_to_include: Sequence[Any]):
        if "sample" not in self.core.ds.coords:
            return SpectraSet(self.core.ds.copy())
        mask = np.isin(self.core.ds.coords["sample"].values, sample_ids_to_include)
        return SpectraSet(self.core.ds.isel(obs=mask).copy())

    def subset_by_obs(self, obs_indices_to_include: Sequence[int]):
        if not obs_indices_to_include:
            empty_ds = self.core.ds.isel(obs=slice(0))
            return SpectraSet(empty_ds.copy())
        n_obs = len(self.core.ds["obs"])
        max_index = n_obs - 1
        for idx in obs_indices_to_include:
            if idx < 0 or idx > max_index:
                raise IndexError(f"obs index {idx} out of bounds (0 to {max_index}).")
        return SpectraSet(self.core.ds.isel(obs=list(obs_indices_to_include)).copy())

class SpectraSet:
    def __init__(self, ds: Optional[xr.Dataset] = None):
        self.core = SpectraSetCore(ds)
        self.augmentation = AugmentationManager(self.core)
        self.coordinates = CoordinateManager(self.core)
        self.data = DataRetriever(self.core)
        self.predictions = PredictionManager(self.core)
        self.subset = SubsetManager(self.core)

    @classmethod
    def build(cls, *, spectra: Dict[str, List[List[np.ndarray]]], target: Optional[np.ndarray] = None, metadata: Optional[Dict[str, np.ndarray]] = None, sample_ids: Optional[List[Any]] = None, augmentation_ids: Optional[List[Any]] = None, feature_names: Optional[Dict[str, List[str]]] = None, target_names: Optional[List[str]] = None, splits: Optional[np.ndarray] = None, groups: Optional[Dict[str, np.ndarray]] = None, fold_ids: Optional[np.ndarray] = None, branches: Optional[np.ndarray] = None):
        instance = cls()
        instance.core.build(spectra, target, metadata, sample_ids, augmentation_ids, feature_names, target_names, splits, groups, fold_ids, branches)
        return instance

    def append(self, *, spectra: Dict[str, List[np.ndarray]], target: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None, sample_ids: Optional[List[Any]] = None, augmentation_ids: Optional[List[Any]] = None, splits: Optional[np.ndarray] = None, groups: Optional[Dict[str, np.ndarray]] = None, fold_ids: Optional[np.ndarray] = None, branches: Optional[np.ndarray] = None):
        if not spectra:
            raise ValueError("'spectra' cannot be empty when appending.")
        first_src_name = next(iter(spectra))
        n_new_obs = len(spectra[first_src_name])
        for src_name, new_obs_list in spectra.items():
            if len(new_obs_list) != n_new_obs:
                raise ValueError(f"All sources in 'spectra' must provide the same number of rows ({n_new_obs}).")
            spectra_var_name = f"spectra_{src_name}"
            if spectra_var_name not in self.core.ds.data_vars:
                raise ValueError(f"Unknown source '{src_name}'. Cannot append new data for it.")
            expected_n_feat = self.core.ds[spectra_var_name].shape[1]
            for vec in new_obs_list:
                if _ensure_1d_float(vec).shape[0] != expected_n_feat:
                    raise ValueError(f"Feature count mismatch for source '{src_name}'. Expected {expected_n_feat}, got {_ensure_1d_float(vec).shape[0]}.")
        if sample_ids is None:
            current_max_sid = np.max(self.core.ds.coords["sample"].values) if "sample" in self.core.ds.coords and self.core.ds.sizes["obs"] > 0 else -1
            sample_ids = list(range(current_max_sid + 1, current_max_sid + 1 + n_new_obs))
        if augmentation_ids is None:
            augmentation_ids = self.core._generate_next_augmentation_ids(n_new_obs).tolist()
        spectra_nested = {src: [[vec] for vec in obs_list] for src, obs_list in spectra.items()}
        def _check_length(arr, name):
            if arr is not None and len(arr) != n_new_obs:
                raise ValueError(f"'{name}' must have length = {n_new_obs}.")
        _check_length(target, "target")
        _check_length(splits, "splits")
        _check_length(fold_ids, "fold_ids")
        _check_length(branches, "branches")
        if metadata:
            for k, v in metadata.items():
                _check_length(v, f"metadata['{k}']")
        if groups:
            for k, v in groups.items():
                _check_length(v, f"groups['{k}']")
        new_part = SpectraSet.build(spectra=spectra_nested, target=target, metadata=metadata, sample_ids=sample_ids, augmentation_ids=augmentation_ids, splits=splits, groups=groups, fold_ids=fold_ids, branches=branches)
        self.core.append(new_part.core.ds)
        return self

    def augment_samples(self, new_spectra_by_source: Dict[str, Sequence[np.ndarray]], original_obs_indices: Optional[Sequence[Union[int, Hashable]]] = None):
        self.augmentation.augment_samples(new_spectra_by_source, original_obs_indices)
        return self

    def add_features(self, source_name: str, new_features: np.ndarray, feature_names: Optional[Sequence[str]] = None, feature_dim_name: Optional[str] = None):
        self.augmentation.add_features(source_name, new_features, feature_names, feature_dim_name)
        return self

    def set_coord(self, name: str, values: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.coordinates.set_coord(name, values, by)
        return self

    def add_split(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.coordinates.add_split(labels, by)
        return self

    def add_fold(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.coordinates.add_fold(labels, by)
        return self

    def add_branch(self, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.coordinates.add_branch(labels, by)
        return self

    def add_group(self, name: str, labels: Sequence[Any], by: Literal["obs", "sample"] = "sample"):
        self.coordinates.add_group(name, labels, by)
        return self

    def X(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, sources: Optional[Union[bool, str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all", feature_shape: Literal["concatenate", "interlace", "2d", "transpose2d"] = "concatenate"):
        return self.data.X(split=split, fold_id=fold_id, groups=groups, branch=branch, sources=sources, augment=augment, feature_shape=feature_shape)

    def X_with_labels(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, sources: Optional[Union[bool, str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all", feature_shape: Literal["concatenate", "interlace", "2d", "transpose2d"] = "concatenate"):
        return self.data.X_with_labels(split=split, fold_id=fold_id, groups=groups, branch=branch, sources=sources, augment=augment, feature_shape=feature_shape)

    def y(self, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None, augment: Literal["all", "original", "augmented"] = "all", encode_labels: bool = False):
        return self.data.y(split=split, fold_id=fold_id, groups=groups, branch=branch, augment=augment, encode_labels=encode_labels)

    def inverse_transform_y(self, encoded_labels: np.ndarray):
        return self.data.inverse_transform_y(encoded_labels)

    def add_prediction(self, model_id: str, y_pred: np.ndarray, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None):
        self.predictions.add_prediction(model_id, y_pred, split=split, fold_id=fold_id, groups=groups, branch=branch)

    def get_prediction(self, model_id: str, split: Optional[Union[str, Sequence[str]]] = None, fold_id: Optional[Union[int, Sequence[int]]] = None, groups: Optional[Dict[str, Any]] = None, branch: Optional[Union[str, Sequence[str]]] = None):
        return self.predictions.get_prediction(model_id, split=split, fold_id=fold_id, groups=groups, branch=branch)

    def subset_by_samples(self, sample_ids_to_include: Sequence[Any]):
        return self.subset.subset_by_samples(sample_ids_to_include)

    def subset_by_obs(self, obs_indices_to_include: Sequence[int]):
        return self.subset.subset_by_obs(obs_indices_to_include)

    def __len__(self):
        return len(self.core.ds["obs"])

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return SpectraSet(self.core.ds.isel(obs=key).copy())
        elif isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype == bool:
            if len(key) != len(self):
                raise IndexError(f"Boolean index length ({len(key)}) mismatch with dataset length ({len(self)}).")
            return SpectraSet(self.core.ds.isel(obs=key).copy())
        elif isinstance(key, (np.ndarray, list, tuple)):
            indices = np.atleast_1d(key).astype(int)
            return self.subset_by_obs(indices.tolist())
        else:
            raise TypeError(f"Invalid key type for slicing: {type(key)}")

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        out = {}
        for var in self.core.ds.data_vars:
            if var.startswith("spectra_"):
                src_name = var.replace("spectra_", "")
                feat_dim = self.core.ds[var].dims[1]
                out[src_name] = list(self.core.ds[var].coords[feat_dim].astype(str).values)
        return out

    def sample_ids(self) -> np.ndarray:
        if "sample" not in self.core.ds.coords:
            return np.array([])
        return np.unique(self.core.ds.coords["sample"].values)
    
    @property
    def available_sources(self):
        """
        Returns a list of available spectral sources in the dataset.
        """
        return [str(k).replace("spectra_", "") for k in self.core.ds.data_vars if str(k).startswith("spectra_")]
    
    @property
    def ds(self):
        """
        Provides access to the underlying xarray.Dataset.
        """
        return self.core.ds
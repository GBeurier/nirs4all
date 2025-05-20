"""
spectraset.py

An xarray-backed container for spectral ML pipelines (nirs4all), now supporting:
- Multi-source spectra with *uneven* feature lengths via separate variables
- Uneven augmentation counts per sample and source via a unified "obs" axis
- Zero-copy, NumPy-array-like getters for sklearn compatibility
- Filtering by split, folds, groups, branch
- Predictions storage
- Categorical target encoding/inversion

Usage example:
    # Build from dict of sources -> nested list of aug arrays per sample
    spectra = {
        'raman': [[...], [...], ...],  # list length n_samples, each sublist len vary, arrays len f_raman
        'nirs': [[...], [...], ...],   # arrays len f_nirs
    }
    target = array([...])  # shape (n_samples,)
    ss = SpectraSet.build(spectra=spectra, target=target)
    X = ss.X(include_augmentations=False, flatten='concat')  # shape (n_obs, total_features)

Dependencies: numpy, pandas, xarray, sklearn
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from typing import Any, Dict, Sequence, Optional, Union, Literal, List # Added List
from sklearn.preprocessing import LabelEncoder

# Assuming SpectraSetConfig is defined elsewhere, or using a placeholder if not critical for this fix.
# For now, let\'s assume it might be a simple config object.
# If it\'s a more complex class, its definition would be needed.
# As a placeholder, if it\'s not found, one might use:
# class SpectraSetConfig: pass
# Or, if it\'s just a dictionary of settings:
# SpectraSetConfig = Dict[str, Any]
# For the purpose of this fix, I will assume it\'s a class that can be instantiated.
# If this causes issues, the actual definition of SpectraSetConfig will be required.
# For now, to make the linter happy if it\'s not defined elsewhere in the project:
try:
    from .config import SpectraSetConfig # Assuming it might be in a local config.py
except ImportError:
    class SpectraSetConfig: # Placeholder
        def __init__(self):
            pass


class SpectraSet:
    """
    Container for spectral data with ragged augmentation and multi-source support.
    """

    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self._label_encoder: Optional[LabelEncoder] = None

    @classmethod
    def build(
        cls,
        spectra: Dict[str, List[List[np.ndarray]]],
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        groups: Optional[Dict[str, np.ndarray]] = None, # Parameter for groups
        splits: Optional[np.ndarray] = None,           # Parameter for splits
        sample_ids: Optional[List[Any]] = None,
        augmentation_ids: Optional[List[Any]] = None,
        feature_names: Optional[Dict[str, List[str]]] = None,
        target_names: Optional[List[str]] = None,
        config: Optional[SpectraSetConfig] = None,
        # Added fold_id_param and branch_param to pass these through if needed for coords
        fold_id_param: Optional[np.ndarray] = None, 
        branch_param: Optional[np.ndarray] = None
    ) -> "SpectraSet":
        current_config = config if config is not None else SpectraSetConfig()

        if not spectra:
            raise ValueError("Spectra data must be provided.")

        first_source_name = next(iter(spectra))
        num_original_samples = len(spectra[first_source_name])

        if num_original_samples == 0 and not any(s for s in spectra.values()):
            raise ValueError("Cannot build SpectraSet with no samples and no spectral data.")

        if sample_ids is None:
            actual_sample_ids = list(range(num_original_samples))
        else:
            if len(sample_ids) != num_original_samples:
                raise ValueError(
                    f"Provided sample_ids length ({len(sample_ids)}) must match "
                    f"number of samples ({num_original_samples})."
                )
            actual_sample_ids = sample_ids

        num_augmentations_per_sample = [
            len(sample_augs) for sample_augs in spectra[first_source_name]
        ]

        for source_name, source_data_list in spectra.items():
            if len(source_data_list) != num_original_samples:
                raise ValueError(f"Source \'{source_name}\' has {len(source_data_list)} samples, expected {num_original_samples}.")
            current_source_augs_per_sample = [len(s_augs) for s_augs in source_data_list]
            if current_source_augs_per_sample != num_augmentations_per_sample:
                raise ValueError(
                    f"Source \'{source_name}\' has inconsistent augmentation counts "
                    f"compared to source \'{first_source_name}\'."
                )

        total_observations = sum(num_augmentations_per_sample)
        
        if total_observations == 0 and num_original_samples > 0:
            pass


        obs_aug_ids_expanded: List[Any] = []
        obs_sample_ids_expanded: List[Any] = []

        default_aug_id_counter = 0
        for i, num_augs in enumerate(num_augmentations_per_sample):
            original_sample_id = actual_sample_ids[i]
            if augmentation_ids and len(augmentation_ids) == total_observations:
                obs_aug_ids_expanded.extend(augmentation_ids[default_aug_id_counter: default_aug_id_counter + num_augs])
            else: # Default augmentation IDs
                obs_aug_ids_expanded.extend(list(range(num_augs)))
            
            obs_sample_ids_expanded.extend([original_sample_id] * num_augs)
            default_aug_id_counter += num_augs
        
        all_data_vars: Dict[str, Any] = {}
        all_coords: Dict[str, Any] = {}

        multi_idx = pd.MultiIndex.from_arrays(
            [obs_sample_ids_expanded, obs_aug_ids_expanded],
            names=("sample", "augmentation"),
        )
        all_coords["observation"] = multi_idx
        observation_coords = all_coords["observation"]


        for source_name, source_data_list in spectra.items():
            if not isinstance(source_data_list, list) or not all(
                isinstance(sample_augs, list) for sample_augs in source_data_list
            ):
                raise TypeError(
                    f"Data for source \'{source_name}\' must be List[List[np.ndarray]]."
                )

            flat_source_spectra: List[np.ndarray] = []
            for sample_idx, sample_augs_list in enumerate(source_data_list):
                for aug_idx, aug_data in enumerate(sample_augs_list):
                    if not isinstance(aug_data, np.ndarray):
                        raise TypeError(
                            f"Spectrum data for source \'{source_name}\', sample {sample_idx}, "
                            f"augmentation {aug_idx} must be a numpy array."
                        )
                    if aug_data.ndim == 1:
                        aug_data = aug_data.reshape(1, -1) # Ensure 2D: (1, n_features)
                    if aug_data.ndim != 2 or aug_data.shape[0] != 1:
                        raise ValueError(
                            f"Each spectrum for source \'{source_name}\', sample {sample_idx}, "
                            f"augmentation {aug_idx} must be effectively 2D with shape (1, n_features). "
                            f"Got {aug_data.shape}."
                        )
                    flat_source_spectra.append(aug_data)
            
            if not flat_source_spectra:
                if total_observations > 0:
                    raise ValueError(f"No spectral data found for source \'{source_name}\' despite expecting {total_observations} observations.")
                else: 
                    continue

            num_features_this_source = flat_source_spectra[0].shape[1]
            if not all(s.shape[1] == num_features_this_source for s in flat_source_spectra):
                raise ValueError(
                    f"All spectra for source \'{source_name}\' must have the same number of features."
                )

            source_feature_dim_name = f"feature_{source_name}" 
            source_feature_coords: Union[List[str], np.ndarray]
            if feature_names and source_name in feature_names:
                source_feature_coords = feature_names[source_name]
                if len(source_feature_coords) != num_features_this_source:
                    raise ValueError(
                        f"Provided feature_names for source \'{source_name}\' has "
                        f"{len(source_feature_coords)} names, but data has "
                        f"{num_features_this_source} features."
                    )
            else:
                source_feature_coords = [
                    f"{source_name}_f{i}" for i in range(num_features_this_source)
                ] 

            if len(flat_source_spectra) != total_observations:
                raise ValueError(
                    f"Internal inconsistency: flattened spectra for source \'{source_name}\' "
                    f"has {len(flat_source_spectra)} items, expected {total_observations}."
                )

            source_np_array = np.vstack(flat_source_spectra) if flat_source_spectra else np.empty((0, num_features_this_source))


            data_array = xr.DataArray(
                source_np_array,
                coords={
                    "observation": observation_coords,
                    source_feature_dim_name: source_feature_coords, 
                },
                dims=("observation", source_feature_dim_name), 
                name=f"spectra_{source_name}",
            )
            all_data_vars[f"spectra_{source_name}"] = data_array
        
        if target is not None:
            targ_arr = np.asarray(target)
            if targ_arr.ndim == 1:
                targ_arr = targ_arr[:, None]
            n_vars = targ_arr.shape[1]
            targ_obs = np.vstack([targ_arr[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded]) # Ensure correct indexing if actual_sample_ids is not 0..N-1
            all_data_vars["target"] = xr.DataArray(
                targ_obs,
                dims=("observation", "variable"),
                coords={
                    "observation": observation_coords,
                    "variable": target_names or np.arange(n_vars)
                }
            )

        if metadata:
            for key, arr in metadata.items():
                vals = [arr[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded] # Ensure correct indexing
                all_data_vars[f"metadata_{key}"] = xr.DataArray(
                    vals,
                    dims=("observation",),
                    coords={"observation": observation_coords}
                )

        ds = xr.Dataset(all_data_vars, coords=all_coords) # Pass all_coords here

        if splits is not None:
            # Ensure splits align with original samples before expanding
            if len(splits) != num_original_samples:
                raise ValueError(f"Splits length ({len(splits)}) must match number of original samples ({num_original_samples})")
            split_vals = [splits[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded]
            ds = ds.assign_coords(split=("observation", split_vals))
        
        # Handle fold_id_param and branch_param similarly to splits
        if fold_id_param is not None:
            if len(fold_id_param) != num_original_samples:
                raise ValueError(f"fold_id_param length ({len(fold_id_param)}) must match number of original samples ({num_original_samples})")
            fold_vals = [fold_id_param[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded]
            ds = ds.assign_coords(fold_id=("observation", fold_vals))

        if branch_param is not None:
            if len(branch_param) != num_original_samples:
                raise ValueError(f"branch_param length ({len(branch_param)}) must match number of original samples ({num_original_samples})")
            br_vals = [branch_param[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded]
            ds = ds.assign_coords(branch=("observation", br_vals))

        if groups:
            for name, arr in groups.items():
                if len(arr) != num_original_samples:
                     raise ValueError(f"Group \'{name}\' length ({len(arr)}) must match number of original samples ({num_original_samples})")
                grp_vals = [arr[actual_sample_ids.index(s)] for s, _ in obs_sample_ids_expanded]
                ds = ds.assign_coords({f"group_id_{name}": ("observation", grp_vals)})

        return cls(ds)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        """Names of features for each spectral source."""
        names: Dict[str, List[str]] = {}
        for var_key in self.ds.data_vars:
            var_name = str(var_key)
            if var_name.startswith("spectra_"):
                source_name = var_name.replace("spectra_", "")
                feature_dim_for_source = f"feature_{source_name}"
                
                data_array = self.ds[var_name]
                if feature_dim_for_source in data_array.dims:
                    if feature_dim_for_source in data_array.coords:
                        names[source_name] = list(data_array.coords[feature_dim_for_source].values.astype(str)) # Ensure strings
                    else:
                        dim_idx = data_array.dims.index(feature_dim_for_source)
                        num_features = data_array.shape[dim_idx]
                        names[source_name] = [f"{source_name}_f{i}" for i in range(num_features)]
        return names

    def X(
        self,
        split: Optional[Union[str, List[str]]] = None,
        fold_id: Optional[Union[int, List[int]]] = None,
        groups: Optional[Dict[str, Union[Any, List[Any]]]] = None,
        branch: Optional[str] = None,
        include_sources: Optional[Union[bool, str, Sequence[str]]] = None,
    ) -> np.ndarray:
        ds_filtered = self._filter_ds(split, fold_id, groups, branch) # Corrected to _filter_ds

        all_spectral_vars_in_ds = sorted([str(k) for k in self.ds.data_vars if str(k).startswith("spectra_")])
        
        source_variable_names_to_use: List[str] = []

        if include_sources is None or include_sources is True:
            source_variable_names_to_use = all_spectral_vars_in_ds
        elif isinstance(include_sources, str):
            var_name = f"spectra_{include_sources}"
            if var_name in all_spectral_vars_in_ds:
                source_variable_names_to_use = [var_name]
        elif isinstance(include_sources, Sequence): 
            for s_name_candidate in include_sources:
                s_name = str(s_name_candidate) 
                var_name = f"spectra_{s_name}"
                if var_name in all_spectral_vars_in_ds:
                    source_variable_names_to_use.append(var_name)
        elif include_sources is False:
            pass 
        else:
            raise TypeError(f"Unsupported type for include_sources: {type(include_sources)}")

        if ds_filtered.observation.size == 0:
            # If no observations match filter, return empty array with correct number of features
            if not source_variable_names_to_use:
                return np.empty((0, 0))
            
            total_features = 0
            for var_name_str in source_variable_names_to_use:
                if var_name_str in self.ds.data_vars: # Check original ds for feature count
                    source_da = self.ds[var_name_str]
                    feature_dim_name = f"feature_{var_name_str.replace('spectra_', '')}"
                    if feature_dim_name in source_da.dims: # Ensure it's a valid dimension
                        dim_idx = source_da.dims.index(feature_dim_name)
                        total_features += source_da.shape[dim_idx]
            return np.empty((0, total_features))

        if not source_variable_names_to_use: # No sources selected or found valid
            return np.empty((ds_filtered.observation.size, 0))

        # If only one source is requested
        if len(source_variable_names_to_use) == 1:
            single_source_var_name = source_variable_names_to_use[0]
            if single_source_var_name in ds_filtered.data_vars:
                return ds_filtered[single_source_var_name].data # Use .data to get numpy array
            else:
                num_features_single_source = 0
                if single_source_var_name in self.ds.data_vars:
                    source_da = self.ds[single_source_var_name]
                    feature_dim_name = f"feature_{single_source_var_name.replace('spectra_', '')}"
                    if feature_dim_name in source_da.dims:
                        dim_idx = source_da.dims.index(feature_dim_name)
                        num_features_single_source = source_da.shape[dim_idx]
                return np.empty((ds_filtered.observation.size, num_features_single_source))

        # Multiple sources: get .data from each and hstack
        arrays_to_stack = []
        for var_name in source_variable_names_to_use: 
            if var_name in ds_filtered.data_vars:
                arrays_to_stack.append(ds_filtered[var_name].data) # Use .data
            else:
                num_features_this_source = 0
                if var_name in self.ds.data_vars:
                    source_da = self.ds[var_name]
                    feature_dim_name = f"feature_{var_name.replace('spectra_', '')}"
                    if feature_dim_name in source_da.dims:
                        dim_idx = source_da.dims.index(feature_dim_name)
                        num_features_this_source = source_da.shape[dim_idx]
                arrays_to_stack.append(np.empty((ds_filtered.observation.size, num_features_this_source)))


        if not arrays_to_stack: # Should be caught by earlier checks on source_variable_names_to_use
            return np.empty((ds_filtered.observation.size, 0))
        
        # Ensure all arrays are 2D before hstack
        processed_arrays_to_stack = []
        for arr in arrays_to_stack:
            if arr.ndim == 1:
                # This case should ideally not happen if data is (obs, features)
                # If it does, it implies an empty feature dimension for some observations,
                # which needs careful handling or prevention upstream.
                # For now, reshape to (n_obs, 1) if it's 1D and has data,
                # or ensure it's (n_obs, 0) if truly empty features.
                if arr.size > 0 : # Has elements
                     processed_arrays_to_stack.append(arr.reshape(-1,1))
                else: # No elements, ensure it's (n_obs, 0)
                     processed_arrays_to_stack.append(arr.reshape(ds_filtered.observation.size,0))

            elif arr.ndim == 2:
                processed_arrays_to_stack.append(arr)
            else:
                # This would be an unexpected shape
                raise ValueError(f"Unexpected array dimension {arr.ndim} for hstack.")
        
        if not processed_arrays_to_stack: # if all arrays were empty and resulted in empty list
             return np.empty((ds_filtered.observation.size, 0))


        return np.hstack(processed_arrays_to_stack)

    def y(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None
    ) -> np.ndarray:
        """Return target array (n_obs, n_vars) with filters."""
        ds = self._filter_ds(split, fold_id, groups, branch)
        arr = ds['target'].values
        # label encode if needed
        if arr.dtype.kind in ('U', 'S', 'O'):
            flat = arr.ravel()
            le = LabelEncoder().fit(flat)
            self._label_encoder = le
            transformed_flat = le.transform(flat)
            arr = transformed_flat.reshape(arr.shape)  # type: ignore[attr-defined]
        return arr

    def add_prediction(
        self,
        model_id: str,
        y_pred: np.ndarray,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None
    ) -> None:
        """Store prediction per obs aligned with filters without alignment errors."""
        name = f"pred_{model_id}"
        ds = self.ds
        # filter dataset to find target obs positions
        ds_f = self._filter_ds(split, fold_id, groups, branch)
        # boolean mask over full obs axis
        mask = np.isin(ds.obs.values, ds_f.obs.values)
        # full-length array (always float to support np.nan)
        full = np.full(len(ds.obs), np.nan, dtype=float)
        full[mask] = y_pred
        # create DataArray without extra coords to avoid re-alignment
        da = xr.DataArray(
            full,
            dims=("obs",),
            coords={"obs": ds.obs},
            name=name
        )
        # assign by directly setting data_vars to bypass alignment
        self.ds = self.ds.assign({name: da})

    def get_prediction(
        self,
        model_id: str,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None
    ) -> np.ndarray:
        name = f"pred_{model_id}"
        if name not in self.ds:
            raise KeyError(f"No predictions for '{model_id}'")
        ds_f = self._filter_ds(split, fold_id, groups, branch)
        mask = np.isin(self.ds.obs.values, ds_f.obs.values)
        return self.ds[name].values[mask]

    def subset_by_samples(self, sample_idx: Sequence[int]) -> SpectraSet:
        """View restricted to given sample indices."""
        mask = np.isin(self.ds.coords['sample'].values, sample_idx)
        return SpectraSet(self.ds.sel(obs=mask))

    def subset_by_obs(self, obs_idx: Sequence[int]) -> SpectraSet:
        """View restricted to given obs indices."""
        return SpectraSet(self.ds.isel(obs=obs_idx))

    def __array__(self) -> np.ndarray:
        """Default array interface: X_train()."""
        return self.X()

    def __len__(self) -> int:
        """Number of observations."""
        return len(self.ds.coords['obs'])

    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> SpectraSet:
        """Index into obs axis, returning a view."""
        if isinstance(idx, slice):
            # Convert slice to a sequence of indices
            start, stop, step = idx.indices(len(self))
            obs_indices = list(range(start, stop, step))
            return self.subset_by_obs(obs_indices)
        return self.subset_by_obs(np.atleast_1d(idx).tolist())  # type: ignore[arg-type]

    def _filter_ds(
        self,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None
    ) -> xr.Dataset:
        """Filter dataset by coord values on obs."""
        ds = self.ds
        mask = np.ones(len(ds.coords['obs']), dtype=bool)
        coords = ds.coords
        if split is not None and 'split' in coords:
            vals = np.atleast_1d(split)
            mask &= np.isin(coords['split'].values, vals)
        if fold_id is not None and 'fold_id' in coords:
            vals = np.atleast_1d(fold_id)
            mask &= np.isin(coords['fold_id'].values, vals)
        if branch is not None and 'branch' in coords:
            vals = np.atleast_1d(branch)
            mask &= np.isin(coords['branch'].values, vals)
        if groups:
            for name, v in groups.items():
                coord = f'group_id_{name}'
                if coord in coords:
                    mask &= np.isin(coords[coord].values, np.atleast_1d(v))
        return ds.sel(obs=mask)

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union, Literal, Tuple, Hashable
import dataclasses  # Added import

import numpy as np
import xarray as xr
from sklearn.preprocessing import LabelEncoder


@dataclasses.dataclass
class PreparedIdsAndCounts:
    """Helper dataclass to hold results from _prepare_ids_and_counts."""
    n_samples: int
    total_obs: int
    final_sample_ids: List[Any]
    sample_id_to_index: Dict[Any, int]
    sample_aug_counts: List[int]
    obs_indices_list: List[int]
    final_augmentation_ids: List[Any]
    obs_to_sample_id_list: List[Any]


# -----------------------------------------------------------------------------#
# Helper utilities                                                           #
# -----------------------------------------------------------------------------#
def _expand_sample_to_obs(
    sample_level_data: Sequence[Any],
    obs_sample_ids: Sequence[Any],
    sample_id_to_index: Dict[Any, int],
) -> List[Any]:
    """Expand a per-sample array to a per-observation array using sample_id_to_index lookup."""
    return [sample_level_data[sample_id_to_index[sid]] for sid in obs_sample_ids]


def _ensure_1d_float(arr: Any) -> np.ndarray:
    """Return *arr* as a 1-D float array; raise if dimensionality is wrong."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Spectral vectors must be 1-D")
    return arr.astype(float, copy=False)


# -----------------------------------------------------------------------------#
# Main container                                                             #
# -----------------------------------------------------------------------------#
class SpectraSet:
    """
    Mutable container for multi-source spectral datasets built on **xarray**.

    It unifies raw measurements, target variables, and metadata, and provides
    flexible mechanisms to apply transformation, augmentation, splitting, grouping,
    and branching for machine learning and deep learning pipelines.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, ds: Optional[xr.Dataset] = None):
        """
        Initializes a SpectraSet instance.

        An empty SpectraSet can be created by calling with no arguments.
        """
        if ds is None:
            # Initialize with an 'obs' coordinate even if empty
            ds = xr.Dataset(coords={"obs": np.array([], dtype=int)})
        self.ds = ds
        self._label_encoder: Optional[LabelEncoder] = None
        # Track the next available observation ID for appending
        self._next_obs_id = int(self.ds.sizes.get("obs", 0))

    # ----------------------------- build ------------------------------ #

    @classmethod
    def _prepare_ids_and_counts(
        cls,
        spectra: Dict[str, List[List[np.ndarray]]],
        sample_ids_input: Optional[List[Any]],
        augmentation_ids_input: Optional[List[Any]],
    ) -> PreparedIdsAndCounts:  # Changed return type
        """Helper to prepare and validate sample/observation IDs and counts."""
        if not spectra:
            raise ValueError("\\'spectra\\' must contain at least one source.")

        first_src_name = next(iter(spectra))
        n_samples = len(spectra[first_src_name])

        final_sample_ids: List[Any]
        if sample_ids_input is None:
            final_sample_ids = list(range(n_samples))
        else:
            final_sample_ids = sample_ids_input

        if len(final_sample_ids) != n_samples:
            raise ValueError("sample_ids length mismatch with samples.")
        sample_id_to_index = {sid: i for i, sid in enumerate(final_sample_ids)}

        sample_aug_counts = [len(s_augs) for s_augs in spectra[first_src_name]]
        for src_name, sample_lists in spectra.items():
            if len(sample_lists) != n_samples:
                raise ValueError(
                    f"Source '{src_name}' has {len(sample_lists)} samples, "
                    f"but '{first_src_name}' has {n_samples}."
                )
            current_aug_counts = [len(s_augs) for s_augs in sample_lists]
            if current_aug_counts != sample_aug_counts:
                raise ValueError(
                    f"Augmentation counts for source '{src_name}' are not "
                    "consistent across samples with other sources.\\n                "
                )

        total_obs = sum(sample_aug_counts)
        obs_indices_list = np.arange(total_obs).tolist()

        obs_to_sample_id_list: List[Any] = [
            final_sample_ids[i] for i, count in enumerate(sample_aug_counts) for _ in range(count)
        ]

        final_augmentation_ids: List[Any]
        if augmentation_ids_input is None:
            final_augmentation_ids = [j for count in sample_aug_counts for j in range(count)]
        else:
            final_augmentation_ids = augmentation_ids_input

        if len(final_augmentation_ids) != total_obs:
            raise ValueError("augmentation_ids length must equal total observations.")
        
        return PreparedIdsAndCounts(
            n_samples=n_samples,
            total_obs=total_obs,
            final_sample_ids=final_sample_ids,
            sample_id_to_index=sample_id_to_index,
            sample_aug_counts=sample_aug_counts,
            obs_indices_list=obs_indices_list,
            final_augmentation_ids=final_augmentation_ids,
            obs_to_sample_id_list=obs_to_sample_id_list,
        )

    @classmethod
    def _build_spectra_data_vars(
        cls,
        spectra: Dict[str, List[List[np.ndarray]]],
        obs_ids: List[Any],
        feature_names_input: Optional[Dict[str, List[str]]],
    ) -> Dict[str, xr.DataArray]:
        """Helper to build DataArrays for spectral data."""
        data_vars: Dict[str, xr.DataArray] = {}
        for src_name, sample_lists in spectra.items():
            all_spectra_rows: List[np.ndarray] = []
            for sample_augs in sample_lists:
                for aug_vec in sample_augs:
                    all_spectra_rows.append(_ensure_1d_float(aug_vec))

            matrix = np.vstack(all_spectra_rows)
            n_feat = matrix.shape[1]
            feat_dim = f"feature_{src_name}"
            current_feature_names = (
                feature_names_input[src_name]
                if feature_names_input and src_name in feature_names_input
                else [f"{src_name}_f{i}" for i in range(n_feat)]
            )
            if len(current_feature_names) != n_feat:
                raise ValueError(f"feature_names length mismatch for '{src_name}'.")

            data_vars[f"spectra_{src_name}"] = xr.DataArray(
                matrix,
                dims=("obs", feat_dim),
                coords={"obs": obs_ids, feat_dim: current_feature_names},
            )
        return data_vars

    @classmethod
    def _build_target_data_var(
        cls,
        target: Optional[np.ndarray],
        n_samples: int,
        obs_ids: List[Any],
        obs_sample_ids: List[Any],
        sample_id_to_index: Dict[Any, int],
        target_names_input: Optional[List[str]],
    ) -> Optional[xr.DataArray]:
        """Helper to build DataArray for target data."""
        if target is None:
            return None
        
        targ_arr = np.asarray(target)
        if targ_arr.ndim == 1:
            targ_arr = targ_arr[:, None]
        
        if targ_arr.shape[0] != n_samples:
            raise ValueError("Target array length mismatch with samples.")

        expanded_target = np.vstack(_expand_sample_to_obs(targ_arr, obs_sample_ids, sample_id_to_index))  # type: ignore[arg-type]
        n_vars = expanded_target.shape[1]
        
        final_target_names = target_names_input
        if final_target_names is None:
            final_target_names = [str(i) for i in range(n_vars)]
        elif len(final_target_names) != n_vars:
            raise ValueError(f"target_names length ({len(final_target_names)}) mismatch with number of target variables ({n_vars}).")

        return xr.DataArray(
            expanded_target,
            dims=("obs", "variable"),
            coords={"obs": obs_ids, "variable": final_target_names},
        )

    @classmethod
    def _build_metadata_data_vars(
        cls,
        metadata: Optional[Dict[str, np.ndarray]],
        n_samples: int,
        obs_ids: List[Any],
        obs_sample_ids: List[Any],
        sample_id_to_index: Dict[Any, int],
    ) -> Dict[str, xr.DataArray]:
        """Helper to build DataArrays for metadata."""
        data_vars: Dict[str, xr.DataArray] = {}
        if metadata:
            for key, arr_val in metadata.items(): # Renamed arr to arr_val to avoid conflict
                arr_np = np.asarray(arr_val) # Renamed arr to arr_np
                if len(arr_np) != n_samples:
                    raise ValueError(f"Metadata '{key}' length mismatch with samples.")
                expanded = _expand_sample_to_obs(arr_np, obs_sample_ids, sample_id_to_index)  # type: ignore[arg-type]
                data_vars[f"metadata_{key}"] = xr.DataArray(
                    expanded, dims=("obs",), coords={"obs": obs_ids}
                )
        return data_vars

    @classmethod
    def _add_sample_level_coords_to_ds(
        cls,
        ds: xr.Dataset,
        n_samples: int,
        obs_sample_ids: List[Any],
        sample_id_to_index: Dict[Any, int],
        splits: Optional[np.ndarray],
        groups: Optional[Dict[str, np.ndarray]],
        fold_ids: Optional[np.ndarray],
        branches: Optional[np.ndarray],
    ):
        """Helper to add sample-level coordinates to the Dataset."""
        def _add_coord_to_ds_helper(name: str, values: Optional[np.ndarray]):
            if values is None:
                return
            if len(values) != n_samples:
                raise ValueError(f"'{name}' array length mismatch with samples.")
            expanded_values = _expand_sample_to_obs(values, obs_sample_ids, sample_id_to_index)  # type: ignore[arg-type]
            ds.coords[name] = ("obs", np.asarray(expanded_values))

        _add_coord_to_ds_helper("split", splits)
        _add_coord_to_ds_helper("fold_id", fold_ids)
        _add_coord_to_ds_helper("branch", branches)
        if groups:
            for gname, arr_val in groups.items(): # Renamed arr to arr_val
                _add_coord_to_ds_helper(f"group_id_{gname}", arr_val)

    @classmethod
    def build(
        cls,
        *,
        spectra: Dict[str, List[List[np.ndarray]]],
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        sample_ids: Optional[List[Any]] = None, # This is sample_ids_input for _prepare_ids_and_counts
        augmentation_ids: Optional[List[Any]] = None, # This is augmentation_ids_input for _prepare_ids_and_counts
        feature_names: Optional[Dict[str, List[str]]] = None,
        target_names: Optional[List[str]] = None, 
        splits: Optional[np.ndarray] = None,
        groups: Optional[Dict[str, np.ndarray]] = None,
        fold_ids: Optional[np.ndarray] = None,
        branches: Optional[np.ndarray] = None,
    ) -> "SpectraSet":
        """
        Constructs a new SpectraSet from raw input data.

        Parameters
        ----------
        spectra : Dict[str, List[List[np.ndarray]]]
            A dictionary where keys are source names (e.g., 'NIRS', 'MIR') and
            values are lists of lists of 1-D NumPy arrays. The outer list corresponds
            to samples, and the inner list corresponds to augmentations for that sample.
            e.g., {'NIRS': [[sample1_aug0, sample1_aug1], [sample2_aug0]]}
        target : Optional[np.ndarray], default=None
            1-D or 2-D array of target variables for each *sample*.
        metadata : Optional[Dict[str, np.ndarray]], default=None
            Dictionary of metadata arrays, where keys are metadata names and values
            are arrays aligned with *samples*.
        sample_ids : Optional[List[Any]], default=None
            Unique identifiers for each *sample*. If None, integer IDs are generated.
        augmentation_ids : Optional[List[Any]], default=None
            Identifiers for each *observation* (augmented or original). If None,
            integer IDs are generated (0 for original, >0 for augmentations).
        feature_names : Optional[Dict[str, List[str]]], default=None
            Dictionary mapping source names to lists of feature names for each source.
        target_names : Optional[List[str]], default=None
            Names for target variables if `target` is 2-D.
        splits : Optional[np.ndarray], default=None
            Array of split labels for each *sample*.
        groups : Optional[Dict[str, np.ndarray]], default=None
            Dictionary of group arrays, where keys are group names and values
            are arrays aligned with *samples*.
        fold_ids : Optional[np.ndarray], default=None
            Array of fold identifiers for each *sample*.
        branches : Optional[np.ndarray], default=None
            Array of branch labels for each *sample*.

        Returns
        -------
        SpectraSet
            A new SpectraSet instance.

        Raises
        ------
        ValueError
            If input data is inconsistent (e.g., mismatched lengths, dimensions).
        """
        id_data = cls._prepare_ids_and_counts(spectra, sample_ids, augmentation_ids)

        coords: Dict[str, Any] = {
            "obs": id_data.obs_indices_list,
            "sample": ("obs", np.array(id_data.obs_to_sample_id_list)),
            "augmentation": ("obs", np.array(id_data.final_augmentation_ids)),
        }
        
        data_vars: Dict[str, xr.DataArray] = {}
        # Pass the original feature_names from build's parameters
        data_vars.update(cls._build_spectra_data_vars(spectra, id_data.obs_indices_list, feature_names))
        
        # Pass the original target and target_names from build's parameters
        target_da = cls._build_target_data_var(
            target, id_data.n_samples, id_data.obs_indices_list, 
            id_data.obs_to_sample_id_list, id_data.sample_id_to_index, target_names
        )
        if target_da is not None:
            data_vars["target"] = target_da

        # Pass the original metadata from build's parameters
        data_vars.update(cls._build_metadata_data_vars(
            metadata, id_data.n_samples, id_data.obs_indices_list, 
            id_data.obs_to_sample_id_list, id_data.sample_id_to_index
        ))
        
        ds = xr.Dataset(data_vars, coords=coords)

        # Pass the original splits, groups, fold_ids, branches from build's parameters
        cls._add_sample_level_coords_to_ds(
            ds, id_data.n_samples, id_data.obs_to_sample_id_list, id_data.sample_id_to_index,
            splits, groups, fold_ids, branches
        )

        return cls(ds)

    # ------------------------------------------------------------------ #
    # Private helpers for mutability                                     #
    # ------------------------------------------------------------------ #
    def _append_ds(self, other: xr.Dataset) -> None:
        """Concatenate *other* along ``obs`` after sanity checks."""
        if self.ds.sizes.get("obs", 0) == 0:
            self.ds = other
        else:
            # Check for consistent dimensions and feature names across sources
            for var_hashable in other.data_vars:
                var = str(var_hashable)  # Ensure var is a string
                if var.startswith("spectra_"):
                    if var not in self.ds.data_vars:
                        raise ValueError(f"New dataset contains unknown source '{var}'.")
                    feat_dim_other = other[var].dims[1]
                    feat_dim_self = self.ds[var].dims[1]
                    if feat_dim_other != feat_dim_self:
                        raise ValueError(
                            f"Feature dimension mismatch for '{var}' "
                            f"(expected '{feat_dim_self}', got '{feat_dim_other}')."
                        )
                    if not np.array_equal(
                        self.ds[var].coords[feat_dim_self].values,
                        other[var].coords[feat_dim_other].values,
                    ):
                        raise ValueError(f"Feature names mismatch for '{var}'.")
            
            # Concatenate along the 'obs' dimension
            self.ds = xr.concat([self.ds, other], dim="obs")

        self._next_obs_id = int(self.ds.sizes["obs"])

    def _generate_next_augmentation_ids(self, n: int) -> np.ndarray:
        """Generate *n* fresh augmentation ids."""
        if "augmentation" not in self.ds.coords or self.ds.sizes["obs"] == 0:
            start = 0
        else:
            # Find the max augmentation ID and start from next
            start = int(self.ds.coords["augmentation"].max().item()) + 1
        return np.arange(start, start + n, dtype=int)

    # ------------------------------------------------------------------ #
    # Public mutator: append new rows                                    #
    # ------------------------------------------------------------------ #
    def append(
        self,
        *,
        spectra: Dict[str, List[np.ndarray]],  # one vector per new obs for each source
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,  # can be dict of lists or arrays
        sample_ids: Optional[List[Any]] = None,
        augmentation_ids: Optional[List[Any]] = None,
        splits: Optional[np.ndarray] = None,
        groups: Optional[Dict[str, np.ndarray]] = None,
        fold_ids: Optional[np.ndarray] = None,
        branches: Optional[np.ndarray] = None,
    ) -> "SpectraSet":
        """
        Append **new observations** (rows) to *self* and return the object.

        *spectra* must map every **existing** source name to a list of 1-D
        vectors. All lists must have the same length ``n_new``. Feature counts
        and order must match the current dataset.

        Parameters
        ----------
        spectra : Dict[str, List[np.ndarray]]
            A dictionary where keys are source names and values are lists of
            1-D NumPy arrays, representing new observations for each source.
            All lists must have the same number of observations.
        target : Optional[np.ndarray], default=None
            1-D array of target variables for the new observations.
        metadata : Optional[Dict[str, Any]], default=None
            Dictionary of metadata for the new observations. Values can be lists or arrays.
        sample_ids : Optional[List[Any]], default=None
            Unique sample IDs for the new observations. If None, new unique
            integer IDs are generated.
        augmentation_ids : Optional[List[Any]], default=None
            Augmentation IDs for the new observations. If None, new unique
            integer IDs are generated.
        splits : Optional[np.ndarray], default=None
            Split labels for the new observations.
        groups : Optional[Dict[str, np.ndarray]], default=None
            Dictionary of group labels for the new observations.
        fold_ids : Optional[np.ndarray], default=None
            Fold IDs for the new observations.
        branches : Optional[np.ndarray], default=None
            Branch labels for the new observations.

        Returns
        -------
        SpectraSet
            The modified SpectraSet instance.

        Raises
        ------
        ValueError
            If input data is inconsistent with existing data (e.g., mismatched
            lengths, unknown sources, feature count mismatches).
        """
        if not spectra:
            raise ValueError("'spectra' cannot be empty when appending.")

        first_src_name = next(iter(spectra))
        n_new_obs = len(spectra[first_src_name])

        # Check existing sources and feature consistency
        for src_name, new_obs_list in spectra.items():
            if len(new_obs_list) != n_new_obs:
                raise ValueError(
                    f"All sources in 'spectra' must provide the same number of rows ({n_new_obs})."
                )
            spectra_var_name = f"spectra_{src_name}"
            if spectra_var_name not in self.ds.data_vars:
                raise ValueError(f"Unknown source '{src_name}'. Cannot append new data for it.")

            expected_n_feat = self.ds[spectra_var_name].shape[1]
            for vec in new_obs_list:
                if _ensure_1d_float(vec).shape[0] != expected_n_feat:
                    raise ValueError(
                        f"Feature count mismatch for source '{src_name}'. "
                        f"Expected {expected_n_feat}, got {_ensure_1d_float(vec).shape[0]}."
                    )
        
        # Determine sample_ids for the new observations
        if sample_ids is None:
            # Generate fresh, non-colliding sample IDs. If no existing samples, start from 0.
            current_max_sid = -1
            if "sample" in self.ds.coords and self.ds.sizes["obs"] > 0:
                current_max_sid = np.max(self.ds.coords["sample"].values)
            sample_ids = list(range(current_max_sid + 1, current_max_sid + 1 + n_new_obs))

        # Determine augmentation_ids for the new observations
        if augmentation_ids is None:
            # For appended data, we assume each is an "original" (augmentation_id = 0)
            # unless a specific augmentation ID is intended. Here, we generate unique ones.
            augmentation_ids = self._generate_next_augmentation_ids(n_new_obs).tolist()

        # Convert to the *nested* format required by `build` (each new obs is an "original" augmentation of itself)
        spectra_nested = {src: [[vec] for vec in obs_list] for src, obs_list in spectra.items()}

        # Validate lengths of optional arrays/dicts
        def _check_length(arr: Optional[Union[np.ndarray, List[Any]]], name: str):
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

        # Build a temporary SpectraSet for the new data
        # Note: target, metadata, splits, groups, fold_ids, branches are all sample-level in build.
        # Since we're appending individual observations, we treat each new observation as a "sample"
        # for the purpose of passing to build.
        new_part = SpectraSet.build(
            spectra=spectra_nested,
            target=target,  # This will be expanded per-obs by build
            metadata=metadata,  # This will be expanded per-obs by build
            sample_ids=sample_ids,
            augmentation_ids=augmentation_ids,
            splits=splits,  # This will be expanded per-obs by build
            groups=groups,  # This will be expanded per-obs by build
            fold_ids=fold_ids,  # This will be expanded per-obs by build
            branches=branches,  # This will be expanded per-obs by build
        )
        self._append_ds(new_part.ds)
        return self

    # ------------------------------------------------------------------ #
    # Public mutator: sample-level augmentation                          #
    # ------------------------------------------------------------------ #
    def augment_samples(
        self,
        new_spectra_by_source: Dict[str, Sequence[np.ndarray]],
        *,
        original_obs_indices: Optional[Sequence[Union[int, Hashable]]] = None,  # Use Union[int, Hashable] for obs_idx
    ) -> SpectraSet:
        """
        Append a batch of precomputed sample-level augmentations for one or more sources.

        The new augmented observations will inherit sample IDs, metadata, splits,
        groups, fold IDs, and branches from their parent original observations.
        New, unique augmentation IDs will be assigned.

        Parameters
        ----------
        new_spectra_by_source : Dict[str, Sequence[np.ndarray]]
            A dictionary where keys are source names and values are sequences of
            1-D NumPy arrays, representing the augmented spectra. All sequences
            must have the same length.
        original_obs_indices : Optional[Sequence[Union[int, Hashable]]], default=None
            A list of **positional** observation indices (0...n_obs-1) or
            `ds.coords["obs"]` labels corresponding to the parent original
            observations for each new augmented spectrum. If None, all new
            augmentations are assumed to originate from the first observation (index 0).

        Returns
        -------
        SpectraSet
            The modified SpectraSet instance with appended augmentations.

        Raises
        ------
        KeyError
            If an unknown source name is provided in `new_spectra_by_source`.
        ValueError
            If input data is inconsistent (e.g., mismatched lengths, dimensions),
            or if `original_obs_indices` is provided and its length does not match
            `new_spectra_by_source`.
        IndexError
            If an `original_obs_indices` label is not found.
        LookupError
            If `aug_idx` coordinate is missing or original parents cannot be found.
        """
        if not new_spectra_by_source:
            raise ValueError("new_spectra_by_source cannot be empty.")

        first_src_name = next(iter(new_spectra_by_source))
        n_new_augmentations = len(new_spectra_by_source[first_src_name])

        # Validate feature dimensions for all new spectra and all sources
        for src_name, new_data_list in new_spectra_by_source.items():
            spectra_var_name = f"spectra_{src_name}"
            if spectra_var_name not in self.ds.data_vars:
                raise KeyError(f"Unknown source '{src_name}' in new_spectra_by_source.")

            expected_n_feat = self.ds[spectra_var_name].shape[1]
            if len(new_data_list) != n_new_augmentations:
                raise ValueError(
                    f"Number of augmentations for source '{src_name}' "
                    f"({len(new_data_list)}) does not match others ({n_new_augmentations})."
                )
            for vec in new_data_list:
                arr = np.asarray(vec)
                if arr.ndim != 1 or arr.shape[0] != expected_n_feat:
                    raise ValueError(
                        f"Each new spectrum for source '{src_name}' must be 1-D of length {expected_n_feat}."
                    )

        # Determine parent observations for copying metadata
        if original_obs_indices is None:
            if self.ds.sizes["obs"] == 0:
                raise ValueError("Cannot augment an empty SpectraSet without original_obs_indices.")
            # Default: all new augmentations come from the first observation
            # Ensure there's at least one observation to pick from
            first_obs_label = self.ds.coords["obs"].values[0]
            original_obs_indices = [first_obs_label] * n_new_augmentations
        
        if len(original_obs_indices) != n_new_augmentations:
            raise ValueError("original_obs_indices must match the number of new augmentations.")

        # Ensure original_obs_indices are actual observation labels from ds.coords["obs"]
        # or convert positional indices to labels.
        current_obs_labels = self.ds.coords["obs"].values
        obs_labels_to_copy_list: List[Hashable] = []  # Use Hashable for labels
        for obs_idx in original_obs_indices:
            if obs_idx in current_obs_labels:  # obs_idx is a label
                obs_labels_to_copy_list.append(obs_idx)
            elif isinstance(obs_idx, int) and 0 <= obs_idx < len(current_obs_labels):
                # If it's a positional index, get the corresponding label
                obs_labels_to_copy_list.append(current_obs_labels[obs_idx])
            else:
                raise IndexError(f"Observation label/position {obs_idx!r} not found in dataset.")

        # Convert to numpy array for advanced indexing
        obs_labels_to_copy = np.array(obs_labels_to_copy_list)

        # Select parent observations. We must select the *original* instance (aug_idx == 0)
        # of each parent observation to ensure correct metadata copying.
        
        if "augmentation" not in self.ds.coords:  # Changed from "aug_idx" to "augmentation" as per build method
            raise LookupError("Coordinate 'augmentation' not found. Cannot determine original samples for augmentation.")
        
        original_parents_mask = (self.ds.coords["augmentation"] == 0)
        
        # Filter the dataset to only include original parent observations
        original_parent_candidates_ds = self.ds.isel(obs=original_parents_mask)
        
        # Now, select the specific original parents based on obs_labels_to_copy
        # We use .reindex() instead of .sel() to handle potential duplicate labels in obs_labels_to_copy
        parent_ds = original_parent_candidates_ds.reindex(obs=obs_labels_to_copy)
        
        # Check if any parent was not found after filtering for originals
        # Use DataArray.isnull().any() for checking NaNs in coordinates/DataArrays
        if parent_ds.obs.size != len(obs_labels_to_copy) or parent_ds.obs.isnull().any():
            missing_parents = []
            for i, label in enumerate(obs_labels_to_copy):
                # Check if the reindexed obs for this label is null (NaN or NaT)
                if parent_ds.obs.isel(obs=i).isnull().item():  # .item() is fine on a 0-d DataArray
                    missing_parents.append(label)
            
            if missing_parents:  # Only raise if there are actually missing parents
                raise LookupError(
                    f"Could not find original (augmentation_id==0) parent observations for labels: {missing_parents}. "
                    f"Ensure original_obs_indices refer to samples that exist as originals."
                )

        # Extract parent sample IDs and metadata for the new observations
        new_sample_ids = parent_ds.coords["sample"].values.tolist()
        new_target_values = parent_ds["target"].values if "target" in parent_ds else None

        new_metadata: Dict[str, Any] = {}
        for var_name_hashable in parent_ds.data_vars:
            var_name = str(var_name_hashable)  # Ensure var_name is a string
            if var_name.startswith("metadata_"):
                # Correctly slice the parent_ds metadata for the (potentially duplicated) obs_labels_to_copy
                new_metadata[var_name.replace("metadata_", "")] = parent_ds[var_name].values
        
        # Extract other sample-level coordinates
        new_splits = parent_ds.coords["split"].values if "split" in parent_ds.coords else None
        new_fold_ids = parent_ds.coords["fold_id"].values if "fold_id" in parent_ds.coords else None
        new_branches = parent_ds.coords["branch"].values if "branch" in parent_ds.coords else None
        
        new_groups: Dict[str, np.ndarray] = {}
        for coord_name_hashable in parent_ds.coords:
            coord_name = str(coord_name_hashable)  # Ensure coord_name is a string
            if coord_name.startswith("group_id_"):  # Check type of coord_name
                new_groups[coord_name.replace("group_id_", "")] = parent_ds.coords[coord_name].values

        # Generate new unique augmentation IDs for the new observations
        # These should be unique across the entire dataset after appending
        max_existing_aug_id = -1
        if "augmentation" in self.ds.coords and self.ds.sizes["obs"] > 0:
            max_existing_aug_id = int(self.ds.coords["augmentation"].max().item())
        
        new_augmentation_ids = np.arange(
            max_existing_aug_id + 1,
            max_existing_aug_id + 1 + n_new_augmentations,
            dtype=int
        )

        # Prepare data for the new observations in the format expected by SpectraSet.build
        # Each new augmentation is treated as a "sample" with one "augmentation" (itself)
        # for the purpose of using the .build() method logic.
        
        # The `spectra` for build needs to be Dict[str, List[List[np.ndarray]]]
        # Outer list is per "sample" (here, per new augmentation)
        # Inner list is per "augmentation of that sample" (here, just one, the new aug itself)
        
        spectra_for_build: Dict[str, List[List[np.ndarray]]] = {}
        for src_name, aug_list in new_spectra_by_source.items():
            spectra_for_build[src_name] = [[aug_list[i]] for i in range(n_new_augmentations)]

        # Metadata for build needs to be Dict[str, np.ndarray] where array is per "sample"
        metadata_for_build: Optional[Dict[str, np.ndarray]] = None
        if new_metadata:
            metadata_for_build = {}
            for meta_key, meta_values_all_parents in new_metadata.items():
                # meta_values_all_parents corresponds to parent_ds, which is already aligned with obs_labels_to_copy
                metadata_for_build[meta_key] = meta_values_all_parents

        # Target for build needs to be np.ndarray (n_samples, n_vars)
        target_for_build: Optional[np.ndarray] = None
        if new_target_values is not None:
            # new_target_values corresponds to parent_ds, aligned with obs_labels_to_copy
            target_for_build = new_target_values

        # Other sample-level attributes for build
        splits_for_build: Optional[np.ndarray] = new_splits
        fold_ids_for_build: Optional[np.ndarray] = new_fold_ids
        branches_for_build: Optional[np.ndarray] = new_branches
        groups_for_build: Optional[Dict[str, np.ndarray]] = None
        if new_groups:
            groups_for_build = {}
            for group_key, group_values_all_parents in new_groups.items():
                groups_for_build[group_key] = group_values_all_parents
        
        # Fetch target_names from the existing dataset if they exist
        _target_names_for_build: Optional[List[str]] = None
        if "target" in self.ds and "variable" in self.ds["target"].coords:
            _target_names_for_build = self.ds["target"].coords["variable"].astype(str).values.tolist()

        # Create a temporary SpectraSet for the new augmentations
        new_augmentations_sset = SpectraSet.build(
            spectra=spectra_for_build,
            target=target_for_build,
            metadata=metadata_for_build,
            sample_ids=new_sample_ids,  # Parent sample IDs are reused
            augmentation_ids=new_augmentation_ids.tolist(),  # New unique aug IDs
            feature_names=self.feature_names,  # Reuse existing feature names (property access)
            target_names=_target_names_for_build,  # Use fetched target names
            splits=splits_for_build,
            groups=groups_for_build,
            fold_ids=fold_ids_for_build,
            branches=branches_for_build,
        )

        # Append the new augmentations to the current dataset
        self._append_ds(new_augmentations_sset.ds)
        return self

    # ------------------------------------------------------------------ #
    # Public mutator: feature-level augmentation / derivation            #
    # ------------------------------------------------------------------ #
    def add_features(
        self,
        source_name: str,
        new_features: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        feature_dim_name: Optional[str] = None,
    ) -> "SpectraSet":
        """
        Adds new derived features to an existing spectral source.

        New features are added as an additional dimension within the existing
        `spectra_{source_name}` DataArray.

        Parameters
        ----------
        source_name : str
            The name of the spectral source to which features will be added.
        new_features : np.ndarray
            A 2D array of new features, where rows correspond to observations
            and columns to new features. Must have `len(self)` rows.
        feature_names : Optional[Sequence[str]], default=None
            Names for the new features. If None, generic names are generated.
            Length must match the number of new features.
        feature_dim_name : Optional[str], default=None
            The name for the new feature dimension. If None, a name like
            'feature_{source_name}_derived' will be used.

        Returns
        -------
        SpectraSet
            The modified SpectraSet instance.

        Raises
        ------
        KeyError
            If `source_name` does not correspond to an existing spectral source.
        ValueError
            If `new_features` has incorrect dimensions or length, or if
            `feature_names` length mismatches.
        """
        spectra_var_name = f"spectra_{source_name}"
        if spectra_var_name not in self.ds.data_vars:
            raise KeyError(f"Source '{source_name}' not found in the dataset.")

        if new_features.ndim != 2 or new_features.shape[0] != len(self):
            raise ValueError(
                f"'new_features' must be a 2D array with {len(self)} rows (observations)."
            )

        n_new_feat = new_features.shape[1]
        if feature_names is None:
            feature_names = [f"derived_f{i}" for i in range(n_new_feat)]
        if len(feature_names) != n_new_feat:
            raise ValueError(
                f"Length of 'feature_names' ({len(feature_names)}) must match "
                f"the number of new features ({n_new_feat})."
            )

        current_feat_dim = self.ds[spectra_var_name].dims[1]
        current_feat_names = self.ds[spectra_var_name].coords[current_feat_dim].values.tolist()

        # Concatenate new features with existing ones
        combined_matrix = np.hstack([self.ds[spectra_var_name].values, new_features])
        combined_feature_names = current_feat_names + list(feature_names)
        
        # Determine the dimension name for the combined features
        if feature_dim_name is None:
            new_feat_dim = current_feat_dim  # Keep the original dimension name if no new one provided
        else:
            new_feat_dim = feature_dim_name  # Use the new dimension name

        # Update the DataArray for the specified source
        self.ds[spectra_var_name] = xr.DataArray(
            combined_matrix,
            dims=("obs", new_feat_dim),
            coords={"obs": self.ds.coords["obs"].values, new_feat_dim: combined_feature_names},
        )
        return self

    # ------------------------------------------------------------------ #
    # Generic coordinate mutator                                         #
    # ------------------------------------------------------------------ #
    def set_coord(
        self,
        name: str,
        values: Sequence[Any],
        *,
        by: Literal["obs", "sample"] = "sample",
    ) -> "SpectraSet":
        """
        Add or overwrite a coordinate (e.g., 'split', 'fold_id', 'group_id_X', 'branch').

        Parameters
        ----------
        name : str
            The name of the coordinate to set.
        values : Sequence[Any]
            The values for the coordinate.
        by : Literal["obs", "sample"], default="sample"
            Determines how `values` align with the dataset:
            - If "obs" (observation-level), `values` must have `len(self)` elements.
            - If "sample" (sample-level), `values` must have `len(self.sample_ids)``
              unique elements and will be broadcast to observations.

        Returns
        -------
        SpectraSet
            The modified SpectraSet instance.

        Raises
        ------
        ValueError
            If `values` length is inconsistent with `by` parameter.
        """
        if by == "obs":
            if len(values) != len(self):
                raise ValueError(f"Length of values ({len(values)}) must equal n_obs ({len(self)}).")
            self.ds.coords[name] = ("obs", np.asarray(values))
        elif by == "sample":
            unique_sample_ids = self.sample_ids()
            if len(values) != len(unique_sample_ids):
                raise ValueError(
                    f"Length of values ({len(values)}) must equal n_unique_samples "
                    f"({len(unique_sample_ids)}) when 'by' is 'sample'."
                )
            
            sample_to_val = dict(zip(unique_sample_ids, values))
            mapped_values = [sample_to_val[sid] for sid in self.ds.coords["sample"].values]
            self.ds.coords[name] = ("obs", np.asarray(mapped_values))
        else:
            raise ValueError(f"Invalid 'by' argument: {by}. Must be 'obs' or 'sample'.")
        return self

    # Convenience wrappers for common coordinates
    def add_split(self, labels: Sequence[Any], *, by: Literal["obs", "sample"] = "sample") -> "SpectraSet":
        """Add or overwrite the 'split' coordinate."""
        return self.set_coord("split", labels, by=by)

    def add_fold(self, labels: Sequence[Any], *, by: Literal["obs", "sample"] = "sample") -> "SpectraSet":
        """Add or overwrite the 'fold_id' coordinate."""
        return self.set_coord("fold_id", labels, by=by)

    def add_branch(self, labels: Sequence[Any], *, by: Literal["obs", "sample"] = "sample") -> "SpectraSet":
        """Add or overwrite the 'branch' coordinate."""
        return self.set_coord("branch", labels, by=by)

    def add_group(self, name: str, labels: Sequence[Any], *, by: Literal["obs", "sample"] = "sample") -> "SpectraSet":
        """Add or overwrite a custom group coordinate (e.g., 'group_id_subject')."""
        return self.set_coord(f"group_id_{name}", labels, by=by)

    # ------------------------------------------------------------------ #
    # READ-ONLY API                                                      #
    # ------------------------------------------------------------------ #

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        """Returns a dictionary mapping source names to their feature names."""
        out: Dict[str, List[str]] = {}
        for var_hashable in self.ds.data_vars:
            var = str(var_hashable)  # Ensure var is a string
            if var.startswith("spectra_"):
                src_name = var.replace("spectra_", "")
                # Get the feature dimension name dynamically
                feat_dim = self.ds[var].dims[1]
                out[src_name] = list(self.ds[var].coords[feat_dim].astype(str).values)
        return out
    
    def sample_ids(self) -> np.ndarray:
        """Returns the unique sample IDs in the dataset."""
        if "sample" not in self.ds.coords:
            return np.array([])
        return np.unique(self.ds.coords["sample"].values)

    # ------------- filtering helper -------------
    def _filter_ds(
        self,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
    ) -> xr.Dataset:
        """
        Applies filtering based on coordinates and augmentation status.

        This private helper method returns a filtered xarray.Dataset view.
        """
        ds = self.ds
        if ds.sizes.get("obs", 0) == 0:
            return ds.copy()  # Return a copy of empty ds to prevent accidental mutation

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
            else:  # If no augmentation coord, all are considered original
                pass  # All data is considered original, so mask remains true for these
        elif augment == "augmented":
            if "augmentation" in ds.coords:
                mask &= ds.coords["augmentation"].values > 0
            else:  # If no augmentation coord, nothing is augmented
                mask[:] = False  # No augmented data exists
        
        return ds.isel(obs=mask)

    # -------------------------- X (Input Samples) --------------------------------
    def X(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        sources: Optional[Union[bool, str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
        feature_shape: Literal["concatenate", "interlace", "2d", "transpose2d"] = "concatenate",
    ) -> np.ndarray:
        """
        Retrieves the input samples (spectra) as a NumPy array.

        Parameters
        ----------
        split : Optional[Union[str, Sequence[str]]], default=None
            Filter by specific split labels.
        fold_id : Optional[Union[int, Sequence[int]]] , default=None
            Filter by specific fold IDs.
        groups : Optional[Dict[str, Any]], default=None
            Filter by custom group names and their values.
        branch : Optional[Union[str, Sequence[str]]] , default=None
            Filter by specific branch labels.
        sources : Optional[Union[bool, str, Sequence[str]]], default=None
            Specify which spectral sources to include:
            - True or None: Include all available sources.
            - False: Include no sources (returns empty array).
            - str: Include a single specified source.
            - Sequence[str]: Include multiple specified sources.
        augment : Literal["all", "original", "augmented"], default="all"
            Filter observations by their augmentation status.
        feature_shape : Literal["concatenate", "interlace", "2d", "transpose2d"], default="concatenate"
            Determines the shape of the returned feature array:
            - "concatenate": (n_obs, sum_of_features_across_sources) - default, flat.
            - "interlace": (n_obs, n_sources * max_features_per_source) - pads with NaNs.
            - "2d": (n_obs, n_sources, max_features_per_source) - 3D array, pads with NaNs.
            - "transpose2d": (n_obs, max_features_per_source, n_sources) - 3D array, pads with NaNs.

        Returns
        -------
        np.ndarray
            A NumPy array of the selected input samples.

        Raises
        ------
        ValueError
            If an unsupported `feature_shape` is requested.
        """
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)
        
        if ds_filtered.sizes.get("obs", 0) == 0:
            return np.empty((0, 0))  # Return empty array if no observations

        all_src_vars_hashable = [v for v in ds_filtered.data_vars if str(v).startswith("spectra_")]
        all_src_vars = sorted(str(v) for v in all_src_vars_hashable)  # Ensure strings for sorting and comparison
        
        chosen_src_vars: List[str] = []
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
            chosen_src_vars = [
                f"spectra_{s}" for s in sources if f"spectra_{s}" in ds_filtered.data_vars
            ]
        
        if not chosen_src_vars:
            return np.empty((ds_filtered.sizes["obs"], 0))

        if feature_shape == "concatenate":
            return np.hstack([ds_filtered[v].values for v in chosen_src_vars])
        
        # For "interlace", "2d", "transpose2d", we need consistent feature lengths
        # across sources. Pad with NaNs if necessary.
        source_data_arrays = [ds_filtered[v] for v in chosen_src_vars]
        max_n_feat = 0
        if source_data_arrays:
            max_n_feat = max(da.shape[1] for da in source_data_arrays)

        if max_n_feat == 0:  # If no features or no sources selected
            return np.empty((ds_filtered.sizes["obs"], 0))

        padded_matrices = []
        for da in source_data_arrays:
            n_feat = da.shape[1]
            if n_feat < max_n_feat:
                padding = np.full((da.shape[0], max_n_feat - n_feat), np.nan)
                padded_matrices.append(np.hstack([da.values, padding]))
            else:
                padded_matrices.append(da.values)
        
        stacked_arrays = np.stack(padded_matrices, axis=1)  # (n_obs, n_sources, max_features)

        if feature_shape == "2d":
            return stacked_arrays
        elif feature_shape == "transpose2d":
            return np.transpose(stacked_arrays, (0, 2, 1))  # (n_obs, max_features, n_sources)
        elif feature_shape == "interlace":
            # Reshape from (n_obs, n_sources, max_features) to (n_obs, n_sources * max_features)
            return stacked_arrays.reshape(ds_filtered.sizes["obs"], -1)
        else:
            raise ValueError(f"Unsupported 'feature_shape': {feature_shape}.")


    def X_with_obs_labels(
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
        Like `.X(...)` but also returns the **original observation labels** (from `ds.coords["obs"]`)

        row–by–row, which can be useful for linking back to the full dataset.
        """
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)
        X_data = self.X(
            split=split, fold_id=fold_id, groups=groups,
            branch=branch, sources=sources, augment=augment,
            feature_shape=feature_shape
        )
        # These are the actual obs labels from the filtered dataset
        obs_labels = ds_filtered.coords["obs"].values
        return X_data, obs_labels

    # -------------------------- y (Targets) --------------------------------

    def y(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
        encode_labels: bool = False,
    ) -> np.ndarray:
        """
        Retrieves the target variables as a NumPy array.

        Automatically encodes categorical labels if `encode_labels` is True.

        Parameters
        ----------
        split, fold_id, groups, branch, augment
            Filtering parameters, same as for `X()`.
        encode_labels : bool, default=False
            If True, categorical targets will be encoded to numerical labels
            using `sklearn.preprocessing.LabelEncoder`. The encoder is stored
            internally for `inverse_transform`.

        Returns
        -------
        np.ndarray
            A NumPy array of the selected target variables.
            Returns an empty array if no 'target' variable exists or no observations.

        Raises
        ------
        KeyError
            If no 'target' variable exists in the dataset.
        """
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment)

        if "target" not in ds_filtered.data_vars:
            # Return empty array if target variable does not exist
            return np.array([])

        arr = ds_filtered["target"].values
        
        if encode_labels:
            # Check if the target is indeed categorical (object, string kind)
            if arr.dtype.kind in ("U", "S", "O"):
                le = LabelEncoder()
                # Fit and transform on the flattened array
                self._label_encoder = le.fit(arr.ravel())
                arr = self._label_encoder.transform(arr.ravel()).reshape(arr.shape)  # type: ignore[attr-defined]
        
        return arr

    def inverse_transform_y(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decodes numerical target labels back to their original categorical values
        using the last fitted LabelEncoder.

        Parameters
        ----------
        encoded_labels : np.ndarray
            Numerical labels to decode.

        Returns
        -------
        np.ndarray
            Original categorical labels.

        Raises
        ------
        RuntimeError
            If a LabelEncoder has not been fitted yet (i.e., `y(encode_labels=True)`)
            has not been called on categorical data).
        """
        if self._label_encoder is None:
            raise RuntimeError("LabelEncoder not fitted. Call .y(encode_labels=True) first.")
        return self._label_encoder.inverse_transform(encoded_labels)


    # ---------------- prediction storage ----------------------
    def add_prediction(
        self,
        model_id: str,
        y_pred: np.ndarray,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Stores model predictions for a specific subset of data.

        Predictions are stored as new DataArrays named `pred_{model_id}`.
        Missing observations (not in the filtered subset) will have NaN values.

        Parameters
        ----------
        model_id : str
            A unique identifier for the model whose predictions are being stored.
        y_pred : np.ndarray
            The array of predictions. Its length must match the number of
            observations in the filtered subset.
        split, fold_id, groups, branch
            Filtering parameters, specifying the subset of data these
            predictions correspond to.
        
        Raises
        ------
        ValueError
            If the length of `y_pred` does not match the size of the filtered subset.
        """
        pred_var_name = f"pred_{model_id}"
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment="all")

        if ds_filtered.sizes.get("obs", 0) == 0 and len(y_pred) > 0:
            raise ValueError("No observations in filtered subset, but y_pred is not empty.")
        elif ds_filtered.sizes.get("obs", 0) != len(y_pred):
            raise ValueError(
                f"Length of y_pred ({len(y_pred)}) does not match "
                f"the number of observations in the filtered subset ({ds_filtered.sizes.get('obs', 0)})."
            )

        # Create a full array with NaNs for all observations
        full_predictions = np.full(self.ds.sizes["obs"], np.nan, dtype=float)
        
        # Map filtered observations back to their original positions in the full dataset
        # by matching 'obs' labels
        obs_labels_filtered = ds_filtered.coords["obs"].values
        obs_labels_full = self.ds.coords["obs"].values
        
        # Find indices in the full dataset corresponding to the filtered observations
        # Use np.isin and then get the corresponding indices
        original_indices_mask = np.isin(obs_labels_full, obs_labels_filtered)
        
        full_predictions[original_indices_mask] = y_pred

        self.ds[pred_var_name] = xr.DataArray(
            full_predictions,
            dims=("obs",),
            coords={"obs": self.ds.coords["obs"].values},
            name=pred_var_name
        )

    def get_prediction(
        self,
        model_id: str,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        """
        Retrieves stored model predictions for a specific subset of data.

        Parameters
        ----------
        model_id : str
            The identifier of the model whose predictions are to be retrieved.
        split, fold_id, groups, branch
            Filtering parameters, defining the subset of data for which
            predictions are desired.

        Returns
        -------
        np.ndarray
            A NumPy array of the retrieved predictions for the specified subset.

        Raises
        ------
        KeyError
            If no prediction is stored for the given `model_id`.
        """
        pred_var_name = f"pred_{model_id}"
        if pred_var_name not in self.ds.data_vars:
            raise KeyError(f"No prediction stored for model_id '{model_id}'.")
        
        ds_filtered = self._filter_ds(split, fold_id, groups, branch, augment="all")

        if ds_filtered.sizes.get("obs", 0) == 0:
            return np.array([])  # Return empty array if no observations in filtered subset
            
        # Select predictions corresponding to the filtered observations
        # Using .sel on the prediction DataArray with filtered 'obs' labels
        # This implicitly handles the NaN values for observations not in the subset
        return self.ds[pred_var_name].sel(obs=ds_filtered.coords["obs"].values).values

    # ---------------- convenience slicing --------------------
    def subset_by_samples(self, sample_ids_to_include: Sequence[Any]) -> "SpectraSet":
        """
        Returns a new SpectraSet containing only observations belonging to specified sample IDs.

        Parameters
        ----------
        sample_ids_to_include : Sequence[Any]
            A sequence of sample IDs to include in the subset.

        Returns
        -------
        SpectraSet
            A new SpectraSet instance containing the subsetted data.
        """
        if "sample" not in self.ds.coords:
            return SpectraSet(self.ds.copy())  # Return a copy if no sample coord
        
        mask = np.isin(self.ds.coords["sample"].values, sample_ids_to_include)
        return SpectraSet(self.ds.isel(obs=mask).copy())  # Use .copy() for branching/distinct processing

    def subset_by_obs(self, obs_indices_to_include: Sequence[int]) -> "SpectraSet":
        """
        Returns a new SpectraSet containing only observations at specified positional indices.

        Parameters
        ----------
        obs_indices_to_include : Sequence[int]
            A sequence of positional indices (0-based) for observations to include.

        Returns
        -------
        SpectraSet
            A new SpectraSet instance with the subsetted data.
        
        Raises
        ------
        IndexError
            If any index in `obs_indices_to_include` is out of bounds.
        """
        if not obs_indices_to_include:  # Handle empty sequence
            # Return empty SpectraSet preserving structure (coords, vars but 0 obs)
            empty_ds = self.ds.isel(obs=slice(0))  # Creates an empty slice
            return SpectraSet(empty_ds.copy())

        # Validate indices
        n_obs = len(self)
        max_index = n_obs - 1
        for idx in obs_indices_to_include:
            if idx < 0 or idx > max_index:
                raise IndexError(f"obs index {idx} out of bounds (0 to {max_index}).")

        # Direct positional selection using isel
        # Return subsetted SpectraSet
        # though isel works with positional indices directly.
        # However, to ensure we are robust if self.ds.obs is not simple 0..N-1 range:
        # obs_labels_to_select = self.ds.obs.isel(obs=list(obs_indices_to_include)).values
        # return SpectraSet(self.ds.sel(obs=obs_labels_to_select).copy())
        # Using isel is generally safer for positional selection if coords are not guaranteed to be simple ranges.
        return SpectraSet(self.ds.isel(obs=list(obs_indices_to_include)).copy())  # Use .copy()

    def __len__(self) -> int:
        """Return the total number of observations in the dataset."""
        return self.ds.sizes["obs"]

    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> "SpectraSet":
        """
        Enables slicing and indexing using familiar Python syntax.
        Returns a new SpectraSet instance for the selected observations.
        """
        if isinstance(key, (int, slice)):
            # Handle integer and slice directly via xarray's isel
            return SpectraSet(self.ds.isel(obs=key).copy())
        elif isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype == bool:
            # Handle boolean indexing directly
            if len(key) != len(self):
                raise IndexError(f"Boolean index length ({len(key)}) mismatch with dataset length ({len(self)}).")
            return SpectraSet(self.ds.isel(obs=key).copy())
        elif isinstance(key, (np.ndarray, list, tuple)):
            # Handle integer array indexing
            indices = np.atleast_1d(key).astype(int)
            return self.subset_by_obs(indices.tolist())
        else:
            raise TypeError(f"Invalid key type for slicing: {type(key)}")

    def __repr__(self) -> str:
        """Return a string representation of the SpectraSet."""
        n_obs = len(self)
        n_samples = len(self.sample_ids())
        
        source_info_parts = []
        # Use property self.feature_names, which returns Dict[str, List[str]]
        for src_name, feats in self.feature_names.items():
            source_info_parts.append(f"{src_name} ({len(feats)} features)")
        source_info = ", ".join(source_info_parts) if source_info_parts else "No spectral sources"

        target_info = "Yes" if "target" in self.ds else "No"
        
        # Count metadata fields
        metadata_count = len([str(k) for k in self.ds.data_vars if str(k).startswith("metadata_")])
        
        # Count prediction fields
        predictions_count = len([str(k) for k in self.ds.data_vars if str(k).startswith("pred_")])

        return (
            f"<SpectraSet: {n_obs} observations ({n_samples} samples)\\n"
            f"  Sources: {source_info}\\n"
            f"  Target: {target_info}\\\\n"
            f"  Metadata fields: {metadata_count}\\\\n"
            f"  Stored predictions: {predictions_count}>"
        )
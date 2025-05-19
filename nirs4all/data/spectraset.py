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
from typing import Any, Dict, Sequence, Optional, Union, Literal
from sklearn.preprocessing import LabelEncoder

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
        spectra: Dict[Any, Sequence[Sequence[np.ndarray]]],
        target: np.ndarray,
        *,
        metadata: Dict[str, np.ndarray] = None,
        splits: np.ndarray | None = None,
        folds: np.ndarray | None = None,
        groups: Dict[str, np.ndarray] = None,
        branch: np.ndarray | None = None,
        target_names: Sequence[Any] | None = None
    ) -> SpectraSet:
        """
        Build SpectraSet from nested dict:
            spectra[source_name][sample_idx] -> list of 1D arrays (augmentation) len f_source
        target: shape (n_samples,) or (n_samples, n_vars)
        """
        # 1) collect obs records across all sources
        obs_records: list[tuple[int, int, Any, np.ndarray]] = []
        for src, sample_list in spectra.items():
            for s_idx, aug_list in enumerate(sample_list):
                for a_idx, arr in enumerate(aug_list):
                    obs_records.append((s_idx, a_idx, src, arr))
        # 2) build index arrays
        sample_ids = [rec[0] for rec in obs_records]
        aug_ids = [rec[1] for rec in obs_records]
        source_ids = [rec[2] for rec in obs_records]
        # 3) MultiIndex for obs
        obs_idx = pd.MultiIndex.from_arrays(
            [sample_ids, aug_ids, source_ids],
            names=("sample", "augmentation", "source")
        )
        N_obs = len(obs_idx)
        # 4) build Dataset: one variable per source (outer join on feature dims)
        spectral_data_arrays = {}
        for src in spectra.keys():
            current_feat_len = 0
            if src in spectra and spectra[src]:
                for aug_list_for_sample in spectra[src]:
                    if aug_list_for_sample:
                        current_feat_len = len(aug_list_for_sample[0])
                        break
            feats = np.full((N_obs, current_feat_len), np.nan, dtype=float)
            for idx, rec_tuple in enumerate(obs_records):
                s_idx_rec, a_idx_rec, rec_src_from_tuple, arr_from_tuple = rec_tuple
                if rec_src_from_tuple == src:
                    if len(arr_from_tuple) == current_feat_len and current_feat_len > 0:
                        feats[idx, :] = arr_from_tuple
            da = xr.DataArray(
                feats,
                dims=("obs", "feature"),
                coords={"obs": obs_idx, "feature": np.arange(current_feat_len)}
            )
            spectral_data_arrays[f"spectra_{src}"] = da
        ds = xr.Dataset(spectral_data_arrays)
        # 5) target: repeat per obs via sample_ids
        targ_arr = np.asarray(target)
        if targ_arr.ndim == 1:
            targ_arr = targ_arr[:, None]
        n_vars = targ_arr.shape[1]
        targ_obs = np.vstack([targ_arr[s] for s in sample_ids])
        ds["target"] = xr.DataArray(
            targ_obs,
            dims=("obs", "variable"),
            coords={"obs": ds.coords["obs"],
                    "variable": target_names or np.arange(n_vars)}
        )
        # 6) metadata per obs
        if metadata:
            for key, arr in metadata.items():
                vals = [arr[s] for s in sample_ids]
                ds[f"metadata_{key}"] = xr.DataArray(
                    vals,
                    dims=("obs",),
                    coords={"obs": ds.coords["obs"]}
                )
        # 7) assign coords for split, folds, branch, groups via sample_ids
        if splits is not None:
            split_vals = [splits[s] for s in sample_ids]
            ds = ds.assign_coords(split=("obs", split_vals))
        if folds is not None:
            fold_vals = [folds[s] for s in sample_ids]
            ds = ds.assign_coords(fold_id=("obs", fold_vals))
        if branch is not None:
            br_vals = [branch[s] for s in sample_ids]
            ds = ds.assign_coords(branch=("obs", br_vals))
        if groups:
            for name, arr in groups.items():
                grp_vals = [arr[s] for s in sample_ids]
                ds = ds.assign_coords({f"group_id_{name}": ("obs", grp_vals)})
        return cls(ds)

    def X(
        self,
        *,
        split: Union[str, Sequence[str]] = None,
        fold_id: Union[int, Sequence[int]] = None,
        groups: Dict[str, Any] = None,
        branch: Union[str, Sequence[str]] = None,
        include_augmentations: bool = True,
        include_sources: Union[bool, Sequence[Any]] = True,
        flatten: Literal["concat","interlaced", None] = "concat"
    ) -> np.ndarray:
        """Return feature matrix (n_obs, sum_n_feats) with filters."""
        ds = self._filter_ds(split, fold_id, groups, branch)
        # select augment
        if not include_augmentations:
            ds = ds.sel(obs=ds.coords['augmentation']==0)
        # determine spectra variables to include
        vars_src = [k for k in ds.data_vars if k.startswith("spectra_")]
        if not include_sources:
            # pick first
            vars_src = vars_src[:1]
        else:
            if isinstance(include_sources, Sequence) and not isinstance(include_sources,str):
                vars_src = [f"spectra_{s}" for s in include_sources if f"spectra_{s}" in ds]
        # collect arrays
        arrs = []
        for var in vars_src:
            da = ds[var]
            arrs.append(da.values)
        # flatten if needed (obs already axis0)
        # horizontal concatenation
        return np.concatenate(arrs, axis=1)

    def y(
        self,
        *,
        split: Union[str, Sequence[str]] = None,
        fold_id: Union[int, Sequence[int]] = None,
        groups: Dict[str, Any] = None,
        branch: Union[str, Sequence[str]] = None
    ) -> np.ndarray:
        """Return target array (n_obs, n_vars) with filters."""
        ds = self._filter_ds(split, fold_id, groups, branch)
        arr = ds['target'].values
        # label encode if needed
        if arr.dtype.kind in ('U','S','O'):
            flat = arr.ravel()
            le = LabelEncoder().fit(flat)
            self._label_encoder = le
            arr = le.transform(flat).reshape(arr.shape)
        return arr

    def add_prediction(
        self,
        model_id: str,
        y_pred: np.ndarray,
        *,
        split: Optional[Union[str,Sequence[str]]] = None,
        fold_id: Optional[Union[int,Sequence[int]]] = None,
        groups: Optional[Dict[str,Any]] = None,
        branch: Optional[Union[str,Sequence[str]]] = None
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
        split: Optional[Union[str,Sequence[str]]] = None,
        fold_id: Optional[Union[int,Sequence[int]]] = None,
        groups: Optional[Dict[str,Any]] = None,
        branch: Optional[Union[str,Sequence[str]]] = None
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
        return self.subset_by_obs(np.atleast_1d(idx))

    def _filter_ds(
        self,
        split: Union[str, Sequence[str]] = None,
        fold_id: Union[int, Sequence[int]] = None,
        groups: Dict[str, Any] = None,
        branch: Union[str, Sequence[str]] = None
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

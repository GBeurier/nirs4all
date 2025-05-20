from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union, Literal

import numpy as np
import xarray as xr
from sklearn.preprocessing import LabelEncoder


def _expand(
    sample_level: Sequence[Any],
    obs_sample_ids: List[Any],
    lookup: Dict[Any, int],
) -> List[Any]:
    return [sample_level[lookup[sid]] for sid in obs_sample_ids]


class SpectraSet:
    def __init__(self, ds: xr.Dataset):
        self.ds = ds
        self._label_encoder: Optional[LabelEncoder] = None

    @classmethod
    def build(
        cls,
        *,
        spectra: Dict[str, List[List[np.ndarray]]],
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        sample_ids: Optional[List[Any]] = None,
        augmentation_ids: Optional[List[Any]] = None,
        feature_names: Optional[Dict[str, List[str]]] = None,
        target_names: Optional[List[str]] = None,
        splits: Optional[np.ndarray] = None,
        groups: Optional[Dict[str, np.ndarray]] = None,
        fold_id_param: Optional[np.ndarray] = None,
        branch_param: Optional[np.ndarray] = None,
    ) -> "SpectraSet":
        if not spectra:
            raise ValueError("'spectra' must contain at least one source.")

        first_src = next(iter(spectra))
        n_samples = len(spectra[first_src])

        if sample_ids is None:
            sample_ids = list(range(n_samples))
        if len(sample_ids) != n_samples:
            raise ValueError("sample_ids length mismatch with samples")
        lookup = {sid: i for i, sid in enumerate(sample_ids)}

        aug_counts = [len(spectra[first_src][i]) for i in range(n_samples)]
        for src, lst in spectra.items():
            if len(lst) != n_samples:
                raise ValueError(f"Source '{src}' length mismatch with '{first_src}'.")
            if [len(lst[i]) for i in range(n_samples)] != aug_counts:
                raise ValueError(
                    f"Augmentation counts for source '{src}' are not consistent across sources.")

        total_obs = sum(aug_counts)

        if augmentation_ids is None:
            augmentation_ids = [j for cnt in aug_counts for j in range(cnt)]
        if len(augmentation_ids) != total_obs:
            raise ValueError("augmentation_ids length must equal total observations")

        obs_sample_ids: List[Any] = [sample_ids[i] for i, cnt in enumerate(aug_counts) for _ in range(cnt)]

        obs_ids = np.arange(total_obs)  # simple integer index
        coords: Dict[str, Any] = {
            "obs": obs_ids,
            "sample": ("obs", np.array(obs_sample_ids)),
            "augmentation": ("obs", np.array(augmentation_ids)),
        }

        data_vars: Dict[str, xr.DataArray] = {}

        for src, lst in spectra.items():
            rows: List[np.ndarray] = []
            for i in range(n_samples):
                for aug in lst[i]:
                    arr = np.asarray(aug)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    if arr.ndim != 2 or arr.shape[0] != 1:
                        raise ValueError("Each augmentation must have shape (n_features,) or (1, n_features)")
                    rows.append(arr)
            matrix = np.vstack(rows)
            n_feat = matrix.shape[1]
            feat_dim = f"feature_{src}"
            feat_names = (
                feature_names[src] if feature_names and src in feature_names else [f"{src}_f{i}" for i in range(n_feat)]
            )
            if len(feat_names) != n_feat:
                raise ValueError(f"feature_names length mismatch for '{src}'")
            data_vars[f"spectra_{src}"] = xr.DataArray(
                matrix,
                dims=("obs", feat_dim),
                coords={"obs": obs_ids, feat_dim: feat_names},
            )

        if target is not None:
            targ = np.asarray(target)
            if targ.ndim == 1:
                targ = targ[:, None]
            n_vars = targ.shape[1]
            expanded_target = np.vstack(_expand(targ, obs_sample_ids, lookup))
            data_vars["target"] = xr.DataArray(
                expanded_target,
                dims=("obs", "variable"),
                coords={"obs": obs_ids, "variable": target_names or np.arange(n_vars)},
            )

        if metadata:
            for key, arr in metadata.items():
                arr = np.asarray(arr)
                if len(arr) != n_samples:
                    raise ValueError(f"Metadata '{key}' length mismatch with samples")
                expanded = _expand(arr, obs_sample_ids, lookup)
                data_vars[f"metadata_{key}"] = xr.DataArray(expanded, dims=("obs",), coords={"obs": obs_ids})

        ds = xr.Dataset(data_vars, coords=coords)

        def _add_coord(name: str, values: Optional[np.ndarray]):
            if values is None:
                return
            if len(values) != n_samples:
                raise ValueError(f"{name} length mismatch with samples")
            ds.coords[name] = ("obs", _expand(values, obs_sample_ids, lookup))

        _add_coord("split", splits)
        _add_coord("fold_id", fold_id_param)
        _add_coord("branch", branch_param)
        if groups:
            for gname, arr in groups.items():
                _add_coord(f"group_id_{gname}", arr)

        return cls(ds)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        """Return a dict ``{source_name: [feature_names...]}``."""
        out: Dict[str, List[str]] = {}
        for var in self.ds.data_vars:
            if var.startswith("spectra_"):
                src = var.replace("spectra_", "")
                out[src] = list(self.ds[var].coords[f"feature_{src}"].astype(str).values)
        return out

    def _filter_ds(
        self,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
    ) -> xr.Dataset:
        ds = self.ds
        mask = np.ones(ds.sizes["obs"], dtype=bool)

        def _apply(coord: str, vals):
            nonlocal mask
            if coord in ds.coords and vals is not None:
                mask &= np.isin(ds.coords[coord].values, np.atleast_1d(vals))

        _apply("split", split)
        _apply("fold_id", fold_id)
        _apply("branch", branch)
        if groups:
            for gname, v in groups.items():
                _apply(f"group_id_{gname}", v)

        # augment filtering (0 ==> original, >0 ==> augmented)
        if augment == "original":
            mask &= ds.coords["augmentation"].values == 0
        elif augment == "augmented":
            mask &= ds.coords["augmentation"].values > 0
        # "all" – no extra mask

        return ds.isel(obs=mask)

    def X(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        sources: Optional[Union[bool, str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
    ) -> np.ndarray:
        ds = self._filter_ds(split, fold_id, groups, branch, augment)
        all_src_vars = sorted(v for v in ds.data_vars if v.startswith("spectra_"))
        if sources is None or sources is True:
            chosen = all_src_vars
        elif sources is False:
            chosen = []
        elif isinstance(sources, str):
            chosen = [f"spectra_{sources}"] if f"spectra_{sources}" in ds.data_vars else []
        else:  # sequence
            chosen = [f"spectra_{s}" for s in sources if f"spectra_{s}" in ds.data_vars]

        if not chosen:
            return np.empty((ds.sizes["obs"], 0))
        return np.hstack([ds[v].values for v in chosen])

    def y(
        self,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
        augment: Literal["all", "original", "augmented"] = "all",
    ) -> np.ndarray:
        ds = self._filter_ds(split, fold_id, groups, branch, augment)
        arr = ds["target"].values
        if arr.dtype.kind in ("U", "S", "O"):
            le = LabelEncoder().fit(arr.ravel())
            self._label_encoder = le
            arr = le.transform(arr.ravel()).reshape(arr.shape)
        return arr

  
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
        name = f"pred_{model_id}"
        ds_f = self._filter_ds(split, fold_id, groups, branch)
        if ds_f.sizes["obs"] != len(y_pred):
            raise ValueError("Length of y_pred does not match filtered observations")
        full = np.full(self.ds.sizes["obs"], np.nan, dtype=float)
        mask = np.isin(self.ds.coords["obs"].values, ds_f.coords["obs"].values)
        full[mask] = y_pred
        self.ds[name] = xr.DataArray(full, dims=("obs",), coords={"obs": self.ds.coords["obs"]})

    def get_prediction(
        self,
        model_id: str,
        *,
        split: Optional[Union[str, Sequence[str]]] = None,
        fold_id: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[Dict[str, Any]] = None,
        branch: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        name = f"pred_{model_id}"
        if name not in self.ds:
            raise KeyError(f"No prediction stored for '{model_id}'")
        ds_f = self._filter_ds(split, fold_id, groups, branch)
        mask = np.isin(self.ds.coords["obs"].values, ds_f.coords["obs"].values)
        return self.ds[name].values[mask]

    def subset_by_samples(self, sample_idx: Sequence[Any]) -> SpectraSet:
        """Return a view containing only the given sample IDs (all augmentations)."""
        mask = np.isin(self.ds.coords['sample'].values, sample_idx)
        return SpectraSet(self.ds.isel(obs=mask))

    def subset_by_obs(self, obs_idx: Sequence[int]) -> SpectraSet:
        """Return a view containing only the given observation indices."""
        return SpectraSet(self.ds.isel(obs=obs_idx))

    def __array__(self) -> np.ndarray:
        """Allow numpy coercion to default to X(all)."""
        return self.X()

    def __len__(self) -> int:
        """Total number of observations in the set."""
        return self.ds.sizes['obs']

    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> SpectraSet:
        """Index along the obs axis (int, slice, or boolean mask)."""
        if isinstance(key, slice):
            indices = np.arange(len(self))[key]
        else:
            indices = np.atleast_1d(key)

        return self.subset_by_obs(list(indices))

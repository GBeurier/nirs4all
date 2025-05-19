Here’s the **final refined spec** for `SpectraSet`, **including** the ability to filter on **any auxiliary coordinate** (groups, folds, branches, etc.) in your `X`/`y` getters:

---

## 1. Core xarray Dataset

* **`spectra`**: DataArray dims
  (`sample`, `source`, `augmentation`, `feature`)
* **`target`**: DataArray dims
  (`sample`, `variable`)
* **auxiliary coords** on `sample`:

  * `split`
  * `fold_id`
  * **any number** of `group_id_<name>` (quantile, cluster, batch…)
  * `branch_id`
* **`metadata_<name>`**: extra DataArrays, dims `sample`
* **`predictions`**: DataArray dims (`sample`, `model`, `fold_id`)

---

## 2. Array‐like interface with full filtering

All getters accept **arbitrary coord‐based filters**:

```python
X(
    *,
    split: str | Sequence[str] = "train",
    fold_id: int | Sequence[int] | None = None,
    groups: dict[str, Scalar | Sequence[Scalar]] = None,
        # e.g. {"group_quantile": [0,2], "group_cluster": 1}
    branch: str | Sequence[str] = None,
    include_augmentations: bool = True,
    include_sources: bool = True,
    flatten: Literal["concat","interlaced",None] = "concat"
) -> np.ndarray
```

* **`split`**, **`fold_id`**, **`branch`**: filter on those coords
* **`groups`**: for each `group_id_<name>` you pass the allowed value(s)
* **all filters are combined** via logical AND on the `sample` axis
* After filtering **samples**, **all** their augmentations/sources follow unless disabled
* **Then** flatten `(augmentation, source, feature)` → obs/features per the `flatten` style
* **Zero-copy** view until you call `.values`

Convenience shorthands:

```python
X_train(   **same args except split fixed to "train" )
X_test(    split="test", ** )
X_val(     split="val",  ** )
```

---

```python
y(
    *,
    split: str | Sequence[str] = "train",
    fold_id: int | Sequence[int] | None = None,
    groups: dict[str, Scalar | Sequence[Scalar]] = None,
    branch: str | Sequence[str] = None
) -> np.ndarray
```

* Applies **exactly the same** sample‐level filtering as `X(...)`
* Returns shape `(n_obs,)` or `(n_obs, n_variables)`

---

## 3. Sub‐selection views (no filtering logic)

* **`subset_by_samples(sample_idx)`**: keep those samples & their augs/sources
* **`subset_by_obs(obs_idx)`**: pick specific obs rows after flattening

*Splitting, fold‐gen, group‐by are done externally*; you feed their indices into these views, or use the high-level filters above.

---

## 4. Predictions

```python
add_prediction(
    model_id: str,
    y_pred: np.ndarray,
    *,
    split: str = "test",
    fold_id: Optional[int] = None,
    groups: dict[str, ...] = None,
    branch: Optional[str] = None
) -> None
```

* Stores under `predictions` DataArray aligned to the **same filtered obs**
* Coordinates (`split`, `fold_id`, `group_*`, `branch`) recorded so you can
  retrieve by the same filters with:

```python
get_prediction(
    model_id: str,
    *,
    split: str | Sequence[str] = "test",
    fold_id: int | Sequence[int] | None = None,
    groups: dict[str, Scalar | Sequence[Scalar]] = None,
    branch: str | Sequence[str] = None
) -> np.ndarray
```

---

### 5. Categorical targets

* Internal encoder stored and applied in `y(...)`
* `inverse_transform(pred_array)` restores original labels

---

### 6. Minimal API skeleton

```python
class SpectraSet:
    ds: xr.Dataset

    @classmethod
    def build( spectra: np.ndarray, target: np.ndarray, *, metadata: dict ) -> SpectraSet: ...

    def X( *, split="train", fold_id=None, groups=None, branch=None,
           include_augmentations=True, include_sources=True,
           flatten="concat" ) -> np.ndarray: ...

    def X_train(self, **kwargs) -> np.ndarray: ...
    def X_test(self, **kwargs) -> np.ndarray: ...
    def X_val(self,  **kwargs) -> np.ndarray: ...

    def y( *, split="train", fold_id=None, groups=None, branch=None ) -> np.ndarray: ...
    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray: ...

    def subset_by_samples(self, sample_idx) -> SpectraSet: ...
    def subset_by_obs(self,    obs_idx)    -> SpectraSet: ...

    def add_prediction(self, model_id, y_pred, *, split, fold_id=None, groups=None, branch=None): ...
    def get_prediction(self, model_id, *, split, fold_id=None, groups=None, branch=None) -> np.ndarray: ...

    @property
    def xr(self) -> xr.Dataset: ...
    def __array__(self) -> np.ndarray: return self.X_train()  # default view
    def __len__(self) -> int: return self.X_train().shape[0]
    def __getitem__(self, idx) -> SpectraSet: return self.subset_by_obs(idx)
```

With this, **any** auxiliary group becomes a first‐class filter in your getters—no more missing samples or augmentations when you request `X(groups={"my_group": 3})`.

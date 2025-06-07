### **Spectroscopy Dataset API — Clean English Specification**

---

#### 1 · General Design Goals

* **Zero-copy everywhere.** All getters must return *views* into the underlying NumPy/Polars buffers; no duplicated data.
* **Immutable reads, explicit writes.** Data can only be modified through dedicated `update_*` or augmentation helpers.
* **Multi-source aware.** A single “dataset” may expose several *feature sources* (e.g., NIR, MIR, Raman), all aligned along the same sample index.
* **Transparent versioning.** Any transformation of features or targets is tracked by a **`processing` hash** so multiple versions can coexist and be queried side-by-side.
* **Fine-grained indexing.** Every table stores rich, typed index columns that let the user slice data along arbitrary dimensions (folds, partitions, augmentation lineage, pipeline branch, etc.).
* **Fast I/O & analytics.** Backed by **Polars** for columnar speed, and native **NumPy** arrays for model I/O. All save/load operations use Parquet.

---

#### 2 · Core Index Schema

| Column           | Description                                                                                                         | Type  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | ----- |
| **`row`**        | Physical row offset inside each feature-source matrix                                                               | `u32` |
| **`sample`**     | Logical sample identifier (unique per original *or* augmented sample)                                               | `u32` |
| **`origin`**     | ID of the *original* sample from which this row was derived ( = `sample` for pristine rows)                         | `u32` |
| **`partition`**  | High-level split label, currently `"train"` or `"test"` (others may be added)                                       | `str` |
| **`group`**      | Optional clustering/grouping code                                                                                   | `u32` |
| **`branch`**     | Pipeline branch identifier for complex, forked workflows                                                            | `u32` |
| **`processing`** | Stable hash (e.g., SHA-1/64-bit) of the exact transformation chain that produced the present feature/target version | `str` |

All data blocks reuse the relevant subset of this schema.

---

#### 3 · Features (`X`)

##### 3.1 Storage

* One **`numpy.ndarray`** per *source*:

  * Shape `(n_rows, n_dims_source)`
  * Rows are aligned across all sources (same `row` index).

##### 3.2 Layouts returned by getters

| Layout            | Returned object                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| `"2d"`            | Per-source `np.ndarray` where all *processing* variants of each sample are **concatenated along `axis=1`**.        |
| `"2d_interlaced"` | Same as `"2d"` but the concatenation order interleaves the different processing variants instead of grouping them. |
| `"3d"`            | Per-source `np.ndarray` with shape `(n_samples, n_variants, n_dims)`, i.e. **stack** variants along a new axis.    |
| `"3d_transpose"`  | Same as `"3d"` but the last two axes are swapped to ease certain deep-learning kernels.                            |

##### 3.3 Public API

```python
add_features(x_list: list[np.ndarray]) -> None
x(filter: dict[str, Any], layout: str = "2d",
  src_concat: bool = False) -> tuple[np.ndarray, ...]
get_indexed_features(filter, layout="2d", src_concat=False)
    -> tuple[tuple[np.ndarray, ...], polars.DataFrame]  # data, indexes
update_x(new_data: tuple[np.ndarray, ...],
         indexes: polars.DataFrame,
         processing_hash: str) -> None
augment_samples_x(spec: list[tuple[int, int]]) -> polars.DataFrame
augment_features_x(copies_per_sample: int) -> polars.DataFrame
update_indexes(indexes: polars.DataFrame,
               updates: dict[str, Any]) -> None
merge(sources: list[int]) -> None
split(ranges: list[tuple[int, int]]) -> None
x_exists(filter: dict[str, Any]) -> bool
```

*All getters (`x`, `get_indexed_features`) **must** return zero-copy views.  Updates only occur through `update_x` or the augmentation helpers.*

---

#### 4 · Targets (`Y`)

##### 4.1 Storage Columns

* `sample`, `processing`, user-defined **target columns** (any dtype).

##### 4.2 Reserved `processing` values

* `"raw"` – untouched ground-truth value as added
* `"regression"` – numeric continuous representation
* `"category"` – integer class index
* `"ohe"` – one-hot encoded vector

##### 4.3 Public API

```python
add_targets(y_df: polars.DataFrame,
            meta_df: polars.DataFrame | None = None) -> None
y(filter, processed: bool = True) -> np.ndarray
get_indexed_targets(filter) -> list[tuple[np.ndarray, polars.DataFrame]]
update_y(new_values: np.ndarray,
         indexes: polars.DataFrame,
         processing_hash: str) -> None
```

---

#### 5 · Metadata (`M`)

Simple key–value store for any non-target, non-feature sample attributes.

```python
add_meta(meta_df: polars.DataFrame) -> None
meta(filter) -> polars.DataFrame
```

Columns: `sample`, arbitrary metadata fields.

---

#### 6 · Folds

In-memory list of folds, each fold = `dict(train=[sample_ids], val=[sample_ids])`.

```python
set_folds(sklearn_splitter, *args, **kwargs) -> None
fold_generator() -> Iterator[dict[str, list[int]]]
get_data(layout="2d") -> Iterator[tuple[X_train, y_train, X_val, y_val]]
```

---

#### 7 · Predictions (`P`)

Polars table with columns

| Column       | Notes                                                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `sample`     | Target sample                                                                                                                 |
| `model`      | Model identifier                                                                                                              |
| `fold`       | Fold index                                                                                                                    |
| `repeat`     | Repeat index (for repeated CV)                                                                                                |
| `partition`  | `"train"`, `"val"`, `"test"`                                                                                                  |
| `processing` | Processing hash of the *training* target version; plus reserved `"regression"` / `"category"` for inverse-transformed outputs |
| `seed`       | Random seed of the model                                                                                                      |
| `metadata`   | Optional extra fields (timestamp, etc.)                                                                                       |

API mirrors `X`/`Y`:

```python
add_prediction(values: np.ndarray, meta: dict[str, Any]) -> None
prediction(filter, processed=True) -> np.ndarray
inverse_transform_prediction(transformers: list[Callable]) -> np.ndarray
```

---

#### 8 · Global Convenience Helpers

```python
add_data(features, targets, metadata) -> None
save(path: str, *, compression: str = "zstd") -> None
load(path: str) -> "SpectroDataset"
log(message: str, level: str = "INFO") -> None
print_summary() -> None
```

---

#### 9 · Implementation Notes & Guarantees

* **All slicing filters** use the column names in §2 and accept any Polars-style boolean expression or simple dict equality (as illustrated).
* **`processing` hashes** should be deterministic for a given ordered list of transformation names + parameters.
* **Augmentation helpers** automatically:

  * allocate new `sample` IDs (monotonically increasing),
  * copy the selected source arrays *by reference*,
  * set `origin` to the parent sample ID.
* **`merge` / `split`** operate *physically* on the stored NumPy matrices; after a `merge`, `src_concat=True` has no effect because only one source remains.
* **`src_concat=True`** (logical concatenation) never mutates data; it merely changes the view returned by getters.
* **Thread-safety:** read-only operations are safe under concurrency; mutation requires external locking by the caller.
* **Debuggability:** every public method logs its action (rows affected, hash used, etc.) through the central `log()` helper.
* **Persistence:** `save()` writes one Parquet file per Polars table plus NumPy `.npy` blobs for each feature source; `load()` re-wires zero-copy relationships on-the-fly.

---

This document captures **all functional requirements you provided, in English, without omission.** Let me know if you’d like clarifications, further structure (e.g. class skeletons), or code examples for specific methods!


---------
### **Lean Prototype Plan — “SpectroDataset” Sub-module**

> Scope reduced to the bare essentials so you can ship a working zero-copy dataset quickly.
> *Processing IDs (hashes) are **provided by the caller** in every `update_*` method — the dataset only stores them.*

---

## 1 Minimal Folder Layout  (`tree -L 2`)

```

nirs4all/                 ← sub-folder inside your bigger project
│
├─ dataset/                 ← implementation code
│   ├─ __init__.py
│   ├─ dataset.py        ← SpectroDataset façade
│   ├─ features.py       ← FeatureBlock + FeatureSource
│   ├─ targets.py        ← TargetBlock
│   ├─ metadata.py       ← MetadataBlock
│   ├─ folds.py          ← FoldsManager
│   └─ predictions.py    ← PredictionBlock
│
examples/                ← smoke tests here in a notebook (smoketest.ipynb) or in a file as described in phases.

```


## 2 Core Classes (only what you need)

| Class / File                             | Responsibility                                                                         | Key Methods                                                                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **`SpectroDataset`** (`dataset.py`)      | Orchestrator that owns four blocks, handles global invariants (row counts, sample IDs) | `add_data`, `x`, `y`, `save`, `load`, `print_summary`                                                                            |
| **`FeatureBlock`** (`features.py`)       | Manages *N* aligned NumPy sources + a Polars index                                     | `add_features`, `x`, `get_indexed_features`, `update_x`, `augment_samples_x`, `augment_features_x`, `merge`, `split`, `x_exists` |
| **`FeatureSource`** (`features.py`)      | Wrapper for a single `(rows, dims)` float array                                        | internal slicing helpers                                                                                                         |
| **`TargetBlock`** (`targets.py`)         | Polars table for target versions                                                       | `add_targets`, `y`, `get_indexed_targets`, `update_y`                                                                            |
| **`MetadataBlock`** (`metadata.py`)      | Key-value metadata store                                                               | `add_meta`, `meta`                                                                                                               |
| **`FoldsManager`** (`folds.py`)          | Simple list-of-dict folds + data generators                                            | `set_folds`, `get_data`                                                                                                          |
| **`PredictionBlock`** (`predictions.py`) | Polars table of model outputs                                                          | `add_prediction`, `prediction`, `inverse_transform_prediction`                                                                   |

**Processing IDs**
*All `update_*` APIs take a `processing_id: str` parameter; the caller generates it (hash, UUID, etc.). Internally we just store it.*


### **Phased Roadmap **


---

## Phase 0 · Bootstrap (1 day)

| #   | Task                                                                                             | File           |
| --- | ------------------------------------------------------------------------------------------------ | -------------- |
| 0-1 | Create folder skeleton under your parent project.                                                | *(filesystem)* |
| 0-2 | `pip install numpy polars` (only runtime deps).                                                  | —              |
| 0-3 | Add **`__init__.py`** in `spectrodataset/` and in every sub-folder so the package is importable. | `__init__.py`  |

**Smoke Test P0 – `smoke/step00_bootstrap.py`**

```python
import spectrodataset
print("SpectroDataset package imported OK")
```

Run ⇒ prints *OK*.

---

## Phase 1 · Feature Sources & “add” Path (1–2 days)

| #   | Task                                                                                                                                                                                                                                                                                                                                                                                                                        | File               | Notes |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----- |
| 1-1 | Implement **`FeatureSource`** class: stores a 2-D float `np.ndarray`; enforces immutable row count.                                                                                                                                                                                                                                                                                                                         | `dataset/features.py` |       |
| 1-2 | Implement **`FeatureBlock`** with minimal methods:<br>  • `__init__(self)` — empty list of sources & Polars index (columns: `row, sample, origin, partition, group, branch, processing`).<br>  • `add_features(x_list)` — append each array as a new `FeatureSource`; build index rows (`row` = 0…N-1, `sample` = same as row for now; `origin` =`sample`, partition=`"train"`, group = 0, branch = 0, processing=`"raw"`). | `dataset/features.py` |       |
| 1-3 | Add method `n_samples()` returning row count (assert equal across sources).                                                                                                                                                                                                                                                                                                                                                 | idem               |       |
| 1-4 | In **`dataset.py`**, create tiny façade:<br>`class SpectroDataset:` with attribute `features = FeatureBlock()`, plus pass-through `add_features`.                                                                                                                                                                                                                                                                           | `dataset/dataset.py`  |       |

**Smoke Test P1 – `smoke/step01_add_features.py`**

```python
import numpy as np
from spectrodataset.core.dataset import SpectroDataset

ds = SpectroDataset()
raman = np.random.rand(10, 500).astype("float32")
nirs  = np.random.rand(10, 2500).astype("float32")
ds.add_features([raman, nirs])          # two sources
print("sources:", len(ds.features.sources))
print("rows   :", ds.features.n_samples())
```

Expect: *sources = 2, rows = 10*.

---

## Phase 2 · Zero-Copy Slicing (2–3 days)

| #   | Task                                                                                                                                                                                                 | File               | Notes |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----- |
| 2-1 | Implement layout helpers inside `FeatureBlock`:<br>`_layout_2d`, `_layout_2d_interlaced`, `_layout_3d`, `_layout_3d_transpose`. Each returns **views**, never copies.                                | `dataset/features.py` |       |
| 2-2 | Implement public `x(filter: dict, layout="2d", src_concat=False)`.<br> • `filter` is a dict of `{column: value}` equality tests.<br> • Start with **simple exact-match** filtering (Polars `is_in`). | idem               |       |
| 2-3 | Implement `get_indexed_features(filter, layout, src_concat)` returning `(tuple[arrays], index_df_view)`.                                                                                             | idem               |       |
| 2-4 | Ensure all outputs share memory (`np.may_share_memory`).                                                                                                                                             | —                  |       |

**Smoke Test P2 – `smoke/step02_slice_features.py`**

```python
from spectrodataset.core.dataset import SpectroDataset
import numpy as np

ds = SpectroDataset()
ds.add_features([np.arange(20*2).reshape(20,2).astype("float32")])  # one source  (rows=20)
# Mark first 15 as train, rest test
ds.features.index_df = ds.features.index_df.with_columns(
    partition= (ds.features.index_df["row"] < 15).replace({True:"train", False:"test"})
)
x_train, = ds.x({"partition":"train"}, layout="2d")
assert np.may_share_memory(x_train, ds.features.sources[0].array)
print("train shape:", x_train.shape)   # (15,2)
```

---

## Phase 3 · Targets & Metadata (1–2 days)

| #   | Task                                                                                                                                                                                   | File               | Notes |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----- |
| 3-1 | **`TargetBlock`**: Polars table (`sample`, `targets` (list col) , `processing`). Implement:<br> • `add_targets(df)` – raw rows only.<br> • `y(filter, processed=True)` – stack arrays. | `dataset/targets.py`  |       |
| 3-2 | `update_y(np_array, index_df, processing_id)` – overwrite / append target version.                                                                                                     | idem               |       |
| 3-3 | **`MetadataBlock`**: simple Polars table (`sample`, …); CRUD methods.                                                                                                                  | `dataset/metadata.py` |       |
| 3-4 | Wire these blocks into `SpectroDataset` (attributes + pass-through).                                                                                                                   | `dataset/dataset.py`  |       |

**Smoke Test P3 – `smoke/step03_targets_meta.py`**

```python
from spectrodataset.core.dataset import SpectroDataset
import polars as pl
import numpy as np

ds = SpectroDataset()
ds.add_features([np.zeros((5,3), dtype="float32")])
# targets
raw_df = pl.DataFrame({"sample":[0,1,2,3,4], "targets":[[0.1],[0.2],[0.3],[0.4],[0.5]], "processing":["raw"]*5})
ds.targets.add_targets(raw_df)
# create a processed version (regression) for first 3 samples
idx = ds.targets.table.filter(pl.col("sample")<3)
vals = np.array([[10.],[20.],[30.]], dtype="float32")
ds.targets.update_y(vals, idx, processing_id="regression")
print(ds.targets.table)
```

---

## Phase 4 · Folds Manager (1 day)

| #   | Task                                                                                                    | File              | Notes |
| --- | ------------------------------------------------------------------------------------------------------- | ----------------- | ----- |
| 4-1 | Implement **`FoldsManager`** storing `folds: list[dict(train=list[int], val=list[int])]`.               | `dataset/folds.py`   |       |
| 4-2 | `set_folds(folds_iterable)` – accept any iterable of `(train_idx, val_idx)` tuples.                     | idem              |       |
| 4-3 | `get_data(dataset, layout)` generator yielding `x_train, y_train, x_val, y_val` using dataset’s blocks. | idem              |       |
| 4-4 | Plug into `SpectroDataset` (`.folds`).                                                                  | `dataset/dataset.py` |       |

**Smoke Test P4 – `smoke/step04_folds.py`**

```python
import numpy as np, polars as pl
from spectrodataset.core.dataset import SpectroDataset
from sklearn.model_selection import KFold

ds = SpectroDataset()
ds.add_features([np.random.rand(12,2).astype("float32")])
raw = pl.DataFrame({"sample":range(12), "targets":[[i] for i in range(12)], "processing":["raw"]*12})
ds.targets.add_targets(raw)

kf = KFold(n_splits=3, shuffle=False)
ds.folds.set_folds(kf.split(np.arange(12)))

for fold in ds.folds.get_data(ds, layout="2d"):
    x_tr, y_tr, x_val, y_val = fold
    print(x_tr[0].shape, x_val[0].shape)
```

---

## Phase 5 · Predictions Block (1 day)

| #   | Task                                                                                                            | File                  | Notes |
| --- | --------------------------------------------------------------------------------------------------------------- | --------------------- | ----- |
| 5-1 | **`PredictionBlock`** with Polars table (`sample, model, fold, repeat, partition, processing, seed, preds`).    | `dataset/predictions.py` |       |
| 5-2 | `add_prediction(np_arr, meta_dict)` – append rows.                                                              | idem                  |       |
| 5-3 | `prediction(filter)` – fetch stacked array view.                                                                | idem                  |       |
| 5-4 | `inverse_transform_prediction(transformers)` – call external functions on fetched arr, return transformed view. | idem                  |       |
| 5-5 | Attach as `dataset.predictions`.                                                                                | `dataset/dataset.py`     |       |

**Smoke Test P5 – `smoke/step05_predictions.py`**

```python
from spectrodataset.core.dataset import SpectroDataset
import numpy as np, polars as pl

ds = SpectroDataset()
ds.add_features([np.random.rand(3,4).astype("float32")])
ds.targets.add_targets(pl.DataFrame({"sample":[0,1,2], "targets":[[0.],[1.],[0.]], "processing":["raw"]*3}))

preds = np.array([[0.2],[0.8],[0.1]], dtype="float32")
meta  = {"model":"logreg", "fold":0, "repeat":0, "partition":"val",
         "processing":"regression", "seed":42}
ds.predictions.add_prediction(preds, meta)
print(ds.predictions.table)
```

---

## Phase 6 · Persistence (2 days)

| #   | Task                                                                                                                                                                                                                                                                                              | File              | Notes |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----- |
| 6-1 | In `dataset/features.py` add `dump_numpy()` / `load_numpy(path)` helpers using `np.save` / `np.load` with `allow_pickle=False`.                                                                                                                                                                      | —                 |       |
| 6-2 | In **`io.py`** (create file) implement:<br>`save(ds, path)` – create folder:<br>  • `features_src#.npy` files<br>  • `targets.parquet`, `metadata.parquet`, `predictions.parquet`, `index.parquet`, `folds.json`.<br>`load(path)` – recreate object using `np.load(mmap_mode="r")` for zero-copy. | `dataset/io.py`      |       |
| 6-3 | Add `SpectroDataset.save / load` thin wrappers.                                                                                                                                                                                                                                                   | `dataset/dataset.py` |       |

**Smoke Test P6 – extend `smoke/step05_predictions.py`**

```python
ds.save("/tmp/spectro_proto")
ds2 = ds.load("/tmp/spectro_proto")
x1, = ds.x({}, layout="2d")
x2, = ds2.x({}, layout="2d")
assert np.array_equal(x1, x2)
print("Save/Load round-trip OK")
```

---

## Phase 7 · Integration & Summary (½ day)

| #   | Task                                                                                                         | File |
| --- | ------------------------------------------------------------------------------------------------------------ | ---- |
| 7-1 | Implement `print_summary()` in `dataset.py` (counts, dimensions, #sources, #targets versions, #predictions). |      |
| 7-2 | Create `smoke/step07_summary.py` calling it after full pipeline creation.                                    |      |

---
# SpectraDataset — Detailed Design Specification

*Last updated: 25 May 2025*

---

## 1 ► Purpose

The **`SpectraDataset`** class is a light‑weight, in‑memory data‐container that tracks **spectroscopic samples** (NIR, MIR, Raman, …) as they flow through an ML pipeline.  It binds four logically separate artefacts under a unified API:

| Artefact                  | Storage                                   | Granularity                                                 | Mutability    |
| ------------------------- | ----------------------------------------- | ----------------------------------------------------------- | ------------- |
| **Features** (X)          | Polars `DataFrame`                        | **row ≈ one spectrum**                                      | *mutable*     |
| **Labels & metadata** (Y) | Polars `DataFrame`                        | **row ≈ one *****sample***** (unique ****`sample_id`****)** | *append‑only* |
| **Predictions**           | Polars `DataFrame`                        | **row ≈ one prediction**                                    | *append‑only* |
| **Folds**                 | `list[list[int]]` *or* Polars `DataFrame` | **indices of samples**                                      | *replaceable* |

All four objects are keyed by a **canonical multi‑index** described below.

---

## 2 ► Canonical Multi‑Index

Each spectrum (row in **Features**) is uniquely identified by the tuple:

```
(sample_id,          # UUID4 str | int – permanent ID of a biological / physical sample
 origin_id,          # same domain as sample_id – None ⇒ raw spectrum, else ID of parent sample
 type,               # str – 'nirs', 'mir', 'raman', …
 transformation_id,  # str – 128‑bit hex hash of the *ordered* transformation pipeline
 set_id,             # str – 'train', 'test', 'val', or arbitrary split label
 branch,             # int – 0 for the main trunk, ≥1 for every fork
 group)              # str|int – optional cluster/grouping label; may be None
```

### 2.1 Reserved Column Names

```
row_id              # int, internal monotonically‑increasing primary key (features‑table only)
model_id            # str, predictions‑table only
fold_id             # int, predictions‑table only
seed                # int, predictions‑table only
stack_index         # int, predictions‑table only
predicted_at        # datetime[μs], predictions‑table only
```

User code **must not** create or overwrite the above columns.

---

## 3 ► Tables & Schemas

### 3.1 Features Table (`self.features`)

Polars `DataFrame` with **one row per spectrum**.

| Column             | dtype           | Description                     |
| ------------------ | --------------- | ------------------------------- |
| `row_id`           | `Int64`         | Internal PK (autoincrement)     |
| Multi‑index fields | see §2          | Joint unique constraint         |
| `spectrum`         | `List(Float64)` | Unpadded vector of intensities  |
| any user tags      | any             | Added via `add_tag` / `set_tag` |

**Constraints**

* `(sample_id, origin_id, type, transformation_id, branch)` *must be unique*.
* `row_id` is the join key into predictions.

### 3.2 Labels / Metadata Table (`self.labels`)

Exactly \*\*one row per \*\****sample*** (unique `sample_id`).

| Column       | dtype                              | Description           |
| ------------ | ---------------------------------- | --------------------- |
| `sample_id`  | same domain as §2                  | PK                    |
| `target`     | `Float64` or `Utf8` or categorical | The supervised target |
| any metadata | any                                | free‑form             |

`origin_id`, `type`, `transformation_id`, etc. **do not appear** here because the label is sample‑level.

### 3.3 Predictions Table (`self.results`)

One row per prediction event.

| Column         | dtype          | Description                 |
| -------------- | -------------- | --------------------------- |
| `row_id`       | `Int64`        | FK → Features.row\_id       |
| `model_id`     | `Utf8`         | Arbitrary model identifier  |
| `fold_id`      | `Int8`         | Outer CV fold index         |
| `seed`         | `Int32`        | RNG seed to reproduce run   |
| `stack_index`  | `Int8`         | For stacked ensembles       |
| `branch`       | `Int8`         | Copied from Features.branch |
| `prediction`   | `Float64`      | Predicted value             |
| `predicted_at` | `Datetime[μs]` | Timestamp                   |
| any extra cols | any            | e.g. variance, class‑prob   |

Composite index **(row\_id, model\_id, fold\_id, seed, stack\_index)** should be unique.

### 3.4 Folds (`self.folds`)

Two interchangeable representations:

* `list[list[int]]` – outer list length = n\_folds; each inner list stores `sample_id`s belonging to that fold’s validation split.
* Polars `DataFrame` – columns: `sample_id`, `fold_id`.

Only one representation is kept in RAM at a time.  Helper mutators convert between the two.

---

## 4 ► Public API

Below are **method signatures, docstrings and rules** to be implemented.  Names follow Python PEP‑8.

### 4.1 Constructor

```python
__init__(self,
         features: pl.DataFrame | None = None,
         labels: pl.DataFrame | None = None,
         folds:  list[list[int]] | pl.DataFrame | None = None,
         *,
         validate: bool = True) -> None
```

* When `validate=True`, invariants in §3 are checked and `ValueError` is raised on failure.

### 4.2 Invariants Validator

```python
validate(self) -> None
```

Ensures uniqueness constraints, dtypes, and join keys.  Called automatically after every mutating op when `self._autovalidate` is `True`.

### 4.3 Sample Ingestion

```python
add_samples(self,
            spectra: Sequence[Sequence[float]],
            sample_ids: Sequence[int | str] | None = None,
            *,
            type: str = "nirs",
            set_id: str = "train",
            branch: int = 0,
            group: str | int | None = None,
            targets: Sequence[Any] | None = None,
            metadata: Mapping[str, Sequence[Any]] | None = None) -> list[int]
```

* If `sample_ids is None`, they are auto‑generated sequential ints.
* Returns the list of **row\_id**s inserted.
* `origin_id=None`, `transformation_id=None` for fresh samples.
* **Side‑effects**: appends to `features`; updates/creates corresponding rows in `labels` if targets/metadata are provided.

### 4.4 Branch Forking

```python
fork_branch(self,
            new_branch: int | None = None,
            source_branch: int = 0,
            copy_predictions: bool = False) -> int
```

* Copies *all rows* whose `branch == source_branch` into `features` with the `branch` field replaced by `new_branch` (or next available int).
* Deep‑copies spectra lists.  `row_id`s are **newly assigned**.
* If `copy_predictions=True`, predictions associated with the copied rows are also forked.
* Returns the numeric id of the new branch.

### 4.5 Sample Augmentation (new **samples**)

```python
augment_samples(self,
                selector: Mapping[str, Any] | None = None,
                transformer: Callable[[np.ndarray], np.ndarray],
                *,
                new_type: str | None = None,
                new_group: str | int | None = None) -> list[int]
```

* Selects a subset of spectra via `selector` (same semantics as `_select`).
* For every selected spectrum:

  * Applies `transformer(vec)` → `vec_aug`.
  * Inserts a **new row** with:

    * `sample_id = self._next_sample_id()` ← brand‑new id ( ≠ parent).
    * `origin_id = parent.sample_id`.
    * `type = new_type if new_type else parent.type`.
    * `transformation_id = hash(parent.transformation_id, transformer)`.
    * `branch, set_id, group` inherited unless overwritten.
* Returns list of new `row_id`s.

### 4.6 Feature Augmentation (same **sample id**)

```python
augment_features(self,
                selector: Mapping[str, Any] | None = None,
                transformer: Callable[[np.ndarray], np.ndarray]) -> list[int]
```

* Similar to §4.5 but **preserves** `sample_id`.  Enforces uniqueness of the extended multi‑index.

### 4.7 Transformation Hash Helper

```python
compute_hash(prev_hash: str | None,
             transformer: Any) -> str
```

Deterministically produces a 128‑bit hex digest based on:

* `prev_hash or "RAW"`
* fully‑qualified class name of `transformer`
* repr of its hyper‑parameters

### 4.8 Tagging Utilities

```python
add_tag(self, name: str, default: Any = None, *, table: str = "features") -> None
set_tag(self, name: str, value: Any | Sequence[Any], selector: Mapping[str, Any] | None = None, *, table: str = "features") -> None
```

* `add_tag` physically inserts a new column with uniform `default`.
* `set_tag` updates an existing column for rows matching `selector`.

### 4.9 Selection & Retrieval

```python
select(self, selector: Mapping[str, Any] | None = None, *, columns: Sequence[str] | None = None) -> pl.DataFrame
X(self, selector: Mapping[str, Any] | None = None, *, pad: bool = False, pad_value: float = np.nan) -> np.ndarray
Y(self, selector: Mapping[str, Any] | None = None) -> np.ndarray
```

* `selector` accepts any subset of column → scalar|collection.
* Table joins are implicit: selecting on label fields auto‑joins `features` & `labels`.

### 4.10 Predictions API

```python
add_predictions(self,
                model_id: str,
                fold_id: int,
                seed: int,
                preds: Sequence[float],
                *,
                selector: Mapping[str, Any] | None = None,
                stack_index: int = 0,
                extra_cols: Mapping[str, Sequence[Any]] | None = None) -> None
get_predictions(self, selector: Mapping[str, Any] | None = None) -> pl.DataFrame
```

### 4.11 Fold Helpers

```python
set_folds(self, splitter: Any, *, n_splits: int, random_state: int | None = None) -> None
get_fold(self, fold_id: int, *, split: str = "train") -> pl.DataFrame
```

`splitter` follows the scikit‑learn splitter API.

### 4.12 Serialization

```python
save(self, path: str, *, fmt: Literal["arrow", "parquet"] = "arrow") -> None
load(cls, path: str, *, fmt: Literal["arrow", "parquet"] = "arrow") -> "SpectraDataset"
```

Stores **three** Arrow/Parquet files: features, labels, results.  Folds are embedded in the features file metadata.

### 4.13 Repr & Len

`__repr__` returns

```
SpectraDataset(n_rows=<len(self)>, n_samples=<len(self.labels)>, n_branches=<…>)
```

---

## 5 ► Internal Helpers

* `_mask(**filters)` – vectorised boolean expression builder.
* `_next_row_id()` – autoincrement counter.
* `_next_sample_id()` – autoincrement if sample IDs are generated internally.
* `_autovalidate` flag to switch expensive checks off in production.

---

## 6 ► Error Handling & Edge Cases

* Any attempt to violate uniqueness constraints must raise `ValueError` with a human‑readable explanation.
* Methods must be **idempotent** where reasonable; e.g. adding identical predictions twice should error, not silently duplicate.
* Hash collisions are considered impossible within reasonable probability (<1e‑36).  If detected, raise `RuntimeError`.

---

## 7 ► Performance Notes

* Store spectra as `List(Float64)` – fastest for variable length.
* Bulk operations preferred; all mutators accept vectorised inputs.
* Branch forking and augmentation avoid deep copy of unchanged polars columns using `with_columns` and `pl.concat_row`.
* When joining large frames, pre‑sort on join keys to exploit hash join internals.

---

## 8 ► Unit‑Test Checklist

1. **Ingestion** – raw add, auto sample\_id, label merge.
2. **Branch fork** – uniqueness, copy count.
3. **Augment samples** – new sample\_id, origin link.
4. **Augment features** – duplicate sample\_id, new transformation\_id.
5. **Tagging** – creation, overwrite, selector filtering.
6. **Predictions** – mis‑sized preds, duplication error.
7. **Folds** – scikit splitter integration, retrieval.
8. **Serialization** – round‑trip equality.

---

## 9 ► Future Extensions (non‑blocking)

* Lazy out‑of‑core backend (Arrow Dataset).
* Dask/Polars lazy execution paths.
* Differential privacy noise injectors.
* Native PyTorch / TensorFlow `Dataset` views.

---

> **End of Specification**

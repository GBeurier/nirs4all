"""Host-dataset materialization for the dag-ml backend.

Accept ANY input ``nirs4all.run()`` takes (path / config / DatasetConfigs / live SpectroDataset /
array / tuple), build the host ``SpectroDataset``, and resolve how the subprocess adapter
re-materializes the byte-identical dataset (reloadable path vs. pickle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nirs4all.data.config import DatasetConfigs

from .identity import mint_identity


def _materialize_dataset(dataset: Any) -> Any:
    """Materialize the host ``SpectroDataset`` from ANY input ``nirs4all.run()`` accepts.

    ``run_via_dagml`` must accept the SAME dataset arg as legacy ``run()``: a path / config dict /
    JSON-YAML file (via :class:`DatasetConfigs`), an already-built :class:`DatasetConfigs`, a live
    :class:`SpectroDataset` (e.g. ``nirs4all.generate.regression(...)``), a single feature array, or
    an ``(X, y[, partition_info])`` tuple. ``DatasetConfigs`` SILENTLY SKIPS the in-memory inputs
    (its normalizer returns ``None`` for a ``SpectroDataset`` / ``ndarray`` / ``tuple``), leaving
    ``configs`` empty so ``get_dataset_at(0)`` raises a misleading ``IndexError`` — so those inputs
    are wrapped HERE with the SAME normalization the legacy orchestrator uses
    (:meth:`PipelineOrchestrator._wrap_dataset`), never handed to ``DatasetConfigs`` raw.

    A list of datasets is rejected loud: the dag-ml backend runs ONE dataset (the cartesian product
    over datasets stays a legacy-orchestrator concern).
    """
    from nirs4all.data.dataset import SpectroDataset

    if isinstance(dataset, DatasetConfigs):
        return dataset.get_dataset_at(0)
    if isinstance(dataset, SpectroDataset):
        return dataset
    if isinstance(dataset, list):
        raise NotImplementedError("engine='dag-ml' runs a single dataset; pass one dataset, not a list of datasets")
    if isinstance(dataset, (np.ndarray, tuple)):
        return _wrap_in_memory_arrays(dataset)
    return DatasetConfigs(dataset).get_dataset_at(0)


def _wrap_in_memory_arrays(dataset: np.ndarray | tuple) -> Any:
    """Build a ``SpectroDataset`` from a feature array or ``(X, y[, partition_info])`` tuple.

    Mirrors the legacy :meth:`PipelineOrchestrator._wrap_dataset` array/tuple branch exactly (public
    ``add_samples`` / ``add_targets``), so the dag-ml backend materializes the IDENTICAL dataset the
    legacy engine would for the same input. A bare array goes to the ``test`` partition (prediction
    shape); an ``(X, y)`` tuple goes to ``train`` with targets; an optional third element is a
    partition map (``{"train": 80}`` / explicit slices / index lists).
    """
    from nirs4all.data.dataset import SpectroDataset

    spectro = SpectroDataset(name="array_dataset")
    if isinstance(dataset, np.ndarray):
        spectro.add_samples(dataset, indexes={"partition": "test"})
        return spectro

    x = dataset[0]
    y = dataset[1] if len(dataset) > 1 else None
    partition_info = dataset[2] if len(dataset) > 2 else None
    if partition_info is None:
        spectro.add_samples(x, indexes={"partition": "train"})
        if y is not None:
            spectro.add_targets(y)
    else:
        _split_and_add_in_memory(spectro, x, y, partition_info)
    return spectro


def _split_and_add_in_memory(spectro: Any, x: np.ndarray, y: np.ndarray | None, partition_info: dict[str, Any]) -> None:
    """Add ``(X, y)`` to ``spectro`` per ``partition_info`` — the legacy ``_split_and_add_data`` rules.

    ``partition_info`` values: an int (``{"train": 80}`` = first N samples), a ``slice``/list/array of
    indices. If only ``train`` is given, the remaining rows become ``test``. Mirrors
    :meth:`PipelineOrchestrator._split_and_add_data` so the materialized dataset is identical.
    """
    n_samples = x.shape[0]
    partition_indices: dict[str, slice | list[Any] | np.ndarray] = {}
    for partition_name, partition_spec in partition_info.items():
        if isinstance(partition_spec, int):
            partition_indices[partition_name] = slice(0, partition_spec)
        elif isinstance(partition_spec, (slice, list, np.ndarray)):
            partition_indices[partition_name] = partition_spec
        else:
            raise ValueError(f"Invalid partition spec for '{partition_name}': {partition_spec}")

    if "train" in partition_indices and "test" not in partition_indices:
        train_spec = partition_indices["train"]
        if isinstance(train_spec, slice):
            train_end = train_spec.stop if train_spec.stop is not None else train_spec.start
        else:
            train_array = np.array(train_spec)
            train_end = int(train_array.max()) + 1 if len(train_array) > 0 else 0
        if train_end < n_samples:
            partition_indices["test"] = slice(train_end, n_samples)

    for partition_name, indices_spec in partition_indices.items():
        x_partition = x[indices_spec]
        y_partition = y[indices_spec] if y is not None else None
        if len(x_partition) > 0:
            spectro.add_samples(x_partition, indexes={"partition": partition_name})
            if y_partition is not None:
                spectro.add_targets(y_partition)


def _dataset_inputs(dataset: Any, spectro: Any, base_dir: Path) -> tuple[str, str | None]:
    """Resolve how the adapter materializes ``spectro``: ``(dataset_path, dataset_pickle)``.

    The adapter (in a subprocess) must materialize the EXACT dataset the host built the
    envelope/identity from — the wire ids are ``f"{content_hash()}.s{sample_int}"``, so any drift in
    the re-loaded dataset's content hash or sample order makes the run wrong (host-audit H-P1-1).

    Prefer the cheap path re-load when it is provably faithful: for a file-path / config input that
    exposes a reloadable path, re-load ``DatasetConfigs(path).get_dataset_at(0)`` IN-PROCESS and keep
    the path ONLY if its identity fingerprint matches the host's. If the input is in-memory (a
    ``SpectroDataset`` / array / tuple, with no reloadable path) OR the path re-load diverges, PICKLE
    the host ``spectro`` and ship it via the existing ``N4A_DAGML_DATASET_PICKLE`` channel — the
    adapter then loads the byte-identical dataset (the same mechanism augmentation/rep-fusion use).
    """
    path = _reloadable_path(dataset)
    if path is not None:
        try:
            reloaded = DatasetConfigs(path).get_dataset_at(0)
            if mint_identity(reloaded).fingerprint == mint_identity(spectro).fingerprint:
                return path, None
        except (IndexError, ValueError, OSError):
            pass  # path not faithfully reloadable — fall through to pickle

    base_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = base_dir / "host_dataset.pkl"
    import pickle

    pickle_path.write_bytes(pickle.dumps(spectro))
    return (path or str(pickle_path)), str(pickle_path)


def _reloadable_path(dataset: Any) -> str | None:
    """A filesystem path the adapter can reload via ``DatasetConfigs(path)``, or ``None``.

    A raw path string / ``Path`` is reloadable as-is; an already-built :class:`DatasetConfigs` (e.g.
    ``DatasetConfigs(folder, repetition="col")``) exposes its source files in ``configs[0]`` — the
    folder-style dataset reloads from the common parent directory of the ``train_x`` file. In-memory
    inputs (``SpectroDataset`` / array / tuple) have no reloadable path and return ``None``.
    """
    if isinstance(dataset, (str, Path)):
        return str(dataset)
    if isinstance(dataset, DatasetConfigs):
        config_dict = dataset.configs[0][0]
        train_x = config_dict.get("train_x")
        if isinstance(train_x, str) and train_x:
            return str(Path(train_x).parent)
    return None

"""Scientific dataset previews for thin application hosts (no jobs or stores).

CSV/Parquet assembly uses the installed native IO owner. Other formats are
selected before loading through the existing library loaders; this is an
explicit format route, not retry/fallback after a native error. Presentation
arrays always accompany textual/numeric summaries and keep original axes.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from nirs4all.api.dataset_documents import normalize_dataset_document

_NATIVE_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".csv.zip", ".parquet", ".pq")
_DATA_KEYS = frozenset(f"{partition}_{role}" for partition in ("train", "test") for role in ("x", "y", "group"))


def load_prediction_file(
    path: str | Path, *, params: dict[str, Any] | None = None,
    load_limits: dict[str, int] | None = None, max_input_bytes: int = 512 * 1024 * 1024,
) -> tuple[Any, dict[str, Any], list[str] | None]:
    """Load an authorized prediction upload through its existing reader owner.

    All numeric columns remain features in file order; the first string column
    supplies optional display labels, never scientific sample identifiers.
    CSV/Parquet use an explicit IO DatasetSpec, not target-column inference.
    Excel uses the registered Excel loader with typed cells preserved. No
    training, target synthesis, parser retry or implicit legacy execution occurs.

    The host must authorize the path and provide confirmed CSV parsing options
    (for example from IO's bounded text detector and a header checkbox). Without
    overrides IO's defaults apply, including a header and semicolon delimiter.
    Native limits are rejected for Excel because its loader cannot guarantee
    them; its separate raw-file admission budget remains effective.
    """
    path = Path(path).resolve(strict=True)
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be a positive integer")
    if not path.is_file():
        raise ValueError("Prediction input must be a regular file")
    size = path.stat().st_size
    if size > max_input_bytes:
        raise ValueError("Prediction file exceeds raw input byte budget")
    if params is not None and not isinstance(params, dict):
        raise TypeError("params must be a mapping")
    options = dict(params or {})
    reader: dict[str, Any] = {"raw_input_bytes": size, "max_input_bytes": max_input_bytes}
    if str(path).lower().endswith(_NATIVE_SUFFIXES):
        nio = importlib.import_module("nirs4all_io")
        importlib.import_module("nirs4all_io._native")
        spec = {"name": path.stem, "params": options, "sources": [{
            "id": "upload", "role": "mixed", "input": str(path), "strict_columns": False,
            "columns": {"features": {"dtype": "numeric"}, "metadata": {"dtype": "string"}},
        }]}
        dataset = nio.load(spec, target="spectrodataset", limits=load_limits)
        reader.update({"backend": "nirs4all-io.native", "version": nio.__version__,
                       "native_load_limits_applied": True, "load_limits": load_limits or "native_defaults"})
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        if load_limits is not None:
            raise ValueError("Native LoadLimits cannot be guaranteed by non-native format loaders")
        from nirs4all.data.dataset import SpectroDataset
        from nirs4all.data.loaders.excel_loader import ExcelLoader
        if options.get("categorical_mode", "preserve") != "preserve":
            raise ValueError("Prediction uploads require categorical_mode='preserve' for display labels")
        options["categorical_mode"] = "preserve"
        result = ExcelLoader().load(path, **options)
        if result.report.get("error") or result.data is None:
            raise ValueError(result.report.get("error") or "Excel reader returned no data")
        frame = result.data
        numeric = frame.select_dtypes(include=["number"])
        if numeric.shape[1] == 0:
            raise ValueError("Prediction input contains no numeric feature columns")
        dataset = SpectroDataset(path.stem)
        dataset.add_samples(numeric.to_numpy(), {"partition": "train"},
                            headers=[str(column) for column in numeric.columns], header_unit=result.header_unit)
        labels = frame.select_dtypes(include=["object", "string"])
        if labels.shape[1]:
            dataset.add_metadata(labels, headers=[str(column) for column in labels.columns])
        reader.update({"backend": "nirs4all.loaders", "loader": ExcelLoader.name,
                       "native_load_limits_applied": False, "load_limits": None,
                       "limitations": ["native decompression/shape budgets do not cover this format route"],
                       "loading_report": result.report})
    else:
        raise ValueError("Prediction file must be CSV, Parquet or Excel")
    if np.asarray(dataset.x({"partition": "train"}, layout="2d")).shape[1] == 0:
        raise ValueError("Prediction input contains no numeric feature columns")
    columns = dataset.metadata_columns
    sample_labels = [str(value) for value in dataset.metadata_column(columns[0])] if columns else None
    reader["sample_labels_are_identifiers"] = False
    return dataset, reader, sample_labels


def _references(value: Any, is_path: bool = False) -> list[Path]:
    if isinstance(value, str) and is_path:
        return [Path(value)]
    if isinstance(value, list):
        return [path for item in value for path in _references(item, is_path)]
    if isinstance(value, dict):
        return [path for key, item in value.items() for path in _references(item, key in _DATA_KEYS or key in {"path", "input", "file", "index_file", "folds"})]
    return []


def load_dataset_for_analysis(
    config: dict[str, Any], *, load_limits: dict[str, int] | None = None,
    max_input_bytes: int = 512 * 1024 * 1024,
) -> tuple[Any, dict[str, Any]]:
    """Load one authorized explicit config and return dataset/reader evidence.

    Paths are normalized but must be authorized by the host before this call.
    Native CSV/Parquet errors propagate; other formats are selected before
    loading through the existing registered library loaders. The returned
    dataset retains all sources, partitions, targets and metadata.
    """
    config = normalize_dataset_document(config)
    paths = list(dict.fromkeys(_references(config)))
    if not paths:
        raise ValueError("Dataset inspection requires explicit input file references")
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be a positive integer")
    total = 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Dataset input is not a regular file: {path}")
        total += path.stat().st_size
        if total > max_input_bytes:
            raise ValueError("Dataset inspection exceeds aggregate raw input byte budget")
    # Fold/index config files do not select the matrix reader.
    data_paths = [path for key, value in config.items() if key in _DATA_KEYS for path in _references(value, True)]
    native = bool(data_paths) and all(str(path).lower().endswith(_NATIVE_SUFFIXES) for path in data_paths)
    reader: dict[str, Any] = {"raw_input_bytes": total, "max_input_bytes": max_input_bytes}
    if native:
        nio = importlib.import_module("nirs4all_io")
        # Do not silently import the source oracle and claim native budgets.
        importlib.import_module("nirs4all_io._native")
        dataset = nio.load(config, target="spectrodataset", limits=load_limits)
        reader.update({"backend": "nirs4all-io.native", "version": nio.__version__,
                       "native_load_limits_applied": True, "load_limits": load_limits or "native_defaults"})
    else:
        if load_limits is not None:
            raise ValueError("Native LoadLimits cannot be guaranteed by non-native format loaders")
        from nirs4all.data.config import DatasetConfigs
        datasets = DatasetConfigs(config).get_datasets()
        if len(datasets) != 1:
            raise ValueError("Dataset inspection requires one concrete assembled dataset")
        dataset = datasets[0]
        reader.update({"backend": "nirs4all.loaders", "native_load_limits_applied": False,
                       "load_limits": None, "limitations": ["native decompression/shape budgets do not cover this format route"]})
    return dataset, reader


def _json_values(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    return value


def _json_dict(value: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], _json_values(value))


def _sources(dataset: Any, partition: str) -> list[np.ndarray]:
    values = dataset.x({"partition": partition}, layout="2d", concat_source=False)
    return [np.asarray(value) for value in values] if isinstance(values, list) else [np.asarray(values)]


def _axis(dataset: Any, source: int, width: int) -> np.ndarray:
    headers = dataset.headers(source)
    if headers is not None and len(headers) == width:
        try:
            return np.asarray(headers, dtype=float)
        except (TypeError, ValueError):
            pass
    return np.arange(width)


def _spectra(values: np.ndarray, axis: np.ndarray, max_samples: int) -> dict[str, Any]:
    count = len(values)
    selected = np.linspace(0, count - 1, max_samples, dtype=int) if count > max_samples else np.arange(count)
    sample = values[selected]
    return _json_dict({
        "wavelengths": axis, "mean_spectrum": np.mean(sample, axis=0),
        "std_spectrum": np.std(sample, axis=0), "min_spectrum": np.min(sample, axis=0),
        "max_spectrum": np.max(sample, axis=0), "sample_spectra": sample[:5],
        "n_samples": len(sample), "total_samples": count, "sample_indices": selected,
        "statistics_scope": "display_sample" if len(sample) < count else "full_partition",
    })


def _target_distribution(values: np.ndarray, regression: bool) -> dict[str, Any]:
    values = np.asarray(values).reshape(-1)
    if regression:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return {"type": "regression", "n_samples": len(values), "missing_count": len(values), "histogram": []}
        counts, edges = np.histogram(finite, bins=20)
        return _json_dict({
            "type": "regression", "n_samples": len(values), "missing_count": len(values) - len(finite),
            "min": np.min(finite), "max": np.max(finite), "mean": np.mean(finite), "std": np.std(finite),
            "histogram": [{"bin": edge, "count": int(count)} for edge, count in zip(edges[:-1], counts, strict=True)],
        })
    classes, counts = np.unique(values, return_counts=True)
    return {"type": "classification", "n_samples": len(values), "classes": [str(value) for value in classes],
            "class_counts": {str(value): int(count) for value, count in zip(classes, counts, strict=True)}}


def preview_dataset(
    config: dict[str, Any], *, max_samples: int = 100, load_limits: dict[str, int] | None = None,
    max_input_bytes: int = 512 * 1024 * 1024, max_response_bytes: int = 32 * 1024 * 1024,
) -> dict[str, Any]:
    """Return partition/source-aware spectral previews and exact dataset counts.

    Spectral statistics retain the historical evenly spaced display-sample
    semantics, now explicitly labelled. Targets use the complete partition.
    No feature-axis truncation, fit, server, job or workspace is introduced.
    """
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    dataset, reader = load_dataset_for_analysis(config, load_limits=load_limits, max_input_bytes=max_input_bytes)
    train, test = _sources(dataset, "train"), _sources(dataset, "test")
    # Admit presentation cardinality before duplicating partition/source arrays
    # into historical response aliases. This is not a process RSS guarantee;
    # hosts must additionally bound streaming JSON bytes during serialization.
    estimated_bytes = 65536
    for source, source_train in enumerate(train):
        for partition, count in (("train", len(source_train)), ("test", len(test[source])),
                                 ("all", len(source_train) + len(test[source]))):
            if count:
                sample_count = min(count, max_samples)
                cells = source_train.shape[1] * (5 + min(5, sample_count)) + sample_count
                copies = (int(source == 0) + int(dataset.n_sources > 1)) * (2 if partition == "train" else 1)
                estimated_bytes += cells * copies * 32
    if not dataset.is_regression:
        estimated_bytes += dataset.num_samples * 256
    if estimated_bytes > max_response_bytes:
        raise ValueError("Dataset preview exceeds presentation byte budget before array construction")
    per_source: dict[int, dict[str, Any]] = {}
    for source in range(dataset.n_sources):
        partitions = {name: arrays[source] for name, arrays in (("train", train), ("test", test)) if len(arrays[source])}
        if not partitions:
            continue
        partitions["all"] = np.concatenate(list(partitions.values()), axis=0)
        axis = _axis(dataset, source, partitions["all"].shape[1])
        per_source[source] = {name: _spectra(values, axis, max_samples) for name, values in partitions.items()}
    targets: dict[str, dict[str, Any]] = {}
    target_distributions: dict[str, dict[str, Any]] = {}
    has_targets = dataset.describe()["has_targets"]
    if has_targets:
        for partition, sources in (("train", train), ("test", test)):
            partition_count = len(sources[0])
            # Targets' legacy empty-index selection means "all targets".
            # An absent X cohort must never acquire the training targets.
            if partition_count == 0:
                continue
            values = np.asarray(dataset.y({"partition": partition}))
            if values.size:
                if len(values) != partition_count:
                    raise ValueError(f"Target rows do not match the {partition} observation cohort")
                values = values.reshape(len(values), -1)
                targets[partition] = _target_distribution(values[:, 0], dataset.is_regression)
        values = np.asarray(dataset.y({})).reshape(dataset.num_samples, -1)
        if values.size:
            targets["all"] = _target_distribution(values[:, 0], dataset.is_regression)
            target_distributions = {f"target_{column}": _target_distribution(values[:, column], dataset.is_regression) for column in range(values.shape[1])}
    first = per_source.get(0, {})
    summary = {"num_samples": dataset.num_samples, "num_features": train[0].shape[1],
               "n_sources": dataset.n_sources, "features_per_source": [values.shape[1] for values in train],
               "train_samples": len(train[0]), "test_samples": len(test[0]),
               "has_targets": has_targets, "has_metadata": bool(dataset.metadata_columns),
               "metadata_columns": dataset.metadata_columns, "header_unit": dataset.header_unit(0),
               "signal_type": dataset.signal_types[0].value if dataset.signal_types else None}
    return _json_dict({
        "success": True, "error": None, "summary": summary, "reader": reader,
        "spectra_preview": first.get("train", first.get("all")), "spectra_preview_by_partition": first,
        "spectra_per_source": {source: partitions.get("train", partitions["all"]) for source, partitions in per_source.items()} if dataset.n_sources > 1 else None,
        "spectra_per_source_by_partition": per_source if dataset.n_sources > 1 else None,
        "target_distribution": targets.get("train"), "target_distribution_by_partition": targets,
        "target_distributions": target_distributions,
    })


def dataset_statistics(config: dict[str, Any], *, partition: str = "train", **load_options: Any) -> dict[str, Any]:
    """Compute historical first-source global statistics with reader evidence."""
    if partition not in {"train", "test", "all"}:
        raise ValueError("partition must be train, test or all")
    dataset, reader = load_dataset_for_analysis(config, **load_options)
    sources = dataset.x({} if partition == "all" else {"partition": partition}, layout="2d", concat_source=False)
    values = np.asarray(sources[0] if isinstance(sources, list) else sources)
    if not len(values):
        raise ValueError("Selected partition is empty")
    targets = None
    if dataset.describe()["has_targets"]:
        y = np.asarray(dataset.y({} if partition == "all" else {"partition": partition})).reshape(len(values), -1)
        targets = _target_distribution(y[:, 0], dataset.is_regression)
    return _json_dict({"partition": partition, "reader": reader,
                         "global": {"num_samples": len(values), "num_features": values.shape[1],
                                    "global_mean": np.mean(values), "global_std": np.std(values),
                                    "global_min": np.min(values), "global_max": np.max(values)}, "targets": targets})


def inspect_format_file(path: str, *, params: dict[str, Any] | None = None, sample_rows: int = 10,
                        max_input_bytes: int = 512 * 1024 * 1024) -> dict[str, Any]:
    """Inspect a non-native format through its existing registered file loader.

    CSV/Parquet file inspection belongs to native IO in the Rust host. No
    native decompression/shape limits are claimed for this explicit format lane.
    """
    from nirs4all.data.loaders import get_loader_for_file, load_file
    if str(path).lower().endswith(_NATIVE_SUFFIXES):
        raise ValueError("CSV/Parquet file inspection must use the native IO reader")
    if type(sample_rows) is not int or not 0 <= sample_rows <= 1000:
        raise ValueError("sample_rows must be an integer between 0 and 1000")
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError("max_input_bytes must be a positive integer")
    if not Path(path).is_file() or Path(path).stat().st_size > max_input_bytes:
        raise ValueError("Format inspection requires a regular file within raw input byte budget")
    data, report, _, headers, unit = load_file(path, **{"na_policy": "ignore", **(params or {})})
    if data is None or report.get("error"):
        raise ValueError(str(report.get("error") or "File loader returned no data"))
    return _json_dict({"success": True, "num_rows": len(data), "num_columns": len(data.columns),
                         "headers": list(headers), "header_unit": unit, "sample_data": data.head(sample_rows).astype(str).values.tolist(),
                         "reader": {"backend": "nirs4all.loaders", "loader": type(get_loader_for_file(path)).__name__,
                                    "native_load_limits_applied": False, "max_input_bytes": max_input_bytes}})

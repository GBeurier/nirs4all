"""Parquet-based storage for prediction arrays.

Stores dense numerical arrays (y_true, y_pred, y_proba, sample_indices,
weights) in one Parquet file per dataset, with Zstd compression.  Each
file embeds lightweight metadata columns (model_name, fold_id, partition,
metric, val_score, task_type) making it self-describing and portable.

The relational metadata (runs, pipelines, chains, full scores) stays in
DuckDB.  This module only manages the array sidecar files.

Workspace layout::

    workspace/
        store.duckdb          # Metadata only
        arrays/
            wheat.parquet     # All prediction arrays for dataset "wheat"
            corn.parquet      # All prediction arrays for dataset "corn"
            _tombstones.json  # Pending deletes (prediction_ids)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Zstd level 3: good compression ratio, fast decompression
_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 3
_TOMBSTONE_FILE = "_tombstones.json"

# Characters invalid in filenames across platforms
_UNSAFE_FILENAME_RE = re.compile(r'[/\\:*?"<>|\s.]+')

def _sanitize_dataset_filename(dataset_name: str) -> str:
    """Convert a dataset name to a safe filename (without extension).

    Replaces any character that is invalid in filenames on Windows/Linux
    (slashes, colons, wildcards, spaces, etc.) with underscores.  Strips
    leading/trailing underscores to avoid hidden files or empty stems.
    """
    sanitized = _UNSAFE_FILENAME_RE.sub("_", dataset_name)
    return sanitized.strip("_") or "unnamed"

def _arr_to_list(arr: np.ndarray | None, dtype: str = "float") -> list | None:
    """Convert a numpy array to a Python list for Arrow storage."""
    if arr is None:
        return None
    flat = arr.flatten()
    if dtype == "int":
        return flat.astype(np.int32).tolist()
    return flat.astype(np.float64).tolist()

def _shape_to_list(arr: np.ndarray | None) -> list[int] | None:
    """Return the shape of an array as a list of ints, or None."""
    if arr is None:
        return None
    return list(arr.shape)

# Shared Arrow schema for all Parquet files
_PARQUET_SCHEMA = pa.schema([
    ("prediction_id", pa.utf8()),
    ("dataset_name", pa.utf8()),
    ("model_name", pa.utf8()),
    ("fold_id", pa.utf8()),
    ("partition", pa.utf8()),
    ("metric", pa.utf8()),
    ("val_score", pa.float64()),
    ("task_type", pa.utf8()),
    ("y_true", pa.list_(pa.float64())),
    ("y_pred", pa.list_(pa.float64())),
    ("y_proba", pa.list_(pa.float64())),
    ("y_proba_shape", pa.list_(pa.int32())),
    ("sample_indices", pa.list_(pa.int32())),
    ("weights", pa.list_(pa.float64())),
])

class ArrayStore:
    """Parquet-backed storage for prediction arrays.

    Arrays live under ``base_dir / 'arrays'``, one ``.parquet`` file per
    dataset.  Writes append row groups; deletes use a tombstone file that
    is applied during :meth:`compact`.

    Args:
        base_dir: Workspace root directory.  The ``arrays/`` subdirectory
            is created automatically.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._arrays_dir = self._base_dir / "arrays"
        self._arrays_dir.mkdir(parents=True, exist_ok=True)

    @property
    def arrays_dir(self) -> Path:
        """Directory containing the Parquet array files."""
        return self._arrays_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parquet_path(self, dataset_name: str) -> Path:
        """Return the Parquet file path for a given dataset."""
        return self._arrays_dir / f"{_sanitize_dataset_filename(dataset_name)}.parquet"

    def _tombstone_path(self) -> Path:
        return self._arrays_dir / _TOMBSTONE_FILE

    def _read_tombstones(self) -> dict[str, str]:
        """Read the tombstone file.  Returns {prediction_id: timestamp}."""
        path = self._tombstone_path()
        if not path.exists():
            return {}
        with open(path) as f:
            return dict(json.load(f))

    def _write_tombstones(self, tombstones: dict[str, str]) -> None:
        """Write the tombstone file atomically via temp + rename."""
        path = self._tombstone_path()
        if not tombstones:
            if path.exists():
                path.unlink()
            return
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(tombstones, f)
        tmp.replace(path)

    def _records_to_table(self, records: list[dict]) -> pa.Table:
        """Convert a list of record dicts to a PyArrow Table."""
        columns: dict[str, list] = {field.name: [] for field in _PARQUET_SCHEMA}

        for rec in records:
            columns["prediction_id"].append(rec["prediction_id"])
            columns["dataset_name"].append(rec["dataset_name"])
            columns["model_name"].append(rec.get("model_name", ""))
            columns["fold_id"].append(rec.get("fold_id", ""))
            columns["partition"].append(rec.get("partition", ""))
            columns["metric"].append(rec.get("metric", ""))
            columns["val_score"].append(rec.get("val_score"))
            columns["task_type"].append(rec.get("task_type", ""))

            y_true = rec.get("y_true")
            y_pred = rec.get("y_pred")
            y_proba = rec.get("y_proba")
            sample_indices = rec.get("sample_indices")
            weights = rec.get("weights")

            columns["y_true"].append(_arr_to_list(y_true))
            columns["y_pred"].append(_arr_to_list(y_pred))
            columns["y_proba"].append(_arr_to_list(y_proba))
            columns["y_proba_shape"].append(_shape_to_list(y_proba))
            columns["sample_indices"].append(_arr_to_list(sample_indices, dtype="int"))
            columns["weights"].append(_arr_to_list(weights))

        arrays = [pa.array(columns[field.name], type=field.type) for field in _PARQUET_SCHEMA]
        return pa.table(arrays, schema=_PARQUET_SCHEMA)

    @staticmethod
    def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
        """Write a Parquet table atomically via temp file + rename.

        Prevents data loss if a crash occurs mid-write: the original
        file remains intact until the new file is fully written and
        ``os.replace()`` atomically swaps the name.
        """
        tmp = path.with_suffix(".parquet.tmp")
        try:
            pq.write_table(
                table,
                tmp,
                compression=_COMPRESSION,
                compression_level=_COMPRESSION_LEVEL,
            )
            os.replace(tmp, path)
        except BaseException:
            # Clean up partial temp file on any failure
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_batch(self, records: list[dict]) -> int:
        """Write prediction arrays to Parquet.

        Each dict should contain::

            {prediction_id, dataset_name, model_name, fold_id, partition,
             metric, val_score, task_type, y_true, y_pred, y_proba,
             y_proba_shape, sample_indices, weights}

        Records are grouped by ``dataset_name``; each group is appended
        to its dataset Parquet file.  Writes are idempotent by
        ``prediction_id`` (duplicates resolved on :meth:`compact`).

        Args:
            records: List of prediction array records.

        Returns:
            Number of rows written.
        """
        if not records:
            return 0

        # Clear tombstones for prediction_ids being written (handles upserts)
        written_ids = {rec["prediction_id"] for rec in records}
        tombstones = self._read_tombstones()
        cleared = {k: v for k, v in tombstones.items() if k not in written_ids}
        if len(cleared) != len(tombstones):
            self._write_tombstones(cleared)

        # Group records by dataset_name
        groups: dict[str, list[dict]] = {}
        for rec in records:
            ds = rec["dataset_name"]
            groups.setdefault(ds, []).append(rec)

        total_written = 0
        for dataset_name, group_records in groups.items():
            table = self._records_to_table(group_records)
            path = self._parquet_path(dataset_name)

            if path.exists():
                existing = pq.read_table(path)
                combined = pa.concat_tables([existing, table])
                self._atomic_write_parquet(combined, path)
            else:
                self._atomic_write_parquet(table, path)

            total_written += len(group_records)

        return total_written

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_batch(
        self,
        prediction_ids: list[str],
        dataset_name: str | None = None,
    ) -> dict[str, dict[str, np.ndarray | None]]:
        """Load arrays for multiple predictions.

        Uses predicate pushdown on ``prediction_id`` for efficient reads.

        Args:
            prediction_ids: List of prediction IDs to load.
            dataset_name: If given, reads only that dataset's file.

        Returns:
            ``{prediction_id: {y_true, y_pred, y_proba, sample_indices, weights}}``
        """
        if not prediction_ids:
            return {}

        id_set = set(prediction_ids)
        result: dict[str, dict[str, np.ndarray | None]] = {}

        files = [self._parquet_path(dataset_name)] if dataset_name else sorted(self._arrays_dir.glob("*.parquet"))

        for path in files:
            if not path.exists():
                continue

            # Read with predicate pushdown via Polars for efficiency
            try:
                df = pl.scan_parquet(path).filter(
                    pl.col("prediction_id").is_in(list(id_set))
                ).collect()
            except Exception:
                logger.warning("Failed to read Parquet file %s", path)
                continue

            if df.is_empty():
                continue

            for row in df.iter_rows(named=True):
                pid = row["prediction_id"]
                arrays: dict[str, np.ndarray | None] = {}

                for field in ("y_true", "y_pred", "y_proba", "weights"):
                    val = row.get(field)
                    if val is not None:
                        arrays[field] = np.array(val, dtype=np.float64)
                    else:
                        arrays[field] = None

                # Reconstruct y_proba shape if available
                if arrays["y_proba"] is not None:
                    shape = row.get("y_proba_shape")
                    if shape is not None and len(shape) > 1:
                        with contextlib.suppress(ValueError):
                            arrays["y_proba"] = arrays["y_proba"].reshape(shape)

                val = row.get("sample_indices")
                if val is not None:
                    arrays["sample_indices"] = np.array(val, dtype=np.int32)
                else:
                    arrays["sample_indices"] = None

                result[pid] = arrays
                id_set.discard(pid)

            if not id_set:
                break

        return result

    def load_single(
        self,
        prediction_id: str,
        dataset_name: str | None = None,
    ) -> dict[str, np.ndarray | None] | None:
        """Load arrays for one prediction.

        Args:
            prediction_id: Prediction ID to load.
            dataset_name: If given, reads only that dataset's file.

        Returns:
            Dict with array fields, or ``None`` if not found.
        """
        batch = self.load_batch([prediction_id], dataset_name=dataset_name)
        return batch.get(prediction_id)

    def load_dataset(self, dataset_name: str) -> pl.DataFrame:
        """Load the full Parquet file for a dataset.

        Returns all columns (arrays + metadata).  This is the
        'portable read' — the returned DataFrame is self-describing.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            A Polars DataFrame with all rows and columns.

        Raises:
            FileNotFoundError: If no Parquet file exists for this dataset.
        """
        path = self._parquet_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(f"No Parquet file for dataset '{dataset_name}': {path}")
        return pl.read_parquet(path)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_batch(
        self,
        prediction_ids: set[str],
        dataset_name: str | None = None,
    ) -> int:
        """Mark prediction_ids as deleted (tombstone).

        Physical removal happens during :meth:`compact`.

        Args:
            prediction_ids: Set of prediction IDs to mark as deleted.
            dataset_name: Optional dataset filter.

        Returns:
            Number of IDs marked for deletion.
        """
        if not prediction_ids:
            return 0

        from datetime import UTC, datetime

        tombstones = self._read_tombstones()
        ts = datetime.now(UTC).isoformat()
        count = 0
        for pid in prediction_ids:
            if pid not in tombstones:
                tombstones[pid] = ts
                count += 1
        self._write_tombstones(tombstones)
        return count

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete the entire Parquet file for a dataset.

        Args:
            dataset_name: Name of the dataset to delete.

        Returns:
            ``True`` if the file existed and was deleted.
        """
        path = self._parquet_path(dataset_name)
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self, dataset_name: str | None = None) -> dict[str, dict[str, Any]]:
        """Rewrite Parquet file(s): apply tombstones, deduplicate, re-sort.

        Args:
            dataset_name: If given, compact only that dataset's file.
                If ``None``, compact all datasets.

        Returns:
            Stats per dataset::

                {dataset: {rows_before, rows_after, rows_removed,
                           bytes_before, bytes_after}}
        """
        tombstones = self._read_tombstones()
        tombstone_ids = set(tombstones.keys())
        applied_ids: set[str] = set()
        stats: dict[str, dict[str, Any]] = {}

        files = [self._parquet_path(dataset_name)] if dataset_name else sorted(self._arrays_dir.glob("*.parquet"))

        for path in files:
            if not path.exists():
                continue

            ds_name = path.stem
            bytes_before = path.stat().st_size

            df = pl.read_parquet(path)
            rows_before = len(df)

            # Remove tombstoned rows and track which were actually applied
            if tombstone_ids:
                file_ids = set(df["prediction_id"].to_list())
                removed_here = tombstone_ids & file_ids
                applied_ids.update(removed_here)
                if removed_here:
                    df = df.filter(~pl.col("prediction_id").is_in(list(tombstone_ids)))

            # Deduplicate by prediction_id (keep last occurrence)
            df = df.unique(subset=["prediction_id"], keep="last")

            # Sort by partition, fold_id for better compression
            df = df.sort(["partition", "fold_id", "prediction_id"])

            rows_after = len(df)

            if rows_after == 0:
                path.unlink()
                stats[ds_name] = {
                    "rows_before": rows_before,
                    "rows_after": 0,
                    "rows_removed": rows_before,
                    "bytes_before": bytes_before,
                    "bytes_after": 0,
                }
            else:
                self._atomic_write_parquet(df.to_arrow(), path)
                bytes_after = path.stat().st_size
                stats[ds_name] = {
                    "rows_before": rows_before,
                    "rows_after": rows_after,
                    "rows_removed": rows_before - rows_after,
                    "bytes_before": bytes_before,
                    "bytes_after": bytes_after,
                }

        # Only clear tombstones that were actually applied to files
        if applied_ids:
            remaining = {k: v for k, v in tombstones.items() if k not in applied_ids}
            self._write_tombstones(remaining)

        return stats

    def stats(self) -> dict[str, Any]:
        """Return storage stats.

        Returns:
            ``{total_files, total_rows, total_bytes,
               datasets: {name: {rows, bytes}}}``
        """
        datasets: dict[str, dict[str, int]] = {}
        total_files = 0
        total_rows = 0
        total_bytes = 0

        for path in sorted(self._arrays_dir.glob("*.parquet")):
            total_files += 1
            file_bytes = path.stat().st_size
            total_bytes += file_bytes

            try:
                metadata = pq.read_metadata(path)
                rows = metadata.num_rows
            except Exception:
                rows = 0

            total_rows += rows
            datasets[path.stem] = {"rows": rows, "bytes": file_bytes}

        return {
            "total_files": total_files,
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "datasets": datasets,
        }

    def integrity_check(
        self,
        expected_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Check Parquet health.

        Args:
            expected_ids: If given, cross-check against these prediction IDs.

        Returns:
            ``{orphan_ids: [...], missing_ids: [...], corrupt_files: [...]}``
        """
        corrupt_files: list[str] = []
        all_parquet_ids: set[str] = set()

        for path in sorted(self._arrays_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(path, columns=["prediction_id"])
                ids = set(df["prediction_id"].to_list())
                all_parquet_ids.update(ids)
            except Exception:
                corrupt_files.append(str(path))

        orphan_ids: list[str] = []
        missing_ids: list[str] = []

        if expected_ids is not None:
            orphan_ids = sorted(all_parquet_ids - expected_ids)
            missing_ids = sorted(expected_ids - all_parquet_ids)

        return {
            "orphan_ids": orphan_ids,
            "missing_ids": missing_ids,
            "corrupt_files": corrupt_files,
        }

    def list_datasets(self) -> list[str]:
        """Return dataset names that have Parquet files."""
        return sorted(p.stem for p in self._arrays_dir.glob("*.parquet"))

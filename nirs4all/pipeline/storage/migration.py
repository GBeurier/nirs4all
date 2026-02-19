"""Migration from DuckDB prediction_arrays to Parquet sidecar files.

Provides :func:`migrate_arrays_to_parquet` which reads dense prediction
arrays from the legacy ``prediction_arrays`` DuckDB table, writes them
to per-dataset Parquet files via :class:`ArrayStore`, verifies data
integrity, and optionally drops the legacy table.

Usage::

    python -m nirs4all.pipeline.storage.migration /path/to/workspace
    python -m nirs4all.pipeline.storage.migration /path/to/workspace --dry-run
    python -m nirs4all.pipeline.storage.migration /path/to/workspace --verify-only
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

from nirs4all.pipeline.storage.array_store import ArrayStore

logger = logging.getLogger(__name__)

@dataclass
class MigrationReport:
    """Result of a migration run."""

    total_rows: int = 0
    rows_migrated: int = 0
    datasets_migrated: list[str] = field(default_factory=list)
    verification_passed: bool = False
    verification_sample_size: int = 0
    verification_mismatches: int = 0
    duckdb_size_before: int = 0  # bytes
    duckdb_size_after: int = 0  # bytes
    parquet_total_size: int = 0  # bytes
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check whether a table exists in the DuckDB database."""
    result = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = $1 AND table_type = 'BASE TABLE'",
        [table_name],
    ).fetchone()
    return result is not None

def _array_checksum(arr: list | np.ndarray | None) -> str:
    """Compute a stable checksum for an array (or None)."""
    if arr is None:
        return "none"
    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.float64)
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()

def migrate_arrays_to_parquet(
    workspace_path: str | Path,
    *,
    batch_size: int = 10_000,
    verify: bool = True,
    dry_run: bool = False,
) -> MigrationReport:
    """Migrate prediction_arrays from DuckDB to Parquet sidecar files.

    Steps:
        1. Open the DuckDB store.
        2. Query distinct dataset_names from the predictions table.
        3. For each dataset:
           a. Stream prediction_ids + arrays from prediction_arrays in batches.
           b. Join with predictions table for portable metadata columns.
           c. Write to arrays/<dataset_name>.parquet via ArrayStore.
           d. Verify: sample 1% of rows, compare array checksums.
        4. If all datasets pass verification and dry_run is False:
           a. DROP TABLE prediction_arrays.
           b. VACUUM to reclaim space.
        5. Return MigrationReport.

    Rollback: if verify fails or error occurs, delete arrays/ directory.
    DuckDB is untouched until final DROP.

    Args:
        workspace_path: Root directory of the workspace.
        batch_size: Number of rows to process per batch.
        verify: Whether to verify migrated data. Defaults to True.
        dry_run: If True, report what would happen without writing.

    Returns:
        A :class:`MigrationReport` with migration statistics.
    """
    workspace_path = Path(workspace_path)
    report = MigrationReport()
    start_time = time.monotonic()

    db_path = workspace_path / "store.duckdb"
    if not db_path.exists():
        report.errors.append(f"DuckDB file not found: {db_path}")
        report.duration_seconds = time.monotonic() - start_time
        return report

    report.duckdb_size_before = db_path.stat().st_size

    conn = duckdb.connect(str(db_path))
    try:
        if not _table_exists(conn, "prediction_arrays"):
            report.errors.append("No prediction_arrays table found — nothing to migrate.")
            report.duration_seconds = time.monotonic() - start_time
            return report

        # Count total rows
        total_row = conn.execute("SELECT COUNT(*) FROM prediction_arrays").fetchone()
        report.total_rows = total_row[0] if total_row else 0

        if report.total_rows == 0:
            logger.info("prediction_arrays table is empty — nothing to migrate.")
            if not dry_run:
                conn.execute("DROP TABLE prediction_arrays")
                conn.execute("VACUUM")
            report.verification_passed = True
            report.duckdb_size_after = db_path.stat().st_size
            report.duration_seconds = time.monotonic() - start_time
            return report

        # Get distinct dataset names from predictions table
        dataset_rows = conn.execute(
            "SELECT DISTINCT p.dataset_name "
            "FROM predictions p "
            "INNER JOIN prediction_arrays pa ON p.prediction_id = pa.prediction_id "
            "ORDER BY p.dataset_name"
        ).fetchall()
        dataset_names = [row[0] for row in dataset_rows]

        if dry_run:
            logger.info(
                "DRY RUN: Would migrate %d rows across %d datasets: %s",
                report.total_rows, len(dataset_names), dataset_names,
            )
            report.datasets_migrated = dataset_names
            report.rows_migrated = report.total_rows
            report.verification_passed = True
            report.duration_seconds = time.monotonic() - start_time
            return report

        # Create ArrayStore for writing
        array_store = ArrayStore(workspace_path)
        arrays_dir = array_store.arrays_dir

        try:
            for dataset_name in dataset_names:
                _migrate_dataset(conn, array_store, dataset_name, batch_size, report)
                report.datasets_migrated.append(dataset_name)

            # Verification
            if verify:
                _verify_migration(conn, array_store, report)

            if report.verification_mismatches > 0:
                report.verification_passed = False
                report.errors.append(
                    f"Verification failed: {report.verification_mismatches} mismatches."
                )
                # Rollback: delete arrays directory
                if arrays_dir.exists():
                    shutil.rmtree(arrays_dir)
                    logger.warning("Rolled back: deleted %s due to verification failure.", arrays_dir)
            else:
                report.verification_passed = True

                # Drop the legacy table and vacuum
                logger.info("Verification passed. Dropping prediction_arrays table.")
                conn.execute("DROP TABLE prediction_arrays")
                conn.execute("VACUUM")

                report.duckdb_size_after = db_path.stat().st_size
                report.parquet_total_size = sum(
                    f.stat().st_size for f in arrays_dir.glob("*.parquet")
                )

        except Exception as e:
            report.errors.append(f"Migration error: {e}")
            # Rollback: delete arrays directory
            if arrays_dir.exists():
                shutil.rmtree(arrays_dir)
                logger.warning("Rolled back: deleted %s due to error.", arrays_dir)
            raise

    finally:
        conn.close()
        report.duration_seconds = time.monotonic() - start_time

    return report

def _migrate_dataset(
    conn: duckdb.DuckDBPyConnection,
    array_store: ArrayStore,
    dataset_name: str,
    batch_size: int,
    report: MigrationReport,
) -> None:
    """Migrate all prediction arrays for a single dataset."""
    offset = 0
    while True:
        # Join prediction_arrays with predictions to get portable metadata
        rows = conn.execute(
            "SELECT pa.prediction_id, pa.y_true, pa.y_pred, pa.y_proba, "
            "       pa.sample_indices, pa.weights, "
            "       p.model_name, p.fold_id, p.partition, p.metric, "
            "       p.val_score, p.task_type "
            "FROM prediction_arrays pa "
            "INNER JOIN predictions p ON pa.prediction_id = p.prediction_id "
            "WHERE p.dataset_name = $1 "
            "ORDER BY pa.prediction_id "
            "LIMIT $2 OFFSET $3",
            [dataset_name, batch_size, offset],
        ).fetchall()

        if not rows:
            break

        records = []
        for row in rows:
            (prediction_id, y_true, y_pred, y_proba,
             sample_indices, weights,
             model_name, fold_id, partition, metric,
             val_score, task_type) = row

            records.append({
                "prediction_id": prediction_id,
                "dataset_name": dataset_name,
                "model_name": model_name or "",
                "fold_id": fold_id or "",
                "partition": partition or "",
                "metric": metric or "",
                "val_score": val_score,
                "task_type": task_type or "",
                "y_true": np.array(y_true, dtype=np.float64) if y_true is not None else None,
                "y_pred": np.array(y_pred, dtype=np.float64) if y_pred is not None else None,
                "y_proba": np.array(y_proba, dtype=np.float64) if y_proba is not None else None,
                "sample_indices": np.array(sample_indices, dtype=np.int32) if sample_indices is not None else None,
                "weights": np.array(weights, dtype=np.float64) if weights is not None else None,
            })

        array_store.save_batch(records)
        report.rows_migrated += len(records)
        offset += batch_size

        logger.info(
            "Migrated %d rows for dataset '%s' (total: %d/%d)",
            len(records), dataset_name, report.rows_migrated, report.total_rows,
        )

def _verify_migration(
    conn: duckdb.DuckDBPyConnection,
    array_store: ArrayStore,
    report: MigrationReport,
) -> None:
    """Verify migrated data by sampling 1% of rows and comparing checksums."""
    # Sample ~1% of prediction_ids, minimum 1
    sample_size = max(1, report.total_rows // 100)
    sample_rows = conn.execute(
        "SELECT pa.prediction_id, pa.y_true, pa.y_pred, p.dataset_name "
        "FROM prediction_arrays pa "
        "INNER JOIN predictions p ON pa.prediction_id = p.prediction_id "
        "ORDER BY md5(pa.prediction_id) "
        "LIMIT $1",
        [sample_size],
    ).fetchall()

    report.verification_sample_size = len(sample_rows)
    mismatches = 0

    for prediction_id, duckdb_y_true, duckdb_y_pred, dataset_name in sample_rows:
        parquet_arrays = array_store.load_single(prediction_id, dataset_name=dataset_name)
        if parquet_arrays is None:
            mismatches += 1
            logger.warning("Verification: prediction %s missing from Parquet.", prediction_id)
            continue

        # Compare y_true checksums
        duckdb_checksum = _array_checksum(duckdb_y_true)
        parquet_checksum = _array_checksum(parquet_arrays.get("y_true"))
        if duckdb_checksum != parquet_checksum:
            mismatches += 1
            logger.warning(
                "Verification: y_true mismatch for %s (DuckDB=%s, Parquet=%s)",
                prediction_id, duckdb_checksum, parquet_checksum,
            )
            continue

        # Compare y_pred checksums
        duckdb_checksum = _array_checksum(duckdb_y_pred)
        parquet_checksum = _array_checksum(parquet_arrays.get("y_pred"))
        if duckdb_checksum != parquet_checksum:
            mismatches += 1
            logger.warning(
                "Verification: y_pred mismatch for %s (DuckDB=%s, Parquet=%s)",
                prediction_id, duckdb_checksum, parquet_checksum,
            )

    report.verification_mismatches = mismatches

def verify_migrated_store(workspace_path: str | Path) -> MigrationReport:
    """Verify an already-migrated workspace.

    Cross-checks that all prediction_ids in DuckDB have corresponding
    array data in the Parquet files.

    Args:
        workspace_path: Root directory of the workspace.

    Returns:
        A :class:`MigrationReport` with verification results.
    """
    workspace_path = Path(workspace_path)
    report = MigrationReport()
    start_time = time.monotonic()

    db_path = workspace_path / "store.duckdb"
    if not db_path.exists():
        report.errors.append(f"DuckDB file not found: {db_path}")
        report.duration_seconds = time.monotonic() - start_time
        return report

    array_store = ArrayStore(workspace_path)
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # Get all prediction_ids from DuckDB
        pred_ids_rows = conn.execute("SELECT prediction_id FROM predictions").fetchall()
        expected_ids = {row[0] for row in pred_ids_rows}

        integrity = array_store.integrity_check(expected_ids=expected_ids)
        report.verification_sample_size = len(expected_ids)
        report.verification_mismatches = len(integrity["missing_ids"]) + len(integrity["corrupt_files"])
        report.verification_passed = report.verification_mismatches == 0

        if integrity["missing_ids"]:
            report.errors.append(
                f"{len(integrity['missing_ids'])} predictions missing from Parquet: "
                f"{integrity['missing_ids'][:10]}..."
            )
        if integrity["orphan_ids"]:
            report.errors.append(
                f"{len(integrity['orphan_ids'])} orphan rows in Parquet (not in DuckDB): "
                f"{integrity['orphan_ids'][:10]}..."
            )
        if integrity["corrupt_files"]:
            report.errors.append(f"Corrupt Parquet files: {integrity['corrupt_files']}")

        stats = array_store.stats()
        report.parquet_total_size = stats["total_bytes"]
        report.total_rows = len(expected_ids)
        report.rows_migrated = stats["total_rows"]
    finally:
        conn.close()
        report.duration_seconds = time.monotonic() - start_time

    return report

# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Migrate prediction_arrays from DuckDB to Parquet sidecar files.",
    )
    parser.add_argument(
        "workspace",
        help="Path to the workspace directory containing store.duckdb",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing any data.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an already-migrated workspace (no migration).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of rows to process per batch (default: 10000).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-migration verification.",
    )

    args = parser.parse_args()
    workspace = Path(args.workspace)

    if not workspace.exists():
        print(f"Error: workspace path does not exist: {workspace}", file=sys.stderr)
        sys.exit(1)

    print(f"Back up store.duckdb before migrating: cp {workspace / 'store.duckdb'} {workspace / 'store.duckdb.bak'}")
    print()

    if args.verify_only:
        print(f"Verifying migrated workspace: {workspace}")
        report = verify_migrated_store(workspace)
    else:
        print(f"Migrating workspace: {workspace}")
        report = migrate_arrays_to_parquet(
            workspace,
            batch_size=args.batch_size,
            verify=not args.no_verify,
            dry_run=args.dry_run,
        )

    print()
    print("=" * 60)
    print("Migration Report")
    print("=" * 60)
    print(f"  Total rows:              {report.total_rows}")
    print(f"  Rows migrated:           {report.rows_migrated}")
    print(f"  Datasets migrated:       {report.datasets_migrated}")
    print(f"  Verification passed:     {report.verification_passed}")
    print(f"  Verification sample:     {report.verification_sample_size}")
    print(f"  Verification mismatches: {report.verification_mismatches}")
    print(f"  DuckDB size before:      {report.duckdb_size_before:,} bytes")
    print(f"  DuckDB size after:       {report.duckdb_size_after:,} bytes")
    print(f"  Parquet total size:      {report.parquet_total_size:,} bytes")
    print(f"  Duration:                {report.duration_seconds:.1f}s")
    if report.errors:
        print("  Errors:")
        for err in report.errors:
            print(f"    - {err}")
    print("=" * 60)

    if not report.verification_passed and not args.dry_run:
        sys.exit(1)

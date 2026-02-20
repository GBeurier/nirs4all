"""Unit tests for ArrayStore (Parquet-backed prediction array storage).

Covers: round-trip save/load, single-file-per-dataset layout, delete/compact,
integrity checks, y_proba shape preservation, portable columns, and batch flush.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.storage.array_store import ArrayStore

# =========================================================================
# Helpers
# =========================================================================

def _make_store(tmp_path: Path) -> ArrayStore:
    return ArrayStore(tmp_path / "workspace")

def _make_record(
    prediction_id: str = "pred_001",
    dataset_name: str = "wheat",
    model_name: str = "PLSRegression",
    fold_id: str = "fold_0",
    partition: str = "val",
    metric: str = "rmse",
    val_score: float = 0.123,
    task_type: str = "regression",
    n_samples: int = 50,
    *,
    include_proba: bool = False,
) -> dict:
    rng = np.random.default_rng(42)
    rec = {
        "prediction_id": prediction_id,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "fold_id": fold_id,
        "partition": partition,
        "metric": metric,
        "val_score": val_score,
        "task_type": task_type,
        "y_true": rng.standard_normal(n_samples),
        "y_pred": rng.standard_normal(n_samples),
        "y_proba": None,
        "sample_indices": np.arange(n_samples, dtype=np.int32),
        "weights": rng.uniform(size=n_samples),
    }
    if include_proba:
        rec["y_proba"] = rng.uniform(size=(n_samples, 3))
        rec["task_type"] = "classification"
    return rec

# =========================================================================
# Tests
# =========================================================================

class TestArrayStoreSaveLoad:
    """Round-trip: save_batch → load_batch, verify numpy equality."""

    def test_single_record_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec = _make_record()
        written = store.save_batch([rec])
        assert written == 1

        result = store.load_batch(["pred_001"], dataset_name="wheat")
        assert "pred_001" in result
        arrays = result["pred_001"]
        assert arrays is not None
        assert arrays["y_true"] is not None
        assert arrays["y_pred"] is not None
        assert arrays["weights"] is not None
        np.testing.assert_array_almost_equal(arrays["y_true"], rec["y_true"])
        np.testing.assert_array_almost_equal(arrays["y_pred"], rec["y_pred"])
        np.testing.assert_array_equal(arrays["sample_indices"], rec["sample_indices"].astype(np.int32))
        np.testing.assert_array_almost_equal(arrays["weights"], rec["weights"])
        assert arrays["y_proba"] is None

    def test_multiple_records_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [_make_record(prediction_id=f"pred_{i:03d}") for i in range(10)]
        written = store.save_batch(records)
        assert written == 10

        ids = [f"pred_{i:03d}" for i in range(10)]
        result = store.load_batch(ids, dataset_name="wheat")
        assert len(result) == 10

    def test_load_single(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec = _make_record()
        store.save_batch([rec])

        arrays = store.load_single("pred_001", dataset_name="wheat")
        assert arrays is not None
        assert arrays["y_true"] is not None
        np.testing.assert_array_almost_equal(arrays["y_true"], rec["y_true"])

    def test_load_single_not_found(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.load_single("nonexistent") is None

    def test_save_empty_batch(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.save_batch([]) == 0

class TestArrayStoreSingleFilePerDataset:
    """Two datasets → two `.parquet` files, correct content in each."""

    def test_separate_files(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec_wheat = _make_record(prediction_id="p1", dataset_name="wheat")
        rec_corn = _make_record(prediction_id="p2", dataset_name="corn")
        store.save_batch([rec_wheat, rec_corn])

        assert (store.arrays_dir / "wheat.parquet").exists()
        assert (store.arrays_dir / "corn.parquet").exists()

        # Verify content isolation
        wheat_result = store.load_batch(["p1"], dataset_name="wheat")
        assert "p1" in wheat_result
        assert "p2" not in wheat_result

        corn_result = store.load_batch(["p2"], dataset_name="corn")
        assert "p2" in corn_result
        assert "p1" not in corn_result

    def test_list_datasets(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([
            _make_record(prediction_id="p1", dataset_name="wheat"),
            _make_record(prediction_id="p2", dataset_name="corn"),
        ])
        datasets = store.list_datasets()
        assert datasets == ["corn", "wheat"]

class TestArrayStoreDeleteCompact:
    """Tombstone → compact → verify rows removed, file shrinks."""

    def test_delete_and_compact(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        records = [_make_record(prediction_id=f"p{i}") for i in range(5)]
        store.save_batch(records)

        # Mark 2 for deletion
        deleted = store.delete_batch({"p0", "p1"})
        assert deleted == 2

        # Before compact, rows are still in the file
        all_result = store.load_batch([f"p{i}" for i in range(5)], dataset_name="wheat")
        assert len(all_result) == 5

        # After compact, deleted rows are gone
        stats = store.compact("wheat")
        assert "wheat" in stats
        assert stats["wheat"]["rows_before"] == 5
        assert stats["wheat"]["rows_after"] == 3
        assert stats["wheat"]["rows_removed"] == 2

        remaining = store.load_batch([f"p{i}" for i in range(5)], dataset_name="wheat")
        assert len(remaining) == 3
        assert "p0" not in remaining
        assert "p1" not in remaining

    def test_delete_dataset(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1", dataset_name="wheat")])
        assert (store.arrays_dir / "wheat.parquet").exists()

        deleted = store.delete_dataset("wheat")
        assert deleted is True
        assert not (store.arrays_dir / "wheat.parquet").exists()

        # Second delete returns False
        assert store.delete_dataset("wheat") is False

    def test_compact_deduplicates(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # Save same prediction_id twice
        store.save_batch([_make_record(prediction_id="dup")])
        store.save_batch([_make_record(prediction_id="dup")])

        stats = store.compact("wheat")
        assert stats["wheat"]["rows_before"] == 2
        assert stats["wheat"]["rows_after"] == 1

    def test_compact_all_datasets(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([
            _make_record(prediction_id="p1", dataset_name="wheat"),
            _make_record(prediction_id="p2", dataset_name="corn"),
        ])
        store.delete_batch({"p1"})

        stats = store.compact()
        assert "wheat" in stats
        assert stats["wheat"]["rows_after"] == 0
        # corn is untouched (no tombstones)
        assert stats["corn"]["rows_after"] == 1

class TestArrayStoreIntegrityCheck:
    """Inject orphans/missing → check detected."""

    def test_no_issues(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1")])

        result = store.integrity_check(expected_ids={"p1"})
        assert result["orphan_ids"] == []
        assert result["missing_ids"] == []
        assert result["corrupt_files"] == []

    def test_orphan_ids(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1"), _make_record(prediction_id="p2")])

        result = store.integrity_check(expected_ids={"p1"})
        assert "p2" in result["orphan_ids"]
        assert result["missing_ids"] == []

    def test_missing_ids(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1")])

        result = store.integrity_check(expected_ids={"p1", "p2", "p3"})
        assert result["orphan_ids"] == []
        assert "p2" in result["missing_ids"]
        assert "p3" in result["missing_ids"]

class TestArrayStoreYProbaShape:
    """2D y_proba survives round-trip with correct shape."""

    def test_proba_shape_preserved(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec = _make_record(include_proba=True, n_samples=20)
        assert rec["y_proba"].shape == (20, 3)

        store.save_batch([rec])
        arrays = store.load_single("pred_001", dataset_name="wheat")
        assert arrays is not None
        assert arrays["y_proba"] is not None
        assert arrays["y_proba"].shape == (20, 3)
        np.testing.assert_array_almost_equal(arrays["y_proba"], rec["y_proba"])

    def test_proba_none_for_regression(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec = _make_record(include_proba=False)
        store.save_batch([rec])

        arrays = store.load_single("pred_001", dataset_name="wheat")
        assert arrays is not None
        assert arrays["y_proba"] is None

class TestArrayStorePortableColumns:
    """Parquet contains model_name, fold_id, partition, metric, val_score, task_type."""

    def test_portable_metadata_in_parquet(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        rec = _make_record(
            model_name="PLSRegression",
            fold_id="fold_2",
            partition="test",
            metric="r2",
            val_score=0.95,
            task_type="regression",
        )
        store.save_batch([rec])

        df = store.load_dataset("wheat")
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["model_name"] == "PLSRegression"
        assert row["fold_id"] == "fold_2"
        assert row["partition"] == "test"
        assert row["metric"] == "r2"
        assert row["val_score"] == pytest.approx(0.95)
        assert row["task_type"] == "regression"
        assert row["dataset_name"] == "wheat"
        assert row["prediction_id"] == "pred_001"

    def test_load_dataset_not_found(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_dataset("nonexistent")

class TestArrayStoreStats:
    """Stats returns correct counts and sizes."""

    def test_stats(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([
            _make_record(prediction_id="p1", dataset_name="wheat"),
            _make_record(prediction_id="p2", dataset_name="wheat"),
            _make_record(prediction_id="p3", dataset_name="corn"),
        ])

        s = store.stats()
        assert s["total_files"] == 2
        assert s["total_rows"] == 3
        assert s["total_bytes"] > 0
        assert "wheat" in s["datasets"]
        assert "corn" in s["datasets"]
        assert s["datasets"]["wheat"]["rows"] == 2
        assert s["datasets"]["corn"]["rows"] == 1

    def test_stats_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        s = store.stats()
        assert s["total_files"] == 0
        assert s["total_rows"] == 0
        assert s["total_bytes"] == 0

class TestArrayStoreAppend:
    """Verify that multiple save_batch calls append to existing files."""

    def test_append_to_existing(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1")])
        store.save_batch([_make_record(prediction_id="p2")])

        result = store.load_batch(["p1", "p2"], dataset_name="wheat")
        assert len(result) == 2

        s = store.stats()
        assert s["datasets"]["wheat"]["rows"] == 2

class TestArrayStoreLoadWithoutDatasetName:
    """load_batch/load_single without dataset_name scans all files."""

    def test_load_without_dataset_name(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([
            _make_record(prediction_id="p1", dataset_name="wheat"),
            _make_record(prediction_id="p2", dataset_name="corn"),
        ])

        # Load without specifying dataset_name
        result = store.load_batch(["p1", "p2"])
        assert len(result) == 2
        assert "p1" in result
        assert "p2" in result

    def test_load_single_without_dataset_name(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        store.save_batch([_make_record(prediction_id="p1", dataset_name="wheat")])

        arrays = store.load_single("p1")
        assert arrays is not None

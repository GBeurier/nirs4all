"""SQLite-validated compaction tests (debt handoff item #1, step 2).

``ArrayStore.compact()`` historically applied tombstones blindly. A stale tombstone —
one referencing a prediction that is still LIVE in committed SQLite, left behind by a
crash or rollback between a SQLite write and the matching tombstone write — would
physically remove arrays that live metadata still points to. The validated path
(``WorkspaceStore.compact_arrays()`` / ``compact(live_ids=...)``) never removes a
live id's arrays and drops the stale tombstone instead, making compaction
self-correcting. It also refuses to run inside an open ``transaction()`` block,
because a rollback after compaction would resurrect rows whose arrays are gone.
"""

from __future__ import annotations

import pytest

from nirs4all.pipeline.storage.array_store import ArrayStore
from tests.unit.pipeline.storage.test_array_store import _make_record
from tests.unit.pipeline.storage.test_delete_ordering_crash import (
    _create_full_run,
    _live_prediction_ids,
    _make_store,
    _seed_arrays,
)

# =========================================================================
# ArrayStore level
# =========================================================================

class TestArrayStoreValidatedCompact:
    def test_live_id_kept_and_stale_tombstone_dropped_dead_id_removed(self, tmp_path):
        store = ArrayStore(tmp_path)
        store.save_batch([_make_record("live_1"), _make_record("dead_1")])
        store.delete_batch({"live_1", "dead_1"})  # live_1's tombstone is stale

        stats = store.compact(live_ids={"live_1"})

        assert stats["wheat"]["rows_removed"] == 1
        loaded = store.load_batch(["live_1", "dead_1"])
        assert "live_1" in loaded and "dead_1" not in loaded
        # Both tombstones are cleared: dead_1 applied, live_1 dropped as stale.
        assert store._read_tombstones() == {}

    def test_blind_compact_unchanged_without_live_ids(self, tmp_path):
        store = ArrayStore(tmp_path)
        store.save_batch([_make_record("p1")])
        store.delete_batch({"p1"})

        stats = store.compact()  # legacy blind behaviour

        assert stats["wheat"]["rows_removed"] == 1
        assert store.load_batch(["p1"]) == {}

# =========================================================================
# WorkspaceStore level
# =========================================================================

class TestWorkspaceStoreCompactArrays:
    def test_stale_tombstone_cannot_remove_live_rows_arrays(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        # Simulate the rollback/upsert-crash leftover: a tombstone covering a row
        # that is still live in SQLite.
        store._array_store.delete_batch({ids["pred_id"]})

        store.compact_arrays()

        assert _live_prediction_ids(store) == {ids["pred_id"]}
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        assert store._array_store._read_tombstones() == {}

    def test_reclaims_arrays_of_deleted_predictions(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        assert store.delete_prediction(ids["pred_id"]) is True
        stats = store.compact_arrays()

        assert stats["wheat"]["rows_removed"] == 1
        assert store._array_store.load_batch([ids["pred_id"]]) == {}
        assert store._array_store._read_tombstones() == {}

    def test_refuses_inside_open_transaction(self, tmp_path):
        store = _make_store(tmp_path)
        with store.transaction():
            with pytest.raises(RuntimeError, match="transaction"):
                store.compact_arrays()
        # After commit it runs fine.
        assert store.compact_arrays() == {}

    def test_live_snapshot_taken_under_process_lock(self, tmp_path, monkeypatch):
        """Regression for the snapshot/lock race: a delete-then-tombstone landing
        just before compaction acquires the process lock must be RECLAIMED — with a
        pre-lock snapshot the fresh tombstone looked stale, got cleared, and the
        arrays leaked as permanent orphans."""
        import contextlib as _ctx

        from nirs4all.pipeline.storage.array_store import ArrayStore

        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        real_lock = ArrayStore._process_lock
        fired = {"done": False}

        @_ctx.contextmanager
        def racing_lock(array_store_self):
            with real_lock(array_store_self):
                if not fired["done"]:
                    fired["done"] = True
                    # Simulate a concurrent process completing delete-then-tombstone
                    # (SQLite-first per Step 1) immediately before our acquisition.
                    store._conn.execute(
                        "DELETE FROM predictions WHERE prediction_id = ?", [ids["pred_id"]]
                    )
                    tombs = array_store_self._read_tombstones()
                    tombs[ids["pred_id"]] = "race"
                    array_store_self._write_tombstones(tombs)
                yield

        monkeypatch.setattr(ArrayStore, "_process_lock", racing_lock)
        stats = store.compact_arrays()

        # The freshly-dead row is reclaimed; its tombstone is applied, not cleared.
        assert stats["wheat"]["rows_removed"] == 1
        assert store._array_store.load_batch([ids["pred_id"]]) == {}
        assert store._array_store._read_tombstones() == {}

# =========================================================================
# Auto-compaction (threshold-gated, after committed deletes)
# =========================================================================

class TestAutoCompaction:
    def test_below_threshold_keeps_tombstones_pending(self, tmp_path):
        """A single delete under the (default, high) threshold does not compact."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        assert store.delete_prediction(ids["pred_id"]) is True

        assert store._array_store.has_pending_tombstones()
        # Arrays still physically present until a compaction runs.
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])

    def test_threshold_reached_triggers_compaction(self, tmp_path, monkeypatch):
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        monkeypatch.setattr(WorkspaceStore, "AUTO_COMPACT_TOMBSTONE_THRESHOLD", 1)
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        assert store.delete_prediction(ids["pred_id"]) is True

        # Compaction ran automatically: arrays reclaimed, no pending tombstones.
        assert not store._array_store.has_pending_tombstones()
        assert store._array_store.load_batch([ids["pred_id"]]) == {}

    def test_skipped_inside_open_transaction(self, tmp_path, monkeypatch):
        """Deletes inside a transaction never auto-compact (committed-state rule)."""
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        monkeypatch.setattr(WorkspaceStore, "AUTO_COMPACT_TOMBSTONE_THRESHOLD", 1)
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        with store.transaction():
            store.delete_prediction(ids["pred_id"])

        # The guard skipped compaction; the tombstone is still pending after commit
        # (the next delete, manual compact_arrays, or reopen reclaims it).
        assert store._array_store.has_pending_tombstones()
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])

    def test_delete_run_call_site_triggers_auto_compaction(self, tmp_path, monkeypatch):
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        monkeypatch.setattr(WorkspaceStore, "AUTO_COMPACT_TOMBSTONE_THRESHOLD", 1)
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        store.delete_run(ids["run_id"], delete_artifacts=False)

        assert not store._array_store.has_pending_tombstones()
        assert store._array_store.load_batch([ids["pred_id"]]) == {}

    def test_metadata_only_upsert_tombstone_resolved_by_next_auto_compaction(self, tmp_path, monkeypatch):
        """flush() never auto-compacts (it runs inside a transaction). A
        metadata-only upsert tombstones the overwritten id but skips save_batch
        (no arrays to write), so the stale tombstone stays pending. It counts
        toward the threshold, and the next delete-triggered compaction resolves it
        SAFELY: the live row's arrays are kept and the stale tombstone dropped."""
        import numpy as np

        from nirs4all.data.predictions import Predictions
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
        from tests.unit.data.test_predictions_store import _setup_store_hierarchy

        store = _make_store(tmp_path)
        pipeline_id, chain_id = _setup_store_hierarchy(store)
        preds = Predictions(store=store)

        def buffer(with_arrays: bool) -> None:
            preds.add_prediction(
                dataset_name="wheat", model_name="PLS", model_classname="PLSRegression",
                fold_id=0, partition="val",
                y_true=np.array([1.0, 2.0]) if with_arrays else None,
                y_pred=np.array([1.1, 2.1]) if with_arrays else None,
                val_score=0.1, metric="rmse", task_type="regression",
                n_samples=2, n_features=10,
            )

        # 1st flush: full row with arrays.
        buffer(with_arrays=True)
        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)
        pred_id = store.query_predictions()["prediction_id"][0]
        assert pred_id in store.array_store.load_batch([pred_id])

        # 2nd flush: metadata-only upsert of the SAME natural key -> tombstones the
        # id but writes no arrays, leaving a stale tombstone pending.
        preds._buffer.clear()
        buffer(with_arrays=False)
        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)
        assert store.array_store.has_pending_tombstones()
        assert pred_id in store.array_store.load_batch([pred_id])  # not applied

        # Next delete-triggered auto-compaction (threshold counts the stale one).
        monkeypatch.setattr(WorkspaceStore, "AUTO_COMPACT_TOMBSTONE_THRESHOLD", 1)
        ids2 = _create_full_run(store, dataset_name="corn")
        _seed_arrays(store, ids2["pred_id"], dataset_name="corn")
        store.delete_prediction(ids2["pred_id"])

        # Stale tombstone dropped, live arrays KEPT; dead row reclaimed.
        assert not store.array_store.has_pending_tombstones()
        assert pred_id in store.array_store.load_batch([pred_id])
        assert store.array_store.load_batch([ids2["pred_id"]]) == {}

    def test_auto_compact_failure_does_not_break_the_delete(self, tmp_path, monkeypatch):
        from nirs4all.pipeline.storage.array_store import ArrayStore
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        monkeypatch.setattr(WorkspaceStore, "AUTO_COMPACT_TOMBSTONE_THRESHOLD", 1)
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        def boom(self, dataset_name=None, live_ids=None):
            raise RuntimeError("simulated compaction failure")

        monkeypatch.setattr(ArrayStore, "_compact_unlocked", boom)

        assert store.delete_prediction(ids["pred_id"]) is True  # delete unaffected
        assert store._array_store.has_pending_tombstones()  # left for later repair

# =========================================================================
# Startup reconciliation (WorkspaceStore.__init__, gated on pending tombstones)
# =========================================================================

class TestStartupReconciliation:
    def test_reopen_applies_pending_tombstones(self, tmp_path):
        """Tombstones left behind by a session that never compacted are applied on
        the next open: dead arrays reclaimed, tombstone file cleared."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        assert store.delete_prediction(ids["pred_id"]) is True
        assert store._array_store.has_pending_tombstones()
        store.close()

        reopened = _make_store(tmp_path)
        assert not reopened._array_store.has_pending_tombstones()
        assert reopened._array_store.load_batch([ids["pred_id"]]) == {}
        reopened.close()

    def test_reopen_protects_live_rows_with_stale_tombstone(self, tmp_path):
        """A stale tombstone (crash/rollback leftover) covering a live row is dropped
        on reopen and the arrays survive."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        store._array_store.delete_batch({ids["pred_id"]})  # stale: row is live
        store.close()

        reopened = _make_store(tmp_path)
        assert not reopened._array_store.has_pending_tombstones()
        assert ids["pred_id"] in reopened._array_store.load_batch([ids["pred_id"]])
        reopened.close()

    def test_clean_open_skips_reconciliation(self, tmp_path, monkeypatch):
        """No pending tombstones -> the gate never calls compact_arrays."""
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

        calls: list[str] = []
        original = WorkspaceStore.compact_arrays
        monkeypatch.setattr(
            WorkspaceStore, "compact_arrays",
            lambda self, dataset_name=None: calls.append("called") or original(self, dataset_name),
        )
        store = _make_store(tmp_path)
        assert calls == []
        store.close()

    def test_failed_reconciliation_never_blocks_opening(self, tmp_path, monkeypatch):
        """A broken reconciliation logs and the workspace still opens."""
        from nirs4all.pipeline.storage.array_store import ArrayStore

        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        store.delete_prediction(ids["pred_id"])  # leaves a pending tombstone
        store.close()

        def boom(self, dataset_name=None, live_ids=None):
            raise RuntimeError("simulated reconciliation failure")

        monkeypatch.setattr(ArrayStore, "_compact_unlocked", boom)
        reopened = _make_store(tmp_path)  # must not raise
        # Store is functional (the run row survives; only the prediction was deleted)
        assert reopened.get_run(ids["run_id"]) is not None
        # Reconciliation failed, so the tombstone is still pending for the next open.
        assert reopened._array_store.has_pending_tombstones()
        reopened.close()

# =========================================================================
# Predictions facade level
# =========================================================================

class TestPredictionsCompactValidated:
    def test_facade_compact_protects_live_rows_from_stale_tombstone(self, tmp_path):
        """Predictions.compact() must go through the validated path: a stale
        tombstone covering a live row keeps the arrays and drops the tombstone."""
        from nirs4all.data.predictions import Predictions

        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        store._array_store.delete_batch({ids["pred_id"]})  # stale: row is live

        Predictions(store=store).compact()

        assert _live_prediction_ids(store) == {ids["pred_id"]}
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        assert store._array_store._read_tombstones() == {}

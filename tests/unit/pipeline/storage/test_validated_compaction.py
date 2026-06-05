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

"""Crash-injection tests for cross-store delete ordering (debt handoff item #1, step 1).

The workspace is three independent stores (SQLite metadata, Parquet arrays, joblib
artifacts) with no transaction spanning them. Every delete path therefore orders its
two non-transactional steps SQLite-first: the SQLite rows are deleted (autocommitted)
BEFORE the ArrayStore tombstone is written. A crash between the two steps can then
only ORPHAN arrays — rows the committed SQLite no longer references, which are
harmless (all reads resolve through SQLite) and reclaimable later. The reverse
order — the pre-campaign behaviour — could leave LIVE SQLite rows whose arrays a
later ``ArrayStore.compact()`` would physically remove, because compaction applies
tombstones without consulting SQLite.

The "crash" is injected by monkeypatching ``ArrayStore.delete_batch`` to raise,
approximating a process death between the SQLite commit and the tombstone write.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Helpers
# =========================================================================

def _make_store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspace")

def _create_full_run(store: WorkspaceStore, *, dataset_name: str = "wheat") -> dict:
    """Create a run -> pipeline -> chain -> prediction hierarchy and return all IDs."""
    run_id = store.begin_run("crash_run", config={"metric": "rmse"}, datasets=[{"name": dataset_name}])
    pipeline_id = store.begin_pipeline(
        run_id=run_id,
        name="0001_pls_crash",
        expanded_config=[{"step": "MinMaxScaler"}, {"model": "PLSRegression"}],
        generator_choices=[],
        dataset_name=dataset_name,
        dataset_hash="abc123",
    )
    chain_id = store.save_chain(
        pipeline_id=pipeline_id,
        steps=[{"step_idx": 0, "operator_class": "PLSRegression", "params": {}, "artifact_id": None, "stateless": True}],
        model_step_idx=0,
        model_class="sklearn.cross_decomposition.PLSRegression",
        preprocessings="raw",
        fold_strategy="per_fold",
        fold_artifacts={},
        shared_artifacts={},
    )
    pred_id = store.save_prediction(
        pipeline_id=pipeline_id,
        chain_id=chain_id,
        dataset_name=dataset_name,
        model_name="PLSRegression",
        model_class="sklearn.cross_decomposition.PLSRegression",
        fold_id="fold_0",
        partition="val",
        val_score=0.12,
        test_score=0.15,
        train_score=0.08,
        metric="rmse",
        task_type="regression",
        n_samples=10,
        n_features=20,
        scores={"val": {"rmse": 0.12}},
        best_params={},
        branch_id=None,
        branch_name=None,
        exclusion_count=0,
        exclusion_rate=0.0,
        preprocessings="raw",
    )
    return {"run_id": run_id, "pipeline_id": pipeline_id, "chain_id": chain_id, "pred_id": pred_id}

def _seed_arrays(store: WorkspaceStore, pred_id: str, dataset_name: str = "wheat") -> None:
    """Write prediction arrays for *pred_id* into the ArrayStore."""
    rng = np.random.default_rng(42)
    store._array_store.save_batch([{
        "prediction_id": pred_id,
        "dataset_name": dataset_name,
        "model_name": "PLSRegression",
        "fold_id": "fold_0",
        "partition": "val",
        "metric": "rmse",
        "val_score": 0.12,
        "task_type": "regression",
        "y_true": rng.standard_normal(10),
        "y_pred": rng.standard_normal(10),
        "y_proba": None,
        "sample_indices": np.arange(10, dtype=np.int32),
        "weights": None,
    }])

def _crash_delete_batch(store: WorkspaceStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ArrayStore.delete_batch die before writing the tombstone."""
    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash before tombstone write")

    monkeypatch.setattr(store._array_store, "delete_batch", boom)

def _live_prediction_ids(store: WorkspaceStore) -> set[str]:
    df = store.query_predictions()
    return set(df["prediction_id"].to_list()) if not df.is_empty() else set()

# =========================================================================
# Crash between SQLite commit and tombstone write
# =========================================================================

class TestDeleteCrashLeavesOrphansNotDeadRows:
    """A crash between the SQLite delete and the tombstone write must leave
    orphaned arrays (safe) — never live SQLite rows with a covering tombstone."""

    def test_delete_run_crash(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        _crash_delete_batch(store, monkeypatch)

        with pytest.raises(RuntimeError, match="simulated crash"):
            store.delete_run(ids["run_id"], delete_artifacts=False)

        # SQLite side is already committed: run + prediction rows are gone.
        assert store.get_run(ids["run_id"]) is None
        assert _live_prediction_ids(store) == set()
        # Arrays survive as orphans (readable, but unreachable through SQLite)...
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        # ...with NO tombstone — so no later compaction can act on stale state.
        assert store._array_store._read_tombstones() == {}
        # Reconciliation sees exactly one orphan and no dead links.
        report = store._array_store.integrity_check(expected_ids=_live_prediction_ids(store))
        assert report["orphan_ids"] == [ids["pred_id"]]
        assert report["missing_ids"] == []

    def test_delete_prediction_crash(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        _crash_delete_batch(store, monkeypatch)

        with pytest.raises(RuntimeError, match="simulated crash"):
            store.delete_prediction(ids["pred_id"])

        assert _live_prediction_ids(store) == set()
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        assert store._array_store._read_tombstones() == {}

    def test_upsert_crash(self, tmp_path, monkeypatch):
        """save_prediction's upsert deletes the old row before re-inserting; a crash
        after the SQLite delete must not leave a tombstone covering a live row."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        _crash_delete_batch(store, monkeypatch)

        with pytest.raises(RuntimeError, match="simulated crash"):
            store.save_prediction(
                pipeline_id=ids["pipeline_id"],
                chain_id=ids["chain_id"],
                dataset_name="wheat",
                model_name="PLSRegression",
                model_class="sklearn.cross_decomposition.PLSRegression",
                fold_id="fold_0",
                partition="val",  # same natural key -> upsert path
                val_score=0.10,
                test_score=0.14,
                train_score=0.07,
                metric="rmse",
                task_type="regression",
                n_samples=10,
                n_features=20,
                scores={"val": {"rmse": 0.10}},
                best_params={},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
                preprocessings="raw",
            )

        # The old row was deleted (committed) and the re-insert never ran: no live
        # row exists, the arrays are orphaned, and no tombstone was written.
        assert _live_prediction_ids(store) == set()
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        assert store._array_store._read_tombstones() == {}

    def test_pk_collision_crash(self, tmp_path, monkeypatch):
        """save_prediction with an explicit prediction_id colliding with an existing
        row under a DIFFERENT natural key takes the PK-guard delete path; a crash
        after the SQLite delete must not leave a tombstone covering a live row."""
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])
        _crash_delete_batch(store, monkeypatch)

        with pytest.raises(RuntimeError, match="simulated crash"):
            store.save_prediction(
                pipeline_id=ids["pipeline_id"],
                chain_id=ids["chain_id"],
                dataset_name="wheat",
                model_name="PLSRegression",
                model_class="sklearn.cross_decomposition.PLSRegression",
                fold_id="fold_1",
                partition="test",  # different natural key -> misses the upsert lookup
                prediction_id=ids["pred_id"],  # ...but collides on the primary key
                val_score=0.20,
                test_score=0.25,
                train_score=0.18,
                metric="rmse",
                task_type="regression",
                n_samples=10,
                n_features=20,
                scores={"test": {"rmse": 0.25}},
                best_params={},
                branch_id=None,
                branch_name=None,
                exclusion_count=0,
                exclusion_rate=0.0,
                preprocessings="raw",
            )

        assert _live_prediction_ids(store) == set()
        assert ids["pred_id"] in store._array_store.load_batch([ids["pred_id"]])
        assert store._array_store._read_tombstones() == {}

# =========================================================================
# Happy path: the normal delete -> tombstone -> compact flow still works
# =========================================================================

class TestDeleteThenCompactStillReclaims:
    def test_delete_prediction_then_compact_reclaims_arrays(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _create_full_run(store)
        _seed_arrays(store, ids["pred_id"])

        assert store.delete_prediction(ids["pred_id"]) is True

        # Tombstone written after the SQLite delete; row already gone.
        assert _live_prediction_ids(store) == set()
        assert ids["pred_id"] in store._array_store._read_tombstones()

        stats = store._array_store.compact()
        assert stats["wheat"]["rows_removed"] == 1
        assert store._array_store.load_batch([ids["pred_id"]]) == {}
        assert store._array_store._read_tombstones() == {}

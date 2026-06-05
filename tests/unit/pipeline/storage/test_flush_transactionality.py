"""Flush transactionality tests (debt handoff item #1, step 3).

``Predictions.flush()`` wraps its row-by-row SQLite writes AND the final Parquet
batch in one ``WorkspaceStore.transaction()``. A crash anywhere mid-flush therefore
rolls the SQLite metadata back: the workspace can at worst gain orphan Parquet
arrays (harmless — reads resolve through SQLite), never committed prediction rows
without arrays ("dead links", which previously required clean_dead_links repair).
``transaction()`` is re-entrant — an inner block joins the enclosing transaction —
so flush keeps working unchanged inside the orchestrator's transaction.

Also covers the clean_dead_links generalization: array orphans are now tombstoned
and physically reclaimed through the validated compaction path.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from tests.unit.data.test_predictions_store import _make_predictions_with_store
from tests.unit.pipeline.storage.test_array_store import _make_record
from tests.unit.pipeline.storage.test_delete_ordering_crash import _make_store


def _buffer_one(preds: Predictions, *, model_name: str = "PLS_1", op_counter: int = 0) -> None:
    preds.add_prediction(
        dataset_name="wheat",
        dataset_path="/data/wheat",
        config_name="pls_config",
        config_path="/configs/pls.yaml",
        pipeline_uid=f"pipe_{op_counter:03d}",
        step_idx=0,
        op_counter=op_counter,
        model_name=model_name,
        model_classname="PLSRegression",
        fold_id=0,
        partition="val",
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.1, 2.1, 2.9]),
        val_score=0.1,
        test_score=0.12,
        train_score=0.08,
        metric="rmse",
        task_type="regression",
        n_samples=3,
        n_features=10,
    )

def _row_count(store) -> int:
    df = store.query_predictions()
    return 0 if df.is_empty() else len(df)

class TestFlushTransactionality:
    def test_crash_before_parquet_rolls_back_sqlite(self, tmp_path, monkeypatch):
        """Crash between the SQLite loop and the Parquet batch: no committed rows."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)
        _buffer_one(preds)

        def boom(*args, **kwargs):
            raise RuntimeError("simulated crash before parquet write")

        monkeypatch.setattr(store.array_store, "save_batch", boom)

        with pytest.raises(RuntimeError, match="simulated crash"):
            preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        # The pre-step-3 behaviour left dead links (rows without arrays); now the
        # transaction rolls the metadata back entirely.
        assert _row_count(store) == 0

    def test_crash_mid_sqlite_loop_rolls_back_everything(self, tmp_path, monkeypatch):
        """Crash on the 2nd of 2 rows: the 1st row must roll back too (atomic flush)."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)
        _buffer_one(preds, model_name="PLS_1", op_counter=0)
        _buffer_one(preds, model_name="PLS_2", op_counter=1)

        original = store.save_prediction
        calls = {"n": 0}

        def failing(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash mid-flush")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "save_prediction", failing)

        with pytest.raises(RuntimeError, match="simulated crash"):
            preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        assert _row_count(store) == 0

    def test_flush_joins_enclosing_transaction(self, tmp_path):
        """Inside an outer transaction() (the orchestrator pattern) flush works and
        everything is durable after the outer commit."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)
        _buffer_one(preds)

        with store.transaction():
            preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        assert _row_count(store) == 1
        pred_id = store.query_predictions()["prediction_id"][0]
        assert pred_id in store.array_store.load_batch([pred_id])

    def test_outer_rollback_after_flush_leaves_only_orphans(self, tmp_path):
        """If the OUTER transaction fails after flush, the SQLite rows roll back and
        the already-written Parquet arrays become harmless orphans."""
        preds, store, pipeline_id, chain_id = _make_predictions_with_store(tmp_path)
        _buffer_one(preds)

        with pytest.raises(RuntimeError, match="outer failure"):
            with store.transaction():
                preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)
                raise RuntimeError("outer failure after flush")

        assert _row_count(store) == 0
        # The arrays were written before the rollback: orphaned, not dead-linked.
        assert store.array_store.stats()["total_rows"] == 1
        report = store.array_store.integrity_check(expected_ids=set())
        assert len(report["orphan_ids"]) == 1 and report["missing_ids"] == []

class TestTransactionReentrancy:
    def test_nested_transaction_joins_outer(self, tmp_path):
        store = _make_store(tmp_path)
        with store.transaction():
            with store.transaction():  # joins; must not raise "within a transaction"
                run_id = store.begin_run("nested", config={}, datasets=[])
        assert store.get_run(run_id) is not None

class TestCleanDeadLinksReclaimsOrphans:
    def test_array_orphans_are_physically_reclaimed(self, tmp_path):
        """Array rows with no SQLite metadata are tombstoned and compacted away
        (previously they were only counted: compact() never touched untombstoned rows)."""
        store = _make_store(tmp_path)
        store.array_store.save_batch([_make_record("ghost_1")])

        report = Predictions(store=store).clean_dead_links()

        assert report["array_orphans_found"] == 1
        assert store.array_store.load_batch(["ghost_1"]) == {}
        assert store.array_store.stats()["total_rows"] == 0
        assert store.array_store._read_tombstones() == {}

"""Integration tests for persisted refit labels and contexts."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import polars as pl

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import ExecutionPhase, RuntimeContext
from nirs4all.pipeline.execution.executor import PipelineExecutor
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STANDALONE
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def test_refit_predictions_are_persisted_with_final_fold_and_context(tmp_path):
    """Flush path should persist REFIT overrides into store rows."""
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)

    try:
        run_id = store.begin_run("refit_contract", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="refit_pipe",
            expanded_config=[{"model": "DummyModel"}],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="hash-ds",
        )
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 2, "operator_class": "DummyModel", "params": {}, "stateless": False}],
            model_step_idx=2,
            model_class="DummyModel",
            preprocessings="pp_target",
            fold_strategy="shared",
            fold_artifacts={},
            shared_artifacts={},
            branch_path=None,
        )

        predictions = Predictions()
        predictions.add_prediction(
            dataset_name="ds",
            model_name="DummyModel",
            model_classname="DummyModel",
            fold_id="fold_0",
            partition="test",
            metric="rmse",
            y_true=np.array([1.0, 2.0], dtype=float),
            y_pred=np.array([1.1, 1.9], dtype=float),
            step_idx=2,
            preprocessings="pp_target",
        )

        runtime_context = RuntimeContext(
            phase=ExecutionPhase.REFIT,
            refit_fold_id="final",
            refit_context_name=REFIT_CONTEXT_STANDALONE,
        )

        executor = PipelineExecutor(step_runner=Mock(), mode="train")
        executor._flush_predictions_to_store(
            store=store,
            pipeline_id=pipeline_id,
            prediction_store=predictions,
            runtime_context=runtime_context,
        )

        persisted = store.query_predictions(pipeline_id=pipeline_id)
        assert len(persisted) == 1
        assert "refit_context" in persisted.columns
        assert "fold_id" in persisted.columns

        row = persisted.row(0, named=True)
        assert row["chain_id"] == chain_id
        assert row["fold_id"] == "final"
        assert row["refit_context"] == REFIT_CONTEXT_STANDALONE

        final_rows = persisted.filter(pl.col("fold_id") == "final")
        assert len(final_rows) == 1
        assert final_rows["refit_context"].null_count() == 0
    finally:
        store.close()

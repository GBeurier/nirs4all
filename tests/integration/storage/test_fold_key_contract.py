"""Integration tests for fold-key contract normalization across storage replay."""

from __future__ import annotations

import numpy as np

from nirs4all.pipeline.storage.chain_builder import ChainBuilder
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.pipeline.trace.execution_trace import ExecutionStep, ExecutionTrace


class _ShiftTransformer:
    """Simple transformer used for replay assertions."""

    def __init__(self, shift: float):
        self.shift = float(shift)

    def transform(self, X):  # noqa: ANN001
        return np.asarray(X, dtype=float) + self.shift

class _ScaledSumModel:
    """Simple model used for replay assertions."""

    def __init__(self, scale: float):
        self.scale = float(scale)

    def predict(self, X):  # noqa: ANN001
        return np.asarray(X, dtype=float).sum(axis=1) * self.scale

def test_fold_key_roundtrip_trace_chain_store_replay(tmp_path):
    """Fold keys stay canonical across trace -> chain -> store -> replay."""
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)

    try:
        run_id = store.begin_run("fold_contract", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="fold_contract_pipe",
            expanded_config=[{"step": "Shift"}, {"model": "ScaledSum"}],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="hash-ds",
        )

        shared_artifact_id = store.save_artifact(
            _ShiftTransformer(shift=1.0),
            operator_class="tests._ShiftTransformer",
            artifact_type="transformer",
            format="joblib",
        )
        fold0_model_id = store.save_artifact(
            _ScaledSumModel(scale=10.0),
            operator_class="tests._ScaledSumModel",
            artifact_type="model",
            format="joblib",
        )
        final_model_id = store.save_artifact(
            _ScaledSumModel(scale=2.0),
            operator_class="tests._ScaledSumModel",
            artifact_type="model",
            format="joblib",
        )

        trace = ExecutionTrace(trace_id="trace_fold", pipeline_uid="fold_contract_pipe")

        transform_step = ExecutionStep(
            step_index=0,
            operator_type="transform",
            operator_class="Shift",
        )
        transform_step.artifacts.add_artifact(shared_artifact_id, is_primary=True)
        trace.add_step(transform_step)

        model_step = ExecutionStep(
            step_index=1,
            operator_type="model",
            operator_class="ScaledSum",
        )
        # Intentionally use legacy-style keys ("0", "final") to assert normalization.
        model_step.artifacts.add_fold_artifact("0", fold0_model_id)
        model_step.artifacts.add_fold_artifact("final", final_model_id)
        trace.add_step(model_step)
        trace.set_model_step(1)

        assert model_step.artifacts.fold_artifact_ids == {
            "fold_0": fold0_model_id,
            "fold_final": final_model_id,
        }

        chain_payload = ChainBuilder(trace).build()
        assert set(chain_payload["fold_artifacts"]) == {"fold_0", "fold_final"}

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=chain_payload["steps"],
            model_step_idx=chain_payload["model_step_idx"],
            model_class=chain_payload["model_class"],
            preprocessings=chain_payload["preprocessings"],
            fold_strategy=chain_payload["fold_strategy"],
            fold_artifacts=chain_payload["fold_artifacts"],
            shared_artifacts={"0": [shared_artifact_id]},
            branch_path=chain_payload["branch_path"],
        )

        stored_chain = store.get_chain(chain_id)
        assert stored_chain is not None
        assert set(stored_chain["fold_artifacts"]) == {"fold_0", "fold_final"}

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        y_pred = store.replay_chain(chain_id=chain_id, X=X)

        # replay_chain should prefer fold_final instead of averaging all fold models.
        expected = (X + 1.0).sum(axis=1) * 2.0
        assert np.allclose(y_pred, expected)
    finally:
        store.close()

"""Unit tests for the public ``nirs4all.predict`` helper."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import nirs4all
from nirs4all.api.result import PredictResult
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def test_public_predict_can_publish_workspace_prediction_evidence(monkeypatch, tmp_path) -> None:
    """``predict(..., save_to_workspace=True)`` writes reloadable replay evidence."""

    predict_module = importlib.import_module("nirs4all.api.predict")
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    def _fake_predict_from_model(**_kwargs):
        return PredictResult(
            y_pred=np.asarray([1.2, 1.8], dtype=float),
            sample_indices=np.asarray([0, 1], dtype=np.int64),
            model_name="PLSRegression",
            preprocessing_steps=["SNV"],
        )

    monkeypatch.setattr(predict_module, "_predict_from_model", _fake_predict_from_model)

    result = nirs4all.predict(
        model={"model_name": "PLSRegression"},
        data={"X": X},
        name="wheat",
        workspace_path=tmp_path / "workspace",
        save_to_workspace=True,
        workspace_result_metadata={"robustness_evidence": {"predictor_bundle": "model.n4a"}},
    )
    prediction_id = result.metadata["workspace_prediction_id"]
    restored = nirs4all.load_workspace_predict_result(tmp_path / "workspace", prediction_id)

    assert result.metadata["workspace_prediction_published"] is True
    assert result.metadata["workspace_path"] == str(tmp_path / "workspace")
    np.testing.assert_allclose(restored.y_pred, [1.2, 1.8])
    np.testing.assert_array_equal(restored.sample_indices, [0, 1])
    np.testing.assert_allclose(restored.metadata["X"], X)
    assert restored.model_name == "PLSRegression"
    assert restored.preprocessing_steps == ["SNV"]
    assert restored.robustness_evidence == {"predictor_bundle": "model.n4a"}
    assert restored.metadata["result_metadata"]["publisher"] == "nirs4all.predict"
    assert restored.spectral_replay_evidence_status["status"] == "ready_for_spectral_replay"


def test_public_predict_workspace_metadata_is_strict_json_native(monkeypatch, tmp_path) -> None:
    """``predict(..., save_to_workspace=True)`` rejects non-JSON-native metadata."""

    predict_module = importlib.import_module("nirs4all.api.predict")

    def _fake_predict_from_model(**_kwargs):
        return PredictResult(
            y_pred=np.asarray([1.2, 1.8], dtype=float),
            sample_indices=np.asarray([0, 1], dtype=np.int64),
            model_name="PLSRegression",
        )

    monkeypatch.setattr(predict_module, "_predict_from_model", _fake_predict_from_model)

    for payload in ({" bad": 1}, {"bad": object()}, {"bad": float("nan")}, {"bad": (1, 2)}):
        with pytest.raises(ValueError, match=r"save_workspace_predict_result.metadata"):
            nirs4all.predict(
                model={"model_name": "PLSRegression"},
                data={"X": np.asarray([[1.0], [2.0]], dtype=float)},
                workspace_path=tmp_path / "workspace",
                save_to_workspace=True,
                workspace_metadata=payload,
            )
        with pytest.raises(ValueError, match=r"save_workspace_predict_result.result_metadata"):
            nirs4all.predict(
                model={"model_name": "PLSRegression"},
                data={"X": np.asarray([[1.0], [2.0]], dtype=float)},
                workspace_path=tmp_path / "workspace",
                save_to_workspace=True,
                workspace_result_metadata=payload,
            )


def test_public_predict_publish_with_chain_id_uses_existing_pipeline(monkeypatch, tmp_path) -> None:
    """Publishing a chain replay prediction keeps the owning chain/pipeline."""

    predict_module = importlib.import_module("nirs4all.api.predict")
    workspace = tmp_path / "workspace"
    store = WorkspaceStore(workspace)
    try:
        run_id = store.begin_run("run", config={}, datasets=[{"name": "wheat"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pls",
            expanded_config=[{"model": "PLSRegression"}],
            generator_choices=[],
            dataset_name="wheat",
            dataset_hash="sha256:wheat",
        )
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLSRegression", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLSRegression",
            preprocessings='["SNV"]',
            fold_strategy="final",
            fold_artifacts={},
            shared_artifacts={},
            dataset_name="wheat",
        )
    finally:
        store.close()

    def _fake_predict_from_chain(**_kwargs):
        return PredictResult(
            y_pred=np.asarray([1.0, 2.0], dtype=float),
            model_name="PLSRegression",
            preprocessing_steps=["SNV"],
        )

    monkeypatch.setattr(predict_module, "_predict_from_chain", _fake_predict_from_chain)

    result = nirs4all.predict(
        chain_id=chain_id,
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        workspace_path=workspace,
        save_to_workspace=True,
    )

    store = WorkspaceStore(workspace)
    try:
        row = store.get_prediction(result.metadata["workspace_prediction_id"], load_arrays=True)
    finally:
        store.close()

    assert row is not None
    assert row["pipeline_id"] == pipeline_id
    assert row["chain_id"] == chain_id

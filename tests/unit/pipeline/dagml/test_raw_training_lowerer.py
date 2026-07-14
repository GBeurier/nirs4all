"""Unit tests for the P3-R1b raw-array DAG-ML training lowerer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator
from nirs4all.pipeline.dagml.fit_identity import normalize_fit_identity
from nirs4all.pipeline.dagml.raw_training_lowerer import (
    RawArrayDagMLTrainingCompiler,
    identity_from_fit_frame,
    lower_raw_array_training_contracts,
    raw_arrays_to_spectro_dataset,
)


class _FakeTrainingResult:
    outcome = {"ok": True}
    outputs = [{"output_id": "output:prediction"}]

    def export_portable_predictor_package(self, package_id: str) -> dict[str, str]:
        return {"package_id": package_id}


class _FakeNativeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute_training(self, *args: Any, **kwargs: Any) -> _FakeTrainingResult:
        self.calls.append({"args": args, "kwargs": kwargs})
        return _FakeTrainingResult()


def _pipeline() -> list[Any]:
    return [KFold(n_splits=2), {"model": KNeighborsRegressor(n_neighbors=1)}]


def _require_recent_dag_ml() -> None:
    dag_ml = pytest.importorskip("dag_ml")
    if not all(hasattr(dag_ml, name) for name in ("sign_training_request", "sample_relation_set_fingerprint_json")):
        pytest.skip("installed dag_ml does not expose the native training signer/fingerprint surface")


def test_raw_arrays_to_spectro_dataset_and_identity_preserve_sample_ids() -> None:
    X = np.arange(12, dtype=float).reshape(3, 4)
    y = np.arange(3, dtype=float)
    frame = normalize_fit_identity(X, y, sample_ids=["a", "b", "c"])

    dataset = raw_arrays_to_spectro_dataset(X, y, identity_frame=frame)
    identity = identity_from_fit_frame(frame)

    assert dataset.index_column("sample", {"partition": "train"}) == [0, 1, 2]
    assert identity.observation_ids() == ["a", "b", "c"]
    assert np.asarray(dataset.x_rows([2, 0], layout="2d")).tolist() == [X[2].tolist(), X[0].tolist()]


def test_lower_raw_array_training_contracts_builds_request_envelopes_and_influence() -> None:
    _require_recent_dag_ml()
    X = np.arange(20, dtype=float).reshape(5, 4)
    y = np.arange(5, dtype=float)
    frame = normalize_fit_identity(
        X,
        y,
        sample_ids=[f"s{index}" for index in range(5)],
        groups=["g0", "g0", "g1", "g1", "g1"],
        metadata={"instrument": ["demo"] * 5},
    )

    contracts = lower_raw_array_training_contracts(_pipeline(), X, y, identity_frame=frame)
    request = contracts.to_prepared().request

    assert request["request_fingerprint"] != "0" * 64
    assert request["campaign"]["root_seed"] == request["options"]["seed"]
    assert list(contracts.data_envelopes) == ["model:compat.0.x"]
    envelope = contracts.data_envelopes["model:compat.0.x"]
    assert envelope["data_content_fingerprint"] == request["data_identities"][0]["data_content_fingerprint"]
    assert envelope["target_content_fingerprint"] == request["data_identities"][0]["target_content_fingerprint"]
    assert {row["observation_id"] for row in contracts.relations["records"]} == {f"s{index}" for index in range(5)}
    assert {entry["kind"] for entry in contracts.training_influence["entries"]} == {"model_fit", "hpo_selection"}


def test_lower_raw_array_training_contracts_maps_deterministic_finetune_grid() -> None:
    _require_recent_dag_ml()
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.arange(6, dtype=float)
    frame = normalize_fit_identity(X, y, sample_ids=[f"s{index}" for index in range(6)])
    pipeline = [
        KFold(n_splits=3),
        {
            "model": KNeighborsRegressor(),
            "finetune_params": {
                "engine": "dag-ml",
                "metric": "mae",
                "direction": "minimize",
                "model_params": {"n_neighbors": [1, 2]},
            },
        },
    ]

    contracts = lower_raw_array_training_contracts(pipeline, X, y, identity_frame=frame)
    request = contracts.to_prepared().request

    assert request["options"]["selection"]["metric"] == {"name": "mae", "objective": "minimize"}
    generation = request["campaign"]["generation"]
    assert generation["strategy"] == "cartesian"
    assert [choice["value"] for choice in generation["dimensions"][0]["choices"]] == [{"n_neighbors": 1}, {"n_neighbors": 2}]


def test_lower_raw_array_training_contracts_rejects_adaptive_finetune_engine_until_adapter_exists() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    y = np.arange(5, dtype=float)
    frame = normalize_fit_identity(X, y, sample_ids=[f"s{index}" for index in range(5)])
    pipeline = [
        KFold(n_splits=2),
        {
            "model": KNeighborsRegressor(),
            "finetune_params": {
                "engine": "n4m",
                "model_params": {"n_neighbors": [1, 2]},
            },
        },
    ]

    with pytest.raises(ValueError, match="deterministic DAG-ML generation"):
        lower_raw_array_training_contracts(pipeline, X, y, identity_frame=frame)


def test_lower_raw_array_training_contracts_rejects_step_level_train_params_until_supported() -> None:
    X = np.arange(20, dtype=float).reshape(5, 4)
    y = np.arange(5, dtype=float)
    frame = normalize_fit_identity(X, y, sample_ids=[f"s{index}" for index in range(5)])
    pipeline = [
        KFold(n_splits=2),
        {
            "model": KNeighborsRegressor(),
            "train_params": {"sample_weight": [1.0] * 5},
        },
    ]

    with pytest.raises(NotImplementedError, match="train_params"):
        lower_raw_array_training_contracts(pipeline, X, y, identity_frame=frame)


def test_raw_array_training_compiler_integrates_with_estimator_fit() -> None:
    _require_recent_dag_ml()
    X = np.arange(20, dtype=float).reshape(5, 4)
    y = np.arange(5, dtype=float)
    client = _FakeNativeClient()
    estimator = DagMLPipelineEstimator(
        pipeline=_pipeline(),
        native_client=client,
        training_compiler=RawArrayDagMLTrainingCompiler(),
    )

    estimator.fit(X, y, sample_ids=[f"s{index}" for index in range(5)])

    request = client.calls[0]["args"][0]
    assert request["request_id"] == "training:nirs4all.raw_fit"
    assert client.calls[0]["kwargs"]["diagnostics"]["nirs4all_lowerer"] == "raw_array_p3_r1b"
    assert estimator.training_outcome_ == {"ok": True}


def test_raw_array_training_compiler_requires_splitter() -> None:
    X = np.arange(12, dtype=float).reshape(3, 4)
    y = np.arange(3, dtype=float)
    estimator = DagMLPipelineEstimator(
        pipeline=[{"model": KNeighborsRegressor(n_neighbors=1)}],
        native_client=_FakeNativeClient(),
        training_compiler=RawArrayDagMLTrainingCompiler(),
    )

    with pytest.raises(ValueError, match="requires a splitter"):
        estimator.fit(X, y, sample_ids=["a", "b", "c"])


def test_raw_array_training_compiler_executes_native_training_when_supported() -> None:
    _require_recent_dag_ml()
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.arange(6, dtype=float)
    estimator = DagMLPipelineEstimator(
        pipeline=[KFold(n_splits=3), {"model": KNeighborsRegressor(n_neighbors=1)}],
        training_compiler=RawArrayDagMLTrainingCompiler(dagml_module="dag_ml"),
        dagml_module="dag_ml",
    )

    estimator.fit(X, y, sample_ids=[f"s{index}" for index in range(6)])

    assert estimator.training_outcome_.to_dict()["refit"]["status"] == "completed"
    assert estimator.outputs_[0]["binding"]["binding_id"] == "output:prediction"
    assert estimator.predictor_package_.to_dict()["package_id"] == "outcome:nirs4all.raw_fit-predictor"


def test_raw_array_training_compiler_executes_native_deterministic_finetune_when_supported() -> None:
    _require_recent_dag_ml()
    X = np.arange(24, dtype=float).reshape(6, 4)
    y = np.arange(6, dtype=float)
    estimator = DagMLPipelineEstimator(
        pipeline=[
            KFold(n_splits=3),
            {
                "model": KNeighborsRegressor(),
                "finetune_params": {
                    "engine": "dag-ml",
                    "metric": "mae",
                    "direction": "minimize",
                    "model_params": {"n_neighbors": [1, 2]},
                },
            },
        ],
        training_compiler=RawArrayDagMLTrainingCompiler(dagml_module="dag_ml"),
        dagml_module="dag_ml",
    )

    estimator.fit(X, y, sample_ids=[f"s{index}" for index in range(6)])

    outcome = estimator.training_outcome_.to_dict()
    assert outcome["refit"]["status"] == "completed"
    assert len(outcome["execution_bundle"]["selections"]["selection:mae"]["ranked_candidates"]) == 2
    assert outcome["parameter_patches"] == [
        {
            "schema_version": 1,
            "node_id": "model:compat.0",
            "namespace": "operator",
            "path": ["n_neighbors"],
            "value": 1,
        }
    ]

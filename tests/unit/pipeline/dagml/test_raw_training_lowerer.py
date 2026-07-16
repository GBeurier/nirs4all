"""Unit tests for the P3-R1b raw-array DAG-ML training lowerer."""

from __future__ import annotations

import contextlib
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

with contextlib.suppress(ImportError):
    import torch

with contextlib.suppress(ImportError):
    import tensorflow as tf


if "torch" in globals():

    class _TinyTorchRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)
            torch.nn.init.zeros_(self.linear.weight)

        def forward(self, features: Any) -> Any:
            return self.linear(features)


if "tf" in globals():

    @tf.keras.utils.register_keras_serializable(package="nirs4all_tests")
    class _TinyTensorFlowRegressor(tf.keras.Model):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("name", "tiny_tensorflow_regressor")
            super().__init__(**kwargs)
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(
                1,
                use_bias=False,
                kernel_initializer="zeros",
            )

        def call(self, features: Any) -> Any:
            return self.dense(self.flatten(features))


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


def test_model_controller_routing_uses_framework_mro_without_substring_false_positives() -> None:
    from nirs4all.pipeline.dagml_bridge import (
        _PYTORCH_MODEL_CONTROLLER_ID,
        _TENSORFLOW_MODEL_CONTROLLER_ID,
        _model_controller_id,
    )

    class _SciKerasLike:
        pass

    _SciKerasLike.__module__ = "scikeras.wrappers"

    assert _model_controller_id(_SciKerasLike()) is None
    if "torch" in globals():
        assert _model_controller_id(_TinyTorchRegressor()) == _PYTORCH_MODEL_CONTROLLER_ID
    if "tf" in globals():
        assert _model_controller_id(_TinyTensorFlowRegressor()) == _TENSORFLOW_MODEL_CONTROLLER_ID


def test_differentiable_backend_seed_maps_wide_core_seed_deterministically() -> None:
    from nirs4all.pipeline.dagml.node_runner import (
        _PYTORCH_MODEL_CONTROLLER_ID,
        _TENSORFLOW_MODEL_CONTROLLER_ID,
        _seed_differentiable_backend,
    )

    seed = 2**63 + 17
    if "torch" in globals():
        _seed_differentiable_backend(_PYTORCH_MODEL_CONTROLLER_ID, seed)
        first_torch = torch.rand(3)
        _seed_differentiable_backend(_PYTORCH_MODEL_CONTROLLER_ID, seed)
        assert torch.equal(first_torch, torch.rand(3))
    if "tf" in globals():
        _seed_differentiable_backend(_TENSORFLOW_MODEL_CONTROLLER_ID, seed)
        first_tensorflow = tf.random.uniform((3,))
        _seed_differentiable_backend(_TENSORFLOW_MODEL_CONTROLLER_ID, seed)
        np.testing.assert_array_equal(first_tensorflow.numpy(), tf.random.uniform((3,)).numpy())


def test_node_runner_rejects_missing_registry_and_multiple_loss_requirements() -> None:
    from nirs4all.pipeline.dagml.node_runner import (
        _PYTORCH_MODEL_CONTROLLER_ID,
        _bind_training_loss,
    )

    task = {
        "phase": "FIT_CV",
        "required_loss_attestations": [{"phase": "FIT_CV", "loss_id": "loss@1"}],
    }
    with pytest.raises(RuntimeError, match="process-local loss registry"):
        _bind_training_loss(task, _PYTORCH_MODEL_CONTROLLER_ID, None)

    task["required_loss_attestations"].append(
        {"phase": "FIT_CV", "loss_id": "other-loss@1"}
    )
    with pytest.raises(NotImplementedError, match="one training loss per task"):
        _bind_training_loss(task, _PYTORCH_MODEL_CONTROLLER_ID, object())


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


@pytest.mark.torch
@pytest.mark.xdist_group("torch")
def test_raw_array_training_executes_local_torch_loss_and_attests_each_phase() -> None:
    _require_recent_dag_ml()
    if "torch" not in globals():
        pytest.skip("PyTorch not available")
    import dag_ml

    if not hasattr(dag_ml, "LocalImplementationRegistry"):
        pytest.skip("installed dag_ml does not expose local loss contracts yet")

    calls: list[tuple[tuple[int, ...], tuple[int, ...], bool, bool]] = []

    def squared_loss(target: Any, prediction: Any) -> Any:
        calls.append(
            (
                tuple(target.shape),
                tuple(prediction.shape),
                bool(target.requires_grad),
                bool(prediction.requires_grad),
            )
        )
        return torch.mean(torch.square(prediction - target))

    registry = dag_ml.LocalImplementationRegistry()
    loss_reference = registry.register_local_loss(
        {
            "schema_version": 1,
            "loss_id": "example.loss.nirs4all-torch-squared@1",
            "kind": "custom",
            "task_kinds": ["regression"],
            "prediction_kinds": ["regression_point"],
            "objective": "minimize",
            "reduction": "mean",
            "required_inputs": ["target", "prediction"],
            "capabilities": ["differentiable"],
            "parameters": {},
        },
        squared_loss,
        registry_key="loss:nirs4all:test-torch-squared",
        implementation_fingerprint="a" * 64,
        capabilities=["differentiable"],
    )
    bind_calls: list[tuple[str, str | None, str | None, int]] = []

    class _CountingRegistry:
        def bind_training_loss(
            self,
            task: dict[str, Any],
            *,
            role_index: int,
        ) -> Any:
            bind_calls.append(
                (
                    task["phase"],
                    task.get("fold_id"),
                    task.get("variant_id"),
                    role_index,
                )
            )
            return registry.bind_training_loss(task, role_index=role_index)

    role = {
        "schema_version": 1,
        "node_id": "model:compat.0",
        "output_id": "oof",
        "phases": ["FIT_CV", "REFIT"],
        "loss": loss_reference,
    }
    estimator = DagMLPipelineEstimator(
        pipeline=[
            KFold(n_splits=2),
            {
                "model": _TinyTorchRegressor(),
                "train_params": {
                    "epochs": 1,
                    "batch_size": 2,
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "patience": 1,
                    "verbose": 0,
                },
            },
        ],
        training_compiler=RawArrayDagMLTrainingCompiler(
            dagml_module="dag_ml",
            training_losses=(role,),
            local_implementations=_CountingRegistry(),
        ),
        dagml_module="dag_ml",
    )
    X = np.ones((6, 2), dtype=np.float32)
    y = np.ones(6, dtype=np.float32)

    estimator.fit(X, y, sample_ids=[f"s{index}" for index in range(6)])

    assert calls
    assert all(not target_requires_grad for _, _, target_requires_grad, _ in calls)
    assert any(prediction_requires_grad for _, _, _, prediction_requires_grad in calls)
    assert {phase for phase, _, _, _ in bind_calls} == {"FIT_CV", "REFIT"}
    assert all(role_index == 0 for _, _, _, role_index in bind_calls)
    outcome = estimator.training_outcome_.to_dict()
    model_lineage = [
        record
        for record in outcome["lineage"]
        if record["node_id"] == "model:compat.0"
        and record["phase"] in {"FIT_CV", "REFIT"}
    ]
    assert {record["phase"] for record in model_lineage} == {"FIT_CV", "REFIT"}
    assert all(
        [attestation["loss_id"] for attestation in record["loss_attestations"]]
        == ["example.loss.nirs4all-torch-squared@1"]
        for record in model_lineage
    )
    assert outcome["execution_bundle"]["refit_artifacts"][0][
        "training_loss_fingerprint"
    ]


@pytest.mark.tensorflow
@pytest.mark.xdist_group("tensorflow")
def test_raw_array_training_executes_local_tensorflow_loss_and_detaches_refit_artifact() -> None:
    _require_recent_dag_ml()
    if "tf" not in globals():
        pytest.skip("TensorFlow not available")
    import dag_ml

    if not hasattr(dag_ml, "LocalImplementationRegistry"):
        pytest.skip("installed dag_ml does not expose local loss contracts yet")

    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def squared_loss(target: Any, prediction: Any) -> Any:
        calls.append((tuple(target.shape), tuple(prediction.shape)))
        return tf.reduce_mean(tf.square(prediction - target))

    registry = dag_ml.LocalImplementationRegistry()
    loss_reference = registry.register_local_loss(
        {
            "schema_version": 1,
            "loss_id": "example.loss.nirs4all-tensorflow-squared@1",
            "kind": "custom",
            "task_kinds": ["regression"],
            "prediction_kinds": ["regression_point"],
            "objective": "minimize",
            "reduction": "mean",
            "required_inputs": ["target", "prediction"],
            "capabilities": ["differentiable"],
            "parameters": {},
        },
        squared_loss,
        registry_key="loss:nirs4all:test-tensorflow-squared",
        implementation_fingerprint="b" * 64,
        capabilities=["differentiable"],
    )
    bind_calls: list[tuple[str, str | None, str | None, int]] = []

    class _CountingRegistry:
        def bind_training_loss(
            self,
            task: dict[str, Any],
            *,
            role_index: int,
        ) -> Any:
            bind_calls.append(
                (
                    task["phase"],
                    task.get("fold_id"),
                    task.get("variant_id"),
                    role_index,
                )
            )
            return registry.bind_training_loss(task, role_index=role_index)

    estimator = DagMLPipelineEstimator(
        pipeline=[
            KFold(n_splits=2),
            {
                "model": _TinyTensorFlowRegressor(),
                "train_params": {
                    "epochs": 1,
                    "batch_size": 2,
                    "optimizer": "sgd",
                    "learning_rate": 0.1,
                    "patience": 1,
                    "best_model_memory": False,
                    "verbose": 0,
                },
            },
        ],
        training_compiler=RawArrayDagMLTrainingCompiler(
            dagml_module="dag_ml",
            training_losses=(
                {
                    "schema_version": 1,
                    "node_id": "model:compat.0",
                    "output_id": "oof",
                    "phases": ["FIT_CV", "REFIT"],
                    "loss": loss_reference,
                },
            ),
            local_implementations=_CountingRegistry(),
        ),
        dagml_module="dag_ml",
    )
    X = np.ones((6, 2), dtype=np.float32)
    y = np.ones(6, dtype=np.float32)

    estimator.fit(X, y, sample_ids=[f"s{index}" for index in range(6)])

    assert calls
    assert {phase for phase, _, _, _ in bind_calls} == {"FIT_CV", "REFIT"}
    assert all(role_index == 0 for _, _, _, role_index in bind_calls)
    outcome = estimator.training_outcome_.to_dict()
    model_lineage = [
        record
        for record in outcome["lineage"]
        if record["node_id"] == "model:compat.0"
        and record["phase"] in {"FIT_CV", "REFIT"}
    ]
    assert {record["phase"] for record in model_lineage} == {"FIT_CV", "REFIT"}
    assert all(
        [attestation["loss_id"] for attestation in record["loss_attestations"]]
        == ["example.loss.nirs4all-tensorflow-squared@1"]
        for record in model_lineage
    )


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

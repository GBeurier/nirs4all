"""Unit tests for native DAG-ML training contract assembly helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from nirs4all.pipeline.dagml.training_contracts import (
    DagMLTrainingRequestSpec,
    assemble_training_request,
    tcv1_fingerprint_without,
    training_data_identity_from_binding,
    validate_training_request_with_dagml,
)


def _dagml_training_fixture() -> dict:
    relative_path = Path("dag-ml/examples/fixtures/training/python_training_smoke.v1.json")
    fixture_path = next(
        (
            parent / relative_path
            for parent in Path(__file__).resolve().parents
            if (parent / relative_path).exists()
        ),
        None,
    )
    if fixture_path is None:
        pytest.skip("sibling dag-ml training fixture is not available")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_tcv1_request_fingerprint_matches_native_fixture() -> None:
    request = _dagml_training_fixture()["request"]

    assert tcv1_fingerprint_without(request, "request_fingerprint") == request["request_fingerprint"]


def test_training_data_identity_from_binding_matches_native_fixture() -> None:
    fixture = _dagml_training_fixture()
    request = fixture["request"]
    identity = request["data_identities"][0]
    binding = request["campaign"]["data_bindings"]["model:base"][0]

    rebuilt = training_data_identity_from_binding(
        binding,
        data_content_fingerprint=identity["data_content_fingerprint"],
        target_content_fingerprint=identity["target_content_fingerprint"],
    )

    assert rebuilt == identity


def test_assemble_training_request_reproduces_and_validates_native_fixture() -> None:
    fixture = _dagml_training_fixture()
    source = fixture["request"]
    spec = DagMLTrainingRequestSpec(
        request_id=source["request_id"],
        plan_id=source["plan_id"],
        graph=source["graph"],
        campaign=source["campaign"],
        controller_manifests=source["controller_manifests"],
        data_identities=source["data_identities"],
        selection_metric=source["options"]["selection"]["metric"]["name"],
        selection_objective=source["options"]["selection"]["metric"]["objective"],
        selection_output_id=source["options"]["selection_output_id"],
        output_requests=source["options"]["outputs"],
        seed=source["options"]["seed"],
        refit=source["options"]["refit"],
        scheduler_workers=source["options"]["scheduler"]["workers"],
        cpu_threads=source["options"]["resources"]["cpu_threads"],
        cv_artifacts=source["options"]["artifacts"]["cv_artifacts"],
        prediction_caches=source["options"]["artifacts"]["prediction_caches"],
        fitted_artifacts=source["options"]["artifacts"]["fitted_artifacts"],
        selection_required_metric_level=source["options"]["selection"]["required_metric_level"],
        selection_evaluation_scope=source["options"]["selection"]["evaluation_scope"],
    )

    assembled = assemble_training_request(spec)

    assert assembled == source
    dag_ml = pytest.importorskip("dag_ml")
    if not hasattr(dag_ml, "TrainingRequest"):
        pytest.skip("installed dag_ml does not expose TrainingRequest validation yet")
    validated = validate_training_request_with_dagml(assembled, dag_ml)
    assert validated.to_dict() == source


def test_assemble_training_request_transports_and_orders_loss_roles_without_fixture() -> None:
    later_role = _custom_training_loss_role()
    later_role["node_id"] = "model:z"
    later_role["output_id"] = None
    later_role["phases"] = ["REFIT", "FIT_CV"]
    earlier_role = deepcopy(later_role)
    earlier_role["node_id"] = "model:a"
    request = assemble_training_request(
        DagMLTrainingRequestSpec(
            request_id="training:test.loss-order",
            plan_id="plan:test.loss-order",
            graph={},
            campaign={},
            controller_manifests=(),
            data_identities=(),
            training_losses=(later_role, earlier_role),
            output_requests=(),
        )
    )

    assert [role["node_id"] for role in request["training_losses"]] == [
        "model:a",
        "model:z",
    ]
    assert all(
        role["phases"] == ["FIT_CV", "REFIT"]
        for role in request["training_losses"]
    )
    assert request["request_fingerprint"] == tcv1_fingerprint_without(
        request, "request_fingerprint"
    )


def test_assemble_training_request_validates_native_training_loss_role() -> None:
    fixture = _dagml_training_fixture()
    source = fixture["request"]
    controller_manifests = deepcopy(source["controller_manifests"])
    model_manifest = next(
        manifest
        for manifest in controller_manifests
        if manifest["controller_id"] == "controller:model.mock"
    )
    model_manifest["capabilities"] = sorted(
        {
            *model_manifest["capabilities"],
            "needs_python_gil",
            "supports_configurable_loss",
            "supports_custom_loss",
            "supports_differentiable_loss",
        }
    )
    training_loss = _custom_training_loss_role()
    spec = DagMLTrainingRequestSpec(
        request_id=source["request_id"],
        plan_id=source["plan_id"],
        graph=source["graph"],
        campaign=source["campaign"],
        controller_manifests=controller_manifests,
        data_identities=source["data_identities"],
        training_losses=(training_loss,),
        selection_metric=source["options"]["selection"]["metric"]["name"],
        selection_objective=source["options"]["selection"]["metric"]["objective"],
        selection_output_id=source["options"]["selection_output_id"],
        output_requests=source["options"]["outputs"],
        seed=source["options"]["seed"],
        refit=source["options"]["refit"],
        scheduler_workers=source["options"]["scheduler"]["workers"],
        cpu_threads=source["options"]["resources"]["cpu_threads"],
        cv_artifacts=source["options"]["artifacts"]["cv_artifacts"],
        prediction_caches=source["options"]["artifacts"]["prediction_caches"],
        fitted_artifacts=source["options"]["artifacts"]["fitted_artifacts"],
        selection_required_metric_level=source["options"]["selection"]["required_metric_level"],
        selection_evaluation_scope=source["options"]["selection"]["evaluation_scope"],
    )

    assembled = assemble_training_request(spec)

    assert assembled["training_losses"] == [training_loss]
    assert assembled["request_fingerprint"] == tcv1_fingerprint_without(
        assembled, "request_fingerprint"
    )
    dag_ml = pytest.importorskip("dag_ml")
    if not hasattr(dag_ml, "LocalImplementationRegistry"):
        pytest.skip("installed dag_ml does not expose local loss contracts yet")
    validated = validate_training_request_with_dagml(assembled, dag_ml)
    assert validated.to_dict()["training_losses"] == [training_loss]


def test_tcv1_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="NaN and infinity"):
        tcv1_fingerprint_without({"x": float("nan"), "fp": "0" * 64}, "fp")


def _custom_training_loss_role() -> dict:
    spec = {
        "schema_version": 1,
        "loss_id": "example.loss.asymmetric@1",
        "kind": "custom",
        "task_kinds": ["regression"],
        "prediction_kinds": ["regression_point"],
        "objective": "minimize",
        "reduction": "mean",
        "required_inputs": ["target", "prediction"],
        "capabilities": ["differentiable"],
        "parameters": {"over_weight": 1.0, "under_weight": 2.0},
        "spec_fingerprint": "cf661225cc7137ab5ef9b87871ed5736a8479dd21587b7e17150c442b1e43eb0",
    }
    implementation = {
        "schema_version": 1,
        "semantic_kind": "loss",
        "semantic_id": spec["loss_id"],
        "semantic_fingerprint": spec["spec_fingerprint"],
        "provider_id": "provider:python-local",
        "binding_id": "binding:python",
        "implementation_version": "1.0.0",
        "implementation_fingerprint": "1f4c71b0b758c5ed25b4e38b132b9ad56fb2f5ff2cf490f7eb8786c4350a62f7",
        "supported_controller_families": [],
        "runtime_requirements": [],
        "capabilities": ["deterministic", "differentiable", "needs_gil"],
        "portability": "host_local",
        "replayability": "registry_required",
        "registry_key": "loss:run-123:asymmetric",
        "descriptor_fingerprint": "031c7b120740620dade9ba14a5f2e142831ecb1be47ea7181f68511dfafdd807",
    }
    return {
        "schema_version": 1,
        "node_id": "model:base",
        "output_id": "oof",
        "phases": ["FIT_CV", "REFIT"],
        "loss": {"spec": spec, "implementation": implementation},
    }

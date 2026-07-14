"""Unit tests for native DAG-ML training contract assembly helpers."""

from __future__ import annotations

import json
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
    fixture_path = Path(__file__).resolve().parents[5] / "dag-ml" / "examples" / "fixtures" / "training" / "python_training_smoke.v1.json"
    if not fixture_path.exists():
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


def test_tcv1_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="NaN and infinity"):
        tcv1_fingerprint_without({"x": float("nan"), "fp": "0" * 64}, "fp")

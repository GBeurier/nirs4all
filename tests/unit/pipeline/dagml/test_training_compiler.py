"""Unit tests for the prepared DAG-ML training compiler seam."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from nirs4all.pipeline.dagml.estimator import DagMLPipelineEstimator, DagMLTrainingExecution
from nirs4all.pipeline.dagml.fit_identity import normalize_fit_identity
from nirs4all.pipeline.dagml.training_compiler import (
    DagMLPreparedTrainingContracts,
    DagMLTrainingContractFactoryCompiler,
    DagMLTrainingRequestCompiler,
    DagMLTrainingRequestContracts,
    PreparedDagMLTrainingCompiler,
    compile_prepared_training_contracts,
)
from nirs4all.pipeline.dagml.training_contracts import DagMLTrainingRequestSpec


class _FakeTrainingResult:
    outcome = {"ok": True}
    outputs = [{"output_id": "prediction"}]

    def export_portable_predictor_package(self, package_id: str) -> dict[str, str]:
        return {"package_id": package_id}


class _FakeNativeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute_training(self, *args: Any, **kwargs: Any) -> _FakeTrainingResult:
        self.calls.append({"args": args, "kwargs": kwargs})
        return _FakeTrainingResult()


def _contracts(**overrides: Any) -> DagMLPreparedTrainingContracts:
    values: dict[str, Any] = {
        "request": {"phase": "train"},
        "data_envelopes": {"fit.data": {"rows": 3}},
        "relations": {"relations": []},
        "training_influence": {"entries": []},
        "op_callback": lambda task: {"task": task},
        "outcome_id": "outcome-1",
        "run_id": "run-1",
        "bundle_id": "bundle-1",
        "warnings": ("prepared-warning",),
        "diagnostics": {"source": "prepared"},
    }
    values.update(overrides)
    return DagMLPreparedTrainingContracts(**values)


def test_compile_prepared_contracts_adds_identity_diagnostics() -> None:
    identity_frame = normalize_fit_identity(
        np.ones((3, 2)),
        np.arange(3),
        sample_ids=["s1", "s2", "s3"],
        groups=["g1", "g1", "g2"],
        metadata={"instrument": ["a", "a", "b"]},
    )

    execution = compile_prepared_training_contracts(
        _contracts(),
        identity_frame=identity_frame,
        additional_diagnostics={"compiler": "prepared"},
    )

    assert isinstance(execution, DagMLTrainingExecution)
    assert execution.request == {"phase": "train"}
    assert execution.data_envelopes == {"fit.data": {"rows": 3}}
    assert execution.warnings == ("prepared-warning",)
    assert execution.diagnostics == {
        "compiler": "prepared",
        "nirs4all_fit_identity_explicit_sample_ids": True,
        "nirs4all_fit_identity_fingerprint": identity_frame.fingerprint,
        "nirs4all_fit_identity_n_samples": 3,
        "source": "prepared",
    }


def test_compile_prepared_contracts_warns_for_compatibility_ids() -> None:
    identity_frame = normalize_fit_identity(np.ones((2, 2)), np.arange(2))

    execution = compile_prepared_training_contracts(
        _contracts(warnings=()),
        identity_frame=identity_frame,
    )

    assert len(execution.warnings) == 1
    assert "compatibility sample ids" in execution.warnings[0]
    assert execution.diagnostics["nirs4all_fit_identity_explicit_sample_ids"] is False


def test_prepared_compiler_integrates_with_estimator_fit() -> None:
    client = _FakeNativeClient()
    compiler = PreparedDagMLTrainingCompiler(
        _contracts(),
        additional_diagnostics={"lane": "p3-r0"},
    )
    estimator = DagMLPipelineEstimator(
        pipeline=("prepared",),
        native_client=client,
        training_compiler=compiler,
    )

    estimator.fit(
        np.ones((3, 4)),
        np.arange(3),
        sample_ids=["s1", "s2", "s3"],
        groups=["batch-a", "batch-a", "batch-b"],
        metadata=[{"source": "a"}, {"source": "a"}, {"source": "b"}],
    )

    assert client.calls[0]["args"][:4] == (
        {"phase": "train"},
        {"fit.data": {"rows": 3}},
        {"relations": []},
        {"entries": []},
    )
    assert client.calls[0]["kwargs"]["diagnostics"]["lane"] == "p3-r0"
    assert client.calls[0]["kwargs"]["diagnostics"]["nirs4all_fit_identity_n_samples"] == 3
    assert estimator.training_outcome_ == {"ok": True}
    assert estimator.predictor_package_ == {"package_id": "outcome-1-predictor"}


def test_contract_factory_compiler_receives_normalized_fit_identity() -> None:
    client = _FakeNativeClient()
    factory_calls: list[dict[str, Any]] = []

    def factory(
        estimator: DagMLPipelineEstimator,
        X: Any,
        y: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
        identity_frame: Any = None,
    ) -> DagMLPreparedTrainingContracts:
        factory_calls.append(
            {
                "estimator": estimator,
                "X": X,
                "y": y,
                "sample_ids": sample_ids,
                "groups": groups,
                "metadata": metadata,
                "identity_frame": identity_frame,
            }
        )
        return _contracts(diagnostics={"factory": True})

    estimator = DagMLPipelineEstimator(
        pipeline=("future-lowered",),
        native_client=client,
        training_compiler=DagMLTrainingContractFactoryCompiler(factory),
    )

    estimator.fit(
        np.ones((2, 3)),
        np.arange(2),
        sample_ids=["s1", "s2"],
        groups=[1, 2],
        metadata={"instrument": ["a", "b"]},
    )

    assert factory_calls[0]["sample_ids"] == ("s1", "s2")
    assert factory_calls[0]["groups"] == ("1", "2")
    assert factory_calls[0]["metadata"] == {
        "s1": {"instrument": "a"},
        "s2": {"instrument": "b"},
    }
    assert factory_calls[0]["identity_frame"].fingerprint == estimator.fit_identity_frame_.fingerprint
    assert client.calls[0]["kwargs"]["diagnostics"]["factory"] is True
    assert client.calls[0]["kwargs"]["diagnostics"]["nirs4all_fit_identity_explicit_sample_ids"] is True


def test_training_request_compiler_assembles_signed_request_for_estimator_fit() -> None:
    client = _FakeNativeClient()
    request_spec = DagMLTrainingRequestSpec(
        request_id="training:nirs4all.unit",
        plan_id="plan:nirs4all.unit",
        graph={"id": "graph:unit", "interface": {"inputs": [], "outputs": []}, "nodes": [], "edges": [], "search_space_fingerprint": None, "metadata": {}},
        campaign={
            "id": "campaign:unit",
            "root_seed": 12345,
            "leakage_policy": {},
            "aggregation_policy": {},
            "split_invocation": None,
            "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
            "shape_plans": {},
            "data_bindings": {},
            "branch_view_plans": [],
            "inner_cv": None,
            "metadata": {},
        },
        controller_manifests=(),
        data_identities=(),
    )
    compiler = DagMLTrainingRequestCompiler(
        DagMLTrainingRequestContracts(
            request_spec=request_spec,
            data_envelopes={},
            relations={"records": []},
            training_influence={"schema_version": 1, "entries": [], "manifest_fingerprint": "0" * 64, "relation_fingerprint": "0" * 64},
            op_callback=lambda task: {"task": task},
            outcome_id="outcome-1",
            run_id="run-1",
            bundle_id="bundle-1",
        ),
        additional_diagnostics={"compiler": "request-spec"},
    )
    estimator = DagMLPipelineEstimator(
        pipeline=("request-spec",),
        native_client=client,
        training_compiler=compiler,
    )

    estimator.fit(np.ones((2, 2)), np.arange(2), sample_ids=["s1", "s2"])

    request = client.calls[0]["args"][0]
    assert request["request_id"] == "training:nirs4all.unit"
    assert request["request_fingerprint"] != "0" * 64
    assert client.calls[0]["kwargs"]["diagnostics"]["compiler"] == "request-spec"


def test_training_request_compiler_uses_dagml_signer_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class SignedRequest:
        def __init__(self, request: dict[str, Any]) -> None:
            self.request = request

        def to_dict(self) -> dict[str, Any]:
            return {**self.request, "signed_by": "dag_ml"}

    fake_dagml = types.SimpleNamespace(sign_training_request=lambda request: SignedRequest(request))
    monkeypatch.setitem(sys.modules, "dag_ml_signer_test", fake_dagml)
    request_spec = DagMLTrainingRequestSpec(
        request_id="training:nirs4all.unit",
        plan_id="plan:nirs4all.unit",
        graph={"id": "graph:unit", "interface": {"inputs": [], "outputs": []}, "nodes": [], "edges": [], "search_space_fingerprint": None, "metadata": {}},
        campaign={
            "id": "campaign:unit",
            "root_seed": 12345,
            "leakage_policy": {},
            "aggregation_policy": {},
            "split_invocation": None,
            "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
            "shape_plans": {},
            "data_bindings": {},
            "branch_view_plans": [],
            "inner_cv": None,
            "metadata": {},
        },
        controller_manifests=(),
        data_identities=(),
    )
    compiler = DagMLTrainingRequestCompiler(
        DagMLTrainingRequestContracts(
            request_spec=request_spec,
            data_envelopes={},
            relations={"records": []},
            training_influence={"schema_version": 1, "entries": [], "manifest_fingerprint": "0" * 64, "relation_fingerprint": "0" * 64},
            op_callback=lambda task: {"task": task},
            outcome_id="outcome-1",
            run_id="run-1",
            bundle_id="bundle-1",
        ),
        dagml_module="dag_ml_signer_test",
    )

    execution = compiler.compile_fit(
        DagMLPipelineEstimator(),
        np.ones((1, 2)),
        np.ones(1),
        identity_frame=normalize_fit_identity(np.ones((1, 2)), np.ones(1), sample_ids=["s1"]),
    )

    assert execution.request["signed_by"] == "dag_ml"


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"data_envelopes": {1: {"rows": 1}}}, ValueError, "data_envelopes keys"),
        ({"op_callback": object()}, TypeError, "op_callback"),
        ({"outcome_id": ""}, ValueError, "outcome_id"),
        ({"run_id": ""}, ValueError, "run_id"),
        ({"bundle_id": ""}, ValueError, "bundle_id"),
        ({"warnings": ("",)}, ValueError, "warnings"),
        ({"diagnostics": []}, TypeError, "diagnostics"),
    ],
)
def test_prepared_contract_validation_is_fail_closed(
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    identity_frame = normalize_fit_identity(np.ones((1, 2)), np.ones(1), sample_ids=["s1"])

    with pytest.raises(error, match=match):
        compile_prepared_training_contracts(
            _contracts(**overrides),
            identity_frame=identity_frame,
        )

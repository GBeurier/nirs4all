"""Unit tests for the nirs4all-side DAG-ML native client seam."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any

import pytest

from nirs4all.pipeline.dagml.errors import DagMlUnavailable
from nirs4all.pipeline.dagml.native_client import DagMLNativeClient, DagMLNativeCoverageError


def _fake_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_constructor_does_not_import_dag_ml(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "_n4a_absent_dag_ml_for_deferred_import"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    client = DagMLNativeClient(module_name)

    assert module_name not in sys.modules
    with pytest.raises(DagMlUnavailable, match="not importable"):
        client.capabilities()


def test_capabilities_report_training_replay_and_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "_n4a_fake_dag_ml_capabilities"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(
            module_name,
            contract_manifest_json=lambda: json.dumps(
                {
                    "python_package_version": "0.8.0-test",
                    "training_contracts": ["training_request"],
                }
            ),
            execute_training=lambda *args, **kwargs: object(),
            replay_loaded_predictor_package=lambda *args, **kwargs: object(),
        ),
    )

    capabilities = DagMLNativeClient(module_name).capabilities()

    assert capabilities.package_version == "0.8.0-test"
    assert capabilities.contract_manifest == {
        "python_package_version": "0.8.0-test",
        "training_contracts": ["training_request"],
    }
    assert capabilities.training is True
    assert capabilities.loaded_predictor_replay is True


def test_missing_training_entrypoint_is_typed_coverage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "_n4a_fake_dag_ml_missing_training"
    monkeypatch.setitem(sys.modules, module_name, _fake_module(module_name))

    client = DagMLNativeClient(module_name)

    assert client.capabilities().training is False
    with pytest.raises(DagMLNativeCoverageError, match=r"execute_training\(\)"):
        client.execute_training(
            {},
            {},
            {},
            {},
            lambda task: task,
            outcome_id="outcome",
            run_id="run",
            bundle_id="bundle",
        )


def test_execute_training_forwards_to_facade_without_contract_reimplementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_n4a_fake_dag_ml_execute_training"
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def execute_training(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, execute_training=execute_training),
    )

    op_callback = lambda task: {"task": task}  # noqa: E731
    request = {
        "request": True,
        "portable_prediction_caches": {
            "caches": [
                {
                    "cache_id": "cache:selected:oof",
                    "cache_namespace_fingerprints": ["a" * 64],
                }
            ]
        },
    }
    result = DagMLNativeClient(module_name).execute_training(
        request,
        {"envelope": True},
        {"relations": True},
        {"influence": True},
        op_callback,
        outcome_id="outcome-1",
        run_id="run-1",
        bundle_id="bundle-1",
        warnings=["warning"],
        diagnostics={"native": True},
    )

    assert result is sentinel
    assert calls[0][0][0] is request
    assert calls[0][0][0]["portable_prediction_caches"]["caches"][0]["cache_namespace_fingerprints"] == ["a" * 64]
    assert calls == [
        (
            (
                request,
                {"envelope": True},
                {"relations": True},
                {"influence": True},
                op_callback,
            ),
            {
                "outcome_id": "outcome-1",
                "run_id": "run-1",
                "bundle_id": "bundle-1",
                "warnings": ["warning"],
                "diagnostics": {"native": True},
            },
        )
    ]


def test_replay_loaded_predictor_package_forwards_to_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "_n4a_fake_dag_ml_replay"
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def replay_loaded_predictor_package(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, replay_loaded_predictor_package=replay_loaded_predictor_package),
    )

    op_callback = lambda task: {"task": task}  # noqa: E731
    request = {
        "phase": "EXPLAIN",
        "request": True,
        "cache_namespace_fingerprints": ["b" * 64],
    }
    result = DagMLNativeClient(module_name).replay_loaded_predictor_package(
        {"package": True},
        request,
        {"envelope": True},
        {"artifact": "handle"},
        op_callback,
        outcome_id="replay-outcome",
        run_id="replay-run",
        warnings=[],
        diagnostics={"replay": True},
    )

    assert result is sentinel
    assert calls[0][0][1] is request
    assert calls[0][0][1]["cache_namespace_fingerprints"] == ["b" * 64]
    assert calls == [
        (
            (
                {"package": True},
                request,
                {"envelope": True},
                {"artifact": "handle"},
                op_callback,
            ),
            {
                "outcome_id": "replay-outcome",
                "run_id": "replay-run",
                "warnings": [],
                "diagnostics": {"replay": True},
            },
        )
    ]


def test_non_object_contract_manifest_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "_n4a_fake_dag_ml_bad_manifest"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        _fake_module(module_name, contract_manifest_json=lambda: "[]"),
    )

    with pytest.raises(DagMLNativeCoverageError, match="manifest must be a JSON object"):
        DagMLNativeClient(module_name).capabilities()

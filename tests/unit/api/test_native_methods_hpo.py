"""API-001 coverage for controller-owned portable Methods HPO."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.api.native_archive_training import (
    NativeMethodsArchiveRunResult,
    NativeMethodsHpoCapabilityError,
    _attach_portable_methods_hpo,
    _extract_portable_methods_hpo,
)
from nirs4all.pipeline.dagml.training_compiler import DagMLTrainingRequestContracts
from nirs4all.pipeline.dagml.training_contracts import DagMLTrainingRequestSpec


def _pipeline(finetune: dict[str, Any]) -> list[Any]:
    return [
        KFold(n_splits=3),
        {
            "model": PLSRegression(n_components=1),
            "finetune_params": finetune,
        },
    ]


def _finetune(**overrides: Any) -> dict[str, Any]:
    return {
        "engine": "n4m",
        "n_trials": 4,
        "sampler": "random",
        "pruner": "none",
        "approach": "grouped",
        "seed": 7,
        "metric": "rmse",
        "direction": "minimize",
        "model_params": {"n_components": ["int", 1, 3]},
        **overrides,
    }


def _contracts() -> DagMLTrainingRequestContracts:
    manifest = {
        "controller_id": "controller:methods.pls",
        "controller_version": "n4m-abi-2.2",
        "operator_kind": "model",
        "priority": 100,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one"}],
        "output_ports": [{"name": "oof", "kind": "prediction", "representation": None, "cardinality": "one"}],
        "capabilities": ["deterministic", "emits_predictions", "emits_artifacts", "stateful"],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable",
    }
    spec = DagMLTrainingRequestSpec(
        request_id="training:test",
        plan_id="plan:test",
        graph={
            "id": "graph:test",
            "nodes": [
                {
                    "id": "model:compat.0",
                    "kind": "model",
                    "operator": "sklearn.cross_decomposition._pls.PLSRegression",
                    "params": {"n_components": 1},
                }
            ],
            "edges": [],
        },
        campaign={
            "id": "campaign:test",
            "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
            "metadata": {},
        },
        controller_manifests=[manifest],
        data_identities=[],
    )
    return DagMLTrainingRequestContracts(
        request_spec=spec,
        data_envelopes={},
        relations={},
        training_influence={},
        op_callback=lambda _operation: None,
        outcome_id="outcome:test",
        run_id="run:test",
        bundle_id="bundle:test",
    )


def test_extracts_bounded_hpo_without_mutating_the_public_pipeline() -> None:
    pipeline = _pipeline(_finetune())

    stripped, hpo = _extract_portable_methods_hpo(pipeline, seed=12345)

    assert hpo is not None
    assert (hpo.trials, hpo.seed, hpo.low, hpo.high, hpo.step) == (4, 7, 1, 3, 1)
    assert set(stripped[1]) == {"model"}
    assert "finetune_params" in pipeline[1]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sampler": "tpe"}, "sampler='random'"),
        ({"pruner": "median"}, "pruner='none'"),
        ({"metric": "mae"}, "OOF RMSE"),
        ({"n_trials": 0}, "1..256"),
    ],
)
def test_refuses_unimplemented_hpo_shapes_before_native_runtime(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    match: str,
) -> None:
    from nirs4all.api import native_archive_training as module

    monkeypatch.setattr(
        module,
        "_require_archive_runtime",
        lambda: pytest.fail("invalid HPO request reached the native runtime"),
    )
    dataset = {
        "X": np.arange(18, dtype=float).reshape(6, 3),
        "y": np.arange(6, dtype=float),
        "sample_ids": [f"sample.{index}" for index in range(6)],
    }

    with pytest.raises(ValueError, match=match):
        module.run_native_methods_archive(
            _pipeline(_finetune(**overrides)),
            dataset,
            save_charts=False,
        )


@pytest.mark.parametrize(
    "space",
    [
        ["int", 1, 4],
        ["int", 1, 3, 2],
        ["int", 2, 3],
        ["float", 1, 3],
        ["int", 1, 3, 1, 1],
        {"type": "int", "low": 1, "high": 3, "step": 2},
        {"type": "int", "low": 1, "high": 3, "log": True},
    ],
)
def test_refuses_spaces_outside_exact_methods_v1_before_data_or_session(
    monkeypatch: pytest.MonkeyPatch,
    space: Any,
) -> None:
    from nirs4all.api import native_archive_training as module
    from nirs4all.api.session import Session

    native_session = Session()
    monkeypatch.setattr(
        native_session,
        "_prepare_native_run",
        lambda: pytest.fail("invalid HPO request prepared a native session"),
    )
    monkeypatch.setattr(
        module,
        "_normalize_training_arrays",
        lambda _dataset: pytest.fail("invalid HPO request consumed dataset arrays"),
    )

    with pytest.raises(
        NativeMethodsHpoCapabilityError,
        match="portable Methods HPO v1",
    ):
        module.run_native_methods_archive(
            _pipeline(_finetune(model_params={"n_components": space})),
            {"X": object(), "y": object(), "sample_ids": object()},
            session=native_session,
        )


def test_refuses_too_narrow_feature_matrix_before_session_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nirs4all.api import native_archive_training as module
    from nirs4all.api.session import Session

    native_session = Session()
    monkeypatch.setattr(
        native_session,
        "_prepare_native_run",
        lambda: pytest.fail("unsupported HPO request prepared a native session"),
    )
    monkeypatch.setattr(
        module,
        "_require_archive_runtime",
        lambda: pytest.fail("unsupported HPO request reached the native runtime"),
    )

    with pytest.raises(NativeMethodsHpoCapabilityError, match="at least 3 features"):
        module.run_native_methods_archive(
            _pipeline(_finetune()),
            {
                "X": np.arange(12, dtype=float).reshape(6, 2),
                "y": np.arange(6, dtype=float),
                "sample_ids": [f"sample.{index}" for index in range(6)],
            },
            session=native_session,
        )


@pytest.mark.parametrize(
    "space",
    [
        ["int", 1, 3],
        ["int", 1, 3, 1],
        {"type": "int", "low": 1, "high": 3, "step": 1},
        {"type": "int", "min": 1, "max": 3},
    ],
)
def test_accepts_only_equivalent_exact_methods_v1_space_spellings(space: Any) -> None:
    _pipeline_without_hpo, hpo = _extract_portable_methods_hpo(
        _pipeline(_finetune(model_params={"n_components": space})),
        seed=12345,
    )

    assert hpo is not None
    assert (hpo.low, hpo.high, hpo.step) == (1, 3, 1)


def test_attaches_one_signed_scheduler_operation_and_tuner_controller() -> None:
    _stripped, hpo = _extract_portable_methods_hpo(_pipeline(_finetune()), seed=12345)
    assert hpo is not None

    attached = _attach_portable_methods_hpo(_contracts(), hpo)
    request = attached.to_prepared().request

    assert request["request_fingerprint"] != "0" * 64
    assert request["graph"]["nodes"][0]["operator"] == "pls"
    operation = request["campaign"]["metadata"]["methods_hpo_operation"]
    assert operation == {
        "operation_id": "hpo:nirs4all.native.methods",
        "study": {
            "controller_id": "controller:tuner.methods",
            "study_id": "study:nirs4all.native.methods",
            "methods_abi": "n4m-abi-2.2",
            "search_space": {
                "parameters": [
                    {
                        "kind": "int",
                        "name": "n_components",
                        "low": 1,
                        "high": 3,
                        "step": 1,
                        "log": False,
                    }
                ]
            },
            "optimizer": {
                "sampler": "random",
                "pruner": "none",
                "direction": "minimize",
                "metric": "rmse",
                "seed": 7,
                "n_startup_trials": 0,
                "max_resource": 0,
                "reduction_factor": 0,
            },
        },
        "trials": 4,
        "target_node_id": "model:compat.0",
        "parameter_paths": {"n_components": "n_components"},
    }
    controllers = {item["controller_id"]: item for item in request["controller_manifests"]}
    assert set(controllers) == {"controller:methods.pls", "controller:tuner.methods"}
    assert controllers["controller:tuner.methods"]["operator_kind"] == "tuner"
    assert attached.diagnostics == {
        "nirs4all_execution": "methods_controller_owned_hpo_archive_v2",
        "nirs4all_methods_hpo_resume": "not_exposed_by_public_api_v1",
    }


def test_projects_only_the_scheduler_checked_native_incumbent() -> None:
    result = object.__new__(NativeMethodsArchiveRunResult)
    result._native_outcome = {
        "methods_hpo_resume_state": {
            "incumbent": {"trial_id": 2, "score": 0.125},
            "terminal_trials": [
                {
                    "trial": {
                        "id": 2,
                        "parameters": {
                            "n_components": {
                                "name": "n_components",
                                "value": 3.0,
                                "native_kind": "int",
                                "integer": True,
                                "active": True,
                            }
                        },
                    }
                }
            ],
        }
    }

    assert result.tuning_best_params == {"model.n_components": 3}
    assert result.tuning_best_value == 0.125


def test_hpo_attachment_is_immutable() -> None:
    contracts = _contracts()
    _stripped, hpo = _extract_portable_methods_hpo(_pipeline(_finetune()), seed=12345)
    assert hpo is not None
    original = replace(contracts.request_spec)

    _attach_portable_methods_hpo(contracts, hpo)

    assert contracts.request_spec == original
    assert contracts.request_spec.campaign["metadata"] == {}

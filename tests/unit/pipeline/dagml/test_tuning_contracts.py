"""Contract tests for native DAG-ML public tuning specs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import pytest

from nirs4all.pipeline.dagml.tuning_contracts import (
    TUNING_SPACE_SCHEMA_ID,
    DagMLTuningSpec,
    OrderedSearchSpaceSpec,
    ParameterPatch,
    SearchSpaceParameter,
    TrialResult,
    TuningResult,
    get_tuning_space_schema,
    parse_tuning_spec,
    tuning_space_schema_json,
)


def test_parse_tuning_spec_normalizes_public_contract_and_fingerprint() -> None:
    spec = parse_tuning_spec(
        {
            "engine": "N4M",
            "space": {
                " model.n_components ": [2, 3],
                "preprocess.savgol.window_length": {"type": "int", "low": 5, "high": 15, "step": 2},
            },
            "force_params": {" model.n_components ": 3},
            "metric": " RMSE ",
            "direction": "MINIMIZE",
            "n_trials": 12,
            "sampler": " TPE ",
            "pruner": " Median ",
            "seed": 42,
            "resume": True,
            "storage": " sqlite:///study.db ",
            "study_name": " pls_tuning ",
        }
    )

    assert isinstance(spec, DagMLTuningSpec)
    assert spec.engine == "n4m"
    assert spec.metric == "rmse"
    assert spec.direction == "minimize"
    assert spec.sampler == "tpe"
    assert spec.pruner == "median"
    assert spec.storage == "sqlite:///study.db"
    assert spec.study_name == "pls_tuning"
    assert spec.space == {
        "model.n_components": [2, 3],
        "preprocess.savgol.window_length": {"type": "int", "low": 5, "high": 15, "step": 2},
    }
    assert spec.force_params == {"model.n_components": 3}
    assert spec.to_dict()["space"] == {
        "model.n_components": [2, 3],
        "preprocess.savgol.window_length": {"type": "int", "low": 5, "high": 15, "step": 2},
    }
    assert spec.to_dict()["force_params"] == {"model.n_components": 3}
    assert len(spec.fingerprint) == 64
    assert (
        spec.fingerprint
        == parse_tuning_spec(
            {
                "space": {
                    "preprocess.savgol.window_length": {"type": "int", "low": 5, "high": 15, "step": 2},
                    " model.n_components ": [2, 3],
                },
                "engine": "n4m",
                "n_trials": 12,
                "metric": "rmse",
                "direction": "minimize",
                "sampler": "tpe",
                "pruner": "median",
                "seed": 42,
                "resume": True,
                "storage": "sqlite:///study.db",
                "study_name": "pls_tuning",
                "force_params": {"model.n_components": 3},
            }
        ).fingerprint
    )


def test_ordered_search_space_canonicalizes_public_patch_paths_and_patches() -> None:
    spec = parse_tuning_spec(
        {
            "engine": "optuna",
            "space": {
                " ridge__alpha ": [0.1, 0.2],
                "scale.with_mean": [False],
            },
            "force_params": {
                "ridge.alpha": 0.2,
            },
        }
    )

    ordered = spec.ordered_search_space

    assert isinstance(ordered, OrderedSearchSpaceSpec)
    assert ordered.paths == ("ridge.alpha", "scale.with_mean")
    assert ordered.to_space_mapping() == {
        "ridge.alpha": [0.1, 0.2],
        "scale.with_mean": [False],
    }
    assert ordered.to_dict() == {
        "format": "nirs4all.tuning.ordered_search_space",
        "parameters": [
            {
                "index": 0,
                "path": "ridge.alpha",
                "segments": ["ridge", "alpha"],
                "spec": [0.1, 0.2],
            },
            {
                "index": 1,
                "path": "scale.with_mean",
                "segments": ["scale", "with_mean"],
                "spec": [False],
            },
        ],
        "schema_version": 1,
    }
    assert len(ordered.fingerprint) == 64
    assert spec.force_params == {"ridge.alpha": 0.2}
    assert spec.parameter_patches({"scale__with_mean": False, "ridge.alpha": 0.1}) == (
        ParameterPatch(path="ridge.alpha", segments=("ridge", "alpha"), value=0.1),
        ParameterPatch(path="scale.with_mean", segments=("scale", "with_mean"), value=False),
    )


def test_tuning_contracts_direct_construction_normalizes_public_contracts() -> None:
    parameter = SearchSpaceParameter(
        index=0,
        path=" ridge__alpha ",
        segments=("ridge", "alpha"),
        spec=[0.1, 0.2],
    )
    patch = ParameterPatch(path=" ridge__alpha ", segments=("ridge", "alpha"), value=0.1)
    tuning = DagMLTuningSpec(
        engine=" OPTUNA ",
        space={" ridge__alpha ": [0.1, 0.2]},
        force_params={" ridge__alpha ": 0.2},
        metric=" RMSE ",
        direction=" MINIMIZE ",
        n_trials=2,
        sampler=" TPE ",
        pruner=" Median ",
        seed=42,
        resume=True,
        storage=" sqlite:///study.db ",
        study_name=" pls-study ",
    )
    trial = TrialResult(
        number=0,
        params={" ridge__alpha ": 0.1},
        value=0.1,
        state=" complete ",
        diagnostics={"metric": "rmse"},
    )
    result = TuningResult(
        tuning=tuning,
        best_params={" ridge__alpha ": 0.1},
        best_value=0.1,
        trials=(trial,),
        optimizer=" optuna ",
    )

    assert parameter == SearchSpaceParameter(index=0, path="ridge.alpha", segments=("ridge", "alpha"), spec=[0.1, 0.2])
    assert patch == ParameterPatch(path="ridge.alpha", segments=("ridge", "alpha"), value=0.1)
    assert tuning.engine == "optuna"
    assert tuning.space == {"ridge.alpha": [0.1, 0.2]}
    assert tuning.force_params == {"ridge.alpha": 0.2}
    assert tuning.metric == "rmse"
    assert tuning.direction == "minimize"
    assert tuning.sampler == "tpe"
    assert tuning.pruner == "median"
    assert tuning.storage == "sqlite:///study.db"
    assert tuning.study_name == "pls-study"
    assert trial.params == {"ridge.alpha": 0.1}
    assert trial.state == "COMPLETE"
    assert result.best_params == {"ridge.alpha": 0.1}
    assert result.optimizer == "optuna"
    assert result.trials == (trial,)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: SearchSpaceParameter(index=True, path="ridge.alpha", segments=("ridge", "alpha"), spec=[0.1]),
            "index must be a non-negative integer",
        ),
        (
            lambda: SearchSpaceParameter(index=0, path="ridge.alpha", segments=("ridge", "beta"), spec=[0.1]),
            "segments must match path",
        ),
        (
            lambda: SearchSpaceParameter(index=0, path="ridge.alpha", segments=("ridge", "alpha"), spec=object()),
            "spec must contain TCV1-compatible JSON-native values",
        ),
        (
            lambda: ParameterPatch(path="ridge.alpha", segments=("ridge", "beta"), value=0.1),
            "segments must match path",
        ),
        (
            lambda: ParameterPatch(path="ridge.alpha", segments=("ridge", "alpha"), value=object()),
            "value must contain TCV1-compatible JSON-native values",
        ),
        (
            lambda: OrderedSearchSpaceSpec(parameters=(SearchSpaceParameter(index=1, path="ridge.alpha", segments=("ridge", "alpha"), spec=[0.1]),)),
            "indexes must be contiguous from zero",
        ),
        (
            lambda: OrderedSearchSpaceSpec(
                parameters=(
                    SearchSpaceParameter(index=0, path="zeta.alpha", segments=("zeta", "alpha"), spec=[0.1]),
                    SearchSpaceParameter(index=1, path="ridge.alpha", segments=("ridge", "alpha"), spec=[0.1]),
                )
            ),
            "must be sorted by canonical path",
        ),
        (
            lambda: DagMLTuningSpec(engine="bad", space={"ridge.alpha": [0.1]}),
            "engine must be one of",
        ),
        (
            lambda: DagMLTuningSpec(engine="optuna", space={"ridge.alpha": [0.1]}, n_trials=True),
            "n_trials must be a positive integer",
        ),
        (
            lambda: DagMLTuningSpec(engine="optuna", space={"ridge.alpha": [0.1]}, resume="yes"),  # type: ignore[arg-type]
            "resume must be a boolean",
        ),
        (
            lambda: TrialResult(number=True, params={"ridge.alpha": 0.1}, value=0.1, state="COMPLETE", diagnostics={}),
            "number must be a non-negative integer",
        ),
        (
            lambda: TrialResult(number=0, params={"ridge.alpha": object()}, value=0.1, state="COMPLETE", diagnostics={}),
            "params\\['ridge.alpha'\\] must contain TCV1-compatible JSON-native values",
        ),
        (
            lambda: TrialResult(number=0, params={"ridge.alpha": 0.1}, value=math.nan, state="COMPLETE", diagnostics={}),
            "value must be a finite number",
        ),
        (
            lambda: TrialResult(number=0, params={"ridge.alpha": 0.1}, value=0.1, state="", diagnostics={}),
            "state must be a non-empty string",
        ),
        (
            lambda: TrialResult(number=0, params={"ridge.alpha": 0.1}, value=0.1, state="COMPLETE", diagnostics={"bad": object()}),
            "diagnostics must contain TCV1-compatible JSON-native values",
        ),
        (
            lambda: TuningResult(
                tuning=DagMLTuningSpec(engine="optuna", space={"ridge.alpha": [0.1]}),
                best_params={"ridge.beta": 0.1},
                best_value=0.1,
                trials=(),
                optimizer="optuna",
            ),
            "not present in tuning.space",
        ),
        (
            lambda: TuningResult(
                tuning=DagMLTuningSpec(engine="optuna", space={"ridge.alpha": [0.1]}),
                best_params={"ridge.alpha": 0.1},
                best_value=math.inf,
                trials=(),
                optimizer="optuna",
            ),
            "best_value must be a finite number",
        ),
        (
            lambda: TuningResult(
                tuning=DagMLTuningSpec(engine="optuna", space={"ridge.alpha": [0.1]}),
                best_params={"ridge.alpha": 0.1},
                best_value=0.1,
                trials=(
                    TrialResult(number=0, params={"ridge.alpha": 0.1}, value=0.1, state="COMPLETE", diagnostics={}),
                    TrialResult(number=0, params={"ridge.alpha": 0.1}, value=0.1, state="COMPLETE", diagnostics={}),
                ),
                optimizer="optuna",
            ),
            "duplicate trial numbers",
        ),
    ],
)
def test_tuning_contracts_direct_construction_rejects_invalid_payloads(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_ordered_search_space_rejects_ambiguous_or_non_json_patch_syntax() -> None:
    with pytest.raises(ValueError, match="duplicate patch path 'ridge.alpha'"):
        parse_tuning_spec(
            {
                "engine": "optuna",
                "space": {
                    "ridge.alpha": [0.1],
                    "ridge__alpha": [0.2],
                },
            }
        )

    spec = parse_tuning_spec({"engine": "optuna", "space": {"ridge.alpha": [0.1]}})
    with pytest.raises(ValueError, match="not present in tuning.space"):
        spec.parameter_patches({"ridge.beta": 1.0})
    with pytest.raises(ValueError, match="TCV1-compatible JSON-native value"):
        spec.parameter_patches({"ridge.alpha": object()})
    with pytest.raises(ValueError, match="trial params path must be a non-empty string"):
        spec.parameter_patches({1: 0.1})  # type: ignore[dict-item]


def test_tuning_space_schema_publishes_public_ordered_contract() -> None:
    schema = get_tuning_space_schema()

    assert schema["$id"] == TUNING_SPACE_SCHEMA_ID
    assert schema["properties"]["format"]["const"] == "nirs4all.tuning.ordered_search_space"
    assert schema["properties"]["schema_version"]["const"] == 1
    assert schema["properties"]["parameters"]["items"]["required"] == ["index", "path", "segments", "spec"]
    assert schema["properties"]["force_params"]["items"]["required"] == ["path", "segments", "value"]
    assert schema["required"] == [
        "fingerprint",
        "force_params",
        "format",
        "parameters",
        "schema_version",
        "tuning_fingerprint",
    ]
    compact = tuning_space_schema_json(indent=None)
    assert compact.endswith("\n")
    assert json.loads(compact) == schema


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"space": {"model.n_components": [2, 3]}}, "engine is required"),
        ({"engine": "grid", "space": {"model.n_components": [2, 3]}}, "engine must be one of"),
        ({"engine": "n4m", "space": {}}, "space must be a non-empty mapping"),
        ({"engine": "optuna", "space": {"": [2, 3]}}, "space key must be a non-empty string"),
        ({"engine": "optuna", "space": {"x": [1]}, "n_trials": 0}, "n_trials must be a positive integer"),
        ({"engine": "optuna", "space": {"x": [1]}, "resume": "yes"}, "resume must be a boolean"),
        ({"engine": "optuna", "space": {"x": [1]}, "storage": "study.db"}, "storage must be a URI"),
        ({"engine": "optuna", "space": {"x": [1]}, "storage": "not a uri"}, "storage must be a URI"),
        (
            {"engine": "optuna", "space": {"x": [1]}, "study_name": "bad\x00name"},
            "study_name must not contain NUL",
        ),
        ({"engine": "optuna", "space": {"x": [1]}, "unknown": True}, "does not support keys"),
        ({"engine": "optuna", "space": {"scale": [object()]}}, "TCV1-compatible JSON-native"),
        ({"engine": "optuna", "space": {"alpha": [math.inf]}}, "TCV1-compatible JSON-native"),
        ({"engine": "optuna", "space": {"alpha": [0.2]}, "force_params": []}, "force_params must be a non-empty mapping"),
        (
            {"engine": "optuna", "space": {"alpha": [0.2]}, "force_params": {"beta": 1.0}},
            "force_params keys must be a subset",
        ),
        (
            {"engine": "optuna", "space": {"alpha": [0.2]}, "force_params": {"alpha": object()}},
            "TCV1-compatible JSON-native",
        ),
    ],
)
def test_parse_tuning_spec_rejects_ambiguous_or_unsupported_payloads(payload: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        parse_tuning_spec(payload)


def test_tuning_result_round_trips_with_verified_fingerprint() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 0.2]}, "n_trials": 2})
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        optimizer="optuna",
        trials=(
            TrialResult(
                number=0,
                params={"alpha": 0.2},
                value=0.2,
                state="COMPLETE",
                diagnostics={"metric": "rmse"},
            ),
            TrialResult(
                number=1,
                params={"alpha": 0.1},
                value=0.1,
                state="COMPLETE",
                diagnostics={"metric": "rmse"},
            ),
        ),
    )

    payload = result.to_dict()
    restored = TuningResult.from_dict(payload)

    assert restored == result
    assert payload["fingerprint"] == result.fingerprint
    assert restored.fingerprint == result.fingerprint


def test_tuning_summary_publishes_safe_persistence_controls_without_storage_uri() -> None:
    tuning = parse_tuning_spec(
        {
            "engine": "optuna",
            "space": {"alpha": [0.1, 0.2]},
            "storage": "sqlite:///study.db",
            "study_name": "pls-study",
            "resume": True,
        }
    )
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        optimizer="optuna",
        trials=(),
    )

    summary = result.summary_artifact()

    assert summary["persistence"] == {
        "optimizer_state_resume_supported": True,
        "resume": True,
        "storage_configured": True,
        "study_name": "pls-study",
    }
    assert "sqlite:///study.db" not in result.to_summary_json()


def test_tuning_summary_marks_n4m_optimizer_state_resume_as_supported() -> None:
    tuning = parse_tuning_spec(
        {
            "engine": "n4m",
            "space": {"alpha": [0.1, 0.2]},
            "storage": "file:///tmp/n4m-checkpoints",
            "study_name": "pls-study",
            "resume": True,
        }
    )
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        optimizer="n4m",
        trials=(),
    )

    assert result.summary_artifact()["persistence"] == {
        "optimizer_state_resume_supported": True,
        "resume": True,
        "storage_configured": True,
        "study_name": "pls-study",
    }


def test_tuning_summary_publishes_compact_trial_diagnostics_without_raw_errors() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 0.2]}})
    result = TuningResult(
        tuning=tuning,
        best_params={},
        best_value=1.0e308,
        optimizer="optuna",
        trials=(
            TrialResult(
                number=0,
                params={"alpha": 0.2},
                value=None,
                state="FAIL",
                diagnostics={
                    "direction": "minimize",
                    "engine": "optuna",
                    "error": "candidate stack trace that must stay out of summary cards",
                    "error_type": "RuntimeError",
                    "metric": "rmse",
                    "score_extractor": "failed",
                    "score_family": "objective",
                    "search_space_fingerprint": "a" * 64,
                    "tuning_fingerprint": tuning.fingerprint,
                },
            ),
        ),
    )

    trial = result.summary_artifact()["trials"][0]

    assert trial == {
        "diagnostics": {
            "direction": "minimize",
            "engine": "optuna",
            "error_type": "RuntimeError",
            "metric": "rmse",
            "score_extractor": "failed",
            "score_family": "objective",
            "search_space_fingerprint": "a" * 64,
            "tuning_fingerprint": tuning.fingerprint,
        },
        "number": 0,
        "state": "FAIL",
        "value": None,
    }
    assert "candidate stack trace" not in result.to_summary_json()


def test_tuning_summary_rejects_non_scalar_whitelisted_diagnostics() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 0.2]}})
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.2},
        best_value=0.2,
        optimizer="optuna",
        trials=(
            TrialResult(
                number=0,
                params={"alpha": 0.2},
                value=0.2,
                state="COMPLETE",
                diagnostics={"metric": ["rmse"]},
            ),
        ),
    )

    with pytest.raises(ValueError, match="summary diagnostics.metric must be a scalar JSON-native value"):
        result.summary_artifact()


def test_tuning_result_saves_and_loads_deterministic_json(tmp_path) -> None:
    tuning = parse_tuning_spec({"engine": "n4m", "space": {"alpha": [0.1, 0.2]}, "sampler": "tpe"})
    result = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.2},
        best_value=0.2,
        optimizer="n4m",
        trials=(
            TrialResult(
                number=1,
                params={"alpha": 0.2},
                value=0.2,
                state="COMPLETE",
                diagnostics={"direction": "minimize", "metric": "rmse"},
            ),
        ),
    )
    target = tmp_path / "tuning-result.json"

    saved = result.save_json(target)
    first = target.read_text(encoding="utf-8")
    result.save_json(target)
    second = target.read_text(encoding="utf-8")

    assert saved == target
    assert first == second
    assert first.endswith("\n")
    assert TuningResult.load_json(target) == result


def test_tuning_result_rejects_fingerprint_mismatch() -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 0.2]}})
    payload = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        optimizer="optuna",
        trials=(),
    ).to_dict()
    payload["fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        TuningResult.from_dict(payload)


@pytest.mark.parametrize("best_value", [True, "0.1"])
def test_tuning_result_from_dict_rejects_coercive_best_value(best_value: object) -> None:
    tuning = parse_tuning_spec({"engine": "optuna", "space": {"alpha": [0.1, 0.2]}})
    payload = TuningResult(
        tuning=tuning,
        best_params={"alpha": 0.1},
        best_value=0.1,
        optimizer="optuna",
        trials=(),
    ).to_dict()
    payload["best_value"] = best_value
    payload.pop("fingerprint")

    with pytest.raises(ValueError, match="TuningResult.best_value must be a finite number"):
        TuningResult.from_dict(payload)

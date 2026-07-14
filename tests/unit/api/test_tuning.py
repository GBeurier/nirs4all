"""Unit tests for public native tuning result helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import nirs4all
from nirs4all.pipeline.dagml.tuning_contracts import (
    SUPPORTED_TUNING_DIRECTIONS,
    SUPPORTED_TUNING_ENGINES,
    SUPPORTED_TUNING_KEYS,
    TrialResult,
    TuningResult,
    parse_tuning_spec,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "tuning"


def _make_result() -> TuningResult:
    tuning = parse_tuning_spec(
        {
            "engine": "optuna",
            "space": {"model.n_components": [2, 3]},
            "metric": "rmse",
            "direction": "minimize",
            "n_trials": 2,
            "sampler": "grid",
        }
    )
    return TuningResult(
        tuning=tuning,
        best_params={"model.n_components": 2},
        best_value=0.12,
        trials=(
            TrialResult(number=0, params={"model.n_components": 3}, value=0.2, state="COMPLETE", diagnostics={}),
            TrialResult(number=1, params={"model.n_components": 2}, value=0.12, state="COMPLETE", diagnostics={}),
        ),
        optimizer="optuna",
    )


class _PublicTuningEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, **_kwargs: Any) -> _PublicTuningEstimator:
        self.fit_X_ = X
        self.fit_y_ = y
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), float(self.alpha))


class _PublicIdentityAwareTuningEstimator(BaseEstimator):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(self, X: Any, y: Any, **_kwargs: Any) -> _PublicIdentityAwareTuningEstimator:
        return self

    def predict(
        self,
        X: Any,
        *,
        sample_ids: Any = None,
        groups: Any = None,
        metadata: Any = None,
    ) -> np.ndarray:
        identities_match = list(sample_ids or []) == ["score-a", "score-b"] and list(groups or []) == ["batch-1", "batch-2"] and list(metadata or []) == [{"site": "north"}, {"site": "south"}]
        if identities_match:
            return np.full(len(X), float(self.alpha))
        return np.full(len(X), 99.0)


def test_public_workspace_tuning_result_helpers_roundtrip(tmp_path) -> None:
    result = _make_result()

    tuning_id = nirs4all.save_workspace_tuning_result(
        tmp_path / "workspace",
        result,
        tuning_id="tune-main",
        name="main tuning",
        metadata={"purpose": "unit-test"},
    )
    restored = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", tuning_id)
    restored_by_fingerprint = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", result.fingerprint)

    assert tuning_id == "tune-main"
    assert restored.to_dict() == result.to_dict()
    assert restored_by_fingerprint.to_dict() == result.to_dict()


def test_public_workspace_tuning_result_helper_rejects_coercive_ids(tmp_path) -> None:
    result = _make_result()

    generated = nirs4all.save_workspace_tuning_result(tmp_path / "workspace", result)
    assert isinstance(generated, str)
    assert generated

    for tuning_id in ("", " tune-main ", "bad\x00id", 123):
        with pytest.raises(ValueError, match="save_tuning_result.tuning_id must be a canonical non-empty string"):
            nirs4all.save_workspace_tuning_result(
                tmp_path / "workspace",
                result,
                tuning_id=tuning_id,  # type: ignore[arg-type]
            )


def test_public_workspace_tuning_result_helper_rejects_coercive_link_ids(tmp_path) -> None:
    result = _make_result()

    for kwargs, label in (
        ({"run_id": " run-main "}, "save_tuning_result.run_id"),
        ({"pipeline_id": "pipe\x00main"}, "save_tuning_result.pipeline_id"),
        ({"chain_id": 123}, "save_tuning_result.chain_id"),
    ):
        with pytest.raises(ValueError, match=f"{label} must be a canonical non-empty string"):
            nirs4all.save_workspace_tuning_result(
                tmp_path / "workspace",
                result,
                **kwargs,  # type: ignore[arg-type]
            )


def test_public_tuning_result_summary_artifact_is_deterministic(tmp_path) -> None:
    result = _make_result()

    summary = result.summary_artifact()
    summary_json = result.to_summary_json(indent=None)
    summary_path = result.save_summary(tmp_path / "tuning-summary.json")
    schema = nirs4all.get_tuning_summary_schema()

    assert summary == {
        "best_params": {"model.n_components": 2},
        "best_value": 0.12,
        "direction": "minimize",
        "engine": "optuna",
        "fingerprint": result.fingerprint,
        "format": "nirs4all.tuning.summary",
        "metric": "rmse",
        "n_trials": 2,
        "optimizer": "optuna",
        "persistence": {
            "optimizer_state_resume_supported": True,
            "resume": False,
            "storage_configured": False,
            "study_name": None,
        },
        "pruner": None,
        "sampler": "grid",
        "schema_version": 1,
        "seed": None,
        "trial_states": {"COMPLETE": 2},
        "trials": [
            {"diagnostics": {}, "number": 0, "state": "COMPLETE", "value": 0.2},
            {"diagnostics": {}, "number": 1, "state": "COMPLETE", "value": 0.12},
        ],
        "version": 1,
    }
    assert json.loads(summary_json) == summary
    assert summary_path.read_text(encoding="utf-8") == result.to_summary_json()
    assert schema["$id"] == nirs4all.TUNING_SUMMARY_SCHEMA_ID
    assert schema["properties"]["format"]["const"] == "nirs4all.tuning.summary"
    assert schema["properties"]["sampler"] == {"type": ["string", "null"]}
    assert schema["properties"]["pruner"] == {"type": ["string", "null"]}
    assert schema["properties"]["seed"] == {"type": ["integer", "null"]}
    assert json.loads(nirs4all.tuning_summary_schema_json(indent=None)) == schema


def test_public_tuning_optimizer_persistence_keys_are_discoverable() -> None:
    entries = {entry["id"]: entry for entry in nirs4all.get_keyword_registry()["entries"]}

    assert nirs4all.TUNING_OPTIMIZER_PERSISTENCE_KEYS == ("storage", "study_name")
    assert set(nirs4all.TUNING_OPTIMIZER_PERSISTENCE_KEYS) <= set(nirs4all.TUNING_CONTRACT_KEYS)
    assert {f"run.tuning.{key}" for key in nirs4all.TUNING_OPTIMIZER_PERSISTENCE_KEYS} == {entry_id for entry_id, entry in entries.items() if entry["scope"] == "optimizer_persistence"}
    assert entries["run.tuning.storage"]["value_schema"]["pattern"] == "^[A-Za-z][A-Za-z0-9+.-]*://"
    assert entries["run.tuning.study_name"]["value_schema"]["pattern"] == "^[^\\u0000]+$"


def test_public_native_tuning_typed_payload_normalizes_to_existing_mapping() -> None:
    tuning = nirs4all.NativeTuning(
        engine="optuna",
        space={" alpha ": [0.2, 0.9], "scale": [nirs4all.TuningPassthrough()]},
        force_params={"alpha": 0.2, "scale": nirs4all.TuningPassthrough()},
        metric="RMSE",
        direction="MINIMIZE",
        n_trials=2,
        sampler="grid",
        score_data=nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            sample_ids=["score-a", "score-b"],
            groups=["batch-1", "batch-2"],
            metadata=[{"site": "north"}, {"site": "south"}],
        ),
        winner=nirs4all.TuningWinner(
            X=np.asarray([[30.0], [40.0]]),
            y_true=np.asarray([0.0, 0.0]),
            score=0.2,
            metric="rmse",
            sample_ids=["winner-a", "winner-b"],
            model_name="TypedWinner",
        ),
        calibration=nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            coverage=[0.8, 0.9],
            workspace_conformal_id="typed-conformal",
        ),
        workspace_tuning_id="typed-tuning",
        workspace_metadata={"purpose": "typed"},
    )

    payload = tuning.to_dict()
    spec = tuning.to_tuning_spec()

    assert payload["score_data"]["sample_ids"] == ["score-a", "score-b"]
    assert payload["winner"]["model_name"] == "TypedWinner"
    assert payload["calibration"]["workspace_conformal_id"] == "typed-conformal"
    assert payload["workspace_tuning_id"] == "typed-tuning"
    assert payload["space"]["scale"] == [{"kind": "passthrough"}]
    assert payload["force_params"] == {"alpha": 0.2, "scale": {"kind": "passthrough"}}
    assert nirs4all.TuningPassthrough().to_dict() == {"kind": "passthrough"}
    assert spec.space == {"alpha": [0.2, 0.9], "scale": [{"kind": "passthrough"}]}
    assert spec.force_params == {"alpha": 0.2, "scale": {"kind": "passthrough"}}
    assert spec.metric == "rmse"
    assert spec.direction == "minimize"


def test_public_native_tuning_to_tuning_spec_validates_storage_and_study_name() -> None:
    spec = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        storage=" sqlite:///study.db ",
        study_name=" pls_tuning ",
    ).to_tuning_spec()

    assert spec.storage == "sqlite:///study.db"
    assert spec.study_name == "pls_tuning"

    with pytest.raises(ValueError, match="storage must be a URI"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            storage="study.db",
        ).to_tuning_spec()

    with pytest.raises(ValueError, match="study_name must not contain NUL"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            study_name="bad\x00name",
        ).to_tuning_spec()


def test_public_tuning_winner_typed_payload_accepts_sample_id_aliases() -> None:
    assert nirs4all.TuningWinner(
        X=np.asarray([[30.0], [40.0]]),
        y_true=np.asarray([0.0, 0.0]),
        physical_sample_ids=["phys-a", "phys-b"],
    ).to_dict()["sample_ids"] == ["phys-a", "phys-b"]

    assert nirs4all.TuningWinner(
        dataset="dataset.json",
        selector={"partition": "winner"},
        prediction_sample_ids=["pred-a", "pred-b"],
    ).to_dict()["sample_ids"] == ["pred-a", "pred-b"]

    canonical = nirs4all.TuningWinner(
        X=np.asarray([[30.0], [40.0]]),
        y_true=np.asarray([0.0, 0.0]),
        metric=" RMSE ",
        dataset_name=" winner ",
        model_name=" WinnerModel ",
        task_type=" Regression ",
    ).to_dict()
    assert canonical["metric"] == "rmse"
    assert canonical["dataset_name"] == "winner"
    assert canonical["model_name"] == "WinnerModel"
    assert canonical["task_type"] == "regression"

    with pytest.raises(ValueError, match="multiple aliases"):
        nirs4all.TuningWinner(
            X=np.asarray([[30.0], [40.0]]),
            y_true=np.asarray([0.0, 0.0]),
            sample_ids=["a", "b"],
            winner_sample_ids=["c", "d"],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningWinner.metadata keys must be canonical non-empty strings"):
        nirs4all.TuningWinner(
            X=np.asarray([[30.0], [40.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata={1: "coerced"},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningWinner.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.TuningWinner(
            X=np.asarray([[30.0], [40.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata=[{1: "coerced"}],
        ).to_dict()

    json_metadata = nirs4all.TuningWinner(
        X=np.asarray([[30.0], [40.0]]),
        y_true=np.asarray([0.0, 0.0]),
        metadata={"site": "north", "flags": [True, None], "nested": {"fold": 1}},
    ).to_dict()["metadata"]
    assert json_metadata == {"site": "north", "flags": [True, None], "nested": {"fold": 1}}

    for metadata, message in (
        ({"bad": object()}, r"TuningWinner.metadata\[bad\] must be JSON-native"),
        ({"bad": float("nan")}, r"TuningWinner.metadata\[bad\] must be JSON-native and finite"),
        ({"bad": (1, 2)}, r"TuningWinner.metadata\[bad\] must be JSON-native"),
        ([{"bad": object()}], r"TuningWinner.metadata\[0\]\[bad\] must be JSON-native"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.TuningWinner(
                X=np.asarray([[30.0], [40.0]]),
                y_true=np.asarray([0.0, 0.0]),
                metadata=metadata,
            ).to_dict()

    for kwargs, message in (
        ({"metric": 1}, "TuningWinner.metric"),
        ({"metric": " "}, "TuningWinner.metric"),
        ({"dataset_name": 1}, "TuningWinner.dataset_name"),
        ({"dataset_name": " "}, "TuningWinner.dataset_name"),
        ({"model_name": 1}, "TuningWinner.model_name"),
        ({"model_name": " "}, "TuningWinner.model_name"),
        ({"task_type": 1}, "TuningWinner.task_type"),
        ({"task_type": " "}, "TuningWinner.task_type"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.TuningWinner(
                X=np.asarray([[30.0], [40.0]]),
                y_true=np.asarray([0.0, 0.0]),
                **kwargs,
            ).to_dict()


def test_public_native_tuning_winner_mapping_rejects_ambiguous_aliases() -> None:
    with pytest.raises(ValueError, match="winner features"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={
                "X": [[30.0]],
                "winner_x": [[30.0]],
                "y_true": [0.0],
                "score": 0.2,
                "metric": "rmse",
                "sample_ids": ["cal-a"],
            },
        ).to_dict()

    with pytest.raises(ValueError, match="winner sample_ids"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={
                "X": [[30.0]],
                "y_true": [0.0],
                "score": 0.2,
                "metric": "rmse",
                "sample_ids": ["cal-a"],
                "physical_sample_ids": ["cal-a"],
            },
        ).to_dict()

    with pytest.raises(ValueError, match="dataset-backed form must not also provide X/y_true"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={
                "dataset": "dataset.json",
                "selector": {"partition": "winner"},
                "winner_x": [[30.0]],
            },
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.score_data.metadata keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data={"X": [[10.0]], "y": [0.0], "metadata": {1: "coerced"}},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.score_data.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data={"X": [[10.0]], "y": [0.0], "score_metadata": [{1: "coerced"}]},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.score_data.metadata\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data={"X": [[10.0]], "y": [0.0], "metadata": {"bad": object()}},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.winner.metadata keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={"X": [[30.0]], "y_true": [0.0], "metadata": {1: "coerced"}},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.winner.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={"X": [[30.0]], "y_true": [0.0], "winner_metadata": [{1: "coerced"}]},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.winner.metadata\[bad\] must be JSON-native and finite"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
            winner={"X": [[30.0]], "y_true": [0.0], "metadata": {"bad": float("inf")}},
        ).to_dict()

    canonical = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        winner={
            "X": [[30.0]],
            "y_true": [0.0],
            "winner_metric": " RMSE ",
            "winner_dataset_name": " winner ",
            "winner_model_name": " WinnerModel ",
            "winner_task_type": " Regression ",
        },
    ).to_dict()["winner"]
    assert canonical["metric"] == "rmse"
    assert canonical["dataset_name"] == "winner"
    assert canonical["model_name"] == "WinnerModel"
    assert canonical["task_type"] == "regression"
    assert "winner_metric" not in canonical
    assert "winner_dataset_name" not in canonical
    assert "winner_model_name" not in canonical
    assert "winner_task_type" not in canonical

    for winner, message in (
        ({"X": [[30.0]], "y_true": [0.0], "metric": 1}, "NativeTuning.winner.metric"),
        ({"X": [[30.0]], "y_true": [0.0], "metric": " "}, "NativeTuning.winner.metric"),
        ({"X": [[30.0]], "y_true": [0.0], "dataset_name": 1}, "NativeTuning.winner.dataset_name"),
        ({"X": [[30.0]], "y_true": [0.0], "dataset_name": " "}, "NativeTuning.winner.dataset_name"),
        ({"X": [[30.0]], "y_true": [0.0], "model_name": 1}, "NativeTuning.winner.model_name"),
        ({"X": [[30.0]], "y_true": [0.0], "model_name": " "}, "NativeTuning.winner.model_name"),
        ({"X": [[30.0]], "y_true": [0.0], "task_type": 1}, "NativeTuning.winner.task_type"),
        ({"X": [[30.0]], "y_true": [0.0], "task_type": " "}, "NativeTuning.winner.task_type"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.NativeTuning(space={"alpha": [0.2]}, winner=winner).to_dict()


def test_public_tuning_score_data_typed_payload_accepts_sample_id_aliases() -> None:
    assert nirs4all.TuningScoreData(
        X=np.asarray([[10.0], [20.0]]),
        y=np.asarray([0.0, 0.0]),
        score_sample_ids=["score-a", "score-b"],
    ).to_dict()["sample_ids"] == ["score-a", "score-b"]

    score_alias_payload = nirs4all.TuningScoreData(
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        score_metric="mae",
        score_groups=["g1", "g2"],
        score_metadata=[{"site": "a"}, {"site": "b"}],
    ).to_dict()
    assert score_alias_payload["X"].shape == (2, 1)
    assert score_alias_payload["y"].shape == (2,)
    assert score_alias_payload["metric"] == "mae"
    assert score_alias_payload["groups"] == ["g1", "g2"]
    assert score_alias_payload["metadata"] == [{"site": "a"}, {"site": "b"}]
    assert "X_score" not in score_alias_payload
    assert "y_score" not in score_alias_payload
    assert "score_groups" not in score_alias_payload
    assert "score_metadata" not in score_alias_payload

    assert (
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            score_metric=" RMSE ",
        ).to_dict()["metric"]
        == "rmse"
    )

    with pytest.raises(ValueError, match="TuningScoreData metric"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            metric="rmse",
            score_metric="mae",
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData groups"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            groups=["g1", "g2"],
            score_groups=["h1", "h2"],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData metadata"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            metadata=[{"site": "a"}, {"site": "b"}],
            score_metadata=[{"site": "c"}, {"site": "d"}],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData.metadata keys must be canonical non-empty strings"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            metadata={1: "coerced"},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningScoreData.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            metadata=[{1: "coerced"}],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData.metadata keys must be canonical non-empty strings"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            score_metadata={" site ": "north"},
        ).to_dict()

    json_metadata = nirs4all.TuningScoreData(
        X=np.asarray([[10.0], [20.0]]),
        y=np.asarray([0.0, 0.0]),
        metadata={"site": "north", "flags": [True, None], "nested": {"fold": 1}},
    ).to_dict()["metadata"]
    assert json_metadata == {"site": "north", "flags": [True, None], "nested": {"fold": 1}}

    for metadata, message in (
        ({"bad": object()}, r"TuningScoreData.metadata\[bad\] must be JSON-native"),
        ({"bad": float("nan")}, r"TuningScoreData.metadata\[bad\] must be JSON-native and finite"),
        ({"bad": (1, 2)}, r"TuningScoreData.metadata\[bad\] must be JSON-native"),
        ([{"bad": object()}], r"TuningScoreData.metadata\[0\]\[bad\] must be JSON-native"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
                metadata=metadata,
            ).to_dict()

    for metric in (1, " "):
        with pytest.raises(ValueError, match="TuningScoreData.metric"):
            nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
                metric=metric,
            ).to_dict()

    assert nirs4all.TuningScoreData(
        dataset="dataset.json",
        selector={"partition": "score"},
        physical_sample_ids=["phys-a", "phys-b"],
    ).to_dict()["sample_ids"] == ["phys-a", "phys-b"]

    with pytest.raises(ValueError, match="TuningScoreData.selector keys must be canonical non-empty strings"):
        nirs4all.TuningScoreData(
            dataset="dataset.json",
            selector={1: "score"},  # type: ignore[dict-item]
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData.selector keys must be canonical non-empty strings"):
        nirs4all.TuningScoreData(
            dataset="dataset.json",
            selector={" partition ": "score"},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningScoreData.selector\[bad\] must be JSON-native"):
        nirs4all.TuningScoreData(
            dataset="dataset.json",
            selector={"bad": object()},
        ).to_dict()

    with pytest.raises(ValueError, match="TuningScoreData.include_augmented must be a boolean"):
        nirs4all.TuningScoreData(
            dataset="dataset.json",
            selector={"partition": "score"},
            include_augmented="yes",  # type: ignore[arg-type]
        ).to_dict()

    with pytest.raises(ValueError, match="multiple aliases"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            X_score=np.asarray([[10.0], [20.0]]),
            y_score=np.asarray([0.0, 0.0]),
        ).to_dict()

    with pytest.raises(ValueError, match="both X_score and y_score"):
        nirs4all.TuningScoreData(
            X_score=np.asarray([[10.0], [20.0]]),
        ).to_dict()

    with pytest.raises(ValueError, match="multiple aliases"):
        nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            sample_ids=["a", "b"],
            prediction_sample_ids=["c", "d"],
        ).to_dict()


def test_public_tuning_score_data_validates_conformal_coverage_contract() -> None:
    payload = nirs4all.TuningScoreData(
        X=np.asarray([[10.0], [20.0]]),
        y=np.asarray([0.0, 0.0]),
        conformal_coverage=0.8,
    ).to_dict()

    assert payload["conformal_coverage"] == pytest.approx(0.8)

    for invalid in (0.0, 1.0, float("nan"), True, "0.8", [0.8]):
        with pytest.raises(ValueError, match="TuningScoreData.conformal_coverage"):
            nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
                conformal_coverage=invalid,
            ).to_dict()


def test_public_tuning_conformal_score_calibration_accepts_sample_id_aliases() -> None:
    assert nirs4all.TuningConformalScoreCalibration(
        X=np.asarray([[3.0], [4.0]]),
        y_true=np.asarray([0.0, 0.0]),
        calibration_sample_ids=["cal-a", "cal-b"],
    ).to_dict()["sample_ids"] == ["cal-a", "cal-b"]

    y_alias_payload = nirs4all.TuningConformalScoreCalibration(
        X=np.asarray([[3.0], [4.0]]),
        y=np.asarray([0.0, 0.0]),
    ).to_dict()
    assert "y" not in y_alias_payload
    assert y_alias_payload["y_true"].shape == (2,)

    calibration_alias_payload = nirs4all.TuningConformalScoreCalibration(
        X_calibration=np.asarray([[3.0], [4.0]]),
        y_calibration=np.asarray([0.0, 0.0]),
        calibration_groups=["g1", "g2"],
        calibration_metadata=[{"site": "a"}, {"site": "b"}],
    ).to_dict()
    assert calibration_alias_payload["X"].shape == (2, 1)
    assert calibration_alias_payload["y_true"].shape == (2,)
    assert calibration_alias_payload["groups"] == ["g1", "g2"]
    assert calibration_alias_payload["metadata"] == [{"site": "a"}, {"site": "b"}]
    assert "X_calibration" not in calibration_alias_payload
    assert "y_calibration" not in calibration_alias_payload
    assert "calibration_groups" not in calibration_alias_payload
    assert "calibration_metadata" not in calibration_alias_payload

    feature_target_payload = nirs4all.TuningConformalScoreCalibration(
        features=np.asarray([[3.0], [4.0]]),
        target=np.asarray([0.0, 0.0]),
    ).to_dict()
    assert feature_target_payload["X"].shape == (2, 1)
    assert feature_target_payload["y_true"].shape == (2,)

    assert nirs4all.TuningConformalScoreCalibration(
        X=np.asarray([[3.0], [4.0]]),
        y_true=np.asarray([0.0, 0.0]),
        physical_sample_ids=["phys-a", "phys-b"],
    ).to_dict()["sample_ids"] == ["phys-a", "phys-b"]

    with pytest.raises(ValueError, match="multiple aliases"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            sample_ids=["a", "b"],
            calibration_sample_ids=["c", "d"],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration target"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            y=np.asarray([1.0, 1.0]),
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration features"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            features=np.asarray([[5.0], [6.0]]),
            y_true=np.asarray([0.0, 0.0]),
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration groups"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            groups=["g1", "g2"],
            calibration_groups=["h1", "h2"],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration metadata"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata=[{"site": "a"}, {"site": "b"}],
            calibration_metadata=[{"site": "c"}, {"site": "d"}],
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration.metadata keys must be canonical non-empty strings"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata={1: "coerced"},
        ).to_dict()

    with pytest.raises(ValueError, match="TuningConformalScoreCalibration.metadata keys must be canonical non-empty strings"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata={" site ": "north"},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningConformalScoreCalibration.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata=[{1: "coerced"}],
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningConformalScoreCalibration.metadata\[0\] must be a mapping"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata=["not-a-row-mapping"],
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningConformalScoreCalibration.metadata\[bad\] must be JSON-native"):
        nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0]]),
            y_true=np.asarray([0.0, 0.0]),
            metadata={"bad": object()},
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration.metadata keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            metric="conformal_mean_width",
            score_data={
                "X": [[1.0]],
                "y": [1.0],
                "conformal_calibration": {
                    "X": [[1.0]],
                    "y_true": [1.0],
                    "metadata": {1: "coerced"},
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match=r"score_data.conformal_calibration.metadata\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            metric="conformal_mean_width",
            score_data={
                "X": [[1.0]],
                "y": [1.0],
                "conformal_calibration": {
                    "X": [[1.0]],
                    "y_true": [1.0],
                    "metadata": {"bad": object()},
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match=r"score_data.conformal_calibration.metadata\[0\] keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            metric="conformal_mean_width",
            score_data={
                "X": [[1.0]],
                "y": [1.0],
                "conformal_calibration": {
                    "X": [[1.0]],
                    "y_true": [1.0],
                    "metadata": [{1: "coerced"}],
                },
            },
        ).to_dict()


def test_public_tuning_calibration_normalizes_and_validates_method_unit() -> None:
    payload = nirs4all.TuningCalibration(
        y_pred=np.asarray([1.0]),
        prediction_sample_ids=["pred-a"],
        method=" SPLIT_ABSOLUTE_RESIDUAL ",
        unit=" PHYSICAL_SAMPLE ",
    ).to_dict()

    assert payload["method"] == "split_absolute_residual"
    assert payload["unit"] == "physical_sample"

    with pytest.raises(ValueError, match="TuningCalibration.method"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            method="unsupported",
        ).to_dict()

    with pytest.raises(ValueError, match="TuningCalibration.unit"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            unit="observation",
        ).to_dict()


def test_public_tuning_calibration_validates_coverage_contract() -> None:
    payload = nirs4all.TuningCalibration(
        y_pred=np.asarray([1.0]),
        prediction_sample_ids=["pred-a"],
        coverage=(0.8, 0.9),
    ).to_dict()

    assert payload["coverage"] == [0.8, 0.9]

    for invalid in (0.0, 1.0, [], [0.9, 0.9], [0.8, float("nan")], [0.8, "0.9"], True, "0.8"):
        with pytest.raises(ValueError, match="TuningCalibration.coverage"):
            nirs4all.TuningCalibration(
                y_pred=np.asarray([1.0]),
                prediction_sample_ids=["pred-a"],
                coverage=invalid,
            ).to_dict()


def test_public_tuning_calibration_extra_cannot_override_reserved_keys() -> None:
    payload = nirs4all.TuningCalibration(
        y_pred=np.asarray([1.0]),
        prediction_sample_ids=["pred-a"],
        workspace_metadata={"site": "north", "flags": [True, None], "nested": {"fold": 1}},
        extra={"target_name": "protein", "nested": {"fold": 1}},
    ).to_dict()

    assert payload["target_name"] == "protein"
    assert payload["nested"] == {"fold": 1}
    assert payload["workspace_metadata"] == {"site": "north", "flags": [True, None], "nested": {"fold": 1}}

    with pytest.raises(ValueError, match="TuningCalibration.as_predict_result must be a boolean"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            as_predict_result="yes",  # type: ignore[arg-type]
        ).to_dict()

    with pytest.raises(ValueError, match="TuningCalibration.extra keys must be canonical non-empty strings"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            extra={1: "coerced"},  # type: ignore[dict-item]
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningCalibration.extra\[bad\] must be JSON-native"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            extra={"bad": object()},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningCalibration.workspace_metadata\[bad\] must be JSON-native and finite"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            workspace_metadata={"bad": float("-inf")},
        ).to_dict()

    with pytest.raises(ValueError, match="TuningCalibration.workspace_conformal_id must be a canonical non-empty string"):
        nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0]),
            prediction_sample_ids=["pred-a"],
            workspace_conformal_id="bad\x00id",
        ).to_dict()

    for key, value in (
        ("calibration_data", {"y_true": [1.0]}),
        ("coverage", [0.9, 0.9]),
        ("method", "unsupported"),
        ("prediction_sample_ids", ["other"]),
        ("unit", "observation"),
        ("y_pred", [2.0]),
    ):
        with pytest.raises(ValueError, match="TuningCalibration.extra"):
            nirs4all.TuningCalibration(
                y_pred=np.asarray([1.0]),
                prediction_sample_ids=["pred-a"],
                extra={key: value},
            ).to_dict()


def test_public_tuning_winner_and_native_tuning_reject_coercive_payloads() -> None:
    canonical = nirs4all.NativeTuning(
        space={" alpha ": [0.1, 0.2]},
        force_params={" alpha ": 0.1},
        engine=" OPTUNA ",
        metric=" RMSE ",
        direction=" MINIMIZE ",
        n_trials=2,
        sampler=" GRID ",
        pruner=" NONE ",
        seed=3,
        resume=False,
        storage=" sqlite:///study.db ",
        study_name=" native-study ",
        workspace_tuning_id="native-tuning",
    ).to_dict()

    assert canonical["space"] == {"alpha": [0.1, 0.2]}
    assert canonical["force_params"] == {"alpha": 0.1}
    assert canonical["engine"] == "optuna"
    assert canonical["metric"] == "rmse"
    assert canonical["direction"] == "minimize"
    assert canonical["n_trials"] == 2
    assert canonical["sampler"] == "grid"
    assert canonical["pruner"] == "none"
    assert canonical["seed"] == 3
    assert canonical["resume"] is False
    assert canonical["storage"] == "sqlite:///study.db"
    assert canonical["study_name"] == "native-study"
    assert canonical["workspace_tuning_id"] == "native-tuning"

    for kwargs, message in (
        ({"n_trials": True}, "NativeTuning.n_trials must be a positive integer"),
        ({"n_trials": "2"}, "NativeTuning.n_trials must be a positive integer"),
        ({"seed": True}, "NativeTuning.seed must be an integer"),
        ({"seed": "1"}, "NativeTuning.seed must be an integer"),
        ({"resume": "yes"}, "NativeTuning.resume must be a boolean"),
        ({"engine": 1}, "NativeTuning.engine must be a non-empty string"),
        ({"metric": 1}, "metric must be a non-empty string"),
        ({"direction": 1}, "direction must be a non-empty string"),
        ({"sampler": 1}, "sampler must be a non-empty string"),
        ({"pruner": 1}, "pruner must be a non-empty string"),
        ({"storage": 1}, "storage must be a non-empty string"),
        ({"study_name": 1}, "study_name must be a non-empty string"),
        ({"workspace_tuning_id": "bad\x00id"}, "NativeTuning.workspace_tuning_id must be a canonical non-empty string"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.NativeTuning(space={"alpha": [0.1]}, **kwargs).to_dict()

    with pytest.raises(ValueError, match="TuningWinner.selector keys must be canonical non-empty strings"):
        nirs4all.TuningWinner(
            dataset="dataset.json",
            selector={" partition ": "winner"},
        ).to_dict()

    with pytest.raises(ValueError, match=r"TuningWinner.selector\[bad\] must be JSON-native"):
        nirs4all.TuningWinner(
            dataset="dataset.json",
            selector={"bad": object()},
        ).to_dict()

    with pytest.raises(ValueError, match="TuningWinner.include_augmented must be a boolean"):
        nirs4all.TuningWinner(
            dataset="dataset.json",
            selector={"partition": "winner"},
            include_augmented="yes",  # type: ignore[arg-type]
        ).to_dict()

    with pytest.raises(ValueError, match="TuningWinner.score must be a finite number"):
        nirs4all.TuningWinner(X=[[1.0]], y_true=[1.0], score=True).to_dict()

    with pytest.raises(ValueError, match="TuningWinner.score must be a finite number"):
        nirs4all.TuningWinner(X=[[1.0]], y_true=[1.0], score="1.0").to_dict()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="NativeTuning.space keys must be non-empty strings"):
        nirs4all.NativeTuning(space={1: [0.1]}).to_dict()  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="NativeTuning.workspace_metadata keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(space={"alpha": [0.1]}, workspace_metadata={1: "coerced"}).to_dict()  # type: ignore[dict-item]

    with pytest.raises(ValueError, match=r"NativeTuning.workspace_metadata\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(space={"alpha": [0.1]}, workspace_metadata={"bad": object()}).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.score_data.selector keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            score_data={"dataset": "dataset.json", "selector": {1: "score"}},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.score_data.selector\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            score_data={"dataset": "dataset.json", "selector": {"bad": object()}},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.score_data.include_augmented must be a boolean"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            score_data={"dataset": "dataset.json", "selector": {"partition": "score"}, "include_augmented": "yes"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.winner.score must be a finite number"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            winner={"X": [[1.0]], "y_true": [1.0], "score": True},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.winner.score must be a finite number"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            winner={"X": [[1.0]], "y_true": [1.0], "score": "1.0"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.winner.selector keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            winner={"dataset": "dataset.json", "selector": {1: "winner"}},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.winner.selector\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            winner={"dataset": "dataset.json", "selector": {"bad": object()}},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.winner.include_augmented must be a boolean"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            winner={"dataset": "dataset.json", "selector": {"partition": "winner"}, "include_augmented": "yes"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration.as_predict_result must be a boolean"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "as_predict_result": "yes"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], 1: "coerced"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration must not include calibration_data"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "calibration_data": {"y_true": [1.0]}},
        ).to_dict()

    for invalid_coverage in (True, "0.9", [0.9, "0.8"], [0.9, 0.9]):
        with pytest.raises(ValueError, match="NativeTuning.calibration.coverage"):
            nirs4all.NativeTuning(
                space={"alpha": [0.1]},
                calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "coverage": invalid_coverage},
            ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration.method"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "method": "unsupported"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration.unit"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "unit": "observation"},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration.workspace_metadata keys must be canonical non-empty strings"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "workspace_metadata": {1: "coerced"}},
        ).to_dict()

    with pytest.raises(ValueError, match=r"NativeTuning.calibration.workspace_metadata\[bad\] must be JSON-native"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "workspace_metadata": {"bad": object()}},
        ).to_dict()

    with pytest.raises(ValueError, match="NativeTuning.calibration.workspace_conformal_id must be a canonical non-empty string"):
        nirs4all.NativeTuning(
            space={"alpha": [0.1]},
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"], "workspace_conformal_id": "bad\x00id"},
        ).to_dict()

    normalized_calibration = nirs4all.NativeTuning(
        space={"alpha": [0.1]},
        calibration={
            "y_pred": [1.0],
            "prediction_sample_ids": ["pred-a"],
            "coverage": (0.8, 0.9),
            "method": " SPLIT_ABSOLUTE_RESIDUAL ",
            "unit": " PHYSICAL_SAMPLE ",
            "workspace_metadata": {"purpose": "raw"},
            "workspace_conformal_id": "raw-conformal",
        },
    ).to_dict()["calibration"]

    assert normalized_calibration["coverage"] == [0.8, 0.9]
    assert normalized_calibration["method"] == "split_absolute_residual"
    assert normalized_calibration["unit"] == "physical_sample"
    assert normalized_calibration["workspace_metadata"] == {"purpose": "raw"}
    assert normalized_calibration["workspace_conformal_id"] == "raw-conformal"


def test_public_tuning_vocabulary_constants_match_registry_and_runtime() -> None:
    registry_entries = {entry["id"]: entry for entry in nirs4all.get_keyword_registry()["entries"]}

    assert nirs4all.TUNING_ENGINES == ("optuna", "n4m")
    assert nirs4all.TUNING_DIRECTIONS == ("minimize", "maximize")
    assert set(nirs4all.TUNING_ENGINES) == set(SUPPORTED_TUNING_ENGINES)
    assert set(nirs4all.TUNING_DIRECTIONS) == set(SUPPORTED_TUNING_DIRECTIONS)
    assert nirs4all.TUNING_CONTRACT_KEYS == (
        "direction",
        "engine",
        "force_params",
        "metric",
        "n_trials",
        "pruner",
        "resume",
        "sampler",
        "seed",
        "space",
        "storage",
        "study_name",
    )
    assert set(nirs4all.TUNING_CONTRACT_KEYS) == set(SUPPORTED_TUNING_KEYS)
    assert nirs4all.TUNING_RUNTIME_KEYS == (
        "calibration",
        "score_data",
        "tuning_id",
        "winner",
        "workspace_metadata",
        "workspace_tuning_id",
    )
    assert registry_entries["run.tuning.engine"]["value_schema"]["enum"] == list(nirs4all.TUNING_ENGINES)

    for engine in nirs4all.TUNING_ENGINES:
        spec = parse_tuning_spec({"engine": engine, "space": {"alpha": [0.2]}, "n_trials": 1})
        assert spec.engine == engine

    for direction in nirs4all.TUNING_DIRECTIONS:
        spec = nirs4all.NativeTuning(space={"alpha": [0.2]}, direction=direction, n_trials=1).to_tuning_spec()
        assert spec.direction == direction

    assert "conformal_mean_width" in nirs4all.CONFORMAL_TUNING_SCORE_METRICS
    conformal_payload = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        metric=nirs4all.CONFORMAL_TUNING_SCORE_METRICS[3],
        score_data=nirs4all.TuningScoreData(
            X=np.asarray([[10.0], [20.0]]),
            y=np.asarray([0.0, 0.0]),
            conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                X=np.asarray([[3.0], [4.0], [5.0], [6.0]]),
                y=np.asarray([0.0, 0.0, 0.0, 0.0]),
            ),
        ),
    ).to_dict()
    assert conformal_payload["metric"] in nirs4all.CONFORMAL_TUNING_SCORE_METRICS


def test_public_inspect_tuning_space_returns_ordered_patch_contract() -> None:
    artifact = nirs4all.inspect_tuning_space(
        nirs4all.NativeTuning(
            space={
                "ridge__alpha": [0.1, 0.2],
                "scale.with_mean": [False],
            },
            force_params={"ridge.alpha": 0.1},
            sampler="grid",
            n_trials=2,
        )
    )

    assert artifact["format"] == "nirs4all.tuning.ordered_search_space"
    assert artifact["schema_version"] == 1
    assert len(artifact["fingerprint"]) == 64
    assert len(artifact["tuning_fingerprint"]) == 64
    assert artifact["parameters"] == [
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
    ]
    assert artifact["force_params"] == [
        {
            "path": "ridge.alpha",
            "segments": ["ridge", "alpha"],
            "value": 0.1,
        }
    ]


def test_public_inspect_tuning_space_matches_cross_repo_fixture() -> None:
    expected = json.loads((FIXTURES / "ordered_search_space_v1.json").read_text(encoding="utf-8"))

    artifact = nirs4all.inspect_tuning_space(
        {
            "engine": "optuna",
            "n_trials": 50,
            "space": {
                "model.n_components": {"type": "int", "low": 2, "high": 12, "step": 1},
                "model.alpha": {"type": "log_float", "low": 1e-4, "high": 1, "log": True},
                "train.batch_size": [16, 32, 64],
            },
            "force_params": {
                "model.n_components": 6,
                "train.batch_size": 32,
            },
        }
    )

    assert artifact == expected


def test_public_tuning_space_schema_helpers_are_exported() -> None:
    schema = nirs4all.get_tuning_space_schema()

    assert nirs4all.TUNING_SPACE_SCHEMA_ID == "https://nirs4all.org/schemas/tuning-ordered-search-space/v1"
    assert schema["$id"] == nirs4all.TUNING_SPACE_SCHEMA_ID
    assert schema["properties"]["format"]["const"] == "nirs4all.tuning.ordered_search_space"
    assert json.loads(nirs4all.tuning_space_schema_json(indent=None)) == schema


def test_public_native_tuning_exposes_space_inspection_method() -> None:
    tuning = nirs4all.NativeTuning(space={"alpha": [0.2]}, n_trials=1)

    assert tuning.inspect_space() == nirs4all.inspect_tuning_space(tuning)
    assert isinstance(tuning.to_tuning_spec().ordered_search_space, nirs4all.OrderedSearchSpaceSpec)
    [parameter] = tuning.to_tuning_spec().ordered_search_space.parameters
    assert isinstance(parameter, nirs4all.SearchSpaceParameter)
    [patch] = tuning.to_tuning_spec().parameter_patches({"alpha": 0.2})
    assert isinstance(patch, nirs4all.ParameterPatch)


def test_public_finetune_vocabulary_constants_match_registry_and_runtimes() -> None:
    from nirs4all.optimization.n4m_engine import _PRUNER_MAP as n4m_pruner_map
    from nirs4all.optimization.n4m_engine import _SAMPLER_MAP as n4m_sampler_map
    from nirs4all.optimization.optuna import VALID_APPROACHES, VALID_EVAL_MODES, VALID_PRUNERS, VALID_SAMPLERS
    from nirs4all.pipeline.dagml.finetune_lowering import (
        DETERMINISTIC_FINETUNE_ENGINES,
        PUBLIC_DAGML_SELECTION_METRICS,
        SUPPORTED_FINETUNE_META_KEYS,
        lower_deterministic_finetune_params_to_generators,
    )

    registry_entries = {entry["id"]: entry for entry in nirs4all.get_keyword_registry()["entries"]}
    engine_entry = registry_entries["pipeline.step.finetune_params.engine"]
    sampler_entry = registry_entries["pipeline.step.finetune_params.sampler"]
    eval_mode_entry = registry_entries["pipeline.step.finetune_params.eval_mode"]

    assert nirs4all.FINETUNE_ENGINES == tuple(engine_entry["value_schema"]["enum"])
    assert nirs4all.FINETUNE_ENGINE_ALIASES == tuple((alias["name"], alias["canonical"]) for alias in engine_entry["aliases"])
    assert nirs4all.FINETUNE_SAMPLER_KEY_ALIASES == tuple((alias["name"], alias["canonical"]) for alias in sampler_entry["aliases"])
    assert nirs4all.FINETUNE_EVAL_MODES == tuple(eval_mode_entry["value_schema"]["enum"])
    assert nirs4all.FINETUNE_EVAL_MODE_ALIASES == tuple((alias["name"], alias["canonical"]) for alias in eval_mode_entry["aliases"])

    assert set(nirs4all.FINETUNE_APPROACHES) == VALID_APPROACHES
    assert set(nirs4all.FINETUNE_EVAL_MODES) == VALID_EVAL_MODES
    assert set(nirs4all.FINETUNE_OPTUNA_SAMPLERS) == VALID_SAMPLERS
    assert set(nirs4all.FINETUNE_OPTUNA_PRUNERS) == VALID_PRUNERS
    assert set(nirs4all.FINETUNE_N4M_SAMPLERS) == set(n4m_sampler_map) - {"sample"}
    assert set(nirs4all.FINETUNE_N4M_PRUNERS) == set(n4m_pruner_map)
    assert set(nirs4all.FINETUNE_DAGML_DETERMINISTIC_ENGINES) | {"", "dagml", "native"} == DETERMINISTIC_FINETUNE_ENGINES
    assert set(nirs4all.FINETUNE_DAGML_META_KEYS) == SUPPORTED_FINETUNE_META_KEYS
    assert set(nirs4all.FINETUNE_DAGML_SELECTION_METRICS) == PUBLIC_DAGML_SELECTION_METRICS
    assert nirs4all.FINETUNE_DAGML_APPROACHES == ("grouped",)
    assert nirs4all.FINETUNE_DAGML_EVAL_MODES == ("mean", "best")

    for engine in (*nirs4all.FINETUNE_DAGML_DETERMINISTIC_ENGINES, "dagml", "native"):
        steps, overrides = lower_deterministic_finetune_params_to_generators(
            [
                {
                    "model": object(),
                    "finetune_params": {
                        "engine": engine,
                        "metric": nirs4all.FINETUNE_DAGML_SELECTION_METRICS[0],
                        "direction": nirs4all.TUNING_DIRECTIONS[0],
                        "model_params": {"alpha": [0.1, 1.0]},
                    },
                }
            ],
            supported_selection_metrics=PUBLIC_DAGML_SELECTION_METRICS,
        )
        assert overrides == {"selection_metric": "rmse", "selection_objective": "minimize"}
        assert steps[0]["_grid_"] == {"alpha": [0.1, 1.0]}


def test_public_native_tuning_conformal_score_payload_validation() -> None:
    conformal_score = nirs4all.TuningScoreData(
        X=np.asarray([[10.0], [20.0]]),
        y=np.asarray([0.0, 0.0]),
        conformal_calibration=nirs4all.TuningConformalScoreCalibration(
            X=np.asarray([[3.0], [4.0], [5.0], [6.0]]),
            y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
        ),
        conformal_coverage=0.8,
    )
    payload = nirs4all.NativeTuning(
        engine="optuna",
        space={"alpha": [0.2, 0.9]},
        metric="conformal_mean_width",
        direction="minimize",
        n_trials=2,
        sampler="grid",
        score_data=conformal_score,
    ).to_dict()

    assert payload["score_data"]["conformal_coverage"] == pytest.approx(0.8)
    assert payload["score_data"]["conformal_calibration"]["y_true"].shape == (4,)

    with pytest.raises(ValueError, match="conformal score metric"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="rmse",
            score_data=conformal_score,
        ).to_dict()

    with pytest.raises(ValueError, match="conformal_calibration"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0]]),
                y=np.asarray([0.0]),
            ),
        ).to_dict()

    with pytest.raises(ValueError, match="conformal_coverage"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0]]),
                y=np.asarray([0.0]),
                conformal_coverage=0.8,
            ),
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_coverage"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
                "coverage": "0.8",
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data metric"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "metric": "conformal_mean_width",
                "score_metric": "conformal_median_width",
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data features"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "X_score": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data conformal_calibration"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
                "conformal_score_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data sample_ids"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "sample_ids": ["a"],
                "score_sample_ids": ["b"],
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data groups"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "groups": ["g"],
                "score_groups": ["h"],
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data metadata"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "metadata": [{"site": "a"}],
                "score_metadata": [{"site": "b"}],
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration features"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "features": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration target"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "target": np.asarray([0.0, 0.0]),
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration sample_ids"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "sample_ids": ["a", "b"],
                    "physical_sample_ids": ["c", "d"],
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration groups"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "groups": ["g1", "g2"],
                    "calibration_groups": ["h1", "h2"],
                },
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data.conformal_calibration metadata"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="conformal_mean_width",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "conformal_calibration": {
                    "X": np.asarray([[3.0], [4.0]]),
                    "y_true": np.asarray([0.0, 0.0]),
                    "metadata": [{"site": "a"}, {"site": "b"}],
                    "calibration_metadata": [{"site": "c"}, {"site": "d"}],
                },
            },
        ).to_dict()


def test_public_native_tuning_standard_score_data_rejects_ambiguous_aliases() -> None:
    sequence_payload = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        metric="rmse",
        score_data=(
            np.asarray([[10.0]]),
            np.asarray([0.0]),
            ["score-a"],
            ["batch-a"],
            ({"site": "north"},),
        ),
    ).to_dict()["score_data"]
    assert isinstance(sequence_payload, list)
    assert sequence_payload[2] == ["score-a"]
    assert sequence_payload[3] == ["batch-a"]
    assert sequence_payload[4] == [{"site": "north"}]

    for score_data, message in (
        ((np.asarray([[10.0]]),), "NativeTuning.score_data tuple/list must contain"),
        (
            (np.asarray([[10.0]]), np.asarray([0.0]), ["s"], ["g"], [{"site": "north"}], "extra"),
            "NativeTuning.score_data tuple/list supports at most",
        ),
        (
            (np.asarray([[10.0]]), np.asarray([0.0]), None, None, {1: "coerced"}),
            "NativeTuning.score_data.metadata keys must be canonical non-empty strings",
        ),
        (
            (np.asarray([[10.0]]), np.asarray([0.0]), None, None, [{1: "coerced"}]),
            r"NativeTuning.score_data.metadata\[0\] keys must be canonical non-empty strings",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.NativeTuning(
                space={"alpha": [0.2]},
                metric="rmse",
                score_data=score_data,
            ).to_dict()

    for score_data in (1, "bad", b"bad", object()):
        with pytest.raises(TypeError, match="NativeTuning.score_data must be a mapping, tuple, or list"):
            nirs4all.NativeTuning(
                space={"alpha": [0.2]},
                metric="rmse",
                score_data=score_data,
            ).to_dict()

    canonical = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        metric="rmse",
        score_data={
            "X": np.asarray([[10.0]]),
            "y": np.asarray([0.0]),
            "score_metric": " RMSE ",
        },
    ).to_dict()["score_data"]
    assert canonical["metric"] == "rmse"
    assert "score_metric" not in canonical

    for metric in (1, " "):
        with pytest.raises(ValueError, match="NativeTuning.score_data.metric"):
            nirs4all.NativeTuning(
                space={"alpha": [0.2]},
                metric="rmse",
                score_data={
                    "X": np.asarray([[10.0]]),
                    "y": np.asarray([0.0]),
                    "metric": metric,
                },
            ).to_dict()

    with pytest.raises(ValueError, match="score_data features"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="rmse",
            score_data={
                "X": np.asarray([[10.0]]),
                "X_score": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
            },
        ).to_dict()

    with pytest.raises(ValueError, match="score_data target"):
        nirs4all.NativeTuning(
            space={"alpha": [0.2]},
            metric="rmse",
            score_data={
                "X": np.asarray([[10.0]]),
                "y": np.asarray([0.0]),
                "y_score": np.asarray([0.0]),
            },
        ).to_dict()


def test_public_native_tuning_dataset_backed_typed_payload_requires_selector() -> None:
    with pytest.raises(ValueError, match="requires selector"):
        nirs4all.TuningScoreData(dataset="dataset.json").to_dict()

    with pytest.raises(ValueError, match="TuningScoreData dataset-backed form must not also provide X/y"):
        nirs4all.TuningScoreData(
            dataset="dataset.json",
            selector={"partition": "score"},
            X=np.asarray([[1.0]]),
            y=np.asarray([1.0]),
        ).to_dict()

    with pytest.raises(ValueError, match="TuningWinner dataset-backed form must not also provide X/y_true"):
        nirs4all.TuningWinner(
            dataset="dataset.json",
            selector={"partition": "calibration"},
            X=np.asarray([[1.0]]),
            y_true=np.asarray([1.0]),
        ).to_dict()

    payload = nirs4all.TuningWinner(
        dataset="dataset.json",
        selector={"partition": "calibration"},
        sample_id_column="Sample_ID",
        metadata_columns=["Site"],
        score=0.2,
        metric="rmse",
    ).to_dict()

    assert payload["dataset"] == "dataset.json"
    assert payload["selector"] == {"partition": "calibration"}
    assert payload["sample_id_column"] == "Sample_ID"
    assert payload["metadata_columns"] == ["Site"]

    score_payload = nirs4all.TuningScoreData(
        dataset="dataset.json",
        selector={"partition": "score"},
        sample_id_column="Sample_ID",
        group_column="Batch",
        metadata_columns=("Site", "Instrument"),
    ).to_dict()
    assert score_payload["sample_id_column"] == "Sample_ID"
    assert score_payload["group_column"] == "Batch"
    assert score_payload["metadata_columns"] == ["Site", "Instrument"]

    for kwargs, message in (
        ({"sample_id_column": 1}, "TuningScoreData.sample_id_column"),
        ({"sample_id_column": " Sample_ID "}, "TuningScoreData.sample_id_column"),
        ({"group_column": 1}, "TuningScoreData.group_column"),
        ({"group_column": " "}, "TuningScoreData.group_column"),
        ({"metadata_columns": 1}, "TuningScoreData.metadata_columns"),
        ({"metadata_columns": ["Site", 1]}, r"TuningScoreData.metadata_columns\[1\]"),
        ({"metadata_columns": ["Site", " Site "]}, r"TuningScoreData.metadata_columns\[1\]"),
        ({"metadata_columns": ["Site", "Site"]}, "TuningScoreData.metadata_columns contains duplicate column names"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.TuningScoreData(
                dataset="dataset.json",
                selector={"partition": "score"},
                **kwargs,
            ).to_dict()

    for kwargs, message in (
        ({"sample_id_column": 1}, "TuningWinner.sample_id_column"),
        ({"group_column": 1}, "TuningWinner.group_column"),
        ({"metadata_columns": ["Site", 1]}, r"TuningWinner.metadata_columns\[1\]"),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.TuningWinner(
                dataset="dataset.json",
                selector={"partition": "calibration"},
                **kwargs,
            ).to_dict()

    raw_score = nirs4all.NativeTuning(
        space={"alpha": [0.2]},
        score_data={
            "dataset": "dataset.json",
            "selector": {"partition": "score"},
            "metadata_columns": ("Site", "Instrument"),
        },
    ).to_dict()["score_data"]
    assert raw_score["metadata_columns"] == ["Site", "Instrument"]

    for score_data, message in (
        (
            {"dataset": "dataset.json", "selector": {"partition": "score"}, "sample_id_column": 1},
            "NativeTuning.score_data.sample_id_column",
        ),
        (
            {"dataset": "dataset.json", "selector": {"partition": "score"}, "group_column": " "},
            "NativeTuning.score_data.group_column",
        ),
        (
            {"dataset": "dataset.json", "selector": {"partition": "score"}, "metadata_columns": ["Site", 1]},
            r"NativeTuning.score_data.metadata_columns\[1\]",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.NativeTuning(space={"alpha": [0.2]}, score_data=score_data).to_dict()

    for winner, message in (
        (
            {"dataset": "dataset.json", "selector": {"partition": "winner"}, "sample_id_column": 1},
            "NativeTuning.winner.sample_id_column",
        ),
        (
            {"dataset": "dataset.json", "selector": {"partition": "winner"}, "metadata_columns": ["Site", "Site"]},
            "NativeTuning.winner.metadata_columns contains duplicate column names",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            nirs4all.NativeTuning(space={"alpha": [0.2]}, winner=winner).to_dict()


def test_public_run_accepts_typed_native_tuning_config() -> None:
    result = nirs4all.run(
        pipeline=[{"model": _PublicTuningEstimator(alpha=1.0)}],
        dataset=(
            np.asarray([[1.0], [2.0]]),
            np.asarray([1.0, 2.0]),
            ["train-a", "train-b"],
        ),
        engine="dag-ml",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
            ),
            winner=nirs4all.TuningWinner(
                X=np.asarray([[30.0], [40.0]]),
                y_true=np.asarray([0.0, 0.0]),
                score=0.2,
                metric="rmse",
                sample_ids=["test-a", "test-b"],
                model_name="TypedRunWinner",
            ),
        ),
        verbose=0,
    )

    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.best["model_name"] == "TypedRunWinner"
    assert result.best_score == pytest.approx(0.2)


def test_public_run_calibration_uses_tuning_winner_not_score_data() -> None:
    result = nirs4all.run(
        pipeline=[{"model": _PublicTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
            ),
            winner=nirs4all.TuningWinner(
                X=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
                y_true=np.asarray([10.0, 10.0, 10.0, 10.0]),
                score=9.8,
                metric="rmse",
                sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
                model_name="TypedRunWinner",
            ),
        ),
        calibration=nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0, 2.0]),
            prediction_sample_ids=["pred-a", "pred-b"],
            coverage=0.8,
            extra={"result_metadata": {"operator": "unit-test"}},
        ),
        verbose=0,
    )

    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.run.tuning_best_params == {"alpha": 0.2}
    assert result.run.best["model_name"] == "TypedRunWinner"
    np.testing.assert_allclose(result.interval(0.8).lower, [-8.8, -7.8])
    np.testing.assert_allclose(result.interval(0.8).upper, [10.8, 11.8])
    assert result.calibrated.metadata["operator"] == "unit-test"
    assert result.calibrated.tuning_calibration_source == {
        "source": "tuning.winner",
        "score_data_role": "hpo_objective_only",
        "score_data_used": False,
    }


def test_public_run_accepts_typed_conformal_score_calibration() -> None:
    result = nirs4all.run(
        pipeline=[{"model": _PublicTuningEstimator(alpha=1.0)}],
        dataset=(np.asarray([[1.0], [2.0]]), np.asarray([1.0, 2.0])),
        engine="dag-ml",
        tuning=nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="conformal_mean_width",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
                conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                    X=np.asarray([[3.0], [4.0], [5.0], [6.0]]),
                    y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
                ),
                conformal_coverage=0.8,
            ),
        ),
        verbose=0,
    )

    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.4)
    assert result.tuning_result.trials[0].diagnostics["score_family"] == "conformal"


def test_public_tune_single_estimator_accepts_typed_native_tuning_core() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
        ),
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
    )

    assert result.tuning_best_params == {"alpha": 0.2}


def test_public_tune_single_estimator_accepts_direct_sklearn_string_model() -> None:
    result = nirs4all.tune_single_estimator(
        "sklearn.dummy.DummyRegressor",
        np.asarray([[1.0], [2.0]]),
        np.asarray([10.0, 20.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"strategy": ["mean"]},
            sampler="grid",
            n_trials=1,
            metric="rmse",
            direction="minimize",
        ),
        X_score=np.asarray([[3.0], [4.0]]),
        y_score=np.asarray([15.0, 15.0]),
    )

    assert result.tuning_best_params == {"strategy": "mean"}
    assert result.tuning_best_value == pytest.approx(0.0)


@pytest.mark.parametrize("wrapper_key", ["steps", "pipeline"])
def test_public_tune_single_estimator_accepts_public_linear_wrappers(wrapper_key: str) -> None:
    result = nirs4all.tune_single_estimator(
        {
            "name": "wrapped-linear",
            wrapper_key: [{"name": "ridge", "model": _PublicTuningEstimator(alpha=1.0)}],
        },
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"ridge.alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
        ),
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
    )

    assert result.tuning_best_params == {"ridge.alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)


def test_public_tune_single_estimator_uses_native_tuning_score_data() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
            ),
        ),
    )

    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)


def test_public_tune_single_estimator_accepts_json_array_score_data() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=[
                np.asarray([[10.0], [20.0]]),
                np.asarray([0.0, 0.0]),
                ["score-a", "score-b"],
            ],
        ),
    )

    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)


def test_public_tune_single_estimator_uses_native_tuning_conformal_score_data() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="conformal_mean_width",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
                conformal_calibration=nirs4all.TuningConformalScoreCalibration(
                    X=np.asarray([[3.0], [4.0], [5.0], [6.0]]),
                    y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
                ),
                conformal_coverage=0.8,
            ),
        ),
    )

    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.4)
    assert result.tuning_result.trials[0].diagnostics["score_family"] == "conformal"
    assert result.tuning_result.trials[0].diagnostics["score_extractor"] == "conformal_temporary_calibration"


def test_public_tuning_calibration_rejects_manual_calibration_data() -> None:
    with pytest.raises(ValueError, match="TuningCalibration.extra"):
        nirs4all.TuningCalibration(
            y_pred=[1.0],
            prediction_sample_ids=["pred-a"],
            extra={"calibration_data": {"y_true": [1.0]}},
        ).to_dict()


def test_public_tune_single_estimator_runs_native_lane_and_exports_result(tmp_path) -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        sample_ids=["train-a", "train-b"],
        workspace_path=tmp_path / "workspace",
        workspace_tuning_id="public-single-estimator-tune",
        winner_x=np.asarray([[30.0], [40.0]]),
        winner_y_true=np.asarray([0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["test-a", "test-b"],
        winner_dataset_name="external",
        winner_model_name="PublicWinner",
    )

    restored = nirs4all.load_workspace_tuning_result(tmp_path / "workspace", "public-single-estimator-tune")

    assert result.tuning_id == "public-single-estimator-tune"
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(0.2)
    assert result.best["model_name"] == "PublicWinner"
    assert result.best_score == pytest.approx(0.2)
    assert restored.to_dict() == result.tuning_result.to_dict()


def test_public_tune_single_estimator_score_data_transports_identities() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicIdentityAwareTuningEstimator(alpha=0.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.0, 5.0]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([5.0, 5.0]),
        score_sample_ids=["score-a", "score-b"],
        score_groups=["batch-1", "batch-2"],
        score_metadata=[{"site": "north"}, {"site": "south"}],
    )

    assert result.tuning_best_params == {"alpha": 5.0}


def test_public_tune_single_estimator_resume_ignores_resume_bit_in_contract() -> None:
    first = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
    )

    resumed = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
            "resume": True,
        },
        # This score cohort would prefer alpha=0.9 if the optimizer re-ran.
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([1.0, 1.0]),
        resume_tuning_result=first.tuning_result,
    )

    assert first.tuning_best_params == {"alpha": 0.2}
    assert resumed.tuning_best_params == {"alpha": 0.2}
    assert resumed.tuning_result.to_dict() == first.tuning_result.to_dict()


def test_public_tune_single_estimator_result_best_can_feed_conformal_calibration() -> None:
    tuned = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        winner_x=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
        winner_y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        winner_dataset_name="calibration",
        winner_model_name="PublicWinner",
    )

    calibrated = nirs4all.calibrate(
        calibration_data=tuned.best,
        y_pred=np.asarray([1.0, 2.0]),
        prediction_sample_ids=["pred-a", "pred-b"],
        coverage=0.8,
        as_predict_result=True,
    )

    assert calibrated.conformal_guarantee_status is not None
    assert calibrated.conformal_guarantee_status["status"] == "active"
    assert calibrated.conformal_guarantee_status["unit"] == "physical_sample"
    np.testing.assert_allclose(calibrated.interval(0.8).lower, [0.8, 1.8])
    np.testing.assert_allclose(calibrated.interval(0.8).upper, [1.2, 2.2])


def test_public_tune_single_estimator_can_return_integrated_conformal_result() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        winner_x=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
        winner_y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        calibration={
            "y_pred": np.asarray([1.0, 2.0]),
            "prediction_sample_ids": ["pred-a", "pred-b"],
            "coverage": 0.8,
        },
    )

    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.tuning_result is result.run.tuning_result
    assert result.tuning_best_params == result.run.tuning_best_params == {"alpha": 0.2}
    assert result.tuning_best_value == pytest.approx(result.run.tuning_best_value)
    assert result.run.tuning_best_params == {"alpha": 0.2}
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    assert result.interval_coverages == (0.8,)
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.calibrated.interval(0.8).lower, [0.8, 1.8])
    np.testing.assert_allclose(result.interval(0.8).lower, [0.8, 1.8])
    assert result.metrics([1.0, 2.0])[0.8].observed_coverage == pytest.approx(1.0)
    report = result.robustness(y_true=[1.0, 2.0])
    assert report.mode == "clean_frozen"
    assert report.scenarios[0].metrics.n_samples == 2


def test_public_tune_single_estimator_final_calibration_uses_projected_winner_not_score_data() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        # The HPO scoring cohort selects alpha=0.2 and would produce a narrow
        # residual of 0.2 if it leaked into final calibration.
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        # The final conformal cohort is intentionally different: the refit
        # winner predicts 0.2, so residuals are 9.8 and intervals must be wide.
        winner_x=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
        winner_y_true=np.asarray([10.0, 10.0, 10.0, 10.0]),
        winner_score=9.8,
        winner_metric="rmse",
        winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        calibration={
            "y_pred": np.asarray([1.0, 2.0]),
            "prediction_sample_ids": ["pred-a", "pred-b"],
            "coverage": 0.8,
        },
    )

    assert result.run.tuning_best_params == {"alpha": 0.2}
    np.testing.assert_allclose(result.interval(0.8).lower, [-8.8, -7.8])
    np.testing.assert_allclose(result.interval(0.8).upper, [10.8, 11.8])
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    assert result.calibrated.tuning_calibration_source == {
        "source": "tuning.winner",
        "score_data_role": "hpo_objective_only",
        "score_data_used": False,
    }


def test_public_tune_single_estimator_can_proxy_rich_calibrated_result() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        {
            "engine": "optuna",
            "space": {"alpha": [0.9, 0.2]},
            "sampler": "grid",
            "n_trials": 2,
            "metric": "rmse",
            "direction": "minimize",
        },
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        winner_x=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
        winner_y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        calibration=nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0, 2.0]),
            prediction_sample_ids=["pred-a", "pred-b"],
            coverage=0.8,
            as_predict_result=False,
        ),
    )

    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert isinstance(result.calibrated, nirs4all.CalibratedRunResult)
    assert result.interval_coverages == (0.8,)
    assert result.conformal_guarantee_status is not None
    assert result.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.interval(0.8).upper, [1.2, 2.2])
    assert result.metrics([1.0, 2.0])[0.8].n_samples == 2
    report = result.robustness(y_true=[1.0, 2.0])
    assert report.mode == "clean_frozen"
    assert report.scenarios[0].metrics.n_samples == 2


def test_public_tune_single_estimator_accepts_typed_calibration_helper() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
        ),
        X_score=np.asarray([[10.0], [20.0]]),
        y_score=np.asarray([0.0, 0.0]),
        winner_x=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
        winner_y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
        winner_score=0.2,
        winner_metric="rmse",
        winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        calibration=nirs4all.TuningCalibration(
            y_pred=np.asarray([1.0, 2.0]),
            prediction_sample_ids=["pred-a", "pred-b"],
            coverage=0.8,
        ),
    )

    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.tuning_best_params == {"alpha": 0.2}
    assert result.calibrated.conformal_guarantee_status is not None
    assert result.calibrated.conformal_guarantee_status["status"] == "active"
    np.testing.assert_allclose(result.calibrated.interval(0.8).upper, [1.2, 2.2])


def test_public_tune_single_estimator_accepts_typed_winner_and_calibration_from_tuning() -> None:
    result = nirs4all.tune_single_estimator(
        [{"model": _PublicTuningEstimator(alpha=1.0)}],
        np.asarray([[1.0], [2.0]]),
        np.asarray([1.0, 2.0]),
        nirs4all.NativeTuning(
            engine="optuna",
            space={"alpha": [0.9, 0.2]},
            sampler="grid",
            n_trials=2,
            metric="rmse",
            direction="minimize",
            score_data=nirs4all.TuningScoreData(
                X=np.asarray([[10.0], [20.0]]),
                y=np.asarray([0.0, 0.0]),
            ),
            winner=nirs4all.TuningWinner(
                X=np.asarray([[30.0], [40.0], [50.0], [60.0]]),
                y_true=np.asarray([0.0, 0.0, 0.0, 0.0]),
                score=0.2,
                metric="rmse",
                physical_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
                model_name="TypedWinner",
            ),
            calibration=nirs4all.TuningCalibration(
                y_pred=np.asarray([1.0, 2.0]),
                prediction_sample_ids=["pred-a", "pred-b"],
                coverage=0.8,
            ),
        ),
    )

    assert isinstance(result, nirs4all.TunedSingleEstimatorConformalResult)
    assert result.run.best["model_name"] == "TypedWinner"
    assert result.run.best["metadata"]["physical_sample_id"] == ["cal-a", "cal-b", "cal-c", "cal-d"]
    assert result.tuning_best_params == {"alpha": 0.2}
    np.testing.assert_allclose(result.calibrated.interval(0.8).lower, [0.8, 1.8])


def test_public_tune_single_estimator_calibration_requires_projected_winner() -> None:
    with pytest.raises(ValueError, match="projected winner"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0], [2.0]],
            [1.0, 2.0],
            {"engine": "optuna", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
            X_score=[[10.0], [20.0]],
            y_score=[0.0, 0.0],
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"]},
        )

    with pytest.raises(ValueError, match="must not include calibration_data"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0], [2.0]],
            [1.0, 2.0],
            {"engine": "optuna", "space": {"alpha": [0.9, 0.2]}, "sampler": "grid", "n_trials": 2},
            X_score=[[10.0], [20.0]],
            y_score=[0.0, 0.0],
            winner_x=[[30.0], [40.0], [50.0], [60.0]],
            winner_y_true=[0.0, 0.0, 0.0, 0.0],
            winner_score=0.2,
            winner_metric="rmse",
            winner_sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
            calibration={"calibration_data": {"y_true": [0.0]}, "y_pred": [1.0], "prediction_sample_ids": ["pred-a"]},
        )


def test_public_tune_single_estimator_requires_unambiguous_scoring() -> None:
    with pytest.raises(ValueError, match="requires score_extractor"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            {"engine": "optuna", "space": {"alpha": [0.2]}, "sampler": "grid", "n_trials": 1},
        )

    with pytest.raises(ValueError, match="exactly one"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            {"engine": "optuna", "space": {"alpha": [0.2]}, "sampler": "grid", "n_trials": 1},
            score_extractor=lambda _fitted: 0.0,
            X_score=[[1.0]],
            y_score=[1.0],
        )

    with pytest.raises(ValueError, match="exactly one"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            {"engine": "optuna", "space": {"alpha": [0.2]}, "sampler": "grid", "n_trials": 1},
            score_extractor=lambda _fitted: 0.0,
            score_sample_ids=["unused"],
        )

    with pytest.raises(ValueError, match="tuning.score_data or X_score"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            nirs4all.NativeTuning(
                engine="optuna",
                space={"alpha": [0.2]},
                sampler="grid",
                n_trials=1,
                score_data=nirs4all.TuningScoreData(X=[[1.0]], y=[1.0]),
            ),
            X_score=[[1.0]],
            y_score=[1.0],
        )

    with pytest.raises(ValueError, match="dataset-backed score_data"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            nirs4all.NativeTuning(
                engine="optuna",
                space={"alpha": [0.2]},
                sampler="grid",
                n_trials=1,
                score_data=nirs4all.TuningScoreData(
                    dataset="dataset.json",
                    selector={"partition": "score"},
                ),
            ),
        )


def test_public_tune_single_estimator_rejects_ambiguous_tuning_winner_and_calibration() -> None:
    tuning = nirs4all.NativeTuning(
        engine="optuna",
        space={"alpha": [0.2]},
        sampler="grid",
        n_trials=1,
        score_data=nirs4all.TuningScoreData(X=[[10.0]], y=[0.0]),
        winner=nirs4all.TuningWinner(
            X=[[30.0], [40.0], [50.0], [60.0]],
            y_true=[0.0, 0.0, 0.0, 0.0],
            score=0.2,
            metric="rmse",
            sample_ids=["cal-a", "cal-b", "cal-c", "cal-d"],
        ),
        calibration=nirs4all.TuningCalibration(
            y_pred=[1.0],
            prediction_sample_ids=["pred-a"],
        ),
    )

    with pytest.raises(ValueError, match=r"tuning\.winner or winner_\* arguments"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            tuning,
            winner_x=[[30.0]],
        )

    with pytest.raises(ValueError, match=r"either as calibration=\.\.\. or tuning\.calibration"):
        nirs4all.tune_single_estimator(
            _PublicTuningEstimator(),
            [[1.0]],
            [1.0],
            tuning,
            calibration={"y_pred": [1.0], "prediction_sample_ids": ["pred-a"]},
        )

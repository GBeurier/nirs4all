"""Real Methods fold-local HPO, independent OOF oracle and installed RAW replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import dag_ml
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold

import nirs4all
from nirs4all.pipeline.dagml.training_contracts import tcv1_sha256
from tests.integration.api.test_native_pls_phase_controls import (
    _REPLAY as _PHASE_REPLAY,
)
from tests.integration.api.test_native_pls_phase_controls import (
    _dataset,
    _inspect,
    _oof,
    _run,
)
from tests.integration.api.test_native_pls_phase_controls import (
    _pipeline as _phase_pipeline,
)

_REQUIRE_ENV = "NIRS4ALL_REQUIRE_NATIVE_PLS_FOLD_HPO"
_LIBRARY_ENV = "NIRS4ALL_CORE_LIVE_METHODS_LIBRARY"
pytestmark = [
    pytest.mark.methods,
    pytest.mark.skipif(os.environ.get(_REQUIRE_ENV) != "1", reason=f"set {_REQUIRE_ENV}=1 with matching installed native wheels"),
]


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> str:
    selected = os.environ.get(_LIBRARY_ENV)
    assert selected and Path(selected).is_file(), f"{_LIBRARY_ENV} must select the exact Methods library"
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    return str(Path(selected).resolve())


def _pipeline(*, trials: int = 4, scope: str | None = "fold", splitter: Any = None,
              resume: Any = None, refit: Any = None, train: Any = None,
              space: Any = None, smooth: bool = False) -> list[Any]:
    result = _phase_pipeline(trials=trials, resume=resume, refit=refit, train=train, space=space, smooth=smooth)
    if scope is not None:
        result[-1]["finetune_params"]["scope"] = scope
    if splitter is not None:
        result[0] = splitter
    return result


def _scopes(result: Any) -> list[dict[str, Any]]:
    state = result._native_outcome["methods_hpo_fold_state"]
    public_state = dag_ml.methods_hpo_fold_state_from_package(result.tuning_resume_package)
    assert isinstance(public_state, dag_ml.MethodsFoldHpoState)
    assert public_state.to_dict() == state
    snapshot = public_state.to_dict()
    snapshot["outer_scopes"].clear()
    assert public_state.to_dict() == state
    assert state == result._native_package["execution_bundle"]["methods_hpo_fold_state"]
    assert not result._native_outcome.get("methods_hpo_resume_state")
    assert not result._native_package["execution_bundle"].get("methods_hpo_resume_state")
    summary = result.tuning_scope_results
    assert summary == {"outer_scopes": state["outer_scopes"], "refit_scope": state["refit_scope"]}
    return [*summary["outer_scopes"], summary["refit_scope"]]


def _resume_state_semantics(original: dict[str, Any], library: str) -> str:
    """Validate native checkpoints while excluding elapsed time from equality."""
    from n4m._ffi import lib
    from n4m.model_selection import Optimizer

    assert Path(lib._name).resolve() == Path(library).resolve()
    state = deepcopy(original)
    checkpoint = state["checkpoint"]
    payload = bytes(checkpoint.pop("opaque_payload"))
    assert hashlib.sha256(payload).hexdigest() == checkpoint.pop("payload_sha256")
    optimizer = Optimizer.load(payload)
    try:
        records = []
        for record, terminal in zip(optimizer.get_trials(), state["terminal_trials"], strict=True):
            native = asdict(record)
            duration = native.pop("duration")
            reported_duration = terminal["trial"].pop("duration")
            assert type(duration) is float and math.isfinite(duration) and duration >= 0
            assert type(reported_duration) is float and reported_duration == duration
            native["parameter_order"] = list(record.params)
            records.append(native)
        best = optimizer.best()
        checkpoint["decoded_state"] = {
            "trials": records,
            "best": None if best is None else {"trial_id": best[0].id, "score": best[1]},
        }
    finally:
        optimizer.close()
    # JSON equality also distinguishes boolean, integer and float values and
    # preserves the native parameter, trial and intermediate report orders.
    return json.dumps(state, allow_nan=False)


def _params(trial: dict[str, Any]) -> dict[str, Any]:
    parameters = trial["parameters"]
    scale = parameters["scale"]
    assert scale["native_kind"] == "categorical" and scale["category_type"] == "boolean"
    assert type(scale["category_index"]) is int and scale["category_index"] in (0, 1)
    assert parameters["n_components"]["integer"] is True
    return {"n_components": int(parameters["n_components"]["value"]), "scale": scale["category_index"] == 1}


def _splits(splitter: Any, dataset: dict[str, Any], indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = dataset.get("groups")
    kwargs = {"groups": np.asarray(groups)[indices]} if groups is not None else {}
    return [(indices[train], indices[validation]) for train, validation in splitter.split(dataset["X"][indices], dataset["y"][indices], **kwargs)]


def _assert_memberships(scopes: list[dict[str, Any]], dataset: dict[str, Any], splitter: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    universe = np.arange(len(dataset["sample_ids"]))
    outer = _splits(splitter, dataset, universe)
    assert len(scopes) == len(outer) + 1
    global_ids: set[str] = set()
    for index, scope in enumerate(scopes):
        refit = index == len(outer)
        pool = universe if refit else outer[index][0]
        assert scope["phase"] == ("REFIT" if refit else "FIT_CV")
        assert scope["outer_fold_id"] == (None if refit else f"fold{index}")
        assert scope["scope_id"].endswith(":scope:refit" if refit else f":scope:fit_cv:fold{index}")
        inner = scope["inner_fold_set"]
        assert inner["sample_ids"] == [dataset["sample_ids"][position] for position in pool]
        if "groups" in dataset:
            assert inner["sample_groups"] == {dataset["sample_ids"][position]: str(dataset["groups"][position]) for position in pool}
        expected = _splits(splitter, dataset, pool)
        assert len(inner["folds"]) == len(expected)
        for fold, (train, validation) in zip(inner["folds"], expected, strict=True):
            assert fold["fold_id"] not in global_ids
            global_ids.add(fold["fold_id"])
            assert fold["train_sample_ids"] == [dataset["sample_ids"][position] for position in train]
            assert fold["validation_sample_ids"] == [dataset["sample_ids"][position] for position in validation]
            assert set(fold["train_sample_ids"]).isdisjoint(fold["validation_sample_ids"])
            if not refit:
                validation_ids = {dataset["sample_ids"][position] for position in outer[index][1]}
                assert validation_ids.isdisjoint(fold["train_sample_ids"] + fold["validation_sample_ids"])
            if "groups" in dataset:
                groups = np.asarray(dataset["groups"])
                assert set(groups[train]).isdisjoint(groups[validation])
    return outer


_ORACLE = r'''
import json, pathlib, sys
import numpy as np
import n4m
from n4m._ffi import lib
from n4m.roles import RolePipeline
data = json.loads(sys.stdin.read())
assert pathlib.Path(lib._name).resolve() == pathlib.Path(sys.argv[1]).resolve()
# The canonical pls4all 1.3.2 wheel must carry native Methods 1.3.2 / ABI 2.17.
assert n4m.version() in {"1.2.1+abi.2.14.0", "1.2.1+abi.2.15.0", "1.2.1+abi.2.16.0", "1.2.1+abi.2.17.0", "1.3.2+abi.2.17.0"}, "this gate qualifies the selected public ABI 2.14/2.15/2.16/2.17 runtime"
X, y = np.asarray(data["X"]), np.asarray(data["y"])
row_of = {identifier: row for row, identifier in enumerate(data["sample_ids"])}
def positions(ids): return [row_of[identifier] for identifier in ids]
def model(params):
    steps = []
    if data["smooth"]:
        steps += [("preprocessing.scatter.snv", {"with_mean": True, "with_std": True, "ddof": 0}),
                  ("preprocessing.derivatives.savitzky_golay", {"window_length": 5, "polyorder": 2, "deriv": 0, "delta": 1.0, "mode": "interp", "cval": 0.0})]
    return RolePipeline(steps + [("models.pls.pls_regression", {
        "n_components": params["n_components"], "solver": "nipals", "center_x": True, "center_y": True,
        "scale_x": params["scale"], "scale_y": params["scale"],
    })])
scores = []
oof = np.empty_like(y)
for index, scope in enumerate(data["scopes"]):
    trial_scores = []
    for params in scope["trial_params"]:
        predictions, targets = [], []
        for fold in scope["inner_fold_set"]["folds"]:
            train, validation = positions(fold["train_sample_ids"]), positions(fold["validation_sample_ids"])
            fitted = model(params).fit(X[train], y[train])
            predictions.extend(np.asarray(fitted.predict(X[validation])).reshape(-1).tolist())
            targets.extend(y[validation].tolist())
            del fitted
        trial_scores.append(float(np.sqrt(np.mean((np.asarray(predictions) - targets)**2))))
    scores.append(trial_scores)
    if index < len(data["outer"]):
        train, validation = data["outer"][index]
        fitted = model(scope["winner_params"]).fit(X[train], y[train])
        oof[validation] = np.asarray(fitted.predict(X[validation])).reshape(-1)
        del fitted
fitted = model(data["saved_params"]).fit(X, y)
heldout = np.asarray(fitted.predict(np.asarray(data["heldout"]))).reshape(-1, 1)
print(json.dumps({"scores": scores, "oof": oof.tolist(), "heldout": heldout.tolist(), "library": str(pathlib.Path(lib._name).resolve())}))
'''


def _oracle(dataset: dict[str, Any], library: str, scopes: list[dict[str, Any]], outer: Any,
            heldout: np.ndarray, saved_params: dict[str, Any], tmp_path: Path, *, smooth: bool = False) -> dict[str, Any]:
    payload = {
        "X": dataset["X"].tolist(), "y": dataset["y"].tolist(), "sample_ids": dataset["sample_ids"],
        "outer": [[train.tolist(), validation.tolist()] for train, validation in outer],
        "scopes": [{"inner_fold_set": scope["inner_fold_set"], "winner_params": scope["winner_params"],
                    "trial_params": [_params(entry["trial"]) for entry in scope["resume_state"]["terminal_trials"]]} for scope in scopes],
        "heldout": heldout.tolist(), "saved_params": saved_params, "smooth": smooth,
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _ORACLE, library], input=json.dumps(payload), cwd=tmp_path,
        env={**os.environ, "N4M_LIB_PATH": library}, capture_output=True, text=True, check=True, timeout=120,
    )
    result = json.loads(completed.stdout)
    assert result["library"] == library
    return result


def _fresh_fold_predict(archive: Path, library: str, heldout: np.ndarray, tmp_path: Path) -> dict[str, Any]:
    # Reuse the phase gate's FIT/HPO-forbidden replay script, while binding
    # the new fold planner as well as its ten pre-existing production files.
    root = Path(nirs4all.__file__).parent
    files = [
        "__init__.py", "api/__init__.py", "api/run.py", "api/portable_archive.py", "api/native_archive_training.py",
        "pipeline/dagml/native_pls_phase_controls.py", "pipeline/dagml/native_pls_phase_replay.py", "pipeline/dagml/native_pls_fold_hpo.py",
        "pipeline/dagml/core_archive_replay.py", "pipeline/dagml/raw_training_lowerer.py", "pipeline/dagml/raw_replay_lowerer.py",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    data = {"X": heldout.tolist(), "sample_ids": [f"heldout.fold.{index}" for index in range(len(heldout))], "source_hashes": hashes}
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["N4M_LIB_PATH"] = library
    executable = os.environ.get("NIRS4ALL_NATIVE_PLS_INSTALLED_PYTHON", sys.executable)
    assert Path(executable).is_file(), "fresh replay must use the installed-wheel qualification Python"
    completed = subprocess.run(
        [executable, "-I", "-B", "-c", _PHASE_REPLAY, str(archive), library], input=json.dumps(data), cwd=tmp_path,
        env=environment, check=True, capture_output=True, text=True, timeout=60,
    )
    result = json.loads(completed.stdout)
    assert result["sample_ids"] == data["sample_ids"] and result["target_names"] == ["response"]
    assert result["source_hashes"] == hashes
    return result


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_exact_inner_folds_scores_outer_oof_and_saved_refit_match_methods_oracle(library: str, tmp_path: Path, grouped: bool, smooth: bool) -> None:
    dataset = _dataset()
    splitter: Any = KFold(3, shuffle=True, random_state=17)
    if grouped:
        dataset["groups"] = np.repeat([1, 2, 10, 11, 12, 20, 21], [2, 3, 4, 5, 6, 7, 9])
        splitter = GroupKFold(3)
    overrides = {"n_components": 2, "scale": False}
    heldout = dataset["X"][[2, 15, 31]] * 0.89 + 0.003
    with _run(_pipeline(splitter=splitter, refit=overrides, smooth=smooth), dataset, library) as result:
        scopes = _scopes(result)
        outer = _assert_memberships(scopes, dataset, splitter)
        if grouped:
            assert len({len(validation) for _, validation in outer}) > 1
        oracle = _oracle(dataset, library, scopes, outer, heldout, overrides, tmp_path, smooth=smooth)
        variants: set[str] = set()
        for scope, scores in zip(scopes, oracle["scores"], strict=True):
            assert type(scope["winner_params"]["scale"]) is bool and type(scope["winner_params"]["n_components"]) is int
            state = scope["resume_state"]
            assert state["trial_history_len"] == len(state["terminal_trials"]) == len(state["completed_reports"]) == 4
            reports = {report["variant_id"]: report for report in state["completed_reports"]}
            for terminal, score in zip(state["terminal_trials"], scores, strict=True):
                assert terminal["trial"]["status"] == "completed"
                variant = terminal["variant_id"]
                assert variant not in variants
                variants.add(variant)
                assert reports[variant]["score"] == pytest.approx(score, abs=1e-10)
            assert reports[scope["winner_variant_id"]]["score"] == pytest.approx(min(scores), abs=1e-10)
        np.testing.assert_allclose(_oof(result, dataset), oracle["oof"], atol=1e-10, rtol=1e-10)
        assert result.cv_best_score == pytest.approx(np.sqrt(np.mean((np.asarray(oracle["oof"]) - dataset["y"])**2)), abs=1e-10)
        refit_scope = scopes[-1]
        assert result.tuning_best_params == {"model." + name: value for name, value in refit_scope["winner_params"].items()}
        assert result.tuning_best_value == pytest.approx(min(oracle["scores"][-1]), abs=1e-10)
        archive = Path(result.export(tmp_path / "fold-winner.n4a"))
        _inspect(archive, library, overrides, steps=3 if smooth else 1)
        assert result._native_outcome["selected_variant_id"] == refit_scope["winner_variant_id"]
        lineage = result._native_outcome["lineage"]
        for scope in scopes:
            records = [record for record in lineage if record["node_id"] == result._native_outcome["methods_hpo_fold_state"]["target_node_id"]
                       and record["phase"] == ("REFIT" if scope["phase"] == "REFIT" else "FIT_CV") and record["fold_id"] == scope["outer_fold_id"]]
            assert records and all(record["params_fingerprint"] == scope["params_fingerprint"] for record in records)
    assert result.native_execution_is_live is False
    del dataset
    detached = _fresh_fold_predict(archive, library, heldout, tmp_path)
    np.testing.assert_allclose(detached["predictions"], oracle["heldout"], atol=1e-10, rtol=1e-10)


def test_outer_validation_targets_never_select_its_local_winner(library: str) -> None:
    original = _dataset()
    changed = _dataset()
    _, outer_validation = next(KFold(3, shuffle=True, random_state=17).split(original["X"]))
    changed["y"][outer_validation] += np.linspace(90, 180, len(outer_validation))
    with _run(_pipeline(), original, library) as first:
        first_scope = _scopes(first)[0]
        first_predictions = _oof(first, original)[outer_validation]
        first_score = first.cv_best_score
    with _run(_pipeline(), changed, library) as second:
        second_scope = _scopes(second)[0]
        assert first_scope["winner_params"] == second_scope["winner_params"]
        assert first_scope["inner_fold_set"] == second_scope["inner_fold_set"]
        first_trials = first_scope["resume_state"]["terminal_trials"]
        second_trials = second_scope["resume_state"]["terminal_trials"]
        assert [_params(entry["trial"]) for entry in first_trials] == [_params(entry["trial"]) for entry in second_trials]
        assert [report["score"] for report in first_scope["resume_state"]["completed_reports"]] == [report["score"] for report in second_scope["resume_state"]["completed_reports"]]
        np.testing.assert_array_equal(first_predictions, _oof(second, changed)[outer_validation])
        assert second.cv_best_score > first_score + 10


def test_outer_studies_can_select_different_effective_model_parameters(library: str) -> None:
    dataset = _dataset()
    # One noisy contiguous calibration regime is absent from fold0's train
    # pool and present in the other pools; inner validation evaluates that
    # difference instead of forcing one campaign-wide winner onto all folds.
    dataset["y"][:12] += 12 * np.random.default_rng(411).normal(size=12)
    with _run(_pipeline(trials=16, splitter=KFold(3)), dataset, library) as result:
        scopes = _scopes(result)
        winners = {(scope["winner_params"]["n_components"], scope["winner_params"]["scale"]) for scope in scopes[:-1]}
        assert len(winners) > 1, "the unequal-regime witness must exercise different outer-fold winners"
        assert all(scope["resume_state"]["trial_history_len"] == 16 for scope in scopes)
        assert np.all(np.isfinite(_oof(result, dataset)))


def test_every_scope_budget_extension_matches_uninterrupted_search(library: str) -> None:
    dataset = _dataset()
    with _run(_pipeline(trials=2), dataset, library) as initial:
        resume = initial.tuning_resume_package
        assert resume is not None
        assert all(scope["resume_state"]["trial_history_len"] == 2 for scope in _scopes(initial))
    with _run(_pipeline(resume=resume), dataset, library) as continued, _run(_pipeline(), dataset, library) as full:
        resumed_scopes, full_scopes = _scopes(continued), _scopes(full)
        for resumed, reference in zip(resumed_scopes, full_scopes, strict=True):
            assert resumed["scope_id"] == reference["scope_id"]
            assert resumed["winner_params"] == reference["winner_params"]
            assert resumed["resume_state"]["trial_history_len"] == 4
            assert _resume_state_semantics(resumed["resume_state"], library) == _resume_state_semantics(reference["resume_state"], library)
        assert continued.tuning_best_params == full.tuning_best_params
        assert continued.tuning_best_value == full.tuning_best_value
        np.testing.assert_array_equal(_oof(continued, dataset), _oof(full, dataset))


@pytest.mark.parametrize("change", ["data", "targets", "folds", "train", "refit", "scope_swap"])
def test_complete_checkpoint_rejects_changed_or_swapped_scope_authority(library: str, change: str) -> None:
    import dag_ml

    dataset = _dataset()
    options: dict[str, Any] = {"space": {"scale": [False, True]}, "train": {"n_components": 2}, "refit": {"scale": False}}
    with _run(_pipeline(trials=2, **options), dataset, library) as initial:
        resume = initial.tuning_resume_package
        assert resume is not None
    if change == "data":
        dataset["X"][0, 0] += 0.1
    elif change == "targets":
        dataset["y"][0] += 0.1
    elif change == "folds":
        options["splitter"] = KFold(3, shuffle=True, random_state=18)
    elif change == "train":
        options["train"] = {"n_components": 3}
    elif change == "refit":
        options["refit"] = {"scale": True}
    else:
        scopes = resume["execution_bundle"]["methods_hpo_fold_state"]["outer_scopes"]
        scopes[0]["resume_state"], scopes[1]["resume_state"] = scopes[1]["resume_state"], scopes[0]["resume_state"]
        # Re-sign only the containing package: scope provenance must still refuse it.
        resume["package_fingerprint"] = tcv1_sha256({key: value for key, value in resume.items() if key != "package_fingerprint"})
    with pytest.raises((ValueError, RuntimeError, dag_ml.DagMlRuntimeError), match="resume|checkpoint|scope|provenance"):
        _run(_pipeline(resume=resume, **options), dataset, library)


@pytest.mark.parametrize("change", ["missing", "partial", "kfold_with_groups"])
def test_group_grain_is_never_silently_discarded(library: str, change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import dag_ml

    dataset = _dataset()
    splitter: Any = GroupKFold(3)
    if change != "missing":
        dataset["groups"] = np.repeat(np.arange(7), [2, 3, 4, 5, 6, 7, 9])
    if change == "partial":
        dataset["groups"] = dataset["groups"][:-1]
    elif change == "kfold_with_groups":
        splitter = KFold(3)
    monkeypatch.setattr(dag_ml, "execute_methods_training", lambda *_args, **_kwargs: pytest.fail("invalid group contract reached FIT/HPO"))
    with pytest.raises((ValueError, TypeError), match="group|GroupKFold"):
        _run(_pipeline(splitter=splitter), dataset, library)


def test_default_campaign_descriptor_history_and_predictions_remain_unchanged(library: str) -> None:
    dataset = _dataset()
    with _run(_pipeline(scope=None), dataset, library) as default, _run(_pipeline(scope="campaign"), dataset, library) as explicit:
        assert not default._native_outcome.get("methods_hpo_fold_state")
        assert not explicit._native_outcome.get("methods_hpo_fold_state")
        assert _resume_state_semantics(default._native_outcome["methods_hpo_resume_state"], library) == _resume_state_semantics(explicit._native_outcome["methods_hpo_resume_state"], library)
        operation = default._native_outcome["effective_plan"]["campaign"]["metadata"]["methods_hpo_operation"]
        assert operation["schema_version"] == 2 and "inner_fold_sets" not in operation
        assert default.tuning_best_params == explicit.tuning_best_params
        assert default.tuning_best_value == explicit.tuning_best_value
        np.testing.assert_array_equal(_oof(default, dataset), _oof(explicit, dataset))

"""Public fold-HPO boundaries remain independent of optional native runtimes."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, ShuffleSplit

import nirs4all
from nirs4all.api.native_archive_training import NativeArchiveTrainingError, NativeMethodsArchiveRunResult
from nirs4all.pipeline.dagml.native_pls_phase_controls import normalize_native_pls_phase_controls
from tests.unit.api.test_native_pls_phase_controls import _forbid_work

PROFILE = "n4m.pls_role_pipeline.v1"


def _pipeline(scope: Any = "fold", **options: Any) -> list[Any]:
    return [KFold(3, shuffle=True, random_state=17), {
        "model": PLSRegression(n_components=1),
        "finetune_params": {
            "engine": "n4m", "approach": "grouped", "scope": scope,
            "sampler": "random", "pruner": "none", "n_trials": 4, "seed": 6,
            "metric": "rmse", "direction": "minimize",
            "model_params": {"n_components": ["int", 1, 3], "scale": [False, True]},
            **options,
        },
    }]


@pytest.mark.parametrize("scope", [None, True, 1, "", "Fold", "outer", "nested", []])
def test_unknown_scope_fails_before_data_fit_or_optimizer(monkeypatch: pytest.MonkeyPatch, scope: Any) -> None:
    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError), match="scope"):
        nirs4all.run(_pipeline(scope), {"X": object(), "y": object(), "sample_ids": object()}, engine="native", native_profile=PROFILE, save_charts=False)


def test_fold_scope_requires_the_explicit_role_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError), match="scope|profile|finetune_params"):
        nirs4all.run(_pipeline(), {"X": object(), "y": object(), "sample_ids": object()}, engine="native", save_charts=False)


def test_fold_scope_refuses_resampling_before_any_native_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_work(monkeypatch)
    pipeline = _pipeline()
    pipeline[0] = ShuffleSplit(n_splits=3, test_size=0.25, random_state=17)
    with pytest.raises((ValueError, TypeError), match="fold|KFold|splitter"):
        nirs4all.run(pipeline, {"X": object(), "y": object(), "sample_ids": object()}, engine="native", native_profile=PROFILE, save_charts=False)


def test_unseeded_shuffled_studies_are_refused_before_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_work(monkeypatch)
    pipeline = _pipeline()
    pipeline[0] = KFold(3, shuffle=True)
    with pytest.raises((ValueError, TypeError), match="seed|random_state|determin"):
        nirs4all.run(pipeline, {"X": object(), "y": object(), "sample_ids": object()}, engine="native", native_profile=PROFILE, save_charts=False)


@pytest.mark.parametrize("options", [{"approach": "individual"}, {"inner_cv": 2}])
def test_fold_mode_does_not_accept_unimplemented_inner_controls(monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]) -> None:
    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        nirs4all.run(_pipeline(**options), {"X": object(), "y": object(), "sample_ids": object()}, engine="native", native_profile=PROFILE, save_charts=False)


@pytest.mark.parametrize("options", [{"model_params": {}}, {"model_params": None}, {"n_trials": 0}])
def test_scope_requires_a_real_search_and_positive_per_study_budget(monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]) -> None:
    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        nirs4all.run(_pipeline(**options), {"X": object(), "y": object(), "sample_ids": object()}, engine="native", native_profile=PROFILE, save_charts=False)


def test_scope_declaration_is_caller_independent_and_campaign_stays_default() -> None:
    pipeline = _pipeline()
    before = copy.deepcopy(pipeline[-1]["finetune_params"])
    normalized = normalize_native_pls_phase_controls(pipeline, seed=12345)
    assert normalized.hpo_scope == "fold"
    assert normalized.search_axes == (
        {"kind": "int", "name": "n_components", "low": 1, "high": 3, "step": 1, "log": False},
        {"kind": "categorical", "name": "scale", "values": [False, True]},
    )
    assert pipeline[-1]["finetune_params"] == before
    default = _pipeline()
    del default[-1]["finetune_params"]["scope"]
    campaign = _pipeline("campaign")
    default_controls = normalize_native_pls_phase_controls(default, seed=12345)
    campaign_controls = normalize_native_pls_phase_controls(campaign, seed=12345)
    assert default_controls.hpo_scope == campaign_controls.hpo_scope == "campaign"
    assert default_controls.base_params == campaign_controls.base_params
    assert default_controls.search_axes == campaign_controls.search_axes
    assert default_controls.hpo == campaign_controls.hpo


def _result() -> NativeMethodsArchiveRunResult:
    result = object.__new__(NativeMethodsArchiveRunResult)
    state = {
        "schema_version": 1,
        "selected_variant_id": "variant:refit.winner",
        "relations": {"records": [{"observation_id": "obs:1", "sample_id": "sample:1", "origin_sample_id": "sample:1"}]},
        "base_plan": {"campaign": {"metadata": {"methods_hpo_operation": {
            "schema_version": 3, "scope": "fold", "native_profile": PROFILE,
            "parameter_paths": {"n_components": "n_components", "scale": "scale"},
        }}}},
        "outer_scopes": [{"scope_id": "hpo:test:scope:fit_cv:fold0", "phase": "FIT_CV", "outer_fold_id": "fold0", "winner_params": {"n_components": 1, "scale": True}}],
        "refit_scope": {
            "scope_id": "hpo:test:scope:refit", "phase": "REFIT", "outer_fold_id": None,
            "winner_params": {"n_components": 3, "scale": False},
            "resume_state": {"incumbent": {"score": 0.125}},
        },
    }
    result._native_outcome = {"methods_hpo_fold_state": state, "selected_variant_id": "variant:refit.winner"}
    result._native_package = {"schema_version": 2, "execution_bundle": {"methods_hpo_fold_state": copy.deepcopy(state)}}
    result._native_package_json = json.dumps(result._native_package)
    return result


def test_public_summary_uses_refit_winner_and_is_an_independent_snapshot() -> None:
    result = _result()
    assert result.tuning_best_params == {"model.n_components": 3, "model.scale": False}
    assert type(result.tuning_best_params["model.scale"]) is bool
    assert result.tuning_best_value == 0.125
    first = result.tuning_scope_results
    assert first["outer_scopes"][0]["winner_params"] != first["refit_scope"]["winner_params"]
    first["outer_scopes"][0]["winner_params"]["scale"] = False
    first["refit_scope"]["winner_params"]["n_components"] = 1
    assert result.tuning_scope_results["outer_scopes"][0]["winner_params"]["scale"] is True
    assert result.tuning_best_params["model.n_components"] == 3
    resume = result.tuning_resume_package
    assert resume is not None and resume == result._native_package
    resume["execution_bundle"] = {}
    assert result.tuning_resume_package == result._native_package


@pytest.mark.parametrize("params", [{"n_components": True}, {"n_components": 0}, {"n_components": 1.5}, {"scale": 1}, {"scale": "false"}, {"epochs": 1}])
def test_fold_winner_projection_cannot_coerce_invalid_public_types(params: dict[str, Any]) -> None:
    result = _result()
    result._native_outcome["methods_hpo_fold_state"]["refit_scope"]["winner_params"].update(params)
    with pytest.raises(NativeArchiveTrainingError):
        _ = result.tuning_best_params


@pytest.mark.parametrize("mutation", ["owner", "scope", "version", "foreign_path", "global_ledger", "selected_variant"])
def test_fold_projection_rejects_foreign_authority_and_mixed_campaign_ledger(mutation: str) -> None:
    result = _result()
    state = result._native_outcome["methods_hpo_fold_state"]
    operation = state["base_plan"]["campaign"]["metadata"]["methods_hpo_operation"]
    if mutation == "owner":
        operation["native_profile"] = "foreign.profile.v1"
    elif mutation == "scope":
        operation["scope"] = "campaign"
    elif mutation == "version":
        operation["schema_version"] = 2
    elif mutation == "foreign_path":
        operation["parameter_paths"]["scale"] = "foreign.scale"
    elif mutation == "global_ledger":
        result._native_outcome["methods_hpo_resume_state"] = {"incumbent": {"score": 0.0}}
    else:
        result._native_outcome["selected_variant_id"] = "variant:outer.winner"
    with pytest.raises(NativeArchiveTrainingError):
        _ = result.tuning_best_params

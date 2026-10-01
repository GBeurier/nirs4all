"""Closed public phase declarations must fail before data, FIT or optimizer work."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.api.native_archive_training import NativeArchiveTrainingError, NativeMethodsArchiveRunResult, _extract_portable_methods_hpo
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
from nirs4all.pipeline.dagml.native_pls_phase_controls import normalize_native_pls_phase_controls
from nirs4all.pipeline.dagml.raw_training_lowerer import _portable_methods_pls_params

PROFILE = "n4m.pls_role_pipeline.v1"


def _finetune(space: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": "n4m", "n_trials": 2, "sampler": "random", "pruner": "none",
        "approach": "grouped", "seed": 7, "metric": "rmse", "direction": "minimize",
        "model_params": space,
    }


def _pipeline(**step: Any) -> list[Any]:
    return [KFold(n_splits=3), {"model": PLSRegression(n_components=1), **step}]


def _forbid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.api import native_archive_training as module

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid native phase request consumed data or allocated FIT/optimizer runtime")

    monkeypatch.setattr(module, "_normalize_training_arrays", forbidden)
    monkeypatch.setattr(module, "_require_archive_runtime", forbidden)


@pytest.mark.parametrize("phase", ["train_params", "refit_params"])
@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 10}, {"warm_start": True}, {"tol": 1.0e-8}, {"max_iter": 100},
        {"copy": False}, {"scale_x": False}, {"scale_y": False},
        {"scale": True, "scale_x": True}, {"n_components": True},
        {"n_components": 0}, {"n_components": 1.5}, {"n_components": 2**31},
        {"scale": 0}, {"scale": "false"}, {"scale": np.bool_(False)}, None,
    ],
)
def test_invalid_phase_values_are_refused_before_data_fit_or_ask(
    monkeypatch: pytest.MonkeyPatch, phase: str, values: Any,
) -> None:
    import nirs4all

    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        nirs4all.run(
            _pipeline(**{phase: values}), {"X": object(), "y": object(), "sample_ids": object()},
            engine="native", native_profile=PROFILE, save_charts=False,
        )


@pytest.mark.parametrize(
    "model",
    [PLSRegression(n_components=2**31), PLSRegression(n_components=True),
     PLSRegression(scale=0), PLSRegression(max_iter=10), PLSRegression(tol=1.0e-8), PLSRegression(copy=False)],
)
def test_unexecuted_constructor_settings_are_refused_before_data(
    monkeypatch: pytest.MonkeyPatch, model: PLSRegression,
) -> None:
    import nirs4all

    _forbid_work(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        nirs4all.run(
            [KFold(3), {"model": model}], {"X": object(), "y": object(), "sample_ids": object()},
            engine="native", native_profile=PROFILE, save_charts=False,
        )


@pytest.mark.parametrize("name,axis,train", [
    ("n_components", ["int", 1, 3], 2), ("scale", [False, True], False),
])
def test_train_search_ownership_collision_precedes_fit_and_ask(
    monkeypatch: pytest.MonkeyPatch, name: str, axis: Any, train: Any,
) -> None:
    import nirs4all

    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="ownership overlaps"):
        nirs4all.run(
            _pipeline(train_params={name: train}, finetune_params=_finetune({name: axis})),
            {"X": object(), "y": object(), "sample_ids": object()},
            engine="native", native_profile=PROFILE, save_charts=False,
        )


@pytest.mark.parametrize("axis", [[0, 1], [False, False], [False], ["false", "true"], ["float", 0, 1]])
def test_scale_axis_requires_two_distinct_native_bools(axis: Any) -> None:
    with pytest.raises(ValueError, match="bool|distinct"):
        normalize_native_pls_phase_controls(_pipeline(finetune_params=_finetune({"scale": axis})), seed=3)


@pytest.mark.parametrize("space", [
    {"n_components": ["int", 1, 4]}, {"tol": ["float", 1.0e-8, 1.0e-6]},
    {"scale_x": [False, True]}, {"epochs": ["int", 1, 3]}, {},
])
def test_search_cannot_expand_the_executable_profile(space: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_native_pls_phase_controls(_pipeline(finetune_params=_finetune(space)), seed=3)


def test_normalization_preserves_separate_phase_ownership_and_caller_objects() -> None:
    pipeline = _pipeline(
        train_params={"n_components": 2}, refit_params={"n_components": 3, "scale": False},
        finetune_params=_finetune({"scale": {"type": "categorical", "choices": [False, True]}}),
    )
    before = copy.deepcopy(pipeline[-1])
    normalized = normalize_native_pls_phase_controls(pipeline, seed=12345)
    assert normalized.base_params == {"native_profile": PROFILE, "n_components": 1, "scale": True}
    assert normalized.train_params == {"n_components": 2}
    assert normalized.refit_params == {"n_components": 3, "scale": False}
    assert normalized.search_axes == ({"kind": "categorical", "name": "scale", "values": [False, True]},)
    assert pipeline[-1]["train_params"] == before["train_params"]
    assert pipeline[-1]["refit_params"] == before["refit_params"]
    assert pipeline[-1]["finetune_params"] == before["finetune_params"]
    normalized.train_params["n_components"] = 1
    assert pipeline[-1]["train_params"] == {"n_components": 2}


def test_exact_existing_snv_savgol_order_remains_closed() -> None:
    terminal = {"model": PLSRegression(n_components=1, scale=False)}
    valid = [KFold(3), StandardNormalVariate(), SavitzkyGolay(window_length=5, polyorder=2), terminal]
    normalized = normalize_native_pls_phase_controls(valid, seed=3)
    assert normalized.base_params["pipeline"] == {
        "schema_version": 1, "pipeline_type": "n4m.snv_savgol_smooth.v1",
        "savgol_window": 5, "savgol_poly_degree": 2,
    }
    for transforms in (
        [SavitzkyGolay(window_length=5, polyorder=2), StandardNormalVariate()],
        [StandardNormalVariate(), SavitzkyGolay(window_length=5, polyorder=2, deriv=1)],
    ):
        with pytest.raises(ValueError):
            normalize_native_pls_phase_controls([KFold(3), *transforms, terminal], seed=3)


def test_historical_native_profile_keeps_default_scale_and_hpo_v1() -> None:
    assert _portable_methods_pls_params([{"model": PLSRegression(n_components=2)}]) == {"n_components": 2}
    with pytest.raises(ValueError, match="default scale"):
        _portable_methods_pls_params([{"model": PLSRegression(n_components=2, scale=False)}])
    _, hpo = _extract_portable_methods_hpo(_pipeline(finetune_params=_finetune({"n_components": ["int", 1, 3]})), seed=3)
    assert hpo is not None and (hpo.low, hpo.high, hpo.step) == (1, 3, 1)
    with pytest.raises(ValueError):
        _extract_portable_methods_hpo(_pipeline(finetune_params=_finetune({"scale": [False, True]})), seed=3)


def test_unknown_profile_is_refused_before_native_work(monkeypatch: pytest.MonkeyPatch) -> None:
    import nirs4all

    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="native_profile"):
        nirs4all.run(
            _pipeline(), {"X": object(), "y": object(), "sample_ids": object()},
            engine="native", native_profile="n4m.pls_role_pipeline.v2", save_charts=False,
        )


@pytest.mark.parametrize("engine", ["legacy", "host"])
def test_profile_never_implicitly_selects_another_engine(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    import nirs4all

    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="native_profile|engine"):
        nirs4all.run(_pipeline(), {}, engine=engine, native_profile=PROFILE, save_charts=False)


def test_scale_category_order_is_canonical_without_mutating_declaration() -> None:
    options = _finetune({"scale": [True, False]})
    normalized = normalize_native_pls_phase_controls(_pipeline(finetune_params=options), seed=3)
    assert normalized.search_axes == ({"kind": "categorical", "name": "scale", "values": [False, True]},)
    assert options["model_params"]["scale"] == [True, False]


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("components", [None, 2])
def test_native_winner_projects_typed_bool_with_optional_components(scale: bool, components: int | None) -> None:
    parameters = {
        "scale": {"value": float(scale), "active": True, "native_kind": "categorical", "category_type": "boolean", "category_index": int(scale)},
    }
    expected: dict[str, Any] = {"model.scale": scale}
    if components is not None:
        parameters["n_components"] = {"value": float(components), "integer": True, "active": True}
        expected["model.n_components"] = components
    result = object.__new__(NativeMethodsArchiveRunResult)
    result._native_outcome = {
        "selected_variant_id": "variant:dag-winner",
        "effective_plan": {"campaign": {"metadata": {"methods_hpo_operation": {
            "schema_version": 2, "native_profile": PROFILE, "parameter_paths": {name: name for name in parameters},
        }}}},
        "methods_hpo_resume_state": {"incumbent": {"trial_id": 6}, "terminal_trials": [
            {"variant_id": "variant:native-incumbent", "trial": {"id": 6, "parameters": {"scale": {"category_index": int(not scale)}}}},
            {"variant_id": "variant:dag-winner", "trial": {"id": 7, "parameters": parameters}},
        ]},
    }
    actual = result.tuning_best_params
    assert actual == expected and type(actual["model.scale"]) is bool
    parameters["scale"]["category_type"] = "integer"
    with pytest.raises(NativeArchiveTrainingError, match="incumbent parameters"):
        _ = result.tuning_best_params

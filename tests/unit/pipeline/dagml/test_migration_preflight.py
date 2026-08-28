"""Regression locks for the stateful-concat DAG-ML migration refusal."""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs
from nirs4all.pipeline.config.generator import expand_spec
from nirs4all.pipeline.dagml.errors import DagMlPipelinePreflightRequired, DagMlStatefulConcatTransformMigrationRequired
from nirs4all.pipeline.dagml.migration_preflight import _operator_is_stateless, preflight_dagml_pipeline_migration
from nirs4all.pipeline.engine import legacy_fallback_metrics


def _stateful_concat_pipeline() -> list[Any]:
    return [
        {"concat_transform": [PCA(n_components=1)]},
        KFold(n_splits=2),
        {"model": PLSRegression(n_components=1)},
    ]


def _safe_pipeline() -> list[Any]:
    return [
        {"concat_transform": [StandardNormalVariate()]},
        KFold(n_splits=2),
        {"model": PLSRegression(n_components=1)},
    ]


def _refit_noop_pipeline() -> list[Any]:
    return [
        KFold(n_splits=2),
        {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}},
    ]


def _serialized_stateful_concat_pipeline() -> dict[str, Any]:
    return {
        "steps": [
            {
                "concat_transform": [
                    {"class": "sklearn.decomposition.PCA", "params": {"n_components": 1}},
                ]
            },
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 2}},
            {
                "model": {
                    "class": "sklearn.cross_decomposition.PLSRegression",
                    "params": {"n_components": 1},
                }
            },
        ]
    }


def _assert_public_refusal(pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove every public wrapper refuses before native, data, or legacy work begins."""

    run_module = importlib.import_module("nirs4all.api.run")
    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")
    reached: list[str] = []

    def _bomb(label: str):
        def fail(*_args: Any, **_kwargs: Any) -> None:
            reached.append(label)
            raise AssertionError(f"semantic preflight unexpectedly reached {label}")

        return fail

    monkeypatch.setattr(run_module, "PipelineRunner", _bomb("legacy-runner"))
    monkeypatch.setattr(run_backend, "run_via_dagml", _bomb("native-run"))
    monkeypatch.setattr(run_backend, "preflight_dagml_backend", _bomb("native-backend"))
    monkeypatch.setattr(run_backend, "_materialize_dataset", _bomb("dataset-materialization"))
    before = legacy_fallback_metrics()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(DagMlStatefulConcatTransformMigrationRequired, match="migration"):
            nirs4all.run(
                pipeline=pipeline,
                dataset=object(),
                engine="dag-ml",
                allow_legacy_fallback=True,
                verbose=0,
            )

    assert reached == []
    assert not any("falling back to the legacy engine" in str(item.message) for item in caught)
    assert legacy_fallback_metrics() == before


@pytest.mark.parametrize("form", ["list", "dict_steps", "dict_pipeline", "pipeline_configs", "json_path", "yaml_path", "batch"])
def test_public_pipeline_forms_refuse_before_fallback(
    form: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented wrapper reaches the non-catchable concat boundary first."""

    if form == "list":
        pipeline: Any = _stateful_concat_pipeline()
    elif form == "dict_steps":
        pipeline = {"steps": _stateful_concat_pipeline()}
    elif form == "dict_pipeline":
        pipeline = {"pipeline": _stateful_concat_pipeline()}
    elif form == "pipeline_configs":
        pipeline = PipelineConfigs(_stateful_concat_pipeline())
    elif form in {"json_path", "yaml_path"}:
        path = tmp_path / f"stateful_concat.{form.split('_')[0]}"
        path.write_text(json.dumps(_serialized_stateful_concat_pipeline()), encoding="utf-8")
        pipeline = path if form == "json_path" else str(path)
    else:
        pipeline = [_safe_pipeline(), _stateful_concat_pipeline()]

    _assert_public_refusal(pipeline, monkeypatch)


def test_reloaded_pipeline_configs_refuse_serialized_stateful_concat_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale public re-export remains a PipelineConfigs input after its implementation reloads."""

    from nirs4all.pipeline.config import pipeline_config as pipeline_config_module

    stale_pipeline_configs = PipelineConfigs
    importlib.reload(pipeline_config_module)
    assert stale_pipeline_configs is not pipeline_config_module.PipelineConfigs
    try:

        class ReloadedPipelineConfigs(stale_pipeline_configs):
            pass

        pipeline = ReloadedPipelineConfigs(
            [
                {"concat_transform": [PCA(n_components=1), TruncatedSVD(n_components=1)]},
                KFold(n_splits=2),
                {"model": PLSRegression(n_components=1)},
            ]
        )
        operations = pipeline.steps[0][0]["concat_transform"]
        assert [operation["class"].rsplit(".", 1)[-1] for operation in operations] == ["PCA", "TruncatedSVD"]
        assert all(not _operator_is_stateless(operation) for operation in operations)

        _assert_public_refusal(pipeline, monkeypatch)
        _assert_public_refusal([_safe_pipeline(), pipeline], monkeypatch)
    finally:
        # Restore the implementation binding so this regression does not leave a distinct class identity
        # behind for unrelated tests. The public re-export intentionally stays the held stale reference.
        pipeline_config_module.PipelineConfigs = stale_pipeline_configs


def test_malformed_nominal_pipeline_configs_requires_preflight() -> None:
    """A recognized config object with invalid serialized variants cannot pass as a safe pipeline."""

    malformed = object.__new__(PipelineConfigs)
    malformed.steps = [{"concat_transform": [PCA(n_components=1)]}]

    with pytest.raises(DagMlPipelinePreflightRequired, match="PipelineConfigs.steps"):
        preflight_dagml_pipeline_migration(malformed)


def test_direct_backend_refuses_before_backend_or_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct seam has the same ordering guarantee as the public API."""

    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("semantic preflight unexpectedly performed backend or dataset work")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", fail)
    monkeypatch.setattr(run_backend, "_materialize_dataset", fail)

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        run_backend.run_via_dagml(_stateful_concat_pipeline(), object())


@pytest.mark.parametrize("extra", [{"tuning": {}}, {"tuning": {}, "calibration": {}}])
def test_tuning_and_calibration_refuse_before_their_early_dagml_route(
    extra: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The early native tuning route cannot bypass the concat migration refusal."""

    run_module = importlib.import_module("nirs4all.api.run")
    reached: list[str] = []

    def fail(*_args: Any, **_kwargs: Any) -> None:
        reached.append("tuning-route")
        raise AssertionError("semantic preflight unexpectedly reached the native tuning route")

    monkeypatch.setattr(run_module, "_run_single_estimator_tuning_subset", fail)

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        nirs4all.run(
            pipeline=_stateful_concat_pipeline(),
            dataset=object(),
            engine="dag-ml",
            allow_legacy_fallback=True,
            verbose=0,
            **extra,
        )

    assert reached == []


def test_refit_params_use_all_partitions_is_not_a_preflight_boundary() -> None:
    """The legacy compatibility no-op must never become a typed migration refusal."""

    generator_payload = {
        "_or_": [
            {"use_all_partitions": False},
            {"use_all_partitions": True},
        ],
        "count": 1,
    }
    cyclic_payload: dict[str, Any] = {"use_all_partitions": True}
    cyclic_payload["self"] = cyclic_payload

    preflight_dagml_pipeline_migration(_refit_noop_pipeline())
    preflight_dagml_pipeline_migration([{"model": PLSRegression(n_components=1), "refit_params": generator_payload}])
    preflight_dagml_pipeline_migration([{"model": PLSRegression(n_components=1), "refit_params": cyclic_payload}])
    preflight_dagml_pipeline_migration([{"_zip_": {"params": {"_or_": [{"alpha": 1}, {"alpha": 2}]}}}])


def test_concat_refusal_is_limited_to_the_pre_cv_prefix() -> None:
    """A stateful concat after a splitter remains an ordinary capability boundary."""

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration(_stateful_concat_pipeline())

    preflight_dagml_pipeline_migration(
        [
            KFold(n_splits=2),
            {"concat_transform": [PCA(n_components=1)]},
            {"model": PLSRegression(n_components=1)},
        ]
    )
    preflight_dagml_pipeline_migration(_safe_pipeline())


def test_named_concat_operation_dict_is_scanned_without_an_operations_wrapper() -> None:
    """Legacy accepts named operation maps, which are not automatically stateful."""

    preflight_dagml_pipeline_migration(
        [
            {"concat_transform": {"snv": StandardNormalVariate()}},
            KFold(n_splits=2),
            {"model": PLSRegression(n_components=1)},
        ]
    )

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration(
            [
                {"concat_transform": {"pca": PCA(n_components=1)}},
                KFold(n_splits=2),
                {"model": PLSRegression(n_components=1)},
            ]
        )


def test_unseeded_choices_and_late_branch_generators_cannot_hide_stateful_concat() -> None:
    """Preflight mirrors the differing top-level and duplication-branch seed contracts."""

    safe = {"concat_transform": [StandardNormalVariate()]}
    unsafe = {"concat_transform": [PCA(n_components=1)]}

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration([{"_sample_": {"distribution": "choice", "values": [safe, unsafe], "num": 1}}])

    # The top-level random_state is honored by PipelineConfigs and chooses the safe option here.
    preflight_dagml_pipeline_migration(
        {
            "random_state": 1,
            "steps": [{"_sample_": {"distribution": "choice", "values": [safe, unsafe], "num": 1}}],
        }
    )

    # BranchController expands duplication generators later without PipelineConfigs.random_state;
    # a root seed must therefore not hide the unsafe first chain item.
    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration(
            {
                "random_state": 0,
                "steps": [
                    {
                        "branch": [
                            {
                                "name": "late-chain",
                                "steps": [{"_chain_": [unsafe, safe], "count": 1}],
                            }
                        ]
                    }
                ],
            }
        )


def test_weighted_unseeded_choice_is_uninspectable_not_a_false_concat_refusal() -> None:
    """A zero-weight stateful option is not reported as definitely active."""

    safe = {"concat_transform": [StandardNormalVariate()]}
    unsafe = {"concat_transform": [PCA(n_components=1)]}

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([{"_or_": [unsafe, safe], "count": 1, "_weights_": [0.0, 1.0]}])


def test_opaque_zip_concat_column_is_never_decided_from_one_random_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opaque generated sibling cannot truncate a stateful concat candidate."""

    sampler = importlib.import_module("nirs4all.pipeline.config._generator.strategies.or_strategy")

    def pick_first(
        population: list[Any],
        count: int,
        seed: int | None = None,
        weights: list[float] | None = None,
    ) -> list[Any]:
        del seed, weights
        return list(population[:count])

    def pick_last(
        population: list[Any],
        count: int,
        seed: int | None = None,
        weights: list[float] | None = None,
    ) -> list[Any]:
        del seed, weights
        return list(population[-count:])

    concat_zip = {
        "_zip_": {
            "params": {"_or_": [{"alpha": 1}, {"alpha": 2}]},
            "concat_transform": {
                "_or_": [[StandardNormalVariate()], [PCA(n_components=1)]],
                "count": 1,
            },
        }
    }

    monkeypatch.setattr(sampler, "sample_with_seed", pick_first)
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([concat_zip])

    monkeypatch.setattr(sampler, "sample_with_seed", pick_last)
    assert isinstance(expand_spec([concat_zip])[0][0]["concat_transform"][0], PCA)

    # A concrete splitter closes the pre-CV semantic boundary.
    preflight_dagml_pipeline_migration([KFold(n_splits=2), concat_zip])


def test_selected_generator_population_aggregates_opaque_and_concat_siblings() -> None:
    """Opaque expansion anywhere in a selector protects stateful concat anywhere in it."""

    safe = {
        "model": {"_or_": [PLSRegression(n_components=1), PLSRegression(n_components=1)]},
        "concat_transform": [StandardNormalVariate()],
    }
    unsafe = {
        "model": PLSRegression(n_components=1),
        "concat_transform": [PCA(n_components=1)],
    }
    selector = {"_or_": [safe, unsafe], "count": 1, "_seed_": 6}

    assert isinstance(expand_spec([selector])[0][0]["concat_transform"][0], PCA)
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([selector])

    seeded_grid = {
        "_grid_": {
            "params": {"_or_": [{"alpha": 1}, {"alpha": 2}]},
            "concat_transform": [[StandardNormalVariate()], [PCA(n_components=1)]],
        },
        "count": 1,
        "_seed_": 1,
    }
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([seeded_grid])


def test_active_branch_body_is_visible_but_payloads_remain_opaque() -> None:
    """Executable branch bodies are scanned without mistaking payload keys for workflow steps."""

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration([{"branch": [{"name": "unsafe", "steps": [{"concat_transform": [PCA(n_components=1)]}]}]}])

    metadata_cycle: dict[str, Any] = {"concat_transform": [PCA(n_components=1)]}
    metadata_cycle["self"] = metadata_cycle
    preflight_dagml_pipeline_migration(
        [
            {"concat_transform": [StandardNormalVariate()], "metadata": metadata_cycle},
            KFold(n_splits=2),
            {"model": PLSRegression(n_components=1)},
        ]
    )


def test_only_concat_relevant_cycles_trigger_generic_preflight() -> None:
    """Nontracked grammar cycles keep their ordinary path; concat cycles fail closed."""

    cycle: dict[str, Any] = {"_or_": []}
    cycle["_or_"].append(cycle)

    preflight_dagml_pipeline_migration([cycle])

    concat_cycle: dict[str, Any] = {"concat_transform": []}
    concat_cycle["concat_transform"].append(concat_cycle)

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([concat_cycle])

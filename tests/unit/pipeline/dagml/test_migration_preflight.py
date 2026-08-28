"""Regression locks for semantic DAG-ML migration refusal before legacy fallback."""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.controllers.data.branch import BranchController
from nirs4all.operators.transforms import StandardNormalVariate
from nirs4all.pipeline import PipelineConfigs
from nirs4all.pipeline.config.generator import expand_spec
from nirs4all.pipeline.dagml.errors import (
    DagMlPipelinePreflightRequired,
    DagMlRefitParamsMigrationRequired,
    DagMlStatefulConcatTransformMigrationRequired,
)
from nirs4all.pipeline.dagml.migration_preflight import preflight_dagml_pipeline_migration
from nirs4all.pipeline.engine import legacy_fallback_metrics


def _refit_pipeline() -> list[Any]:
    return [
        KFold(n_splits=2),
        {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}},
    ]


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


def _serialized_refit_pipeline() -> dict[str, Any]:
    return {
        "steps": [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 2}},
            {
                "model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 1}},
                "refit_params": {"use_all_partitions": True},
            },
        ]
    }


def _assert_public_refusal(pipeline: Any, expected: type[RuntimeError], monkeypatch: pytest.MonkeyPatch) -> None:
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
        with pytest.raises(expected, match="migration"):
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


@pytest.mark.parametrize(
    ("form", "expected"),
    [
        ("list", DagMlRefitParamsMigrationRequired),
        ("dict_steps", DagMlRefitParamsMigrationRequired),
        ("dict_pipeline", DagMlStatefulConcatTransformMigrationRequired),
        ("pipeline_configs", DagMlStatefulConcatTransformMigrationRequired),
        ("json_path", DagMlRefitParamsMigrationRequired),
        ("yaml_path", DagMlRefitParamsMigrationRequired),
        ("batch", DagMlRefitParamsMigrationRequired),
    ],
)
def test_public_pipeline_forms_refuse_before_fallback(
    form: str,
    expected: type[RuntimeError],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List/dict/path/config/batch forms all hit the non-catchable boundary first."""

    if form == "list":
        pipeline: Any = _refit_pipeline()
    elif form == "dict_steps":
        pipeline = {"steps": _refit_pipeline()}
    elif form == "dict_pipeline":
        pipeline = {"pipeline": _stateful_concat_pipeline()}
    elif form == "pipeline_configs":
        pipeline = PipelineConfigs(_stateful_concat_pipeline())
    elif form == "json_path":
        path = tmp_path / "refit.json"
        path.write_text(json.dumps(_serialized_refit_pipeline()), encoding="utf-8")
        pipeline = path
    elif form == "yaml_path":
        path = tmp_path / "refit.yaml"
        path.write_text(json.dumps(_serialized_refit_pipeline()), encoding="utf-8")
        pipeline = str(path)
    else:
        pipeline = [_safe_pipeline(), _refit_pipeline()]

    _assert_public_refusal(pipeline, expected, monkeypatch)


def test_direct_backend_refuses_before_backend_or_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct seam has the same ordering guarantee as the public API."""

    run_backend = importlib.import_module("nirs4all.pipeline.dagml.run_backend")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("semantic preflight unexpectedly performed backend or dataset work")

    monkeypatch.setattr(run_backend, "preflight_dagml_backend", fail)
    monkeypatch.setattr(run_backend, "_materialize_dataset", fail)

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        run_backend.run_via_dagml(_refit_pipeline(), object())


@pytest.mark.parametrize("extra", [{"tuning": {}}, {"tuning": {}, "calibration": {}}])
def test_tuning_and_calibration_refuse_before_their_early_dagml_route(extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The early native tuning route cannot bypass semantic migration refusal."""

    run_module = importlib.import_module("nirs4all.api.run")
    reached: list[str] = []

    def fail(*_args: Any, **_kwargs: Any) -> None:
        reached.append("tuning-route")
        raise AssertionError("semantic preflight unexpectedly reached the native tuning route")

    monkeypatch.setattr(run_module, "_run_single_estimator_tuning_subset", fail)

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        nirs4all.run(
            pipeline=_refit_pipeline(),
            dataset=object(),
            engine="dag-ml",
            allow_legacy_fallback=True,
            verbose=0,
            **extra,
        )

    assert reached == []


def test_unseeded_sample_is_exhaustive_but_unseeded_chain_keeps_its_order() -> None:
    """Random legacy re-normalization cannot hide an unsafe sample, without false-positive chain scans."""

    safe_model = {"model": PLSRegression(n_components=1)}
    unsafe_model = {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}}

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration([{"_sample_": {"distribution": "choice", "values": [safe_model, unsafe_model], "num": 1}}])

    preflight_dagml_pipeline_migration([{"_chain_": [safe_model, unsafe_model], "count": 1}])


def test_seeded_samples_are_exact_and_unseeded_cartesian_counts_are_exhaustive() -> None:
    """Honor the public root seed, but inspect every later-random cartesian candidate."""

    safe_model = {"model": PLSRegression(n_components=1)}
    unsafe_model = {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}}

    # ``random.Random(1).choice([safe, unsafe])`` chooses the safe entry. The root configuration
    # seed is the one PipelineConfigs passes to generator-core, so preflight must preserve it.
    preflight_dagml_pipeline_migration(
        {
            "random_state": 1,
            "steps": [{"_sample_": {"distribution": "choice", "values": [safe_model, unsafe_model], "num": 1}}],
        }
    )

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration(
            [
                {
                    "_cartesian_": [
                        {"_or_": [KFold(n_splits=2)]},
                        {"_or_": [safe_model, unsafe_model]},
                    ],
                    "count": 1,
                }
            ]
        )


def test_duplication_branch_generators_do_not_inherit_the_root_seed() -> None:
    """BranchController expands duplication generators later and without PipelineConfigs.random_state."""

    safe_model = {"model": PLSRegression(n_components=1)}
    unsafe_model = {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}}

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration(
            {
                "random_state": 1,
                "steps": [
                    {
                        "branch": [
                            {
                                "name": "late-generator",
                                "steps": [{"_sample_": {"distribution": "choice", "values": [safe_model, unsafe_model], "num": 1}}],
                            }
                        ]
                    }
                ],
            }
        )

    # generator-core's mixed-``_or_`` helper keeps `_seed_` as payload and therefore ignores it for
    # selection. The preflight must likewise consider this duplication branch unseeded.
    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration(
            {
                "random_state": 0,
                "steps": [
                    {
                        "branch": [
                            {
                                "name": "late-mixed-or",
                                "steps": [
                                    {
                                        "_or_": [unsafe_model, safe_model],
                                        "count": 1,
                                        "_seed_": 1,
                                        "name": "mixed",
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
        )

    # A branch ``_chain_`` uses its first item when unseeded. The outer root seed 0 would make
    # generator-core sample the safe item unless the duplication subtree blocks inherited seeding.
    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration(
            {
                "random_state": 0,
                "steps": [
                    {
                        "branch": [
                            {
                                "name": "late-chain",
                                "steps": [{"_chain_": [unsafe_model, safe_model], "count": 1}],
                            }
                        ]
                    }
                ],
            }
        )


def test_weighted_unseeded_or_is_uninspectable_not_an_active_unsafe_choice() -> None:
    """A zero-weight unsafe option must not be reported as an active semantic migration."""

    safe_model = {"model": PLSRegression(n_components=1)}
    unsafe_model = {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": True}}

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([{"_or_": [unsafe_model, safe_model], "count": 1, "_weights_": [0.0, 1.0]}])


def test_grid_and_zip_preserve_active_variant_semantics() -> None:
    """Grid unsafe values refuse, while a zip-truncated unsafe tail remains inactive."""

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration(
            [
                {
                    "_grid_": {
                        "model": [PLSRegression(n_components=1)],
                        "refit_params": [{"use_all_partitions": True}],
                    }
                }
            ]
        )

    preflight_dagml_pipeline_migration(
        [
            {
                "_zip_": {
                    "model": [PLSRegression(n_components=1)],
                    "refit_params": [{"use_all_partitions": False}, {"use_all_partitions": True}],
                }
            }
        ]
    )


def test_zip_with_opaque_generated_column_refuses_only_when_it_can_hide_a_migration() -> None:
    """An opaque generated ``params`` map cannot shorten a relevant ZipStrategy scan."""

    opaque_params = {"_or_": [{"alpha": 1}, {"alpha": 2}]}
    unsafe_zip = {
        "_zip_": {
            "model": [PLSRegression(n_components=1), PLSRegression(n_components=1)],
            "params": opaque_params,
            "refit_params": [
                {"use_all_partitions": False},
                {"use_all_partitions": True},
            ],
        }
    }

    # Runtime expands two aligned variants, but the preflight intentionally keeps ``params``
    # semantically opaque. It must therefore refuse as uninspectable instead of silently dropping
    # the second held-out-refit candidate or misreporting it as definitely active.
    assert [variant[0]["refit_params"]["use_all_partitions"] for variant in expand_spec([unsafe_zip])] == [False, True]
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([unsafe_zip])

    # The same opaque cardinality is harmless when every reachable refit override is safe.
    preflight_dagml_pipeline_migration(
        [
            {
                "_zip_": {
                    "model": [PLSRegression(n_components=1), PLSRegression(n_components=1)],
                    "params": opaque_params,
                    "refit_params": [
                        {"use_all_partitions": False},
                        {"use_all_partitions": False},
                    ],
                }
            }
        ]
    )

    # Nor does a generated opaque column by itself become a semantic-migration refusal.
    preflight_dagml_pipeline_migration(
        [
            {
                "_zip_": {
                    "model": [PLSRegression(n_components=1), PLSRegression(n_components=1)],
                    "params": opaque_params,
                }
            }
        ]
    )


def test_zip_opaque_alignment_keeps_concat_refusal_within_the_pre_cv_prefix() -> None:
    """A hidden stateful concat is generic preflight only before a concrete splitter."""

    opaque_params = {"_or_": [{"alpha": 1}, {"alpha": 2}]}
    concat_zip = {
        "_zip_": {
            "params": opaque_params,
            "concat_transform": [[StandardNormalVariate()], [PCA(n_components=1)]],
        }
    }

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([concat_zip])

    preflight_dagml_pipeline_migration([KFold(n_splits=2), concat_zip])


def test_unseeded_zip_migration_columns_are_never_decided_from_one_random_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relevant zip columns expand all unseeded choices before their opaque alignment is assessed."""

    sampler = importlib.import_module("nirs4all.pipeline.config._generator.strategies.or_strategy")

    def pick_first(population: list[Any], count: int, seed: int | None = None, weights: list[float] | None = None) -> list[Any]:
        del seed, weights
        return list(population[:count])

    def pick_last(population: list[Any], count: int, seed: int | None = None, weights: list[float] | None = None) -> list[Any]:
        del seed, weights
        return list(population[-count:])

    opaque_params = {"_or_": [{"alpha": 1}, {"alpha": 2}]}
    refit_zip = {
        "_zip_": {
            "model": [PLSRegression(n_components=1), PLSRegression(n_components=1)],
            "params": opaque_params,
            "refit_params": {
                "_or_": [
                    {"use_all_partitions": False},
                    {"use_all_partitions": True},
                ],
                "count": 1,
            },
        }
    }
    concat_zip = {
        "_zip_": {
            "params": opaque_params,
            "concat_transform": {
                "_or_": [[StandardNormalVariate()], [PCA(n_components=1)]],
                "count": 1,
            },
        }
    }

    # This is the old false-negative direction: one raw preflight draw was safe. The new guard
    # must refuse before consuming the sampler at all, because a later legacy expansion can differ.
    monkeypatch.setattr(sampler, "sample_with_seed", pick_first)
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([refit_zip])
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([concat_zip])

    # The same unseeded source can select the unsafe item at runtime.
    monkeypatch.setattr(sampler, "sample_with_seed", pick_last)
    assert expand_spec([refit_zip])[0][0]["refit_params"]["use_all_partitions"] is True
    assert isinstance(expand_spec([concat_zip])[0][0]["concat_transform"][0], PCA)


def test_seeded_grid_population_with_an_opaque_generated_mapping_is_generic_preflight() -> None:
    """A deterministic count still cannot use a projection with a different candidate population."""

    opaque_params = {"_or_": [{"alpha": 1}, {"alpha": 2}]}
    seeded_grid = {
        "_grid_": {
            "model": [PLSRegression(n_components=1)],
            "params": opaque_params,
            "refit_params": [
                {"use_all_partitions": False},
                {"use_all_partitions": True},
            ],
        },
        "count": 1,
        "_seed_": 1,
    }

    assert expand_spec([seeded_grid])[0][0]["refit_params"]["use_all_partitions"] is True
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([seeded_grid])

    # The same seeded selection remains admissible when no tracked semantic field is present.
    preflight_dagml_pipeline_migration(
        [
            {
                "_grid_": {
                    "model": [PLSRegression(n_components=1)],
                    "params": opaque_params,
                },
                "count": 1,
                "_seed_": 1,
            }
        ]
    )


def test_selected_cartesian_and_mixed_or_guard_opaque_migration_candidates() -> None:
    """The bounded population rule also reaches enclosing cartesian and mixed-OR selectors."""

    opaque_params = {"_or_": [{"alpha": 1}, {"alpha": 2}]}
    mixed_or = {
        "model": PLSRegression(n_components=1),
        "params": opaque_params,
        "_or_": [
            {"refit_params": {"use_all_partitions": False}},
            {"refit_params": {"use_all_partitions": True}},
        ],
        "count": 1,
    }
    cartesian = {
        "_cartesian_": [
            {
                "model": PLSRegression(n_components=1),
                "params": opaque_params,
                "refit_params": {"_or_": [{"use_all_partitions": False}, {"use_all_partitions": True}]},
            }
        ],
        "count": 1,
        "_seed_": 1,
    }

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration({"random_state": 1, "steps": [mixed_or]})
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([cartesian])


def test_selected_or_aggregates_opaque_and_migration_across_siblings() -> None:
    """An opaque safe sibling can still shift a seeded choice onto an unsafe sibling."""

    safe = {
        "model": {"_or_": [PLSRegression(n_components=1), PLSRegression(n_components=1)]},
        "refit_params": {"use_all_partitions": False},
    }
    unsafe = {
        "model": PLSRegression(n_components=1),
        "refit_params": {"use_all_partitions": True},
    }
    selector = {"_or_": [safe, unsafe], "count": 1, "_seed_": 6}

    assert expand_spec([selector])[0][0]["refit_params"]["use_all_partitions"] is True
    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([selector])

    preflight_dagml_pipeline_migration(
        [
            {
                "_or_": [
                    safe,
                    {"model": PLSRegression(n_components=1), "refit_params": {"use_all_partitions": False}},
                ],
                "count": 1,
                "_seed_": 6,
            }
        ]
    )


def test_duplication_branch_later_expansion_cannot_hide_a_zip_refit_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BranchController's second generator phase is included in the zip uncertainty boundary."""

    sampler = importlib.import_module("nirs4all.pipeline.config._generator.strategies.or_strategy")

    def pick_last(population: list[Any], count: int, seed: int | None = None, weights: list[float] | None = None) -> list[Any]:
        del seed, weights
        return list(population[-count:])

    nested_zip = {
        "_zip_": {
            "model": [PLSRegression(n_components=1), PLSRegression(n_components=1)],
            "params": {"_or_": [{"alpha": 1}, {"alpha": 2}]},
            "refit_params": [
                {
                    "_or_": [
                        {"use_all_partitions": False},
                        {"use_all_partitions": True},
                    ],
                    "count": 1,
                }
            ],
        }
    }
    pipeline = [{"branch": [{"name": "late-zip", "steps": [nested_zip]}]}]

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration(pipeline)

    monkeypatch.setattr(sampler, "sample_with_seed", pick_last)
    runtime_variants = BranchController.__new__(BranchController)._expand_list_with_generators([nested_zip])
    assert runtime_variants[0][0]["refit_params"]["use_all_partitions"] is True


def test_concat_refusal_is_limited_to_the_pre_cv_prefix() -> None:
    """A stateful concat after a splitter remains an ordinary native capability boundary."""

    with pytest.raises(DagMlStatefulConcatTransformMigrationRequired):
        preflight_dagml_pipeline_migration(_stateful_concat_pipeline())

    preflight_dagml_pipeline_migration([KFold(n_splits=2), {"concat_transform": [PCA(n_components=1)]}, {"model": PLSRegression(n_components=1)}])


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


def test_active_branch_body_is_visible_but_metadata_and_params_are_opaque() -> None:
    """Semantic-looking payload fields cannot create a false refusal or cycle failure."""

    with pytest.raises(DagMlRefitParamsMigrationRequired):
        preflight_dagml_pipeline_migration([{"branch": [{"name": "unsafe", "steps": [_refit_pipeline()[-1]]}]}])

    metadata_cycle: dict[str, Any] = {"_or_": []}
    metadata_cycle["_or_"].append(metadata_cycle)
    preflight_dagml_pipeline_migration(
        [
            {"concat_transform": [StandardNormalVariate()], "metadata": {"concat_transform": [PCA(n_components=1)]}},
            KFold(n_splits=2),
            {
                "model": {
                    "class": "sklearn.cross_decomposition.PLSRegression",
                    "params": {
                        "n_components": 1,
                        "refit_params": {"use_all_partitions": True},
                        "concat_transform": [PCA(n_components=1)],
                    },
                },
                "metadata": {"refit_params": metadata_cycle},
            },
        ]
    )


def test_active_cycle_refuses_before_any_fallback() -> None:
    """A cycle in the declarative grammar is uncertain, unlike a metadata cycle."""

    cycle: dict[str, Any] = {"_or_": []}
    cycle["_or_"].append(cycle)

    with pytest.raises(DagMlPipelinePreflightRequired):
        preflight_dagml_pipeline_migration([cycle])

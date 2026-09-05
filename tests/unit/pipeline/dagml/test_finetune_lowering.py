"""Unit locks for native deterministic ``finetune_params`` lowering."""

from __future__ import annotations

from collections import UserDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit

from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.pipeline.dagml.detect import _generation_kind
from nirs4all.pipeline.dagml.finetune_lowering import (
    _is_supported_native_refit_params_noop,
    lower_deterministic_finetune_params_to_generators,
    reject_native_training_param_overrides,
)
from nirs4all.pipeline.dagml.run_backend import _derive_config_name, _training_loss_generation_kind


class _PLSRegressionSubclass(PLSRegression):
    pass


class _CustomEstimator:
    pass


class _RefitParamsDict(dict[str, Any]):
    pass


class _StringSubclass(str):
    pass


class _FitPredictSplit:
    def fit(self) -> _FitPredictSplit:
        return self

    def predict(self) -> None:
        return None

    def split(self) -> tuple[()]:
        return ()


def test_local_loss_routing_treats_allowed_train_params_as_concrete_metadata() -> None:
    """A differentiable controller's fixed fit arguments are not a generator."""

    pipeline = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "train_params": {"epochs": 1}},
    ]

    assert _generation_kind(pipeline) == "operator"
    assert _training_loss_generation_kind(pipeline) == "none"


def test_native_refit_noop_proof_is_exact_and_preserves_original_config() -> None:
    """The accepted legacy no-op keeps the untouched pipeline as hash authority."""

    model = PLSRegression(n_components=8)
    refit_params = {"use_all_partitions": True}
    model_step = {"name": "PLS_with_refit", "model": model, "refit_params": refit_params}
    pipeline = [SNV(), ShuffleSplit(n_splits=3, random_state=42), model_step]

    assert _is_supported_native_refit_params_noop(pipeline)
    assert pipeline[2] is model_step
    assert model_step["model"] is model
    assert model_step["refit_params"] is refit_params
    assert _derive_config_name(pipeline, "") == "config_b2d6a46d"


def test_native_refit_noop_is_not_serialized_as_pls_fit_metadata() -> None:
    """The admitted controller no-op must not leak into ``PLSRegression.fit``."""

    from nirs4all.pipeline.dagml_bridge import _step_to_dsl

    refit_params = {"use_all_partitions": True}
    model_step = {"model": PLSRegression(n_components=2), "refit_params": refit_params}

    dsl_step = _step_to_dsl(model_step)

    assert "metadata" not in dsl_step
    assert model_step["refit_params"] is refit_params


def test_public_dispatch_routes_proven_refit_noop_without_mutating_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accepted key reaches native lowering unchanged and retains its legacy hash."""

    from nirs4all.pipeline.dagml import run_backend

    class FakeSpectro:
        name = "regression"

        def y(self, query: dict[str, str]) -> np.ndarray:
            assert query == {"partition": "train"}
            return np.asarray([0.1, 1.5, 2.7, 3.9])

        def features_sources(self) -> int:
            return 1

    model = PLSRegression(n_components=8)
    refit_params = {"use_all_partitions": True}
    model_step = {"name": "PLS_with_refit", "model": model, "refit_params": refit_params}
    pipeline = [SNV(), ShuffleSplit(n_splits=3, random_state=42), model_step]
    captured: dict[str, Any] = {}

    def fake_run_concrete_scores(variant: list[Any], *_args: Any, **_kwargs: Any) -> tuple[Any, str, bool, list[Any], dict[str, Any], list[Any]]:
        captured["variant"] = variant
        return object(), "PLSRegression", False, [], {}, []

    def fake_scores_to_run_result(*_args: Any, **kwargs: Any) -> str:
        captured["config_name"] = kwargs["config_name"]
        return "native-result"

    monkeypatch.setattr(run_backend, "_is_repetition_dataset", lambda _spectro: False)
    monkeypatch.setattr(run_backend, "_resolve_exclude", lambda steps, _spectro: (steps, [0, 1, 2, 3], set()))
    monkeypatch.setattr(run_backend, "_resolve_tags", lambda steps, _spectro, _cv_pool: (steps, {}))
    monkeypatch.setattr(run_backend, "_expand_operator_generators", lambda steps: [steps])
    monkeypatch.setattr(run_backend, "_run_concrete_scores", fake_run_concrete_scores)
    monkeypatch.setattr(run_backend, "_scores_to_run_result", fake_scores_to_run_result)

    result = run_backend._dispatch_run(
        pipeline,
        FakeSpectro(),
        Path("/tmp/nirs4all-test"),
        "dataset",
        None,
        "dag-ml-cli",
        None,
    )

    assert result == "native-result"
    assert captured["config_name"] == "config_b2d6a46d"
    assert captured["variant"][2] is model_step
    assert model_step["model"] is model
    assert model_step["refit_params"] is refit_params


@pytest.mark.parametrize(
    ("pipeline", "expected_key"),
    [
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"use_all_partitions": False}}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"use_all_partitions": 1}}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"use_all_partitions": np.bool_(True)}}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"other": True}}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {_StringSubclass("use_all_partitions"): True}}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": _RefitParamsDict({"use_all_partitions": True})}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": UserDict({"use_all_partitions": True})}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": MappingProxyType({"use_all_partitions": True})}], "refit_params"),
        ([KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"use_all_partitions": True, "future": "value"}}], "refit_params"),
        ([KFold(n_splits=2), {"model": _PLSRegressionSubclass(), "refit_params": {"use_all_partitions": True}}], "refit_params"),
        ([KFold(n_splits=2), {"model": _CustomEstimator(), "refit_params": {"use_all_partitions": True}}], "refit_params"),
        (
            [
                KFold(n_splits=2),
                PLSRegression(),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"branch": [[{"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}]]},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
                {"model": PLSRegression()},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
                {"branch": [[{"model": PLSRegression()}]]},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
                {"branch": [{"name": "x", "steps": [{"model": PLSRegression()}], "metadata": "x"}]},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {
                    "branch": [
                        {
                            "name": "x",
                            "steps": [{"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}],
                            "metadata": "x",
                        }
                    ]
                },
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
                {"refit_params": {"use_all_partitions": True}},
            ],
            "refit_params",
        ),
        (
            [
                KFold(n_splits=2),
                {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
                {"branch": [[{"refit_params": {"use_all_partitions": True}}]]},
            ],
            "refit_params",
        ),
        ([KFold(n_splits=2), {"model": PLSRegression()}, {"refit_params": {"use_all_partitions": True}}], "refit_params"),
        (
            [
                KFold(n_splits=2),
                {
                    "model": PLSRegression(),
                    "refit_params": {"use_all_partitions": True},
                    "train_params": {"epochs": 1},
                },
            ],
            "train_params",
        ),
    ],
    ids=(
        "false",
        "integer-one",
        "numpy-bool",
        "wrong-sole-key",
        "string-subclass-key",
        "dict-subclass",
        "user-dict",
        "mapping-proxy",
        "extra-key",
        "pls-subclass",
        "custom-estimator",
        "bare-model",
        "nested-only-pair",
        "multiple-models",
        "nested-model",
        "branch-list-wrapper-nested-model",
        "branch-list-wrapper-nested-refit",
        "duplicate-refit-key",
        "nested-refit-key",
        "refit-key-off-model-step",
        "train-params",
    ),
)
def test_public_dispatch_rejects_every_invalid_refit_shape_outside_proven_noop(pipeline: list[Any], expected_key: str) -> None:
    """Nearby malformed or unrecognized controls remain fail-closed."""

    from nirs4all.pipeline.dagml import run_backend

    assert not _is_supported_native_refit_params_noop(pipeline), expected_key
    with pytest.raises(NotImplementedError, match=expected_key):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_cyclic_refit_pipeline_without_recursing() -> None:
    """A cyclic pipeline is not a no-op proof and still reaches the native refusal."""

    from nirs4all.pipeline.dagml import run_backend

    model_step = {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}
    pipeline: list[Any] = [KFold(n_splits=2), model_step]
    pipeline.append(pipeline)

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_counts_aliased_model_step_as_two_occurrences() -> None:
    """An alias in a second branch position is another pipeline occurrence."""

    from nirs4all.pipeline.dagml import run_backend

    model_step = {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}
    pipeline = [KFold(n_splits=2), model_step, {"branch": [[model_step]]}]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


@pytest.mark.parametrize(
    "branch_step",
    [
        {"branch": [{"_or_": [{"model": PLSRegression()}]}]},
        {"branch": [{"_or_": [{"refit_params": {"use_all_partitions": True}}]}]},
        {"branch": {"x": {"_or_": [{"model": PLSRegression()}]}}},
        {"branch": {"x": {"_or_": [{"refit_params": {"use_all_partitions": True}}]}}},
        {"branch": {"_or_": [{"model": PLSRegression()}]}},
        {"branch": {"_or_": [{"refit_params": {"use_all_partitions": True}}]}},
    ],
    ids=("list-model", "list-refit", "named-model", "named-refit", "branch-model", "branch-refit"),
)
def test_public_dispatch_rejects_refit_noop_with_branch_generator(branch_step: dict[str, Any]) -> None:
    """An unexpanded branch generator cannot hide a second model or refit key."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
        branch_step,
    ]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_nested_refit_inside_branch_generator_before_runtime() -> None:
    """A generator-contained refit override reaches the native rejection guard."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline = [
        KFold(n_splits=2),
        {
            "branch": [
                {"_or_": [{"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}]}
            ]
        },
    ]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


@pytest.mark.parametrize(
    ("candidate", "branch_hidden"),
    [
        ({"class": "sklearn.ensemble.RandomForestClassifier"}, False),
        ({"function": "sklearn.linear_model.LogisticRegression"}, False),
        ({"instance": "sklearn.ensemble.RandomForestClassifier"}, False),
        ("sklearn.ensemble.RandomForestClassifier", False),
        ({"class": "sklearn.ensemble.RandomForestClassifier"}, True),
        ({"function": "sklearn.linear_model.LogisticRegression"}, True),
        ({"instance": "sklearn.ensemble.RandomForestClassifier"}, True),
        ("sklearn.ensemble.RandomForestClassifier", True),
    ],
    ids=("top-class", "top-function", "top-instance", "top-fqcn", "branch-class", "branch-function", "branch-instance", "branch-fqcn"),
)
def test_public_dispatch_rejects_refit_noop_with_ambiguous_model_representation(candidate: Any, branch_hidden: bool) -> None:
    """Serialized or string model candidates cannot join the exact PLS proof."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline: list[Any] = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
    ]
    pipeline.append({"branch": [[candidate]]} if branch_hidden else candidate)

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


@pytest.mark.parametrize("branch_hidden", [False, True], ids=("top", "branch"))
def test_public_dispatch_rejects_refit_noop_with_framework_factory(branch_hidden: bool) -> None:
    """A framework-marked factory is too ambiguous for the exact PLS proof."""

    from nirs4all.pipeline.dagml import run_backend

    def framework_model_factory() -> None:
        return None

    setattr(framework_model_factory, "framework", "tensorflow")
    pipeline: list[Any] = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
    ]
    pipeline.append({"branch": [[framework_model_factory]]} if branch_hidden else framework_model_factory)

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_refit_noop_with_direct_unknown_item() -> None:
    """An unclassified direct item cannot join the exact PLS proof."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
        object(),
    ]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_refit_noop_with_fit_predict_split_item() -> None:
    """A model-like object cannot be admitted merely because it has ``split``."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
        _FitPredictSplit(),
    ]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


@pytest.mark.parametrize(
    "workflow_step",
    [
        {"preprocessing": PLSRegression()},
        {"split": PLSRegression()},
        {"custom": PLSRegression()},
    ],
    ids=("preprocessing", "split", "custom"),
)
def test_public_dispatch_rejects_refit_noop_with_non_model_workflow_dict(workflow_step: dict[str, Any]) -> None:
    """Only the exact top-level model dict can accompany the PLS no-op."""

    from nirs4all.pipeline.dagml import run_backend

    pipeline = [
        KFold(n_splits=2),
        {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}},
        workflow_step,
    ]

    assert not _is_supported_native_refit_params_noop(pipeline)
    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            pipeline,
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_refit_noop_if_pls_gains_matching_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A future sklearn PLS parameter must never be silently ignored as a no-op."""

    from nirs4all.pipeline.dagml import run_backend

    original_get_params = PLSRegression.get_params

    def get_params_with_future_refit_option(self: PLSRegression, deep: bool = True) -> dict[str, Any]:
        params = cast(dict[str, Any], original_get_params(self, deep=deep))
        if not deep:
            params["use_all_partitions"] = False
        return params

    monkeypatch.setattr(PLSRegression, "get_params", get_params_with_future_refit_option)

    with pytest.raises(NotImplementedError, match="refit_params"):
        run_backend._dispatch_run(
            [KFold(n_splits=2), {"model": PLSRegression(), "refit_params": {"use_all_partitions": True}}],
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_training_parameter_rejection_ignores_non_structural_payload_mappings() -> None:
    """Reserved names inside model and label payloads are not pipeline controls."""

    reject_native_training_param_overrides(
        [
            {"model": RandomForestClassifier(), "class_weight": {"refit_params": 1.0}},
            {"label": {"train_params": "cohort"}},
        ],
        context="engine='dag-ml'",
    )


def test_deterministic_finetune_plain_grid_lowers_to_step_grid() -> None:
    steps, overrides = lower_deterministic_finetune_params_to_generators(
        [
            KFold(n_splits=3),
            {
                "model": PLSRegression(),
                "finetune_params": {
                    "engine": "dag-ml",
                    "metric": "mae",
                    "direction": "minimize",
                    "model_params": {"n_components": [2, 3], "scale": [True, False]},
                },
            },
        ]
    )

    assert overrides == {"selection_metric": "mae", "selection_objective": "minimize"}
    assert steps[1]["_grid_"] == {"n_components": [2, 3], "scale": [True, False]}
    assert "finetune_params" not in steps[1]
    assert _generation_kind(steps) == "param_model"


def test_deterministic_finetune_range_lowers_to_model_param_generator() -> None:
    steps, overrides = lower_deterministic_finetune_params_to_generators(
        [
            {
                "model": PLSRegression(),
                "finetune_params": {
                    "engine": "native",
                    "model_params": {"n_components": {"_range_": [2, 6, 2]}},
                },
            }
        ]
    )

    assert overrides == {}
    assert steps == [{"model": steps[0]["model"], "n_components": {"_range_": [2, 6, 2]}}]
    assert _generation_kind(steps) == "param_model"


def test_deterministic_finetune_refuses_silently_ignored_best_eval_mode() -> None:
    with pytest.raises(NotImplementedError, match="cannot honor eval_mode='best'.*host Optuna"):
        lower_deterministic_finetune_params_to_generators(
            [
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "engine": "dag-ml",
                        "eval_mode": "best",
                        "model_params": {"n_components": [2, 3]},
                    },
                }
            ]
        )


@pytest.mark.parametrize("engine", ["dagml", "native", "grid"])
def test_deterministic_finetune_engine_aliases_remain_read_only_compatibility(engine: str) -> None:
    steps, overrides = lower_deterministic_finetune_params_to_generators(
        [
            {
                "model": PLSRegression(),
                "finetune_params": {
                    "engine": engine,
                    "model_params": {"n_components": [2, 3]},
                },
            }
        ]
    )

    assert overrides == {}
    assert steps[0]["_grid_"] == {"n_components": [2, 3]}
    assert "finetune_params" not in steps[0]


def test_deterministic_finetune_rejects_adaptive_engines() -> None:
    with pytest.raises(ValueError, match="n4m/Optuna"):
        lower_deterministic_finetune_params_to_generators(
            [
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "engine": "n4m",
                        "model_params": {"n_components": [2, 3]},
                    },
                }
            ]
        )


def test_deterministic_finetune_rejects_trial_train_params_until_optimizer_adapter_exists() -> None:
    with pytest.raises(ValueError, match="train_params"):
        lower_deterministic_finetune_params_to_generators(
            [
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "engine": "dag-ml",
                        "model_params": {"n_components": [2, 3]},
                        "train_params": {"sample_weight": [1.0, 1.0]},
                    },
                }
            ]
        )


def test_public_dispatch_lowers_deterministic_finetune_to_native_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from nirs4all.pipeline.dagml import run_backend

    class FakeSpectro:
        def y(self, query: dict[str, str]) -> np.ndarray:
            assert query == {"partition": "train"}
            return np.asarray([0.1, 1.5, 2.7, 3.9])

        def features_sources(self) -> int:
            return 1

    captured: dict[str, Any] = {}

    def fake_run_native_generation(
        pipeline: list[Any],
        _spectro: Any,
        _dataset_arg: str,
        _cli: str,
        _venv_python: str,
        _run_dir: Path,
        metric: str,
        task_type: str,
        cv_pool: list[int] | None = None,
        excluded: set[int] | None = None,
        tags_by_sample: dict[int, list[str]] | None = None,
        **_kwargs: Any,
    ) -> str:
        captured.update(
            {
                "pipeline": pipeline,
                "metric": metric,
                "task_type": task_type,
                "cv_pool": cv_pool,
                "excluded": excluded,
                "tags_by_sample": tags_by_sample,
            }
        )
        return "native-generation-result"

    monkeypatch.setattr(run_backend, "_is_repetition_dataset", lambda _spectro: False)
    monkeypatch.setattr(run_backend, "_resolve_exclude", lambda pipeline, _spectro: (pipeline, [0, 1, 2, 3], set()))
    monkeypatch.setattr(run_backend, "_resolve_tags", lambda pipeline, _spectro, _cv_pool: (pipeline, {}))
    monkeypatch.setattr(run_backend, "_run_native_generation", fake_run_native_generation)

    result = run_backend._dispatch_run(
        [
            KFold(n_splits=2),
            {
                "model": PLSRegression(),
                "finetune_params": {
                    "engine": "dag-ml",
                    "metric": "rmse",
                    "direction": "minimize",
                    "model_params": {"n_components": [2, 3]},
                },
            },
        ],
        FakeSpectro(),
        Path("/tmp/nirs4all-test"),
        "dataset",
        None,
        "dag-ml-cli",
        None,
    )

    assert result == "native-generation-result"
    assert captured["metric"] == "rmse"
    assert captured["task_type"] == "regression"
    assert captured["cv_pool"] == [0, 1, 2, 3]
    assert captured["excluded"] == set()
    assert captured["tags_by_sample"] == {}
    assert captured["pipeline"][1]["_grid_"] == {"n_components": [2, 3]}
    assert "finetune_params" not in captured["pipeline"][1]


def test_public_dispatch_rejects_direction_that_contradicts_metric() -> None:
    from nirs4all.pipeline.dagml import run_backend

    class FakeSpectro:
        def y(self, _query: dict[str, str]) -> np.ndarray:
            return np.asarray([0.1, 1.5, 2.7, 3.9])

        def features_sources(self) -> int:
            return 1

    with pytest.raises(NotImplementedError, match="overriding the native selection direction"):
        run_backend._dispatch_run(
            [
                KFold(n_splits=2),
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "engine": "dag-ml",
                        "metric": "rmse",
                        "direction": "maximize",
                        "model_params": {"n_components": [2, 3]},
                    },
                },
            ],
            FakeSpectro(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_metrics_not_supported_by_public_native_selection() -> None:
    from nirs4all.pipeline.dagml import run_backend

    class FakeSpectro:
        def y(self, _query: dict[str, str]) -> np.ndarray:
            return np.asarray([0.1, 1.5, 2.7, 3.9])

        def features_sources(self) -> int:
            return 1

    with pytest.raises(NotImplementedError, match="metric='mae'"):
        run_backend._dispatch_run(
            [
                KFold(n_splits=2),
                {
                    "model": PLSRegression(),
                    "finetune_params": {
                        "engine": "dag-ml",
                        "metric": "mae",
                        "direction": "minimize",
                        "model_params": {"n_components": [2, 3]},
                    },
                },
            ],
            FakeSpectro(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )


def test_public_dispatch_rejects_unknown_step_level_training_controls_before_native_routing() -> None:
    from nirs4all.pipeline.dagml import run_backend

    with pytest.raises(NotImplementedError, match=r"(?=.*train_params)(?=.*refit_params)"):
        run_backend._dispatch_run(
            [
                KFold(n_splits=2),
                {
                    "model": PLSRegression(),
                    "train_params": {"sample_weight": [1.0, 1.0]},
                    "refit_params": {"sample_weight": [1.0, 1.0]},
                },
            ],
            object(),
            Path("/tmp/nirs4all-test"),
            "dataset",
            None,
            "dag-ml-cli",
            None,
        )

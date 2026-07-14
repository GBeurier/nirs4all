"""Unit locks for native deterministic ``finetune_params`` lowering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml.detect import _generation_kind
from nirs4all.pipeline.dagml.finetune_lowering import (
    lower_deterministic_finetune_params_to_generators,
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


def test_public_dispatch_rejects_step_level_train_and_refit_params_before_native_routing() -> None:
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

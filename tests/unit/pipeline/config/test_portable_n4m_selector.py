"""The shared n4m.Selector recipe resolves to supervised native Methods."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.steps.parser import StepParser

pytest.importorskip("n4m.feature_selection")


@pytest.mark.parametrize("extension", ["json", "yaml"])
@pytest.mark.parametrize(
    "method,method_params",
    [
        ("spa_select", {"top_k": 5}),
        ("random_frog_select", {"top_k": 5, "n_iterations": 8, "initial_size": 6, "min_size": 2, "max_size": 10, "seed": 7}),
    ],
)
def test_selector_recipe_files_fit_training_only(
    tmp_path,
    extension: str,
    method: str,
    method_params: dict[str, int],
) -> None:
    from n4m.feature_selection import Selector

    rows = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(rows * bands / 11) + np.cos(rows / 3 + bands / 7) + rows * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    node = {
        "class": "n4m.Selector",
        "params": {
            "method": method,
            "n_components": 2,
            "method_params": method_params,
        },
    }
    recipe = {"pipeline": [node]}
    path = tmp_path / f"selector.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        import yaml

        path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    config = PipelineConfigs(str(path))
    resolved = StepParser().parse(config.steps[0][0]).operator
    assert isinstance(resolved, Selector)
    assert TransformerMixinController._uses_y(resolved)
    resolved.fit(X[:28], y[:28])
    direct = Selector(method, 2, method_params).fit(X[:28], y[:28])
    np.testing.assert_array_equal(resolved.selected_indices_, direct.selected_indices_)
    np.testing.assert_allclose(resolved.transform(X[28:]), X[28:, np.sort(direct.selected_indices_)], rtol=0, atol=0)
    if method == "spa_select":
        leaked = Selector(method, 2, method_params).fit(X, y)
        assert not np.array_equal(leaked.selected_indices_, resolved.selected_indices_)


def test_selector_recipe_resolves_inside_feature_branch() -> None:
    from n4m.feature_selection import Selector

    node = {
        "class": "n4m.Selector",
        "params": {
            "method": "wvc_select",
            "n_components": 2,
            "method_params": {"top_k": 5, "normalize": False},
        },
    }
    config = PipelineConfigs(
        {
            "pipeline": [
                {
                    "branch": {
                        "selected": [node],
                        "original": [{"class": "n4m.SNV"}],
                    }
                },
                {"merge": "features"},
            ]
        }
    )
    branch = config.steps[0][0]["branch"]["selected"][0]
    assert isinstance(StepParser().parse(branch).operator, Selector)


def test_selector_empty_method_params_remains_json_object() -> None:
    node = {
        "class": "n4m.Selector",
        "params": {
            "method": "cars_select",
            "n_components": 2,
            "method_params": {},
        },
    }
    config = PipelineConfigs({"pipeline": [node]})
    assert config.steps[0][0] == node
    assert json.loads(json.dumps(config.steps[0][0]))["params"]["method_params"] == {}

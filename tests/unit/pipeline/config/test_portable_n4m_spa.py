"""The portable SPA recipe uses train-only Methods feature selection."""

import json

import numpy as np
import pytest

from nirs4all.controllers.transforms.transformer import TransformerMixinController
from nirs4all.pipeline.config.component_serialization import deserialize_component
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.steps.parser import StepParser

pytest.importorskip("n4m.feature_selection.wrapper")


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_spa_recipe_files_fit_training_rows_and_replay_selected_columns(tmp_path, extension: str) -> None:
    # Same matrix and parameters as nirs4all-r/tests/spa-selection.R. R uses
    # one-based indices; the Methods Python binding returns zero-based indices.
    samples = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 11) + np.cos(samples / 3 + bands / 7) + samples * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    recipe = {"pipeline": [{"class": "n4m.SPA", "params": {"top_k": 5, "n_components": 2}}]}
    path = tmp_path / f"spa.{extension}"
    if extension == "json":
        path.write_text(json.dumps(recipe), encoding="utf-8")
    else:
        path.write_text("pipeline:\n  - class: n4m.SPA\n    params:\n      top_k: 5\n      n_components: 2\n", encoding="utf-8")

    config = PipelineConfigs(str(path))
    selector = StepParser().parse(config.steps[0][0]).operator
    assert type(selector).__name__ == "SPA"
    assert TransformerMixinController._uses_y(selector)
    selector.fit(X[:28], y[:28])

    expected_python_indices = np.array([8, 3, 1, 10, 2])
    np.testing.assert_array_equal(selector.selected_indices_, expected_python_indices)
    np.testing.assert_array_equal(selector.selected_indices_ + 1, [9, 4, 2, 11, 3])
    np.testing.assert_array_equal(selector.get_support(indices=True), np.sort(expected_python_indices))
    np.testing.assert_allclose(selector.transform(X[28:]), X[28:, np.sort(expected_python_indices)])

    # An accidental fit on validation rows changes the answer for this fixture.
    leaked = deserialize_component(recipe["pipeline"][0]).fit(X, y)
    assert not np.array_equal(leaked.selected_indices_, selector.selected_indices_)

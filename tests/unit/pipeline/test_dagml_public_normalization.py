"""Public bare-estimator syntax must survive the DAG migration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.dagml.public_normalization import normalize_model_steps


@pytest.mark.parametrize("model", [Ridge(), LogisticRegression(), PLSRegression(2), Ridge])
def test_bare_model_normalization_preserves_identity(model):
    scaler = StandardScaler()
    splitter = KFold(3)
    original = [scaler, splitter, model]
    normalized = normalize_model_steps(original)
    assert normalized == [scaler, splitter, {"model": model}]
    assert original == [scaler, splitter, model]
    assert normalized[-1]["model"] is model


def test_explicit_transform_and_parameters_are_not_reclassified():
    pls = PLSRegression(2)
    steps = [{"preprocessing": pls}, {"y_processing": pls}, {"model": Ridge(), "train_params": {"nested": pls}}, {"concat_transform": [pls]}]
    normalized = normalize_model_steps(steps)
    assert all(left is right for left, right in zip(steps, normalized, strict=True))


def test_nested_and_named_branch_models_preserve_order_and_options():
    first, second = Ridge(1), Ridge(2)
    steps = [{"branch": {"first": [first], "second": second, "parallel": False, "n_jobs": 1, "_metadata": {"owner": "test"}}}, {"merge": "predictions"}]
    result = normalize_model_steps(steps)
    branch = result[0]["branch"]
    assert list(branch) == list(steps[0]["branch"])
    assert branch["first"] == [{"model": first}]
    assert branch["second"] == [{"model": second}]
    assert branch["parallel"] is False
    assert branch["_metadata"] is steps[0]["branch"]["_metadata"]
    assert steps[0]["branch"]["first"] == [first]
    assert normalize_model_steps([[first]]) == [[{"model": first}]]


@pytest.mark.parametrize("selector", ["by_source", "by_metadata", "by_tag", "by_filter"])
def test_separation_branch_only_normalizes_executable_body(selector):
    model = Ridge()
    branch = {selector: "cohort", "steps": [model], "values": [1, 2]}
    result = normalize_model_steps([{"branch": branch}])[0]["branch"]
    assert result == {**branch, "steps": [{"model": model}]}
    assert branch["steps"] == [model]


@pytest.mark.parametrize("classification", [False, True])
def test_normalized_shorthand_executes_real_dag_without_legacy(classification, monkeypatch):
    import nirs4all
    from nirs4all.pipeline.runner import PipelineRunner

    def forbidden(*args, **kwargs):
        raise AssertionError("implicit legacy execution")

    monkeypatch.setattr(PipelineRunner, "run", forbidden)
    rng = np.random.default_rng(314159)
    X = rng.normal(size=(48, 300))
    y = X[:, 0] * 2 - X[:, 1]
    model = LogisticRegression(max_iter=300) if classification else Ridge()
    splitter = StratifiedKFold(3, shuffle=True, random_state=42) if classification else KFold(3, shuffle=True, random_state=42)
    if classification:
        y = (y > np.median(y)).astype(int)
    result = nirs4all.run(normalize_model_steps([StandardScaler(), splitter, model]), (X, y), save_charts=False, verbose=0, random_state=42)
    assert result.execution_engine == "dag-ml"
    assert result.num_predictions > 0
    assert result._dagml_score_set is not None
    assert result._dagml_refit_artifacts
    assert np.isfinite(result.cv_best_score)

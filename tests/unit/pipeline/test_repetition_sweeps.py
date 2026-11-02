"""Repetition execution expands canonical and live sweeps before estimator parameters."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.pipeline.dagml import run_paths
from nirs4all.pipeline.dagml.steps import _apply_model_params, _split_pipeline


@pytest.mark.parametrize("model_form", ["instance", "class", "canonical"])
@pytest.mark.parametrize("sweep", ["range", "grid", "sibling"])
def test_repetition_expands_model_sweeps_before_concrete_execution(monkeypatch, tmp_path, model_form, sweep):
    canonical = model_form == "canonical"
    model = {"class": "sklearn.cross_decomposition.PLSRegression"} if canonical else PLSRegression if model_form == "class" else PLSRegression()
    splitter = {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 2}} if canonical else KFold(2)
    if sweep == "range":
        step = {"model": model, "_range_": [1, 3, 1], "param": "n_components"}
    elif sweep == "grid":
        step = {"model": model, "_grid_": {"n_components": [1, 2, 3]}}
    else:
        step = {"model": model, "n_components": {"_range_": [1, 3, 1]}}
    observed = []

    def concrete(pipeline, *args, **kwargs):
        steps, parsed = _split_pipeline(pipeline)
        assert isinstance(parsed, KFold)
        resolved = _apply_model_params(steps)
        components = resolved[0]["model"].n_components
        observed.append(components)
        return SimpleNamespace(cv_best_score=abs(components - 2), components=components)

    monkeypatch.setattr(run_paths, "_run_repetition_concrete", concrete)
    result = run_paths._run_repetition([splitter, step], None, "", "", "", Path(tmp_path), "rmse", "regression")
    assert observed == [1, 2, 3]
    assert result.components == 2


def test_repetition_combines_neighboring_preprocessing_and_model_sweeps(monkeypatch, tmp_path):
    observed = []

    def concrete(pipeline, *args, **kwargs):
        steps, _ = _split_pipeline(pipeline)
        resolved = _apply_model_params(steps)
        observed.append((type(resolved[0]).__name__, resolved[1]["model"].n_components))
        return SimpleNamespace(cv_best_score=1.0)

    monkeypatch.setattr(run_paths, "_run_repetition_concrete", concrete)
    run_paths._run_repetition([
        {"_or_": [StandardScaler(), MinMaxScaler()]}, KFold(2),
        {"model": PLSRegression(), "_range_": [1, 3, 1], "param": "n_components"},
    ], None, "", "", "", tmp_path, "rmse", "regression")
    assert set(observed) == {(name, nc) for name in ["StandardScaler", "MinMaxScaler"] for nc in [1, 2, 3]}
    assert len(observed) == 6


@pytest.mark.parametrize("fusion", ["rep_to_pp", "rep_to_sources"])
def test_repetition_fusion_expands_model_sweeps_and_keeps_variant_names(monkeypatch, tmp_path, fusion):
    observed = []

    def concrete(body, *args, **kwargs):
        steps, _ = _split_pipeline(body)
        nc = _apply_model_params(steps)[0]["model"].n_components
        observed.append(nc)
        return {"nc": nc}, "PLSRegression", []

    def project(scores, name, metric, task, classification, names, **kwargs):
        assert names == ["one", "two", "three"]
        assert [score[0]["nc"] for score in scores] == [1, 2, 3]
        return "projected"

    monkeypatch.setattr(run_paths, "_run_rep_fusion_concrete_scores", concrete)
    monkeypatch.setattr(run_paths, "_project_operator_sweep", project)
    rep_step = {fusion: "sample"}
    result = run_paths._run_rep_fusion(
        [rep_step, KFold(2), {"model": PLSRegression(), "_range_": [1, 3, 1], "param": "n_components"}],
        rep_step, SimpleNamespace(name="fusion"), "", "", "", tmp_path, "rmse", "regression",
        variant_config_names=["one", "two", "three"],
    )
    assert observed == [1, 2, 3]
    assert result == "projected"

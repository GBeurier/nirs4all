"""Repetition execution expands canonical and live sweeps before estimator parameters."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from nirs4all.pipeline.dagml import run_paths
from nirs4all.pipeline.dagml.steps import _apply_model_params, _split_pipeline


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("sweep", ["range", "grid", "sibling"])
def test_repetition_expands_model_sweeps_before_concrete_execution(monkeypatch, tmp_path, canonical, sweep):
    model = {"class": "sklearn.cross_decomposition.PLSRegression"} if canonical else PLSRegression()
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

"""General Python API workflows execute on DAG-ML without implicit legacy."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import nirs4all


@pytest.mark.parametrize("input_form", ["tuple", "mapping", "spectro"])
@pytest.mark.parametrize("scale_target", [False, True])
def test_default_general_run_accepts_historical_inputs(input_form, scale_target, monkeypatch):
    monkeypatch.delenv("N4A_ENGINE", raising=False)
    monkeypatch.setattr("nirs4all.pipeline.PipelineRunner.run", lambda *args, **kwargs: pytest.fail("implicit legacy execution"))
    rng = np.random.default_rng(37)
    X = rng.normal(size=(24, 6))
    y = X @ np.arange(1.0, 7.0) + rng.normal(scale=0.05, size=24)
    if input_form == "spectro":
        from nirs4all.data import SpectroDataset

        dataset = SpectroDataset("general-api")
        dataset.add_samples(X[:18], {"partition": "train"})
        dataset.add_targets(y[:18])
        dataset.add_samples(X[18:], {"partition": "test"})
        dataset.add_targets(y[18:])
    else:
        dataset = (X, y) if input_form == "tuple" else {"X": X, "y": y}
    pipeline = [MinMaxScaler()]
    if scale_target:
        pipeline.append({"y_processing": MinMaxScaler()})
    pipeline.extend([KFold(3), {"model": PLSRegression(2)}])
    with nirs4all.run(pipeline, dataset, verbose=0) as result:
        assert result.execution_engine == "dag-ml"
        assert result.num_predictions > 0
        assert np.isfinite(result.cv_best_score)

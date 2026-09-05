"""Unsplit variants remain independent REFIT executions, not invented CV."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@pytest.mark.parametrize("parameter_sweep", [False, True])
def test_full_training_model_choices_preserve_children_names_and_exports(tmp_path, monkeypatch, parameter_sweep):
    import nirs4all
    from nirs4all.pipeline.dagml.full_train import NoSplitEvaluationWarning
    from nirs4all.pipeline.dagml.full_train_variants import expand_full_train_variants

    rng = np.random.default_rng(814)
    X = rng.normal(size=(24, 5))
    y = X @ np.arange(1.0, 6.0)
    model_step = {"model": Ridge(), "alpha": {"_or_": [0.1, 10]}} if parameter_sweep else {"model": {"_or_": [Ridge(0.1), Ridge(10)]}}
    pipeline = [StandardScaler(), model_step]
    # Public run labels its first pipeline p0 before PipelineConfigs expansion.
    names = {name for _, name in expand_full_train_variants(pipeline, name="unsplit_p0")}
    with pytest.warns(NoSplitEvaluationWarning):
        result = nirs4all.run(pipeline, (X, y), name="unsplit", save_artifacts=False)
    assert len(result.runs) == 2
    assert np.isnan(result.cv_best_score)
    assert {row["config_name"] for row in result.predictions.filter_predictions()} == names
    for child in result.runs:
        assert child._dagml_score_set is not None
        assert {frame["lineage"]["phase"] for frame in child._dagml_node_results} == {"REFIT"}
        assert all(not item["evaluation"]["cross_validation"] for item in child.per_dataset.values())
    selected = result._source_run(None)
    expected = selected._dagml_refit_artifacts[0]["estimator"].predict(X.astype(np.float32))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("variant export retrained"))
    path = result.export(tmp_path / "selected.n4a")
    np.testing.assert_array_equal(nirs4all.predict(path, X).y_pred, expected)

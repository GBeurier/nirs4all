"""The direct run API respects Session configuration and explicit overrides."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def test_direct_run_inherits_session_options_and_explicit_false_wins(tmp_path, capsys):
    import nirs4all

    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 4))
    y = X @ np.arange(1.0, 5.0)
    with nirs4all.Session(verbose=0, save_artifacts=False, save_charts=False, workspace_path=tmp_path, report_naming="ml", random_state=73) as session:
        first = nirs4all.run([KFold(3), Ridge()], (X, y), session=session)
        assert first._dagml_results_dir is None
        assert first._dagml_export_spec["random_state"] == 73
        assert "Training pipeline" not in capsys.readouterr().out
        second = nirs4all.run(
            [KFold(3), Ridge()], (X, y), session=session,
            verbose=1, save_artifacts=True, random_state=None,
        )
        assert second._dagml_results_dir is not None
        assert second._dagml_export_spec["random_state"] is None
        assert "Training pipeline" in capsys.readouterr().out
        assert session._runner is None


def test_explicit_false_overrides_enabled_session_persistence(tmp_path):
    import nirs4all

    X = np.arange(96.0).reshape(24, 4)
    with nirs4all.Session(save_artifacts=True, save_charts=True, workspace_path=tmp_path) as session:
        result = nirs4all.run([KFold(3), Ridge()], (X, X[:, 0] + 0.23), session=session, save_artifacts=False, save_charts=False)
        assert result._dagml_results_dir is None
        assert session._runner is None


def test_run_signature_retains_effective_default_representations():
    import inspect

    import nirs4all

    parameters = inspect.signature(nirs4all.run).parameters
    expected = {"verbose": "1", "save_artifacts": "True", "save_charts": "None", "plots_visible": "False", "random_state": "None", "refit": "True", "cache": "None", "project": "None", "report_naming": "'nirs'"}
    assert {name: repr(parameters[name].default) for name in expected} == expected
    assert parameters["refit"].default is True
    assert parameters["save_artifacts"].default is True
    assert parameters["plots_visible"].default is False
    assert parameters["random_state"].default is None

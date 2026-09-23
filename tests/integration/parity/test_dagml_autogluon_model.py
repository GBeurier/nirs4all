"""Public AutoGluon controller parity with a directory-backed protocol double.

The optional AutoGluon distribution is not required for this contract test.
The double exercises the same DataFrame fit/predict and saved-directory API;
native AutoGluon training remains a separate optional-dependency check.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytest.importorskip("dag_ml")


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_autogluon_config_trains_refits_and_replays_archive(tmp_path, monkeypatch, mechanism: str) -> None:
    package = tmp_path / "autogluon"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tabular.py").write_text(
        """from pathlib import Path
import pickle
import numpy as np

class TabularPredictor:
    fit_calls = []

    def __init__(self, label, path, problem_type=None, verbosity=0, eval_metric=None):
        self.label = label
        self.path = path
        self.problem_type = problem_type
        self.verbosity = verbosity
        self.eval_metric = eval_metric

    def fit(self, train_data, **kwargs):
        type(self).fit_calls.append((len(train_data), len(kwargs['tuning_data']) if 'tuning_data' in kwargs else None))
        self.fit_options = kwargs
        x = train_data.drop(columns=[self.label]).to_numpy(dtype=float)
        y = train_data[self.label].to_numpy(dtype=float)
        self.weights = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        self.save()
        return self

    def predict(self, data):
        x = data.to_numpy(dtype=float)
        return np.column_stack([np.ones(len(x)), x]) @ self.weights

    def save(self, path=None):
        if path is not None:
            self.path = path
        directory = Path(self.path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'predictor.pkl').open('wb') as handle:
            pickle.dump({k: v for k, v in self.__dict__.items() if k != 'path'}, handle)

    @classmethod
    def load(cls, path):
        with (Path(path) / 'predictor.pkl').open('rb') as handle:
            state = pickle.load(handle)
        predictor = cls.__new__(cls)
        predictor.__dict__.update(state)
        predictor.path = path
        return predictor
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml CLI binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    sys.modules.pop("autogluon.tabular", None)
    sys.modules.pop("autogluon", None)
    importlib.invalidate_caches()

    from nirs4all.controllers.models import autogluon_model
    from nirs4all.utils.backend import clear_availability_cache

    clear_availability_cache()
    autogluon_model._ag_modules.clear()  # noqa: SLF001 - isolate optional fake package between mechanisms
    monkeypatch.setattr(autogluon_model, "AUTOGLUON_AVAILABLE", True)
    import nirs4all

    rng = np.random.default_rng(13)
    x = rng.normal(size=(12, 4)).astype(np.float32)
    y = (2 * x[:, 0] - x[:, 1] + 0.5).astype(np.float32)
    pipeline = [
        KFold(n_splits=2, shuffle=True, random_state=4),
        {"model": {"framework": "autogluon"}, "params": {"random_state": 9},
         "train_params": {"time_limit": 3, "presets": "medium_quality"}},
    ]
    legacy = nirs4all.run(pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy", save_charts=False, save_artifacts=False, verbose=0)
    assert np.isfinite(legacy.cv_best_score)
    from autogluon.tabular import TabularPredictor

    assert any(held_out == 6 for _, held_out in TabularPredictor.fit_calls)
    legacy.close()

    result = nirs4all.run(pipeline, (x, y), engine="dag-ml", save_charts=False)
    assert result.execution_engine == "dag-ml"
    assert np.isfinite(result.cv_best_score)
    fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - verify native refit identity
    assert fitted.predictor_.fit_options["time_limit"] == 3
    assert fitted.predictor_.fit_options["ag_args_fit"]["random_seed"] == 9
    expected = np.asarray(fitted.predict(x[:3])).reshape(-1)
    archive = result.export(tmp_path / "autogluon.n4a")
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    restored = load_general_archive(archive)["artifact"]["estimator"]
    np.testing.assert_allclose(np.asarray(restored.predict(x[:3])).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    import dag_ml._dag_ml as dag_ml_ext

    if callable(getattr(dag_ml_ext, "execute_phase_in_process", None)):
        actual = np.asarray(nirs4all.predict(archive, x[:3]).y_pred).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    result.close()

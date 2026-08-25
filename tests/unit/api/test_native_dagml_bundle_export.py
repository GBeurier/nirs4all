"""The dag-ml native bundle export must not touch the legacy runner."""

from __future__ import annotations

import importlib

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from nirs4all.api.result import RunResult
from nirs4all.pipeline.bundle import BundleLoader


def test_dagml_result_exports_captured_refit_without_legacy_run(tmp_path, monkeypatch) -> None:
    x_train = np.asarray([[-2.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 0.5]])
    y_train = 1.5 + 2.0 * x_train[:, 0] - 0.75 * x_train[:, 1]
    estimator = PLSRegression(n_components=1).fit(x_train, y_train)
    result = RunResult(predictions=object(), per_dataset={"native": {"engine": "dag-ml"}})
    result._dagml_export_spec = {"pipeline": [], "dataset": object()}  # noqa: SLF001
    result._dagml_results_dir = tmp_path / "captured"  # noqa: SLF001

    def fake_read_native_results(_path):
        return {
            "manifest": {"run_id": "run-native", "model_names": ["PLSRegression"]},
            "artifacts": [{"estimator": estimator, "y_transform": None}],
        }

    native_results = importlib.import_module("nirs4all.pipeline.dagml.native_results")
    monkeypatch.setattr(native_results, "read_native_results", fake_read_native_results)
    run_module = importlib.import_module("nirs4all.api.run")
    monkeypatch.setattr(run_module, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy run used")))

    bundle = result.export(tmp_path / "captured.n4a")
    x_test = np.asarray([[0.5, 1.0]])
    assert np.allclose(BundleLoader(bundle).predict(x_test), estimator.predict(x_test))
    assert result._dagml_legacy_result is None  # noqa: SLF001

"""Optional public parity oracle against the real AutoGluon Tabular runtime.

Run with ``autogluon.tabular[lightgbm]`` installed. The ordinary parity suite
keeps its directory-backed protocol double so this dependency stays optional.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_real_autogluon_cv_refit_and_portable_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mechanism: str, task: str) -> None:
    """A saved directory replays after its original AutoGluon path disappears."""
    # The protocol-double tests may have populated these optional-import caches.
    sys.modules.pop("autogluon.tabular", None)
    sys.modules.pop("autogluon", None)
    importlib.invalidate_caches()
    pytest.importorskip("autogluon.tabular")
    from nirs4all.controllers.models import autogluon_model
    from nirs4all.utils.backend import clear_availability_cache

    clear_availability_cache()
    autogluon_model._ag_modules.clear()  # noqa: SLF001 - isolate the preceding protocol double
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1" if mechanism == "in_process" else "0")
    if mechanism == "subprocess":
        from ._dagml_cli import dagml_cli_path

        cli = dagml_cli_path()
        if not cli.exists():
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))

    import nirs4all

    rng = np.random.default_rng(43)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    if task == "classification":
        y = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, 2, 5)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=4)
    else:
        y = (2 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=100)).astype(np.float32)
        splitter = KFold(n_splits=2, shuffle=True, random_state=4)
    pipeline = [
        splitter,
        {"model": {"framework": "autogluon"}, "train_params": {
            "hyperparameters": {"GBM": {"num_boost_round": 10}},
            "presets": "medium_quality", "time_limit": 30,
        }},
    ]
    if mechanism == "in_process":
        legacy = nirs4all.run(
            pipeline, (x, y), engine="legacy", workspace_path=tmp_path / "legacy",
            save_charts=False, verbose=0,
        )
        try:
            assert np.isfinite(legacy.cv_best_score)
        finally:
            legacy.close()

    with nirs4all.run(
        pipeline, (x, y), engine="dag-ml", allow_fallback=False,
        workspace_path=tmp_path / mechanism, save_charts=False, verbose=0,
    ) as result:
        assert result.execution_engine == "dag-ml"
        assert np.isfinite(result.cv_best_score)
        fitted = result._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - inspect actual refit
        assert fitted.predictor_.model_names()
        source = Path(fitted.predictor_.path)
        direct = np.asarray(fitted.predict(x[:5])).reshape(-1)
        if task == "classification":
            # nirs4all normalizes the public {2, 5} target to {0, 1} before
            # DAG-ML dispatches the AutoGluon operator.
            np.testing.assert_array_equal(fitted.classes_, [0, 1])
            np.testing.assert_array_equal(fitted.predictor_.class_labels, fitted.classes_)
            probabilities = fitted.predict_proba(x[:5])
            assert probabilities.shape == (5, 2)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        archive = result.export(tmp_path / f"{mechanism}_{task}.n4a")
        with zipfile.ZipFile(archive) as bundle:
            assert any(name.startswith("host_artifacts/") for name in bundle.namelist())

    shutil.rmtree(source)
    assert not source.exists()
    replay = np.asarray(nirs4all.predict(archive, x[:5]).y_pred).reshape(-1)
    if task == "classification":
        np.testing.assert_array_equal(replay, direct)
    else:
        np.testing.assert_allclose(replay, direct, rtol=1e-6, atol=1e-6)

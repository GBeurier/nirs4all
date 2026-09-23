"""Public AutoGluon controller parity with a directory-backed protocol double.

The optional AutoGluon distribution is not required for this contract test.
The double exercises the same DataFrame fit/predict and saved-directory API;
native AutoGluon training remains a separate optional-dependency check.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

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
        (directory / 'large_model.bin').write_bytes(b'x' * 65536)

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
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
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
    # A small synthetic file crosses a patched inline limit without allocating
    # hundreds of MiB; the model payload itself must remain small.
    from nirs4all.pipeline.dagml import general_archive

    monkeypatch.setattr(general_archive, "_MAX_INLINE_MODEL_BYTES", 8192)
    archive = result.export(tmp_path / "autogluon.n4a")
    from nirs4all.pipeline.dagml.general_archive import load_general_archive

    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        sidecar = manifest["host_artifacts"][0]["files"]
        assert {entry["uri"].split("/")[-1] for entry in sidecar} == {"predictor.pkl", "large_model.bin"}
        model_member = next(name for name in bundle.namelist() if name.endswith(".joblib"))
        assert bundle.getinfo(model_member).file_size < 8192

    restored = load_general_archive(archive)["artifact"]["estimator"]
    np.testing.assert_allclose(np.asarray(restored.predict(x[:3])).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    from nirs4all.pipeline.bundle import BundleLoader

    np.testing.assert_allclose(np.asarray(BundleLoader(archive).predict(x[:3])).reshape(-1), expected, rtol=1e-6, atol=1e-6)
    import dag_ml._dag_ml as dag_ml_ext

    if callable(getattr(dag_ml_ext, "execute_phase_in_process", None)):
        actual = np.asarray(nirs4all.predict(archive, x[:3]).y_pred).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if mechanism == "in_process":
        # Older archives inlined the predictor directory in joblib; they still replay.
        import joblib

        payload = io.BytesIO()
        joblib.dump(fitted, payload)
        old_member = "artifacts/step_1_foldfinal_old.joblib"
        old_archive = tmp_path / "old_inline.n4a"
        with zipfile.ZipFile(old_archive, "w") as bundle:
            bundle.writestr("manifest.json", json.dumps({
                "source_type": "dagml_native",
                "artifact_integrity": {old_member: "sha256:" + sha256(payload.getvalue()).hexdigest()},
            }))
            bundle.writestr(old_member, payload.getvalue())
        monkeypatch.setattr(general_archive, "_MAX_INLINE_MODEL_BYTES", 512 * 1024 * 1024)
        old_model = load_general_archive(old_archive)["artifact"]["estimator"]
        np.testing.assert_allclose(np.asarray(old_model.predict(x[:3])).reshape(-1), expected)

        # Tampering a sidecar is detected before pickle opcodes execute.
        corrupt = tmp_path / "corrupt.n4a"
        with zipfile.ZipFile(archive) as original, zipfile.ZipFile(corrupt, "w") as altered:
            for name in original.namelist():
                data = original.read(name)
                altered.writestr(name, b"changed" if name.endswith("large_model.bin") else data)
        monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("unverified model deserialized"))
        with pytest.raises(ValueError, match="sidecar integrity mismatch"):
            load_general_archive(corrupt)
        from nirs4all.pipeline.dagml.native_results import read_native_results

        run_dir = result._dagml_results_dir  # noqa: SLF001 - inspect the actual native persistence contract
        assert run_dir is not None
        native_ref = json.loads((run_dir / "manifest.json").read_text())["artifacts"][0]
        native_uri = next(file_ref["uri"] for item in native_ref["host_artifacts"]
                          for file_ref in item["files"] if file_ref["uri"].endswith("large_model.bin"))
        (run_dir / native_uri).write_bytes(b"changed")
        with pytest.raises(ValueError, match="sidecar integrity mismatch"):
            read_native_results(run_dir)
    result.close()


@pytest.mark.parametrize("mechanism", ["in_process", "subprocess"])
@pytest.mark.parity
def test_autogluon_classifier_exposes_classes_and_replays_archive(tmp_path, monkeypatch, mechanism: str) -> None:
    """The DAG host adapter preserves AutoGluon's class labels for CV and replay."""
    package = tmp_path / "autogluon"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tabular.py").write_text(
        """from pathlib import Path
import pickle
import numpy as np
import pandas as pd

class TabularPredictor:
    can_predict_proba = True

    def __init__(self, label, path, problem_type=None, verbosity=0, eval_metric=None):
        self.label = label
        self.path = path
        self.problem_type = problem_type

    def fit(self, train_data, **kwargs):
        x = train_data.drop(columns=[self.label]).to_numpy(dtype=float)
        y = train_data[self.label].to_numpy()
        self.class_labels = np.unique(y)
        self.centers = np.stack([x[y == label].mean(axis=0) for label in self.class_labels])
        self.save()
        return self

    def predict_proba(self, data):
        x = data.to_numpy(dtype=float)
        scores = -np.square(x[:, None, :] - self.centers[None, :, :]).sum(axis=2)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        return pd.DataFrame(weights / weights.sum(axis=1, keepdims=True), columns=self.class_labels)

    def predict(self, data):
        probabilities = self.predict_proba(data).to_numpy()
        return self.class_labels[probabilities.argmax(axis=1)]

    def save(self, path=None):
        if path is not None:
            self.path = path
        directory = Path(self.path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'predictor.pkl').open('wb') as handle:
            pickle.dump({key: value for key, value in self.__dict__.items() if key != 'path'}, handle)

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
            pytest.skip(f"dag-ml-cli binary not built at {cli}")
        monkeypatch.setenv("N4A_DAGML_CLI", str(cli))
    sys.modules.pop("autogluon.tabular", None)
    sys.modules.pop("autogluon", None)
    importlib.invalidate_caches()
    from nirs4all.controllers.models import autogluon_model
    from nirs4all.utils.backend import clear_availability_cache

    clear_availability_cache()
    autogluon_model._ag_modules.clear()  # noqa: SLF001 - isolate fake optional package
    monkeypatch.setattr(autogluon_model, "AUTOGLUON_AVAILABLE", True)
    import nirs4all

    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    pipeline = [
        StratifiedKFold(n_splits=2, shuffle=True, random_state=4),
        {"model": {"framework": "autogluon"}},
    ]
    legacy = nirs4all.run(
        pipeline, (x, y), engine="legacy", refit=False,
        workspace_path=tmp_path / "legacy", save_charts=False, save_artifacts=False, verbose=0,
    )
    try:
        assert np.isfinite(legacy.cv_best_score)
    finally:
        legacy.close()
    native = nirs4all.run(
        pipeline, (x, y), engine="dag-ml", allow_fallback=False,
        workspace_path=tmp_path / mechanism, save_charts=False, save_artifacts=False, verbose=0,
    )
    try:
        assert native.execution_engine == "dag-ml"
        assert np.isfinite(native.cv_best_score)
        fitted = native._dagml_refit_artifacts[0]["estimator"]  # noqa: SLF001 - check native artifact
        np.testing.assert_array_equal(fitted.classes_, [0, 1])
        probabilities = fitted.predict_proba(x[:5])
        assert probabilities.shape == (5, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        archive = native.export(tmp_path / f"classifier_{mechanism}.n4a")
        replay = np.asarray(nirs4all.predict(archive, x[:5]).y_pred).ravel()
        np.testing.assert_array_equal(replay, np.asarray(fitted.predict(x[:5])).ravel())
    finally:
        native.close()

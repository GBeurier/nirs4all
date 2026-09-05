"""Public transfer must freeze captured transforms and train only the new model."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.pipeline.dagml.general_archive import load_general_archive


@pytest.mark.parametrize("source_kind,sweep", [("prediction", False), ("bundle", False), ("prediction", True), ("bundle", True)])
@pytest.mark.parametrize("replace", [False, True])
def test_transfer_freezes_preprocessing_and_replays_without_fit(tmp_path, monkeypatch, source_kind, sweep, replace):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(32)
    old_x = rng.normal(size=(36, 5))
    old_y = (old_x[:, 0] + 50).astype(np.float32)
    new_x = rng.normal(size=(24, 5)) + 5
    new_y = (new_x[:, 1] - 80).astype(np.float32)
    model = {"_or_": [Ridge(0.1), Ridge(1.0)]} if sweep else Ridge(0.1)
    initial = nirs4all.run([StandardScaler(), {"y_processing": MinMaxScaler()}, KFold(3), {"model": model}],
                           (old_x, old_y), verbose=0, save_charts=False)
    with initial:
        bundle = initial.export(tmp_path / "source.n4a")
        loaded = load_general_archive(bundle)["artifact"]["estimator"]
        new_model = Ridge(9) if replace else None
        expected_model = clone(new_model if replace else loaded.estimator[-1])
        expected_model.fit(loaded.estimator[:-1].transform(new_x), loaded.y_transform.transform(new_y[:, None]).ravel())
        expected = loaded.y_transform.inverse_transform(expected_model.predict(loaded.estimator[:-1].transform(new_x))[:, None]).ravel()
        source = initial.best if source_kind == "prediction" else bundle
        database = Path(initial.best["workspace_path"]) / "store.sqlite"
        before = (database.read_bytes(), database.stat().st_mtime_ns, bundle.read_bytes())

        def forbid_fit(*args, **kwargs):
            raise AssertionError("transfer fitted a captured transformation")

        monkeypatch.setattr(StandardScaler, "fit", forbid_fit)
        monkeypatch.setattr(MinMaxScaler, "fit", forbid_fit)
        monkeypatch.setattr("nirs4all.pipeline.PipelineRunner", forbid_fit)
        with nirs4all.retrain(source, (new_x, new_y), mode="transfer", new_model=new_model,
                             verbose=0, save_charts=False, save_artifacts=False, results_path=tmp_path / "new") as result:
            assert result.execution_engine == "dag-ml"
            assert len(result._dagml_refit_artifacts) == 1
            assert np.isnan(result.cv_best_score)
            lineage = result._retrain_lineage
            assert lineage["preprocessing_frozen"] is True
            assert lineage["model_learned_state_reused"] is False
            assert lineage["model_replaced"] is replace
            exported = result.export(tmp_path / "transferred.n4a")
            with zipfile.ZipFile(exported) as archive:
                assert json.loads(archive.read("manifest.json"))["retrain_lineage"] == lineage
            monkeypatch.setattr(Ridge, "fit", forbid_fit)
            actual = nirs4all.predict(exported, new_x, verbose=0).y_pred
            np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=1e-6, atol=1e-6)
        assert before == (database.read_bytes(), database.stat().st_mtime_ns, bundle.read_bytes())


@pytest.mark.parametrize("source_kind", ["prediction", "bundle"])
def test_full_retrain_after_transfer_relearns_preprocessing(tmp_path, monkeypatch, source_kind):
    monkeypatch.chdir(tmp_path)
    x = np.random.default_rng(7).normal(size=(30, 4))
    with nirs4all.run([StandardScaler(), KFold(3), {"model": Ridge()}], (x, x[:, 0]), verbose=0, save_charts=False) as initial:
        with nirs4all.retrain(initial.best, (x + 40, x[:, 1]), mode="transfer", verbose=0, save_charts=False) as transferred:
            source = transferred.best if source_kind == "prediction" else transferred.export(tmp_path / "transfer-source.n4a")
            with nirs4all.retrain(source, (x - 80, x[:, 2]), verbose=0, save_charts=False, save_artifacts=False) as full:
                exported = full.export(tmp_path / "full.n4a")
                model = load_general_archive(exported)["artifact"]["estimator"].estimator
                means = [value.mean_ for value in model.get_params(deep=True).values() if isinstance(value, StandardScaler)]
                assert means
                for mean in means:
                    np.testing.assert_allclose(mean, (x - 80).mean(axis=0), atol=1e-5)
                assert full._retrain_lineage["learned_state_reused"] is False


@pytest.mark.parametrize("labels", [(3, 7), ("apple", "pear")])
def test_transfer_classification_preserves_recorded_class_axis_and_probabilities(tmp_path, monkeypatch, labels):
    monkeypatch.chdir(tmp_path)
    x = np.random.default_rng(4).normal(size=(40, 4)).astype(np.float32)
    y = np.where(x[:, 0] > 0, labels[0], labels[1])
    with nirs4all.run([StandardScaler(), KFold(2), {"model": LogisticRegression()}], (x, y), verbose=0, save_charts=False) as initial:
        # Dataset classification encoding precedes estimator training. The
        # fitted model retains that encoded axis, while the captured target
        # transformer and public prediction preserve the user's labels.
        source_bundle = initial.export(tmp_path / "source-classifier.n4a")
        source_capture = load_general_archive(source_bundle)["artifact"]["estimator"]
        assert len(source_capture.estimator.classes_) == 2
        with nirs4all.retrain(initial.best, (x + 2, y), mode="transfer", verbose=0, save_charts=False, save_artifacts=False) as result:
            exported = result.export(tmp_path / "classifier.n4a")
            captured = load_general_archive(exported)["artifact"]["estimator"]
            np.testing.assert_array_equal(captured.estimator.classes_, source_capture.estimator.classes_)
            np.testing.assert_array_equal(captured.y_transform.classes_, sorted(labels))
            np.testing.assert_allclose(captured.estimator.predict_proba(x + 2).sum(axis=1), 1)
            expected = captured.predict(x + 2)
            assert set(np.asarray(expected).ravel()) <= set(labels)
            np.testing.assert_array_equal(
                np.asarray(nirs4all.predict(exported, x + 2, verbose=0).y_pred).ravel(),
                np.asarray(expected).ravel(),
            )

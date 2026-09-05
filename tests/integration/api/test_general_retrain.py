"""Captured winners retrain on new data without replaying the source search."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import nirs4all
from nirs4all.pipeline.dagml.general_archive import load_general_archive
from nirs4all.pipeline.dagml.general_workspace import load_general_workspace_chain
from nirs4all.pipeline.dagml.rt import RtError


@pytest.mark.parametrize("source_kind,sweep", [("prediction", False), ("prediction", True), ("bundle", True)])
def test_selected_predictor_retrains_fresh_without_search_or_legacy(tmp_path, monkeypatch, source_kind, sweep):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(42)
    old_x = rng.normal(size=(36, 6))
    old_y = old_x[:, 0] * 2.2 - old_x[:, 1]
    new_x = rng.normal(size=(33, 6)) + 3
    new_y = (new_x[:, 0] * -1.8 + new_x[:, 2]).astype(np.float32)
    model = {"_or_": [Ridge(0.1), Ridge(1.0)]} if sweep else Ridge(0.1)
    with nirs4all.run([StandardScaler(), KFold(3), {"model": model}], (old_x, old_y), verbose=0, save_charts=False) as initial:
        bundle = initial.export(tmp_path / "selected.n4a")
        loaded = load_general_archive(bundle)
        fitted = loaded["artifact"]["estimator"].estimator
        expected = clone(fitted).fit(new_x, new_y).predict(new_x)
        old_prediction = nirs4all.predict(bundle, new_x, verbose=0).y_pred.copy()
        source = initial.best if source_kind == "prediction" else bundle
        database = Path(initial.best["workspace_path"]) / "store.sqlite"
        before = (database.read_bytes(), database.stat().st_mtime_ns, bundle.read_bytes())

        import nirs4all.pipeline as pipeline_module

        def forbid_legacy(*args, **kwargs):
            raise AssertionError("full retrain constructed the legacy runner")

        monkeypatch.setattr(pipeline_module, "PipelineRunner", forbid_legacy)
        # Native results are a new destination; library workspace writes are off.
        with nirs4all.retrain(source, (new_x, new_y), verbose=0, save_charts=False, results_path=tmp_path / "new-results", save_artifacts=False) as retrained:
            assert retrained.execution_engine == "dag-ml"
            assert len(retrained._dagml_refit_artifacts) == 1
            assert np.isnan(retrained.cv_best_score), "full winner retrain must not invent a new CV search"
            exported = retrained.export(tmp_path / "retrained.n4a")
            prediction = nirs4all.predict(exported, new_x, verbose=0).y_pred
            np.testing.assert_allclose(np.asarray(prediction).ravel(), expected, rtol=1e-6, atol=1e-6)
            assert not np.allclose(np.asarray(prediction).ravel(), np.asarray(old_prediction).ravel())
            lineage = retrained._retrain_lineage
            assert lineage["learned_state_reused"] is False
            assert lineage["parameter_search_repeated"] is False
            assert lineage["source_integrity_verified"] is True
            assert lineage["source_kind"] == ("workspace_chain" if source_kind == "prediction" else "n4a_bundle")
            with zipfile.ZipFile(exported) as archive:
                assert json.loads(archive.read("manifest.json"))["retrain_lineage"] == lineage
        assert before == (database.read_bytes(), database.stat().st_mtime_ns, bundle.read_bytes())
        np.testing.assert_array_equal(nirs4all.predict(bundle, new_x, verbose=0).y_pred, old_prediction)


def test_workspace_target_transform_is_refitted_with_selected_estimator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(16)
    x = rng.normal(size=(30, 4))
    y = (x[:, 0] * 4 + 50).astype(np.float32)
    with nirs4all.run([StandardScaler(), {"y_processing": MinMaxScaler()}, KFold(3), {"model": Ridge(0.2)}], (x, y), verbose=0, save_charts=False) as initial:
        source = initial.best
        loaded = load_general_workspace_chain(source["workspace_path"], source["chain_id"])
        assert loaded is not None
        new_y = y + 300
        target = clone(loaded["artifact"]["y_transform"])
        transformed = target.fit_transform(new_y.reshape(-1, 1)).ravel()
        model = clone(loaded["artifact"]["estimator"]).fit(x, transformed)
        expected = target.inverse_transform(model.predict(x).reshape(-1, 1)).ravel()
        with nirs4all.retrain(source, (x, new_y), verbose=0, save_charts=False, save_artifacts=False, results_path=tmp_path / "new") as retrained:
            exported = retrained.export(tmp_path / "target-retrained.n4a")
            actual = nirs4all.predict(exported, x, verbose=0).y_pred
            np.testing.assert_allclose(np.asarray(actual).ravel(), expected, rtol=1e-6, atol=1e-6)


def test_corrupt_workspace_artifact_fails_before_any_new_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.random.default_rng(9).normal(size=(24, 4))
    with nirs4all.run([KFold(3), {"model": Ridge()}], (x, x[:, 0]), verbose=0, save_charts=False) as initial:
        source = initial.best
        from nirs4all.pipeline.storage.store_queries import GET_ARTIFACT, GET_CHAIN

        root = Path(source["workspace_path"])
        connection = sqlite3.connect(f"{(root / 'store.sqlite').as_uri()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            chain = connection.execute(GET_CHAIN, [source["chain_id"]]).fetchone()
            artifact_id = json.loads(chain["fold_artifacts"])["final"]
            record = connection.execute(GET_ARTIFACT, [artifact_id]).fetchone()
            artifact = root / "artifacts" / record["artifact_path"]
        finally:
            connection.close()
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"corruption")
        before = hashlib.sha256(artifact.read_bytes()).hexdigest()

        def forbid_fit(*args, **kwargs):
            raise AssertionError("a corrupted source reached fit")

        monkeypatch.setattr(Ridge, "fit", forbid_fit)
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            nirs4all.retrain(source, (x, x[:, 0]), verbose=0, save_charts=False)
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before


def test_clone_retaining_source_estimator_is_refused_before_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.random.default_rng(18).normal(size=(24, 4))
    with nirs4all.run([KFold(3), {"model": Ridge()}], (x, x[:, 0]), verbose=0, save_charts=False) as initial:
        source = initial.best
        monkeypatch.setattr("sklearn.base.clone", lambda estimator: estimator)

        def forbid_fit(*args, **kwargs):
            raise AssertionError("a retained source estimator reached fit")

        monkeypatch.setattr(Ridge, "fit", forbid_fit)
        with pytest.raises(RtError, match="cloning retained a captured estimator"):
            nirs4all.retrain(source, (x, x[:, 0]), verbose=0, save_charts=False)

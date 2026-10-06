"""Data-integrity regressions for STO-01/02/03/04/05/06/08/10/11/12/16."""

from __future__ import annotations

import io
import json
import runpy
import sqlite3
import zipfile
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.bundle.generator import BundleGenerator, write_single_model_bundle
from nirs4all.pipeline.bundle.loader import BundleLoader
from nirs4all.pipeline.config.context import MapArtifactProvider
from nirs4all.pipeline.resolver import ResolvedPrediction
from nirs4all.pipeline.storage import workspace_store as storage_module
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


@pytest.fixture
def store(tmp_path):
    with WorkspaceStore(tmp_path / "workspace") as value:
        yield value


def _hierarchy(store, *, shared_id=None, run_id=None, final=False):
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float) ** 2
    scaler = StandardScaler().fit(X)
    if shared_id is None:
        shared_id = store.save_artifact(scaler, "StandardScaler", "transformer", "joblib")
    model = LinearRegression().fit(scaler.transform(X), y)
    model_id = store.save_artifact(model, "LinearRegression", "model", "joblib")
    run_id = run_id or store.begin_run("audit", {}, [{"name": "wheat"}])
    pipeline_id = store.begin_pipeline(run_id, "audit", [{"transform": "StandardScaler"}, {"model": "LinearRegression"}], [], "wheat", "hash")
    fold = "fold_final" if final else "fold_0"
    chain_id = store.save_chain(
        pipeline_id, [{"step_idx": 1}, {"step_idx": 2}], 2, "LinearRegression", "StandardScaler", "per_fold",
        {fold: model_id}, {"1": [shared_id]},
    )
    pred_id = store.save_prediction(
        pipeline_id=pipeline_id, chain_id=chain_id, dataset_name="wheat", model_name="LinearRegression", model_class="LinearRegression",
        fold_id=fold, partition="val", val_score=0.2, test_score=0.3, train_score=0.1, metric="rmse", task_type="regression",
        n_samples=8, n_features=3, scores={}, best_params={}, branch_id=None, branch_name=None, exclusion_count=0, exclusion_rate=0.0,
    )
    return {"run_id": run_id, "pipeline_id": pipeline_id, "chain_id": chain_id, "pred_id": pred_id, "shared_id": shared_id, "model_id": model_id, "X": X, "model": model}


@pytest.mark.parametrize("delete_scope", ["prediction", "run"])
def test_shared_artifact_survives_sibling_deletion(store, tmp_path, delete_scope):
    first = _hierarchy(store)
    second = _hierarchy(store, shared_id=first["shared_id"])
    expected = store.replay_chain(second["chain_id"], second["X"])
    assert store._ensure_open().execute("SELECT ref_count FROM artifacts WHERE artifact_id = ?", [first["shared_id"]]).fetchone()[0] == 1
    if delete_scope == "prediction":
        store.delete_prediction(first["pred_id"])
    else:
        store.delete_run(first["run_id"])
    np.testing.assert_allclose(store.replay_chain(second["chain_id"], second["X"]), expected)
    bundle = store.export_chain(second["chain_id"], tmp_path / "survivor.n4a")
    np.testing.assert_allclose(BundleLoader(bundle).predict(second["X"]), expected)
    store.delete_run(second["run_id"])
    with pytest.raises(KeyError):
        store.load_artifact(first["shared_id"])


def test_transient_cleanup_keeps_winners_shared_preprocessing(store):
    loser = _hierarchy(store)
    winner = _hierarchy(store, run_id=loser["run_id"], shared_id=loser["shared_id"], final=True)
    store._ensure_open().execute("DELETE FROM predictions")
    expected = store.replay_chain(winner["chain_id"], winner["X"])
    store.cleanup_transient_artifacts(loser["run_id"], "wheat", [winner["pipeline_id"]])
    np.testing.assert_allclose(store.replay_chain(winner["chain_id"], winner["X"]), expected)


def _auxiliary_rows(store, ids, link):
    """Seed FK shapes directly; serialization payloads are irrelevant to cascade safety."""
    conn = store._ensure_open()
    value = ids[{"run_id": "run_id", "pipeline_id": "pipeline_id", "chain_id": "chain_id", "prediction_id": "pred_id"}[link]]
    conn.execute(
        f"INSERT INTO conformal_results (conformal_id, {link}, artifact_fingerprint, result_fingerprint, coverages, artifact_json, result_json) VALUES ('c', ?, 'a', 'r', '[]', '{{}}', '{{}}')",
        [value],
    )
    conn.execute(
        "INSERT INTO robustness_results (robustness_id, conformal_id, result_fingerprint, mode, scenario_count, slice_by, report_json) VALUES ('r', 'c', 'r', 'audit', 0, '[]', '{}')",
    )
    if link != "prediction_id":
        conn.execute(
            f"INSERT INTO tuning_results (tuning_id, {link}, tuning_fingerprint, result_fingerprint, engine, metric, direction, best_value, n_trials, tuning_json, result_json) VALUES ('t', ?, 't', 'r', 'optuna', 'rmse', 'minimize', 0.2, 1, '{{}}', '{{}}')",
            [value],
        )


@pytest.mark.parametrize("link", ["run_id", "pipeline_id", "chain_id", "prediction_id"])
@pytest.mark.parametrize("scope", ["run", "prediction"])
def test_deletes_cascade_auxiliary_result_foreign_keys(store, link, scope):
    ids = _hierarchy(store)
    _auxiliary_rows(store, ids, link)
    if scope == "run":
        store.delete_run(ids["run_id"])
        assert store.get_run(ids["run_id"]) is None
    else:
        store.delete_prediction(ids["pred_id"])
    expected = 1 if scope == "prediction" and link == "run_id" else 0
    for table in ("conformal_results", "robustness_results", "tuning_results"):
        count = store._ensure_open().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count == (0 if table == "tuning_results" and link == "prediction_id" else expected)
    assert store._ensure_open().execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("scope", ["run", "prediction"])
def test_failed_cascade_rolls_back_metadata_refs_and_auxiliary_results(store, monkeypatch, scope):
    ids = _hierarchy(store)
    _auxiliary_rows(store, ids, "chain_id")
    conn = store._ensure_open()
    conn.execute("CREATE TRIGGER refuse_chain_delete BEFORE DELETE ON chains BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
    tombstone = Mock()
    monkeypatch.setattr(store.array_store, "delete_batch", tombstone)
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        if scope == "run":
            store.delete_run(ids["run_id"])
        else:
            store.delete_prediction(ids["pred_id"])
    tombstone.assert_not_called()
    assert not conn.in_transaction
    assert store.get_chain(ids["chain_id"]) is not None
    assert store.query_predictions().height == 1
    assert conn.execute("SELECT COUNT(*) FROM tuning_results").fetchone()[0] == 1
    assert conn.execute("SELECT ref_count FROM artifacts WHERE artifact_id = ?", [ids["shared_id"]]).fetchone()[0] == 1
    assert store.gc_artifacts() == 0
    store.replay_chain(ids["chain_id"], ids["X"])


def test_outer_rollback_preserves_artifact_files(store):
    ids = _hierarchy(store)
    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            store.delete_run(ids["run_id"])
            raise RuntimeError("rollback")
    assert store.get_chain(ids["chain_id"]) is not None
    store.replay_chain(ids["chain_id"], ids["X"])


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_transaction_rolls_back_base_exceptions_and_releases_writer(store, interruption):
    conn = store._ensure_open()
    with pytest.raises(interruption), store.transaction():
        store.begin_run("interrupted", {}, [])
        raise interruption()
    assert not conn.in_transaction
    assert store.list_runs().is_empty()
    with WorkspaceStore(store.workspace_path) as other:
        other.begin_run("next", {}, [])
    assert store.list_runs().height == 1


@pytest.mark.parametrize("wrapper", ["decorator", "statement"])
@pytest.mark.parametrize("message", ["no such table: missing", "database or disk is full", "attempt to write a readonly database"])
def test_permanent_sqlite_errors_are_not_retried(store, monkeypatch, wrapper, message):
    sleep = Mock()
    monkeypatch.setattr(storage_module.time, "sleep", sleep)
    fail = Mock(side_effect=sqlite3.OperationalError(message))
    if wrapper == "decorator":
        call = storage_module._retry_on_lock(fail)
    else:
        monkeypatch.setattr(store, "_ensure_open", lambda: Mock(execute=fail))

        def call(value):
            value._execute_with_retry("SELECT 1")
    with pytest.raises(sqlite3.OperationalError, match=message):
        call(store)
    assert fail.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("query", ["corn", "co_n_2024", "%", "corn_2024", "quote\"name"])
def test_list_runs_matches_exact_dataset_name(store, query):
    expected = None
    for name in ("corn_2024", "wheat", "quote\"name"):
        run_id = store.begin_run(name, {}, [{"name": name}])
        if name == query:
            expected = run_id
    rows = store.list_runs(dataset=query)
    assert ([row["run_id"] for row in rows.to_dicts()]) == ([expected] if expected else [])


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("route", ["store", "generator", "resolver"])
def test_portable_scripts_embed_and_replay_store_models(store, tmp_path, final, route):
    ids = _hierarchy(store, final=final)
    path = tmp_path / "model.n4a.py"
    generator = BundleGenerator(store.workspace_path, store=store)
    if route == "store":
        path = store.export_chain(ids["chain_id"], path, format="n4a.py")
    elif route == "generator":
        path = generator.export_from_chain(ids["chain_id"], path, fmt="n4a.py")
    else:
        prediction = store.query_predictions().to_dicts()[0]
        prediction["pipeline_uid"] = ids["pipeline_id"]
        path = generator.export(prediction, path, format="n4a.py")
    assert not zipfile.is_zipfile(path)
    script = runpy.run_path(str(path))
    np.testing.assert_allclose(script["predict"](ids["X"]), store.replay_chain(ids["chain_id"], ids["X"]))


@pytest.mark.parametrize("final", [False, True])
def test_resolver_script_preserves_uuid_fold_membership_and_final_priority(store, tmp_path, final):
    ids = _hierarchy(store)
    scaler = store.load_artifact(ids["shared_id"])
    model = LinearRegression().fit(scaler.transform(ids["X"]), np.arange(8) ** 2 + 10)
    other_id = store.save_artifact(model, "LinearRegression", "model", "joblib")
    fold = "fold_final" if final else "fold_1"
    store._ensure_open().execute("UPDATE chains SET fold_artifacts = ? WHERE chain_id = ?", [json.dumps({"fold_0": ids["model_id"], fold: other_id}), ids["chain_id"]])
    prediction = store.query_predictions().to_dicts()[0]
    prediction["pipeline_uid"] = ids["pipeline_id"]
    path = BundleGenerator(store.workspace_path, store=store).export(prediction, tmp_path / "folds.n4a.py", format="n4a.py")
    script = runpy.run_path(str(path))
    np.testing.assert_allclose(script["predict"](ids["X"]), store.replay_chain(ids["chain_id"], ids["X"]))


@pytest.mark.parametrize("route", ["store", "resolver"])
def test_portable_multisource_script_uses_source_metadata(store, tmp_path, route):
    ids = _hierarchy(store)
    X = ids["X"]
    second_X = X * 3 + np.array([1, 5, 13])
    scaler1 = store.load_artifact(ids["shared_id"])
    scaler2 = StandardScaler().fit(second_X)
    scaler2_id = store.save_artifact(scaler2, "StandardScaler", "transformer", "joblib")
    features = np.hstack([scaler1.transform(X), scaler2.transform(second_X)])
    model = LinearRegression().fit(features, np.arange(8) ** 2)
    model_id = store.save_artifact(model, "LinearRegression", "model", "joblib")
    conn = store._ensure_open()
    conn.execute("UPDATE chains SET fold_artifacts = ?, shared_artifacts = ? WHERE chain_id = ?", [
        json.dumps({"fold_0": model_id}), json.dumps({"1": [ids["shared_id"], scaler2_id], "_source_map": {"1": {"0": [ids["shared_id"]], "1": [scaler2_id]}}}), ids["chain_id"],
    ])
    generator = BundleGenerator(store.workspace_path, store=store)
    path = tmp_path / "sources.n4a.py"
    if route == "store":
        generator.export_from_chain(ids["chain_id"], path, fmt="n4a.py")
    else:
        prediction = store.query_predictions().to_dicts()[0]
        prediction["pipeline_uid"] = ids["pipeline_id"]
        generator.export(prediction, path, format="n4a.py")
    script = runpy.run_path(str(path))
    np.testing.assert_allclose(script["predict"]([X, second_X]), model.predict(features))
    with pytest.raises(ValueError, match="list of source arrays"):
        script["predict"](features)


def test_store_reexport_failure_preserves_previous_bundle(store, tmp_path):
    ids = _hierarchy(store)
    path = store.export_chain(ids["chain_id"], tmp_path / "model.n4a")
    original = path.read_bytes()
    expected = BundleLoader(path).predict(ids["X"])
    store.get_artifact_path(ids["shared_id"]).unlink()
    with pytest.raises(FileNotFoundError):
        store.export_chain(ids["chain_id"], path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
    np.testing.assert_allclose(BundleLoader(path).predict(ids["X"]), expected)


@pytest.mark.parametrize("route", ["single_model", "resolver"])
def test_zip_write_failure_preserves_previous_export(store, tmp_path, monkeypatch, route):
    ids = _hierarchy(store)
    path = tmp_path / "model.n4a"
    generator = BundleGenerator(store.workspace_path)
    resolved = ResolvedPrediction(model_step_index=1, artifact_provider=MapArtifactProvider({1: [("pipeline$hash:final", ids["model"])]}))

    def export():
        if route == "single_model":
            return write_single_model_bundle(ids["model"], path)
        return generator._export_n4a(resolved, path, True, True)

    export()
    original = path.read_bytes()
    write = zipfile.ZipFile.writestr

    def fail_after_manifest(self, name, *args, **kwargs):
        if name == "pipeline.json":
            raise OSError("injected disk failure")
        return write(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail_after_manifest)
    with pytest.raises(OSError, match="injected disk failure"):
        export()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_duplicate_transformer_zip_members_remain_individually_loadable(store, tmp_path):
    X = np.arange(18).reshape(6, 3)
    first = StandardScaler().fit(X)
    second = StandardScaler().fit(X + 10)
    resolved = ResolvedPrediction(artifact_provider=MapArtifactProvider({1: [("pipeline$one:all", first), ("pipeline$two:all", second)]}))
    path = BundleGenerator(store.workspace_path)._export_n4a(resolved, tmp_path / "unique.n4a", True, True)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.startswith("artifacts/")]
        assert len(names) == len(set(names)) == 2
        loaded = [joblib.load(io.BytesIO(archive.read(name))) for name in names]
        np.testing.assert_allclose(loaded[0].mean_, first.mean_)
        np.testing.assert_allclose(loaded[1].mean_, second.mean_)
    assert len(BundleLoader(path)._artifact_index) == 2


class _UnserializableArtifact:
    def __reduce__(self):
        raise ValueError("injected serialization failure")


def test_serialization_failure_aborts_export_and_preserves_bundle(store, tmp_path):
    ids = _hierarchy(store)
    generator = BundleGenerator(store.workspace_path)
    path = tmp_path / "model.n4a"
    good = ResolvedPrediction(model_step_index=1, artifact_provider=MapArtifactProvider({1: [("pipeline$good:final", ids["model"])]}))
    generator._export_n4a(good, path, True, True)
    original = path.read_bytes()
    broken = ResolvedPrediction(model_step_index=1, artifact_provider=MapArtifactProvider({1: [("pipeline$broken:final", _UnserializableArtifact())]}))
    with pytest.raises(RuntimeError, match=r"pipeline\$broken:final") as error:
        generator._export_n4a(broken, path, True, True)
    assert isinstance(error.value.__cause__, ValueError)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_export_chain_rejects_unknown_format_before_creating_file(store, tmp_path):
    ids = _hierarchy(store)
    path = tmp_path / "unexpected"
    with pytest.raises(ValueError, match="Unsupported bundle format"):
        store.export_chain(ids["chain_id"], path, format="unknown")
    assert not path.exists()

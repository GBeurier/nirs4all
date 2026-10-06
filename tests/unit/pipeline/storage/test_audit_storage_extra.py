"""Regression coverage for the second storage audit batch."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from nirs4all.cli.commands import artifacts as artifact_cli
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.types import ArtifactType
from nirs4all.pipeline.storage.library import PipelineLibrary
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
from nirs4all.pipeline.trace.recorder import TraceRecorder


def _register(registry, route, obj):
    if route == "chain":
        return registry.register_with_chain(obj, chain="s1.StandardScaler", step_index=1, artifact_type=ArtifactType.TRANSFORMER)
    return registry.register(obj, "pipeline:1:all", ArtifactType.TRANSFORMER)


@pytest.mark.parametrize("route", ["register", "chain", "deferred"])
@pytest.mark.parametrize("fresh_registry", [False, True])
def test_registration_repairs_truncated_content(tmp_path, route, fresh_registry):
    obj = StandardScaler().fit(np.arange(18).reshape(6, 3))
    registry = ArtifactRegistry(tmp_path, "A")
    record = _register(registry, "chain" if route == "chain" else "register", obj)
    path = registry.binaries_dir / record.path
    original = path.read_bytes()
    path.write_bytes(original[:len(original) // 2])
    if fresh_registry:
        registry = ArtifactRegistry(tmp_path, "A")
    if route == "deferred":
        registry.begin_deferred()
    repaired = _register(registry, "chain" if route == "chain" else "register", obj)
    if route == "deferred":
        registry.commit_deferred()
    assert repaired.path == record.path
    assert path.read_bytes() == original
    np.testing.assert_allclose(registry.load_artifact(repaired).transform(obj.mean_.reshape(1, -1)), 0)
    assert not list(path.parent.glob("*.tmp"))


def test_failed_atomic_artifact_write_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "artifact.pkl"
    path.write_bytes(b"previous complete content")
    monkeypatch.setattr("nirs4all.pipeline.storage.artifacts.artifact_registry.os.replace", Mock(side_effect=OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        ArtifactRegistry._write_artifact(path, b"replacement content")
    assert path.read_bytes() == b"previous complete content"
    assert not list(tmp_path.glob("*.tmp"))


def test_cleanup_protects_all_dataset_manifests_and_store_rows(tmp_path):
    with WorkspaceStore(tmp_path) as store:
        aid = store.save_artifact(StandardScaler(), "StandardScaler", "transformer", "joblib")
        stored_path = store.get_artifact_path(aid)
        paths = []
        for index, extension in enumerate(("yaml", "json", "yml")):
            blob = tmp_path / "artifacts" / f"d{index}" / "shared.pkl"
            blob.parent.mkdir()
            blob.write_bytes(b"live fitted artifact")
            manifest = tmp_path / "runs" / f"dataset_{index}" / f"manifest.{extension}"
            manifest.parent.mkdir(parents=True)
            relative = blob.relative_to(tmp_path / "artifacts").as_posix()
            manifest.write_text(json.dumps({"artifacts": {"items": [{"path": relative}]}}))
            paths.append(blob)
        orphan = tmp_path / "artifacts" / "ee" / "orphan.pkl"
        orphan.parent.mkdir()
        orphan.write_bytes(b"unreferenced")
        registry = ArtifactRegistry(tmp_path, "unrelated_dataset")
        assert registry.find_orphaned_artifacts() == ["ee/orphan.pkl"]
        deleted, _ = registry.delete_orphaned_artifacts(dry_run=False)
        assert deleted == ["ee/orphan.pkl"]
        assert not orphan.exists()
        assert stored_path.exists()
        assert all(path.exists() for path in paths)


def test_store_gc_preserves_artifacts_referenced_by_other_dataset_manifest(tmp_path):
    with WorkspaceStore(tmp_path) as store:
        aid = store.save_artifact(StandardScaler(), "StandardScaler", "transformer", "joblib")
        store._ensure_open().execute("UPDATE artifacts SET ref_count = 0 WHERE artifact_id = ?", [aid])
        blob = store.get_artifact_path(aid)
        manifest = tmp_path / "runs" / "other_dataset" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"artifacts": [{"path": blob.relative_to(tmp_path / "artifacts").as_posix()}]}))
        assert store.gc_artifacts() == 0
        assert store.get_artifact_path(aid).exists()
        manifest.unlink()
        assert store.gc_artifacts() == 1
        assert not blob.exists()


def test_invalid_manifest_blocks_destructive_cleanup(tmp_path):
    registry = ArtifactRegistry(tmp_path, "A")
    registry.binaries_dir.mkdir()
    orphan = registry.binaries_dir / "orphan.pkl"
    orphan.write_bytes(b"preserve until references can be checked")
    (tmp_path / "manifest.json").write_text("{invalid")
    with pytest.raises(ValueError):
        registry.delete_orphaned_artifacts(dry_run=False)
    assert orphan.exists()


def test_purge_preserves_other_dataset_shared_and_unknown_blobs(tmp_path):
    registry = ArtifactRegistry(tmp_path, "A")
    record = registry.register(StandardScaler(), "A:1:all", ArtifactType.TRANSFORMER)
    shared = registry.binaries_dir / record.path
    manifest = tmp_path / "runs" / "B" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"artifacts": [{"path": record.path}]}))
    unknown = registry.binaries_dir / "stray.pkl"
    unknown.write_bytes(b"unknown ownership")
    assert registry.purge_dataset_artifacts(confirm=True) == (0, 0)
    artifact_cli.artifacts_purge(Namespace(workspace=str(tmp_path), dataset="A", force=True, yes=True))
    assert shared.exists() and unknown.exists()
    assert registry.resolve(record.artifact_id) is not None


def test_artifact_cli_stats_and_cleanup_use_shared_shards(tmp_path, monkeypatch):
    blob = tmp_path / "artifacts" / "aa" / "orphan.pkl"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"orphan")
    log = Mock()
    monkeypatch.setattr(artifact_cli, "logger", log)
    args = Namespace(workspace=str(tmp_path), dataset=None, force=False, verbose=True)
    artifact_cli.artifacts_stats(args)
    assert any("Files on disk: 1" in str(call) for call in log.info.call_args_list)
    artifact_cli.artifacts_cleanup(args)
    assert blob.exists()
    args.dataset = "A"
    args.force = True
    artifact_cli.artifacts_cleanup(args)
    assert blob.exists()
    args.dataset = None
    artifact_cli.artifacts_cleanup(args)
    assert not blob.exists()


@pytest.mark.parametrize("component", ["", ".", "..", "../outside", "/outside", "a\\b", "\0"])
@pytest.mark.parametrize("operation", ["save", "load", "delete", "import"])
def test_library_rejects_unsafe_names_and_categories(tmp_path, component, operation):
    library = PipelineLibrary(tmp_path)
    keep = library.save_template({"steps": []}, "Keep Model")
    other = library.save_template({}, "Other Model", category="other")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "metadata.json").write_text(json.dumps({"name": "imported"}))
    for name, category in ((component, "general"), ("keep_model", component)):
        with pytest.raises(ValueError):
            if operation == "save":
                library.save_template({}, name, category=category)
            elif operation == "import":
                (incoming / "metadata.json").write_text(json.dumps({"name": name}))
                library.import_template(incoming, category=category)
            elif operation == "load":
                library.load_template(name, category=category)
            else:
                library.delete_template(name, category=category)
    assert keep.is_dir() and other.is_dir()
    assert library.load_template("Keep Model") == {"steps": []}


@pytest.mark.parametrize("symlink_level", ["category", "template"])
def test_library_rejects_symlinks_outside_library(tmp_path, symlink_level):
    library = PipelineLibrary(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "metadata.json"
    marker.write_text(json.dumps({"name": "private"}))
    if symlink_level == "category":
        (library.library_path / "general").symlink_to(outside, target_is_directory=True)
    else:
        (library.library_path / "general").mkdir()
        (library.library_path / "general" / "private").symlink_to(outside, target_is_directory=True)
    for action in (lambda: library.delete_template("private", "general"),
                   lambda: library.save_template({}, "private", overwrite=True),
                   lambda: library.list_templates("general")):
        with pytest.raises(ValueError, match="escapes library"):
            action()
    assert marker.exists()


@pytest.mark.parametrize("uid,provided,expected", [("", "provided", "provided"), ("abc_run", "", "abc"),
                                                   ("abc_run", "provided", "provided"), ("", "", "")])
def test_trace_identity_survives_empty_uid(uid, provided, expected):
    recorder = TraceRecorder(pipeline_uid=uid, pipeline_id=provided)
    assert recorder.pipeline_id == expected
    assert recorder.current_chain().pipeline_id == expected


def _scope_predictions(store, refit_via_context=False):
    run = store.begin_run("scopes", config={}, datasets=[])
    pipeline = store.begin_pipeline(run, "scopes", [], [], "dataset", "hash")
    chains, ids = [], {}
    for model, cv_score, final_score in (("A", 0.1, 0.9), ("B", 0.4, 0.2), ("CVOnly", 0.3, None)):
        chain = store.save_chain(pipeline, [{"step_idx": 0, "operator_class": model}], 0, model, "", "per_fold", {}, {})
        chains.append(chain)
        for fold, score in (("fold_0", cv_score), ("fold_final", final_score), ("fold_avg", 999)):
            if score is None:
                continue
            persisted_fold = "uuid-refit-fold" if fold == "fold_final" and refit_via_context else fold
            context = "{\"refit\": true}" if fold == "fold_final" and refit_via_context else None
            pid = store.save_prediction(pipeline, chain, "dataset", model, model, persisted_fold, "test",
                                        score, score, score, "rmse", "regression", 3, 2, {}, {}, None, None, 0, 0, refit_context=context)
            ids[(chain, fold)] = pid
        store.update_chain_summary(chain)
    return chains, ids


@pytest.mark.parametrize("refit_via_context", [False, True])
def test_score_scopes_change_membership_aggregates_and_ranked_pagination(tmp_path, refit_via_context):
    with WorkspaceStore(tmp_path) as store:
        chains, ids = _scope_predictions(store, refit_via_context)
        cv = store.query_aggregated_predictions(score_scope="cv")
        final = store.query_aggregated_predictions(score_scope="final")
        all_rows = store.query_aggregated_predictions(score_scope="all")
        assert cv.height == all_rows.height == 3 and final.height == 2
        assert set(final["chain_id"]) == set(chains[:2])
        for scope, frame, count, expected in (("cv", cv, 1, .1), ("final", final, 1, .9), ("all", all_rows, 2, .5)):
            row = frame.filter(frame["chain_id"] == chains[0]).row(0, named=True)
            assert row["score_scope"] == scope
            assert row["avg_test_score"] == pytest.approx(expected)
            assert row["prediction_count"] == row["fold_count"] == count
            assert ids[(chains[0], "fold_avg")] not in json.loads(row["prediction_ids"])
        assert store.query_top_aggregated_predictions("rmse", n=1, score_scope="cv")["chain_id"][0] == chains[0]
        assert store.query_top_aggregated_predictions("rmse", n=1, score_scope="final")["chain_id"][0] == chains[1]
        assert store.query_top_aggregated_predictions("rmse", n=1, offset=1, score_scope="final")["chain_id"][0] == chains[0]
        assert store.query_top_aggregated_predictions("rmse", n=1, score_scope="all")["chain_id"][0] == chains[2]
        for query in (store.query_aggregated_predictions, lambda **kw: store.query_top_aggregated_predictions("rmse", **kw)):
            with pytest.raises(ValueError, match="score_scope"):
                query(score_scope="typo")


def test_full_trained_jax_artifact_roundtrip(tmp_path, record_property):
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    from nirs4all.controllers.models.jax_model import JaxModelController
    from nirs4all.operators.models.jax.generic import JaxMLPRegressor

    x = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
    y = (x[:, :1] * 2).astype(np.float32)
    trained = JaxModelController()._train_model(JaxMLPRegressor(features=(4,)), x, y, epochs=2, batch_size=4)
    expected = trained.predict(x)
    assert int(trained.state.step) == 4
    registry = ArtifactRegistry(tmp_path, "jax_dataset")
    record = registry.register(trained, "jax:1:all", ArtifactType.MODEL)
    assert record.format == "cloudpickle"
    restored = ArtifactRegistry(tmp_path, "jax_dataset").load_artifact(record)
    assert type(restored) is type(trained)
    assert restored is not trained and restored.state is not trained.state
    assert type(restored.model) is type(trained.model)
    assert int(restored.state.step) == int(trained.state.step)
    before = jax.tree.leaves((trained.state.params, trained.state.opt_state, trained.state.batch_stats))
    after = jax.tree.leaves((restored.state.params, restored.state.opt_state, restored.state.batch_stats))
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    actual = restored.predict(x)
    record_property("serialization_format", record.format)
    record_property("artifact_bytes", (registry.binaries_dir / record.path).stat().st_size)
    record_property("training_steps", int(restored.state.step))
    record_property("max_prediction_absolute_error", float(np.max(np.abs(actual - expected))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

"""Tests for store-based chain export via BundleGenerator.

Verifies that BundleGenerator.export_from_chain() correctly delegates
to WorkspaceStore.export_chain() for .n4a format export.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nirs4all.pipeline.bundle.generator import BundleGenerator
from nirs4all.pipeline.dagml.host_container import BUNDLE_MANIFEST_MEMBER, JOBLIB_CONTENT_TYPE, open_n4a_source
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


class TestExportFromChain:
    """Tests for BundleGenerator.export_from_chain()."""

    def test_requires_store(self, tmp_path):
        """export_from_chain should raise if no store is set."""
        gen = BundleGenerator(workspace_path=tmp_path)
        with pytest.raises(RuntimeError, match="store"):
            gen.export_from_chain("abc123", tmp_path / "out.n4a")

    def test_delegates_to_store_for_n4a(self, tmp_path):
        """export_from_chain should call store.export_chain for .n4a format."""
        store = MagicMock()
        expected_path = tmp_path / "model.n4a"
        store.export_chain.return_value = expected_path

        gen = BundleGenerator(workspace_path=tmp_path, store=store)
        result = gen.export_from_chain("chain_abc", expected_path, fmt="n4a")

        store.export_chain.assert_called_once_with("chain_abc", expected_path, format="n4a")
        assert result == expected_path

    def test_default_format_is_n4a(self, tmp_path):
        """export_from_chain should default to n4a format."""
        store = MagicMock()
        store.export_chain.return_value = tmp_path / "out.n4a"

        gen = BundleGenerator(workspace_path=tmp_path, store=store)
        gen.export_from_chain("chain_xyz", tmp_path / "out.n4a")

        store.export_chain.assert_called_once()


@pytest.fixture
def real_core():
    core = pytest.importorskip("nirs4all_core")
    if not callable(getattr(core, "open_n4a_file", None)) or not callable(getattr(core, "write_n4a_file", None)):
        pytest.skip("installed nirs4all-core lacks the file-backed .n4a container API")
    return core


def _chain_store(tmp_path, *, stale: str | None = None):
    """A store double exposing exactly what ``WorkspaceStore.export_chain`` reads; 12 shared sub-artifacts force the sub10/sub2 order trap."""
    shared = [f"shared-{index}" for index in range(12)]
    payloads = {artifact_id: f"fitted state of {artifact_id}".encode() for artifact_id in (*shared, "model-final", "model-fold0")}
    paths = {}
    for artifact_id, payload in payloads.items():
        paths[artifact_id] = tmp_path / "store" / f"{artifact_id}.joblib"
        paths[artifact_id].parent.mkdir(exist_ok=True)
        paths[artifact_id].write_bytes(payload)
    declared = {artifact_id: "sha256:" + hashlib.sha256(payload).hexdigest() for artifact_id, payload in payloads.items()}
    if stale is not None:
        declared[stale] = "sha256:" + "0" * 64
    chain = {
        "model_class": "PLSRegression", "model_step_idx": 3, "preprocessings": "SNV", "fold_strategy": "kfold", "pipeline_id": "pipe",
        "steps": [{"step_idx": 1}], "fold_artifacts": {"fold_0": "model-fold0", "fold_final": "model-final"}, "shared_artifacts": {"1": shared},
    }
    store = SimpleNamespace(
        _validate_chain_export=lambda chain_id: None, get_chain=lambda chain_id: chain, get_pipeline=lambda pipeline_id: {"expanded_config": {"steps": ["snv"]}},
        get_artifact_path=lambda artifact_id: paths[artifact_id], _artifact_declared_fingerprint=lambda artifact_id: declared[artifact_id],
    )
    return store, shared, payloads, paths


class TestExportChainHostContainer:
    """``WorkspaceStore.export_chain`` writes a Core host container from store file paths."""

    def test_declared_order_paths_and_bytes(self, tmp_path, real_core):
        store, shared, payloads, _ = _chain_store(tmp_path)
        out = WorkspaceStore.export_chain(store, "chain-1", tmp_path / "chain.n4a")

        expected = [BUNDLE_MANIFEST_MEMBER, "pipeline.json", "chain.json", *[f"artifacts/step_1_sub{index}.joblib" for index in range(12)], "artifacts/step_3_foldfinal.joblib"]
        with open_n4a_source(out) as source:
            container = source.require_host()
            assert container.namelist() == expected
            assert container.namelist() != sorted(container.namelist()), "declared order is not the physical sorted order"
            for index, artifact_id in enumerate(shared):
                assert container.read(f"artifacts/step_1_sub{index}.joblib") == payloads[artifact_id]
            assert container.read("artifacts/step_3_foldfinal.joblib") == payloads["model-final"]
            assert container.artifact("artifacts/step_3_foldfinal.joblib")["content_type"] == {"id": JOBLIB_CONTENT_TYPE, "version": 1}
            manifest = container.bundle_manifest
            assert manifest is not None and manifest["chain_id"] == "chain-1" and manifest["has_refit"] is True
            assert manifest["fold_strategy"] == "single_refit" and manifest["model_step_index"] == 3
            assert container.read_json("chain.json")["fold_artifacts"] == {"fold_final": "model-final"}
            assert container.read_json("pipeline.json") == {"steps": ["snv"]}

    def test_a_stale_store_hash_is_refused_before_publication_and_keeps_the_destination(self, tmp_path, real_core):
        store, *_ = _chain_store(tmp_path, stale="shared-3")
        destination = tmp_path / "chain.n4a"
        destination.write_bytes(b"previous valid destination")
        before = frozenset(path.name for path in tmp_path.iterdir())
        with pytest.raises(ValueError, match="fingerprint"):
            WorkspaceStore.export_chain(store, "chain-1", destination)
        assert destination.read_bytes() == b"previous valid destination"
        after = frozenset(path.name for path in tmp_path.iterdir())
        assert after == before and {"chain.n4a", "store"} <= before, "the refused export left no staging entry and removed nothing"

    def test_a_missing_artifact_file_is_refused_and_nothing_is_published(self, tmp_path, real_core):
        store, _, _, paths = _chain_store(tmp_path)
        paths["shared-5"].unlink()
        destination = tmp_path / "chain.n4a"
        with pytest.raises(FileNotFoundError, match="Artifact file missing"):
            WorkspaceStore.export_chain(store, "chain-1", destination)
        assert not destination.exists()

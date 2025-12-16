"""
Unit tests for manifest v2 schema support.

Tests the v2 schema format in ManifestManager:
- Schema version detection
- v2 artifact section handling
- Migration from v1 to v2
"""

import pytest
from pathlib import Path
import yaml

from nirs4all.pipeline.storage.manifest_manager import (
    ManifestManager,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SCHEMA_V2,
    CURRENT_MANIFEST_SCHEMA,
)


class TestSchemaVersionDetection:
    """Tests for schema version detection."""

    def test_detect_v1_manifest(self):
        """Test detecting v1 manifest from flat artifacts list."""
        manifest = {
            "uid": "test",
            "pipeline_id": "0001_test",
            "artifacts": [
                {"step": 0, "name": "scaler"}
            ]
        }
        version = ManifestManager.get_schema_version(manifest)
        assert version == MANIFEST_SCHEMA_V1

    def test_detect_v2_manifest_from_root(self):
        """Test detecting v2 manifest from root schema_version."""
        manifest = {
            "schema_version": "2.0",
            "uid": "test",
            "artifacts": {
                "schema_version": "2.0",
                "items": []
            }
        }
        version = ManifestManager.get_schema_version(manifest)
        assert version == MANIFEST_SCHEMA_V2

    def test_detect_v2_manifest_from_artifacts(self):
        """Test detecting v2 manifest from artifacts section."""
        manifest = {
            "uid": "test",
            "artifacts": {
                "schema_version": "2.0",
                "items": []
            }
        }
        version = ManifestManager.get_schema_version(manifest)
        assert version == MANIFEST_SCHEMA_V2

    def test_is_v2_artifacts(self):
        """Test _is_v2_artifacts helper."""
        # v2 format
        assert ManifestManager._is_v2_artifacts({"items": []}) is True
        assert ManifestManager._is_v2_artifacts(
            {"schema_version": "2.0", "items": []}
        ) is True

        # v1 format
        assert ManifestManager._is_v2_artifacts([]) is False
        assert ManifestManager._is_v2_artifacts([{"step": 0}]) is False

        # Invalid
        assert ManifestManager._is_v2_artifacts({}) is False
        assert ManifestManager._is_v2_artifacts(None) is False


class TestV2ArtifactSection:
    """Tests for v2 artifact section handling."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create temporary results directory."""
        results_dir = tmp_path / "runs" / "test_dataset"
        results_dir.mkdir(parents=True)
        return results_dir

    @pytest.fixture
    def manager(self, temp_results_dir):
        """Create ManifestManager for testing."""
        return ManifestManager(temp_results_dir)

    def test_create_pipeline_uses_v2(self, manager):
        """Test that create_pipeline creates v2 manifests."""
        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={"steps": []},
            pipeline_hash="abc123"
        )

        manifest = manager.load_manifest(pipeline_id)

        assert manifest.get("schema_version") == CURRENT_MANIFEST_SCHEMA
        assert isinstance(manifest["artifacts"], dict)
        assert "items" in manifest["artifacts"]
        assert manifest["artifacts"]["schema_version"] == CURRENT_MANIFEST_SCHEMA

    def test_append_artifacts_v1_format(self, manager):
        """Test appending to v1 format artifacts (backward compat)."""
        # Create pipeline with v1 format by modifying after creation
        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={},
            pipeline_hash="abc123"
        )

        # Downgrade to v1 format manually
        manifest = manager.load_manifest(pipeline_id)
        manifest["artifacts"] = []  # v1: flat list
        manager.save_manifest(pipeline_id, manifest)

        # Append should work with v1 format
        manager.append_artifacts(pipeline_id, [{"step": 0, "name": "scaler"}])

        manifest = manager.load_manifest(pipeline_id)
        assert len(manifest["artifacts"]) == 1
        assert manifest["artifacts"][0]["name"] == "scaler"

    def test_append_artifacts_v2_format(self, manager):
        """Test appending to v2 format artifacts."""
        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={},
            pipeline_hash="abc123"
        )

        manager.append_artifacts(pipeline_id, [{"step": 0, "name": "scaler"}])

        manifest = manager.load_manifest(pipeline_id)
        assert len(manifest["artifacts"]["items"]) == 1
        assert manifest["artifacts"]["items"][0]["name"] == "scaler"

    def test_get_artifacts_list_v1(self, manager):
        """Test get_artifacts_list with v1 manifest."""
        manifest = {
            "artifacts": [
                {"step": 0, "name": "scaler"},
                {"step": 1, "name": "model"}
            ]
        }

        artifacts = manager.get_artifacts_list(manifest)

        assert len(artifacts) == 2
        assert artifacts[0]["name"] == "scaler"

    def test_get_artifacts_list_v2(self, manager):
        """Test get_artifacts_list with v2 manifest."""
        manifest = {
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {"artifact_id": "0001:0:all", "class_name": "StandardScaler"},
                    {"artifact_id": "0001:1:all", "class_name": "PLSRegression"}
                ]
            }
        }

        artifacts = manager.get_artifacts_list(manifest)

        assert len(artifacts) == 2
        assert artifacts[0]["class_name"] == "StandardScaler"


class TestConvertToV2:
    """Tests for converting manifests to v2 format."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create temporary results directory."""
        results_dir = tmp_path / "runs" / "test_dataset"
        results_dir.mkdir(parents=True)
        return results_dir

    @pytest.fixture
    def manager(self, temp_results_dir):
        """Create ManifestManager for testing."""
        return ManifestManager(temp_results_dir)

    def test_convert_artifacts_to_v2(self):
        """Test converting v1 artifacts to v2 format."""
        v1_artifacts = [
            {"step": 0, "name": "scaler"},
            {"step": 1, "name": "model"}
        ]

        v2_section = ManifestManager._convert_artifacts_to_v2(v1_artifacts)

        assert v2_section["schema_version"] == MANIFEST_SCHEMA_V2
        assert len(v2_section["items"]) == 2
        assert v2_section["items"][0]["name"] == "scaler"

    def test_upgrade_manifest_to_v2(self, manager):
        """Test upgrading a v1 manifest to v2."""
        # Create a v1 manifest manually
        pipeline_id = "0001_test"
        pipeline_dir = manager.results_dir / pipeline_id
        pipeline_dir.mkdir(parents=True)

        v1_manifest = {
            "uid": "test-uid",
            "pipeline_id": pipeline_id,
            "name": "test",
            "version": "1.0",
            "artifacts": [
                {"step": 0, "name": "scaler"}
            ],
            "predictions": []
        }

        manifest_path = pipeline_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(v1_manifest, f)

        # Upgrade
        manager.upgrade_manifest_to_v2(pipeline_id)

        # Verify
        manifest = manager.load_manifest(pipeline_id)
        assert manifest["schema_version"] == MANIFEST_SCHEMA_V2
        assert "version" not in manifest  # Old field removed
        assert isinstance(manifest["artifacts"], dict)
        assert len(manifest["artifacts"]["items"]) == 1

    def test_upgrade_already_v2_is_noop(self, manager):
        """Test that upgrading a v2 manifest is a no-op."""
        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={},
            pipeline_hash="abc123"
        )

        # Get original
        original = manager.load_manifest(pipeline_id)

        # Upgrade (should be no-op)
        manager.upgrade_manifest_to_v2(pipeline_id)

        # Should be unchanged
        upgraded = manager.load_manifest(pipeline_id)
        assert upgraded == original


class TestAppendArtifactsV2:
    """Tests for append_artifacts_v2 method."""

    @pytest.fixture
    def temp_results_dir(self, tmp_path):
        """Create temporary results directory."""
        results_dir = tmp_path / "runs" / "test_dataset"
        results_dir.mkdir(parents=True)
        return results_dir

    @pytest.fixture
    def manager(self, temp_results_dir):
        """Create ManifestManager for testing."""
        return ManifestManager(temp_results_dir)

    def test_append_artifact_records(self, manager):
        """Test appending ArtifactRecord instances."""
        from nirs4all.pipeline.storage.artifacts.types import (
            ArtifactRecord,
            ArtifactType
        )

        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={},
            pipeline_hash="abc123"
        )

        records = [
            ArtifactRecord(
                artifact_id="0001:0:all",
                content_hash="sha256:abc123",
                path="transformer_StandardScaler_abc123.pkl",
                pipeline_id=pipeline_id,
                artifact_type=ArtifactType.TRANSFORMER,
                class_name="StandardScaler"
            )
        ]

        manager.append_artifacts_v2(pipeline_id, records)

        manifest = manager.load_manifest(pipeline_id)
        items = manifest["artifacts"]["items"]

        assert len(items) == 1
        assert items[0]["artifact_id"] == "0001:0:all"
        assert items[0]["artifact_type"] == "transformer"

    def test_append_dict_records(self, manager):
        """Test appending dict records (already serialized)."""
        pipeline_id, _ = manager.create_pipeline(
            name="test",
            dataset="test_dataset",
            pipeline_config={},
            pipeline_hash="abc123"
        )

        records = [
            {
                "artifact_id": "0001:0:all",
                "content_hash": "sha256:abc123",
                "path": "transformer_StandardScaler_abc123.pkl",
                "artifact_type": "transformer"
            }
        ]

        manager.append_artifacts_v2(pipeline_id, records)

        manifest = manager.load_manifest(pipeline_id)
        items = manifest["artifacts"]["items"]

        assert len(items) == 1
        assert items[0]["artifact_id"] == "0001:0:all"

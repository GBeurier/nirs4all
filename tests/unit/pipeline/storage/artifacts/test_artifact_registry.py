"""
Unit tests for ArtifactRegistry.

Tests the central registry for artifact management including:
- ID generation
- Deduplication
- Dependency graph
- Cleanup utilities
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nirs4all.pipeline.storage.artifacts.artifact_registry as artifact_registry_module
from nirs4all.pipeline.storage.artifacts import generate_artifact_id_v3
from nirs4all.pipeline.storage.artifacts.artifact_registry import (
    ArtifactRegistry,
    DependencyGraph,
)
from nirs4all.pipeline.storage.artifacts.types import (
    ArtifactRecord,
    ArtifactType,
    MetaModelConfig,
)


def make_v3_id(pipeline_id: str, step: int, fold_id=None, operator: str = "Model", branch_path=None):
    """Helper to generate V3 artifact IDs for tests."""
    branch_str = ""
    if branch_path:
        branch_str = f"[br={','.join(map(str, branch_path))}]"
    chain_path = f"s{step}.{operator}{branch_str}"
    return generate_artifact_id_v3(pipeline_id, chain_path, fold_id)

class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    def test_add_dependency(self):
        """Test adding a single dependency."""
        graph = DependencyGraph()
        graph.add_dependency("model", "scaler")

        assert graph.get_dependencies("model") == ["scaler"]
        assert graph.get_dependents("scaler") == ["model"]

    def test_add_multiple_dependencies(self):
        """Test adding multiple dependencies."""
        graph = DependencyGraph()
        graph.add_dependencies("meta_model", ["model1", "model2"])

        deps = graph.get_dependencies("meta_model")
        assert "model1" in deps
        assert "model2" in deps

    def test_get_dependencies_empty(self):
        """Test getting dependencies for unknown artifact."""
        graph = DependencyGraph()
        assert graph.get_dependencies("unknown") == []

    def test_resolve_dependencies_simple(self):
        """Test resolving simple dependency chain."""
        graph = DependencyGraph()
        graph.add_dependency("model", "scaler")
        graph.add_dependency("scaler", "splitter")

        deps = graph.resolve_dependencies("model")
        # Should return in topological order: splitter, scaler
        assert deps == ["splitter", "scaler"]

    def test_resolve_dependencies_diamond(self):
        """Test resolving diamond-shaped dependencies."""
        graph = DependencyGraph()
        # meta -> model1 -> scaler
        # meta -> model2 -> scaler
        graph.add_dependencies("meta", ["model1", "model2"])
        graph.add_dependency("model1", "scaler")
        graph.add_dependency("model2", "scaler")

        deps = graph.resolve_dependencies("meta")
        # scaler should appear once, before both models
        assert deps.count("scaler") == 1
        assert deps.index("scaler") < deps.index("model1")
        assert deps.index("scaler") < deps.index("model2")

    def test_resolve_dependencies_cycle_detection(self):
        """Test that cycles are detected."""
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("b", "c")
        graph.add_dependency("c", "a")  # Creates cycle

        with pytest.raises(ValueError, match="Cycle detected"):
            graph.resolve_dependencies("a")

    def test_resolve_dependencies_max_depth(self):
        """Test max depth limit."""
        graph = DependencyGraph()

        # Create a very deep chain
        for i in range(150):
            graph.add_dependency(f"node_{i}", f"node_{i+1}")

        with pytest.raises(ValueError, match="Maximum dependency depth"):
            graph.resolve_dependencies("node_0", max_depth=100)

    def test_remove_artifact(self):
        """Test removing an artifact from graph."""
        graph = DependencyGraph()
        graph.add_dependency("model", "scaler")
        graph.add_dependency("meta", "model")

        graph.remove_artifact("model")

        assert graph.get_dependencies("meta") == []
        assert graph.get_dependents("scaler") == []

    def test_clear(self):
        """Test clearing the graph."""
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("c", "d")

        graph.clear()

        assert graph.get_dependencies("a") == []
        assert graph.get_dependencies("c") == []

class TestArtifactRegistry:
    """Tests for ArtifactRegistry class."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def registry(self, temp_workspace):
        """Create a registry for testing."""
        return ArtifactRegistry(
            workspace=temp_workspace,
            dataset="test_dataset"
        )

    def test_initialization(self, registry, temp_workspace):
        """Test registry initialization."""
        assert registry.workspace == temp_workspace
        assert registry.dataset == "test_dataset"
        # binaries_dir is created lazily when first artifact is saved
        # V3 uses workspace/artifacts (shared across datasets, content-addressed)
        assert registry.binaries_dir == temp_workspace / "artifacts"

    def test_generate_id(self, registry):
        """Test ID generation with V3 chain-based format."""
        chain_path = "s3.PLS[br=0]"
        artifact_id = registry.generate_id(
            chain=chain_path,
            fold_id=0,
            pipeline_id="0001_pls"
        )
        # V3 format: {pipeline_id}${chain_hash}:{fold_id}
        assert artifact_id.startswith("0001_pls$")
        assert artifact_id.endswith(":0")
        assert "$" in artifact_id

    def test_register_artifact(self, registry):
        """Test registering an artifact."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        record = registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        assert record.artifact_id == "0001:0:all"
        assert record.class_name == "StandardScaler"
        assert record.artifact_type == ArtifactType.TRANSFORMER
        assert record.content_hash.startswith("sha256:")

    def test_register_with_dependencies(self, registry):
        """Test registering artifact with dependencies."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        record = registry.register(
            obj=scaler,
            artifact_id="0001:1:all",
            artifact_type=ArtifactType.TRANSFORMER,
            depends_on=["0001:0:all"]
        )

        deps = registry.get_dependencies("0001:1:all")
        assert deps == ["0001:0:all"]

    def test_deduplication(self, registry):
        """Test that duplicate content is deduplicated."""
        from sklearn.preprocessing import StandardScaler

        # Register same scaler twice with different IDs
        scaler = StandardScaler()

        record1 = registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        record2 = registry.register(
            obj=scaler,
            artifact_id="0002:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Should have same content hash and path
        assert record1.content_hash == record2.content_hash
        assert record1.path == record2.path

        # Should only have one file
        files = list(registry.binaries_dir.glob("*"))
        assert len(files) == 1

    def test_find_existing_by_hash_verifies_full_content(self, registry, monkeypatch):
        """Short-hash matches must be validated against full content hash."""
        target_content = b"target-bytes"
        target_hash = f"sha256:{hashlib.sha256(target_content).hexdigest()}"
        shard = target_hash[7:9]
        shard_dir = registry.binaries_dir / shard
        shard_dir.mkdir(parents=True, exist_ok=True)

        forced_short_hash = "deadbeefcafe"
        monkeypatch.setattr(
            artifact_registry_module,
            "get_short_hash",
            lambda *_args, **_kwargs: forced_short_hash,
        )

        # Wrong content with matching forced short-hash pattern must be rejected.
        wrong_file = shard_dir / f"model_Wrong_{forced_short_hash}.joblib"
        wrong_file.write_bytes(b"different-content")
        assert registry._find_existing_by_hash(target_hash) is None

        # Correct content should then be accepted.
        good_file = shard_dir / f"model_Good_{forced_short_hash}.joblib"
        good_file.write_bytes(target_content)
        assert registry._find_existing_by_hash(target_hash) == f"{shard}/{good_file.name}"

    def test_resolve(self, registry):
        """Test resolving artifact by ID."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        record = registry.resolve("0001:0:all")
        assert record is not None
        assert record.class_name == "StandardScaler"

    def test_resolve_unknown(self, registry):
        """Test resolving unknown ID returns None."""
        assert registry.resolve("unknown") is None

    def test_load_artifact(self, registry):
        """Test loading artifact from disk."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit([[1, 2], [3, 4]])  # Fit so it has state

        record = registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        loaded = registry.load_artifact(record)
        assert isinstance(loaded, StandardScaler)
        assert hasattr(loaded, 'mean_')

    def test_get_artifacts_for_step(self, registry):
        """Test getting artifacts for a step."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        chain0 = "s0.StandardScaler"
        chain1 = "s1.MinMaxScaler"
        registry.register(
            obj=StandardScaler(),
            artifact_id=make_v3_id("0001", 0, None, "StandardScaler"),
            artifact_type=ArtifactType.TRANSFORMER,
            chain_path=chain0
        )
        registry.register(
            obj=MinMaxScaler(),
            artifact_id=make_v3_id("0001", 1, None, "MinMaxScaler"),
            artifact_type=ArtifactType.TRANSFORMER,
            chain_path=chain1
        )

        step0_artifacts = registry.get_artifacts_for_step(
            pipeline_id="0001",
            step_index=0
        )
        assert len(step0_artifacts) == 1
        assert step0_artifacts[0].class_name == "StandardScaler"

    def test_get_fold_models(self, registry):
        """Test getting fold-specific models."""
        from sklearn.linear_model import LinearRegression

        # Register models for different folds
        for fold_id in range(3):
            chain_path = "s3.LinearRegression"
            registry.register(
                obj=LinearRegression(),
                artifact_id=make_v3_id("0001", 3, fold_id, "LinearRegression"),
                artifact_type=ArtifactType.MODEL,
                chain_path=chain_path
            )

        fold_models = registry.get_fold_models(
            pipeline_id="0001",
            step_index=3
        )

        assert len(fold_models) == 3
        assert [m.fold_id for m in fold_models] == [0, 1, 2]

    def test_export_to_manifest(self, registry):
        """Test exporting registry to manifest format."""
        from sklearn.preprocessing import StandardScaler

        chain_path = "s0.StandardScaler"
        artifact_id = make_v3_id("0001", 0, None, "StandardScaler")
        registry.register(
            obj=StandardScaler(),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.TRANSFORMER,
            chain_path=chain_path
        )

        manifest_section = registry.export_to_manifest()

        assert manifest_section["schema_version"] == "3.0"
        assert len(manifest_section["items"]) == 1
        # V3 format artifact ID
        assert manifest_section["items"][0]["artifact_id"] == artifact_id

    def test_import_from_manifest(self, registry):
        """Test importing from manifest format."""
        manifest = {
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:all",
                        "content_hash": "sha256:abc123",
                        "path": "transformer_StandardScaler_abc123.pkl",
                        "pipeline_id": "0001",
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "depends_on": []
                    }
                ]
            }
        }

        registry.import_from_manifest(manifest, registry.binaries_dir)

        record = registry.resolve("0001:0:all")
        assert record is not None
        assert record.class_name == "StandardScaler"

    def test_find_orphaned_artifacts(self, registry):
        """Test finding orphaned artifacts."""
        from sklearn.preprocessing import StandardScaler

        # Register an artifact
        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Create an orphaned file directly
        orphan_path = registry.binaries_dir / "orphan_file.pkl"
        orphan_path.write_bytes(b"orphan data")

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=False)
        assert "orphan_file.pkl" in orphans

    def test_delete_orphaned_artifacts_dry_run(self, registry):
        """Test deleting orphaned artifacts in dry run mode."""
        # Create binaries dir first (normally created when saving artifacts)
        registry.binaries_dir.mkdir(parents=True, exist_ok=True)
        # Create an orphaned file
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan")

        deleted, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=True,
            scan_all_manifests=False
        )

        assert "orphan.pkl" in deleted
        assert bytes_freed == 6
        assert orphan_path.exists()  # Still exists in dry run

    def test_delete_orphaned_artifacts(self, registry):
        """Test actually deleting orphaned artifacts."""
        # Create binaries dir first (normally created when saving artifacts)
        registry.binaries_dir.mkdir(parents=True, exist_ok=True)
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan")

        deleted, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=False,
            scan_all_manifests=False
        )

        assert "orphan.pkl" in deleted
        assert not orphan_path.exists()

    def test_delete_pipeline_artifacts(self, registry):
        """Test deleting all artifacts for a pipeline."""
        from sklearn.preprocessing import StandardScaler

        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )
        registry.register(
            obj=StandardScaler(),
            artifact_id="0002:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        count = registry.delete_pipeline_artifacts("0001")

        assert count == 1
        assert registry.resolve("0001:0:all") is None
        assert registry.resolve("0002:0:all") is not None

    def test_delete_pipeline_artifacts_with_files(self, registry):
        """Test deleting pipeline artifacts including files."""
        from sklearn.preprocessing import StandardScaler

        # Register and get the record to know the path
        record = registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Verify file exists
        filepath = registry.binaries_dir / record.path
        assert filepath.exists()

        # Delete with file deletion
        count = registry.delete_pipeline_artifacts("0001", delete_files=True)

        assert count == 1
        # File should be deleted since no other artifact references it
        assert not filepath.exists()

    def test_cleanup_failed_run(self, registry):
        """Test cleaning up artifacts from a failed run."""
        from sklearn.preprocessing import StandardScaler

        registry.start_run()
        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        count = registry.cleanup_failed_run()

        assert count == 1
        assert registry.resolve("0001:0:all") is None

    def test_purge_dataset_artifacts_requires_confirm(self, registry):
        """Test that purge requires confirmation."""
        with pytest.raises(ValueError, match="confirm=True"):
            registry.purge_dataset_artifacts(confirm=False)

    def test_purge_dataset_artifacts(self, registry):
        """Test purging all artifacts for a dataset."""
        from sklearn.preprocessing import StandardScaler

        # Register multiple artifacts
        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )
        registry.register(
            obj=StandardScaler(),
            artifact_id="0002:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Verify files exist (now in shard directories)
        def count_files(path):
            count = 0
            for item in path.iterdir():
                if item.is_file():
                    count += 1
                elif item.is_dir():
                    count += count_files(item)
            return count

        file_count_before = count_files(registry.binaries_dir)
        assert file_count_before >= 1

        # Purge
        files_deleted, bytes_freed = registry.purge_dataset_artifacts(confirm=True)

        assert files_deleted >= 1
        assert bytes_freed > 0

        # All in-memory state should be cleared
        assert len(registry._artifacts) == 0
        assert len(registry._by_content_hash) == 0

        # All files should be deleted (recursive check)
        file_count_after = count_files(registry.binaries_dir) if registry.binaries_dir.exists() else 0
        assert file_count_after == 0

    def test_get_stats(self, registry):
        """Test getting storage statistics."""
        from sklearn.preprocessing import StandardScaler

        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["total_artifacts"] == 1
        assert stats["unique_files"] == 1
        assert "transformer" in stats["by_type"]
        assert stats["by_type"]["transformer"] == 1
        # Note: get_stats counts files recursively in shard directories
        assert stats["disk_usage_bytes"] > 0
        assert stats["disk_file_count"] >= 1
        assert stats["dataset"] == "test_dataset"

    def test_get_stats_with_orphans(self, registry):
        """Test stats includes orphan information."""
        from sklearn.preprocessing import StandardScaler

        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Create an orphan
        orphan_data = b"orphan data - 12 bytes"
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(orphan_data)

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["orphaned_count"] == 1
        assert stats["orphaned_size_bytes"] == len(orphan_data)

class TestMetaModelHandling:
    """Tests for meta-model (stacking) artifact handling."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def registry(self, temp_workspace):
        """Create a registry for testing."""
        return ArtifactRegistry(
            workspace=temp_workspace,
            dataset="test_dataset"
        )

    def test_register_meta_model(self, registry):
        """Test registering a meta-model with source models."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge

        # Register source models first
        registry.register(
            obj=LinearRegression(),
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL
        )
        registry.register(
            obj=RandomForestRegressor(n_estimators=10),
            artifact_id="0001:4:all",
            artifact_type=ArtifactType.MODEL
        )

        # Register meta-model using convenience method
        record = registry.register_meta_model(
            obj=Ridge(),
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all", "0001:4:all"],
            feature_columns=["LinearRegression_pred", "RandomForestRegressor_pred"]
        )

        assert record.artifact_type == ArtifactType.META_MODEL
        assert record.meta_config is not None
        assert len(record.meta_config.source_models) == 2
        assert record.meta_config.feature_columns == [
            "LinearRegression_pred",
            "RandomForestRegressor_pred"
        ]
        assert "0001:3:all" in record.depends_on
        assert "0001:4:all" in record.depends_on

    def test_register_meta_model_auto_feature_columns(self, registry):
        """Test that feature columns are auto-generated from source class names."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge

        # Register source models
        registry.register(
            obj=LinearRegression(),
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL
        )
        registry.register(
            obj=RandomForestRegressor(n_estimators=10),
            artifact_id="0001:4:all",
            artifact_type=ArtifactType.MODEL
        )

        # Register meta-model without explicit feature columns
        record = registry.register_meta_model(
            obj=Ridge(),
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all", "0001:4:all"]
        )

        # Feature columns should be auto-generated
        assert record.meta_config.feature_columns == [
            "LinearRegression_pred",
            "RandomForestRegressor_pred"
        ]

    def test_register_meta_model_missing_source_fails(self, registry):
        """Test that registering meta-model with missing sources fails."""
        from sklearn.linear_model import Ridge

        # Try to register meta-model without registering sources first
        with pytest.raises(ValueError, match="missing source model dependencies"):
            registry.register_meta_model(
                obj=Ridge(),
                artifact_id="0001:5:all",
                source_model_ids=["0001:3:all", "0001:4:all"]
            )

    def test_meta_model_dependency_validation(self, registry):
        """Test that meta-model dependency validation works."""
        from sklearn.linear_model import Ridge

        # Register one source model
        registry.register(
            obj=Ridge(),
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL
        )

        # Try to register meta-model with one missing source
        with pytest.raises(ValueError, match="0001:4:all"):
            registry.register_meta_model(
                obj=Ridge(),
                artifact_id="0001:5:all",
                source_model_ids=["0001:3:all", "0001:4:all"]
            )

    def test_meta_model_dependencies_tracked(self, registry):
        """Test that meta-model dependencies are in dependency graph."""
        from sklearn.linear_model import LinearRegression, Ridge

        # Register source
        registry.register(
            obj=LinearRegression(),
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL
        )

        # Register meta-model
        registry.register_meta_model(
            obj=Ridge(),
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all"]
        )

        # Check dependency graph
        deps = registry.get_dependencies("0001:5:all")
        assert "0001:3:all" in deps

    def test_resolve_meta_model_dependencies(self, registry):
        """Test resolving transitive dependencies for meta-model."""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        # Register preprocessing -> model -> meta-model chain
        registry.register(
            obj=StandardScaler(),
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )
        registry.register(
            obj=LinearRegression(),
            artifact_id="0001:3:all",
            artifact_type=ArtifactType.MODEL,
            depends_on=["0001:0:all"]  # Model depends on scaler
        )
        registry.register_meta_model(
            obj=Ridge(),
            artifact_id="0001:5:all",
            source_model_ids=["0001:3:all"]
        )

        # Resolve all dependencies
        all_deps = registry.resolve_dependencies("0001:5:all")
        dep_ids = [r.artifact_id for r in all_deps]

        # Should include scaler and source model
        assert "0001:0:all" in dep_ids
        assert "0001:3:all" in dep_ids

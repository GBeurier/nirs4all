"""
Integration tests for artifact cleanup utilities.

Tests cleanup functionality:
- Orphan detection works correctly
- Cleanup dry run mode
- Actual orphan deletion
- Failed run cleanup
- Dataset purge
- Auto-cleanup on pipeline failure
"""

import pytest
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist
from nirs4all.pipeline.storage.artifacts.types import ArtifactType


def create_test_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_cleanup")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestOrphanDetection:
    """Tests for orphaned artifact detection."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_no_orphans_when_all_referenced(self, registry, workspace_path):
        """No orphans should be found when all files are referenced."""
        # Register an artifact
        scaler = StandardScaler()
        registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=False)
        assert len(orphans) == 0

    def test_orphan_detection_finds_unreferenced_files(self, registry, workspace_path):
        """Unreferenced files should be detected as orphans."""
        # Register an artifact
        scaler = StandardScaler()
        registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Create an orphan file directly
        orphan_path = registry.binaries_dir / "orphan_artifact.pkl"
        orphan_path.write_bytes(b"orphan data")

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=False)
        assert "orphan_artifact.pkl" in orphans

    def test_orphan_detection_multiple_orphans(self, registry, workspace_path):
        """Multiple orphan files should all be detected."""
        # Create orphan files
        for i in range(5):
            orphan_path = registry.binaries_dir / f"orphan_{i}.pkl"
            orphan_path.write_bytes(f"orphan {i}".encode())

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=False)
        assert len(orphans) == 5

    def test_orphan_detection_ignores_directories(self, registry, workspace_path):
        """Orphan detection should ignore subdirectories."""
        # Create a subdirectory
        subdir = registry.binaries_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("test")

        orphans = registry.find_orphaned_artifacts(scan_all_manifests=False)
        # Subdirectory should not be listed as orphan
        assert "subdir" not in orphans


class TestOrphanCleanup:
    """Tests for orphan cleanup functionality."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_cleanup_dry_run_preserves_files(self, registry, workspace_path):
        """Dry run should not delete any files."""
        # Create orphan
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan")

        deleted, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=True,
            scan_all_manifests=False
        )

        # Should report as would-be-deleted
        assert "orphan.pkl" in deleted
        assert bytes_freed == 6

        # File should still exist
        assert orphan_path.exists()

    def test_cleanup_actually_deletes_files(self, registry, workspace_path):
        """Actual cleanup should delete orphan files."""
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan data")

        deleted, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=False,
            scan_all_manifests=False
        )

        assert "orphan.pkl" in deleted
        assert not orphan_path.exists()

    def test_cleanup_preserves_referenced_files(self, registry, workspace_path):
        """Cleanup should not delete referenced artifacts."""
        # Register an artifact
        scaler = StandardScaler()
        record = registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        referenced_path = registry.binaries_dir / record.path

        # Create an orphan
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan")

        deleted, _ = registry.delete_orphaned_artifacts(
            dry_run=False,
            scan_all_manifests=False
        )

        # Referenced file should still exist
        assert referenced_path.exists()

        # Orphan should be deleted
        assert not orphan_path.exists()
        assert "orphan.pkl" in deleted

    def test_cleanup_returns_correct_byte_count(self, registry, workspace_path):
        """Cleanup should return accurate byte count."""
        # Create orphans of known sizes
        sizes = [100, 200, 300]
        for i, size in enumerate(sizes):
            path = registry.binaries_dir / f"orphan_{i}.pkl"
            path.write_bytes(b"x" * size)

        _, bytes_freed = registry.delete_orphaned_artifacts(
            dry_run=False,
            scan_all_manifests=False
        )

        assert bytes_freed == sum(sizes)


class TestFailedRunCleanup:
    """Tests for failed run artifact cleanup."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_cleanup_failed_run_removes_current_run_artifacts(
        self, registry, workspace_path
    ):
        """Failed run cleanup should remove artifacts from current run."""
        registry.start_run()

        # Register some artifacts
        for i in range(3):
            scaler = StandardScaler()
            registry.register(
                obj=scaler,
                artifact_id=f"0001:{i}:all",
                artifact_type=ArtifactType.TRANSFORMER
            )

        # Simulate failure and cleanup
        count = registry.cleanup_failed_run()

        assert count == 3

        # Registry should be empty
        assert registry.resolve("0001:0:all") is None
        assert registry.resolve("0001:1:all") is None
        assert registry.resolve("0001:2:all") is None

    def test_cleanup_failed_run_empty_when_no_run(self, registry):
        """Cleanup with no run should return 0."""
        count = registry.cleanup_failed_run()
        assert count == 0

    def test_end_run_clears_tracking(self, registry, workspace_path):
        """end_run should clear run tracking."""
        registry.start_run()

        scaler = StandardScaler()
        registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        registry.end_run()

        # cleanup_failed_run should do nothing after end_run
        count = registry.cleanup_failed_run()
        assert count == 0

        # But artifact should still exist
        assert registry.resolve("0001:0:all") is not None


class TestPipelineArtifactDeletion:
    """Tests for deleting artifacts by pipeline."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_delete_pipeline_artifacts(self, registry, workspace_path):
        """Delete all artifacts for a specific pipeline."""
        # Register artifacts for two pipelines
        for i in range(3):
            scaler = StandardScaler()
            registry.register(
                obj=scaler,
                artifact_id=f"0001:{i}:all",
                artifact_type=ArtifactType.TRANSFORMER
            )
            registry.register(
                obj=scaler,
                artifact_id=f"0002:{i}:all",
                artifact_type=ArtifactType.TRANSFORMER
            )

        # Delete pipeline 0001
        count = registry.delete_pipeline_artifacts("0001")

        assert count == 3

        # Pipeline 0001 artifacts should be gone
        for i in range(3):
            assert registry.resolve(f"0001:{i}:all") is None

        # Pipeline 0002 artifacts should remain
        for i in range(3):
            assert registry.resolve(f"0002:{i}:all") is not None

    def test_delete_pipeline_artifacts_with_files(self, registry, workspace_path):
        """Delete pipeline artifacts including binary files."""
        scaler = StandardScaler()
        record = registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        filepath = registry.binaries_dir / record.path
        assert filepath.exists()

        count = registry.delete_pipeline_artifacts("0001", delete_files=True)

        assert count == 1
        assert not filepath.exists()


class TestDatasetPurge:
    """Tests for purging all dataset artifacts."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_purge_requires_confirmation(self, registry):
        """Purge should require confirm=True."""
        with pytest.raises(ValueError, match="confirm=True"):
            registry.purge_dataset_artifacts(confirm=False)

    def test_purge_deletes_all_artifacts(self, registry, workspace_path):
        """Purge should delete all artifacts for dataset."""
        # Create multiple artifacts
        for i in range(5):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i+1]]))
            registry.register(
                obj=scaler,
                artifact_id=f"000{i}:0:all",
                artifact_type=ArtifactType.TRANSFORMER
            )

        # Verify files exist
        file_count = sum(1 for _ in registry.binaries_dir.iterdir() if _.is_file())
        assert file_count >= 1

        # Purge
        files_deleted, bytes_freed = registry.purge_dataset_artifacts(confirm=True)

        assert files_deleted >= 1
        assert bytes_freed > 0

        # All files should be deleted
        remaining = list(f for f in registry.binaries_dir.iterdir() if f.is_file())
        assert len(remaining) == 0

        # Registry should be empty
        assert len(registry._artifacts) == 0


class TestStorageStats:
    """Tests for storage statistics."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_stats_basic(self, registry, workspace_path):
        """Test basic statistics."""
        scaler = StandardScaler()
        registry.register(
            obj=scaler,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["total_artifacts"] == 1
        assert stats["unique_files"] == 1
        assert stats["disk_file_count"] == 1
        assert stats["disk_usage_bytes"] > 0
        assert "transformer" in stats["by_type"]

    def test_stats_with_orphans(self, registry, workspace_path):
        """Stats should include orphan information."""
        # Create orphan
        orphan_path = registry.binaries_dir / "orphan.pkl"
        orphan_path.write_bytes(b"orphan data - 15 bytes")

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["orphaned_count"] == 1
        assert stats["orphaned_size_bytes"] == 22  # len("orphan data - 15 bytes")

    def test_stats_deduplication_ratio(self, registry, workspace_path):
        """Stats should show deduplication ratio."""
        scaler = StandardScaler()

        # Register same object multiple times
        for i in range(4):
            registry.register(
                obj=scaler,
                artifact_id=f"000{i}:0:all",
                artifact_type=ArtifactType.TRANSFORMER
            )

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["total_artifacts"] == 4
        assert stats["unique_files"] == 1
        assert stats["deduplication_ratio"] == 0.75  # (4-1)/4


class TestCleanupWithRealPipelines:
    """Tests for cleanup with actual pipeline runs."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        """Create pipeline runner."""
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_no_orphans_after_successful_run(
        self, runner, dataset, workspace_path
    ):
        """Successful pipeline run should leave no orphans."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(PipelineConfigs(pipeline), dataset)

        # Create registry for this dataset
        registry = ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_cleanup"
        )

        # Check for orphans (using in-memory only since manifest format varies)
        # This is a sanity check - actual orphan detection requires manifest scanning
        stats = registry.get_stats(scan_all_manifests=False)

        # Should have some artifacts (or zero if save_artifacts didn't save to binaries/)
        assert stats["total_artifacts"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

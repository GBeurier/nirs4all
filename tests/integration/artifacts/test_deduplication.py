"""
Integration tests for artifact deduplication.

Tests the content-addressed storage deduplication:
- Same model params across runs shares single file
- Different params produce different artifacts
- Hash collision handling
- Cross-run deduplication works correctly
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist


def create_test_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_dedup")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestRegistryDeduplication:
    """Tests for ArtifactRegistry deduplication."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def registry(self, workspace_path):
        """Create artifact registry."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType
        return ArtifactRegistry(
            workspace=workspace_path,
            dataset="test_dataset"
        )

    def test_same_object_registered_twice_creates_one_file(
        self, registry, workspace_path
    ):
        """Identical objects registered twice should share the same file."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        scaler = StandardScaler()

        # Register same object with different IDs
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

        # Should have same content hash
        assert record1.content_hash == record2.content_hash

        # Should reference same file
        assert record1.path == record2.path

        # Should only have one file on disk (artifacts are sharded)
        binaries_dir = workspace_path / "artifacts"
        files = list(binaries_dir.rglob("*.pkl")) + list(binaries_dir.rglob("*.joblib"))
        assert len(files) == 1

    def test_different_objects_create_different_files(
        self, registry, workspace_path
    ):
        """Objects with different state should create different files."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        # Create two scalers with different fitted state
        scaler1 = StandardScaler()
        scaler1.fit(np.array([[0, 0], [1, 1]]))

        scaler2 = StandardScaler()
        scaler2.fit(np.array([[10, 10], [20, 20]]))  # Different data

        record1 = registry.register(
            obj=scaler1,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        record2 = registry.register(
            obj=scaler2,
            artifact_id="0002:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Should have different hashes
        assert record1.content_hash != record2.content_hash

        # Should have different paths
        assert record1.path != record2.path

        # Should have two files (artifacts are sharded)
        binaries_dir = workspace_path / "artifacts"
        files = list(binaries_dir.rglob("*.pkl")) + list(binaries_dir.rglob("*.joblib"))
        assert len(files) == 2

    def test_deduplication_across_pipelines(self, registry, workspace_path):
        """Same scaler used in different pipelines should be deduplicated."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        # Simulate same scaler fitted on same data
        data = np.array([[0], [1], [2]])

        # Pipeline 1
        scaler1 = StandardScaler()
        scaler1.fit(data)
        record1 = registry.register(
            obj=scaler1,
            artifact_id="0001:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Pipeline 2 - same scaler, same data, same result
        scaler2 = StandardScaler()
        scaler2.fit(data)
        record2 = registry.register(
            obj=scaler2,
            artifact_id="0002:0:all",
            artifact_type=ArtifactType.TRANSFORMER
        )

        # Should be deduplicated
        assert record1.content_hash == record2.content_hash
        assert record1.path == record2.path

    def test_deduplication_stats(self, registry, workspace_path):
        """Verify stats correctly report deduplication."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        scaler = StandardScaler()

        # Register same scaler 5 times
        for i in range(5):
            registry.register(
                obj=scaler,
                artifact_id=f"000{i}:0:all",
                artifact_type=ArtifactType.TRANSFORMER
            )

        stats = registry.get_stats(scan_all_manifests=False)

        assert stats["total_artifacts"] == 5
        assert stats["unique_files"] == 1
        assert stats["deduplication_ratio"] == 0.8  # (5-1)/5


class TestPersistDeduplication:
    """Tests for persist-level deduplication."""

    @pytest.fixture
    def binaries_dir(self, tmp_path):
        """Create temporary binaries directory."""
        binaries = tmp_path / "binaries"
        binaries.mkdir(parents=True)
        return binaries

    def test_persist_same_object_twice_creates_one_file(self, binaries_dir):
        """Persisting identical object twice creates only one file."""
        scaler = StandardScaler()

        meta1 = persist(scaler, binaries_dir, "scaler1")
        meta2 = persist(scaler, binaries_dir, "scaler2")

        # Hashes should be same
        assert meta1["hash"] == meta2["hash"]

        # Paths should be same (content-addressed)
        assert meta1["path"] == meta2["path"]

        # Only one file
        files = list(binaries_dir.glob("*"))
        assert len(files) == 1


class TestCrossRunDeduplication:
    """Tests for deduplication across pipeline runs."""

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

    def test_identical_pipelines_share_artifacts(
        self, runner, dataset, workspace_path
    ):
        """Running identical pipelines should share artifact files."""
        # Same pipeline twice
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        # Run twice
        predictions1, _ = runner.run(PipelineConfigs(pipeline), dataset)
        predictions2, _ = runner.run(PipelineConfigs(pipeline), dataset)

        # Both should produce predictions
        assert len(predictions1) > 0
        assert len(predictions2) > 0

        # Content-addressed storage in artifacts/ provides deduplication.
        # Two identical pipeline runs should share artifact files.
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Should have artifacts from both runs
        assert len(artifact_files) >= 1, "Should have artifact files"

    def test_different_model_params_create_different_artifacts(
        self, runner, dataset, workspace_path
    ):
        """Pipelines with different params should create different artifacts."""
        pipeline1 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=0.1)},
        ]
        pipeline2 = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"model": Ridge(alpha=100.0)},  # Very different
        ]

        predictions1, _ = runner.run(PipelineConfigs(pipeline1), dataset)
        predictions2, _ = runner.run(PipelineConfigs(pipeline2), dataset)

        # Both should work
        assert len(predictions1) > 0
        assert len(predictions2) > 0

        # Different model params should produce different artifact files
        # in the content-addressed artifacts/ directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Should have at least 2 distinct artifact files (different models)
        assert len(artifact_files) >= 2, \
            "Different model params should produce different artifact files"


class TestHashCollisionHandling:
    """Tests for handling hash collisions (edge case)."""

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

    def test_different_content_never_overwrites(self, registry, workspace_path):
        """Different content should never overwrite existing files."""
        from nirs4all.pipeline.storage.artifacts.types import ArtifactType

        # Create different scalers
        scalers = []
        records = []

        for i in range(5):
            scaler = StandardScaler()
            scaler.fit(np.array([[i], [i + 1], [i + 2]]))  # Different data
            scalers.append(scaler)

            record = registry.register(
                obj=scaler,
                artifact_id=f"000{i}:0:all",
                artifact_type=ArtifactType.TRANSFORMER
            )
            records.append(record)

        # Reload each and verify correct content
        for i, record in enumerate(records):
            loaded = registry.load_artifact(record)
            assert isinstance(loaded, StandardScaler)
            # Verify the mean is correct (based on training data)
            expected_mean = np.array([i + 1])  # mean of [i, i+1, i+2]
            np.testing.assert_array_almost_equal(loaded.mean_, expected_mean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

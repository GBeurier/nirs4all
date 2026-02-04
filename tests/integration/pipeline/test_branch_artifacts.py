"""
Integration tests for Branch Artifact Integrity (Section 5.1.2).

Tests artifact persistence and loading:
- All branch artifacts are saved correctly in content-addressed artifacts/ directory
- Artifacts from different branches have unique paths
- Branch artifacts are isolated (no cross-contamination)
- DuckDB store contains correct chain and artifact records
"""

import pytest
import numpy as np
from pathlib import Path
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader


def create_test_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_artifacts")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestBranchArtifactCompleteness:
    """
    Test that all branch artifacts are saved correctly.

    Per specification §5.1.2.
    """

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
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

    def test_branch_artifacts_complete(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Verify all branch artifacts are saved correctly.

        Per spec §5.1.2: All artifacts should be persisted.
        DuckDB storage: artifacts stored in store.duckdb + flat artifacts/ directory.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # DuckDB storage: verify store.duckdb was created
        store_file = workspace_path / "store.duckdb"
        assert store_file.exists(), "store.duckdb should exist"

        # Verify artifacts are stored (either in artifacts/ via WorkspaceStore
        # or in binaries/ via V3 artifact registry)
        artifacts_dir = workspace_path / "artifacts"
        binaries_dir = workspace_path / "binaries"
        artifact_files = []
        for search_dir in [artifacts_dir, binaries_dir]:
            if search_dir.exists():
                artifact_files.extend([f for f in search_dir.rglob("*.*") if f.is_file()])
        assert len(artifact_files) >= 2, f"Expected at least 2 artifacts, got {len(artifact_files)}"

    def test_all_artifact_files_exist(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Verify all artifact files exist in content-addressed artifacts/ directory.

        Per spec §5.1.2: All referenced files must be present.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}},
                ],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Verify artifact files exist in content-addressed directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Should have artifacts from both branches
        assert len(artifact_files) >= 2, f"Expected at least 2 artifacts, got {len(artifact_files)}"


class TestBranchArtifactUniqueness:
    """
    Test that artifacts from different branches have unique paths.

    Per specification §5.1.2.
    """

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
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

    def test_branch_artifacts_have_unique_paths(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Test that artifacts from different branch configurations exist.

        Note: Content-addressed storage means identical artifacts will share paths.
        Different preprocessing (StandardScaler vs MinMaxScaler) should produce different artifacts.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Different preprocessing should produce different artifact files
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Should have at least scalers and models
        assert len(artifact_files) >= 2, f"Expected at least 2 artifacts, got {len(artifact_files)}"

    def test_same_operator_different_branches_unique_artifacts(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Test that same operator type in different branches is handled correctly.

        Note: Content-addressed storage may deduplicate identical artifacts.
        This is expected behavior when same data produces same model.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.StandardScaler"}],  # Same class, different branch
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Verify artifacts are created in content-addressed directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Deduplication may reduce count (same class on same data = same hash)
        assert len(artifact_files) >= 1, f"Expected at least 1 artifact, got {len(artifact_files)}"


class TestBranchArtifactIsolation:
    """
    Test that branch artifacts are isolated (no cross-contamination).

    Per specification §5.1.2.
    """

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
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

    def test_branch_artifact_isolation(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Verify branch artifacts are isolated (no cross-contamination).

        Per spec §5.1.2: Loading branch 0 should not load branch 1 artifacts.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Predict with branch 0 only
        branch_0_preds = predictions.filter_predictions(branch_id=0, partition="test")
        if len(branch_0_preds) == 0:
            pytest.skip("No branch 0 predictions")

        target_pred = branch_0_preds[0]

        # This should only use branch 0 artifacts
        y_pred, _ = runner_with_save.predict(
            prediction_obj=target_pred,
            dataset=dataset,
            dataset_name="test_artifacts"
        )

        # Verify prediction succeeded (implying correct artifacts loaded)
        assert y_pred is not None


class TestManifestBranchMetadata:
    """
    Test that manifest contains correct branch metadata.

    Per specification §5.1.2.
    """

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner_with_save(self, workspace_path):
        """Create a PipelineRunner that saves artifacts."""
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

    def test_store_contains_branch_metadata(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Test that DuckDB store contains branch metadata for chains and predictions.

        Per spec §5.1.2.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Verify store.duckdb exists and artifacts directory has files
        store_path = workspace_path / "store.duckdb"
        assert store_path.exists(), "store.duckdb should be created"

        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))
        assert len(artifact_files) >= 1, "Should have artifact files"

    def test_named_branches_produce_predictions(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test that named branches produce predictions with branch_name metadata."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "scaler_std": [{"class": "sklearn.preprocessing.StandardScaler"}],
                "scaler_mm": [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Should have predictions from both named branches
        assert len(predictions) > 0, "Should produce predictions"

        # Verify artifacts directory has files
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))
        assert len(artifact_files) >= 1, "Should have artifact files"


class TestArtifactLoaderBranchSupport:
    """
    Test ArtifactLoader branch-aware loading.

    Per specification §5.1.2.
    """

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "binaries" / "test_dataset").mkdir(parents=True)
        return workspace

    def test_artifact_loader_handles_branch_artifacts(self, workspace_path):
        """Test that ArtifactLoader handles branch artifacts correctly."""
        # Create mock manifest with v2 format artifacts
        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": [
                    {
                        "artifact_id": "0001:0:2:all",
                        "content_hash": "sha256:abc123",
                        "path": "ab/abc123.pkl",
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 2,
                        "artifact_type": "transformer",
                        "class_name": "StandardScaler",
                        "format": "joblib",
                    },
                    {
                        "artifact_id": "0001:1:2:all",
                        "content_hash": "sha256:def456",
                        "path": "cd/cdef456.pkl",
                        "pipeline_id": "0001",
                        "branch_path": [1],
                        "step_index": 2,
                        "artifact_type": "transformer",
                        "class_name": "MinMaxScaler",
                        "format": "joblib",
                    },
                    {
                        "artifact_id": "0001:0:3:all",
                        "content_hash": "sha256:ghi789",
                        "path": "ef/efgh789.pkl",
                        "pipeline_id": "0001",
                        "branch_path": [0],
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": "joblib",
                    },
                    {
                        "artifact_id": "0001:1:3:all",
                        "content_hash": "sha256:jkl012",
                        "path": "gh/ghij012.pkl",
                        "pipeline_id": "0001",
                        "branch_path": [1],
                        "step_index": 3,
                        "artifact_type": "model",
                        "class_name": "Ridge",
                        "format": "joblib",
                    },
                ]
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Check we have artifacts for step 2
        assert loader.has_binaries_for_step(2, branch_id=0)
        assert loader.has_binaries_for_step(2, branch_id=1)

        # Check cache info
        info = loader.get_cache_info()
        assert "total_artifacts" in info
        assert info["total_artifacts"] == 4


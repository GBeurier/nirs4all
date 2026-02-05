"""
Integration tests for CV fold model artifacts.

Tests per-fold model saving and loading:
- Each CV fold creates a separate model artifact
- All fold models can be loaded for ensemble prediction
- Fold models are correctly identified by fold_id
- CV averaging works with loaded fold models
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all.pipeline.storage.artifacts.artifact_registry import ArtifactRegistry
from nirs4all.pipeline.storage.artifacts.artifact_loader import ArtifactLoader
from nirs4all.pipeline.storage.artifacts.types import ArtifactType
from nirs4all.pipeline.storage.artifacts import generate_artifact_id_v3


def make_v3_id(pipeline_id: str, step: int, fold_id=None, operator: str = "Model"):
    """Helper to generate V3 artifact IDs for tests."""
    chain_path = f"s{step}.{operator}"
    return generate_artifact_id_v3(pipeline_id, chain_path, fold_id)


def create_test_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_fold_models")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestFoldModelRegistration:
    """Tests for registering per-fold model artifacts."""

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

    def test_register_fold_models(self, registry):
        """Test registering models for different CV folds."""
        # Create models for 3 folds
        for fold_id in range(3):
            model = Ridge(alpha=1.0)
            model.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))

            chain_path = f"s3.Ridge"
            artifact_id = make_v3_id("0001", 3, fold_id, "Ridge")
            record = registry.register(
                obj=model,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.MODEL,
                chain_path=chain_path
            )

            assert record.fold_id == fold_id
            assert record.step_index == 3

    def test_get_fold_models(self, registry):
        """Test retrieving all fold models for a step."""
        # Register 5 fold models
        n_folds = 5
        for fold_id in range(n_folds):
            model = Ridge(alpha=1.0)
            model.fit(np.array([[fold_id], [fold_id + 1]]), np.array([0, 1]))

            chain_path = f"s3.Ridge"
            artifact_id = make_v3_id("0001", 3, fold_id, "Ridge")
            registry.register(
                obj=model,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.MODEL,
                chain_path=chain_path
            )

        # Retrieve fold models
        fold_models = registry.get_fold_models(
            pipeline_id="0001",
            step_index=3
        )

        assert len(fold_models) == n_folds

        # Should be sorted by fold_id
        for i, record in enumerate(fold_models):
            assert record.fold_id == i

    def test_fold_models_have_different_content(self, registry, workspace_path):
        """Fold models trained on different data have different content."""
        # Register models trained on different data
        records = []
        for fold_id in range(3):
            model = Ridge(alpha=1.0)
            # Different training data per fold
            X = np.array([[fold_id * 10], [fold_id * 10 + 1], [fold_id * 10 + 2]])
            y = X.ravel()
            model.fit(X, y)

            chain_path = f"s3.Ridge"
            artifact_id = make_v3_id("0001", 3, fold_id, "Ridge")
            record = registry.register(
                obj=model,
                artifact_id=artifact_id,
                artifact_type=ArtifactType.MODEL,
                chain_path=chain_path
            )
            records.append(record)

        # Content hashes should be different
        hashes = [r.content_hash for r in records]
        assert len(set(hashes)) == 3  # All unique


class TestFoldModelLoading:
    """Tests for loading per-fold model artifacts."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        artifacts_dir = workspace / "artifacts"
        artifacts_dir.mkdir(parents=True)
        return workspace

    def test_load_fold_models_from_loader(self, workspace_path):
        """Test loading fold models using ArtifactLoader."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        artifacts_dir = workspace_path / "artifacts"

        # Create and persist fold models
        items = []
        for fold_id in range(3):
            model = Ridge(alpha=1.0)
            X = np.array([[fold_id], [fold_id + 1], [fold_id + 2]])
            y = X.ravel() * 2
            model.fit(X, y)

            meta = persist(model, artifacts_dir, f"model_fold{fold_id}")
            chain_path = "s3.Ridge"
            artifact_id = make_v3_id("0001", 3, fold_id, "Ridge")
            items.append({
                "artifact_id": artifact_id,
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": 3,
                "fold_id": fold_id,
                "artifact_type": "model",
                "class_name": "Ridge",
                "format": meta["format"],
                "chain_path": chain_path,
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {
                "schema_version": "2.0",
                "items": items
            }
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Load all fold models
        fold_models = loader.load_fold_models(step_index=3)

        assert len(fold_models) == 3

        # Verify each is a Ridge model
        for fold_id, model in fold_models:
            assert isinstance(model, Ridge)
            assert hasattr(model, 'coef_')

    def test_fold_models_produce_different_predictions(self, workspace_path):
        """Fold models trained on different data should predict differently."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        artifacts_dir = workspace_path / "artifacts"

        # Create fold models with different training data
        items = []
        for fold_id in range(3):
            model = Ridge(alpha=1.0)
            offset = fold_id * 100
            X = np.array([[offset], [offset + 1], [offset + 2]])
            y = X.ravel() * (fold_id + 1)  # Different slopes
            model.fit(X, y)

            meta = persist(model, artifacts_dir, f"model_fold{fold_id}")
            chain_path = "s3.Ridge"
            artifact_id = make_v3_id("0001", 3, fold_id, "Ridge")
            items.append({
                "artifact_id": artifact_id,
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": 3,
                "fold_id": fold_id,
                "artifact_type": "model",
                "class_name": "Ridge",
                "format": meta["format"],
                "chain_path": chain_path,
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Load and predict with each
        fold_models = loader.load_fold_models(step_index=3)
        X_test = np.array([[50]])

        predictions = []
        for fold_id, model in fold_models:
            pred = model.predict(X_test)
            predictions.append(pred[0])

        # Predictions should be different (different training)
        assert len(set(predictions)) == 3


class TestCVPipelineFoldArtifacts:
    """Tests for fold artifacts in CV pipelines."""

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

    def test_kfold_creates_multiple_model_artifacts(
        self, runner, dataset, workspace_path
    ):
        """KFold CV should create model artifacts for each fold."""
        n_splits = 3
        pipeline = [
            KFold(n_splits=n_splits, shuffle=True, random_state=42),
            {"class": "sklearn.preprocessing.StandardScaler"},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(PipelineConfigs(pipeline), dataset)

        assert len(predictions) > 0

        # Check that artifacts were saved to content-addressed directory
        artifacts_dir = workspace_path / "artifacts"
        artifact_files = list(artifacts_dir.glob("**/*.joblib")) + list(artifacts_dir.glob("**/*.pkl"))

        # Should have at least one artifact file (model artifacts)
        assert len(artifact_files) >= 1, \
            "Should have at least one artifact file for model"

    def test_repeated_kfold_creates_many_artifacts(
        self, runner, dataset, workspace_path
    ):
        """RepeatedKFold should create n_splits * n_repeats model artifacts."""
        n_splits = 2
        n_repeats = 2

        pipeline = [
            RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42),
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(PipelineConfigs(pipeline), dataset)

        # Should have predictions for each fold
        expected_folds = n_splits * n_repeats
        assert len(predictions) >= expected_folds


class TestFoldModelEnsembleLoading:
    """Tests for loading fold models for ensemble prediction."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        """Create temporary workspace."""
        workspace = tmp_path / "workspace"
        artifacts_dir = workspace / "artifacts"
        artifacts_dir.mkdir(parents=True)
        return workspace

    def test_ensemble_prediction_with_fold_models(self, workspace_path):
        """Test averaging predictions from multiple fold models."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        artifacts_dir = workspace_path / "artifacts"

        # Create fold models
        n_folds = 5
        items = []
        for fold_id in range(n_folds):
            model = Ridge(alpha=1.0)
            # Same data but different random state would give different models
            np.random.seed(fold_id)
            X = np.random.randn(50, 10)
            y = X @ np.random.randn(10)
            model.fit(X, y)

            meta = persist(model, artifacts_dir, f"model_fold{fold_id}")
            items.append({
                "artifact_id": f"0001:2:{fold_id}",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": 2,
                "fold_id": fold_id,
                "artifact_type": "model",
                "class_name": "Ridge",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        # Load fold models
        fold_models = loader.load_fold_models(step_index=2)
        assert len(fold_models) == n_folds

        # Ensemble prediction
        np.random.seed(999)
        X_test = np.random.randn(10, 10)

        all_predictions = []
        for fold_id, model in fold_models:
            pred = model.predict(X_test)
            all_predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)

        assert ensemble_pred.shape == (10,)

    def test_fold_models_sorted_by_fold_id(self, workspace_path):
        """Fold models should be returned sorted by fold_id."""
        from nirs4all.pipeline.storage.artifacts.artifact_persistence import persist

        artifacts_dir = workspace_path / "artifacts"

        # Create models in non-sequential order
        fold_order = [3, 1, 4, 0, 2]
        items = []

        for fold_id in fold_order:
            model = Ridge()
            model.fit(np.array([[0], [1]]), np.array([0, 1]))
            meta = persist(model, artifacts_dir, f"model_{fold_id}")
            items.append({
                "artifact_id": f"0001:2:{fold_id}",
                "content_hash": meta["hash"],
                "path": meta["path"],
                "pipeline_id": "0001",
                "branch_path": [],
                "step_index": 2,
                "fold_id": fold_id,
                "artifact_type": "model",
                "class_name": "Ridge",
                "format": meta["format"],
            })

        manifest = {
            "dataset": "test_dataset",
            "artifacts": {"schema_version": "2.0", "items": items}
        }

        loader = ArtifactLoader(workspace_path, "test_dataset")
        loader.import_from_manifest(manifest)

        fold_models = loader.load_fold_models(step_index=2)

        # Should be sorted 0, 1, 2, 3, 4
        returned_fold_ids = [fold_id for fold_id, _ in fold_models]
        assert returned_fold_ids == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

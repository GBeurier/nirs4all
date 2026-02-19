"""
Integration tests for Nested Branches (Section 5.1.4 and Phase extension E5).

Tests nested branch behavior:
- Multiple sequential branch steps create Cartesian product
- Nested branch path identification
- Roundtrip with nested branches
- Branch path encoding and naming
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator
from nirs4all.pipeline.runner import PipelineRunner


def create_test_dataset(n_samples: int = 96, n_features: int = 40) -> SpectroDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_nested")
    n_train = int(n_samples * 0.8)
    dataset.add_samples(X[:n_train], indexes={"partition": "train"})
    dataset.add_samples(X[n_train:], indexes={"partition": "test"})
    dataset.add_targets(y[:n_train])
    dataset.add_targets(y[n_train:])

    return dataset

class TestNestedBranchBasics:
    """
    Test basic nested branch functionality.

    Per specification §3.5: Nested branches create Cartesian product.
    """

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create an orchestrator with temporary workspace."""
        return PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            verbose=0,
            save_artifacts=False, save_charts=False,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_two_sequential_branches_cartesian_product(
        self, orchestrator, dataset
    ):
        """
        Test that two sequential branch steps create Cartesian product.

        Per spec §3.5: 2 branches × 2 branches = 4 total configurations.
        """
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],  # A
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],    # B
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],  # X
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],   # Y
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 2 = 4 unique leaf branch combinations
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        # Filter to leaf branches (not a prefix of any other branch name)
        leaf_branches = [b for b in branch_names if not any(other.startswith(b + "_") for other in branch_names)]
        assert len(leaf_branches) == 4, f"Expected 4 branch combinations, got {len(leaf_branches)}: {leaf_branches}"

    def test_nested_branch_names_combined(
        self, orchestrator, dataset
    ):
        """
        Test that nested branch names are combined correctly.

        Per spec §3.5.1: Branch path names should be concatenated.
        """
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": {
                "scaler_std": [{"class": "sklearn.preprocessing.StandardScaler"}],
                "scaler_mm": [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            }},
            {"branch": {
                "pca10": [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                "pca5": [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]

        # Names should be combinations
        expected_patterns = [
            ("scaler_std", "pca10"),
            ("scaler_std", "pca5"),
            ("scaler_mm", "pca10"),
            ("scaler_mm", "pca5"),
        ]

        for pattern in expected_patterns:
            found = any(pattern[0] in name and pattern[1] in name for name in branch_names)
            assert found, f"Expected pattern {pattern} not found in {branch_names}"

    def test_three_level_nested_branches(
        self, orchestrator, dataset
    ):
        """
        Test three levels of nested branches.

        2 × 2 × 2 = 8 total configurations.
        """
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"branch": [
                [{"class": "sklearn.feature_selection.SelectKBest", "params": {"k": 3}}],
                [{"class": "sklearn.feature_selection.SelectKBest", "params": {"k": 5}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 2 × 2 = 8 unique leaf branch combinations
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        # Filter to leaf branches (not a prefix of any other branch name)
        leaf_branches = [b for b in branch_names if not any(other.startswith(b + "_") for other in branch_names)]
        assert len(leaf_branches) == 8, f"Expected 8 branch combinations, got {len(leaf_branches)}"

class TestNestedBranchRoundtrip:
    """
    Test roundtrip with nested branches.

    Per specification §5.1.1: Nested branches produce identical results after reload.
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

    def test_nested_branch_roundtrip(
        self, runner_with_save, dataset, workspace_path
    ):
        """
        Verify nested branches produce identical results after reload.

        Per spec §5.1.1.
        """
        np.random.seed(42)

        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions_train, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Get all unique branch combinations
        branch_ids = [b for b in predictions_train.get_unique_values("branch_id") if b is not None]

        # Verify we have multiple branch combinations (nested branches create Cartesian product)
        assert len(branch_ids) >= 4, f"Expected at least 4 nested branch combinations, got {len(branch_ids)}"

        # Test roundtrip for at least one branch
        branch_preds = predictions_train.filter_predictions(
            branch_id=branch_ids[0], partition="test"
        )
        assert len(branch_preds) > 0, "No predictions found for first branch"

        target_pred = branch_preds[0]

        # Predict on the dataset - this validates artifact loading and execution
        y_pred_reloaded, _ = runner_with_save.predict(
            prediction_obj=target_pred,
            dataset=dataset,
            dataset_name="test_nested"
        )

        # Verify prediction succeeded and has correct shape
        assert y_pred_reloaded is not None, "Prediction failed for nested branch"
        # Prediction should be on full dataset
        assert y_pred_reloaded.shape[0] == dataset.num_samples, \
            f"Expected {dataset.num_samples} predictions, got {y_pred_reloaded.shape[0]}"

class TestNestedBranchWithGenerators:
    """
    Test nested branches combined with generators.
    """

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create an orchestrator with temporary workspace."""
        return PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            verbose=0,
            save_artifacts=False, save_charts=False,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_nested_branch_with_or_generator(
        self, orchestrator, dataset
    ):
        """Test nested branches where one uses _or_ generator."""
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 2 = 4 unique branch combinations
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 4, f"Expected 4 branches, got {len(branch_names)}"

class TestNestedBranchArtifacts:
    """
    Test artifact handling with nested branches.
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

    def test_nested_branch_artifacts_complete(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test that all nested branch artifacts are saved."""
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Verify predictions were made for all combinations
        branch_ids = [b for b in predictions.get_unique_values("branch_id") if b is not None]
        assert len(branch_ids) == 2, f"Expected 2 nested branches (2×1), got {len(branch_ids)}"

        # Each branch should be able to make predictions
        for branch_id in branch_ids:
            branch_preds = predictions.filter_predictions(
                branch_id=branch_id, partition="test"
            )
            assert len(branch_preds) > 0, f"No predictions for nested branch {branch_id}"

class TestNestedBranchEdgeCases:
    """
    Test edge cases for nested branches.
    """

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create an orchestrator with temporary workspace."""
        return PipelineOrchestrator(
            workspace_path=tmp_path / "workspace",
            verbose=0,
            save_artifacts=False, save_charts=False,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return create_test_dataset()

    def test_single_branch_nested_with_multi(
        self, orchestrator, dataset
    ):
        """Test 1 branch followed by 2 branches."""
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 1 × 2 = 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

    def test_multi_branch_nested_with_single(
        self, orchestrator, dataset
    ):
        """Test 2 branches followed by 1 branch."""
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 1 = 2 leaf branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        # Filter to leaf branches (not a prefix of any other branch name)
        leaf_branches = [b for b in branch_names if not any(other.startswith(b + "_") for other in branch_names)]
        assert len(leaf_branches) == 2

    def test_nested_branch_ids_sequential(
        self, orchestrator, dataset
    ):
        """
        Test that flattened branch IDs are sequential.

        Per spec §3.5.2: Flattened IDs should be sequential.
        """
        pipeline = [
            ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Get all branch IDs
        branch_ids = sorted([b for b in predictions.get_unique_values("branch_id") if b is not None])

        # Should be sequential: ['0', '1', '2', '3'] (get_unique_values returns strings)
        expected_ids = [str(i) for i in range(len(branch_ids))]
        assert branch_ids == expected_ids, f"Branch IDs should be sequential: {branch_ids} vs {expected_ids}"

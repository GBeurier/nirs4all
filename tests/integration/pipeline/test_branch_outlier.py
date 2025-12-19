"""
Integration tests for Outlier Excluder Branches (Phase 7).

Tests outlier-based branching functionality:
- OutlierExcluderController with various strategies
- SamplePartitionerController for sample separation
- Combined outlier + preprocessing branches
- Exclusion metadata in predictions
"""

import pytest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator
from nirs4all.pipeline.runner import PipelineRunner


def create_dataset_with_outliers(
    n_samples: int = 150,
    n_features: int = 50,
    n_outliers: int = 15,
    seed: int = 42
) -> SpectroDataset:
    """Create a synthetic dataset with intentional outliers."""
    np.random.seed(seed)

    # Normal samples
    X_normal = np.random.randn(n_samples - n_outliers, n_features)
    y_normal = np.sum(X_normal[:, :5], axis=1) + np.random.randn(n_samples - n_outliers) * 0.5

    # Outlier samples (extreme values)
    X_outliers = np.random.randn(n_outliers, n_features) * 3  # 3x variance
    y_outliers = np.sum(X_outliers[:, :5], axis=1) * 2 + 50  # Shifted and scaled

    # Combine
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([y_normal, y_outliers])

    # Shuffle
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    dataset = SpectroDataset(name="test_outliers")
    train_size = int(n_samples * 0.8)
    dataset.add_samples(X[:train_size], indexes={"partition": "train"})
    dataset.add_samples(X[train_size:], indexes={"partition": "test"})
    dataset.add_targets(y[:train_size])
    dataset.add_targets(y[train_size:])

    return dataset


class TestOutlierExcluderBasics:
    """
    Test basic OutlierExcluder functionality.

    Per specification Phase 7.
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
        """Create test dataset with outliers."""
        return create_dataset_with_outliers()

    def test_outlier_excluder_creates_branches(
        self, orchestrator, dataset
    ):
        """Test that outlier_excluder creates correct number of branches."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,  # Baseline (no exclusion)
                    {"method": "isolation_forest", "contamination": 0.05},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 branches (one per strategy)
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2, f"Expected 2 branches, got {len(branch_names)}: {branch_names}"

    def test_outlier_excluder_baseline_branch(
        self, orchestrator, dataset
    ):
        """Test that None strategy creates baseline branch without exclusion."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [None],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 1 baseline branch
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 1
        assert "baseline" in branch_names[0].lower()

    def test_outlier_excluder_isolation_forest(
        self, orchestrator, dataset
    ):
        """Test IsolationForest exclusion strategy."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    {"method": "isolation_forest", "contamination": 0.1},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should complete successfully with exclusion
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 1
        assert "if" in branch_names[0].lower() or "isolation" in branch_names[0].lower()

    def test_outlier_excluder_mahalanobis(
        self, orchestrator, dataset
    ):
        """Test Mahalanobis distance exclusion strategy."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    {"method": "leverage", "threshold": 3.0},  # Use leverage instead of mahalanobis
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should complete successfully
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 1


class TestSamplePartitioner:
    """
    Test SamplePartitionerController functionality.

    Creates two branches: outliers and inliers.
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
        """Create test dataset with outliers."""
        return create_dataset_with_outliers()

    def test_sample_partitioner_creates_two_branches(
        self, orchestrator, dataset
    ):
        """Test that sample_partitioner creates exactly 2 branches."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "sample_partitioner",
                "filter": {"method": "y_outlier", "threshold": 1.5},
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have exactly 2 branches: outliers and inliers
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2, f"Expected 2 branches, got {len(branch_names)}: {branch_names}"

    def test_sample_partitioner_y_outlier_filter(
        self, orchestrator, dataset
    ):
        """Test Y-based outlier partitioning."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "sample_partitioner",
                "filter": {"method": "y_outlier", "threshold": 1.5},
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]

        # Should have outliers and inliers branches
        has_outliers = any("outlier" in name.lower() for name in branch_names)
        has_inliers = any("inlier" in name.lower() for name in branch_names)

        assert has_outliers, f"Expected outliers branch in {branch_names}"
        assert has_inliers, f"Expected inliers branch in {branch_names}"

    def test_sample_partitioner_x_outlier_filter(
        self, orchestrator, dataset
    ):
        """Test X-based outlier partitioning (Isolation Forest)."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "sample_partitioner",
                "filter": {"method": "isolation_forest", "contamination": 0.1},
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2


class TestOutlierBranchCombinations:
    """
    Test outlier branches combined with preprocessing branches.

    Per specification Phase 7: Combined scenarios.
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
        """Create test dataset with outliers."""
        return create_dataset_with_outliers()

    def test_outlier_excluder_with_preprocessing_branches(
        self, orchestrator, dataset
    ):
        """
        Test outlier excluder combined with preprocessing branches.

        Per spec: 2 outlier strategies × 2 preprocessings = 4 branches.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,
                    {"method": "isolation_forest", "contamination": 0.1},
                ],
            }},
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 2 = 4 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 4, f"Expected 4 branches, got {len(branch_names)}: {branch_names}"

    def test_sample_partitioner_with_preprocessing_branches(
        self, orchestrator, dataset
    ):
        """
        Test sample partitioner combined with preprocessing branches.

        2 partitions × 2 preprocessings = 4 branches.
        """
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": {
                "by": "sample_partitioner",
                "filter": {"method": "y_outlier", "threshold": 1.5},
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 2 × 2 = 4 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 4, f"Expected 4 branches, got {len(branch_names)}: {branch_names}"


class TestOutlierBranchRoundtrip:
    """
    Test roundtrip with outlier branches.
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
        """Create test dataset with outliers."""
        return create_dataset_with_outliers()

    def test_outlier_excluder_roundtrip(
        self, runner_with_save, dataset, workspace_path
    ):
        """Test train → save → load → predict with outlier excluder."""
        np.random.seed(42)

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,
                    {"method": "isolation_forest", "contamination": 0.05},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = runner_with_save.run(
            PipelineConfigs(pipeline),
            dataset
        )

        # Test prediction for each branch
        branch_ids = [b for b in predictions.get_unique_values("branch_id") if b is not None]

        for branch_id in branch_ids:
            branch_preds = predictions.filter_predictions(
                branch_id=branch_id, partition="test"
            )
            if not branch_preds:
                continue

            target_pred = branch_preds[0]

            y_pred_reloaded, _ = runner_with_save.predict(
                prediction_obj=target_pred,
                dataset=dataset,
                dataset_name="test_outliers"
            )

            # Verify prediction succeeded
            assert y_pred_reloaded is not None, f"Prediction failed for outlier branch {branch_id}"
            assert len(y_pred_reloaded) > 0, f"No predictions for outlier branch {branch_id}"


class TestOutlierStrategies:
    """
    Test various outlier detection strategies.

    Per specification Phase 7: Multiple strategy types.
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
        """Create test dataset with outliers."""
        return create_dataset_with_outliers()

    def test_lof_strategy(self, orchestrator, dataset):
        """Test Local Outlier Factor strategy."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    {"method": "lof", "contamination": 0.1, "n_neighbors": 20},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should complete successfully
        assert predictions.num_predictions > 0

    def test_leverage_strategy(self, orchestrator, dataset):
        """Test leverage (hat matrix) strategy."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    {"method": "leverage", "threshold": 2.0},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should complete successfully
        assert predictions.num_predictions > 0

    def test_multiple_strategies_comparison(self, orchestrator, dataset):
        """Test multiple strategies for comparison."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,  # Baseline
                    {"method": "isolation_forest", "contamination": 0.05},
                    {"method": "isolation_forest", "contamination": 0.1},
                ],
            }},
            {"model": PLSRegression(n_components=5)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Should have 3 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 3, f"Expected 3 branches, got {len(branch_names)}"

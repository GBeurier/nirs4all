"""
Integration tests for Branch Controller with Generator syntax (Phase 3).

Tests end-to-end scenarios with:
- _or_ generator inside branches
- _range_ generator inside branches
- Complex nested generator combinations
- Post-branch model execution
"""

import pytest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator


def create_simple_dataset(n_samples: int = 100, n_features: int = 50) -> SpectroDataset:
    """Create a simple synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="test_branch_gen")
    # Split 80/20 train/test
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset


class TestBranchGeneratorIntegration:
    """Integration tests for branching with generator syntax."""

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
        return create_simple_dataset()

    def test_branch_with_or_generator_creates_multiple_branches(
        self, orchestrator, dataset
    ):
        """Test that _or_ inside branch creates correct number of branches."""
        pipeline = [
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Each branch produces predictions - check we have 2 unique branch names
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2
        assert any("StandardScaler" in name for name in branch_names)
        assert any("MinMaxScaler" in name for name in branch_names)

    def test_branch_with_list_of_or_generators(
        self, orchestrator, dataset
    ):
        """Test multiple _or_ generators in a list."""
        pipeline = [
            {"branch": [
                {"_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]},
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches from _or_
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

    def test_branch_with_explicit_and_generator_branches(
        self, orchestrator, dataset
    ):
        """Test mixing explicit branches with generator syntax."""
        pipeline = [
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],  # Explicit branch
                {"_or_": [
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                    {"class": "sklearn.preprocessing.MaxAbsScaler"},
                ]},  # Generator expands to 2 branches
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 1 explicit + 2 from generator = 3 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 3

    def test_branch_names_are_descriptive(
        self, orchestrator, dataset
    ):
        """Test that branch names are derived from step class names."""
        pipeline = [
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # Get branch names from predictions
        branch_names = predictions.get_unique_values("branch_name")

        # Branch names should include class short names
        assert any("StandardScaler" in str(name) for name in branch_names if name)
        assert any("MinMaxScaler" in str(name) for name in branch_names if name)

    def test_branch_with_named_dict_and_generator(
        self, orchestrator, dataset
    ):
        """Test named branches mixed with generator."""
        pipeline = [
            {"branch": {
                "standard": [{"class": "sklearn.preprocessing.StandardScaler"}],
                "minmax": [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

        # Check named branches
        assert "standard" in branch_names
        assert "minmax" in branch_names

    def test_branch_with_multi_step_generator(
        self, orchestrator, dataset
    ):
        """Test generator producing multi-step branches."""
        pipeline = [
            {"branch": {
                "_or_": [
                    [
                        {"class": "sklearn.preprocessing.StandardScaler"},
                        {"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}},
                    ],
                    [
                        {"class": "sklearn.preprocessing.MinMaxScaler"},
                    ],
                ]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

    def test_branch_with_count_limit(
        self, orchestrator, dataset
    ):
        """Test generator with count limit."""
        pipeline = [
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                    {"class": "sklearn.preprocessing.MaxAbsScaler"},
                    {"class": "sklearn.preprocessing.RobustScaler"},
                ],
                "count": 2  # Limit to 2 branches
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches (limited by count)
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2


class TestBranchGeneratorWithCrossValidation:
    """Test branches with generators work with cross-validation."""

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
        return create_simple_dataset(n_samples=150)

    def test_branch_generator_with_shuffle_split(
        self, orchestrator, dataset
    ):
        """Test branch generator with ShuffleSplit cross-validation."""
        from sklearn.model_selection import ShuffleSplit

        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

    def test_branch_generator_with_kfold(
        self, orchestrator, dataset
    ):
        """Test branch generator with KFold cross-validation."""
        from sklearn.model_selection import KFold

        pipeline = [
            KFold(n_splits=3, shuffle=True, random_state=42),
            {"branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = orchestrator.execute(
            pipeline=pipeline,
            dataset=dataset
        )

        # 2 branches
        branch_names = [b for b in predictions.get_unique_values("branch_name") if b]
        assert len(branch_names) == 2

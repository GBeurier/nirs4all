"""
Integration tests for new unified branch modes (Phase 4).

Tests the new separation branch syntax:
- Mode detection (duplication vs separation)
- by_metadata separation branches (basic)
- Legacy pattern error handling

Note: Full separation branch testing with merge will be in Phase 5.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.runner import PipelineRunner


def create_simple_dataset() -> SpectroDataset:
    """Create a simple single-source test dataset."""
    np.random.seed(42)

    n_samples = 100
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    dataset = SpectroDataset(name="simple_test")
    dataset.add_samples(X[:80], indexes={"partition": "train"})
    dataset.add_samples(X[80:], indexes={"partition": "test"})
    dataset.add_targets(y[:80])
    dataset.add_targets(y[80:])

    return dataset

def create_metadata_dataset() -> SpectroDataset:
    """Create a dataset with metadata for per-site testing."""
    np.random.seed(42)

    n_samples = 120
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    # Different y distributions per site
    sites = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    site_effects = {"A": 0, "B": 5, "C": -3}
    y = np.array([np.sum(X[i, :3]) + site_effects[sites[i]] for i in range(n_samples)])
    y += np.random.randn(n_samples) * 0.5

    dataset = SpectroDataset(name="metadata_test")
    # Split 80/20 train/test per site
    train_mask = np.array([True] * 32 + [False] * 8 + [True] * 32 + [False] * 8 + [True] * 32 + [False] * 8)
    test_mask = ~train_mask

    dataset.add_samples(X[train_mask], indexes={"partition": "train"})
    dataset.add_samples(X[test_mask], indexes={"partition": "test"})
    dataset.add_targets(y[train_mask])
    dataset.add_targets(y[test_mask])

    # Add metadata
    metadata_df = pd.DataFrame({
        "site": np.concatenate([sites[train_mask], sites[test_mask]]),
        "sample_id": list(range(n_samples)),
    })
    dataset.add_metadata(metadata_df)

    return dataset

class TestDuplicationBranchesStillWork:
    """Test that existing duplication branch syntax still works."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=True,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        return create_simple_dataset()

    def test_list_syntax_duplication_branch(self, runner, dataset):
        """Test list syntax duplication branch still works."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": [
                [StandardScaler()],
                [MinMaxScaler()],
            ]},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

    def test_named_dict_syntax_duplication_branch(self, runner, dataset):
        """Test named dict syntax duplication branch still works."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {"branch": {
                "standard": [StandardScaler()],
                "minmax": [MinMaxScaler()],
            }},
            {"model": Ridge(alpha=1.0)},
        ]

        predictions, _ = runner.run(
            PipelineConfigs(pipeline),
            dataset
        )

        assert predictions is not None
        assert len(predictions) > 0

class TestModeDetection:
    """Test branch mode detection logic."""

    def test_list_syntax_detected_as_duplication(self):
        """Verify list syntax triggers duplication mode."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()

        # List of lists is duplication
        assert ctrl._detect_branch_mode([[]]) == "duplication"
        assert ctrl._detect_branch_mode([["step1"], ["step2"]]) == "duplication"

    def test_by_tag_detected_as_separation(self):
        """Verify by_tag syntax triggers separation mode."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()
        assert ctrl._detect_branch_mode({"by_tag": "outlier"}) == "separation"

    def test_by_metadata_detected_as_separation(self):
        """Verify by_metadata syntax triggers separation mode."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()
        assert ctrl._detect_branch_mode({"by_metadata": "site"}) == "separation"

    def test_by_filter_detected_as_separation(self):
        """Verify by_filter syntax triggers separation mode."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()
        assert ctrl._detect_branch_mode({"by_filter": "some_filter"}) == "separation"

    def test_by_source_detected_as_separation(self):
        """Verify by_source syntax triggers separation mode."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()
        assert ctrl._detect_branch_mode({"by_source": True}) == "separation"

    def test_named_dict_detected_as_duplication(self):
        """Verify named dict (without by_*) is duplication."""
        from nirs4all.controllers.data.branch import BranchController

        ctrl = BranchController()
        assert ctrl._detect_branch_mode({"snv": [], "msc": []}) == "duplication"

class TestLegacyPatternErrors:
    """Test that legacy 'by' patterns raise clear errors."""

    @pytest.fixture
    def workspace_path(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        return workspace

    @pytest.fixture
    def runner(self, workspace_path):
        return PipelineRunner(
            workspace_path=workspace_path,
            save_artifacts=False,
            verbose=0,
            enable_tab_reports=False,
            show_spinner=False
        )

    @pytest.fixture
    def dataset(self):
        return create_simple_dataset()

    def test_outlier_excluder_pattern_raises(self, runner, dataset):
        """Test legacy outlier_excluder pattern raises error."""
        pipeline = [
            ShuffleSplit(n_splits=2),
            {"branch": {"by": "outlier_excluder"}},
            {"model": Ridge()},
        ]

        # Error is wrapped in RuntimeError by pipeline executor
        with pytest.raises((ValueError, RuntimeError), match="no longer supported"):
            runner.run(PipelineConfigs(pipeline), dataset)

    def test_sample_partitioner_pattern_raises(self, runner, dataset):
        """Test legacy sample_partitioner pattern raises error."""
        pipeline = [
            ShuffleSplit(n_splits=2),
            {"branch": {"by": "sample_partitioner"}},
            {"model": Ridge()},
        ]

        with pytest.raises((ValueError, RuntimeError), match="no longer supported"):
            runner.run(PipelineConfigs(pipeline), dataset)

    def test_metadata_partitioner_pattern_raises(self, runner, dataset):
        """Test legacy metadata_partitioner pattern raises error."""
        pipeline = [
            ShuffleSplit(n_splits=2),
            {"branch": {"by": "metadata_partitioner"}},
            {"model": Ridge()},
        ]

        with pytest.raises((ValueError, RuntimeError), match="no longer supported"):
            runner.run(PipelineConfigs(pipeline), dataset)

class TestBranchControllerMatches:
    """Test BranchController.matches() method."""

    def test_matches_branch_keyword(self):
        """Controller should match 'branch' keyword."""
        from nirs4all.controllers.data.branch import BranchController

        step = {"branch": [[StandardScaler()], [MinMaxScaler()]]}
        assert BranchController.matches(step, None, "branch") is True

    def test_not_matches_other_keywords(self):
        """Controller should not match other keywords."""
        from nirs4all.controllers.data.branch import BranchController

        assert BranchController.matches({}, None, "model") is False
        assert BranchController.matches({}, None, "preprocessing") is False
        assert BranchController.matches({}, None, "merge") is False

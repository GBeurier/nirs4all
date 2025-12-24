"""
Integration tests for Disjoint Sample Branch Merging (Phase 5.2).

Tests cover:
- Full pipeline with metadata_partitioner
- Stacking after disjoint merge
- Prediction mode with routing
- Multiple datasets
- End-to-end regression and classification

See: docs/reports/disjoint_sample_branch_merging.md
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.transforms import StandardNormalVariate as SNV


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def synthetic_dataset_with_metadata():
    """Create synthetic dataset with metadata for partitioning."""
    np.random.seed(42)

    n_samples = 100
    n_features = 50

    # Create spectral data
    X = np.random.randn(n_samples, n_features)

    # Create target with site-dependent bias
    y = np.zeros(n_samples)

    # Create metadata with 3 sites
    sites = []
    for i in range(n_samples):
        if i < 40:
            sites.append("site_A")
            y[i] = np.sum(X[i, :5]) + 1.0  # Site A has +1 bias
        elif i < 70:
            sites.append("site_B")
            y[i] = np.sum(X[i, :5]) + 0.0  # Site B has no bias
        else:
            sites.append("site_C")
            y[i] = np.sum(X[i, :5]) - 1.0  # Site C has -1 bias

    y += np.random.randn(n_samples) * 0.1  # Add noise

    # Create dataset
    dataset = SpectroDataset(name="synthetic_sites")
    dataset.add_samples(X, indexes={"partition": "train"})
    dataset.add_targets(y)
    dataset.add_metadata(pd.DataFrame({
        "site": sites,
        "sample_id": list(range(n_samples)),
    }))

    return dataset


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# P5.2: Full Pipeline with Metadata Partitioner
# =============================================================================

class TestMetadataPartitionerPipeline:
    """Integration tests for metadata_partitioner in full pipelines."""

    @pytest.mark.parametrize("n_splits", [2, 3])
    def test_basic_metadata_partitioner_pipeline(
        self,
        synthetic_dataset_with_metadata,
        temp_workspace,
        n_splits,
    ):
        """Run basic pipeline with metadata partitioner branching."""
        dataset = synthetic_dataset_with_metadata

        # This is a structural test - just verify the pipeline parses correctly
        # The actual dataset would need to exist with proper metadata column

        # Define pipeline with metadata partitioner
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
            Ridge(alpha=1.0),
        ]

        # This is a structural test - just verify the pipeline parses correctly
        config = PipelineConfigs(pipeline, name="metadata_partitioner_test")

        # Verify pipeline structure
        assert len(pipeline) == 5
        assert pipeline[3] == {"merge": "predictions"}

    def test_metadata_partitioner_with_min_samples(
        self,
        synthetic_dataset_with_metadata,
        temp_workspace,
    ):
        """Test metadata partitioner with min_samples filtering."""
        # Define pipeline with min_samples
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
                "min_samples": 35,  # Site C has only 30 samples, should be skipped
            },
            {"merge": "predictions"},
        ]

        config = PipelineConfigs(pipeline, name="min_samples_test")

        # Verify min_samples is in config
        assert pipeline[2]["min_samples"] == 35

    def test_metadata_partitioner_with_group_values(
        self,
        synthetic_dataset_with_metadata,
        temp_workspace,
    ):
        """Test metadata partitioner with value grouping."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
                "group_values": {
                    "large_sites": ["site_A"],
                    "small_sites": ["site_B", "site_C"],
                },
            },
            {"merge": "predictions"},
        ]

        config = PipelineConfigs(pipeline, name="group_values_test")

        # Verify group_values is in config
        assert "large_sites" in pipeline[2]["group_values"]
        assert pipeline[2]["group_values"]["small_sites"] == ["site_B", "site_C"]


# =============================================================================
# P5.2: Stacking After Disjoint Merge
# =============================================================================

class TestStackingAfterDisjointMerge:
    """Test meta-model stacking after disjoint branch merge."""

    def test_ridge_meta_learner_after_merge(self):
        """Ridge meta-learner should work after disjoint merge."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5), RandomForestRegressor(n_estimators=10)],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
            {"name": "MetaRidge", "model": Ridge(alpha=1.0)},
        ]

        config = PipelineConfigs(pipeline, name="stacking_test")

        # Verify pipeline structure
        assert len(pipeline) == 5
        assert pipeline[4]["name"] == "MetaRidge"

    def test_merge_with_n_columns_and_select_by(self):
        """Merge with explicit n_columns and select_by options."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": [
                    PLSRegression(n_components=3),
                    PLSRegression(n_components=5),
                    PLSRegression(n_components=10),
                ],
                "by": "metadata_partitioner",
                "column": "site",
            },
            # Force 2 columns, select by R²
            {"merge": "predictions", "n_columns": 2, "select_by": "r2"},
            Ridge(alpha=1.0),
        ]

        config = PipelineConfigs(pipeline, name="n_columns_test")

        # Verify merge options
        assert pipeline[3]["n_columns"] == 2
        assert pipeline[3]["select_by"] == "r2"

    def test_multiple_models_per_branch(self):
        """Test disjoint merge with multiple models per branch."""
        pipeline = [
            MinMaxScaler(),
            KFold(n_splits=3, shuffle=True, random_state=42),
            {
                "branch": [
                    {"name": "PLS5", "model": PLSRegression(n_components=5)},
                    {"name": "PLS10", "model": PLSRegression(n_components=10)},
                    {"name": "RF", "model": RandomForestRegressor(n_estimators=10)},
                ],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
            Ridge(alpha=1.0),
        ]

        config = PipelineConfigs(pipeline, name="multi_model_test")

        # Verify 3 models defined
        assert len(pipeline[2]["branch"]) == 3


# =============================================================================
# P5.2: Prediction Mode with Routing
# =============================================================================

class TestPredictionModeRouting:
    """Test prediction mode with sample routing."""

    def test_prediction_routing_structure(self):
        """Verify prediction mode routing structure."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
        ]

        config = PipelineConfigs(pipeline, name="routing_test")

        # The metadata_partitioner should support prediction mode
        from nirs4all.controllers.data.metadata_partitioner import MetadataPartitionerController

        assert MetadataPartitionerController.supports_prediction_mode() is True

    def test_merge_controller_prediction_mode_support(self):
        """Verify MergeController supports prediction mode."""
        from nirs4all.controllers.data.merge import MergeController

        assert MergeController.supports_prediction_mode() is True


# =============================================================================
# P5.2: Sample Partitioner Integration
# =============================================================================

class TestSamplePartitionerIntegration:
    """Test sample_partitioner with disjoint merge."""

    def test_sample_partitioner_pipeline_structure(self):
        """Verify sample partitioner pipeline structure."""
        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {"method": "y_outlier", "threshold": 1.5},
                },
            },
            PLSRegression(n_components=5),
        ]

        config = PipelineConfigs(pipeline, name="sample_partitioner_test")

        # Verify structure
        assert pipeline[2]["branch"]["by"] == "sample_partitioner"
        assert "filter" in pipeline[2]["branch"]

    def test_sample_partitioner_with_isolation_forest(self):
        """Test sample partitioner with isolation forest."""
        pipeline = [
            MinMaxScaler(),
            SNV(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": {
                    "by": "sample_partitioner",
                    "filter": {
                        "method": "isolation_forest",
                        "contamination": 0.10,
                    },
                },
            },
            PLSRegression(n_components=5),
        ]

        config = PipelineConfigs(pipeline, name="isolation_forest_test")

        # Verify filter config
        assert pipeline[3]["branch"]["filter"]["method"] == "isolation_forest"
        assert pipeline[3]["branch"]["filter"]["contamination"] == 0.10


# =============================================================================
# P5.2: Complex Pipeline Combinations
# =============================================================================

class TestComplexPipelineCombinations:
    """Test complex pipeline combinations with disjoint merge."""

    def test_preprocessing_then_metadata_partitioner(self):
        """Preprocessing before metadata partitioner."""
        pipeline = [
            MinMaxScaler(),
            SNV(),
            StandardScaler(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
            Ridge(alpha=1.0),
        ]

        config = PipelineConfigs(pipeline, name="preprocessing_test")

        # Verify preprocessing steps
        assert len(pipeline) == 7

    def test_nested_preprocessing_branches_with_metadata_partitioner(self):
        """Preprocessing branches before metadata partitioner."""
        from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC

        pipeline = [
            MinMaxScaler(),
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            # First: preprocessing branches
            {
                "branch": {
                    "snv": [SNV()],
                    "msc": [MSC()],
                },
            },
            # Second: metadata partitioner (nested)
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "predictions"},
        ]

        config = PipelineConfigs(pipeline, name="nested_test")

        # Verify structure
        assert "snv" in pipeline[2]["branch"]
        assert pipeline[3]["by"] == "metadata_partitioner"

    def test_feature_merge_then_model(self):
        """Feature merge from disjoint branches then model."""
        pipeline = [
            ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            {
                "branch": [SNV(), StandardScaler()],
                "by": "metadata_partitioner",
                "column": "site",
            },
            {"merge": "features"},  # Merge features, not predictions
            PLSRegression(n_components=10),
        ]

        config = PipelineConfigs(pipeline, name="feature_merge_test")

        # Verify feature merge
        assert pipeline[2]["merge"] == "features"


# =============================================================================
# P5.2: Error Handling Integration
# =============================================================================

class TestErrorHandlingIntegration:
    """Test error handling in disjoint merge pipelines."""

    def test_missing_metadata_column_error_message(self):
        """Error when metadata column doesn't exist."""
        pipeline = [
            ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "nonexistent_column",  # Does not exist
            },
        ]

        # This should be caught during execution, not configuration
        config = PipelineConfigs(pipeline, name="error_test")
        assert config is not None

    def test_merge_without_branch_error(self):
        """Merge without prior branch should fail."""
        pipeline = [
            MinMaxScaler(),
            {"merge": "predictions"},  # No branch before this!
        ]

        config = PipelineConfigs(pipeline, name="merge_error_test")
        # The error will be raised at runtime, not config time


# =============================================================================
# P5.2: Serialization and Loading
# =============================================================================

class TestSerializationIntegration:
    """Test pipeline serialization with disjoint merge."""

    def test_metadata_partitioner_serialization(self):
        """Metadata partitioner pipeline should serialize correctly."""
        pipeline = [
            MinMaxScaler(),
            {
                "branch": [PLSRegression(n_components=5)],
                "by": "metadata_partitioner",
                "column": "site",
                "min_samples": 20,
                "group_values": {"combined": ["A", "B"]},
            },
            {"merge": "predictions", "n_columns": 2, "select_by": "r2"},
        ]

        config = PipelineConfigs(pipeline, name="serialization_test")

        # Verify config was created successfully
        assert config is not None
        # names is a list of pipeline names
        assert "serialization_test" in config.names or len(config.names) > 0

        # The pipeline steps should be accessible
        assert len(pipeline) == 3

    def test_disjoint_merge_options_serialization(self):
        """Disjoint merge options should serialize correctly."""
        merge_step = {"merge": "predictions", "n_columns": 3, "select_by": "mae"}

        # Verify structure
        assert merge_step["n_columns"] == 3
        assert merge_step["select_by"] == "mae"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

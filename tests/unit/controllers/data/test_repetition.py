"""
Unit tests for repetition transformation controllers.

Tests for rep_to_sources and rep_to_pp pipeline keywords that transform
spectral repetitions into separate sources or additional preprocessings.
"""

import numpy as np
import pytest

from nirs4all.controllers.data.repetition import RepToPPController, RepToSourcesController
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.data.repetition import RepetitionConfig, UnequelRepsStrategy

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_repetition_dataset():
    """Create dataset with 4 reps per sample (30 unique samples = 120 total)."""
    np.random.seed(42)
    n_unique = 30
    n_reps = 4
    n_features = 100

    X = np.random.randn(n_unique * n_reps, n_features)
    y = np.repeat(np.arange(n_unique, dtype=float), n_reps)
    sample_ids = np.repeat(np.arange(n_unique), n_reps)

    ds = SpectroDataset("test_reps")
    ds.add_samples(X, {"partition": "train"})
    ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])
    ds.add_targets(y)

    return ds

@pytest.fixture
def unequal_repetition_dataset():
    """Create dataset with unequal reps: 4, 4, 3 reps for 3 samples."""
    np.random.seed(42)
    n_features = 50

    X = np.random.randn(11, n_features)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=float)
    sample_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

    ds = SpectroDataset("unequal_reps")
    ds.add_samples(X, {"partition": "train"})
    ds.add_metadata(np.array(sample_ids).reshape(-1, 1), headers=["Sample_ID"])
    ds.add_targets(y)

    return ds

@pytest.fixture
def multi_source_dataset():
    """Create dataset with 2 sources and 4 reps per sample."""
    np.random.seed(42)
    n_unique = 10
    n_reps = 4
    n_features_1 = 100
    n_features_2 = 50

    X1 = np.random.randn(n_unique * n_reps, n_features_1)
    X2 = np.random.randn(n_unique * n_reps, n_features_2)
    y = np.repeat(np.arange(n_unique, dtype=float), n_reps)
    sample_ids = np.repeat(np.arange(n_unique), n_reps)

    ds = SpectroDataset("multi_source")
    # Pass as list for multi-source
    ds.add_samples([X1, X2], {"partition": "train"})
    ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])
    ds.add_targets(y)

    return ds

# ============================================================================
# RepetitionConfig Tests
# ============================================================================

class TestRepetitionConfig:
    """Tests for RepetitionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RepetitionConfig()
        assert config.column is None
        assert config.on_unequal == "error"
        assert config.expected_reps is None
        assert config.preserve_order is True
        assert config.uses_dataset_aggregate is True

    def test_from_step_value_string(self):
        """Test parsing from string column name."""
        config = RepetitionConfig.from_step_value("Sample_ID")
        assert config.column == "Sample_ID"
        assert not config.uses_dataset_aggregate

    def test_from_step_value_true(self):
        """Test parsing from True (use aggregate)."""
        config = RepetitionConfig.from_step_value(True)
        assert config.column is None
        assert config.uses_dataset_aggregate

    def test_from_step_value_none(self):
        """Test parsing from None (use aggregate)."""
        config = RepetitionConfig.from_step_value(None)
        assert config.column is None
        assert config.uses_dataset_aggregate

    def test_from_step_value_dict(self):
        """Test parsing from dict with options."""
        config = RepetitionConfig.from_step_value({
            "column": "Sample_ID",
            "on_unequal": "drop",
            "expected_reps": 4
        })
        assert config.column == "Sample_ID"
        assert config.on_unequal == "drop"
        assert config.expected_reps == 4

    def test_resolve_column_explicit(self):
        """Test column resolution with explicit column."""
        config = RepetitionConfig(column="Sample_ID")
        assert config.resolve_column("other_col") == "Sample_ID"

    def test_resolve_column_from_aggregate(self):
        """Test column resolution from dataset aggregate."""
        config = RepetitionConfig(column=None)
        assert config.resolve_column("Sample_ID") == "Sample_ID"

    def test_resolve_column_no_aggregate_error(self):
        """Test error when no column and no aggregate."""
        config = RepetitionConfig(column=None)
        with pytest.raises(ValueError, match="REP-E001"):
            config.resolve_column(None)

    def test_get_source_name_default(self):
        """Test default source naming."""
        config = RepetitionConfig()
        assert config.get_source_name(0) == "rep_0"
        assert config.get_source_name(2) == "rep_2"

    def test_get_source_name_template(self):
        """Test custom source naming template."""
        config = RepetitionConfig(source_names="measurement_{i}")
        assert config.get_source_name(0) == "measurement_0"
        assert config.get_source_name(3) == "measurement_3"

    def test_get_pp_name_default(self):
        """Test default preprocessing naming."""
        config = RepetitionConfig()
        assert config.get_pp_name(0, "raw") == "raw_rep0"
        assert config.get_pp_name(2, "snv") == "snv_rep2"

    def test_get_pp_name_template(self):
        """Test custom preprocessing naming template."""
        config = RepetitionConfig(pp_names="{pp}_scan{i}")
        assert config.get_pp_name(0, "raw") == "raw_scan0"
        assert config.get_pp_name(1, "snv") == "snv_scan1"

    def test_invalid_on_unequal(self):
        """Test validation of on_unequal parameter."""
        with pytest.raises(ValueError, match="on_unequal"):
            RepetitionConfig(on_unequal="invalid")

    def test_invalid_expected_reps(self):
        """Test validation of expected_reps parameter."""
        with pytest.raises(ValueError, match="expected_reps"):
            RepetitionConfig(expected_reps=0)
        with pytest.raises(ValueError, match="expected_reps"):
            RepetitionConfig(expected_reps=-1)

# ============================================================================
# Dataset Method Tests: _get_sample_groups
# ============================================================================

class TestGetSampleGroups:
    """Tests for SpectroDataset._get_sample_groups method."""

    def test_basic_grouping(self, basic_repetition_dataset):
        """Test basic grouping by metadata column."""
        ds = basic_repetition_dataset
        groups = ds._get_sample_groups("Sample_ID")

        assert len(groups) == 30  # 30 unique samples
        assert all(len(v) == 4 for v in groups.values())  # 4 reps each

    def test_grouping_preserves_indices(self, basic_repetition_dataset):
        """Test that grouping returns correct row indices."""
        ds = basic_repetition_dataset
        groups = ds._get_sample_groups("Sample_ID")

        # First sample (ID=0) should have indices 0,1,2,3
        assert groups[0] == [0, 1, 2, 3]
        # Second sample (ID=1) should have indices 4,5,6,7
        assert groups[1] == [4, 5, 6, 7]

    def test_grouping_by_y(self, basic_repetition_dataset):
        """Test grouping by target values."""
        ds = basic_repetition_dataset
        groups = ds._get_sample_groups("y")

        assert len(groups) == 30  # 30 unique target values
        assert all(len(v) == 4 for v in groups.values())

    def test_column_not_found_error(self, basic_repetition_dataset):
        """Test error when column doesn't exist."""
        ds = basic_repetition_dataset
        with pytest.raises(ValueError, match="REP-E001"):
            ds._get_sample_groups("NonExistent")

    def test_y_grouping_no_targets_error(self):
        """Test error when grouping by y without targets."""
        ds = SpectroDataset("no_targets")
        ds.add_samples(np.random.randn(10, 50), {"partition": "train"})

        with pytest.raises(ValueError, match="REP-E003"):
            ds._get_sample_groups("y")

# ============================================================================
# Dataset Method Tests: reshape_reps_to_sources
# ============================================================================

class TestReshapeRepsToSources:
    """Tests for SpectroDataset.reshape_reps_to_sources method."""

    def test_basic_transformation(self, basic_repetition_dataset):
        """Test basic rep_to_sources transformation."""
        ds = basic_repetition_dataset

        assert ds.num_samples == 120
        assert ds.n_sources == 1

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        assert ds.num_samples == 30
        assert ds.n_sources == 4

    def test_feature_shapes(self, basic_repetition_dataset):
        """Test that feature shapes are correct after transformation."""
        ds = basic_repetition_dataset
        original_features = ds.x({}).shape[1]  # 100 features

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        X_sources = ds.x({}, concat_source=False)
        assert len(X_sources) == 4
        for X_src in X_sources:
            assert X_src.shape == (30, original_features)

    def test_data_integrity(self):
        """Test that data values are correctly mapped."""
        np.random.seed(123)
        n_unique = 5
        n_reps = 3
        n_features = 20

        # Create deterministic data
        X = np.zeros((n_unique * n_reps, n_features))
        for sample_idx in range(n_unique):
            for rep_idx in range(n_reps):
                idx = sample_idx * n_reps + rep_idx
                X[idx] = sample_idx * 100 + rep_idx

        sample_ids = np.repeat(np.arange(n_unique), n_reps)

        ds = SpectroDataset("integrity_test")
        ds.add_samples(X, {"partition": "train"})
        ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        X_sources = ds.x({}, concat_source=False)

        # Source 0 should have rep 0 from each sample
        assert np.allclose(X_sources[0][0], 0)    # Sample 0, rep 0
        assert np.allclose(X_sources[0][1], 100)  # Sample 1, rep 0

        # Source 1 should have rep 1 from each sample
        assert np.allclose(X_sources[1][0], 1)    # Sample 0, rep 1
        assert np.allclose(X_sources[1][1], 101)  # Sample 1, rep 1

    def test_multi_source_transformation(self, multi_source_dataset):
        """Test transformation with multiple existing sources."""
        ds = multi_source_dataset

        assert ds.num_samples == 40
        assert ds.n_sources == 2

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        # 2 sources × 4 reps = 8 sources
        assert ds.num_samples == 10
        assert ds.n_sources == 8

    def test_using_aggregate(self, basic_repetition_dataset):
        """Test transformation using dataset.aggregate."""
        ds = basic_repetition_dataset
        ds.set_aggregate("Sample_ID")

        config = RepetitionConfig(column=None)  # Use aggregate
        ds.reshape_reps_to_sources(config)

        assert ds.num_samples == 30
        assert ds.n_sources == 4

# ============================================================================
# Dataset Method Tests: reshape_reps_to_preprocessings
# ============================================================================

class TestReshapeRepsToPreprocessings:
    """Tests for SpectroDataset.reshape_reps_to_preprocessings method."""

    def test_basic_transformation(self, basic_repetition_dataset):
        """Test basic rep_to_pp transformation."""
        ds = basic_repetition_dataset

        assert ds.num_samples == 120
        assert len(ds.features_processings(0)) == 1

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_preprocessings(config)

        assert ds.num_samples == 30
        assert len(ds.features_processings(0)) == 4

    def test_feature_shapes(self, basic_repetition_dataset):
        """Test that feature shapes are correct after transformation."""
        ds = basic_repetition_dataset
        original_features = ds.x({}).shape[1]  # 100 features

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_preprocessings(config)

        # 2D: (30, 400) - all pp concatenated
        X_2d = ds.x({})
        assert X_2d.shape == (30, original_features * 4)

        # 3D: (30, 4, 100)
        X_3d = ds.x({}, layout="3d")
        assert X_3d.shape == (30, 4, original_features)

    def test_processing_names(self, basic_repetition_dataset):
        """Test that preprocessing names are generated correctly."""
        ds = basic_repetition_dataset

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_preprocessings(config)

        pp_names = ds.features_processings(0)
        assert pp_names == ["raw_rep0", "raw_rep1", "raw_rep2", "raw_rep3"]

    def test_multi_source_transformation(self, multi_source_dataset):
        """Test transformation with multiple existing sources."""
        ds = multi_source_dataset

        assert ds.num_samples == 40
        assert ds.n_sources == 2

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_preprocessings(config)

        # Still 2 sources, but each has 4 preprocessings
        assert ds.num_samples == 10
        assert ds.n_sources == 2
        assert len(ds.features_processings(0)) == 4
        assert len(ds.features_processings(1)) == 4

# ============================================================================
# Unequal Repetitions Tests
# ============================================================================

class TestUnequelRepetitions:
    """Tests for handling samples with unequal repetition counts."""

    def test_error_strategy(self, unequal_repetition_dataset):
        """Test that error strategy raises on unequal reps."""
        ds = unequal_repetition_dataset

        config = RepetitionConfig(column="Sample_ID", on_unequal="error")
        with pytest.raises(ValueError, match="REP-E002"):
            ds.reshape_reps_to_sources(config)

    def test_drop_strategy(self, unequal_repetition_dataset):
        """Test that drop strategy keeps only matching samples."""
        ds = unequal_repetition_dataset

        config = RepetitionConfig(column="Sample_ID", on_unequal="drop", expected_reps=4)
        ds.reshape_reps_to_sources(config)

        # Only 2 samples with 4 reps remain
        assert ds.num_samples == 2
        assert ds.n_sources == 4

    def test_pad_strategy(self, unequal_repetition_dataset):
        """Test that pad strategy fills missing reps with NaN."""
        ds = unequal_repetition_dataset

        config = RepetitionConfig(column="Sample_ID", on_unequal="pad")
        ds.reshape_reps_to_sources(config)

        # All 3 samples kept
        assert ds.num_samples == 3
        assert ds.n_sources == 4

        # Last source should have NaN for sample 2
        X_sources = ds.x({}, concat_source=False)
        assert np.isnan(X_sources[3][2]).all()  # Sample 2 (idx 2) in source 3

    def test_truncate_strategy(self, unequal_repetition_dataset):
        """Test that truncate strategy uses minimum count."""
        ds = unequal_repetition_dataset

        config = RepetitionConfig(column="Sample_ID", on_unequal="truncate")
        ds.reshape_reps_to_sources(config)

        # All samples kept, but only 3 sources (minimum reps)
        assert ds.num_samples == 3
        assert ds.n_sources == 3

# ============================================================================
# Controller Tests
# ============================================================================

class TestRepToSourcesController:
    """Tests for RepToSourcesController."""

    def test_matches_keyword(self):
        """Test that controller matches correct keyword."""
        assert RepToSourcesController.matches({}, None, "rep_to_sources")
        assert not RepToSourcesController.matches({}, None, "rep_to_pp")
        assert not RepToSourcesController.matches({}, None, "other")

    def test_supports_prediction_mode(self):
        """Test that controller doesn't run in prediction mode."""
        assert not RepToSourcesController.supports_prediction_mode()

    def test_use_multi_source(self):
        """Test that controller operates on whole dataset."""
        assert not RepToSourcesController.use_multi_source()

class TestRepToPPController:
    """Tests for RepToPPController."""

    def test_matches_keyword(self):
        """Test that controller matches correct keyword."""
        assert RepToPPController.matches({}, None, "rep_to_pp")
        assert not RepToPPController.matches({}, None, "rep_to_sources")
        assert not RepToPPController.matches({}, None, "other")

    def test_supports_prediction_mode(self):
        """Test that controller doesn't run in prediction mode."""
        assert not RepToPPController.supports_prediction_mode()

    def test_use_multi_source(self):
        """Test that controller operates on whole dataset."""
        assert not RepToPPController.use_multi_source()

# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_repetition(self):
        """Test with single repetition per sample (no-op effectively)."""
        ds = SpectroDataset("single_rep")
        X = np.random.randn(30, 50)
        sample_ids = np.arange(30)  # Each sample is unique

        ds.add_samples(X, {"partition": "train"})
        ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        # 1 rep = 1 source, same sample count
        assert ds.num_samples == 30
        assert ds.n_sources == 1

    def test_large_repetition_count(self):
        """Test with many repetitions per sample."""
        n_reps = 20
        n_unique = 5
        n_features = 30

        ds = SpectroDataset("many_reps")
        X = np.random.randn(n_unique * n_reps, n_features)
        sample_ids = np.repeat(np.arange(n_unique), n_reps)

        ds.add_samples(X, {"partition": "train"})
        ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        assert ds.num_samples == n_unique
        assert ds.n_sources == n_reps

    def test_preserve_targets(self, basic_repetition_dataset):
        """Test that targets are preserved after transformation."""
        ds = basic_repetition_dataset
        original_unique_targets = set(ds.y({}).flatten())

        config = RepetitionConfig(column="Sample_ID")
        ds.reshape_reps_to_sources(config)

        # Targets should still exist and be unique per sample
        new_targets = set(ds.y({}).flatten())
        assert new_targets == original_unique_targets

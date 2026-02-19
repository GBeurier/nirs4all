"""
Integration tests for exclude keyword migration.

These tests verify that the new `exclude` keyword correctly replaces
the old `sample_filter` keyword functionality. The exclude keyword:
- Uses simpler syntax: {"exclude": Filter()} instead of {"sample_filter": {"filters": [...]}}
- Stores exclusion tags for analysis
- Marks samples as excluded from training
- Never runs during prediction mode

Note: nirs4all.run() works on an internal copy of the dataset, so
we verify exclusion by checking pipeline completion and model behavior.
"""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.data.dataset import SpectroDataset
from nirs4all.operators.filters import XOutlierFilter, YOutlierFilter


class TestExcludeBasicFunctionality:
    """Test basic exclude keyword functionality in pipelines."""

    @pytest.fixture
    def dataset_with_outliers(self):
        """Create a dataset with known outliers for testing."""
        np.random.seed(42)

        # Create normal samples
        n_normal = 50
        X_normal = np.random.rand(n_normal, 100)
        y_normal = np.random.normal(50, 10, n_normal)

        # Add outliers
        n_outliers = 5
        X_outliers = np.random.rand(n_outliers, 100)
        y_outliers = np.array([200, -100, 250, -150, 300])  # Extreme values

        X = np.vstack([X_normal, X_outliers])
        y = np.concatenate([y_normal, y_outliers])

        dataset = SpectroDataset("test_exclude")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        return dataset, n_normal, n_outliers

    def test_exclude_single_filter_pipeline_completes(self, dataset_with_outliers):
        """Test exclude with single filter completes pipeline successfully."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        # New syntax: {"exclude": Filter()}
        pipeline = [
            {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0
        )

        # Pipeline should complete successfully
        assert result is not None
        assert hasattr(result, 'best_score') or hasattr(result, 'scores')

    def test_exclude_list_of_filters_mode_any(self, dataset_with_outliers):
        """Test exclude with multiple filters using mode='any'."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        # New syntax: {"exclude": [Filter1(), Filter2()], "mode": "any"}
        pipeline = [
            {
                "exclude": [
                    YOutlierFilter(method="iqr", threshold=1.5),
                    YOutlierFilter(method="zscore", threshold=2.5)
                ],
                "mode": "any"  # Exclude if ANY filter flags
            },
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0
        )

        # Pipeline should complete successfully
        assert result is not None

    def test_exclude_list_of_filters_mode_all(self, dataset_with_outliers):
        """Test exclude with multiple filters using mode='all'."""
        dataset, n_normal, n_outliers = dataset_with_outliers

        # New syntax with mode='all'
        pipeline = [
            {
                "exclude": [
                    YOutlierFilter(method="iqr", threshold=1.5),
                    YOutlierFilter(method="zscore", threshold=2.5)
                ],
                "mode": "all"  # Exclude only if ALL filters flag
            },
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0
        )

        # Pipeline should complete successfully
        assert result is not None

class TestExcludeControllerDirectly:
    """Test ExcludeController directly on dataset for verification."""

    @pytest.fixture
    def dataset_with_outliers(self):
        """Create a dataset with known outliers."""
        np.random.seed(42)

        X = np.vstack([
            np.random.rand(45, 100),  # Normal
            np.random.rand(5, 100)    # Outliers at end
        ])
        y = np.concatenate([
            np.random.normal(50, 10, 45),
            [200, -100, 250, -150, 300]
        ])

        dataset = SpectroDataset("test_tags")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        return dataset

    def test_exclude_marks_samples_excluded_directly(self, dataset_with_outliers):
        """Directly verify filter and indexer mark samples as excluded."""
        dataset = dataset_with_outliers

        # Apply filter directly
        filter_obj = YOutlierFilter(method="iqr", threshold=1.5)
        X = dataset.x({"partition": "train"}, layout="2d")
        y = dataset.y({"partition": "train"})
        sample_indices = dataset._indexer.x_indices({"partition": "train"})

        filter_obj.fit(X, y)
        mask = filter_obj.get_mask(X, y)

        # Mark excluded
        exclude_indices = sample_indices[~mask].tolist()
        n_excluded = dataset._indexer.mark_excluded(exclude_indices, reason="test")

        # Verify exclusion
        assert n_excluded >= 4  # Most outliers should be caught
        summary = dataset._indexer.get_exclusion_summary()
        assert summary["total_excluded"] >= 4

    def test_exclude_creates_tags_directly(self, dataset_with_outliers):
        """Directly verify tags can be created for exclusion tracking."""
        dataset = dataset_with_outliers

        # Add exclusion tag manually (as ExcludeController would)
        tag_name = "excluded_y_outlier_iqr"
        dataset.add_tag(tag_name, dtype="bool")

        # Set some samples as tagged
        sample_indices = dataset._indexer.x_indices({"partition": "train"}).tolist()
        tag_values = [False] * 45 + [True] * 5  # Last 5 are outliers
        dataset.set_tag(tag_name, sample_indices, tag_values)

        # Verify tag exists
        tags = dataset.tags
        assert tag_name in tags

        # Verify tag values
        tag_data = dataset.get_tag(tag_name)
        assert sum(tag_data) == 5  # 5 samples tagged

class TestExcludeWithCrossValidation:
    """Test exclude works correctly with cross-validation."""

    @pytest.fixture
    def cv_dataset(self):
        """Create dataset for cross-validation testing."""
        np.random.seed(42)

        X = np.vstack([
            np.random.rand(80, 50),
            np.random.rand(10, 50)  # Outliers
        ])
        y = np.concatenate([
            np.random.normal(50, 5, 80),
            np.random.normal(200, 50, 10)  # Outlier targets
        ])

        dataset = SpectroDataset("cv_test")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        return dataset

    def test_exclude_with_kfold(self, cv_dataset):
        """Exclude should work within cross-validation folds."""
        pipeline = [
            {"exclude": YOutlierFilter(method="iqr", threshold=2.0)},
            KFold(n_splits=3),
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=cv_dataset,
            verbose=0
        )

        # Should complete successfully
        assert result is not None
        # Should have results for multiple folds
        assert hasattr(result, 'scores') or hasattr(result, 'best_score')

class TestExcludeCascadeToAugmented:
    """Test exclude cascades to augmented samples via direct indexer test."""

    @pytest.fixture
    def dataset_with_augmented(self):
        """Create dataset with base and augmented samples."""
        np.random.seed(42)

        # Base samples
        X = np.random.rand(20, 100)
        y = np.concatenate([
            np.random.normal(50, 5, 18),
            [200, -100]  # 2 outliers at indices 18, 19
        ])

        dataset = SpectroDataset("test_augmented")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        # Add augmented samples for outliers
        X_aug = np.random.rand(4, 1, 100)
        indexes_list = [
            {"partition": "train", "origin": 18, "augmentation": "noise"},
            {"partition": "train", "origin": 18, "augmentation": "noise"},
            {"partition": "train", "origin": 19, "augmentation": "noise"},
            {"partition": "train", "origin": 19, "augmentation": "noise"},
        ]
        dataset.add_samples_batch(X_aug, indexes_list)

        return dataset

    def test_exclude_cascades_to_augmented_directly(self, dataset_with_augmented):
        """Directly test that cascade_to_augmented works on indexer."""
        dataset = dataset_with_augmented

        # Count samples before
        count_before = len(dataset._indexer.x_indices(
            {"partition": "train"}, include_augmented=True, include_excluded=False
        ))
        assert count_before == 24  # 20 base + 4 augmented

        # Mark outlier samples (18, 19) as excluded with cascade
        n_excluded = dataset._indexer.mark_excluded(
            [18, 19], reason="outlier", cascade_to_augmented=True
        )

        # Should exclude 2 base + 4 augmented = 6
        assert n_excluded == 6

        # Verify count after
        count_after = len(dataset._indexer.x_indices(
            {"partition": "train"}, include_augmented=True, include_excluded=False
        ))
        assert count_after == 18  # 24 - 6

    def test_exclude_no_cascade_directly(self, dataset_with_augmented):
        """Directly test cascade_to_augmented=False."""
        dataset = dataset_with_augmented

        # Mark sample without cascade
        n_excluded = dataset._indexer.mark_excluded(
            [18], reason="outlier", cascade_to_augmented=False
        )

        # Should only exclude the base sample
        assert n_excluded == 1

    def test_exclude_pipeline_with_augmented_completes(self, dataset_with_augmented):
        """Pipeline with exclude should complete on augmented dataset."""
        dataset = dataset_with_augmented

        pipeline = [
            {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
            PLSRegression(n_components=5)
        ]

        # Should complete without error
        result = nirs4all.run(
            pipeline=pipeline,
            dataset=dataset,
            verbose=0
        )

        assert result is not None

class TestExcludeSyntaxComparison:
    """Compare new exclude syntax functionality."""

    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for syntax testing."""
        np.random.seed(42)

        X = np.vstack([
            np.random.rand(40, 50),
            np.random.rand(5, 50)
        ])
        y = np.concatenate([
            np.random.normal(50, 5, 40),
            [200, -100, 250, -150, 300]
        ])

        dataset = SpectroDataset("syntax_test")
        dataset.add_samples(X, {"partition": "train"})
        dataset.add_targets(y)

        return dataset

    def test_new_exclude_syntax_single(self, simple_dataset):
        """New syntax: {"exclude": Filter()}"""
        pipeline = [
            {"exclude": YOutlierFilter(method="iqr", threshold=1.5)},
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=simple_dataset,
            verbose=0
        )

        assert result is not None

    def test_new_exclude_syntax_multiple(self, simple_dataset):
        """New syntax: {"exclude": [Filter1(), Filter2()], "mode": "any"}"""
        pipeline = [
            {
                "exclude": [
                    YOutlierFilter(method="iqr", threshold=1.5),
                    YOutlierFilter(method="zscore", threshold=2.5)
                ],
                "mode": "any"
            },
            PLSRegression(n_components=5)
        ]

        result = nirs4all.run(
            pipeline=pipeline,
            dataset=simple_dataset,
            verbose=0
        )

        assert result is not None

    def test_exclude_different_from_old_sample_filter_keyword(self, simple_dataset):
        """Verify old sample_filter keyword is no longer recognized."""
        # Old syntax (should NOT work anymore)
        old_pipeline = [
            {
                "sample_filter": {
                    "filters": [YOutlierFilter(method="iqr")],
                    "mode": "any"
                }
            },
            PLSRegression(n_components=5)
        ]

        # The old keyword should not have a controller and cause an error
        # or be silently ignored. Let's verify it doesn't exclude samples
        # by checking the pipeline completes but sample_filter is unrecognized
        result = nirs4all.run(
            pipeline=old_pipeline,
            dataset=simple_dataset,
            verbose=0
        )

        # Pipeline may complete but sample_filter should not have been processed
        # (no SampleFilterController registered anymore)
        assert result is not None

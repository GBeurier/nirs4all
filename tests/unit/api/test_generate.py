"""
Unit tests for nirs4all.generate API.
"""

import pytest
import numpy as np


class TestGenerateFunction:
    """Tests for the main generate() function."""

    def test_basic_generation(self):
        """Test basic dataset generation."""
        import nirs4all

        dataset = nirs4all.generate(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.num_samples == 100

    def test_generate_as_arrays(self):
        """Test generation returning arrays."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=100, as_dataset=False, random_state=42)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 100

    def test_generate_reproducibility(self):
        """Test reproducibility with random_state."""
        import nirs4all

        X1, y1 = nirs4all.generate(n_samples=50, as_dataset=False, random_state=42)
        X2, y2 = nirs4all.generate(n_samples=50, as_dataset=False, random_state=42)

        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(y1, y2)

    def test_generate_complexity_simple(self):
        """Test simple complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="simple",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))

    def test_generate_complexity_realistic(self):
        """Test realistic complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="realistic",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))

    def test_generate_complexity_complex(self):
        """Test complex complexity generation."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            complexity="complex",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(X))

    def test_generate_wavelength_range(self):
        """Test custom wavelength range."""
        import nirs4all

        X, _ = nirs4all.generate(
            n_samples=50,
            wavelength_range=(1200, 2000),
            as_dataset=False,
            random_state=42,
        )

        # Fewer wavelengths in narrower range
        assert X.shape[1] < 751  # Default is 1000-2500 with step 2

    def test_generate_components(self):
        """Test specifying predefined components."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=50,
            components=["water", "protein"],
            as_dataset=False,
            random_state=42,
        )

        # Should have 2 targets (one per component)
        assert y.shape == (50, 2) or y.ndim == 1

    def test_generate_target_range(self):
        """Test target range scaling."""
        import nirs4all

        X, y = nirs4all.generate(
            n_samples=100,
            target_range=(0, 100),
            as_dataset=False,
            random_state=42,
        )

        assert y.min() >= 0
        assert y.max() <= 100

    def test_generate_train_ratio(self):
        """Test train ratio partitioning."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=100,
            train_ratio=0.7,
            random_state=42,
        )

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 70
        assert test_count == 30

    def test_generate_custom_name(self):
        """Test custom dataset name."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=50,
            name="my_synthetic_data",
            random_state=42,
        )

        assert dataset.name == "my_synthetic_data"


class TestGenerateRegression:
    """Tests for generate.regression() convenience function."""

    def test_regression_basic(self):
        """Test basic regression dataset generation."""
        import nirs4all

        dataset = nirs4all.generate.regression(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.name == "synthetic_regression"

    def test_regression_as_arrays(self):
        """Test regression returning arrays."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            as_dataset=False,
            random_state=42,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_regression_target_range(self):
        """Test regression with target range."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            target_range=(0, 100),
            as_dataset=False,
            random_state=42,
        )

        assert y.min() >= 0
        assert y.max() <= 100

    def test_regression_single_target(self):
        """Test regression with single target component."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            target_component=0,
            as_dataset=False,
            random_state=42,
        )

        # Single target should be 1D
        assert y.ndim == 1

    def test_regression_lognormal_distribution(self):
        """Test regression with lognormal distribution."""
        import nirs4all

        X, y = nirs4all.generate.regression(
            n_samples=100,
            distribution="lognormal",
            as_dataset=False,
            random_state=42,
        )

        assert np.all(np.isfinite(y))


class TestGenerateClassification:
    """Tests for generate.classification() convenience function."""

    def test_classification_basic(self):
        """Test basic classification dataset generation."""
        import nirs4all

        dataset = nirs4all.generate.classification(n_samples=100, random_state=42)

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.name == "synthetic_classification"

    def test_classification_binary(self):
        """Test binary classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=2,
            as_dataset=False,
            random_state=42,
        )

        assert set(np.unique(y)) == {0, 1}

    def test_classification_multiclass(self):
        """Test multiclass classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=4,
            as_dataset=False,
            random_state=42,
        )

        assert len(np.unique(y)) == 4

    def test_classification_imbalanced(self):
        """Test imbalanced classification."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=1000,
            n_classes=3,
            class_weights=[0.6, 0.3, 0.1],
            as_dataset=False,
            random_state=42,
        )

        counts = np.bincount(y.astype(int))
        # Class 0 should have most samples
        assert counts[0] > counts[1] > counts[2]

    def test_classification_separation(self):
        """Test class separation parameter."""
        import nirs4all

        X, y = nirs4all.generate.classification(
            n_samples=100,
            n_classes=2,
            class_separation=2.0,
            as_dataset=False,
            random_state=42,
        )

        assert X.shape[0] == 100
        assert set(np.unique(y)) == {0, 1}


class TestGenerateBuilder:
    """Tests for generate.builder() convenience function."""

    def test_builder_returns_builder(self):
        """Test that builder() returns a SyntheticDatasetBuilder."""
        import nirs4all
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        builder = nirs4all.generate.builder(n_samples=100, random_state=42)

        assert isinstance(builder, SyntheticDatasetBuilder)

    def test_builder_configuration(self):
        """Test builder configuration and building."""
        import nirs4all

        dataset = (
            nirs4all.generate.builder(n_samples=100, random_state=42)
            .with_features(complexity="realistic")
            .with_targets(distribution="lognormal")
            .with_partitions(train_ratio=0.8)
            .build()
        )

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)
        assert dataset.num_samples == 100

    def test_builder_full_chain(self):
        """Test full builder method chain."""
        import nirs4all

        dataset = (
            nirs4all.generate.builder(n_samples=200, random_state=42)
            .with_features(
                wavelength_range=(1100, 2400),
                complexity="realistic",
                components=["water", "protein"],
            )
            .with_targets(
                distribution="lognormal",
                range=(0, 100),
            )
            .with_partitions(train_ratio=0.75)
            .with_batch_effects(n_batches=2)
            .build()
        )

        assert dataset.num_samples == 200


class TestGenerateNamespace:
    """Tests for the generate namespace functionality."""

    def test_generate_is_callable(self):
        """Test that generate is directly callable."""
        import nirs4all

        # Should not raise - generate should be callable
        dataset = nirs4all.generate(n_samples=50, random_state=42)
        assert dataset is not None

    def test_generate_has_methods(self):
        """Test that generate has method attributes."""
        import nirs4all

        assert hasattr(nirs4all.generate, 'regression')
        assert hasattr(nirs4all.generate, 'classification')
        assert hasattr(nirs4all.generate, 'builder')

    def test_generate_methods_callable(self):
        """Test that generate methods are callable."""
        import nirs4all

        assert callable(nirs4all.generate.regression)
        assert callable(nirs4all.generate.classification)
        assert callable(nirs4all.generate.builder)

    def test_generate_repr(self):
        """Test generate namespace string representation."""
        import nirs4all

        repr_str = repr(nirs4all.generate)
        assert "generate" in repr_str
        assert "regression" in repr_str
        assert "classification" in repr_str
        assert "builder" in repr_str


class TestIntegrationWithPipeline:
    """Tests for integration with nirs4all pipeline."""

    def test_generate_with_run(self):
        """Test using generated data with nirs4all.run()."""
        import nirs4all
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import ShuffleSplit

        # Generate synthetic data with single target
        dataset = nirs4all.generate.regression(
            n_samples=200,
            target_component=0,  # Single target for regression
            complexity="simple",
            random_state=42,
        )

        # Run a simple pipeline with cross-validation
        result = nirs4all.run(
            pipeline=[MinMaxScaler(), ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), PLSRegression(n_components=3)],
            dataset=dataset,
            verbose=0,
        )

        assert result is not None

    def test_classification_with_run(self):
        """Test classification data with pipeline."""
        import nirs4all
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import ShuffleSplit

        from sklearn.model_selection import ShuffleSplit

        # Generate classification data
        dataset = nirs4all.generate.classification(
            n_samples=200,
            n_classes=2,
            complexity="simple",
            random_state=42,
        )

        # Run classification pipeline with cross-validation
        result = nirs4all.run(
            pipeline=[StandardScaler(), ShuffleSplit(n_splits=2, test_size=0.2, random_state=42), KNeighborsClassifier(n_neighbors=3)],
            dataset=dataset,
            verbose=0,
        )

        assert result is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_sample_count(self):
        """Test with very small sample count."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=10, as_dataset=False, random_state=42)

        assert X.shape[0] == 10

    def test_large_sample_count(self):
        """Test with larger sample count."""
        import nirs4all

        X, y = nirs4all.generate(n_samples=1000, as_dataset=False, random_state=42)

        assert X.shape[0] == 1000
        assert np.all(np.isfinite(X))

    def test_all_train_no_test(self):
        """Test with train_ratio=1.0 (no test set)."""
        import nirs4all

        dataset = nirs4all.generate(
            n_samples=100,
            train_ratio=1.0,
            random_state=42,
        )

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 100
        assert test_count == 0

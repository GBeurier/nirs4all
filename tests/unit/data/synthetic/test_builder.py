"""
Unit tests for SyntheticDatasetBuilder class.
"""

import numpy as np
import pytest

from nirs4all.synthesis import (
    ComponentLibrary,
    FeatureConfig,
    SyntheticDatasetBuilder,
    SyntheticDatasetConfig,
)


class TestBuilderInit:
    """Tests for SyntheticDatasetBuilder initialization."""

    def test_default_init(self):
        """Test default initialization."""
        builder = SyntheticDatasetBuilder()
        # Default n_samples is 1000 but we test with smaller values
        assert builder.state.n_samples == 1000
        assert builder.state.random_state is None
        assert builder.state.name == "synthetic_nirs"

    def test_custom_init(self):
        """Test custom initialization parameters."""
        builder = SyntheticDatasetBuilder(
            n_samples=500,
            random_state=42,
            name="test_dataset",
        )
        assert builder.state.n_samples == 500
        assert builder.state.random_state == 42
        assert builder.state.name == "test_dataset"

    def test_invalid_n_samples(self):
        """Test error on invalid n_samples."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            SyntheticDatasetBuilder(n_samples=0)

        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            SyntheticDatasetBuilder(n_samples=-10)

    def test_repr(self):
        """Test string representation."""
        builder = SyntheticDatasetBuilder(n_samples=500, random_state=42)
        repr_str = repr(builder)
        assert "SyntheticDatasetBuilder" in repr_str
        assert "500" in repr_str
        assert "42" in repr_str

class TestWithFeatures:
    """Tests for with_features() method."""

    def test_wavelength_range(self):
        """Test setting wavelength range."""
        builder = SyntheticDatasetBuilder()
        builder.with_features(wavelength_range=(1200, 2000))

        assert builder.state.wavelength_start == 1200
        assert builder.state.wavelength_end == 2000

    def test_wavelength_step(self):
        """Test setting wavelength step."""
        builder = SyntheticDatasetBuilder()
        builder.with_features(wavelength_step=5)

        assert builder.state.wavelength_step == 5

    def test_complexity(self):
        """Test setting complexity level."""
        builder = SyntheticDatasetBuilder()
        builder.with_features(complexity="realistic")

        assert builder.state.complexity == "realistic"

    def test_components(self):
        """Test setting predefined components."""
        builder = SyntheticDatasetBuilder()
        builder.with_features(components=["water", "protein"])

        assert builder.state.component_names == ["water", "protein"]

    def test_component_library(self):
        """Test setting custom component library."""
        library = ComponentLibrary.from_predefined(["water", "lipid"])
        builder = SyntheticDatasetBuilder()
        builder.with_features(component_library=library)

        assert builder.state.component_library is library

    def test_cannot_set_both_components_and_library(self):
        """Test error when both components and library are set."""
        library = ComponentLibrary.from_predefined(["water"])
        builder = SyntheticDatasetBuilder()

        with pytest.raises(ValueError, match="Cannot specify both"):
            builder.with_features(
                components=["protein"],
                component_library=library,
            )

    def test_method_chaining(self):
        """Test method chaining returns self."""
        builder = SyntheticDatasetBuilder()
        result = builder.with_features(complexity="simple")

        assert result is builder

class TestWithTargets:
    """Tests for with_targets() method."""

    def test_distribution(self):
        """Test setting distribution method."""
        builder = SyntheticDatasetBuilder()
        builder.with_targets(distribution="lognormal")

        assert builder.state.concentration_method == "lognormal"

    def test_range(self):
        """Test setting target range."""
        builder = SyntheticDatasetBuilder()
        builder.with_targets(range=(0, 100))

        assert builder.state.target_range == (0, 100)

    def test_component(self):
        """Test setting target component."""
        builder = SyntheticDatasetBuilder()
        builder.with_targets(component="protein")

        assert builder.state.target_component == "protein"

    def test_component_by_index(self):
        """Test setting target component by index."""
        builder = SyntheticDatasetBuilder()
        builder.with_targets(component=0)

        assert builder.state.target_component == 0

    def test_transform(self):
        """Test setting target transformation."""
        builder = SyntheticDatasetBuilder()
        builder.with_targets(transform="log")

        assert builder.state.target_transform == "log"

    def test_clears_classification(self):
        """Test that with_targets clears classification settings."""
        builder = SyntheticDatasetBuilder()
        builder.with_classification(n_classes=3)
        builder.with_targets(distribution="uniform")

        assert builder.state.n_classes is None

class TestWithClassification:
    """Tests for with_classification() method."""

    def test_n_classes(self):
        """Test setting number of classes."""
        builder = SyntheticDatasetBuilder()
        builder.with_classification(n_classes=3)

        assert builder.state.n_classes == 3

    def test_separation(self):
        """Test setting class separation."""
        builder = SyntheticDatasetBuilder()
        builder.with_classification(n_classes=2, separation=2.0)

        assert builder.state.class_separation == 2.0

    def test_class_weights(self):
        """Test setting class weights."""
        builder = SyntheticDatasetBuilder()
        builder.with_classification(
            n_classes=3,
            class_weights=[0.5, 0.3, 0.2],
        )

        assert builder.state.class_weights == [0.5, 0.3, 0.2]

    def test_invalid_n_classes(self):
        """Test error on invalid n_classes."""
        builder = SyntheticDatasetBuilder()
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            builder.with_classification(n_classes=1)

    def test_invalid_class_weights_length(self):
        """Test error on mismatched class weights length."""
        builder = SyntheticDatasetBuilder()
        with pytest.raises(ValueError, match="class_weights length"):
            builder.with_classification(n_classes=3, class_weights=[0.5, 0.5])

    def test_invalid_class_weights_sum(self):
        """Test error when class weights don't sum to 1."""
        builder = SyntheticDatasetBuilder()
        with pytest.raises(ValueError, match="must sum to 1.0"):
            builder.with_classification(n_classes=2, class_weights=[0.3, 0.3])

class TestWithMetadata:
    """Tests for with_metadata() method."""

    def test_sample_ids(self):
        """Test enabling sample ID generation."""
        builder = SyntheticDatasetBuilder()
        builder.with_metadata(sample_ids=True)

        assert builder.state.generate_sample_ids is True

    def test_sample_id_prefix(self):
        """Test setting sample ID prefix."""
        builder = SyntheticDatasetBuilder()
        builder.with_metadata(sample_id_prefix="test_")

        assert builder.state.sample_id_prefix == "test_"

    def test_n_groups(self):
        """Test setting number of groups."""
        builder = SyntheticDatasetBuilder()
        builder.with_metadata(n_groups=5)

        assert builder.state.n_groups == 5

    def test_n_repetitions_int(self):
        """Test setting fixed repetitions."""
        builder = SyntheticDatasetBuilder()
        builder.with_metadata(n_repetitions=3)

        assert builder.state.n_repetitions == 3

    def test_n_repetitions_range(self):
        """Test setting repetition range."""
        builder = SyntheticDatasetBuilder()
        builder.with_metadata(n_repetitions=(2, 5))

        assert builder.state.n_repetitions == (2, 5)

class TestWithPartitions:
    """Tests for with_partitions() method."""

    def test_train_ratio(self):
        """Test setting train ratio."""
        builder = SyntheticDatasetBuilder()
        builder.with_partitions(train_ratio=0.75)

        assert builder.state.train_ratio == 0.75

    def test_invalid_train_ratio_low(self):
        """Test error on train_ratio <= 0."""
        builder = SyntheticDatasetBuilder()
        with pytest.raises(ValueError, match="train_ratio must be in"):
            builder.with_partitions(train_ratio=0.0)

    def test_invalid_train_ratio_high(self):
        """Test error on train_ratio > 1."""
        builder = SyntheticDatasetBuilder()
        with pytest.raises(ValueError, match="train_ratio must be in"):
            builder.with_partitions(train_ratio=1.5)

    def test_stratify(self):
        """Test setting stratify option."""
        builder = SyntheticDatasetBuilder()
        builder.with_partitions(stratify=True)

        assert builder.state.stratify is True

    def test_shuffle(self):
        """Test setting shuffle option."""
        builder = SyntheticDatasetBuilder()
        builder.with_partitions(shuffle=False)

        assert builder.state.shuffle is False

class TestWithBatchEffects:
    """Tests for with_batch_effects() method."""

    def test_enable(self):
        """Test enabling batch effects."""
        builder = SyntheticDatasetBuilder()
        builder.with_batch_effects(enabled=True)

        assert builder.state.batch_effects_enabled is True

    def test_n_batches(self):
        """Test setting number of batches."""
        builder = SyntheticDatasetBuilder()
        builder.with_batch_effects(n_batches=5)

        assert builder.state.n_batches == 5

class TestWithOutput:
    """Tests for with_output() method."""

    def test_as_dataset(self):
        """Test setting output format."""
        builder = SyntheticDatasetBuilder()
        builder.with_output(as_dataset=False)

        assert builder.state.as_dataset is False

    def test_include_metadata(self):
        """Test setting metadata inclusion."""
        builder = SyntheticDatasetBuilder()
        builder.with_output(include_metadata=True)

        assert builder.state.include_metadata is True

class TestBuild:
    """Tests for build() method."""

    def test_build_returns_dataset(self):
        """Test that build returns a SpectroDataset by default."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        dataset = builder.build()

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)

    def test_build_returns_arrays(self):
        """Test that build returns arrays when configured."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_output(as_dataset=False)
        X, y = builder.build()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 50

    def test_build_once_only(self):
        """Test that build can only be called once."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.build()

        with pytest.raises(RuntimeError, match="build\\(\\) can only be called once"):
            builder.build()

    def test_build_reproducibility(self):
        """Test that same random_state produces same results."""
        builder1 = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder1.with_output(as_dataset=False)
        X1, y1 = builder1.build()

        builder2 = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder2.with_output(as_dataset=False)
        X2, y2 = builder2.build()

        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(y1, y2)

    def test_build_different_seeds(self):
        """Test that different random_states produce different results."""
        builder1 = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder1.with_output(as_dataset=False)
        X1, _ = builder1.build()

        builder2 = SyntheticDatasetBuilder(n_samples=50, random_state=123)
        builder2.with_output(as_dataset=False)
        X2, _ = builder2.build()

        assert not np.allclose(X1, X2)

class TestBuildArrays:
    """Tests for build_arrays() convenience method."""

    def test_build_arrays(self):
        """Test build_arrays returns tuple."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        X, y = builder.build_arrays()

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 50

class TestBuildDataset:
    """Tests for build_dataset() convenience method."""

    def test_build_dataset(self):
        """Test build_dataset returns SpectroDataset."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        dataset = builder.build_dataset()

        from nirs4all.data import SpectroDataset
        assert isinstance(dataset, SpectroDataset)

class TestTargetProcessing:
    """Tests for target value processing."""

    def test_target_range_scaling(self):
        """Test target range scaling."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_targets(range=(0, 100))
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        assert y.min() >= 0
        assert y.max() <= 100

    def test_single_target_component(self):
        """Test selecting single target component."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_targets(component=0)
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        # Should be 1D for single target
        assert y.ndim == 1

    def test_log_transform(self):
        """Test log transformation."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_targets(transform="log", component=0)
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        # Values should be log-scaled (typically smaller)
        assert np.all(np.isfinite(y))

    def test_sqrt_transform(self):
        """Test sqrt transformation."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_targets(transform="sqrt", component=0)
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        assert np.all(np.isfinite(y))
        assert np.all(y >= 0)

class TestClassificationOutput:
    """Tests for classification target output."""

    def test_binary_classification(self):
        """Test binary classification labels."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_classification(n_classes=2)
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        assert set(np.unique(y)) == {0, 1}

    def test_multiclass_classification(self):
        """Test multiclass classification labels."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_classification(n_classes=3)
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        assert len(np.unique(y)) == 3
        assert set(np.unique(y)) == {0, 1, 2}

    def test_imbalanced_classes(self):
        """Test imbalanced class generation."""
        builder = SyntheticDatasetBuilder(n_samples=300, random_state=42)
        builder.with_classification(
            n_classes=3,
            class_weights=[0.6, 0.3, 0.1],
        )
        builder.with_output(as_dataset=False)
        _, y = builder.build()

        # Check approximate class distribution
        counts = np.bincount(y.astype(int))
        # Class 0 should have most samples
        assert counts[0] > counts[1] > counts[2]

class TestDatasetPartitioning:
    """Tests for dataset partitioning."""

    def test_default_train_ratio(self):
        """Test default 80/20 split."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        dataset = builder.build()

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 40
        assert test_count == 10

    def test_custom_train_ratio(self):
        """Test custom train ratio."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_partitions(train_ratio=0.7)
        dataset = builder.build()

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 35
        assert test_count == 15

    def test_full_train(self):
        """Test train_ratio=1.0 (all training)."""
        builder = SyntheticDatasetBuilder(n_samples=50, random_state=42)
        builder.with_partitions(train_ratio=1.0)
        dataset = builder.build()

        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        assert train_count == 50
        assert test_count == 0

class TestGetConfig:
    """Tests for get_config() method."""

    def test_get_config(self):
        """Test getting configuration object."""
        builder = SyntheticDatasetBuilder(n_samples=500, random_state=42)
        builder.with_features(complexity="realistic")
        builder.with_targets(distribution="lognormal")

        config = builder.get_config()

        assert isinstance(config, SyntheticDatasetConfig)
        assert config.n_samples == 500
        assert config.random_state == 42
        assert config.features.complexity == "realistic"
        assert config.targets.distribution == "lognormal"

class TestFromConfig:
    """Tests for from_config() class method."""

    def test_from_config(self):
        """Test creating builder from config."""
        config = SyntheticDatasetConfig(
            n_samples=500,
            random_state=42,
            features=FeatureConfig(complexity="realistic"),
        )

        builder = SyntheticDatasetBuilder.from_config(config)

        assert builder.state.n_samples == 500
        assert builder.state.random_state == 42
        assert builder.state.complexity == "realistic"

    def test_roundtrip_config(self):
        """Test config roundtrip through get_config/from_config."""
        builder1 = SyntheticDatasetBuilder(n_samples=500, random_state=42)
        builder1.with_features(complexity="realistic")
        builder1.with_targets(distribution="lognormal", range=(0, 50))
        builder1.with_partitions(train_ratio=0.75)

        config = builder1.get_config()
        builder2 = SyntheticDatasetBuilder.from_config(config)

        assert builder2.state.n_samples == 500
        assert builder2.state.complexity == "realistic"
        assert builder2.state.concentration_method == "lognormal"
        assert builder2.state.target_range == (0, 50)
        assert builder2.state.train_ratio == 0.75

class TestComplexScenarios:
    """Tests for complex builder configurations."""

    def test_full_configuration(self):
        """Test full builder configuration."""
        dataset = (
            SyntheticDatasetBuilder(n_samples=60, random_state=42)
            .with_features(
                wavelength_range=(1100, 2400),
                wavelength_step=4,
                complexity="realistic",
                components=["water", "protein", "lipid"],
            )
            .with_targets(
                distribution="lognormal",
                range=(5, 50),
                component="protein",
            )
            .with_metadata(
                sample_ids=True,
                n_groups=3,
            )
            .with_partitions(train_ratio=0.8)
            .with_batch_effects(n_batches=2)
            .build()
        )

        assert dataset.num_samples == 60
        assert dataset.name == "synthetic_nirs"

    def test_classification_with_batch_effects(self):
        """Test classification with batch effects."""
        builder = SyntheticDatasetBuilder(n_samples=60, random_state=42)
        builder.with_features(complexity="realistic")
        builder.with_classification(n_classes=3)
        builder.with_batch_effects(n_batches=3)
        builder.with_output(as_dataset=False)

        X, y = builder.build()

        assert X.shape[0] == 60
        assert len(np.unique(y)) == 3

    def test_realistic_regression_scenario(self):
        """Test realistic regression scenario."""
        dataset = (
            SyntheticDatasetBuilder(n_samples=100, random_state=42)
            .with_features(complexity="realistic")
            .with_targets(
                distribution="correlated",
                range=(0, 100),
            )
            .with_partitions(train_ratio=0.8, shuffle=True)
            .build()
        )

        assert dataset.num_samples == 100

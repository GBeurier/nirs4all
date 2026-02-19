"""
Unit tests for configuration dataclasses.
"""

import pytest

from nirs4all.synthesis.config import (
    BatchEffectConfig,
    FeatureConfig,
    MetadataConfig,
    OutputConfig,
    PartitionConfig,
    SyntheticDatasetConfig,
    TargetConfig,
)


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.wavelength_start == 1000.0
        assert config.wavelength_end == 2500.0
        assert config.wavelength_step == 2.0
        assert config.complexity == "simple"
        assert config.n_components is None
        assert config.component_names is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            wavelength_start=1200,
            wavelength_end=2000,
            complexity="realistic",
            component_names=["water", "protein"],
        )
        assert config.wavelength_start == 1200
        assert config.wavelength_end == 2000
        assert config.complexity == "realistic"
        assert config.component_names == ["water", "protein"]

class TestTargetConfig:
    """Tests for TargetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TargetConfig()
        assert config.distribution == "dirichlet"
        assert config.range is None
        assert config.n_targets is None
        assert config.component_indices is None
        assert config.transform is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TargetConfig(
            distribution="lognormal",
            range=(0, 100),
            transform="log",
        )
        assert config.distribution == "lognormal"
        assert config.range == (0, 100)
        assert config.transform == "log"

class TestMetadataConfig:
    """Tests for MetadataConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MetadataConfig()
        assert config.generate_sample_ids is True
        assert config.sample_id_prefix == "sample"
        assert config.n_groups is None
        assert config.n_repetitions == 1
        assert config.group_names is None
        assert config.additional_columns is None

    def test_repetition_range(self):
        """Test repetition as range tuple."""
        config = MetadataConfig(n_repetitions=(2, 5))
        assert config.n_repetitions == (2, 5)

class TestPartitionConfig:
    """Tests for PartitionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PartitionConfig()
        assert config.train_ratio == 0.8
        assert config.stratify is False
        assert config.shuffle is True
        assert config.group_aware is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PartitionConfig(
            train_ratio=0.7,
            stratify=True,
            shuffle=False,
        )
        assert config.train_ratio == 0.7
        assert config.stratify is True
        assert config.shuffle is False

class TestBatchEffectConfig:
    """Tests for BatchEffectConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchEffectConfig()
        assert config.enabled is False
        assert config.n_batches == 3
        assert config.offset_std == 0.02
        assert config.gain_std == 0.03

    def test_enabled(self):
        """Test enabling batch effects."""
        config = BatchEffectConfig(enabled=True, n_batches=5)
        assert config.enabled is True
        assert config.n_batches == 5

class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OutputConfig()
        assert config.as_dataset is True
        assert config.include_metadata is False
        assert config.include_wavelengths is True

class TestSyntheticDatasetConfig:
    """Tests for SyntheticDatasetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SyntheticDatasetConfig()
        assert config.n_samples == 1000
        assert config.random_state is None
        assert config.name == "synthetic_nirs"
        assert isinstance(config.features, FeatureConfig)
        assert isinstance(config.targets, TargetConfig)
        assert isinstance(config.metadata, MetadataConfig)
        assert isinstance(config.partitions, PartitionConfig)
        assert isinstance(config.batch_effects, BatchEffectConfig)
        assert isinstance(config.output, OutputConfig)

    def test_custom_n_samples(self):
        """Test custom sample count."""
        config = SyntheticDatasetConfig(n_samples=500, random_state=42)
        assert config.n_samples == 500
        assert config.random_state == 42

    def test_invalid_n_samples(self):
        """Test validation of n_samples."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            SyntheticDatasetConfig(n_samples=0)

    def test_invalid_train_ratio(self):
        """Test validation of train_ratio."""
        with pytest.raises(ValueError, match="train_ratio must be in"):
            SyntheticDatasetConfig(
                partitions=PartitionConfig(train_ratio=1.5)
            )

    def test_invalid_complexity(self):
        """Test validation of complexity."""
        with pytest.raises(ValueError, match="complexity must be one of"):
            SyntheticDatasetConfig(
                features=FeatureConfig(complexity="invalid")
            )

    def test_nested_config_modification(self):
        """Test modifying nested configurations."""
        config = SyntheticDatasetConfig(
            n_samples=500,
            features=FeatureConfig(complexity="realistic"),
            targets=TargetConfig(distribution="uniform"),
        )
        assert config.features.complexity == "realistic"
        assert config.targets.distribution == "uniform"

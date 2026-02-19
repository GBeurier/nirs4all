"""
Unit tests for MultiSourceGenerator class.
"""

import numpy as np
import pytest

from nirs4all.synthesis.sources import (
    MultiSourceGenerator,
    MultiSourceResult,
    SourceConfig,
    generate_multi_source,
)


class TestSourceConfig:
    """Tests for SourceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SourceConfig(name="test")
        assert config.name == "test"
        assert config.source_type == "nir"
        assert config.complexity == "simple"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "name": "NIR",
            "type": "nir",
            "wavelength_range": (1000, 2000),
            "complexity": "realistic"
        }
        config = SourceConfig.from_dict(d)

        assert config.name == "NIR"
        assert config.source_type == "nir"
        assert config.wavelength_start == 1000
        assert config.wavelength_end == 2000
        assert config.complexity == "realistic"

    def test_aux_config(self):
        """Test auxiliary source configuration."""
        config = SourceConfig(
            name="markers",
            source_type="aux",
            n_features=20
        )
        assert config.n_features == 20

class TestMultiSourceResult:
    """Tests for MultiSourceResult container."""

    def test_source_names(self):
        """Test getting source names."""
        result = MultiSourceResult(
            sources={"NIR": np.zeros((10, 100)), "markers": np.zeros((10, 20))},
            targets=np.zeros(10),
            source_configs=[],
        )
        assert result.source_names == ["NIR", "markers"]

    def test_combined_features(self):
        """Test combining features."""
        result = MultiSourceResult(
            sources={"A": np.ones((10, 50)), "B": np.ones((10, 30))},
            targets=np.zeros(10),
            source_configs=[],
        )
        combined = result.get_combined_features()
        assert combined.shape == (10, 80)

    def test_n_samples(self):
        """Test n_samples property."""
        result = MultiSourceResult(
            sources={"NIR": np.zeros((25, 100))},
            targets=np.zeros(25),
            source_configs=[],
        )
        assert result.n_samples == 25

    def test_n_features_total(self):
        """Test total features count."""
        result = MultiSourceResult(
            sources={"A": np.zeros((10, 100)), "B": np.zeros((10, 50))},
            targets=np.zeros(10),
            source_configs=[],
        )
        assert result.n_features_total == 150

class TestMultiSourceGeneratorInit:
    """Tests for MultiSourceGenerator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        gen = MultiSourceGenerator()
        assert gen.rng is not None

    def test_with_random_state(self):
        """Test initialization with random state."""
        gen = MultiSourceGenerator(random_state=42)
        assert gen._random_state == 42

class TestMultiSourceGeneration:
    """Tests for multi-source dataset generation."""

    def test_single_nir_source(self):
        """Test generating single NIR source."""
        gen = MultiSourceGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            sources=[
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)}
            ]
        )

        assert "NIR" in result.sources
        assert result.sources["NIR"].shape[0] == 30
        assert "NIR" in result.wavelengths

    def test_multiple_nir_sources(self):
        """Test generating multiple NIR ranges."""
        gen = MultiSourceGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            sources=[
                {"name": "VIS-NIR", "type": "nir", "wavelength_range": (400, 1100)},
                {"name": "SWIR", "type": "nir", "wavelength_range": (1100, 2500)},
            ]
        )

        assert len(result.sources) == 2
        assert "VIS-NIR" in result.sources
        assert "SWIR" in result.sources
        assert len(result.wavelengths) == 2

    def test_nir_plus_aux(self):
        """Test NIR combined with auxiliary source."""
        gen = MultiSourceGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            sources=[
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
                {"name": "markers", "type": "aux", "n_features": 15},
            ]
        )

        assert result.sources["NIR"].shape[0] == 30
        assert result.sources["markers"].shape == (30, 15)
        assert "NIR" in result.wavelengths
        assert "markers" not in result.wavelengths

    def test_target_range(self):
        """Test target range scaling."""
        gen = MultiSourceGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            sources=[{"name": "NIR", "type": "nir"}],
            target_range=(0, 100)
        )

        assert result.targets.min() >= 0
        assert result.targets.max() <= 100

    def test_source_config_objects(self):
        """Test using SourceConfig objects."""
        gen = MultiSourceGenerator(random_state=42)
        result = gen.generate(
            n_samples=30,
            sources=[
                SourceConfig(name="NIR", source_type="nir"),
                SourceConfig(name="aux", source_type="aux", n_features=10),
            ]
        )

        assert len(result.sources) == 2

    def test_duplicate_names_raises(self):
        """Test error on duplicate source names."""
        gen = MultiSourceGenerator(random_state=42)
        with pytest.raises(ValueError, match="names must be unique"):
            gen.generate(
                n_samples=20,
                sources=[
                    {"name": "NIR", "type": "nir"},
                    {"name": "NIR", "type": "aux"},
                ]
            )

    def test_unknown_source_type_raises(self):
        """Test error on unknown source type."""
        gen = MultiSourceGenerator(random_state=42)
        with pytest.raises(ValueError, match="Unknown source type"):
            gen.generate(
                n_samples=20,
                sources=[{"name": "bad", "type": "invalid"}]
            )

class TestMultiSourceCreateDataset:
    """Tests for creating SpectroDataset from multi-source."""

    def test_create_dataset_basic(self):
        """Test basic dataset creation."""
        gen = MultiSourceGenerator(random_state=42)
        dataset = gen.create_dataset(
            n_samples=40,
            sources=[
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
                {"name": "markers", "type": "aux", "n_features": 10},
            ],
            train_ratio=0.8,
        )

        # Check dataset was created
        assert dataset is not None
        assert dataset.name == "multi_source_synthetic"

    def test_create_dataset_partitions(self):
        """Test dataset partition sizes."""
        gen = MultiSourceGenerator(random_state=42)
        dataset = gen.create_dataset(
            n_samples=40,
            sources=[{"name": "NIR", "type": "nir"}],
            train_ratio=0.7,
        )

        # Check partition sizes
        n_train = len(dataset.x({"partition": "train"}))
        n_test = len(dataset.x({"partition": "test"}))
        assert n_train == 28
        assert n_test == 12

class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_results(self):
        """Test same seed produces same results."""
        gen1 = MultiSourceGenerator(random_state=42)
        gen2 = MultiSourceGenerator(random_state=42)

        sources = [{"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)}]

        result1 = gen1.generate(n_samples=50, sources=sources)
        result2 = gen2.generate(n_samples=50, sources=sources)

        np.testing.assert_array_almost_equal(
            result1.sources["NIR"],
            result2.sources["NIR"]
        )
        np.testing.assert_array_almost_equal(result1.targets, result2.targets)

class TestConvenienceFunction:
    """Tests for generate_multi_source convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        dataset = generate_multi_source(
            n_samples=30,
            sources=[
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
                {"name": "markers", "type": "aux", "n_features": 10},
            ],
            random_state=42,
            as_dataset=True,
        )

        assert dataset is not None

    def test_returns_arrays(self):
        """Test returning arrays instead of dataset."""
        result = generate_multi_source(
            n_samples=30,
            sources=[
                {"name": "NIR", "type": "nir", "wavelength_range": (1000, 2500)},
            ],
            random_state=42,
            as_dataset=False,
        )

        # Returns MultiSourceResult when as_dataset=False
        assert isinstance(result, MultiSourceResult)
        assert result.n_samples == 30

    def test_default_sources(self):
        """Test using default sources."""
        dataset = generate_multi_source(
            n_samples=30,
            sources=None,  # Should use defaults
            random_state=42,
        )

        # Default should have NIR + markers
        assert dataset is not None

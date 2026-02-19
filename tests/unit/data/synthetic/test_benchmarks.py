"""
Unit tests for Phase 4 benchmarks module.
"""

import numpy as np
import pytest

from nirs4all.synthesis.benchmarks import (
    BENCHMARK_DATASETS,
    BenchmarkDatasetInfo,
    BenchmarkDomain,
    create_synthetic_matching_benchmark,
    get_benchmark_info,
    get_benchmark_spectral_properties,
    get_datasets_by_domain,
    list_benchmark_datasets,
)


class TestBenchmarkDatasetInfo:
    """Tests for BenchmarkDatasetInfo dataclass."""

    def test_corn_dataset_info(self):
        """Test corn dataset information."""
        info = get_benchmark_info("corn")

        assert info.name == "corn"
        assert info.domain == BenchmarkDomain.AGRICULTURE
        assert info.n_samples == 80
        assert info.wavelength_range == (1100, 2498)
        assert "moisture" in info.targets
        assert "protein" in info.targets

    def test_tecator_dataset_info(self):
        """Test tecator dataset information."""
        info = get_benchmark_info("tecator")

        assert info.name == "tecator"
        assert info.domain == BenchmarkDomain.FOOD
        assert "fat" in info.targets
        assert info.measurement_mode == "transmittance"

    def test_shootout_dataset_info(self):
        """Test shootout2002 dataset information."""
        info = get_benchmark_info("shootout2002")

        assert info.name == "shootout2002"
        assert info.domain == BenchmarkDomain.PHARMACEUTICAL
        assert info.n_samples == 654

    def test_dataset_info_summary(self):
        """Test dataset info summary output."""
        info = get_benchmark_info("corn")
        summary = info.summary()

        assert "corn" in summary.lower()
        assert "80" in summary  # n_samples
        assert "1100" in summary  # wavelength start
        assert "agriculture" in summary.lower()

    def test_unknown_dataset_raises(self):
        """Test that unknown dataset raises KeyError."""
        with pytest.raises(KeyError):
            get_benchmark_info("nonexistent_dataset")

class TestBenchmarkRegistry:
    """Tests for benchmark dataset registry."""

    def test_list_benchmark_datasets(self):
        """Test listing available datasets."""
        datasets = list_benchmark_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) >= 4  # At least our core datasets
        assert "corn" in datasets
        assert "tecator" in datasets

    def test_get_datasets_by_domain(self):
        """Test filtering datasets by domain."""
        pharma = get_datasets_by_domain(BenchmarkDomain.PHARMACEUTICAL)
        assert len(pharma) > 0
        assert "shootout2002" in pharma

        food = get_datasets_by_domain("food")
        assert len(food) > 0
        assert "tecator" in food

        agri = get_datasets_by_domain(BenchmarkDomain.AGRICULTURE)
        assert "corn" in agri

class TestBenchmarkDatasetProperties:
    """Tests for all registered datasets."""

    @pytest.mark.parametrize("name", list_benchmark_datasets())
    def test_dataset_has_required_fields(self, name):
        """Test that all datasets have required fields."""
        info = get_benchmark_info(name)

        # Check required fields
        assert info.name == name
        assert info.full_name is not None
        assert info.domain is not None
        assert info.n_samples > 0
        assert info.n_wavelengths > 0
        assert len(info.wavelength_range) == 2
        assert info.wavelength_range[0] < info.wavelength_range[1]
        assert len(info.targets) > 0
        assert info.sample_type is not None
        assert info.measurement_mode is not None
        assert info.source_url is not None

    @pytest.mark.parametrize("name", list_benchmark_datasets())
    def test_dataset_has_valid_snr_range(self, name):
        """Test that SNR range is reasonable."""
        info = get_benchmark_info(name)

        low, high = info.typical_snr
        assert low > 0
        assert high > low
        assert high < 10000  # Reasonable upper bound

    @pytest.mark.parametrize("name", list_benchmark_datasets())
    def test_dataset_has_valid_peak_density(self, name):
        """Test that peak density range is reasonable."""
        info = get_benchmark_info(name)

        low, high = info.typical_peak_density
        assert low >= 0
        assert high > low
        assert high < 50  # Reasonable upper bound for peaks per 100 nm

class TestBenchmarkDomain:
    """Tests for BenchmarkDomain enum."""

    def test_domain_values(self):
        """Test domain enumeration values."""
        assert BenchmarkDomain.AGRICULTURE.value == "agriculture"
        assert BenchmarkDomain.FOOD.value == "food"
        assert BenchmarkDomain.PHARMACEUTICAL.value == "pharmaceutical"
        assert BenchmarkDomain.PETROCHEMICAL.value == "petrochemical"

    def test_domain_from_string(self):
        """Test creating domain from string."""
        domain = BenchmarkDomain("food")
        assert domain == BenchmarkDomain.FOOD

class TestBenchmarkSpectralProperties:
    """Tests for getting spectral properties from benchmarks."""

    def test_get_benchmark_spectral_properties(self):
        """Test getting properties for synthetic generation."""
        props = get_benchmark_spectral_properties("corn")

        assert "wavelength_start" in props
        assert "wavelength_end" in props
        assert "typical_components" in props
        assert "n_samples" in props

        assert props["wavelength_start"] == 1100
        assert props["wavelength_end"] == 2498
        assert props["n_samples"] == 80

    def test_spectral_properties_include_components(self):
        """Test that properties include relevant components."""
        props = get_benchmark_spectral_properties("tecator")

        assert len(props["typical_components"]) > 0

class TestCreateSyntheticMatchingBenchmark:
    """Tests for creating synthetic data matching benchmark."""

    def test_create_matching_corn(self):
        """Test creating synthetic data matching corn dataset."""
        X, C, E = create_synthetic_matching_benchmark("corn", random_state=42)

        # Should match corn dimensions (approximately)
        assert X.shape[0] == 80  # n_samples
        assert X.shape[1] > 100  # reasonable n_wavelengths
        assert C.shape[0] == 80
        assert E.ndim == 2

    def test_create_matching_custom_samples(self):
        """Test creating with custom sample count."""
        X, C, E = create_synthetic_matching_benchmark(
            "tecator",
            n_samples=50,
            random_state=42
        )

        assert X.shape[0] == 50

    def test_create_matching_reproducible(self):
        """Test that generation is reproducible."""
        X1, C1, E1 = create_synthetic_matching_benchmark("corn", random_state=42)
        X2, C2, E2 = create_synthetic_matching_benchmark("corn", random_state=42)

        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_almost_equal(C1, C2)

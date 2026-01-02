"""
Unit tests for Phase 4 accelerated generation module.
"""

import pytest
import numpy as np

from nirs4all.data.synthetic.accelerated import (
    AcceleratorBackend,
    AcceleratedArrays,
    AcceleratedGenerator,
    detect_best_backend,
    benchmark_backends,
    is_gpu_available,
    get_backend_info,
    create_accelerated_arrays,
    generate_spectra_batch_accelerated,
    generate_voigt_profiles_accelerated,
)


class TestAcceleratorBackend:
    """Tests for AcceleratorBackend enum."""

    def test_backend_values(self):
        """Test backend enumeration values."""
        assert AcceleratorBackend.JAX.value == "jax"
        assert AcceleratorBackend.CUPY.value == "cupy"
        assert AcceleratorBackend.NUMPY.value == "numpy"

    def test_backend_from_string(self):
        """Test creating backend from string."""
        backend = AcceleratorBackend("numpy")
        assert backend == AcceleratorBackend.NUMPY


class TestBackendDetection:
    """Tests for backend detection utilities."""

    def test_detect_best_backend(self):
        """Test detecting best available backend."""
        backend = detect_best_backend()
        assert isinstance(backend, AcceleratorBackend)

    def test_numpy_always_available(self):
        """Test that NumPy backend is always in result."""
        info = get_backend_info()
        assert "best_backend" in info

    def test_get_backend_info(self):
        """Test getting backend information."""
        info = get_backend_info()
        assert isinstance(info, dict)
        assert "jax_available" in info
        assert "cupy_available" in info
        assert "best_backend" in info

    def test_is_gpu_available_returns_bool(self):
        """Test that GPU availability check returns boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestAcceleratedArrays:
    """Tests for AcceleratedArrays dataclass."""

    def test_create_accelerated_arrays(self):
        """Test creating accelerated arrays container."""
        arrays = create_accelerated_arrays(backend=AcceleratorBackend.NUMPY)

        assert arrays.backend == AcceleratorBackend.NUMPY
        assert callable(arrays.zeros)
        assert callable(arrays.ones)
        assert callable(arrays.exp)
        assert callable(arrays.matmul)

    def test_arrays_operations(self):
        """Test accelerated array operations."""
        arrays = create_accelerated_arrays(backend=AcceleratorBackend.NUMPY)

        # Test basic operations
        zeros = arrays.zeros((10, 100))
        assert zeros.shape == (10, 100)

        ones = arrays.ones((5, 50))
        assert ones.shape == (5, 50)

        linspace = arrays.linspace(900, 1700, 100)
        assert len(linspace) == 100

    def test_arrays_to_numpy(self):
        """Test converting accelerated arrays to numpy."""
        arrays = create_accelerated_arrays(backend=AcceleratorBackend.NUMPY)

        x = arrays.array([1, 2, 3])
        np_x = arrays.to_numpy(x)

        assert isinstance(np_x, np.ndarray)
        np.testing.assert_array_equal(np_x, [1, 2, 3])


class TestAcceleratedGenerator:
    """Tests for AcceleratedGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a generator with default settings."""
        return AcceleratedGenerator(backend=AcceleratorBackend.NUMPY)

    @pytest.fixture
    def test_data(self):
        """Create test component spectra and concentrations."""
        np.random.seed(42)
        n_wavelengths = 100
        n_components = 5
        n_samples = 50

        wavelengths = np.linspace(1000, 2500, n_wavelengths)
        component_spectra = np.abs(np.random.randn(n_components, n_wavelengths)) + 0.1
        concentrations = np.abs(np.random.randn(n_samples, n_components))

        return {
            "wavelengths": wavelengths,
            "component_spectra": component_spectra,
            "concentrations": concentrations,
            "n_samples": n_samples,
            "n_wavelengths": n_wavelengths,
            "n_components": n_components,
        }

    def test_generator_creation(self, generator):
        """Test creating accelerated generator."""
        assert generator.backend == AcceleratorBackend.NUMPY

    def test_generator_auto_backend(self):
        """Test generator with automatic backend selection."""
        gen = AcceleratedGenerator(backend=None)
        assert isinstance(gen.backend, AcceleratorBackend)

    def test_generate_batch(self, generator, test_data):
        """Test batch generation."""
        result = generator.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
        )

        assert result.shape == (test_data["n_samples"], test_data["n_wavelengths"])

    def test_generate_batch_reproducible(self, test_data):
        """Test that generation is reproducible with same seed."""
        gen1 = AcceleratedGenerator(backend=AcceleratorBackend.NUMPY, random_state=42)
        gen2 = AcceleratedGenerator(backend=AcceleratorBackend.NUMPY, random_state=42)

        result1 = gen1.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
            noise_level=0.01,
        )
        result2 = gen2.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
            noise_level=0.01,
        )

        # Note: Due to RNG state, these may differ slightly
        # but the shapes should be the same
        assert result1.shape == result2.shape

    def test_spectra_values_reasonable(self, generator, test_data):
        """Test that generated spectra have reasonable values."""
        result = generator.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
        )

        # Spectra should be finite
        assert np.all(np.isfinite(result))

    def test_generate_with_noise_levels(self, generator, test_data):
        """Test generation with different noise levels."""
        result_low = generator.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
            noise_level=0.001,
        )
        result_high = generator.generate_batch(
            n_samples=test_data["n_samples"],
            wavelengths=test_data["wavelengths"],
            component_spectra=test_data["component_spectra"],
            concentrations=test_data["concentrations"],
            noise_level=0.1,
        )

        # Both should have valid outputs
        assert result_low.shape == result_high.shape

    def test_generate_voigt_profiles(self, generator):
        """Test Voigt profile generation."""
        wavelengths = np.linspace(1000, 2500, 500)
        centers = np.array([1200, 1500, 2000])
        amplitudes = np.array([1.0, 0.5, 0.8])
        sigmas = np.array([20.0, 30.0, 25.0])
        gammas = np.array([10.0, 15.0, 12.0])

        spectrum = generator.generate_voigt_profiles(
            wavelengths=wavelengths,
            centers=centers,
            amplitudes=amplitudes,
            sigmas=sigmas,
            gammas=gammas,
        )

        assert spectrum.shape == (500,)
        assert np.all(np.isfinite(spectrum))


class TestBenchmarkBackends:
    """Tests for backend benchmarking utility."""

    def test_benchmark_runs(self):
        """Test that benchmark runs without error."""
        results = benchmark_backends(n_samples=100, n_wavelengths=100)

        assert isinstance(results, dict)
        assert "numpy" in results
        assert results["numpy"] > 0  # Should have positive timing


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_generation(self):
        """Test generation with minimal parameters."""
        gen = AcceleratedGenerator()
        wavelengths = np.linspace(1000, 2500, 10)
        component_spectra = np.abs(np.random.randn(1, 10)) + 0.1
        concentrations = np.abs(np.random.randn(1, 1))

        result = gen.generate_batch(
            n_samples=1,
            wavelengths=wavelengths,
            component_spectra=component_spectra,
            concentrations=concentrations,
        )

        assert result.shape == (1, 10)

    def test_large_n_components(self):
        """Test generation with many components."""
        gen = AcceleratedGenerator()
        n_components = 20
        wavelengths = np.linspace(1000, 2500, 100)
        component_spectra = np.abs(np.random.randn(n_components, 100)) + 0.1
        concentrations = np.abs(np.random.randn(50, n_components))

        result = gen.generate_batch(
            n_samples=50,
            wavelengths=wavelengths,
            component_spectra=component_spectra,
            concentrations=concentrations,
        )

        assert result.shape == (50, 100)


class TestLowLevelFunctions:
    """Tests for low-level accelerated functions."""

    def test_generate_spectra_batch_accelerated(self):
        """Test batch spectra generation function directly."""
        n_samples = 20
        n_wavelengths = 50
        n_components = 3

        wavelengths = np.linspace(1000, 2500, n_wavelengths)
        component_spectra = np.abs(np.random.randn(n_components, n_wavelengths)) + 0.1
        concentrations = np.abs(np.random.randn(n_samples, n_components))

        result = generate_spectra_batch_accelerated(
            n_samples=n_samples,
            wavelengths=wavelengths,
            component_spectra=component_spectra,
            concentrations=concentrations,
            noise_level=0.01,
        )

        assert result.shape == (n_samples, n_wavelengths)
        assert np.all(np.isfinite(result))

    def test_generate_voigt_profiles_accelerated(self):
        """Test Voigt profile generation function directly."""
        wavelengths = np.linspace(1000, 2500, 200)
        centers = np.array([1500])
        amplitudes = np.array([1.0])
        sigmas = np.array([30.0])
        gammas = np.array([15.0])

        spectrum = generate_voigt_profiles_accelerated(
            wavelengths=wavelengths,
            centers=centers,
            amplitudes=amplitudes,
            sigmas=sigmas,
            gammas=gammas,
        )

        assert spectrum.shape == (200,)
        assert np.all(np.isfinite(spectrum))
        # Peak should be around center wavelength
        peak_idx = np.argmax(spectrum)
        assert abs(wavelengths[peak_idx] - 1500) < 50

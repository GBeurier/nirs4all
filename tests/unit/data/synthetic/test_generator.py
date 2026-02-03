"""
Unit tests for SyntheticNIRSGenerator class.
"""

import pytest
import numpy as np

from nirs4all.synthesis import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    COMPLEXITY_PARAMS,
)


class TestSyntheticNIRSGeneratorInit:
    """Tests for SyntheticNIRSGenerator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        gen = SyntheticNIRSGenerator()
        # Phase 2: Default wavelength range extended to Vis-NIR (350-2500nm)
        assert gen.wavelength_start == 350
        assert gen.wavelength_end == 2500
        assert gen.wavelength_step == 2
        assert gen.complexity == "realistic"
        assert gen.n_wavelengths > 0

    def test_simple_complexity(self, simple_generator):
        """Test simple complexity initialization."""
        assert simple_generator.complexity == "simple"
        assert simple_generator.params == COMPLEXITY_PARAMS["simple"]

    def test_realistic_complexity(self, realistic_generator):
        """Test realistic complexity initialization."""
        assert realistic_generator.complexity == "realistic"
        assert realistic_generator.params == COMPLEXITY_PARAMS["realistic"]

    def test_complex_complexity(self, complex_generator):
        """Test complex complexity initialization."""
        assert complex_generator.complexity == "complex"
        assert complex_generator.params == COMPLEXITY_PARAMS["complex"]

    def test_invalid_complexity(self):
        """Test error on invalid complexity."""
        with pytest.raises(ValueError, match="complexity must be one of"):
            SyntheticNIRSGenerator(complexity="invalid")

    def test_custom_wavelength_range(self):
        """Test custom wavelength range."""
        gen = SyntheticNIRSGenerator(
            wavelength_start=1200,
            wavelength_end=2000,
            wavelength_step=5,
        )
        assert gen.wavelengths[0] == 1200
        assert gen.wavelengths[-1] <= 2000
        assert gen.wavelength_step == 5

    def test_custom_component_library(self, predefined_library):
        """Test using custom component library."""
        gen = SyntheticNIRSGenerator(component_library=predefined_library)
        assert gen.library.n_components == 3

    def test_wavelength_grid_generation(self, simple_generator):
        """Test wavelength grid is correctly generated."""
        assert len(simple_generator.wavelengths) == simple_generator.n_wavelengths
        assert simple_generator.wavelengths[0] == simple_generator.wavelength_start
        # Check uniform spacing
        diffs = np.diff(simple_generator.wavelengths)
        np.testing.assert_allclose(diffs, simple_generator.wavelength_step)

    def test_component_spectra_precomputed(self, simple_generator):
        """Test that component spectra are precomputed."""
        assert simple_generator.E.shape == (
            simple_generator.library.n_components,
            simple_generator.n_wavelengths,
        )

    def test_repr(self, simple_generator):
        """Test string representation."""
        repr_str = repr(simple_generator)
        assert "SyntheticNIRSGenerator" in repr_str
        assert "simple" in repr_str


class TestGenerateConcentrations:
    """Tests for concentration generation methods."""

    def test_dirichlet_method(self, simple_generator):
        """Test Dirichlet concentration generation."""
        C = simple_generator.generate_concentrations(30, method="dirichlet")

        assert C.shape == (30, simple_generator.library.n_components)
        assert np.all(C >= 0)
        # Dirichlet should sum to 1
        np.testing.assert_allclose(C.sum(axis=1), 1.0, rtol=1e-10)

    def test_uniform_method(self, simple_generator):
        """Test uniform concentration generation."""
        C = simple_generator.generate_concentrations(30, method="uniform")

        assert C.shape == (30, simple_generator.library.n_components)
        assert np.all(C >= 0)
        assert np.all(C <= 1)

    def test_lognormal_method(self, simple_generator):
        """Test lognormal concentration generation."""
        C = simple_generator.generate_concentrations(30, method="lognormal")

        assert C.shape == (30, simple_generator.library.n_components)
        assert np.all(C >= 0)
        # Lognormal should be normalized
        np.testing.assert_allclose(C.sum(axis=1), 1.0, rtol=1e-10)

    def test_correlated_method(self, simple_generator):
        """Test correlated concentration generation."""
        C = simple_generator.generate_concentrations(30, method="correlated")

        assert C.shape == (30, simple_generator.library.n_components)
        assert np.all(C >= 0)
        # Should be normalized
        np.testing.assert_allclose(C.sum(axis=1), 1.0, rtol=1e-10)

    def test_invalid_method(self, simple_generator):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown concentration method"):
            simple_generator.generate_concentrations(100, method="invalid")

    def test_custom_alpha(self, simple_generator):
        """Test custom Dirichlet alpha parameter."""
        n_components = simple_generator.library.n_components
        alpha = np.ones(n_components) * 10.0  # More uniform
        C = simple_generator.generate_concentrations(30, method="dirichlet", alpha=alpha)

        # With high alpha, concentrations should be more uniform
        stds = C.std(axis=0)
        assert np.all(stds < 0.3)  # Less variance than default


class TestGenerate:
    """Tests for main generate() method."""

    def test_generate_basic(self, simple_generator):
        """Test basic spectrum generation."""
        X, Y, E = simple_generator.generate(n_samples=50)

        assert X.shape == (50, simple_generator.n_wavelengths)
        assert Y.shape == (50, simple_generator.library.n_components)
        assert E.shape == (simple_generator.library.n_components, simple_generator.n_wavelengths)

    def test_generate_all_finite(self, simple_generator):
        """Test that all generated values are finite."""
        X, Y, E = simple_generator.generate(n_samples=30)

        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))
        assert np.all(np.isfinite(E))

    def test_generate_with_metadata(self, simple_generator):
        """Test generation with metadata."""
        X, Y, E, metadata = simple_generator.generate(
            n_samples=50, return_metadata=True
        )

        assert "n_samples" in metadata
        assert "n_components" in metadata
        assert "wavelengths" in metadata
        assert "component_names" in metadata
        assert "complexity" in metadata
        assert metadata["n_samples"] == 50

    def test_generate_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        gen1 = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        gen2 = SyntheticNIRSGenerator(complexity="simple", random_state=42)

        X1, Y1, E1 = gen1.generate(n_samples=50)
        X2, Y2, E2 = gen2.generate(n_samples=50)

        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(Y1, Y2)
        np.testing.assert_allclose(E1, E2)

    def test_generate_different_seeds(self):
        """Test that different seeds produce different results."""
        gen1 = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        gen2 = SyntheticNIRSGenerator(complexity="simple", random_state=123)

        X1, _, _ = gen1.generate(n_samples=50)
        X2, _, _ = gen2.generate(n_samples=50)

        assert not np.allclose(X1, X2)

    def test_generate_with_batch_effects(self, simple_generator):
        """Test generation with batch effects."""
        X, Y, E, metadata = simple_generator.generate(
            n_samples=30,
            include_batch_effects=True,
            n_batches=3,
            return_metadata=True,
        )

        assert "batch_ids" in metadata
        assert len(metadata["batch_ids"]) == 30
        assert set(metadata["batch_ids"]) == {0, 1, 2}

    def test_generate_concentration_methods(self, simple_generator):
        """Test generation with different concentration methods."""
        for method in ["dirichlet", "uniform", "lognormal", "correlated"]:
            X, Y, E = simple_generator.generate(
                n_samples=50, concentration_method=method
            )
            assert X.shape[0] == 50
            assert np.all(np.isfinite(X))

    def test_generate_realistic_values(self, realistic_generator):
        """Test that realistic complexity produces typical NIR values."""
        X, Y, E = realistic_generator.generate(n_samples=60)

        # Absorbance values should be in reasonable range for NIR
        assert X.min() > -1.0  # Not too negative
        assert X.max() < 5.0  # Not unreasonably high


class TestBatchEffects:
    """Tests for batch effect generation."""

    def test_batch_effects_shape(self, simple_generator):
        """Test batch effects have correct shape."""
        offsets, gains = simple_generator.generate_batch_effects(
            n_batches=3, samples_per_batch=[50, 50, 50]
        )

        assert offsets.shape == (3, simple_generator.n_wavelengths)
        assert gains.shape == (3,)

    def test_batch_effects_variation(self, simple_generator):
        """Test that batch effects introduce variation."""
        offsets, gains = simple_generator.generate_batch_effects(
            n_batches=5, samples_per_batch=[20] * 5
        )

        # Gains should vary around 1
        assert np.abs(gains.mean() - 1.0) < 0.1
        assert gains.std() > 0.01


class TestCreateDataset:
    """Tests for create_dataset() method."""

    def test_create_dataset_basic(self, simple_generator):
        """Test basic dataset creation."""
        dataset = simple_generator.create_dataset(n_train=80, n_test=20)

        assert dataset is not None
        assert dataset.name == "synthetic_nirs"

    def test_create_dataset_partition_sizes(self, simple_generator):
        """Test dataset partition sizes."""
        dataset = simple_generator.create_dataset(n_train=80, n_test=20)

        # Get train/test counts from the indexer
        partition_values = dataset._indexer.get_column_values("partition")
        train_count = sum(1 for p in partition_values if p == "train")
        test_count = sum(1 for p in partition_values if p == "test")

        # Verify partition sizes
        assert train_count == 80
        assert test_count == 20
        assert dataset.num_samples == 100

    def test_create_dataset_single_target(self, simple_generator):
        """Test dataset creation with single target component."""
        dataset = simple_generator.create_dataset(
            n_train=80,
            n_test=20,
            target_component=0,
        )
        assert dataset is not None

    def test_create_dataset_named_target(self):
        """Test dataset creation with named target component."""
        library = ComponentLibrary.from_predefined(["water", "protein"])
        gen = SyntheticNIRSGenerator(component_library=library, random_state=42)

        dataset = gen.create_dataset(
            n_train=80,
            n_test=20,
            target_component="protein",
        )
        assert dataset is not None


class TestComplexityLevels:
    """Tests comparing different complexity levels."""

    def test_noise_increases_with_complexity(self):
        """Test that noise level increases with complexity."""
        simple = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        realistic = SyntheticNIRSGenerator(complexity="realistic", random_state=42)
        complex_ = SyntheticNIRSGenerator(complexity="complex", random_state=42)

        assert simple.params["noise_base"] < realistic.params["noise_base"]
        assert realistic.params["noise_base"] < complex_.params["noise_base"]

    def test_artifact_probability_increases(self):
        """Test that artifact probability increases with complexity."""
        simple = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        realistic = SyntheticNIRSGenerator(complexity="realistic", random_state=42)

        assert simple.params["artifact_prob"] == 0.0
        assert realistic.params["artifact_prob"] > 0.0

    def test_different_complexities_produce_different_spectra(self):
        """Test that different complexities produce visibly different spectra."""
        simple_gen = SyntheticNIRSGenerator(complexity="simple", random_state=42)
        complex_gen = SyntheticNIRSGenerator(complexity="complex", random_state=42)

        X_simple, _, _ = simple_gen.generate(n_samples=30)
        X_complex, _, _ = complex_gen.generate(n_samples=30)

        # Complex should have more variance (more effects applied)
        simple_variance = X_simple.var()
        complex_variance = X_complex.var()

        assert complex_variance > simple_variance


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample(self, simple_generator):
        """Test generation of single sample."""
        X, Y, E = simple_generator.generate(n_samples=1)

        assert X.shape == (1, simple_generator.n_wavelengths)
        assert Y.shape == (1, simple_generator.library.n_components)

    def test_large_sample_count(self, simple_generator):
        """Test generation of large number of samples."""
        X, Y, E = simple_generator.generate(n_samples=100)

        assert X.shape[0] == 100
        assert np.all(np.isfinite(X))

    def test_narrow_wavelength_range(self):
        """Test with narrow wavelength range."""
        gen = SyntheticNIRSGenerator(
            wavelength_start=1400,
            wavelength_end=1500,
            wavelength_step=2,
            complexity="simple",
            random_state=42,
        )

        X, Y, E = gen.generate(n_samples=50)
        assert X.shape[1] == len(gen.wavelengths)

    def test_large_wavelength_step(self):
        """Test with large wavelength step."""
        gen = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2500,
            wavelength_step=20,
            complexity="simple",
            random_state=42,
        )

        assert gen.n_wavelengths < 100
        X, Y, E = gen.generate(n_samples=50)
        assert np.all(np.isfinite(X))

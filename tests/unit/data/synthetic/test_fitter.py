"""
Unit tests for RealDataFitter and FittedParameters.

Tests real data fitting and parameter estimation for synthetic generation.
"""

import json
import numpy as np
import pytest
from pathlib import Path


class TestSpectralProperties:
    """Tests for SpectralProperties dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        from nirs4all.data.synthetic import SpectralProperties

        props = SpectralProperties()
        assert props.name == "dataset"
        assert props.n_samples == 0
        assert props.n_wavelengths == 0
        assert props.global_mean == 0.0

    def test_with_values(self):
        """Test initialization with values."""
        from nirs4all.data.synthetic import SpectralProperties

        props = SpectralProperties(
            name="test",
            n_samples=100,
            n_wavelengths=50,
            global_mean=0.5,
        )
        assert props.name == "test"
        assert props.n_samples == 100


class TestFittedParameters:
    """Tests for FittedParameters dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        from nirs4all.data.synthetic import FittedParameters

        params = FittedParameters()
        assert params.wavelength_start == 1000.0
        assert params.wavelength_end == 2500.0
        assert params.complexity == "realistic"

    def test_to_generator_kwargs(self):
        """Test conversion to generator kwargs."""
        from nirs4all.data.synthetic import FittedParameters

        params = FittedParameters(
            wavelength_start=1100,
            wavelength_end=2000,
            wavelength_step=4,
            complexity="simple",
        )

        kwargs = params.to_generator_kwargs()
        assert kwargs["wavelength_start"] == 1100
        assert kwargs["wavelength_end"] == 2000
        assert kwargs["wavelength_step"] == 4
        assert kwargs["complexity"] == "simple"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from nirs4all.data.synthetic import FittedParameters

        params = FittedParameters(
            global_slope_mean=0.05,
            noise_base=0.002,
        )

        d = params.to_dict()
        assert d["global_slope_mean"] == 0.05
        assert d["noise_base"] == 0.002
        assert "source_name" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        from nirs4all.data.synthetic import FittedParameters

        data = {
            "wavelength_start": 1050,
            "wavelength_end": 2200,
            "complexity": "complex",
        }

        params = FittedParameters.from_dict(data)
        assert params.wavelength_start == 1050
        assert params.wavelength_end == 2200
        assert params.complexity == "complex"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading parameters."""
        from nirs4all.data.synthetic import FittedParameters

        params = FittedParameters(
            wavelength_start=1100,
            global_slope_mean=0.03,
            source_name="test_data",
        )

        filepath = tmp_path / "params.json"
        params.save(str(filepath))

        assert filepath.exists()

        loaded = FittedParameters.load(str(filepath))
        assert loaded.wavelength_start == 1100
        assert loaded.global_slope_mean == 0.03
        assert loaded.source_name == "test_data"


class TestComputeSpectralProperties:
    """Tests for compute_spectral_properties function."""

    @pytest.fixture
    def synthetic_spectra(self):
        """Create synthetic spectra for testing."""
        rng = np.random.default_rng(42)
        n_samples, n_wavelengths = 100, 200

        # Create spectra with known properties
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        # Base spectrum with slope
        x_norm = (wavelengths - 1000) / 1500
        base = 0.3 + 0.2 * x_norm

        # Add variation
        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            noise = rng.normal(0, 0.01, n_wavelengths)
            scale = rng.uniform(0.9, 1.1)
            X[i] = base * scale + noise

        return X, wavelengths

    def test_basic_properties(self, synthetic_spectra):
        """Test computation of basic properties."""
        from nirs4all.data.synthetic import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths, name="test")

        assert props.n_samples == 100
        assert props.n_wavelengths == 200
        assert props.name == "test"
        assert props.mean_spectrum is not None
        assert props.std_spectrum is not None
        assert props.global_mean > 0

    def test_slope_analysis(self, synthetic_spectra):
        """Test slope computation."""
        from nirs4all.data.synthetic import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should detect positive slope
        assert props.mean_slope > 0
        assert props.slopes is not None
        assert len(props.slopes) == 100

    def test_noise_estimation(self, synthetic_spectra):
        """Test noise estimation."""
        from nirs4all.data.synthetic import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should have reasonable noise estimate
        assert props.noise_estimate > 0
        assert props.snr_estimate > 0

    def test_pca_analysis(self, synthetic_spectra):
        """Test PCA analysis."""
        from nirs4all.data.synthetic import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should have PCA results
        assert props.pca_explained_variance is not None
        assert len(props.pca_explained_variance) > 0
        assert props.pca_n_components_95 >= 1

    def test_without_wavelengths(self, synthetic_spectra):
        """Test with default wavelengths."""
        from nirs4all.data.synthetic import compute_spectral_properties

        X, _ = synthetic_spectra
        props = compute_spectral_properties(X)

        assert props.wavelengths is not None
        assert len(props.wavelengths) == 200


class TestRealDataFitter:
    """Tests for RealDataFitter class."""

    @pytest.fixture
    def realistic_spectra(self):
        """Create realistic-looking spectra for fitting tests."""
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        # Use the generator to create realistic spectra
        generator = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2500,
            wavelength_step=2,
            complexity="realistic",
            random_state=42,
        )

        X, _, _ = generator.generate(n_samples=200)
        return X, generator.wavelengths

    def test_init(self):
        """Test initialization."""
        from nirs4all.data.synthetic import RealDataFitter

        fitter = RealDataFitter()
        assert fitter.source_properties is None
        assert fitter.fitted_params is None

    def test_fit_array(self, realistic_spectra):
        """Test fitting from numpy array."""
        from nirs4all.data.synthetic import RealDataFitter

        X, wavelengths = realistic_spectra
        fitter = RealDataFitter()

        params = fitter.fit(X, wavelengths=wavelengths, name="test")

        assert params is not None
        assert params.source_name == "test"
        assert params.wavelength_start == 1000
        assert params.wavelength_end == 2500
        assert fitter.source_properties is not None
        assert fitter.fitted_params is not None

    def test_fit_estimates_parameters(self, realistic_spectra):
        """Test that fitting estimates reasonable parameters."""
        from nirs4all.data.synthetic import RealDataFitter

        X, wavelengths = realistic_spectra
        fitter = RealDataFitter()

        params = fitter.fit(X, wavelengths=wavelengths)

        # Check all parameters are set
        assert params.noise_base > 0
        assert params.global_slope_std >= 0
        assert params.complexity in ["simple", "realistic", "complex"]

    def test_fit_validates_input(self):
        """Test input validation."""
        from nirs4all.data.synthetic import RealDataFitter

        fitter = RealDataFitter()

        # 1D array should fail
        with pytest.raises(ValueError, match="2D"):
            fitter.fit(np.random.random(100))

        # Too few samples
        with pytest.raises(ValueError, match="at least 5"):
            fitter.fit(np.random.random((3, 50)))

    def test_evaluate_similarity(self, realistic_spectra):
        """Test similarity evaluation."""
        from nirs4all.data.synthetic import RealDataFitter, SyntheticNIRSGenerator

        X_real, wavelengths = realistic_spectra
        fitter = RealDataFitter()
        params = fitter.fit(X_real, wavelengths=wavelengths)

        # Generate synthetic with fitted params
        generator = SyntheticNIRSGenerator(
            **params.to_generator_kwargs(),
            random_state=42,
        )
        X_synth, _, _ = generator.generate(200)

        metrics = fitter.evaluate_similarity(X_synth, wavelengths)

        assert "overall_score" in metrics
        assert "mean_rel_diff" in metrics
        assert "std_rel_diff" in metrics
        assert metrics["overall_score"] >= 0

    def test_evaluate_similarity_requires_fit(self):
        """Test that evaluate requires prior fit."""
        from nirs4all.data.synthetic import RealDataFitter

        fitter = RealDataFitter()

        with pytest.raises(RuntimeError, match="fit"):
            fitter.evaluate_similarity(np.random.random((50, 100)))

    def test_get_tuning_recommendations(self, realistic_spectra):
        """Test tuning recommendations."""
        from nirs4all.data.synthetic import RealDataFitter

        X, wavelengths = realistic_spectra
        fitter = RealDataFitter()
        fitter.fit(X, wavelengths=wavelengths)

        recs = fitter.get_tuning_recommendations()

        assert isinstance(recs, list)
        # May or may not have recommendations depending on data

    def test_get_recommendations_requires_fit(self):
        """Test that recommendations require prior fit."""
        from nirs4all.data.synthetic import RealDataFitter

        fitter = RealDataFitter()
        recs = fitter.get_tuning_recommendations()
        assert "fit()" in recs[0]


class TestFitToRealData:
    """Tests for fit_to_real_data convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        from nirs4all.data.synthetic import fit_to_real_data

        rng = np.random.default_rng(42)
        X = rng.random((50, 100))
        wavelengths = np.linspace(1000, 2000, 100)

        params = fit_to_real_data(X, wavelengths=wavelengths, name="quick_test")

        assert params is not None
        assert params.source_name == "quick_test"


class TestCompareDatasets:
    """Tests for compare_datasets convenience function."""

    def test_convenience_function(self):
        """Test the comparison function."""
        from nirs4all.data.synthetic import compare_datasets

        rng = np.random.default_rng(42)
        X_real = rng.random((50, 100))
        X_synth = rng.random((50, 100)) * 1.1  # Slightly different

        metrics = compare_datasets(X_synth, X_real)

        assert "overall_score" in metrics
        assert metrics["overall_score"] >= 0


class TestBuilderFitTo:
    """Tests for SyntheticDatasetBuilder.fit_to method."""

    def test_fit_to_array(self):
        """Test fitting builder to array."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        rng = np.random.default_rng(42)
        X_template = rng.random((50, 100))
        wavelengths = np.linspace(1000, 2000, 100)

        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.fit_to(X_template, wavelengths=wavelengths)

        # Check wavelength range was set
        assert builder.state.wavelength_start == 1000
        assert builder.state.wavelength_end == 2000

    def test_fit_to_chains(self):
        """Test that fit_to returns self for chaining."""
        from nirs4all.data.synthetic import SyntheticDatasetBuilder

        X_template = np.random.random((50, 100))
        wavelengths = np.linspace(1000, 2000, 100)

        result = (
            SyntheticDatasetBuilder(n_samples=100)
            .fit_to(X_template, wavelengths=wavelengths)
            .with_partitions(train_ratio=0.8)
        )

        assert isinstance(result, SyntheticDatasetBuilder)


class TestGenerateFromTemplate:
    """Tests for generate.from_template function."""

    def test_from_template_array(self):
        """Test from_template with array input."""
        from nirs4all.api.generate import from_template

        rng = np.random.default_rng(42)
        X_template = rng.random((50, 100))
        wavelengths = np.linspace(1000, 2000, 100)

        dataset = from_template(
            X_template,
            n_samples=100,
            wavelengths=wavelengths,
            random_state=42,
        )

        # Should return a SpectroDataset
        assert hasattr(dataset, "x")

    def test_from_template_as_arrays(self):
        """Test from_template returning arrays."""
        from nirs4all.api.generate import from_template

        X_template = np.random.random((50, 100))
        wavelengths = np.linspace(1000, 2000, 100)

        X, y = from_template(
            X_template,
            n_samples=100,
            wavelengths=wavelengths,
            random_state=42,
            as_dataset=False,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 100

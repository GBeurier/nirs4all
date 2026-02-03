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
        from nirs4all.synthesis import SpectralProperties

        props = SpectralProperties()
        assert props.name == "dataset"
        assert props.n_samples == 0
        assert props.n_wavelengths == 0
        assert props.global_mean == 0.0

    def test_with_values(self):
        """Test initialization with values."""
        from nirs4all.synthesis import SpectralProperties

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
        from nirs4all.synthesis import FittedParameters

        params = FittedParameters()
        assert params.wavelength_start == 1000.0
        assert params.wavelength_end == 2500.0
        assert params.complexity == "realistic"

    def test_to_generator_kwargs(self):
        """Test conversion to generator kwargs."""
        from nirs4all.synthesis import FittedParameters

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
        from nirs4all.synthesis import FittedParameters

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
        from nirs4all.synthesis import FittedParameters

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
        from nirs4all.synthesis import FittedParameters

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
        from nirs4all.synthesis import compute_spectral_properties

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
        from nirs4all.synthesis import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should detect positive slope
        assert props.mean_slope > 0
        assert props.slopes is not None
        assert len(props.slopes) == 100

    def test_noise_estimation(self, synthetic_spectra):
        """Test noise estimation."""
        from nirs4all.synthesis import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should have reasonable noise estimate
        assert props.noise_estimate > 0
        assert props.snr_estimate > 0

    def test_pca_analysis(self, synthetic_spectra):
        """Test PCA analysis."""
        from nirs4all.synthesis import compute_spectral_properties

        X, wavelengths = synthetic_spectra
        props = compute_spectral_properties(X, wavelengths)

        # Should have PCA results
        assert props.pca_explained_variance is not None
        assert len(props.pca_explained_variance) > 0
        assert props.pca_n_components_95 >= 1

    def test_without_wavelengths(self, synthetic_spectra):
        """Test with default wavelengths."""
        from nirs4all.synthesis import compute_spectral_properties

        X, _ = synthetic_spectra
        props = compute_spectral_properties(X)

        assert props.wavelengths is not None
        assert len(props.wavelengths) == 200


class TestRealDataFitter:
    """Tests for RealDataFitter class."""

    @pytest.fixture
    def realistic_spectra(self):
        """Create realistic-looking spectra for fitting tests."""
        from nirs4all.synthesis import SyntheticNIRSGenerator

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
        from nirs4all.synthesis import RealDataFitter

        fitter = RealDataFitter()
        assert fitter.source_properties is None
        assert fitter.fitted_params is None

    def test_fit_array(self, realistic_spectra):
        """Test fitting from numpy array."""
        from nirs4all.synthesis import RealDataFitter

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
        from nirs4all.synthesis import RealDataFitter

        X, wavelengths = realistic_spectra
        fitter = RealDataFitter()

        params = fitter.fit(X, wavelengths=wavelengths)

        # Check all parameters are set
        assert params.noise_base > 0
        assert params.global_slope_std >= 0
        assert params.complexity in ["simple", "realistic", "complex"]

    def test_fit_validates_input(self):
        """Test input validation."""
        from nirs4all.synthesis import RealDataFitter

        fitter = RealDataFitter()

        # 1D array should fail
        with pytest.raises(ValueError, match="2D"):
            fitter.fit(np.random.random(100))

        # Too few samples
        with pytest.raises(ValueError, match="at least 5"):
            fitter.fit(np.random.random((3, 50)))

    def test_evaluate_similarity(self, realistic_spectra):
        """Test similarity evaluation."""
        from nirs4all.synthesis import RealDataFitter, SyntheticNIRSGenerator

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
        from nirs4all.synthesis import RealDataFitter

        fitter = RealDataFitter()

        with pytest.raises(RuntimeError, match="fit"):
            fitter.evaluate_similarity(np.random.random((50, 100)))

    def test_get_tuning_recommendations(self, realistic_spectra):
        """Test tuning recommendations."""
        from nirs4all.synthesis import RealDataFitter

        X, wavelengths = realistic_spectra
        fitter = RealDataFitter()
        fitter.fit(X, wavelengths=wavelengths)

        recs = fitter.get_tuning_recommendations()

        assert isinstance(recs, list)
        # May or may not have recommendations depending on data

    def test_get_recommendations_requires_fit(self):
        """Test that recommendations require prior fit."""
        from nirs4all.synthesis import RealDataFitter

        fitter = RealDataFitter()
        recs = fitter.get_tuning_recommendations()
        assert "fit()" in recs[0]


class TestFitToRealData:
    """Tests for fit_to_real_data convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        from nirs4all.synthesis import fit_to_real_data

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
        from nirs4all.synthesis import compare_datasets

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
        from nirs4all.synthesis import SyntheticDatasetBuilder

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
        from nirs4all.synthesis import SyntheticDatasetBuilder

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


# ============================================================================
# Phase 5: Spectral Fitting Tools (ComponentFitter)
# ============================================================================


class TestComponentFitResult:
    """Tests for ComponentFitResult dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        from nirs4all.synthesis import ComponentFitResult

        result = ComponentFitResult(
            component_names=["water", "protein"],
            concentrations=np.array([0.5, 0.3]),
            baseline_coefficients=np.array([0.1, 0.01]),
            fitted_spectrum=np.zeros(100),
            residuals=np.zeros(100),
            r_squared=0.95,
            rmse=0.01,
        )

        assert result.r_squared == 0.95
        assert result.rmse == 0.01
        assert len(result.component_names) == 2

    def test_to_dict(self):
        """Test to_dict conversion."""
        from nirs4all.synthesis import ComponentFitResult

        result = ComponentFitResult(
            component_names=["water", "protein"],
            concentrations=np.array([0.5, 0.3]),
            baseline_coefficients=None,
            fitted_spectrum=np.zeros(100),
            residuals=np.zeros(100),
            r_squared=0.95,
            rmse=0.01,
        )

        d = result.to_dict()
        assert d["water"] == 0.5
        assert d["protein"] == 0.3

    def test_top_components(self):
        """Test top_components method."""
        from nirs4all.synthesis import ComponentFitResult

        result = ComponentFitResult(
            component_names=["water", "protein", "lipid", "starch"],
            concentrations=np.array([0.5, 0.3, 0.15, 0.05]),
            baseline_coefficients=None,
            fitted_spectrum=np.zeros(100),
            residuals=np.zeros(100),
            r_squared=0.95,
            rmse=0.01,
        )

        top2 = result.top_components(2)
        assert len(top2) == 2
        assert top2[0][0] == "water"
        assert top2[1][0] == "protein"

        # Test with threshold
        top_filtered = result.top_components(10, threshold=0.1)
        assert len(top_filtered) == 3  # water, protein, lipid

    def test_summary(self):
        """Test summary string generation."""
        from nirs4all.synthesis import ComponentFitResult

        result = ComponentFitResult(
            component_names=["water", "protein"],
            concentrations=np.array([0.5, 0.3]),
            baseline_coefficients=np.array([0.1, 0.01, 0.001]),
            fitted_spectrum=np.zeros(100),
            residuals=np.zeros(100),
            r_squared=0.95,
            rmse=0.01,
        )

        summary = result.summary()
        assert "R²" in summary
        assert "water" in summary
        assert "Baseline" in summary


class TestComponentFitter:
    """Tests for ComponentFitter class."""

    @pytest.fixture
    def wavelengths(self):
        """Standard wavelength grid for testing."""
        return np.arange(1000, 2500, 2)

    @pytest.fixture
    def synthetic_spectrum(self, wavelengths):
        """Generate a synthetic spectrum for testing."""
        from nirs4all.synthesis import SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2498,
            wavelength_step=2,
            complexity="simple",
            random_state=42,
        )
        X, _, _ = gen.generate(n_samples=1)
        return X[0], wavelengths

    def test_init_default(self):
        """Test default initialization."""
        from nirs4all.synthesis import ComponentFitter

        fitter = ComponentFitter()

        # Should use all available components
        assert len(fitter.component_names) > 0
        assert fitter.fit_baseline is True
        assert fitter.baseline_order == 2

    def test_init_with_components(self, wavelengths):
        """Test initialization with specific components."""
        from nirs4all.synthesis import ComponentFitter

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
        )

        assert len(fitter.component_names) == 3
        assert "water" in fitter.component_names

    def test_init_with_wavelengths(self):
        """Test initialization with custom wavelengths."""
        from nirs4all.synthesis import ComponentFitter

        custom_wl = np.linspace(1000, 2000, 50)
        fitter = ComponentFitter(wavelengths=custom_wl)

        assert len(fitter.wavelengths) == 50
        assert np.allclose(fitter.wavelengths, custom_wl)

    def test_fit_single_spectrum(self, synthetic_spectrum):
        """Test fitting a single spectrum."""
        from nirs4all.synthesis import ComponentFitter

        spectrum, wavelengths = synthetic_spectrum

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
            fit_baseline=True,
        )

        result = fitter.fit(spectrum)

        assert result.r_squared >= 0  # R² can vary
        assert len(result.concentrations) == 3
        assert result.fitted_spectrum.shape == spectrum.shape
        assert result.residuals.shape == spectrum.shape

    def test_fit_nnls(self, synthetic_spectrum):
        """Test NNLS fitting produces non-negative concentrations."""
        from nirs4all.synthesis import ComponentFitter

        spectrum, wavelengths = synthetic_spectrum

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
        )

        result = fitter.fit(spectrum, method="nnls")

        # NNLS should produce non-negative concentrations
        assert np.all(result.concentrations >= 0)

    def test_fit_lsq(self, synthetic_spectrum):
        """Test unconstrained least squares fitting."""
        from nirs4all.synthesis import ComponentFitter

        spectrum, wavelengths = synthetic_spectrum

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
        )

        result = fitter.fit(spectrum, method="lsq")

        # LSQ can have negative concentrations
        assert isinstance(result.r_squared, float)

    def test_fit_wrong_length(self, wavelengths):
        """Test that wrong spectrum length raises error."""
        from nirs4all.synthesis import ComponentFitter

        fitter = ComponentFitter(wavelengths=wavelengths)

        wrong_spectrum = np.zeros(100)  # Wrong length
        with pytest.raises(ValueError, match="does not match"):
            fitter.fit(wrong_spectrum)

    def test_fit_batch(self, wavelengths):
        """Test batch fitting."""
        from nirs4all.synthesis import ComponentFitter, SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2498,
            wavelength_step=2,
            complexity="simple",
            random_state=42,
        )
        X, _, _ = gen.generate(n_samples=10)

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
        )

        results = fitter.fit_batch(X, n_jobs=1)

        assert len(results) == 10
        assert all(isinstance(r.r_squared, float) for r in results)

    def test_suggest_components(self, synthetic_spectrum):
        """Test component suggestion."""
        from nirs4all.synthesis import ComponentFitter

        spectrum, wavelengths = synthetic_spectrum

        fitter = ComponentFitter(wavelengths=wavelengths)

        suggestions = fitter.suggest_components(spectrum, top_n=5)

        assert len(suggestions) <= 5
        assert all(isinstance(s[0], str) for s in suggestions)
        assert all(isinstance(s[1], float) for s in suggestions)

    def test_get_concentration_matrix(self, wavelengths):
        """Test concentration matrix extraction."""
        from nirs4all.synthesis import ComponentFitter, SyntheticNIRSGenerator

        gen = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2498,
            wavelength_step=2,
            complexity="simple",
            random_state=42,
        )
        X, _, _ = gen.generate(n_samples=5)

        fitter = ComponentFitter(
            component_names=["water", "protein", "lipid"],
            wavelengths=wavelengths,
        )

        C, names = fitter.get_concentration_matrix(X, n_jobs=1)

        assert C.shape == (5, 3)
        assert len(names) == 3


class TestFitComponentsConvenience:
    """Tests for fit_components convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        from nirs4all.synthesis import fit_components, SyntheticNIRSGenerator

        wavelengths = np.arange(1000, 2500, 2)
        gen = SyntheticNIRSGenerator(
            wavelength_start=1000,
            wavelength_end=2498,
            wavelength_step=2,
            complexity="simple",
            random_state=42,
        )
        X, _, _ = gen.generate(n_samples=1)

        result = fit_components(
            X[0],
            wavelengths,
            component_names=["water", "protein"],
        )

        assert isinstance(result.r_squared, float)
        assert len(result.component_names) == 2


class TestPreprocessingInference:
    """Tests for preprocessing type detection."""

    def test_preprocessing_inference_dataclass(self):
        """Test PreprocessingInference dataclass."""
        from nirs4all.synthesis import PreprocessingInference, PreprocessingType

        inference = PreprocessingInference()
        assert inference.preprocessing_type == PreprocessingType.RAW_ABSORBANCE
        assert inference.is_preprocessed is False
        assert inference.confidence == 0.0

    def test_detect_second_derivative(self):
        """Test detection of second derivative data."""
        from nirs4all.synthesis import RealDataFitter, PreprocessingType

        np.random.seed(42)
        n_samples, n_wl = 100, 200
        wl = np.linspace(800, 1600, n_wl)

        # Simulate second derivative: oscillatory, zero mean, small range
        base = np.sin(np.linspace(0, 10 * np.pi, n_wl)) * 0.02
        X = base + np.random.randn(n_samples, n_wl) * 0.005

        fitter = RealDataFitter()
        params = fitter.fit(X, wavelengths=wl, name="test_2nd_deriv")

        assert params.is_preprocessed is True
        assert params.preprocessing_type == "second_derivative"
        assert params.preprocessing_inference is not None
        assert params.preprocessing_inference.preprocessing_type == PreprocessingType.SECOND_DERIVATIVE

    def test_detect_raw_absorbance(self):
        """Test detection of raw absorbance data."""
        from nirs4all.synthesis import RealDataFitter, PreprocessingType

        np.random.seed(42)
        n_samples, n_wl = 100, 200
        wl = np.linspace(800, 1600, n_wl)

        # Simulate raw absorbance: positive values, range 0.2-1.5
        X = np.random.uniform(0.3, 1.2, (n_samples, n_wl))
        # Add some spectral shape
        X += 0.3 * np.sin(np.linspace(0, np.pi, n_wl))

        fitter = RealDataFitter()
        params = fitter.fit(X, wavelengths=wl, name="test_raw")

        assert params.is_preprocessed is False
        assert params.preprocessing_type == "raw_absorbance"
        assert params.preprocessing_inference.preprocessing_type == PreprocessingType.RAW_ABSORBANCE

    def test_detect_snv(self):
        """Test detection of SNV-corrected data."""
        from nirs4all.synthesis import RealDataFitter, PreprocessingType

        np.random.seed(42)
        n_samples, n_wl = 100, 200
        wl = np.linspace(800, 1600, n_wl)

        # Simulate SNV: per-sample mean ~0, std ~1
        X = np.random.randn(n_samples, n_wl)

        fitter = RealDataFitter()
        params = fitter.fit(X, wavelengths=wl, name="test_snv")

        assert params.is_preprocessed is True
        # Should detect as SNV or similar
        assert params.preprocessing_inference.per_sample_std_variation < 0.3

    def test_apply_matching_preprocessing(self):
        """Test apply_matching_preprocessing method."""
        from nirs4all.synthesis import RealDataFitter, SyntheticNIRSGenerator

        np.random.seed(42)
        n_samples, n_wl = 50, 100
        wl = np.linspace(800, 1600, n_wl)

        # Create second derivative real data
        base = np.sin(np.linspace(0, 8 * np.pi, n_wl)) * 0.03
        X_real = base + np.random.randn(n_samples, n_wl) * 0.008

        fitter = RealDataFitter()
        params = fitter.fit(X_real, wavelengths=wl, name="test")

        assert params.is_preprocessed is True

        # Generate raw synthetic
        gen = SyntheticNIRSGenerator(
            wavelength_start=800,
            wavelength_end=1600,
            wavelength_step=(1600 - 800) / (n_wl - 1),
            random_state=42,
        )
        X_raw, _, _ = gen.generate(n_samples=30)

        # Apply preprocessing
        X_matched = fitter.apply_matching_preprocessing(X_raw)

        # Should now match real data range
        real_min, real_max = X_real.min(), X_real.max()
        assert X_matched.min() >= real_min - 0.01
        assert X_matched.max() <= real_max + 0.01

    def test_fitted_parameters_includes_preprocessing(self):
        """Test that FittedParameters includes preprocessing fields."""
        from nirs4all.synthesis import FittedParameters

        params = FittedParameters()
        assert hasattr(params, "preprocessing_type")
        assert hasattr(params, "is_preprocessed")
        assert hasattr(params, "preprocessing_inference")
        assert params.preprocessing_type == "raw_absorbance"
        assert params.is_preprocessed is False

    def test_summary_includes_preprocessing(self):
        """Test that summary() includes preprocessing info."""
        from nirs4all.synthesis import RealDataFitter

        np.random.seed(42)
        X = np.sin(np.linspace(0, 10 * np.pi, 200)) * 0.02
        X = X.reshape(1, -1) + np.random.randn(50, 200) * 0.005

        fitter = RealDataFitter()
        params = fitter.fit(X, name="test")

        summary = params.summary()
        assert "Preprocessing Detection" in summary
        assert "Type:" in summary
        assert "Is preprocessed:" in summary

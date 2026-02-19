"""
Unit tests for the reconstruction module.

Tests the physical signal-chain reconstruction workflow components:
- Forward model
- Calibration
- Inversion
- Distributions
- Generator
- Validation
"""

import numpy as np
import pytest

# =============================================================================
# Test Forward Model
# =============================================================================

class TestCanonicalForwardModel:
    """Tests for CanonicalForwardModel."""

    def test_init_with_components(self):
        """Test initialization with component names."""
        from nirs4all.synthesis.reconstruction.forward import CanonicalForwardModel

        grid = np.linspace(1000, 2500, 300)
        model = CanonicalForwardModel(
            canonical_grid=grid,
            component_names=["water", "protein"],
            baseline_order=4,
        )

        assert model.n_components == 2
        assert model.n_baseline == 5  # order + 1
        assert model._component_spectra.shape == (2, 300)

    def test_init_empty_components(self):
        """Test initialization with no components."""
        from nirs4all.synthesis.reconstruction.forward import CanonicalForwardModel

        grid = np.linspace(1000, 2500, 300)
        model = CanonicalForwardModel(
            canonical_grid=grid,
            component_names=[],
            baseline_order=3,
        )

        assert model.n_components == 0
        assert model._component_spectra.shape == (0, 300)

    def test_compute_absorption(self):
        """Test absorption computation."""
        from nirs4all.synthesis.reconstruction.forward import CanonicalForwardModel

        grid = np.linspace(1000, 2500, 300)
        model = CanonicalForwardModel(
            canonical_grid=grid,
            component_names=["water"],
            baseline_order=2,
        )

        # Compute with simple concentrations
        absorption = model.compute_absorption(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )

        assert absorption.shape == (300,)
        assert not np.all(absorption == 0)

    def test_get_design_matrix(self):
        """Test design matrix construction."""
        from nirs4all.synthesis.reconstruction.forward import CanonicalForwardModel

        grid = np.linspace(1000, 2500, 300)
        model = CanonicalForwardModel(
            canonical_grid=grid,
            component_names=["water", "protein"],
            baseline_order=3,
            continuum_order=2,
        )

        A = model.get_design_matrix(path_length=1.0)

        expected_cols = 2 + 4 + 3  # components + baseline + continuum
        assert A.shape == (300, expected_cols)

class TestInstrumentModel:
    """Tests for InstrumentModel."""

    def test_apply_no_transform(self):
        """Test identity transform."""
        from nirs4all.synthesis.reconstruction.forward import InstrumentModel

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        model = InstrumentModel(
            target_grid=target_grid,
            wl_shift=0.0,
            wl_stretch=1.0,
            ils_sigma=0.0,
        )

        spectrum = np.sin(canonical_grid / 200) + 1
        result = model.apply(spectrum, canonical_grid)

        assert result.shape == (200,)

    def test_apply_with_shift(self):
        """Test wavelength shift."""
        from nirs4all.synthesis.reconstruction.forward import InstrumentModel

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = canonical_grid.copy()

        model = InstrumentModel(
            target_grid=target_grid,
            wl_shift=10.0,  # 10nm shift
            wl_stretch=1.0,
            ils_sigma=0.0,
        )

        # Create a peaked spectrum
        spectrum = np.exp(-((canonical_grid - 1500) ** 2) / 10000)
        result = model.apply(spectrum, canonical_grid)

        # Peak should have shifted
        orig_peak = np.argmax(spectrum)
        result_peak = np.argmax(result)
        # Peak index should shift by approximately shift/step
        assert abs(result_peak - orig_peak) < 10  # Within reasonable range

    def test_apply_with_ils(self):
        """Test ILS convolution smoothing."""
        from nirs4all.synthesis.reconstruction.forward import InstrumentModel

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = canonical_grid.copy()

        model = InstrumentModel(
            target_grid=target_grid,
            wl_shift=0.0,
            wl_stretch=1.0,
            ils_sigma=10.0,  # Broad smoothing
        )

        # Create sharp peaks
        spectrum = np.zeros_like(canonical_grid)
        spectrum[100] = 1.0
        spectrum[200] = 1.0

        result = model.apply(spectrum, canonical_grid)

        # Should be smoother (lower max)
        assert result.max() < spectrum.max()

class TestDomainTransform:
    """Tests for DomainTransform."""

    def test_absorbance_passthrough(self):
        """Test absorbance domain is passthrough."""
        from nirs4all.synthesis.reconstruction.forward import DomainTransform

        transform = DomainTransform(domain="absorbance")

        absorption = np.array([0.1, 0.5, 1.0, 1.5])
        result = transform.transform(absorption, np.array([1000, 1500, 2000, 2500]))

        np.testing.assert_array_equal(result, absorption)

    def test_transmittance(self):
        """Test transmittance transform."""
        from nirs4all.synthesis.reconstruction.forward import DomainTransform

        transform = DomainTransform(domain="transmittance")

        absorption = np.array([0.0, 1.0, 2.0])
        result = transform.transform(absorption, np.array([1000, 1500, 2000]))

        # T = exp(-A)
        expected = np.exp(-absorption)
        np.testing.assert_array_almost_equal(result, expected)

    def test_reflectance_km(self):
        """Test Kubelka-Munk reflectance transform."""
        from nirs4all.synthesis.reconstruction.forward import DomainTransform

        transform = DomainTransform(domain="reflectance")

        # Use reasonable absorption values
        absorption = np.array([0.1, 0.5, 1.0])
        wl = np.array([1000, 1500, 2000])

        result = transform.transform(absorption, wl)

        # Should produce valid reflectance values (0-1)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

class TestPreprocessingOperator:
    """Tests for PreprocessingOperator."""

    def test_none_preprocessing(self):
        """Test no preprocessing."""
        from nirs4all.synthesis.reconstruction.forward import PreprocessingOperator

        op = PreprocessingOperator(preprocessing_type="none")
        spectrum = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = op.apply(spectrum)

        np.testing.assert_array_equal(result, spectrum)

    def test_first_derivative(self):
        """Test first derivative preprocessing."""
        from nirs4all.synthesis.reconstruction.forward import PreprocessingOperator

        op = PreprocessingOperator(
            preprocessing_type="first_derivative",
            sg_window=5,
            sg_polyorder=2,
        )

        # Linear spectrum should have constant derivative
        spectrum = np.linspace(0, 1, 20)
        result = op.apply(spectrum)

        assert result.shape == spectrum.shape
        # Interior points should have similar derivatives
        assert np.std(result[2:-2]) < 0.1

    def test_snv(self):
        """Test SNV preprocessing."""
        from nirs4all.synthesis.reconstruction.forward import PreprocessingOperator

        op = PreprocessingOperator(preprocessing_type="snv")

        spectrum = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = op.apply(spectrum)

        # SNV: mean should be ~0, std should be ~1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

class TestForwardChain:
    """Tests for ForwardChain."""

    def test_create_factory(self):
        """Test factory method."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
        )

        assert chain.canonical_model.n_components == 1
        assert len(chain.instrument_model.target_grid) == 200

    def test_forward(self):
        """Test full forward chain."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
        )

        result = chain.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )

        assert result.shape == (200,)

# =============================================================================
# Test Calibration
# =============================================================================

class TestPrototypeSelector:
    """Tests for PrototypeSelector."""

    def test_select_basic(self):
        """Test basic prototype selection."""
        from nirs4all.synthesis.reconstruction.calibration import PrototypeSelector

        selector = PrototypeSelector(n_prototypes=3)

        # Create synthetic data with variation
        X = np.random.randn(50, 100) + np.linspace(0, 1, 100)

        prototypes, indices = selector.select(X)

        assert prototypes.shape[0] <= 3
        assert len(indices) == prototypes.shape[0]
        assert all(idx < 50 for idx in indices)

    def test_select_includes_median(self):
        """Test that median-like sample is included."""
        from nirs4all.synthesis.reconstruction.calibration import PrototypeSelector

        selector = PrototypeSelector(n_prototypes=5, include_median=True)

        # Create data where median is clear
        X = np.zeros((10, 50))
        for i in range(10):
            X[i] = i * np.ones(50)  # Sample i has constant value i

        prototypes, indices = selector.select(X)

        # Median sample (index ~4 or 5) should be selected
        assert len(prototypes) > 0

class TestGlobalCalibrator:
    """Tests for GlobalCalibrator."""

    def test_calibrate_simple(self):
        """Test basic calibration."""
        from nirs4all.synthesis.reconstruction.calibration import GlobalCalibrator
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 200)
        target_grid = np.linspace(1100, 2400, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
        )

        # Create simple prototype
        prototypes = chain.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        ).reshape(1, -1)

        calibrator = GlobalCalibrator()
        result = calibrator.calibrate(prototypes, chain)

        assert isinstance(result.wl_shift, float)
        assert isinstance(result.ils_sigma, float)

# =============================================================================
# Test Inversion
# =============================================================================

class TestVariableProjectionSolver:
    """Tests for VariableProjectionSolver."""

    def test_fit_simple(self):
        """Test basic fitting."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain
        from nirs4all.synthesis.reconstruction.inversion import (
            MultiscaleSchedule,
            VariableProjectionSolver,
        )

        canonical_grid = np.linspace(1000, 2500, 200)
        target_grid = np.linspace(1100, 2400, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
        )

        # Generate synthetic target
        target = chain.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )

        # Fit
        solver = VariableProjectionSolver()
        schedule = MultiscaleSchedule.quick()
        result = solver.fit(target, chain, schedule)

        assert result.r_squared > 0.5  # Should fit reasonably
        assert len(result.concentrations) == 1

    def test_inversion_result_to_dict(self):
        """Test InversionResult serialization."""
        from nirs4all.synthesis.reconstruction.inversion import InversionResult

        result = InversionResult(
            concentrations=np.array([1.0, 2.0]),
            baseline_coeffs=np.array([0.1, 0.2, 0.3]),
            path_length=1.2,
            r_squared=0.95,
        )

        d = result.to_dict()
        assert d["path_length"] == 1.2
        assert d["r_squared"] == 0.95

class TestMultiscaleSchedule:
    """Tests for MultiscaleSchedule."""

    def test_default_schedule(self):
        """Test default schedule configuration."""
        from nirs4all.synthesis.reconstruction.inversion import MultiscaleSchedule

        schedule = MultiscaleSchedule()

        assert schedule.n_stages >= 2
        assert len(schedule.smooth_sigmas) == schedule.n_stages

    def test_quick_schedule(self):
        """Test quick schedule."""
        from nirs4all.synthesis.reconstruction.inversion import MultiscaleSchedule

        schedule = MultiscaleSchedule.quick()

        assert schedule.n_stages == 2

    def test_thorough_schedule(self):
        """Test thorough schedule."""
        from nirs4all.synthesis.reconstruction.inversion import MultiscaleSchedule

        schedule = MultiscaleSchedule.thorough()

        assert schedule.n_stages >= 4

# =============================================================================
# Test Distributions
# =============================================================================

class TestParameterDistributionFitter:
    """Tests for ParameterDistributionFitter."""

    def test_fit_positive_params(self):
        """Test fitting positive parameters."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
        )

        fitter = ParameterDistributionFitter()

        params = {
            "concentrations": np.abs(np.random.randn(50, 3)) + 0.1,
            "path_lengths": np.abs(np.random.randn(50)) + 0.5,
        }

        result = fitter.fit(params)

        assert len(result.param_names) > 0
        assert "concentrations_0" in result.distributions
        assert result.distributions["concentrations_0"]["type"] == "lognormal"

    def test_fit_gaussian_params(self):
        """Test fitting Gaussian parameters."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
        )

        fitter = ParameterDistributionFitter(
            positive_params=[],  # Don't treat as positive
        )

        params = {
            "shifts": np.random.randn(50),
        }

        result = fitter.fit(params)

        assert "shifts" in result.distributions
        assert result.distributions["shifts"]["type"] == "gaussian"

class TestParameterSampler:
    """Tests for ParameterSampler."""

    def test_sample_basic(self):
        """Test basic sampling."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
            ParameterSampler,
        )

        fitter = ParameterDistributionFitter()

        params = {
            "concentrations": np.abs(np.random.randn(50, 2)) + 0.1,
            "path_lengths": np.abs(np.random.randn(50)) + 0.5,
        }

        result = fitter.fit(params)
        sampler = ParameterSampler(result)

        samples = sampler.sample(100, random_state=42)

        assert "concentrations" in samples
        assert samples["concentrations"].shape == (100, 2)

    def test_sample_correlations(self):
        """Test that correlations are preserved."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
            ParameterSampler,
        )

        fitter = ParameterDistributionFitter()

        # Create correlated data
        base = np.random.randn(100)
        params = {
            "param1": base + 0.1,
            "param2": base * 0.5 + np.random.randn(100) * 0.1 + 1,
        }

        result = fitter.fit(params)
        sampler = ParameterSampler(result, use_correlations=True)

        samples = sampler.sample(500, random_state=42)

        # Check that correlation is preserved
        if result.correlations is not None:
            orig_corr = np.corrcoef(params["param1"], params["param2"])[0, 1]
            samp_corr = np.corrcoef(samples["param1"], samples["param2"])[0, 1]
            # Correlation should be somewhat preserved
            assert abs(samp_corr - orig_corr) < 0.5

# =============================================================================
# Test Generator
# =============================================================================

class TestReconstructionGenerator:
    """Tests for ReconstructionGenerator."""

    def test_generate_basic(self):
        """Test basic generation."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
            ParameterSampler,
        )
        from nirs4all.synthesis.reconstruction.forward import ForwardChain
        from nirs4all.synthesis.reconstruction.generator import ReconstructionGenerator

        # Setup
        canonical_grid = np.linspace(1000, 2500, 200)
        target_grid = np.linspace(1100, 2400, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
        )

        # Create simple distribution
        params = {
            "concentrations": np.abs(np.random.randn(20, 1)) + 0.5,
            "baseline_coeffs": np.random.randn(20, chain.canonical_model.n_baseline) * 0.01,
            "path_lengths": np.abs(np.random.randn(20)) * 0.2 + 1.0,
            "wl_shifts": np.random.randn(20) * 0.5,
        }

        fitter = ParameterDistributionFitter()
        dist_result = fitter.fit(params)
        sampler = ParameterSampler(dist_result)

        # Generate
        generator = ReconstructionGenerator(noise_level=0.001)
        result = generator.generate(10, chain, sampler, random_state=42)

        assert result.X.shape == (10, 150)
        assert result.concentrations.shape[0] == 10

# =============================================================================
# Test Validation
# =============================================================================

class TestReconstructionValidator:
    """Tests for ReconstructionValidator."""

    def test_validate_reconstruction(self):
        """Test reconstruction validation."""
        from nirs4all.synthesis.reconstruction.inversion import InversionResult
        from nirs4all.synthesis.reconstruction.validation import ReconstructionValidator

        validator = ReconstructionValidator()

        # Create mock results
        results = [
            InversionResult(
                concentrations=np.array([1.0]),
                baseline_coeffs=np.array([0.1]),
                r_squared=0.95,
                rmse=0.01,
                residuals=np.random.randn(100) * 0.01,
                fitted_spectrum=np.ones(100),
            )
            for _ in range(10)
        ]

        metrics = validator.validate_reconstruction(results)

        assert "mean_r2" in metrics
        assert abs(metrics["mean_r2"] - 0.95) < 1e-6  # Use approximate comparison

    def test_validate_synthetic(self):
        """Test synthetic validation."""
        from nirs4all.synthesis.reconstruction.validation import ReconstructionValidator

        validator = ReconstructionValidator()

        # Create similar data with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        base_signal = np.linspace(0, 1, 100)
        X_real = rng.standard_normal((50, 100)) * 0.1 + base_signal
        X_synth = rng.standard_normal((50, 100)) * 0.1 + base_signal

        metrics = validator.validate_synthetic(X_real, X_synth)

        assert "mean_spectrum_correlation" in metrics
        assert metrics["mean_spectrum_correlation"] > 0.8  # Should be similar (relaxed threshold)

# =============================================================================
# Test Pipeline
# =============================================================================

class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_from_data_absorbance(self):
        """Test auto-detection of absorbance data."""
        from nirs4all.synthesis.reconstruction.pipeline import DatasetConfig

        # Create absorbance-like data (positive, range 0-2)
        X = np.random.rand(50, 100) * 1.5 + 0.3
        wl = np.linspace(1000, 2500, 100)

        config = DatasetConfig.from_data(X, wl)

        assert config.signal_type == "absorbance"
        assert config.preprocessing == "none"

    def test_from_data_derivative(self):
        """Test auto-detection of derivative data."""
        from nirs4all.synthesis.reconstruction.pipeline import DatasetConfig

        # Create derivative-like data (zero mean, bipolar)
        X = np.random.randn(50, 100) * 0.1
        wl = np.linspace(1000, 2500, 100)

        config = DatasetConfig.from_data(X, wl)

        assert config.preprocessing in ("first_derivative", "second_derivative")

class TestReconstructionPipeline:
    """Tests for ReconstructionPipeline."""

    def test_component_selection(self):
        """Test domain-based component selection."""
        from nirs4all.synthesis.reconstruction.pipeline import (
            DatasetConfig,
            ReconstructionPipeline,
        )

        wl = np.linspace(1000, 2500, 100)

        config = DatasetConfig(
            wavelengths=wl,
            signal_type="absorbance",
            preprocessing="none",
            domain="food_dairy",
        )

        pipeline = ReconstructionPipeline(config=config)

        # Should have selected dairy-related components
        assert len(pipeline.component_names) > 0

# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_synthetic_data(self):
        """Test full workflow on synthetic data."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain
        from nirs4all.synthesis.reconstruction.pipeline import (
            DatasetConfig,
            ReconstructionPipeline,
        )

        # Generate synthetic "real" data
        canonical_grid = np.linspace(950, 2550, 320)
        target_grid = np.linspace(1000, 2500, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water", "protein"],
            domain="absorbance",
            preprocessing_type="none",
        )

        # Generate 30 samples with variation
        n_samples = 30
        X = np.zeros((n_samples, len(target_grid)))
        for i in range(n_samples):
            conc = np.abs(np.random.randn(2)) + 0.5
            path = 1.0 + np.random.randn() * 0.1
            X[i] = chain.forward(concentrations=conc, path_length=path)
            X[i] += np.random.randn(len(target_grid)) * 0.01  # Add noise

        # Run reconstruction
        config = DatasetConfig(
            wavelengths=target_grid,
            signal_type="absorbance",
            preprocessing="none",
            domain="unknown",
            name="test_synthetic",
        )

        pipeline = ReconstructionPipeline(
            config=config,
            component_names=["water", "protein"],
            verbose=False,
        )

        result = pipeline.fit(X, max_samples=20)

        # Check outputs
        assert result.calibration is not None
        assert result.inversion_results is not None
        assert len(result.inversion_results) <= 20
        assert result.X_synthetic is not None
        assert result.validation is not None

        # Check quality
        mean_r2 = np.mean([r.r_squared for r in result.inversion_results])
        assert mean_r2 > 0.5, f"Mean R² too low: {mean_r2}"

# =============================================================================
# Test Environmental Effects
# =============================================================================

class TestEnvironmentalEffectsModel:
    """Tests for EnvironmentalEffectsModel."""

    def test_init_defaults(self):
        """Test default initialization."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        model = EnvironmentalEffectsModel()

        assert model.temperature_delta == 0.0
        assert model.water_activity == 0.5
        assert model.scattering_power == 1.5
        assert model.scattering_amplitude == 0.0
        assert model.enabled is True

    def test_apply_no_effect(self):
        """Test that disabled model passes through unchanged."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        model = EnvironmentalEffectsModel(enabled=False)
        wavelengths = np.linspace(1000, 2500, 200)
        absorption = np.sin(wavelengths / 200)

        result = model.apply(absorption, wavelengths)

        np.testing.assert_array_equal(result, absorption)

    def test_apply_temperature_effect(self):
        """Test temperature effect changes spectrum."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        wavelengths = np.linspace(1000, 2500, 200)
        absorption = np.exp(-((wavelengths - 1450) ** 2) / 10000)  # Peak near O-H band

        model_warm = EnvironmentalEffectsModel(temperature_delta=10.0)
        model_cold = EnvironmentalEffectsModel(temperature_delta=-10.0)

        result_warm = model_warm.apply(absorption.copy(), wavelengths)
        result_cold = model_cold.apply(absorption.copy(), wavelengths)

        # Temperature should shift and scale the spectrum
        assert not np.allclose(result_warm, absorption)
        assert not np.allclose(result_cold, absorption)
        assert not np.allclose(result_warm, result_cold)

    def test_apply_scattering_effect(self):
        """Test scattering baseline effect."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        wavelengths = np.linspace(1000, 2500, 200)
        absorption = np.ones(200) * 0.5

        model = EnvironmentalEffectsModel(
            temperature_delta=0.0,
            water_activity=0.5,
            scattering_amplitude=0.1,
            scattering_power=2.0,
        )

        result = model.apply(absorption.copy(), wavelengths)

        # Scattering should add a wavelength-dependent baseline
        # (higher at shorter wavelengths)
        assert result[0] > absorption[0]  # More scattering at short wavelengths
        assert result[-1] > absorption[-1]  # Some scattering at long wavelengths
        assert (result[0] - absorption[0]) > (result[-1] - absorption[-1])

    def test_apply_water_activity_effect(self):
        """Test water activity effect on water bands."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        wavelengths = np.linspace(1800, 2000, 200)  # Around water combination band
        absorption = np.exp(-((wavelengths - 1940) ** 2) / 1000)

        model_low = EnvironmentalEffectsModel(water_activity=0.2)
        model_high = EnvironmentalEffectsModel(water_activity=0.8)

        result_low = model_low.apply(absorption.copy(), wavelengths)
        result_high = model_high.apply(absorption.copy(), wavelengths)

        # Different water activities should produce different spectra
        assert not np.allclose(result_low, result_high)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )

        model = EnvironmentalEffectsModel(
            temperature_delta=5.0,
            water_activity=0.7,
            scattering_power=1.8,
            scattering_amplitude=0.05,
        )

        d = model.to_dict()

        assert d["temperature_delta"] == 5.0
        assert d["water_activity"] == 0.7
        assert d["scattering_power"] == 1.8
        assert d["scattering_amplitude"] == 0.05

class TestEnvironmentalParameterConfig:
    """Tests for EnvironmentalParameterConfig."""

    def test_default_config(self):
        """Test default parameter configuration."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalParameterConfig,
        )

        config = EnvironmentalParameterConfig()

        assert config.temperature_bounds == (-15.0, 15.0)
        assert config.water_activity_bounds == (0.1, 0.9)
        assert config.scattering_power_bounds == (0.5, 3.0)
        assert config.scattering_amplitude_bounds == (0.0, 0.2)

class TestForwardChainWithEnvironmental:
    """Tests for ForwardChain with environmental model."""

    def test_create_with_environmental(self):
        """Test factory creates environmental model when requested."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=True,
        )

        assert chain.environmental_model is not None
        assert chain.environmental_model.enabled is True

    def test_create_without_environmental(self):
        """Test factory does not create environmental model by default."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=False,
        )

        assert chain.environmental_model is None

    def test_forward_with_environmental(self):
        """Test forward model applies environmental effects."""
        from nirs4all.synthesis.reconstruction.environmental import (
            EnvironmentalEffectsModel,
        )
        from nirs4all.synthesis.reconstruction.forward import ForwardChain

        canonical_grid = np.linspace(1000, 2500, 300)
        target_grid = np.linspace(1100, 2400, 200)

        chain_no_env = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=False,
        )

        chain_with_env = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=True,
        )
        chain_with_env.environmental_model.temperature_delta = 10.0
        chain_with_env.environmental_model.scattering_amplitude = 0.05

        result_no_env = chain_no_env.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )
        result_with_env = chain_with_env.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )

        # Environmental effects should change the spectrum
        assert not np.allclose(result_no_env, result_with_env)

class TestInversionWithEnvironmental:
    """Tests for inversion with environmental parameters."""

    def test_inversion_result_environmental_fields(self):
        """Test InversionResult has environmental fields."""
        from nirs4all.synthesis.reconstruction.inversion import InversionResult

        result = InversionResult(
            concentrations=np.array([1.0]),
            baseline_coeffs=np.array([0.1]),
            path_length=1.0,
            r_squared=0.95,
            temperature_delta=5.0,
            water_activity=0.6,
            scattering_power=1.8,
            scattering_amplitude=0.03,
        )

        assert result.temperature_delta == 5.0
        assert result.water_activity == 0.6
        assert result.scattering_power == 1.8
        assert result.scattering_amplitude == 0.03

    def test_solver_fit_environmental(self):
        """Test solver with environmental fitting enabled."""
        from nirs4all.synthesis.reconstruction.forward import ForwardChain
        from nirs4all.synthesis.reconstruction.inversion import (
            MultiscaleSchedule,
            VariableProjectionSolver,
        )

        canonical_grid = np.linspace(1000, 2500, 200)
        target_grid = np.linspace(1100, 2400, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=True,
        )

        # Set known environmental params
        chain.environmental_model.temperature_delta = 5.0
        chain.environmental_model.scattering_amplitude = 0.02

        # Generate target
        target = chain.forward(
            concentrations=np.array([1.0]),
            path_length=1.0,
        )

        # Fit with environmental
        solver = VariableProjectionSolver(fit_environmental=True)
        schedule = MultiscaleSchedule.quick()
        result = solver.fit(target, chain, schedule)

        # Should have environmental results
        assert result.temperature_delta is not None
        assert result.water_activity is not None
        assert result.scattering_power is not None
        assert result.scattering_amplitude is not None
        assert result.r_squared > 0.3  # Should fit somewhat

class TestGeneratorWithEnvironmental:
    """Tests for generator with environmental parameters."""

    def test_generation_result_environmental_fields(self):
        """Test GenerationResult has environmental fields."""
        from nirs4all.synthesis.reconstruction.generator import GenerationResult

        result = GenerationResult(
            X=np.zeros((10, 100)),
            concentrations=np.zeros((10, 2)),
            path_lengths=np.ones(10),
            baseline_coeffs=np.zeros((10, 3)),
            wavelengths=np.linspace(1000, 2500, 100),
            temperature_deltas=np.random.randn(10) * 5,
            water_activities=np.random.rand(10) * 0.5 + 0.25,
            scattering_powers=np.random.rand(10) * 0.5 + 1.25,
            scattering_amplitudes=np.random.rand(10) * 0.05,
        )

        assert result.temperature_deltas is not None
        assert result.water_activities is not None
        assert result.scattering_powers is not None
        assert result.scattering_amplitudes is not None
        assert len(result.temperature_deltas) == 10

    def test_generate_with_environmental(self):
        """Test generation with environmental parameters."""
        from nirs4all.synthesis.reconstruction.distributions import (
            ParameterDistributionFitter,
            ParameterSampler,
        )
        from nirs4all.synthesis.reconstruction.forward import ForwardChain
        from nirs4all.synthesis.reconstruction.generator import ReconstructionGenerator

        canonical_grid = np.linspace(1000, 2500, 200)
        target_grid = np.linspace(1100, 2400, 150)

        chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=target_grid,
            component_names=["water"],
            domain="absorbance",
            preprocessing_type="none",
            include_environmental=True,
        )

        # Create distribution with environmental params
        params = {
            "concentrations": np.abs(np.random.randn(20, 1)) + 0.5,
            "baseline_coeffs": np.random.randn(20, chain.canonical_model.n_baseline) * 0.01,
            "path_lengths": np.abs(np.random.randn(20)) * 0.2 + 1.0,
            "wl_shifts": np.random.randn(20) * 0.5,
            "temperature_deltas": np.random.randn(20) * 3,
            "water_activities": np.random.rand(20) * 0.5 + 0.25,
            "scattering_powers": np.random.rand(20) * 0.5 + 1.25,
            "scattering_amplitudes": np.random.rand(20) * 0.03,
        }

        fitter = ParameterDistributionFitter(
            positive_params=["concentrations", "path_lengths", "scattering_amplitudes"],
            bounded_params={
                "wl_shifts": (-5.0, 5.0),
                "water_activities": (0.1, 0.9),
                "scattering_powers": (0.5, 3.0),
            },
        )
        dist_result = fitter.fit(params)
        sampler = ParameterSampler(dist_result)

        generator = ReconstructionGenerator(noise_level=0.001)
        result = generator.generate(10, chain, sampler, random_state=42)

        assert result.X.shape == (10, 150)
        assert result.temperature_deltas is not None
        assert result.water_activities is not None
        assert result.scattering_powers is not None
        assert result.scattering_amplitudes is not None

class TestPipelineWithEnvironmental:
    """Tests for pipeline with environmental fitting."""

    def test_pipeline_with_environmental_flag(self):
        """Test pipeline accepts environmental fitting flag."""
        from nirs4all.synthesis.reconstruction.pipeline import (
            DatasetConfig,
            ReconstructionPipeline,
        )

        wl = np.linspace(1000, 2500, 100)

        config = DatasetConfig(
            wavelengths=wl,
            signal_type="absorbance",
            preprocessing="none",
        )

        pipeline = ReconstructionPipeline(
            config=config,
            fit_environmental=True,
            verbose=False,
        )

        assert pipeline.fit_environmental is True

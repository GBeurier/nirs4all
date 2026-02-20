"""
Complete reconstruction pipeline for end-to-end workflow.

Provides a unified interface for:
1. Dataset configuration and preprocessing detection
2. Global calibration
3. Batch inversion
4. Parameter distribution learning
5. Synthetic generation
6. Validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .calibration import CalibrationResult
    from .distributions import DistributionResult
    from .forward import ForwardChain
    from .inversion import InversionResult
    from .validation import ValidationResult

# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Configuration for a dataset to be reconstructed.

    Captures all dataset-specific information needed for reconstruction:
    - Wavelength grid
    - Signal type (absorbance, reflectance)
    - Preprocessing applied
    - Application domain (for component selection)

    Attributes:
        wavelengths: Wavelength grid in nm.
        signal_type: Signal type ('absorbance', 'reflectance').
        preprocessing: Detected or specified preprocessing type.
        domain: Application domain for component selection.
        sg_window: Savitzky-Golay window (for derivatives).
        sg_polyorder: Savitzky-Golay polynomial order.
        name: Optional dataset name.
    """

    wavelengths: np.ndarray
    signal_type: Literal["absorbance", "reflectance", "unknown"] = "absorbance"
    preprocessing: Literal[
        "none", "first_derivative", "second_derivative", "snv", "msc", "unknown"
    ] = "none"
    domain: str = "unknown"
    sg_window: int = 15
    sg_polyorder: int = 2
    name: str = "dataset"

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        wavelengths: np.ndarray,
        name: str = "dataset",
    ) -> DatasetConfig:
        """
        Create configuration by auto-detecting properties from data.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths).
            wavelengths: Wavelength grid.
            name: Dataset name.

        Returns:
            DatasetConfig with detected properties.
        """
        # Detect signal type
        signal_type = cls._detect_signal_type(X, wavelengths)

        # Detect preprocessing
        preprocessing = cls._detect_preprocessing(X)

        return cls(
            wavelengths=wavelengths.copy(),
            signal_type=signal_type,
            preprocessing=preprocessing,
            name=name,
        )

    @staticmethod
    def _detect_signal_type(
        X: np.ndarray, wavelengths: np.ndarray
    ) -> Literal["absorbance", "reflectance", "unknown"]:
        """Detect signal type from data characteristics."""
        mean_val = X.mean()
        max_val = X.max()
        min_val = X.min()

        # Derivative detection
        if min_val < -0.5 or (min_val < 0 and abs(mean_val) < 0.1):
            return "unknown"  # Derivative data

        # Reflectance: typically 0-1 or 0-100
        if min_val >= 0 and max_val <= 1.1:
            return "reflectance"
        if min_val >= 0 and 20 < mean_val < 80:
            return "reflectance"  # Percent reflectance

        # Absorbance: positive, typical range 0-3
        if min_val >= 0 and 0.1 < mean_val < 3.0:
            return "absorbance"

        return "unknown"

    @staticmethod
    def _detect_preprocessing(
        X: np.ndarray,
    ) -> Literal["none", "first_derivative", "second_derivative", "snv", "msc", "unknown"]:
        """Detect preprocessing type from data characteristics."""
        mean_val = float(np.mean(X))
        min_val = float(np.min(X))
        max_val = float(np.max(X))
        global_range = max_val - min_val

        # Zero-crossing ratio
        zero_crossings = np.sum(np.diff(np.sign(X), axis=1) != 0)
        total_transitions = (X.shape[0] * (X.shape[1] - 1))
        zero_crossing_ratio = zero_crossings / max(total_transitions, 1)

        # Second derivative: very small range, zero mean, high oscillation
        if global_range < 0.3 and abs(mean_val) < 0.05 and zero_crossing_ratio > 0.15:
            return "second_derivative"

        # First derivative: bipolar, near-zero mean
        if min_val < -0.001 and max_val > 0.001 and abs(mean_val) < 0.1 and (global_range < 1.0 or zero_crossing_ratio > 0.05):
            return "first_derivative"

        # SNV: per-sample std ~1
        sample_stds = X.std(axis=1)
        if 0.8 < np.mean(sample_stds) < 1.2 and np.std(sample_stds) < 0.2:
            sample_means = X.mean(axis=1)
            if abs(np.mean(sample_means)) < 0.1:
                return "snv"

        # Raw data
        if min_val >= 0 and mean_val > 0.1:
            return "none"

        return "unknown"

# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class PipelineResult:
    """
    Result of reconstruction pipeline.

    Contains all outputs from the reconstruction workflow:
    - Calibration results
    - Inversion results
    - Learned distributions
    - Generated synthetic data
    - Validation metrics

    Attributes:
        config: Dataset configuration used.
        calibration: Global calibration result.
        inversion_results: Per-sample inversion results.
        distribution: Learned parameter distributions.
        X_synthetic: Generated synthetic spectra.
        validation: Validation result.
        forward_chain: Calibrated forward chain.
    """

    config: DatasetConfig
    calibration: CalibrationResult | None = None
    inversion_results: list[InversionResult] | None = None
    distribution: DistributionResult | None = None
    X_synthetic: np.ndarray | None = None
    validation: ValidationResult | None = None
    forward_chain: ForwardChain | None = None

    def summary(self) -> str:
        """Generate pipeline summary."""
        lines = [
            "=" * 70,
            f"Reconstruction Pipeline Result: {self.config.name}",
            "=" * 70,
            "",
            "Dataset Configuration:",
            f"  Signal type: {self.config.signal_type}",
            f"  Preprocessing: {self.config.preprocessing}",
            f"  Wavelengths: {len(self.config.wavelengths)} points",
            f"  Range: {self.config.wavelengths.min():.0f} - {self.config.wavelengths.max():.0f} nm",
        ]

        if self.calibration:
            lines.extend([
                "",
                "Global Calibration:",
                f"  Wavelength shift: {self.calibration.wl_shift:.2f} nm",
                f"  ILS sigma: {self.calibration.ils_sigma:.2f} nm",
                f"  Total loss: {self.calibration.total_loss:.4f}",
            ])

        if self.inversion_results:
            r2_values = [r.r_squared for r in self.inversion_results]
            lines.extend([
                "",
                "Inversion Results:",
                f"  Samples fitted: {len(self.inversion_results)}",
                f"  Mean R²: {np.mean(r2_values):.4f}",
                f"  Min R²: {np.min(r2_values):.4f}",
            ])

        if self.validation:
            lines.extend([
                "",
                "Validation:",
                f"  Overall score: {self.validation.overall_score:.1f}/100",
                f"  Status: {'PASSED' if self.validation.passed else 'NEEDS REVIEW'}",
            ])
            if self.validation.warnings:
                for w in self.validation.warnings:
                    lines.append(f"    Warning: {w}")

        lines.append("=" * 70)
        return "\n".join(lines)

# =============================================================================
# Reconstruction Pipeline
# =============================================================================

@dataclass
class ReconstructionPipeline:
    """
    Complete reconstruction pipeline.

    Orchestrates the full workflow:
    1. Configuration and component selection
    2. Prototype selection and global calibration
    3. Per-sample inversion (optionally with environmental parameters)
    4. Parameter distribution learning
    5. Synthetic generation
    6. Validation

    Attributes:
        config: Dataset configuration.
        component_names: Components to use (auto-selected if None).
        canonical_resolution: Resolution of canonical grid (nm).
        baseline_order: Baseline polynomial order.
        n_prototypes: Number of prototypes for calibration.
        fit_environmental: Whether to fit environmental parameters.
        verbose: Print progress.
    """

    config: DatasetConfig
    component_names: list[str] | None = None
    canonical_resolution: float = 0.5
    baseline_order: int = 5
    continuum_order: int = 3
    n_prototypes: int = 5
    fit_environmental: bool = False
    verbose: bool = True

    def __post_init__(self):
        """Initialize components if not provided."""
        if self.component_names is None:
            self.component_names = self._select_components_for_domain()

    def _select_components_for_domain(self) -> list[str]:
        """Select appropriate components based on domain."""
        # Default components that appear in most NIR datasets
        default_components = [
            "water", "protein", "lipid", "starch", "cellulose",
        ]

        # Domain-specific additions
        domain_components = {
            "food_dairy": ["casein", "lactose", "whey", "lipid"],
            "food_bakery": ["starch", "gluten", "lipid", "glucose"],
            "agriculture_grain": ["starch", "protein", "cellulose", "moisture"],
            "agriculture_fruit": ["fructose", "glucose", "cellulose", "water"],
            "environmental_soil": ["humic_acid", "cellulose", "clay_minerals"],
            "pharma_tablets": ["lactose", "cellulose", "starch"],
            "petrochem_fuels": ["paraffin", "aromatic_hydrocarbons"],
            "beverage_wine": ["ethanol", "glucose", "water"],
        }

        domain = self.config.domain
        components = domain_components.get(domain, default_components)

        # Filter to available components
        from ..components import available_components

        available = set(available_components())
        return [c for c in components if c in available][:10]

    def fit(
        self,
        X: np.ndarray,
        max_samples: int | None = None,
    ) -> PipelineResult:
        """
        Run full reconstruction pipeline.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths).
            max_samples: Max samples to invert (for speed).

        Returns:
            PipelineResult with all outputs.
        """
        from .calibration import GlobalCalibrator, PrototypeSelector, multistage_calibration
        from .distributions import ParameterDistributionFitter, ParameterSampler
        from .forward import ForwardChain
        from .generator import ReconstructionGenerator, estimate_noise_from_residuals
        from .inversion import MultiscaleSchedule, VariableProjectionSolver
        from .validation import ReconstructionValidator

        n_samples = X.shape[0]

        if self.verbose:
            print(f"Starting reconstruction pipeline for {self.config.name}")
            print(f"  Samples: {n_samples}, Wavelengths: {X.shape[1]}")
            print(f"  Signal type: {self.config.signal_type}")
            print(f"  Preprocessing: {self.config.preprocessing}")
            print(f"  Components: {self.component_names}")
            print(f"  Environmental fitting: {self.fit_environmental}")

        # 1. Create canonical grid
        wl_min = self.config.wavelengths.min() - 50
        wl_max = self.config.wavelengths.max() + 50
        canonical_grid = np.arange(wl_min, wl_max, self.canonical_resolution)

        # 2. Create forward chain
        assert self.component_names is not None
        forward_chain = ForwardChain.create(
            canonical_grid=canonical_grid,
            target_grid=self.config.wavelengths,
            component_names=self.component_names,
            domain=self.config.signal_type if self.config.signal_type != "unknown" else "absorbance",
            preprocessing_type=self.config.preprocessing if self.config.preprocessing != "unknown" else "none",
            baseline_order=self.baseline_order,
            continuum_order=self.continuum_order,
            sg_window=self.config.sg_window,
            sg_polyorder=self.config.sg_polyorder,
            include_environmental=self.fit_environmental,
        )

        if self.verbose:
            print("\n1. Global Calibration...")

        # 3. Global calibration
        calibration = multistage_calibration(
            X, forward_chain, n_prototypes=self.n_prototypes
        )

        if self.verbose:
            print(f"   Wavelength shift: {calibration.wl_shift:.2f} nm")
            print(f"   ILS sigma: {calibration.ils_sigma:.2f} nm")
            if calibration.prototype_r2 is not None:
                print(f"   Prototype R²: {np.mean(calibration.prototype_r2):.4f}")

        # 4. Per-sample inversion
        if self.verbose:
            print("\n2. Per-sample Inversion...")

        if max_samples is not None and n_samples > max_samples:
            # Subsample for speed
            idx = np.random.choice(n_samples, max_samples, replace=False)
            X_invert = X[idx]
        else:
            X_invert = X

        solver = VariableProjectionSolver(
            verbose=False,
            fit_environmental=self.fit_environmental,
        )
        schedule = MultiscaleSchedule.quick() if n_samples > 100 else MultiscaleSchedule()

        inversion_results = solver.fit_batch(X_invert, forward_chain, schedule)

        r2_values = [r.r_squared for r in inversion_results]
        if self.verbose:
            print(f"   Fitted {len(inversion_results)} samples")
            print(f"   Mean R²: {np.mean(r2_values):.4f}")
            print(f"   Min R²: {np.min(r2_values):.4f}")

        # 5. Learn parameter distributions
        if self.verbose:
            print("\n3. Learning Parameter Distributions...")

        params = {
            "concentrations": np.array([r.concentrations for r in inversion_results]),
            "baseline_coeffs": np.array([r.baseline_coeffs for r in inversion_results]),
            "path_lengths": np.array([r.path_length for r in inversion_results]),
            "wl_shifts": np.array([r.wl_shift_residual for r in inversion_results]),
        }

        # Add environmental parameters if fitted
        if self.fit_environmental:
            params["temperature_deltas"] = np.array(
                [r.temperature_delta for r in inversion_results]
            )
            params["water_activities"] = np.array(
                [r.water_activity for r in inversion_results]
            )
            params["scattering_powers"] = np.array(
                [r.scattering_power for r in inversion_results]
            )
            params["scattering_amplitudes"] = np.array(
                [r.scattering_amplitude for r in inversion_results]
            )

        # Configure distribution fitter for environmental params
        bounded_params = {"wl_shifts": (-5.0, 5.0)}
        positive_params = ["concentrations", "path_lengths"]

        if self.fit_environmental:
            bounded_params["water_activities"] = (0.0, 1.0)
            bounded_params["scattering_powers"] = (0.5, 3.0)
            positive_params.append("scattering_amplitudes")

        dist_fitter = ParameterDistributionFitter(
            positive_params=positive_params,
            bounded_params=bounded_params,
        )
        distribution = dist_fitter.fit(params)

        if self.verbose:
            print(f"   Fitted distributions for {len(distribution.param_names)} parameters")

        # 6. Generate synthetic data
        if self.verbose:
            print("\n4. Generating Synthetic Data...")

        sampler = ParameterSampler(distribution, use_correlations=True)

        # Estimate noise from residuals
        noise_add, noise_mult = estimate_noise_from_residuals(inversion_results)

        generator = ReconstructionGenerator(
            noise_level=noise_add,
            multiplicative_noise=noise_mult,
            add_noise=True,
        )

        gen_result = generator.generate(
            n_samples=len(X_invert),
            forward_chain=forward_chain,
            sampler=sampler,
            random_state=42,
        )
        X_synthetic = gen_result.X

        if self.verbose:
            print(f"   Generated {len(X_synthetic)} synthetic samples")

        # 7. Validation
        if self.verbose:
            print("\n5. Validation...")

        validator = ReconstructionValidator()
        validation = validator.validate(inversion_results, X_invert, X_synthetic)

        if self.verbose:
            print(f"   Overall score: {validation.overall_score:.1f}/100")
            print(f"   Status: {'PASSED' if validation.passed else 'NEEDS REVIEW'}")

        return PipelineResult(
            config=self.config,
            calibration=calibration,
            inversion_results=inversion_results,
            distribution=distribution,
            X_synthetic=X_synthetic,
            validation=validation,
            forward_chain=forward_chain,
        )

    def generate(
        self,
        n_samples: int,
        result: PipelineResult,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Generate additional synthetic samples using fitted pipeline.

        Args:
            n_samples: Number of samples to generate.
            result: PipelineResult from fit().
            random_state: Random seed.

        Returns:
            Synthetic spectra matrix.
        """
        from .distributions import ParameterSampler
        from .generator import ReconstructionGenerator, estimate_noise_from_residuals

        if result.distribution is None or result.forward_chain is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        sampler = ParameterSampler(result.distribution, use_correlations=True)

        if result.inversion_results is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        noise_add, noise_mult = estimate_noise_from_residuals(result.inversion_results)

        generator = ReconstructionGenerator(
            noise_level=noise_add,
            multiplicative_noise=noise_mult,
        )

        gen_result = generator.generate(
            n_samples=n_samples,
            forward_chain=result.forward_chain,
            sampler=sampler,
            random_state=random_state,
        )

        return gen_result.X

# =============================================================================
# Convenience Functions
# =============================================================================

def reconstruct_and_generate(
    X: np.ndarray,
    wavelengths: np.ndarray,
    n_synthetic: int | None = None,
    domain: str = "unknown",
    component_names: list[str] | None = None,
    fit_environmental: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, PipelineResult]:
    """
    Convenience function for end-to-end reconstruction and generation.

    Args:
        X: Real spectra matrix.
        wavelengths: Wavelength grid.
        n_synthetic: Number of synthetic samples (default: same as X).
        domain: Application domain.
        component_names: Components to use.
        fit_environmental: Whether to fit environmental parameters
            (temperature, water activity, scattering).
        verbose: Print progress.

    Returns:
        Tuple of (X_synthetic, PipelineResult).
    """
    # Create configuration
    config = DatasetConfig.from_data(X, wavelengths)
    config.domain = domain

    # Create and run pipeline
    pipeline = ReconstructionPipeline(
        config=config,
        component_names=component_names,
        fit_environmental=fit_environmental,
        verbose=verbose,
    )

    result = pipeline.fit(X)

    # Generate additional samples if requested
    X_synth = pipeline.generate(n_synthetic, result) if n_synthetic is not None and n_synthetic != len(X) else result.X_synthetic
    assert X_synth is not None

    return X_synth, result

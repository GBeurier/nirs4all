"""
Synthetic data generation from learned parameter distributions.

Generates realistic synthetic spectra by:
1. Sampling physical parameters from learned distributions
2. Running the forward model chain
3. Adding appropriate noise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Generation Result
# =============================================================================


@dataclass
class GenerationResult:
    """
    Result of synthetic generation.

    Attributes:
        X: Generated spectra (n_samples, n_wavelengths).
        concentrations: Sampled concentrations (n_samples, n_components).
        path_lengths: Sampled path lengths (n_samples,).
        baseline_coeffs: Sampled baseline coefficients.
        wavelengths: Wavelength grid.
        noise_level: Applied noise level.
        wl_shifts: Per-sample wavelength shifts.
        temperature_deltas: Per-sample temperature deviations (°C).
        water_activities: Per-sample water activity values.
        scattering_powers: Per-sample scattering exponents.
        scattering_amplitudes: Per-sample scattering amplitudes.
    """

    X: np.ndarray
    concentrations: np.ndarray
    path_lengths: np.ndarray
    baseline_coeffs: np.ndarray
    wavelengths: np.ndarray
    noise_level: float = 0.0
    wl_shifts: Optional[np.ndarray] = None
    # Environmental parameters
    temperature_deltas: Optional[np.ndarray] = None
    water_activities: Optional[np.ndarray] = None
    scattering_powers: Optional[np.ndarray] = None
    scattering_amplitudes: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        """Number of generated samples."""
        return self.X.shape[0]

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return self.X.shape[1]


# =============================================================================
# Reconstruction Generator
# =============================================================================


@dataclass
class ReconstructionGenerator:
    """
    Generate synthetic spectra from learned parameter distributions.

    Uses the calibrated forward chain and learned parameter distributions
    to generate realistic synthetic data that matches the statistical
    properties of the original dataset.

    Attributes:
        forward_chain: Calibrated forward chain.
        sampler: Parameter sampler with learned distributions.
        noise_estimator: Estimated noise level from inversion residuals.
        add_noise: Whether to add noise to generated spectra.
        noise_type: Type of noise ('additive', 'multiplicative', 'both').
    """

    noise_level: float = 0.001
    multiplicative_noise: float = 0.01
    add_noise: bool = True
    noise_type: str = "both"

    def generate(
        self,
        n_samples: int,
        forward_chain: "ForwardChain",
        sampler: "ParameterSampler",
        random_state: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate synthetic spectra.

        Args:
            n_samples: Number of samples to generate.
            forward_chain: Calibrated forward chain.
            sampler: Parameter sampler.
            random_state: Random seed.

        Returns:
            GenerationResult with generated spectra and parameters.
        """
        rng = np.random.default_rng(random_state)

        # Sample parameters
        params = sampler.sample(n_samples, random_state)

        concentrations = params.get("concentrations", np.zeros((n_samples, 1)))
        if concentrations.ndim == 1:
            concentrations = concentrations.reshape(-1, 1)

        baseline_coeffs = params.get("baseline_coeffs", None)
        if baseline_coeffs is not None and baseline_coeffs.ndim == 1:
            baseline_coeffs = baseline_coeffs.reshape(-1, 1)

        path_lengths = params.get("path_lengths", np.ones(n_samples))
        wl_shifts = params.get("wl_shifts", np.zeros(n_samples))

        # Environmental parameters (if present in sampler)
        temperature_deltas = params.get("temperature_deltas", np.zeros(n_samples))
        water_activities = params.get("water_activities", np.full(n_samples, 0.5))
        scattering_powers = params.get("scattering_powers", np.full(n_samples, 1.5))
        scattering_amplitudes = params.get("scattering_amplitudes", np.zeros(n_samples))

        # Generate spectra
        n_wl = len(forward_chain.instrument_model.target_grid)
        X = np.zeros((n_samples, n_wl))

        # Store original state
        orig_wl_shift = forward_chain.instrument_model.wl_shift
        orig_env_state = None
        if forward_chain.environmental_model is not None:
            orig_env_state = forward_chain.environmental_model.to_dict()

        for i in range(n_samples):
            # Apply per-sample wavelength shift
            forward_chain.instrument_model.wl_shift = orig_wl_shift + wl_shifts[i]

            # Apply per-sample environmental parameters
            if forward_chain.environmental_model is not None:
                forward_chain.environmental_model.temperature_delta = temperature_deltas[i]
                forward_chain.environmental_model.water_activity = water_activities[i]
                forward_chain.environmental_model.scattering_power = scattering_powers[i]
                forward_chain.environmental_model.scattering_amplitude = scattering_amplitudes[i]

            # Get baseline coeffs for this sample
            if baseline_coeffs is not None:
                bl_coeffs = baseline_coeffs[i]
            else:
                bl_coeffs = None

            # Forward model
            spectrum = forward_chain.forward(
                concentrations=concentrations[i],
                path_length=path_lengths[i],
                baseline_coeffs=bl_coeffs,
            )

            X[i] = spectrum

        # Restore original state
        forward_chain.instrument_model.wl_shift = orig_wl_shift
        if orig_env_state is not None and forward_chain.environmental_model is not None:
            forward_chain.environmental_model.temperature_delta = orig_env_state["temperature_delta"]
            forward_chain.environmental_model.water_activity = orig_env_state["water_activity"]
            forward_chain.environmental_model.scattering_power = orig_env_state["scattering_power"]
            forward_chain.environmental_model.scattering_amplitude = orig_env_state["scattering_amplitude"]

        # Add noise
        if self.add_noise:
            X = self._add_noise(X, rng)

        return GenerationResult(
            X=X,
            concentrations=concentrations,
            path_lengths=path_lengths,
            baseline_coeffs=baseline_coeffs if baseline_coeffs is not None else np.zeros((n_samples, 1)),
            wavelengths=forward_chain.instrument_model.target_grid.copy(),
            noise_level=self.noise_level,
            wl_shifts=wl_shifts,
            temperature_deltas=temperature_deltas,
            water_activities=water_activities,
            scattering_powers=scattering_powers,
            scattering_amplitudes=scattering_amplitudes,
        )

    def _add_noise(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add noise to generated spectra."""
        X_noisy = X.copy()

        if self.noise_type in ("additive", "both"):
            # Additive Gaussian noise
            noise = rng.normal(0, self.noise_level, X.shape)
            X_noisy = X_noisy + noise

        if self.noise_type in ("multiplicative", "both"):
            # Multiplicative noise (gain variation)
            gain = 1.0 + rng.normal(0, self.multiplicative_noise, (X.shape[0], 1))
            X_noisy = X_noisy * gain

        return X_noisy

    def generate_matched(
        self,
        X_real: np.ndarray,
        forward_chain: "ForwardChain",
        sampler: "ParameterSampler",
        random_state: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate synthetic data matched to real data statistics.

        Generates same number of samples as real data and optionally
        adjusts noise level based on estimated residuals.

        Args:
            X_real: Real data matrix for reference.
            forward_chain: Calibrated forward chain.
            sampler: Parameter sampler.
            random_state: Random seed.

        Returns:
            GenerationResult.
        """
        n_samples = X_real.shape[0]
        return self.generate(n_samples, forward_chain, sampler, random_state)


# =============================================================================
# Noise Estimation
# =============================================================================


def estimate_noise_from_residuals(
    inversion_results: List["InversionResult"],
) -> Tuple[float, float]:
    """
    Estimate noise parameters from inversion residuals.

    Args:
        inversion_results: List of inversion results with residuals.

    Returns:
        Tuple of (additive_noise_std, multiplicative_noise_std).
    """
    residuals = []
    fitted_values = []

    for result in inversion_results:
        if result.residuals is not None and result.fitted_spectrum is not None:
            residuals.append(result.residuals)
            fitted_values.append(result.fitted_spectrum)

    if len(residuals) == 0:
        return 0.001, 0.01

    residuals = np.array(residuals)
    fitted_values = np.array(fitted_values)

    # Additive noise: std of residuals
    additive_std = float(np.std(residuals))

    # Multiplicative noise: check if residual std scales with signal
    # Fit: std(residual) ~ a * |signal| + b
    signal_means = np.abs(fitted_values).mean(axis=1)
    residual_stds = np.std(residuals, axis=1)

    if len(signal_means) > 10:
        try:
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(signal_means, residual_stds)
            multiplicative_std = max(slope, 0.001)
        except Exception:
            multiplicative_std = 0.01
    else:
        multiplicative_std = 0.01

    return additive_std, multiplicative_std


# =============================================================================
# Complete Generation Pipeline
# =============================================================================


def generate_synthetic_dataset(
    forward_chain: "ForwardChain",
    distribution_result: "DistributionResult",
    n_samples: int,
    noise_level: float = 0.001,
    multiplicative_noise: float = 0.01,
    random_state: Optional[int] = None,
) -> GenerationResult:
    """
    Complete pipeline to generate synthetic dataset.

    Args:
        forward_chain: Calibrated forward chain.
        distribution_result: Fitted parameter distributions.
        n_samples: Number of samples to generate.
        noise_level: Additive noise level.
        multiplicative_noise: Multiplicative noise level.
        random_state: Random seed.

    Returns:
        GenerationResult with synthetic spectra.
    """
    from .distributions import ParameterSampler

    # Create sampler
    sampler = ParameterSampler(distribution_result, use_correlations=True)

    # Create generator
    generator = ReconstructionGenerator(
        noise_level=noise_level,
        multiplicative_noise=multiplicative_noise,
        add_noise=True,
        noise_type="both",
    )

    # Generate
    return generator.generate(n_samples, forward_chain, sampler, random_state)

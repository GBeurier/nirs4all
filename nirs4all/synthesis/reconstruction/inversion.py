"""
Variable projection inversion for per-sample spectral fitting.

Implements the core inversion algorithm:
- Outer loop: optimize nonlinear parameters (path_length, per-sample shift)
- Inner loop: solve linear parameters via NNLS/QP (concentrations, baseline)

Uses multiscale schedule to avoid local minima.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Inversion Result
# =============================================================================


@dataclass
class InversionResult:
    """
    Result of per-sample inversion.

    Attributes:
        concentrations: Fitted component concentrations.
        baseline_coeffs: Fitted baseline coefficients.
        continuum_coeffs: Fitted continuum coefficients.
        path_length: Fitted path length factor.
        wl_shift_residual: Per-sample wavelength shift correction.
        scatter_coeffs: Fitted scatter coefficients (reflectance).
        fitted_spectrum: Reconstructed spectrum.
        residuals: Fitting residuals.
        r_squared: Coefficient of determination.
        rmse: Root mean squared error.
        converged: Whether optimization converged.
        temperature_delta: Fitted temperature deviation (°C from reference).
        water_activity: Fitted water activity (0-1 scale).
        scattering_power: Fitted scattering wavelength exponent.
        scattering_amplitude: Fitted scattering baseline amplitude.
    """

    concentrations: np.ndarray
    baseline_coeffs: np.ndarray
    continuum_coeffs: Optional[np.ndarray] = None
    path_length: float = 1.0
    wl_shift_residual: float = 0.0
    scatter_coeffs: Optional[np.ndarray] = None
    fitted_spectrum: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    r_squared: float = 0.0
    rmse: float = float("inf")
    converged: bool = False
    # Environmental parameters
    temperature_delta: float = 0.0
    water_activity: float = 0.5
    scattering_power: float = 1.5
    scattering_amplitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concentrations": self.concentrations.tolist(),
            "baseline_coeffs": self.baseline_coeffs.tolist(),
            "continuum_coeffs": (
                self.continuum_coeffs.tolist()
                if self.continuum_coeffs is not None
                else None
            ),
            "path_length": self.path_length,
            "wl_shift_residual": self.wl_shift_residual,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "converged": self.converged,
            "temperature_delta": self.temperature_delta,
            "water_activity": self.water_activity,
            "scattering_power": self.scattering_power,
            "scattering_amplitude": self.scattering_amplitude,
        }

    @property
    def linear_params(self) -> np.ndarray:
        """Get all linear parameters as single array."""
        params = [self.concentrations, self.baseline_coeffs]
        if self.continuum_coeffs is not None:
            params.append(self.continuum_coeffs)
        return np.concatenate(params)


# =============================================================================
# Multiscale Schedule
# =============================================================================


@dataclass
class MultiscaleSchedule:
    """
    Configuration for multiscale fitting curriculum.

    Fits coarse features first, then progressively adds detail:
    1. Smooth target + no derivatives + strong baseline prior
    2. Less smooth + partial derivative weight
    3. Full resolution + full preprocessing

    Attributes:
        smooth_sigmas: Gaussian sigma values for each stage (0 = no smoothing).
        derivative_weights: Weight on derivative space at each stage.
        baseline_regularization: Baseline regularization at each stage.
        max_iterations: Max iterations at each stage.
    """

    smooth_sigmas: List[float] = field(default_factory=lambda: [15, 8, 4, 0])
    derivative_weights: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0])
    baseline_regularization: List[float] = field(
        default_factory=lambda: [1e-3, 1e-4, 1e-5, 1e-6]
    )
    max_iterations: List[int] = field(default_factory=lambda: [50, 100, 100, 200])

    @property
    def n_stages(self) -> int:
        """Number of stages in schedule."""
        return len(self.smooth_sigmas)

    @classmethod
    def quick(cls) -> "MultiscaleSchedule":
        """Quick schedule for fast fitting."""
        return cls(
            smooth_sigmas=[10, 0],
            derivative_weights=[0.0, 1.0],
            baseline_regularization=[1e-4, 1e-6],
            max_iterations=[50, 100],
        )

    @classmethod
    def thorough(cls) -> "MultiscaleSchedule":
        """Thorough schedule for best accuracy."""
        return cls(
            smooth_sigmas=[20, 12, 6, 3, 0],
            derivative_weights=[0.0, 0.2, 0.5, 0.8, 1.0],
            baseline_regularization=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            max_iterations=[50, 100, 100, 150, 200],
        )


# =============================================================================
# Variable Projection Solver
# =============================================================================


@dataclass
class VariableProjectionSolver:
    """
    Variable projection solver for spectral inversion.

    Separates optimization into:
    - Nonlinear params: path_length, per-sample wl_shift, [environmental] (outer loop)
    - Linear params: concentrations, baseline, continuum (inner NNLS/QP)

    Attributes:
        path_length_bounds: Bounds for path length.
        wl_shift_bounds: Per-sample wavelength shift bounds.
        concentration_regularization: L2 reg on concentrations.
        baseline_smoothness_penalty: Penalty on baseline curvature.
        use_derivatives: Fit in derivative space (for derivative data).
        fit_environmental: Whether to fit environmental parameters.
        temperature_bounds: Bounds for temperature deviation (°C).
        water_activity_bounds: Bounds for water activity.
        scattering_power_bounds: Bounds for scattering exponent.
        scattering_amplitude_bounds: Bounds for scattering amplitude.
        environmental_prior_weight: Weight for environmental parameter priors.
    """

    path_length_bounds: Tuple[float, float] = (0.5, 2.0)
    wl_shift_bounds: Tuple[float, float] = (-2.0, 2.0)
    concentration_regularization: float = 1e-6
    baseline_smoothness_penalty: float = 1e-4
    use_derivatives: bool = False
    verbose: bool = False
    # Environmental fitting options
    fit_environmental: bool = False
    temperature_bounds: Tuple[float, float] = (-15.0, 15.0)
    water_activity_bounds: Tuple[float, float] = (0.1, 0.9)
    scattering_power_bounds: Tuple[float, float] = (0.5, 3.0)
    scattering_amplitude_bounds: Tuple[float, float] = (0.0, 0.2)
    environmental_prior_weight: float = 0.1

    def fit(
        self,
        target: np.ndarray,
        forward_chain: "ForwardChain",
        schedule: Optional[MultiscaleSchedule] = None,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> InversionResult:
        """
        Fit forward model to target spectrum.

        Args:
            target: Target spectrum to fit.
            forward_chain: Forward chain with calibrated global params.
            schedule: Multiscale fitting schedule.
            initial_params: Initial nonlinear parameters.

        Returns:
            InversionResult with fitted parameters.
        """
        if schedule is None:
            schedule = MultiscaleSchedule()

        # Initialize parameters
        if initial_params is None:
            path_length = 1.0
            wl_shift = 0.0
            temp_delta = 0.0
            water_act = 0.5
            scatter_power = 1.5
            scatter_amp = 0.0
        else:
            path_length = initial_params.get("path_length", 1.0)
            wl_shift = initial_params.get("wl_shift_residual", 0.0)
            temp_delta = initial_params.get("temperature_delta", 0.0)
            water_act = initial_params.get("water_activity", 0.5)
            scatter_power = initial_params.get("scattering_power", 1.5)
            scatter_amp = initial_params.get("scattering_amplitude", 0.0)

        # Build parameter vector
        if self.fit_environmental:
            current_params = np.array([
                path_length, wl_shift, temp_delta, water_act, scatter_power, scatter_amp
            ])
        else:
            current_params = np.array([path_length, wl_shift])

        # Run multiscale schedule
        for stage in range(schedule.n_stages):
            sigma = schedule.smooth_sigmas[stage]
            deriv_weight = schedule.derivative_weights[stage]
            baseline_reg = schedule.baseline_regularization[stage]
            max_iter = schedule.max_iterations[stage]

            # Smooth target for this stage
            if sigma > 0:
                target_stage = gaussian_filter1d(target, sigma=sigma)
            else:
                target_stage = target

            # Optimize nonlinear params
            result = self._optimize_stage(
                target_stage,
                forward_chain,
                current_params,
                deriv_weight=deriv_weight,
                baseline_reg=baseline_reg,
                max_iter=max_iter,
            )

            # Update current params
            if self.fit_environmental:
                current_params = np.array([
                    result.path_length, result.wl_shift_residual,
                    result.temperature_delta, result.water_activity,
                    result.scattering_power, result.scattering_amplitude
                ])
            else:
                current_params = np.array([result.path_length, result.wl_shift_residual])

            if self.verbose:
                msg = f"Stage {stage + 1}/{schedule.n_stages}: R²={result.r_squared:.4f}, path_length={result.path_length:.3f}"
                if self.fit_environmental:
                    msg += f", T={result.temperature_delta:.1f}°C, aw={result.water_activity:.2f}"
                print(msg)

        # Final fit with full resolution
        final_result = self._final_fit(target, forward_chain, current_params)

        return final_result

    def _optimize_stage(
        self,
        target: np.ndarray,
        forward_chain: "ForwardChain",
        initial_params: np.ndarray,
        deriv_weight: float = 1.0,
        baseline_reg: float = 1e-6,
        max_iter: int = 100,
    ) -> InversionResult:
        """Optimize nonlinear params for one stage."""
        # Store original instrument wl_shift and environmental state
        orig_wl_shift = forward_chain.instrument_model.wl_shift
        orig_env_state = None
        if forward_chain.environmental_model is not None:
            orig_env_state = forward_chain.environmental_model.to_dict()

        def objective(params: np.ndarray) -> float:
            path_length = params[0]
            wl_shift_delta = params[1]

            # Update instrument model with per-sample shift
            forward_chain.instrument_model.wl_shift = orig_wl_shift + wl_shift_delta

            # Update environmental model if fitting environmental
            if self.fit_environmental and forward_chain.environmental_model is not None:
                temp_delta = params[2]
                water_act = params[3]
                scatter_power = params[4]
                scatter_amp = params[5]

                forward_chain.environmental_model.temperature_delta = temp_delta
                forward_chain.environmental_model.water_activity = water_act
                forward_chain.environmental_model.scattering_power = scatter_power
                forward_chain.environmental_model.scattering_amplitude = scatter_amp

            # Get design matrix
            A = forward_chain.forward_design_matrix(path_length=path_length)

            # Solve inner linear problem
            try:
                linear_params, residual_norm = self._inner_solve(
                    A, target, forward_chain, baseline_reg
                )

                loss = residual_norm

                # Regularization on deviation from nominal
                loss += 0.01 * (path_length - 1.0) ** 2
                loss += 0.1 * wl_shift_delta ** 2

                # Environmental parameter priors
                if self.fit_environmental:
                    weight = self.environmental_prior_weight
                    # Gaussian prior on temperature (mean 0, std 5)
                    loss += weight * (params[2] / 5.0) ** 2
                    # Beta-like prior on water_activity (centered at 0.5)
                    aw_centered = params[3] - 0.5
                    loss += weight * 4 * aw_centered ** 2
                    # Gaussian prior on scattering power (mean 1.5, std 0.5)
                    loss += weight * ((params[4] - 1.5) / 0.5) ** 2
                    # Exponential prior on scattering amplitude (encourage small)
                    loss += weight * 10 * params[5]

                return loss
            except Exception:
                return 1e10

        # Build bounds
        bounds = [self.path_length_bounds, self.wl_shift_bounds]
        if self.fit_environmental:
            bounds.extend([
                self.temperature_bounds,
                self.water_activity_bounds,
                self.scattering_power_bounds,
                self.scattering_amplitude_bounds,
            ])

        result = minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )

        # Extract final parameters
        path_length = result.x[0]
        wl_shift_delta = result.x[1]

        if self.fit_environmental:
            temp_delta = result.x[2]
            water_act = result.x[3]
            scatter_power = result.x[4]
            scatter_amp = result.x[5]
        else:
            temp_delta = 0.0
            water_act = 0.5
            scatter_power = 1.5
            scatter_amp = 0.0

        # Apply final parameters to forward chain
        forward_chain.instrument_model.wl_shift = orig_wl_shift + wl_shift_delta
        if self.fit_environmental and forward_chain.environmental_model is not None:
            forward_chain.environmental_model.temperature_delta = temp_delta
            forward_chain.environmental_model.water_activity = water_act
            forward_chain.environmental_model.scattering_power = scatter_power
            forward_chain.environmental_model.scattering_amplitude = scatter_amp

        A = forward_chain.forward_design_matrix(path_length=path_length)
        linear_params, _ = self._inner_solve(A, target, forward_chain, baseline_reg)

        # Compute fit statistics
        fitted = A @ linear_params
        residuals = target - fitted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Restore original state
        forward_chain.instrument_model.wl_shift = orig_wl_shift
        if orig_env_state is not None and forward_chain.environmental_model is not None:
            forward_chain.environmental_model.temperature_delta = orig_env_state["temperature_delta"]
            forward_chain.environmental_model.water_activity = orig_env_state["water_activity"]
            forward_chain.environmental_model.scattering_power = orig_env_state["scattering_power"]
            forward_chain.environmental_model.scattering_amplitude = orig_env_state["scattering_amplitude"]

        # Parse linear params
        n_comp = forward_chain.canonical_model.n_components
        n_baseline = forward_chain.canonical_model.n_baseline
        n_continuum = forward_chain.canonical_model.n_continuum

        concentrations = linear_params[:n_comp]
        baseline_coeffs = linear_params[n_comp : n_comp + n_baseline]
        continuum_coeffs = (
            linear_params[n_comp + n_baseline :]
            if n_continuum > 0
            else None
        )

        return InversionResult(
            concentrations=concentrations,
            baseline_coeffs=baseline_coeffs,
            continuum_coeffs=continuum_coeffs,
            path_length=path_length,
            wl_shift_residual=wl_shift_delta,
            fitted_spectrum=fitted,
            residuals=residuals,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals ** 2)),
            converged=result.success,
            temperature_delta=temp_delta,
            water_activity=water_act,
            scattering_power=scatter_power,
            scattering_amplitude=scatter_amp,
        )

    def _inner_solve(
        self,
        A: np.ndarray,
        target: np.ndarray,
        forward_chain: "ForwardChain",
        baseline_reg: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve inner linear problem with constraints.

        Constraints:
        - Concentrations >= 0 (NNLS)
        - Baseline/continuum: free (can be negative)
        - Optional smoothness penalty on baseline
        """
        n_comp = forward_chain.canonical_model.n_components
        n_params = A.shape[1]

        # Bounds: concentrations >= 0, baseline free
        lb = np.concatenate([
            np.zeros(n_comp),  # concentrations >= 0
            -np.inf * np.ones(n_params - n_comp),  # baseline free
        ])
        ub = np.inf * np.ones(n_params)

        # Add regularization if needed
        if baseline_reg > 0:
            # Augment system with regularization rows
            n_wl = A.shape[0]
            reg_matrix = baseline_reg * np.eye(n_params)
            reg_matrix[:n_comp, :n_comp] = self.concentration_regularization * np.eye(
                n_comp
            )

            A_aug = np.vstack([A, reg_matrix])
            target_aug = np.concatenate([target, np.zeros(n_params)])

            result = lsq_linear(A_aug, target_aug, bounds=(lb, ub))
        else:
            result = lsq_linear(A, target, bounds=(lb, ub))

        residual_norm = np.sum((A @ result.x - target) ** 2)

        return result.x, residual_norm

    def _final_fit(
        self,
        target: np.ndarray,
        forward_chain: "ForwardChain",
        params: np.ndarray,
    ) -> InversionResult:
        """Final fit with full resolution and statistics."""
        path_length = params[0]
        wl_shift_delta = params[1]

        # Extract environmental parameters if present
        if self.fit_environmental and len(params) >= 6:
            temp_delta = params[2]
            water_act = params[3]
            scatter_power = params[4]
            scatter_amp = params[5]
        else:
            temp_delta = 0.0
            water_act = 0.5
            scatter_power = 1.5
            scatter_amp = 0.0

        # Store original state
        orig_wl_shift = forward_chain.instrument_model.wl_shift
        orig_env_state = None
        if forward_chain.environmental_model is not None:
            orig_env_state = forward_chain.environmental_model.to_dict()

        # Apply per-sample parameters
        forward_chain.instrument_model.wl_shift = orig_wl_shift + wl_shift_delta
        if self.fit_environmental and forward_chain.environmental_model is not None:
            forward_chain.environmental_model.temperature_delta = temp_delta
            forward_chain.environmental_model.water_activity = water_act
            forward_chain.environmental_model.scattering_power = scatter_power
            forward_chain.environmental_model.scattering_amplitude = scatter_amp

        A = forward_chain.forward_design_matrix(path_length=path_length)
        linear_params, _ = self._inner_solve(
            A, target, forward_chain, self.baseline_smoothness_penalty
        )

        # Compute fit
        fitted = A @ linear_params
        residuals = target - fitted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Restore original state
        forward_chain.instrument_model.wl_shift = orig_wl_shift
        if orig_env_state is not None and forward_chain.environmental_model is not None:
            forward_chain.environmental_model.temperature_delta = orig_env_state["temperature_delta"]
            forward_chain.environmental_model.water_activity = orig_env_state["water_activity"]
            forward_chain.environmental_model.scattering_power = orig_env_state["scattering_power"]
            forward_chain.environmental_model.scattering_amplitude = orig_env_state["scattering_amplitude"]

        # Parse params
        n_comp = forward_chain.canonical_model.n_components
        n_baseline = forward_chain.canonical_model.n_baseline
        n_continuum = forward_chain.canonical_model.n_continuum

        return InversionResult(
            concentrations=linear_params[:n_comp],
            baseline_coeffs=linear_params[n_comp : n_comp + n_baseline],
            continuum_coeffs=(
                linear_params[n_comp + n_baseline :]
                if n_continuum > 0
                else None
            ),
            path_length=path_length,
            wl_shift_residual=wl_shift_delta,
            fitted_spectrum=fitted,
            residuals=residuals,
            r_squared=r_squared,
            rmse=np.sqrt(np.mean(residuals ** 2)),
            converged=True,
            temperature_delta=temp_delta,
            water_activity=water_act,
            scattering_power=scatter_power,
            scattering_amplitude=scatter_amp,
        )

    def fit_batch(
        self,
        X: np.ndarray,
        forward_chain: "ForwardChain",
        schedule: Optional[MultiscaleSchedule] = None,
        n_jobs: int = 1,
    ) -> List[InversionResult]:
        """
        Fit multiple spectra.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths).
            forward_chain: Forward chain with calibrated global params.
            schedule: Multiscale fitting schedule.
            n_jobs: Number of parallel jobs (1 = sequential).

        Returns:
            List of InversionResult for each sample.
        """
        n_samples = X.shape[0]
        results = []

        # Use quick schedule for batch to save time
        if schedule is None:
            schedule = MultiscaleSchedule.quick()

        for i in range(n_samples):
            result = self.fit(X[i], forward_chain, schedule)
            results.append(result)

            if self.verbose and (i + 1) % 10 == 0:
                print(f"Fitted {i + 1}/{n_samples} samples")

        return results


# =============================================================================
# Batch Inversion Helper
# =============================================================================


def invert_dataset(
    X: np.ndarray,
    forward_chain: "ForwardChain",
    schedule: Optional[MultiscaleSchedule] = None,
    verbose: bool = False,
    fit_environmental: bool = False,
) -> Tuple[List[InversionResult], Dict[str, np.ndarray]]:
    """
    Invert full dataset and collect parameter arrays.

    Args:
        X: Spectra matrix (n_samples, n_wavelengths).
        forward_chain: Forward chain.
        schedule: Fitting schedule.
        verbose: Print progress.
        fit_environmental: Whether to fit environmental parameters.

    Returns:
        Tuple of (results_list, params_dict) where params_dict contains:
            - concentrations: (n_samples, n_components)
            - baseline_coeffs: (n_samples, n_baseline)
            - path_lengths: (n_samples,)
            - wl_shifts: (n_samples,)
            - r_squared: (n_samples,)
            - temperature_deltas: (n_samples,) [if fit_environmental]
            - water_activities: (n_samples,) [if fit_environmental]
            - scattering_powers: (n_samples,) [if fit_environmental]
            - scattering_amplitudes: (n_samples,) [if fit_environmental]
    """
    solver = VariableProjectionSolver(verbose=verbose, fit_environmental=fit_environmental)
    results = solver.fit_batch(X, forward_chain, schedule)

    n_samples = len(results)
    n_comp = forward_chain.canonical_model.n_components
    n_baseline = forward_chain.canonical_model.n_baseline

    params = {
        "concentrations": np.zeros((n_samples, n_comp)),
        "baseline_coeffs": np.zeros((n_samples, n_baseline)),
        "path_lengths": np.zeros(n_samples),
        "wl_shifts": np.zeros(n_samples),
        "r_squared": np.zeros(n_samples),
    }

    if fit_environmental:
        params["temperature_deltas"] = np.zeros(n_samples)
        params["water_activities"] = np.zeros(n_samples)
        params["scattering_powers"] = np.zeros(n_samples)
        params["scattering_amplitudes"] = np.zeros(n_samples)

    for i, result in enumerate(results):
        params["concentrations"][i] = result.concentrations
        params["baseline_coeffs"][i] = result.baseline_coeffs
        params["path_lengths"][i] = result.path_length
        params["wl_shifts"][i] = result.wl_shift_residual
        params["r_squared"][i] = result.r_squared

        if fit_environmental:
            params["temperature_deltas"][i] = result.temperature_delta
            params["water_activities"][i] = result.water_activity
            params["scattering_powers"][i] = result.scattering_power
            params["scattering_amplitudes"][i] = result.scattering_amplitude

    return results, params

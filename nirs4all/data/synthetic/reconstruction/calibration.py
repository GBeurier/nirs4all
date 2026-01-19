"""
Prototype-based global calibration for instrument parameters.

This module implements global parameter estimation using representative
prototype spectra (median + quantiles + k-medoids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution


# =============================================================================
# Calibration Result
# =============================================================================


@dataclass
class CalibrationResult:
    """
    Result of global calibration.

    Attributes:
        wl_shift: Calibrated wavelength shift.
        wl_stretch: Calibrated wavelength stretch.
        ils_sigma: Calibrated ILS width.
        stray_light: Calibrated stray light fraction.
        gain: Calibrated photometric gain.
        offset: Calibrated photometric offset.
        prototype_residuals: Residuals for each prototype.
        prototype_r2: R² for each prototype.
        total_loss: Total calibration loss.
    """

    wl_shift: float = 0.0
    wl_stretch: float = 1.0
    ils_sigma: float = 4.0
    stray_light: float = 0.0
    gain: float = 1.0
    offset: float = 0.0
    prototype_residuals: Optional[np.ndarray] = None
    prototype_r2: Optional[np.ndarray] = None
    total_loss: float = float("inf")

    def to_dict(self) -> Dict[str, float]:
        """Convert to parameter dictionary."""
        return {
            "wl_shift": self.wl_shift,
            "wl_stretch": self.wl_stretch,
            "ils_sigma": self.ils_sigma,
            "stray_light": self.stray_light,
            "gain": self.gain,
            "offset": self.offset,
        }

    @classmethod
    def from_array(cls, params: np.ndarray) -> "CalibrationResult":
        """Create from parameter array [wl_shift, wl_stretch, ils_sigma]."""
        return cls(
            wl_shift=params[0],
            wl_stretch=params[1] if len(params) > 1 else 1.0,
            ils_sigma=params[2] if len(params) > 2 else 4.0,
        )


# =============================================================================
# Prototype Selector
# =============================================================================


@dataclass
class PrototypeSelector:
    """
    Select representative prototype spectra from a dataset.

    Uses multiple strategies to ensure robust global calibration:
    1. Median spectrum (robust central tendency)
    2. Quantile spectra (25%, 75% in PC1)
    3. K-medoids in PCA space (capture diversity)

    Attributes:
        n_prototypes: Number of prototypes to select.
        include_median: Always include median spectrum.
        include_quantiles: Include quantile spectra.
        pca_components: Number of PCA components for clustering.
    """

    n_prototypes: int = 5
    include_median: bool = True
    include_quantiles: bool = True
    pca_components: int = 5

    def select(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select prototype spectra.

        Args:
            X: Spectra matrix (n_samples, n_wavelengths).

        Returns:
            Tuple of (prototype_spectra, prototype_indices).
        """
        n_samples = X.shape[0]
        indices = []
        prototypes = []

        # 1. Median spectrum (synthesized, not actual sample)
        if self.include_median:
            median_spectrum = np.median(X, axis=0)
            # Find closest actual sample to median
            distances = np.sum((X - median_spectrum) ** 2, axis=1)
            median_idx = np.argmin(distances)
            if median_idx not in indices:
                indices.append(median_idx)
                prototypes.append(X[median_idx])

        # 2. PCA for remaining selections
        if len(indices) < self.n_prototypes:
            from sklearn.decomposition import PCA

            n_comp = min(self.pca_components, n_samples - 1, X.shape[1])
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(X)

            # 2a. Quantile spectra along PC1
            if self.include_quantiles and len(indices) < self.n_prototypes:
                pc1 = scores[:, 0]
                q25_idx = np.argmin(np.abs(pc1 - np.percentile(pc1, 25)))
                q75_idx = np.argmin(np.abs(pc1 - np.percentile(pc1, 75)))

                for idx in [q25_idx, q75_idx]:
                    if idx not in indices and len(indices) < self.n_prototypes:
                        indices.append(idx)
                        prototypes.append(X[idx])

            # 2b. K-medoids clustering for diversity
            remaining = self.n_prototypes - len(indices)
            if remaining > 0:
                kmedoid_indices = self._kmedoids_selection(
                    scores, n_select=remaining, exclude=indices
                )
                for idx in kmedoid_indices:
                    if idx not in indices:
                        indices.append(idx)
                        prototypes.append(X[idx])

        # Ensure we have at least one prototype
        if len(prototypes) == 0:
            indices = [0]
            prototypes = [X[0]]

        return np.array(prototypes), np.array(indices)

    def _kmedoids_selection(
        self,
        scores: np.ndarray,
        n_select: int,
        exclude: List[int],
    ) -> List[int]:
        """Select diverse samples using k-medoids-like approach."""
        n_samples = scores.shape[0]
        available = [i for i in range(n_samples) if i not in exclude]

        if len(available) == 0:
            return []

        selected = []

        # Greedy selection: pick samples maximizing minimum distance to already selected
        if exclude:
            # Start with sample furthest from excluded samples
            excluded_scores = scores[exclude]
            distances = np.min(
                np.sum((scores[available, np.newaxis, :] - excluded_scores) ** 2, axis=2),
                axis=1,
            )
            first_idx = available[np.argmax(distances)]
        else:
            # Start with sample closest to centroid
            centroid = scores.mean(axis=0)
            distances = np.sum((scores[available] - centroid) ** 2, axis=1)
            first_idx = available[np.argmin(distances)]

        selected.append(first_idx)

        # Greedily add samples
        while len(selected) < n_select and len(selected) < len(available):
            remaining = [i for i in available if i not in selected]
            if not remaining:
                break

            selected_scores = scores[selected]
            distances = np.min(
                np.sum(
                    (scores[remaining, np.newaxis, :] - selected_scores) ** 2, axis=2
                ),
                axis=1,
            )
            next_idx = remaining[np.argmax(distances)]
            selected.append(next_idx)

        return selected


# =============================================================================
# Global Calibrator
# =============================================================================


@dataclass
class GlobalCalibrator:
    """
    Calibrate global instrument parameters using prototype spectra.

    Optimizes θ_global = {wl_shift, wl_stretch, ils_sigma} to minimize
    total fitting loss across all prototypes, with per-prototype linear
    parameters solved via NNLS.

    Attributes:
        forward_chain: ForwardChain for computing model predictions.
        wl_shift_bounds: Bounds for wavelength shift.
        wl_stretch_bounds: Bounds for wavelength stretch.
        ils_sigma_bounds: Bounds for ILS sigma.
        regularization: L2 regularization strength.
        use_global_search: Use differential evolution for global search.
    """

    wl_shift_bounds: Tuple[float, float] = (-10.0, 10.0)
    wl_stretch_bounds: Tuple[float, float] = (0.98, 1.02)
    ils_sigma_bounds: Tuple[float, float] = (2.0, 20.0)
    regularization: float = 1e-6
    use_global_search: bool = False

    def calibrate(
        self,
        prototypes: np.ndarray,
        forward_chain: "ForwardChain",
        initial_guess: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Calibrate global parameters on prototype spectra.

        Args:
            prototypes: Prototype spectra (n_prototypes, n_wavelengths).
            forward_chain: Forward chain for model evaluation.
            initial_guess: Initial [wl_shift, wl_stretch, ils_sigma].

        Returns:
            CalibrationResult with optimized parameters.
        """
        from scipy.optimize import nnls

        n_prototypes = prototypes.shape[0]
        n_wl = prototypes.shape[1]

        # Store forward chain reference
        self._forward_chain = forward_chain

        def objective(params: np.ndarray) -> float:
            """Total loss across all prototypes."""
            wl_shift, wl_stretch, ils_sigma = params

            # Update instrument model
            forward_chain.instrument_model.wl_shift = wl_shift
            forward_chain.instrument_model.wl_stretch = wl_stretch
            forward_chain.instrument_model.ils_sigma = ils_sigma

            total_loss = 0.0
            for i in range(n_prototypes):
                try:
                    # Get design matrix with current instrument params
                    A = forward_chain.forward_design_matrix(path_length=1.0)

                    # NNLS for linear params (concentrations >= 0, baseline free)
                    n_comp = forward_chain.canonical_model.n_components
                    n_linear = A.shape[1]

                    # Use bounded least squares to enforce concentration >= 0
                    from scipy.optimize import lsq_linear

                    lb = np.concatenate([
                        np.zeros(n_comp),  # concentrations >= 0
                        -np.inf * np.ones(n_linear - n_comp),  # baseline free
                    ])
                    ub = np.inf * np.ones(n_linear)

                    result = lsq_linear(A, prototypes[i], bounds=(lb, ub))
                    residuals = prototypes[i] - A @ result.x

                    # Weighted by prototype (equal weights)
                    loss = np.sum(residuals ** 2)
                    total_loss += loss

                except Exception:
                    total_loss += 1e6

            # Add regularization on parameter deviation from defaults
            reg_loss = self.regularization * (
                wl_shift ** 2 +
                100 * (wl_stretch - 1.0) ** 2 +
                0.1 * (ils_sigma - 6.0) ** 2
            )

            return total_loss + reg_loss

        # Initial guess
        if initial_guess is None:
            initial_guess = np.array([0.0, 1.0, 6.0])

        bounds = [
            self.wl_shift_bounds,
            self.wl_stretch_bounds,
            self.ils_sigma_bounds,
        ]

        # Optimize
        if self.use_global_search:
            result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                polish=True,
            )
        else:
            result = minimize(
                objective,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200},
            )

        # Extract results
        wl_shift, wl_stretch, ils_sigma = result.x

        # Update forward chain with calibrated params
        forward_chain.instrument_model.wl_shift = wl_shift
        forward_chain.instrument_model.wl_stretch = wl_stretch
        forward_chain.instrument_model.ils_sigma = ils_sigma

        # Compute per-prototype R²
        prototype_r2 = np.zeros(n_prototypes)
        prototype_residuals = []

        for i in range(n_prototypes):
            A = forward_chain.forward_design_matrix(path_length=1.0)
            n_comp = forward_chain.canonical_model.n_components
            n_linear = A.shape[1]

            from scipy.optimize import lsq_linear

            lb = np.concatenate([
                np.zeros(n_comp),
                -np.inf * np.ones(n_linear - n_comp),
            ])
            ub = np.inf * np.ones(n_linear)

            fit_result = lsq_linear(A, prototypes[i], bounds=(lb, ub))
            fitted = A @ fit_result.x
            residuals = prototypes[i] - fitted

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((prototypes[i] - prototypes[i].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            prototype_r2[i] = r2
            prototype_residuals.append(residuals)

        return CalibrationResult(
            wl_shift=wl_shift,
            wl_stretch=wl_stretch,
            ils_sigma=ils_sigma,
            prototype_residuals=np.array(prototype_residuals),
            prototype_r2=prototype_r2,
            total_loss=result.fun,
        )

    def refine(
        self,
        current_result: CalibrationResult,
        prototypes: np.ndarray,
        forward_chain: "ForwardChain",
    ) -> CalibrationResult:
        """
        Refine calibration with tighter bounds around current estimate.

        Args:
            current_result: Current calibration result.
            prototypes: Prototype spectra.
            forward_chain: Forward chain.

        Returns:
            Refined CalibrationResult.
        """
        # Tighten bounds
        margin = 0.5
        self.wl_shift_bounds = (
            current_result.wl_shift - margin * 2,
            current_result.wl_shift + margin * 2,
        )
        self.wl_stretch_bounds = (
            max(0.98, current_result.wl_stretch - 0.005),
            min(1.02, current_result.wl_stretch + 0.005),
        )
        self.ils_sigma_bounds = (
            max(2.0, current_result.ils_sigma - 2),
            min(20.0, current_result.ils_sigma + 2),
        )

        initial = np.array([
            current_result.wl_shift,
            current_result.wl_stretch,
            current_result.ils_sigma,
        ])

        return self.calibrate(prototypes, forward_chain, initial_guess=initial)


# =============================================================================
# Multi-stage Calibration
# =============================================================================


def multistage_calibration(
    X: np.ndarray,
    forward_chain: "ForwardChain",
    n_prototypes: int = 5,
    stages: int = 2,
) -> CalibrationResult:
    """
    Multi-stage calibration with progressive refinement.

    Stage 1: Coarse calibration on smoothed prototypes
    Stage 2: Fine calibration on original prototypes

    Args:
        X: Full dataset (n_samples, n_wavelengths).
        forward_chain: Forward chain for model evaluation.
        n_prototypes: Number of prototypes to select.
        stages: Number of refinement stages.

    Returns:
        Final CalibrationResult.
    """
    from scipy.ndimage import gaussian_filter1d

    # Select prototypes
    selector = PrototypeSelector(n_prototypes=n_prototypes)
    prototypes, indices = selector.select(X)

    calibrator = GlobalCalibrator()

    # Stage 1: Coarse on smoothed
    smooth_sigmas = [10, 5, 0][:stages + 1]
    result = None

    for sigma in smooth_sigmas:
        if sigma > 0:
            protos_smooth = gaussian_filter1d(prototypes, sigma=sigma, axis=1)
        else:
            protos_smooth = prototypes

        if result is None:
            result = calibrator.calibrate(protos_smooth, forward_chain)
        else:
            result = calibrator.refine(result, protos_smooth, forward_chain)

    return result

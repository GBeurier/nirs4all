"""
Validation and diagnostics for reconstruction quality.

Provides tools to evaluate:
1. Reconstruction quality (residual analysis)
2. Synthetic vs real data comparison (PCA, statistics)
3. Parameter plausibility checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of reconstruction validation.

    Attributes:
        reconstruction_metrics: Per-sample reconstruction quality.
        synthetic_metrics: Synthetic vs real comparison metrics.
        parameter_metrics: Parameter plausibility metrics.
        overall_score: Combined quality score (0-100).
        passed: Whether all quality checks passed.
        warnings: List of warning messages.
    """

    reconstruction_metrics: Dict[str, Any] = field(default_factory=dict)
    synthetic_metrics: Dict[str, Any] = field(default_factory=dict)
    parameter_metrics: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = False
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "Reconstruction Validation Summary",
            "=" * 70,
            f"Overall Score: {self.overall_score:.1f}/100",
            f"Status: {'PASSED' if self.passed else 'NEEDS REVIEW'}",
            "",
        ]

        if self.reconstruction_metrics:
            lines.append("Reconstruction Quality:")
            for k, v in self.reconstruction_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        if self.synthetic_metrics:
            lines.append("")
            lines.append("Synthetic vs Real Comparison:")
            for k, v in self.synthetic_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Reconstruction Validator
# =============================================================================


@dataclass
class ReconstructionValidator:
    """
    Validate reconstruction quality and synthetic realism.

    Checks:
    1. Residuals should be structureless (no systematic patterns)
    2. Synthetic should match real in PCA space
    3. Per-wavelength statistics should be similar
    4. Parameters should be physically plausible

    Attributes:
        r2_threshold: Minimum acceptable R² for reconstruction.
        residual_autocorr_threshold: Max autocorrelation in residuals.
        pca_distance_threshold: Max Mahalanobis distance in PCA space.
        concentration_max: Max plausible concentration value.
    """

    r2_threshold: float = 0.90
    residual_autocorr_threshold: float = 0.3
    pca_distance_threshold: float = 3.0
    concentration_max: float = 10.0
    path_length_bounds: Tuple[float, float] = (0.3, 3.0)

    def validate_reconstruction(
        self,
        inversion_results: List["InversionResult"],
    ) -> Dict[str, Any]:
        """
        Validate reconstruction quality.

        Args:
            inversion_results: List of inversion results.

        Returns:
            Dict of reconstruction metrics.
        """
        metrics = {}

        # R² statistics
        r2_values = np.array([r.r_squared for r in inversion_results])
        metrics["mean_r2"] = float(np.mean(r2_values))
        metrics["min_r2"] = float(np.min(r2_values))
        metrics["r2_above_threshold"] = float(np.mean(r2_values >= self.r2_threshold))

        # RMSE statistics
        rmse_values = np.array([r.rmse for r in inversion_results])
        metrics["mean_rmse"] = float(np.mean(rmse_values))
        metrics["median_rmse"] = float(np.median(rmse_values))

        # Residual analysis (check for systematic patterns)
        all_residuals = []
        for r in inversion_results:
            if r.residuals is not None:
                all_residuals.append(r.residuals)

        if len(all_residuals) > 0:
            residuals = np.array(all_residuals)

            # Mean residual (should be near zero)
            mean_residual = np.mean(residuals, axis=0)
            metrics["mean_residual_magnitude"] = float(np.mean(np.abs(mean_residual)))

            # Residual autocorrelation (should be low)
            autocorrs = []
            for res in residuals[:min(50, len(residuals))]:
                ac = np.corrcoef(res[:-1], res[1:])[0, 1]
                if not np.isnan(ac):
                    autocorrs.append(ac)
            if autocorrs:
                metrics["residual_autocorr"] = float(np.mean(autocorrs))

            # Check for oscillatory patterns (high-frequency energy)
            fft_energy = []
            for res in residuals[:min(50, len(residuals))]:
                fft = np.fft.rfft(res)
                high_freq = np.sum(np.abs(fft[len(fft)//2:])**2)
                total = np.sum(np.abs(fft)**2) + 1e-10
                fft_energy.append(high_freq / total)
            metrics["high_freq_residual_energy"] = float(np.mean(fft_energy))

        return metrics

    def validate_synthetic(
        self,
        X_real: np.ndarray,
        X_synth: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Validate synthetic vs real data.

        Args:
            X_real: Real data matrix.
            X_synth: Synthetic data matrix.

        Returns:
            Dict of comparison metrics.
        """
        metrics = {}

        # Basic statistics comparison
        real_mean = X_real.mean(axis=0)
        synth_mean = X_synth.mean(axis=0)
        real_std = X_real.std(axis=0)
        synth_std = X_synth.std(axis=0)

        # Correlation of mean spectra
        mean_corr = np.corrcoef(real_mean, synth_mean)[0, 1]
        metrics["mean_spectrum_correlation"] = float(mean_corr)

        # Correlation of std spectra
        std_corr = np.corrcoef(real_std, synth_std)[0, 1]
        metrics["std_spectrum_correlation"] = float(std_corr)

        # Relative mean difference
        rel_mean_diff = np.mean(np.abs(real_mean - synth_mean) / (np.abs(real_mean) + 1e-10))
        metrics["relative_mean_difference"] = float(rel_mean_diff)

        # Relative std difference
        rel_std_diff = np.mean(np.abs(real_std - synth_std) / (real_std + 1e-10))
        metrics["relative_std_difference"] = float(rel_std_diff)

        # PCA comparison
        try:
            from sklearn.decomposition import PCA

            n_comp = min(10, X_real.shape[0] - 1, X_synth.shape[0] - 1, X_real.shape[1])
            pca = PCA(n_components=n_comp)
            scores_real = pca.fit_transform(X_real)
            scores_synth = pca.transform(X_synth)

            # Compare score distributions
            for i in range(min(3, n_comp)):
                ks_stat, ks_pvalue = stats.ks_2samp(scores_real[:, i], scores_synth[:, i])
                metrics[f"pca_score_{i+1}_ks_stat"] = float(ks_stat)
                metrics[f"pca_score_{i+1}_ks_pvalue"] = float(ks_pvalue)

            # Variance explained comparison
            metrics["pca_variance_real"] = pca.explained_variance_ratio_[:3].tolist()

        except Exception as e:
            metrics["pca_error"] = str(e)

        # Simple discriminator (can synthetic be distinguished from real?)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            # Combine and label
            X_combined = np.vstack([X_real, X_synth])
            y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_synth))])

            # Use PCA features to avoid overfitting
            if "pca_error" not in metrics:
                X_pca = pca.transform(X_combined)[:, :min(5, n_comp)]
            else:
                X_pca = X_combined[:, ::max(1, X_combined.shape[1] // 20)]  # Subsample wavelengths

            # Cross-validated accuracy
            clf = LogisticRegression(max_iter=1000, random_state=42)
            cv_scores = cross_val_score(clf, X_pca, y, cv=min(5, len(X_real)))
            metrics["discriminator_accuracy"] = float(np.mean(cv_scores))
            # Accuracy close to 0.5 = good (indistinguishable)

        except Exception as e:
            metrics["discriminator_error"] = str(e)

        return metrics

    def validate_parameters(
        self,
        inversion_results: List["InversionResult"],
    ) -> Dict[str, Any]:
        """
        Validate parameter plausibility.

        Args:
            inversion_results: List of inversion results.

        Returns:
            Dict of parameter metrics.
        """
        metrics = {}
        warnings = []

        # Concentration statistics
        concentrations = np.array([r.concentrations for r in inversion_results])
        metrics["concentration_mean"] = float(np.mean(concentrations))
        metrics["concentration_max"] = float(np.max(concentrations))
        metrics["concentration_negative_frac"] = float(np.mean(concentrations < 0))

        if np.max(concentrations) > self.concentration_max:
            warnings.append(f"Max concentration ({np.max(concentrations):.2f}) exceeds threshold")
        if np.mean(concentrations < 0) > 0.1:
            warnings.append(f"Significant negative concentrations ({np.mean(concentrations < 0)*100:.1f}%)")

        # Path length statistics
        path_lengths = np.array([r.path_length for r in inversion_results])
        metrics["path_length_mean"] = float(np.mean(path_lengths))
        metrics["path_length_std"] = float(np.std(path_lengths))

        out_of_bounds = (path_lengths < self.path_length_bounds[0]) | (path_lengths > self.path_length_bounds[1])
        metrics["path_length_out_of_bounds_frac"] = float(np.mean(out_of_bounds))

        if np.mean(out_of_bounds) > 0.1:
            warnings.append(f"Path lengths out of bounds ({np.mean(out_of_bounds)*100:.1f}%)")

        # Wavelength shift statistics
        wl_shifts = np.array([r.wl_shift_residual for r in inversion_results])
        metrics["wl_shift_mean"] = float(np.mean(wl_shifts))
        metrics["wl_shift_std"] = float(np.std(wl_shifts))

        metrics["warnings"] = warnings

        return metrics

    def validate(
        self,
        inversion_results: List["InversionResult"],
        X_real: np.ndarray,
        X_synth: np.ndarray,
    ) -> ValidationResult:
        """
        Run full validation.

        Args:
            inversion_results: Inversion results.
            X_real: Real data.
            X_synth: Synthetic data.

        Returns:
            ValidationResult.
        """
        recon_metrics = self.validate_reconstruction(inversion_results)
        synth_metrics = self.validate_synthetic(X_real, X_synth)
        param_metrics = self.validate_parameters(inversion_results)

        warnings = param_metrics.pop("warnings", [])

        # Compute overall score
        score = 0.0
        checks_passed = 0
        total_checks = 0

        # Reconstruction checks
        if recon_metrics.get("mean_r2", 0) >= self.r2_threshold:
            score += 25
            checks_passed += 1
        total_checks += 1

        # Synthetic checks
        if synth_metrics.get("mean_spectrum_correlation", 0) >= 0.95:
            score += 25
            checks_passed += 1
        total_checks += 1

        if synth_metrics.get("discriminator_accuracy", 1.0) <= 0.7:
            score += 25
            checks_passed += 1
        total_checks += 1

        # Parameter checks
        if param_metrics.get("concentration_negative_frac", 1.0) <= 0.05:
            score += 12.5
            checks_passed += 0.5
        if param_metrics.get("path_length_out_of_bounds_frac", 1.0) <= 0.1:
            score += 12.5
            checks_passed += 0.5
        total_checks += 1

        passed = checks_passed >= total_checks * 0.75

        return ValidationResult(
            reconstruction_metrics=recon_metrics,
            synthetic_metrics=synth_metrics,
            parameter_metrics=param_metrics,
            overall_score=score,
            passed=passed,
            warnings=warnings,
        )


# =============================================================================
# Diagnostic Plots (Data Generation)
# =============================================================================


def compute_diagnostic_data(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    inversion_results: Optional[List["InversionResult"]] = None,
    wavelengths: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute data for diagnostic plots.

    Args:
        X_real: Real data.
        X_synth: Synthetic data.
        inversion_results: Optional inversion results for residuals.
        wavelengths: Wavelength grid.

    Returns:
        Dict of diagnostic data arrays.
    """
    if wavelengths is None:
        wavelengths = np.arange(X_real.shape[1])

    data = {
        "wavelengths": wavelengths,
    }

    # Mean and std spectra
    data["real_mean"] = X_real.mean(axis=0)
    data["real_std"] = X_real.std(axis=0)
    data["synth_mean"] = X_synth.mean(axis=0)
    data["synth_std"] = X_synth.std(axis=0)

    # Residuals
    if inversion_results:
        residuals = np.array([
            r.residuals for r in inversion_results
            if r.residuals is not None
        ])
        if len(residuals) > 0:
            data["residual_mean"] = residuals.mean(axis=0)
            data["residual_std"] = residuals.std(axis=0)

    # PCA
    try:
        from sklearn.decomposition import PCA

        n_comp = min(5, X_real.shape[0] - 1, X_synth.shape[0] - 1, X_real.shape[1])
        pca = PCA(n_components=n_comp)
        data["real_pca_scores"] = pca.fit_transform(X_real)
        data["synth_pca_scores"] = pca.transform(X_synth)
        data["pca_explained_variance"] = pca.explained_variance_ratio_
        data["pca_components"] = pca.components_
    except Exception:
        pass

    return data

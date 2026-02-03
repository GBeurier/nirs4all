"""
Parameter distribution fitting and sampling for variance modeling.

Learns distributions of physical parameters from inverted samples,
then samples from these distributions for synthetic generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import stats


# =============================================================================
# Distribution Result
# =============================================================================


@dataclass
class DistributionResult:
    """
    Result of parameter distribution fitting.

    Attributes:
        param_names: Names of parameters.
        distributions: Dict of distribution parameters for each param.
        correlations: Correlation matrix of transformed parameters.
        factor_loadings: Low-rank factor model loadings (optional).
        transform_params: Parameters for transformations (log, etc.).
        n_samples_fitted: Number of samples used for fitting.
    """

    param_names: List[str]
    distributions: Dict[str, Dict[str, Any]]
    correlations: Optional[np.ndarray] = None
    factor_loadings: Optional[np.ndarray] = None
    transform_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_samples_fitted: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Parameter Distribution Summary",
            "=" * 60,
            f"Parameters: {len(self.param_names)}",
            f"Samples fitted: {self.n_samples_fitted}",
            "",
        ]

        for name in self.param_names:
            if name in self.distributions:
                dist = self.distributions[name]
                dist_type = dist.get("type", "unknown")
                lines.append(f"{name}:")
                lines.append(f"  Distribution: {dist_type}")
                if "mean" in dist:
                    lines.append(f"  Mean: {dist['mean']:.4f}")
                if "std" in dist:
                    lines.append(f"  Std: {dist['std']:.4f}")
                if "bounds" in dist:
                    lines.append(f"  Bounds: {dist['bounds']}")

        if self.correlations is not None:
            lines.append("")
            lines.append("Correlation matrix (top-left 5x5):")
            n_show = min(5, self.correlations.shape[0])
            for i in range(n_show):
                row = " ".join(f"{self.correlations[i, j]:6.2f}" for j in range(n_show))
                lines.append(f"  {row}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Parameter Distribution Fitter
# =============================================================================


@dataclass
class ParameterDistributionFitter:
    """
    Fit distributions to parameter samples.

    For positive parameters (concentrations, path_length):
        - Use log-normal or gamma distributions
        - Transform to log space for correlation modeling

    For shift parameters (wl_shift):
        - Use Gaussian distributions

    For bounded parameters:
        - Use truncated normal or beta distributions

    Attributes:
        positive_params: Names of parameters that must be positive.
        bounded_params: Dict of param_name -> (lower, upper) bounds.
        use_factor_model: Use low-rank factor model for correlations.
        n_factors: Number of factors for factor model.
        min_std: Minimum standard deviation to avoid degenerate distributions.
    """

    positive_params: List[str] = field(
        default_factory=lambda: ["concentrations", "path_lengths"]
    )
    bounded_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    use_factor_model: bool = False
    n_factors: int = 3
    min_std: float = 1e-6

    def fit(
        self,
        params: Dict[str, np.ndarray],
        param_names: Optional[List[str]] = None,
    ) -> DistributionResult:
        """
        Fit distributions to parameter samples.

        Args:
            params: Dict of parameter arrays. Each array has shape (n_samples,)
                or (n_samples, n_features) for multi-dimensional params.
            param_names: Optional list of parameter names to fit.

        Returns:
            DistributionResult with fitted distributions.
        """
        if param_names is None:
            param_names = list(params.keys())

        distributions = {}
        transform_params = {}
        transformed_data = []
        feature_names = []

        for name in param_names:
            if name not in params:
                continue

            data = np.asarray(params[name])
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_features = data.shape[1]

            for j in range(n_features):
                col = data[:, j]
                feat_name = f"{name}_{j}" if n_features > 1 else name

                # Determine distribution type
                if any(name.startswith(p) for p in self.positive_params):
                    dist_info, transformed = self._fit_positive(col, feat_name)
                elif name in self.bounded_params:
                    bounds = self.bounded_params[name]
                    dist_info, transformed = self._fit_bounded(col, feat_name, bounds)
                else:
                    dist_info, transformed = self._fit_gaussian(col, feat_name)

                distributions[feat_name] = dist_info
                transform_params[feat_name] = dist_info.get("transform", {})
                transformed_data.append(transformed)
                feature_names.append(feat_name)

        # Compute correlation matrix of transformed parameters
        if len(transformed_data) > 0:
            X_transformed = np.column_stack(transformed_data)

            # Handle constant columns
            stds = np.std(X_transformed, axis=0)
            valid_cols = stds > 1e-10

            if np.sum(valid_cols) > 1:
                X_valid = X_transformed[:, valid_cols]
                correlations = np.corrcoef(X_valid.T)

                # Expand back to full size
                full_corr = np.eye(len(feature_names))
                valid_idx = np.where(valid_cols)[0]
                for i, vi in enumerate(valid_idx):
                    for k, vj in enumerate(valid_idx):
                        full_corr[vi, vj] = correlations[i, k]
            else:
                full_corr = np.eye(len(feature_names))

            # Optional factor model
            factor_loadings = None
            if self.use_factor_model and X_transformed.shape[1] > self.n_factors:
                factor_loadings = self._fit_factor_model(X_transformed)
        else:
            full_corr = None
            factor_loadings = None

        return DistributionResult(
            param_names=feature_names,
            distributions=distributions,
            correlations=full_corr,
            factor_loadings=factor_loadings,
            transform_params=transform_params,
            n_samples_fitted=len(transformed_data[0]) if transformed_data else 0,
        )

    def _fit_positive(
        self, data: np.ndarray, name: str
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Fit distribution for positive parameter (log-normal)."""
        # Remove zeros/negatives
        valid = data > 0
        if np.sum(valid) < 3:
            # Fallback to constant
            mean_val = max(np.mean(data), 1e-6)
            return {
                "type": "constant",
                "value": mean_val,
                "transform": {"type": "log"},
            }, np.log(np.maximum(data, 1e-10))

        data_valid = data[valid]

        # Fit in log space (log-normal)
        log_data = np.log(data_valid)
        mu = np.mean(log_data)
        sigma = max(np.std(log_data), self.min_std)

        # Check if distribution is degenerate
        if sigma < self.min_std:
            return {
                "type": "constant",
                "value": np.exp(mu),
                "transform": {"type": "log"},
            }, np.log(np.maximum(data, 1e-10))

        return {
            "type": "lognormal",
            "mu": float(mu),
            "sigma": float(sigma),
            "mean": float(np.exp(mu + sigma**2 / 2)),
            "std": float(np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))),
            "transform": {"type": "log"},
        }, np.log(np.maximum(data, 1e-10))

    def _fit_bounded(
        self, data: np.ndarray, name: str, bounds: Tuple[float, float]
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Fit distribution for bounded parameter (truncated normal or beta)."""
        lower, upper = bounds

        # Clip to bounds
        data_clipped = np.clip(data, lower + 1e-10, upper - 1e-10)

        # Transform to (0, 1) for beta
        normalized = (data_clipped - lower) / (upper - lower)

        # Fit beta distribution
        alpha, beta_param, _, _ = stats.beta.fit(normalized, floc=0, fscale=1)

        # Also compute Gaussian stats for correlation modeling
        mean = np.mean(data_clipped)
        std = max(np.std(data_clipped), self.min_std)

        # Use logit transform for correlation modeling
        logit_data = np.log(normalized / (1 - normalized + 1e-10))

        return {
            "type": "beta",
            "alpha": float(alpha),
            "beta": float(beta_param),
            "bounds": bounds,
            "mean": float(mean),
            "std": float(std),
            "transform": {"type": "logit", "bounds": bounds},
        }, logit_data

    def _fit_gaussian(
        self, data: np.ndarray, name: str
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """Fit Gaussian distribution."""
        mean = float(np.mean(data))
        std = max(float(np.std(data)), self.min_std)

        return {
            "type": "gaussian",
            "mean": mean,
            "std": std,
            "transform": {"type": "none"},
        }, data

    def _fit_factor_model(self, X: np.ndarray) -> np.ndarray:
        """Fit low-rank factor model to transformed data."""
        from sklearn.decomposition import FactorAnalysis

        n_factors = min(self.n_factors, X.shape[1] - 1, X.shape[0] - 1)
        if n_factors < 1:
            return None

        fa = FactorAnalysis(n_components=n_factors)
        fa.fit(X)

        return fa.components_.T  # (n_features, n_factors)


# =============================================================================
# Parameter Sampler
# =============================================================================


@dataclass
class ParameterSampler:
    """
    Sample parameters from fitted distributions.

    Uses Gaussian copula to maintain correlations between parameters
    while respecting marginal distributions.

    Attributes:
        distribution_result: Fitted DistributionResult.
        use_correlations: Whether to model parameter correlations.
    """

    distribution_result: DistributionResult
    use_correlations: bool = True

    def sample(
        self, n_samples: int, random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample parameters from fitted distributions.

        Args:
            n_samples: Number of samples to generate.
            random_state: Random seed.

        Returns:
            Dict of parameter arrays with same structure as fit input.
        """
        rng = np.random.default_rng(random_state)

        n_features = len(self.distribution_result.param_names)

        if n_features == 0:
            return {}

        # Generate correlated Gaussian samples
        if self.use_correlations and self.distribution_result.correlations is not None:
            corr = self.distribution_result.correlations

            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(corr)
            if np.min(eigvals) < 1e-10:
                corr = corr + np.eye(n_features) * (1e-6 - np.min(eigvals))

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(corr)
                z_uncorr = rng.standard_normal((n_samples, n_features))
                z_corr = z_uncorr @ L.T
            except np.linalg.LinAlgError:
                z_corr = rng.standard_normal((n_samples, n_features))
        else:
            z_corr = rng.standard_normal((n_samples, n_features))

        # Transform to uniform via standard normal CDF
        u = stats.norm.cdf(z_corr)

        # Transform to marginal distributions
        samples = {}

        for j, name in enumerate(self.distribution_result.param_names):
            dist = self.distribution_result.distributions.get(name, {})
            dist_type = dist.get("type", "gaussian")

            if dist_type == "constant":
                samples[name] = np.full(n_samples, dist["value"])

            elif dist_type == "lognormal":
                mu = dist["mu"]
                sigma = dist["sigma"]
                # Transform from uniform to log-normal
                samples[name] = np.exp(stats.norm.ppf(u[:, j]) * sigma + mu)

            elif dist_type == "beta":
                alpha = dist["alpha"]
                beta_param = dist["beta"]
                bounds = dist["bounds"]
                # Transform from uniform to beta, then scale to bounds
                beta_samples = stats.beta.ppf(u[:, j], alpha, beta_param)
                samples[name] = bounds[0] + beta_samples * (bounds[1] - bounds[0])

            elif dist_type == "gaussian":
                mean = dist["mean"]
                std = dist["std"]
                samples[name] = stats.norm.ppf(u[:, j]) * std + mean

            else:
                # Fallback to standard normal
                samples[name] = z_corr[:, j]

        # Reorganize multi-dimensional parameters
        return self._reorganize_samples(samples)

    def _reorganize_samples(
        self, flat_samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Reorganize flat samples into original parameter structure."""
        # Group by base name
        grouped = {}

        for name, values in flat_samples.items():
            # Check if name has index suffix
            if "_" in name:
                parts = name.rsplit("_", 1)
                try:
                    idx = int(parts[1])
                    base_name = parts[0]
                    if base_name not in grouped:
                        grouped[base_name] = {}
                    grouped[base_name][idx] = values
                    continue
                except ValueError:
                    pass

            grouped[name] = values

        # Convert indexed groups to arrays
        result = {}
        for name, data in grouped.items():
            if isinstance(data, dict):
                # Multi-dimensional
                max_idx = max(data.keys())
                n_samples = len(data[0])
                arr = np.zeros((n_samples, max_idx + 1))
                for idx, vals in data.items():
                    arr[:, idx] = vals
                result[name] = arr
            else:
                result[name] = data

        return result

    def sample_single(
        self, random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Sample a single parameter set."""
        samples = self.sample(1, random_state)
        return {k: v[0] if v.ndim > 1 else v[0] for k, v in samples.items()}


# =============================================================================
# Convenience Functions
# =============================================================================


def fit_parameter_distributions(
    inversion_results: List["InversionResult"],
    component_names: Optional[List[str]] = None,
    include_environmental: bool = False,
) -> Tuple[DistributionResult, ParameterSampler]:
    """
    Fit distributions from inversion results.

    Args:
        inversion_results: List of InversionResult from batch inversion.
        component_names: Optional component names for concentrations.
        include_environmental: Whether to include environmental parameter distributions.

    Returns:
        Tuple of (DistributionResult, ParameterSampler).
    """
    n_samples = len(inversion_results)

    if n_samples == 0:
        raise ValueError("No inversion results provided")

    # Extract parameter arrays
    n_comp = len(inversion_results[0].concentrations)
    n_baseline = len(inversion_results[0].baseline_coeffs)

    params = {
        "concentrations": np.zeros((n_samples, n_comp)),
        "baseline_coeffs": np.zeros((n_samples, n_baseline)),
        "path_lengths": np.zeros(n_samples),
        "wl_shifts": np.zeros(n_samples),
    }

    for i, result in enumerate(inversion_results):
        params["concentrations"][i] = result.concentrations
        params["baseline_coeffs"][i] = result.baseline_coeffs
        params["path_lengths"][i] = result.path_length
        params["wl_shifts"][i] = result.wl_shift_residual

    # Add environmental parameters if requested
    if include_environmental:
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

    # Build bounded params dict
    bounded_params = {
        "wl_shifts": (-5.0, 5.0),
    }
    if include_environmental:
        bounded_params["water_activities"] = (0.0, 1.0)
        bounded_params["scattering_powers"] = (0.5, 3.0)

    # Fit distributions
    positive_params = ["concentrations", "path_lengths"]
    if include_environmental:
        positive_params.append("scattering_amplitudes")

    fitter = ParameterDistributionFitter(
        positive_params=positive_params,
        bounded_params=bounded_params,
    )

    result = fitter.fit(params)

    # Create sampler
    sampler = ParameterSampler(result, use_correlations=True)

    return result, sampler

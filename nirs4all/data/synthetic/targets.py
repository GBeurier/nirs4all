"""
Target generation for synthetic NIRS datasets.

This module provides tools for generating target variables for regression
and classification tasks, with configurable distributions and class separation.

Example:
    >>> from nirs4all.data.synthetic.targets import TargetGenerator
    >>>
    >>> generator = TargetGenerator(random_state=42)
    >>>
    >>> # Regression targets
    >>> y = generator.regression(
    ...     n_samples=100,
    ...     concentrations=C,  # From spectra generation
    ...     distribution="lognormal",
    ...     range=(0, 100)
    ... )
    >>>
    >>> # Classification with separable classes
    >>> y = generator.classification(
    ...     n_samples=100,
    ...     concentrations=C,
    ...     n_classes=3,
    ...     separation=2.0
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class ClassSeparationConfig:
    """
    Configuration for class separation in classification tasks.

    Attributes:
        separation: Separation factor (higher = more separable).
            Values around 0.5-1.0 create overlapping classes.
            Values around 2.0-3.0 create well-separated classes.
        method: How to create class differences:
            - "component": Different component concentration profiles per class.
            - "shift": Systematic spectral shifts between classes.
            - "intensity": Different overall intensity levels.
        noise: Noise level to add to class boundaries.
    """

    separation: float = 1.5
    method: Literal["component", "shift", "intensity"] = "component"
    noise: float = 0.1


class TargetGenerator:
    """
    Generate target variables for synthetic NIRS datasets.

    This class creates both regression targets (continuous values correlated
    with component concentrations) and classification targets (discrete labels
    with controllable class separation).

    Attributes:
        rng: NumPy random generator for reproducibility.

    Args:
        random_state: Random seed for reproducibility.

    Example:
        >>> generator = TargetGenerator(random_state=42)
        >>>
        >>> # Generate concentrations first (from SyntheticNIRSGenerator)
        >>> C = np.random.rand(100, 5)  # 5 components
        >>>
        >>> # Regression targets scaled to percentage
        >>> y = generator.regression(
        ...     n_samples=100,
        ...     concentrations=C,
        ...     component=0,  # Use first component
        ...     range=(0, 100)
        ... )
        >>>
        >>> # Multi-class classification
        >>> y = generator.classification(
        ...     n_samples=100,
        ...     concentrations=C,
        ...     n_classes=4,
        ...     separation=2.0
        ... )
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        """
        Initialize the target generator.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def regression(
        self,
        n_samples: int,
        concentrations: Optional[np.ndarray] = None,
        *,
        distribution: Literal["uniform", "normal", "lognormal", "bimodal"] = "uniform",
        range: Optional[Tuple[float, float]] = None,
        component: Optional[Union[int, str, List[int]]] = None,
        component_names: Optional[List[str]] = None,
        correlation: float = 0.9,
        noise: float = 0.1,
        transform: Optional[Literal["log", "sqrt"]] = None,
    ) -> np.ndarray:
        """
        Generate regression target values.

        Args:
            n_samples: Number of samples.
            concentrations: Component concentration matrix (n_samples, n_components).
                If None, generates random base values.
            distribution: Target value distribution.
            range: (min, max) for scaling targets.
            component: Which component(s) to use as target:
                - None: Weighted combination of all components
                - int: Use component at that index
                - str: Use component with that name (requires component_names)
                - List[int]: Multi-output using specified component indices
            component_names: Names of components (for string component selection).
            correlation: Correlation between concentrations and targets (0-1).
            noise: Noise level to add.
            transform: Optional transformation ('log', 'sqrt').

        Returns:
            Target values array. Shape (n_samples,) for single target,
            or (n_samples, n_targets) for multi-output.

        Example:
            >>> y = generator.regression(
            ...     100, C,
            ...     distribution="lognormal",
            ...     range=(5, 50),
            ...     component="protein",
            ...     component_names=["water", "protein", "lipid"]
            ... )
        """
        # Generate base values from concentrations or random
        if concentrations is not None:
            base = self._concentrations_to_base(
                concentrations, component, component_names
            )
        else:
            base = self.rng.uniform(0, 1, size=n_samples)
            if range is not None:
                base = base.reshape(-1, 1) if base.ndim == 1 else base

        # Apply distribution transformation
        y = self._apply_distribution(base, distribution)

        # Scale to range
        if range is not None:
            y = self._scale_to_range(y, range)

        # Add noise (maintaining correlation)
        if noise > 0 and correlation < 1.0:
            y = self._add_controlled_noise(y, correlation, noise)

        # Apply optional transformation
        if transform == "log":
            y = np.log1p(np.maximum(y, 0))
        elif transform == "sqrt":
            y = np.sqrt(np.maximum(y, 0))

        # Flatten if single target
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        return y

    def classification(
        self,
        n_samples: int,
        concentrations: Optional[np.ndarray] = None,
        *,
        n_classes: int = 2,
        class_weights: Optional[List[float]] = None,
        separation: float = 1.5,
        separation_method: Literal["component", "threshold", "cluster"] = "component",
        class_names: Optional[List[str]] = None,
        return_proba: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate classification target labels with controllable class separation.

        The separation parameter controls how distinguishable classes are in
        feature space. Higher values create more separable classes.

        Args:
            n_samples: Number of samples.
            concentrations: Component concentration matrix.
            n_classes: Number of classes to generate.
            class_weights: Class proportions (should sum to 1.0).
                If None, uses balanced classes.
            separation: Class separation factor:
                - 0.5-1.0: Overlapping classes (challenging)
                - 1.5-2.0: Moderate separation (realistic)
                - 2.5+: Well-separated classes (easy)
            separation_method: How to create class differences:
                - "component": Each class has distinct component profiles
                - "threshold": Classes based on concentration thresholds
                - "cluster": K-means-like cluster assignment
            class_names: Optional string labels for classes.
            return_proba: If True, also return class probabilities.

        Returns:
            If return_proba=False: Integer class labels (n_samples,).
            If return_proba=True: Tuple of (labels, probabilities).

        Example:
            >>> # Binary classification with balanced classes
            >>> y = generator.classification(100, C, n_classes=2)
            >>>
            >>> # 3-class with imbalanced weights
            >>> y = generator.classification(
            ...     100, C,
            ...     n_classes=3,
            ...     class_weights=[0.5, 0.3, 0.2],
            ...     separation=2.0
            ... )
        """
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        if class_weights is not None:
            if len(class_weights) != n_classes:
                raise ValueError(
                    f"class_weights length ({len(class_weights)}) must match "
                    f"n_classes ({n_classes})"
                )
            if abs(sum(class_weights) - 1.0) > 0.01:
                raise ValueError(
                    f"class_weights must sum to 1.0, got {sum(class_weights)}"
                )

        # Generate class labels based on method
        if separation_method == "component":
            labels, proba = self._classify_by_component_profile(
                n_samples, concentrations, n_classes, class_weights, separation
            )
        elif separation_method == "threshold":
            labels, proba = self._classify_by_threshold(
                n_samples, concentrations, n_classes, class_weights, separation
            )
        elif separation_method == "cluster":
            labels, proba = self._classify_by_clustering(
                n_samples, concentrations, n_classes, class_weights, separation
            )
        else:
            raise ValueError(f"Unknown separation_method: '{separation_method}'")

        if return_proba:
            return labels, proba
        return labels

    def _concentrations_to_base(
        self,
        concentrations: np.ndarray,
        component: Optional[Union[int, str, List[int]]],
        component_names: Optional[List[str]],
    ) -> np.ndarray:
        """Extract base values from concentration matrix."""
        if component is None:
            # Weighted combination of all components
            weights = self.rng.dirichlet(np.ones(concentrations.shape[1]))
            return concentrations @ weights
        elif isinstance(component, str):
            if component_names is None:
                raise ValueError(
                    "component_names required when component is specified as string"
                )
            idx = component_names.index(component)
            return concentrations[:, idx]
        elif isinstance(component, int):
            return concentrations[:, component]
        elif isinstance(component, list):
            return concentrations[:, component]
        else:
            raise ValueError(f"Invalid component specification: {component}")

    def _apply_distribution(
        self,
        base: np.ndarray,
        distribution: str,
    ) -> np.ndarray:
        """Transform base values to target distribution."""
        if base.ndim == 1:
            base = base.reshape(-1, 1)

        n_samples, n_targets = base.shape

        if distribution == "uniform":
            # Already uniform - just ensure range [0, 1]
            return base / base.max(axis=0, keepdims=True)

        elif distribution == "normal":
            # Transform to approximate normal via inverse CDF
            # Rank transform to uniform, then to normal
            result = np.zeros_like(base)
            for j in range(n_targets):
                ranks = stats.rankdata(base[:, j]) / (n_samples + 1)
                result[:, j] = stats.norm.ppf(ranks)
            return result

        elif distribution == "lognormal":
            # Log-normal: positively skewed
            # Ensure positive base values
            base_pos = np.maximum(base, 1e-10)
            return np.exp(base_pos * 2 - 1)  # Scale and shift

        elif distribution == "bimodal":
            # Create bimodal distribution
            result = np.zeros_like(base)
            for j in range(n_targets):
                # Split samples into two modes
                mid = np.median(base[:, j])
                low_mask = base[:, j] <= mid
                high_mask = ~low_mask

                # Shift modes apart
                result[low_mask, j] = base[low_mask, j] * 0.5
                result[high_mask, j] = base[high_mask, j] * 0.5 + 0.5
            return result

        else:
            raise ValueError(f"Unknown distribution: '{distribution}'")

    def _scale_to_range(
        self,
        y: np.ndarray,
        range: Tuple[float, float],
    ) -> np.ndarray:
        """Scale values to specified range."""
        min_val, max_val = range

        # Handle edge case of constant values
        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-10:
            return np.full_like(y, (min_val + max_val) / 2)

        # Linear scaling
        return (y - y_min) / (y_max - y_min) * (max_val - min_val) + min_val

    def _add_controlled_noise(
        self,
        y: np.ndarray,
        target_correlation: float,
        noise_std: float,
    ) -> np.ndarray:
        """Add noise while maintaining target correlation with original values."""
        # The correlation controls how much of the signal vs noise
        # Higher correlation = less noise influence
        signal_weight = target_correlation
        noise_weight = np.sqrt(1 - target_correlation**2)

        noise = self.rng.normal(0, noise_std, size=y.shape)
        y_noisy = signal_weight * y + noise_weight * noise * np.std(y)

        return y_noisy

    def _classify_by_component_profile(
        self,
        n_samples: int,
        concentrations: Optional[np.ndarray],
        n_classes: int,
        class_weights: Optional[List[float]],
        separation: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify samples based on component concentration profiles.

        Each class is characterized by a different dominant component or
        combination of components.
        """
        if concentrations is None:
            # Generate random concentrations
            concentrations = self.rng.dirichlet(
                np.ones(n_classes), size=n_samples
            )

        n_components = concentrations.shape[1]

        # Create class centroids in component space
        # Each class emphasizes different components
        centroids = np.zeros((n_classes, n_components))
        for c in range(n_classes):
            # Primary component for this class
            primary = c % n_components
            centroids[c, primary] = separation

            # Add some values to other components
            for i in range(n_components):
                if i != primary:
                    centroids[c, i] = self.rng.uniform(0, 0.3)

        # Compute distances to each centroid
        distances = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            diff = concentrations - centroids[c]
            distances[:, c] = np.sqrt((diff ** 2).sum(axis=1))

        # Convert distances to probabilities (inverse distance weighting)
        inv_dist = 1 / (distances + 1e-10)
        proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        # Apply class weights by adjusting probabilities
        if class_weights is not None:
            weights = np.array(class_weights)
            proba = proba * weights
            proba = proba / proba.sum(axis=1, keepdims=True)

        # Assign labels (with some randomness based on separation)
        if separation >= 2.0:
            # High separation - deterministic assignment
            labels = proba.argmax(axis=1)
        else:
            # Lower separation - probabilistic assignment
            labels = np.array([
                self.rng.choice(n_classes, p=p) for p in proba
            ])

        return labels.astype(np.int32), proba

    def _classify_by_threshold(
        self,
        n_samples: int,
        concentrations: Optional[np.ndarray],
        n_classes: int,
        class_weights: Optional[List[float]],
        separation: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify samples using concentration thresholds.

        Classes are assigned based on whether key concentrations are above
        or below certain thresholds.
        """
        if concentrations is None:
            concentrations = self.rng.uniform(0, 1, size=(n_samples, n_classes))

        # Use first component for thresholding
        values = concentrations[:, 0]

        # Determine thresholds based on class weights
        if class_weights is None:
            # Uniform thresholds
            percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        else:
            # Weighted thresholds
            cumsum = np.cumsum(class_weights[:-1])
            percentiles = cumsum * 100

        thresholds = [np.percentile(values, p) for p in percentiles]

        # Add noise to thresholds based on separation
        threshold_noise = (1 - separation / 3) * np.std(values) * 0.5

        # Assign labels
        labels = np.zeros(n_samples, dtype=np.int32)
        for i, threshold in enumerate(thresholds):
            noisy_threshold = threshold + self.rng.normal(0, threshold_noise)
            labels[values > noisy_threshold] = i + 1

        # Compute approximate probabilities
        proba = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            proba[labels == c, c] = 0.8 + self.rng.uniform(0, 0.2, size=(labels == c).sum())
            for other in range(n_classes):
                if other != c:
                    proba[labels == c, other] = self.rng.uniform(
                        0, 0.2 / (n_classes - 1), size=(labels == c).sum()
                    )
        proba = proba / proba.sum(axis=1, keepdims=True)

        return labels, proba

    def _classify_by_clustering(
        self,
        n_samples: int,
        concentrations: Optional[np.ndarray],
        n_classes: int,
        class_weights: Optional[List[float]],
        separation: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify samples using k-means-like clustering in component space.
        """
        if concentrations is None:
            concentrations = self.rng.uniform(0, 1, size=(n_samples, 5))

        # Simple k-means-like assignment
        # Initialize centroids
        n_components = concentrations.shape[1]
        centroid_indices = self.rng.choice(
            n_samples, size=n_classes, replace=False
        )
        centroids = concentrations[centroid_indices].copy()

        # Spread centroids based on separation
        for i in range(n_classes):
            direction = self.rng.normal(0, 1, size=n_components)
            direction = direction / np.linalg.norm(direction)
            centroids[i] += direction * separation * 0.2

        # Assign to nearest centroid
        distances = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            distances[:, c] = np.sqrt(
                ((concentrations - centroids[c]) ** 2).sum(axis=1)
            )

        labels = distances.argmin(axis=1)

        # Adjust for class weights if specified
        if class_weights is not None:
            # Re-balance assignments
            labels = self._rebalance_labels(labels, n_classes, class_weights)

        # Compute probabilities from distances
        inv_dist = 1 / (distances + 1e-10)
        proba = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        return labels.astype(np.int32), proba

    def _rebalance_labels(
        self,
        labels: np.ndarray,
        n_classes: int,
        class_weights: List[float],
    ) -> np.ndarray:
        """Rebalance label distribution to match target weights."""
        n_samples = len(labels)
        target_counts = [int(w * n_samples) for w in class_weights]

        # Adjust to ensure sum equals n_samples
        diff = n_samples - sum(target_counts)
        for i in range(abs(diff)):
            target_counts[i % n_classes] += 1 if diff > 0 else -1

        # Reassign labels to match target counts
        new_labels = labels.copy()
        current_counts = [np.sum(labels == c) for c in range(n_classes)]

        for c in range(n_classes):
            excess = current_counts[c] - target_counts[c]
            if excess > 0:
                # Move excess samples to other classes
                class_samples = np.where(labels == c)[0]
                samples_to_move = self.rng.choice(
                    class_samples, size=excess, replace=False
                )
                # Find classes that need more samples
                for sample_idx in samples_to_move:
                    for other_c in range(n_classes):
                        if current_counts[other_c] < target_counts[other_c]:
                            new_labels[sample_idx] = other_c
                            current_counts[c] -= 1
                            current_counts[other_c] += 1
                            break

        return new_labels


def generate_regression_targets(
    n_samples: int,
    concentrations: Optional[np.ndarray] = None,
    *,
    random_state: Optional[int] = None,
    distribution: str = "uniform",
    range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Convenience function for generating regression targets.

    Args:
        n_samples: Number of samples.
        concentrations: Component concentrations (optional).
        random_state: Random seed.
        distribution: Target distribution type.
        range: Value range (min, max).

    Returns:
        Target values array.
    """
    generator = TargetGenerator(random_state=random_state)
    return generator.regression(
        n_samples=n_samples,
        concentrations=concentrations,
        distribution=distribution,
        range=range,
    )


def generate_classification_targets(
    n_samples: int,
    concentrations: Optional[np.ndarray] = None,
    *,
    random_state: Optional[int] = None,
    n_classes: int = 2,
    class_weights: Optional[List[float]] = None,
    separation: float = 1.5,
) -> np.ndarray:
    """
    Convenience function for generating classification targets.

    Args:
        n_samples: Number of samples.
        concentrations: Component concentrations (optional).
        random_state: Random seed.
        n_classes: Number of classes.
        class_weights: Class proportions.
        separation: Class separation factor.

    Returns:
        Integer class labels array.
    """
    generator = TargetGenerator(random_state=random_state)
    return generator.classification(
        n_samples=n_samples,
        concentrations=concentrations,
        n_classes=n_classes,
        class_weights=class_weights,
        separation=separation,
    )


@dataclass
class NonLinearTargetConfig:
    """
    Configuration for non-linear target complexity.

    Attributes:
        nonlinear_interactions: Type of non-linear interaction.
        interaction_strength: Blend factor (0=linear, 1=fully non-linear).
        hidden_factors: Latent variables not in spectra.
        polynomial_degree: Degree for polynomial interactions.
        signal_to_confound_ratio: Predictability from spectra.
        n_confounders: Confounding variables.
        spectral_masking: Signal in noisy regions.
        temporal_drift: Relationship changes over samples.
        n_regimes: Number of relationship regimes.
        regime_method: How to partition into regimes.
        regime_overlap: Transition zone smoothness.
        noise_heteroscedasticity: Per-regime noise variation.
    """

    # Proposition 1: Non-linear interactions
    nonlinear_interactions: Literal["none", "polynomial", "synergistic", "antagonistic"] = "none"
    interaction_strength: float = 0.5
    hidden_factors: int = 0
    polynomial_degree: int = 2

    # Proposition 2: Confounders
    signal_to_confound_ratio: float = 1.0
    n_confounders: int = 0
    spectral_masking: float = 0.0
    temporal_drift: bool = False

    # Proposition 3: Multi-regime
    n_regimes: int = 1
    regime_method: Literal["concentration", "spectral", "random"] = "concentration"
    regime_overlap: float = 0.2
    noise_heteroscedasticity: float = 0.0


class NonLinearTargetProcessor:
    """
    Process targets with non-linear relationships, confounders, and multi-regime landscapes.

    This class implements three propositions for making synthetic targets harder to predict:

    1. **Non-linear interactions**: Polynomial, synergistic, or antagonistic effects.
    2. **Spectral-target decoupling**: Confounders and partial predictability.
    3. **Multi-regime landscapes**: Different relationships in different regions.

    Args:
        config: NonLinearTargetConfig with all settings.
        random_state: Random seed for reproducibility.

    Example:
        >>> config = NonLinearTargetConfig(
        ...     nonlinear_interactions="polynomial",
        ...     interaction_strength=0.7,
        ...     n_regimes=3
        ... )
        >>> processor = NonLinearTargetProcessor(config, random_state=42)
        >>> y_complex = processor.process(C, y_base)
    """

    def __init__(
        self,
        config: NonLinearTargetConfig,
        random_state: Optional[int] = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(random_state)
        self._random_state = random_state

    def process(
        self,
        concentrations: np.ndarray,
        y_base: np.ndarray,
        spectra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply all configured complexity to base targets.

        Args:
            concentrations: Component concentration matrix (n_samples, n_components).
            y_base: Base target values (n_samples,) or (n_samples, n_targets).
            spectra: Optional spectra matrix for spectral-based regimes.

        Returns:
            Transformed target values with added complexity.
        """
        y = y_base.copy()
        n_samples = len(y)

        # Ensure 1D for processing
        was_1d = y.ndim == 1
        if was_1d:
            y = y.reshape(-1, 1)

        # Step 1: Apply non-linear interactions
        if self.config.nonlinear_interactions != "none":
            y = self._apply_nonlinear_interactions(concentrations, y)

        # Step 2: Add hidden factors (unpredictable component)
        if self.config.hidden_factors > 0:
            y = self._add_hidden_factors(y)

        # Step 3: Apply multi-regime transformation
        if self.config.n_regimes > 1:
            y = self._apply_multi_regime(concentrations, y, spectra)

        # Step 4: Apply confounders
        if self.config.n_confounders > 0:
            y = self._apply_confounders(concentrations, y)

        # Step 5: Apply temporal drift
        if self.config.temporal_drift:
            y = self._apply_temporal_drift(y)

        # Step 6: Apply signal-to-confound ratio (add irreducible noise)
        if self.config.signal_to_confound_ratio < 1.0:
            y = self._apply_signal_ratio(y)

        # Step 7: Apply heteroscedastic noise
        if self.config.noise_heteroscedasticity > 0:
            y = self._apply_heteroscedastic_noise(y, concentrations)

        # Restore original shape
        if was_1d and y.shape[1] == 1:
            y = y.ravel()

        return y

    def _apply_nonlinear_interactions(
        self,
        C: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Apply non-linear interaction terms."""
        n_samples, n_components = C.shape
        strength = self.config.interaction_strength

        if self.config.nonlinear_interactions == "polynomial":
            # Generate polynomial features
            y_nonlinear = self._polynomial_transform(C, y)

        elif self.config.nonlinear_interactions == "synergistic":
            # Synergistic: combinations enhance effect
            y_nonlinear = self._synergistic_transform(C, y)

        elif self.config.nonlinear_interactions == "antagonistic":
            # Antagonistic: saturation/inhibition effects
            y_nonlinear = self._antagonistic_transform(C, y)

        else:
            return y

        # Blend linear and non-linear based on strength
        y_blended = (1 - strength) * y + strength * y_nonlinear

        return y_blended

    def _polynomial_transform(
        self,
        C: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Generate polynomial interaction terms."""
        n_samples, n_components = C.shape
        degree = self.config.polynomial_degree

        # Normalize concentrations to avoid numerical issues
        C_norm = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-8)

        # Build polynomial features
        poly_terms = []

        # Quadratic terms: C_i^2
        for i in range(n_components):
            poly_terms.append(C_norm[:, i] ** 2)

        # Interaction terms: C_i * C_j
        for i in range(n_components):
            for j in range(i + 1, n_components):
                poly_terms.append(C_norm[:, i] * C_norm[:, j])

        # Cubic terms if degree >= 3
        if degree >= 3:
            for i in range(n_components):
                poly_terms.append(C_norm[:, i] ** 3)
            for i in range(min(n_components, 3)):
                for j in range(i + 1, min(n_components, 3)):
                    poly_terms.append(C_norm[:, i] ** 2 * C_norm[:, j])

        # Combine terms with random weights
        poly_features = np.column_stack(poly_terms) if poly_terms else np.zeros((n_samples, 1))
        weights = self.rng.standard_normal(poly_features.shape[1])
        weights = weights / np.linalg.norm(weights)  # Normalize

        # Combine with base y
        nonlinear_component = poly_features @ weights
        nonlinear_component = nonlinear_component.reshape(-1, 1)

        # Scale to match y range
        y_range = y.max() - y.min()
        if y_range > 0:
            nonlinear_component = (
                (nonlinear_component - nonlinear_component.mean())
                / (nonlinear_component.std() + 1e-8)
                * y.std()
                + y.mean()
            )

        return nonlinear_component if y.shape[1] == 1 else np.tile(nonlinear_component, (1, y.shape[1]))

    def _synergistic_transform(
        self,
        C: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Synergistic effects: combinations enhance the target non-linearly."""
        n_samples, n_components = C.shape

        # Select pairs of components for synergy
        n_pairs = min(3, n_components * (n_components - 1) // 2)
        synergy_terms = []

        for _ in range(n_pairs):
            i, j = self.rng.choice(n_components, 2, replace=False)
            # Synergistic term: sqrt(C_i * C_j) * (C_i + C_j)
            term = np.sqrt(C[:, i] * C[:, j] + 1e-8) * (C[:, i] + C[:, j])
            synergy_terms.append(term)

        if synergy_terms:
            synergy = np.column_stack(synergy_terms)
            weights = self.rng.uniform(0.5, 1.5, len(synergy_terms))
            synergy_effect = (synergy * weights).sum(axis=1, keepdims=True)

            # Scale and add to y
            synergy_effect = (
                (synergy_effect - synergy_effect.mean())
                / (synergy_effect.std() + 1e-8)
                * y.std()
            )
            return y + synergy_effect

        return y

    def _antagonistic_transform(
        self,
        C: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Antagonistic effects: saturation and inhibition (Michaelis-Menten-like)."""
        n_samples, n_components = C.shape

        # Apply Michaelis-Menten kinetics to primary component
        primary_idx = self.rng.integers(0, n_components)
        C_primary = C[:, primary_idx]

        # Vmax and Km parameters
        Vmax = y.max() * 1.2
        Km = np.median(C_primary) * 0.5

        # Michaelis-Menten: V = Vmax * [S] / (Km + [S])
        mm_term = Vmax * C_primary / (Km + C_primary + 1e-8)

        # Optional competitive inhibition from another component
        if n_components > 1:
            inhibitor_idx = (primary_idx + 1) % n_components
            Ki = np.median(C[:, inhibitor_idx])
            inhibition_factor = 1 / (1 + C[:, inhibitor_idx] / (Ki + 1e-8))
            mm_term = mm_term * inhibition_factor

        mm_term = mm_term.reshape(-1, 1)

        # Scale to match y
        mm_term = (
            (mm_term - mm_term.mean()) / (mm_term.std() + 1e-8) * y.std() + y.mean()
        )

        return mm_term if y.shape[1] == 1 else np.tile(mm_term, (1, y.shape[1]))

    def _add_hidden_factors(self, y: np.ndarray) -> np.ndarray:
        """Add latent factors that affect y but have no spectral signature."""
        n_samples = y.shape[0]
        n_hidden = self.config.hidden_factors

        # Generate hidden factors
        hidden = self.rng.standard_normal((n_samples, n_hidden))

        # Random weights for hidden factor contribution
        weights = self.rng.uniform(0.1, 0.3, n_hidden)

        # Hidden effect (unobservable from spectra)
        hidden_effect = (hidden * weights).sum(axis=1, keepdims=True)
        hidden_effect = hidden_effect / (hidden_effect.std() + 1e-8) * y.std() * 0.3

        return y + hidden_effect

    def _apply_multi_regime(
        self,
        C: np.ndarray,
        y: np.ndarray,
        spectra: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply different target functions in different regimes."""
        n_samples = y.shape[0]
        n_regimes = self.config.n_regimes

        # Assign samples to regimes
        regime_assignments = self._assign_regimes(C, spectra)

        # Generate different transformation functions per regime
        y_transformed = np.zeros_like(y)

        for regime_id in range(n_regimes):
            mask = regime_assignments == regime_id
            if not np.any(mask):
                continue

            # Each regime has a different relationship
            y_regime = y[mask].copy()
            C_regime = C[mask]

            # Regime-specific transformation
            if regime_id == 0:
                # Linear (baseline)
                y_regime = y_regime
            elif regime_id == 1:
                # Quadratic emphasis
                if C_regime.shape[1] > 0:
                    quad_term = (C_regime[:, 0] ** 2).reshape(-1, 1)
                    quad_term = (quad_term - quad_term.mean()) / (quad_term.std() + 1e-8)
                    y_regime = y_regime + quad_term * y_regime.std() * 0.5
            elif regime_id == 2:
                # Inverse relationship
                y_regime = y_regime.max() + y_regime.min() - y_regime
            else:
                # Random non-linear mixing
                component_idx = regime_id % C_regime.shape[1]
                ratio_term = C_regime[:, component_idx] / (
                    C_regime[:, (component_idx + 1) % C_regime.shape[1]] + 1e-8
                )
                ratio_term = ratio_term.reshape(-1, 1)
                ratio_term = (ratio_term - ratio_term.mean()) / (ratio_term.std() + 1e-8)
                y_regime = y_regime + ratio_term * y_regime.std() * 0.3

            y_transformed[mask] = y_regime

        # Apply overlap smoothing
        if self.config.regime_overlap > 0:
            y_transformed = self._smooth_regime_boundaries(
                y_transformed, regime_assignments, y
            )

        return y_transformed

    def _assign_regimes(
        self,
        C: np.ndarray,
        spectra: Optional[np.ndarray],
    ) -> np.ndarray:
        """Assign samples to regimes based on method."""
        n_samples = C.shape[0]
        n_regimes = self.config.n_regimes

        if self.config.regime_method == "random":
            return self.rng.integers(0, n_regimes, size=n_samples)

        elif self.config.regime_method == "concentration":
            # Use first principal direction of concentrations
            C_centered = C - C.mean(axis=0)
            if C.shape[1] > 1:
                # Simple projection onto first component
                projection = C_centered.sum(axis=1)
            else:
                projection = C_centered[:, 0]

            # Quantile-based assignment
            percentiles = np.linspace(0, 100, n_regimes + 1)
            thresholds = [np.percentile(projection, p) for p in percentiles]
            assignments = np.zeros(n_samples, dtype=int)
            for i in range(n_regimes - 1):
                assignments[projection > thresholds[i + 1]] = i + 1
            return assignments

        elif self.config.regime_method == "spectral":
            if spectra is None:
                # Fall back to concentration-based
                return self._assign_regimes(C, None)

            # Use spectral intensity for regime assignment
            intensity = spectra.mean(axis=1)
            percentiles = np.linspace(0, 100, n_regimes + 1)
            thresholds = [np.percentile(intensity, p) for p in percentiles]
            assignments = np.zeros(n_samples, dtype=int)
            for i in range(n_regimes - 1):
                assignments[intensity > thresholds[i + 1]] = i + 1
            return assignments

        return np.zeros(n_samples, dtype=int)

    def _smooth_regime_boundaries(
        self,
        y_transformed: np.ndarray,
        regime_assignments: np.ndarray,
        y_original: np.ndarray,
    ) -> np.ndarray:
        """Smooth transitions between regimes."""
        # Simple implementation: blend with original at boundaries
        overlap = self.config.regime_overlap
        n_regimes = self.config.n_regimes

        # For each regime boundary, create a soft transition
        # This is a simplified version - more sophisticated approaches possible
        blend_factor = np.ones((y_transformed.shape[0], 1))

        for regime in range(n_regimes):
            mask = regime_assignments == regime
            if np.sum(mask) == 0:
                continue

            # Reduce blend at edges (simple approach)
            edge_samples = int(np.sum(mask) * overlap)
            if edge_samples > 0:
                regime_indices = np.where(mask)[0]
                for idx in regime_indices[:edge_samples]:
                    blend_factor[idx] = 0.5 + 0.5 * (
                        np.where(regime_indices == idx)[0][0] / edge_samples
                    )

        return blend_factor * y_transformed + (1 - blend_factor) * y_original

    def _apply_confounders(
        self,
        C: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Apply confounding variables that affect spectra and target differently."""
        n_samples = y.shape[0]
        n_confounders = self.config.n_confounders

        # Generate confounders correlated with concentrations
        confounder_effect = np.zeros((n_samples, 1))

        for i in range(n_confounders):
            # Each confounder is partially correlated with a component
            component_idx = i % C.shape[1]
            base_confounder = 0.5 * C[:, component_idx] + 0.5 * self.rng.standard_normal(n_samples)

            # Non-linear transformation of confounder effect on y
            # (different from its effect on spectra which is linear)
            effect = np.sin(base_confounder * np.pi) + base_confounder ** 2
            effect = effect.reshape(-1, 1)
            effect = (effect - effect.mean()) / (effect.std() + 1e-8)

            confounder_effect += effect * 0.2 / n_confounders

        return y + confounder_effect * y.std()

    def _apply_temporal_drift(self, y: np.ndarray) -> np.ndarray:
        """Apply temporal drift - relationship changes over samples."""
        n_samples = y.shape[0]

        # Time index (assume samples are in temporal order)
        t = np.linspace(0, 1, n_samples).reshape(-1, 1)

        # Drift function: gradual shift and scale change
        drift_shift = 0.2 * np.sin(2 * np.pi * t)  # Oscillating shift
        drift_scale = 1 + 0.15 * t  # Gradual scale increase

        y_drifted = (y - y.mean()) * drift_scale + y.mean() + drift_shift * y.std()

        return y_drifted

    def _apply_signal_ratio(self, y: np.ndarray) -> np.ndarray:
        """Add irreducible noise based on signal-to-confound ratio."""
        n_samples = y.shape[0]
        ratio = self.config.signal_to_confound_ratio

        # Calculate noise variance to achieve target ratio
        # If ratio = 0.7, then 30% of variance is unexplainable
        unexplainable_var = (1 - ratio) * np.var(y)
        noise_std = np.sqrt(unexplainable_var)

        noise = self.rng.standard_normal(y.shape) * noise_std

        return y + noise

    def _apply_heteroscedastic_noise(
        self,
        y: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        """Apply noise that varies based on concentration/regime."""
        n_samples = y.shape[0]
        hetero_strength = self.config.noise_heteroscedasticity

        # Noise level depends on first concentration component
        if C.shape[1] > 0:
            noise_scale = 1 + hetero_strength * (C[:, 0] - C[:, 0].mean()) / (C[:, 0].std() + 1e-8)
            noise_scale = np.clip(noise_scale, 0.2, 3.0).reshape(-1, 1)
        else:
            noise_scale = np.ones((n_samples, 1))

        base_noise = self.rng.standard_normal(y.shape) * y.std() * 0.1
        heteroscedastic_noise = base_noise * noise_scale

        return y + heteroscedastic_noise

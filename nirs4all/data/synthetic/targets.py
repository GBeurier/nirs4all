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

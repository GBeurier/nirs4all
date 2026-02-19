"""
Configuration dataclasses for synthetic NIRS data generation.

This module provides structured configuration objects for controlling
various aspects of synthetic spectra generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union


@dataclass
class FeatureConfig:
    """
    Configuration for spectral feature generation.

    Attributes:
        wavelength_start: Start wavelength in nm.
        wavelength_end: End wavelength in nm.
        wavelength_step: Wavelength step in nm.
        complexity: Complexity level affecting noise, scatter, etc.
            Options: 'simple', 'realistic', 'complex'.
        n_components: Number of spectral components (auto if None).
        component_names: Specific predefined components to use.
            If None, uses default components based on complexity.
    """

    wavelength_start: float = 1000.0
    wavelength_end: float = 2500.0
    wavelength_step: float = 2.0
    complexity: Literal["simple", "realistic", "complex"] = "simple"
    n_components: int | None = None
    component_names: list[str] | None = None

@dataclass
class TargetConfig:
    """
    Configuration for target variable generation.

    Attributes:
        distribution: Target value distribution method.
            Options: 'dirichlet', 'uniform', 'lognormal', 'correlated'.
        range: Optional (min, max) range for scaling targets.
        n_targets: Number of target variables (auto from components if None).
        component_indices: Which components to use as targets (all if None).
        transform: Optional transformation to apply ('log', 'sqrt', None).
    """

    distribution: Literal["dirichlet", "uniform", "lognormal", "correlated"] = "dirichlet"
    range: tuple[float, float] | None = None
    n_targets: int | None = None
    component_indices: list[int] | None = None
    transform: Literal["log", "sqrt"] | None = None

@dataclass
class MetadataConfig:
    """
    Configuration for sample metadata generation.

    Attributes:
        generate_sample_ids: Whether to generate sample IDs.
        sample_id_prefix: Prefix for sample IDs.
        n_groups: Number of sample groups (e.g., biological replicates).
        n_repetitions: Repetitions per sample, either fixed int or (min, max) range.
        group_names: Optional list of group names.
        additional_columns: Dict of column_name -> generator function or values.
    """

    generate_sample_ids: bool = True
    sample_id_prefix: str = "sample"
    n_groups: int | None = None
    n_repetitions: int | tuple[int, int] = 1
    group_names: list[str] | None = None
    additional_columns: dict[str, Any] | None = None

@dataclass
class PartitionConfig:
    """
    Configuration for data partitioning (train/test split).

    Attributes:
        train_ratio: Proportion of samples for training (0.0-1.0).
        stratify: Whether to stratify by target (for classification).
        shuffle: Whether to shuffle before splitting.
        group_aware: Whether to keep groups together when splitting.
    """

    train_ratio: float = 0.8
    stratify: bool = False
    shuffle: bool = True
    group_aware: bool = True

@dataclass
class BatchEffectConfig:
    """
    Configuration for batch/session effects simulation.

    Attributes:
        enabled: Whether to add batch effects.
        n_batches: Number of measurement batches/sessions.
        offset_std: Standard deviation of batch offset.
        gain_std: Standard deviation of batch gain multiplier.
    """

    enabled: bool = False
    n_batches: int = 3
    offset_std: float = 0.02
    gain_std: float = 0.03

@dataclass
class NonLinearConfig:
    """
    Configuration for non-linear target relationships.

    Enables polynomial, synergistic, or antagonistic interactions between
    component concentrations and targets, making prediction harder.

    Attributes:
        interactions: Type of non-linear interaction.
            Options: 'none', 'polynomial', 'synergistic', 'antagonistic'.
        interaction_strength: Blend factor (0 = linear, 1 = fully non-linear).
        hidden_factors: Number of latent variables affecting target but not spectra.
        polynomial_degree: Degree for polynomial interactions (2 or 3).
    """

    interactions: Literal["none", "polynomial", "synergistic", "antagonistic"] = "none"
    interaction_strength: float = 0.5
    hidden_factors: int = 0
    polynomial_degree: int = 2

@dataclass
class ConfounderConfig:
    """
    Configuration for spectral-target decoupling and confounding effects.

    Introduces factors that make the target only partially predictable
    from spectral features, simulating real-world irreducible error.

    Attributes:
        signal_to_confound_ratio: Proportion of target variance explainable
            from spectra. 1.0 = fully predictable, 0.5 = 50% unexplainable.
        n_confounders: Number of confounding variables that affect both
            spectra and target in different ways.
        spectral_masking: Fraction of predictive signal hidden in high-noise
            wavelength regions (0.0-0.5).
        temporal_drift: If True, the target-spectra relationship gradually
            changes across samples.
    """

    signal_to_confound_ratio: float = 1.0
    n_confounders: int = 0
    spectral_masking: float = 0.0
    temporal_drift: bool = False

@dataclass
class MultiRegimeConfig:
    """
    Configuration for multi-regime target landscapes.

    Creates regions in feature space where the target-spectra relationship
    differs, simulating subpopulations.

    Attributes:
        n_regimes: Number of different relationship regimes.
        regime_method: How to partition samples into regimes:
            'concentration', 'spectral', or 'random'.
        regime_overlap: Overlap between regimes creating transition zones.
            0 = hard boundaries, 0.5 = smooth transitions.
        noise_heteroscedasticity: How much prediction noise varies by regime.
            0 = same noise everywhere, 1 = very different noise levels.
    """

    n_regimes: int = 1
    regime_method: Literal["concentration", "spectral", "random"] = "concentration"
    regime_overlap: float = 0.2
    noise_heteroscedasticity: float = 0.0

@dataclass
class OutputConfig:
    """
    Configuration for output format.

    Attributes:
        as_dataset: Whether to return SpectroDataset (vs tuple).
        include_metadata: Whether to include generation metadata.
        include_wavelengths: Whether to include wavelength array in output.
    """

    as_dataset: bool = True
    include_metadata: bool = False
    include_wavelengths: bool = True

@dataclass
class SyntheticDatasetConfig:
    """
    Complete configuration for synthetic dataset generation.

    This is the main configuration object that combines all sub-configurations
    for generating synthetic NIRS datasets.

    Attributes:
        n_samples: Total number of samples to generate.
        random_state: Random seed for reproducibility.
        features: Feature generation configuration.
        targets: Target variable configuration.
        metadata: Sample metadata configuration.
        partitions: Train/test split configuration.
        batch_effects: Batch effect configuration.
        output: Output format configuration.
        name: Optional dataset name.

    Example:
        >>> config = SyntheticDatasetConfig(
        ...     n_samples=1000,
        ...     random_state=42,
        ...     features=FeatureConfig(complexity="realistic"),
        ...     targets=TargetConfig(distribution="lognormal", range=(0, 100)),
        ... )
    """

    n_samples: int = 1000
    random_state: int | None = None
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    partitions: PartitionConfig = field(default_factory=PartitionConfig)
    batch_effects: BatchEffectConfig = field(default_factory=BatchEffectConfig)
    nonlinear: NonLinearConfig = field(default_factory=NonLinearConfig)
    confounders: ConfounderConfig = field(default_factory=ConfounderConfig)
    multi_regime: MultiRegimeConfig = field(default_factory=MultiRegimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    name: str = "synthetic_nirs"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")

        if not 0.0 < self.partitions.train_ratio <= 1.0:
            raise ValueError(
                f"train_ratio must be in (0, 1], got {self.partitions.train_ratio}"
            )

        valid_complexities = ("simple", "realistic", "complex")
        if self.features.complexity not in valid_complexities:
            raise ValueError(
                f"complexity must be one of {valid_complexities}, "
                f"got '{self.features.complexity}'"
            )

# Convenience type alias for complexity levels
ComplexityLevel = Literal["simple", "realistic", "complex"]

# Convenience type alias for concentration methods
ConcentrationMethod = Literal["dirichlet", "uniform", "lognormal", "correlated"]

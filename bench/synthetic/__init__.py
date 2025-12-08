"""
Synthetic NIRS Spectra Generation Module
=========================================

This module provides tools for generating realistic synthetic NIRS spectra
for training autoencoders, testing preprocessing algorithms, and other ML applications.

Key Components:
- SyntheticNIRSGenerator: Main class for generating synthetic spectra
- ComponentLibrary: Predefined spectral components based on NIR band assignments
- Visualizer: Comprehensive visualization tools for generated data
- Comparator: Tools to compare synthetic spectra with real datasets

Example:
    >>> from examples.synthetic import SyntheticNIRSGenerator, plot_synthetic_spectra
    >>> generator = SyntheticNIRSGenerator(random_state=42)
    >>> X, Y, components = generator.generate(n_samples=1000)
    >>> plot_synthetic_spectra(X, Y, generator.wavelengths)

    # Compare with real data
    >>> from examples.synthetic import compare_with_real_data
    >>> comparator = compare_with_real_data(X, X_real, generator.wavelengths)
"""

from .generator import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    NIRBand,
    PREDEFINED_COMPONENTS,
)

from .visualizer import (
    plot_synthetic_spectra,
    plot_component_library,
    plot_concentration_distributions,
    plot_batch_effects,
    plot_noise_analysis,
    SyntheticSpectraVisualizer,
)

from .comparator import (
    SyntheticRealComparator,
    SpectralProperties,
    compute_spectral_properties,
    compare_with_real_data,
)

__all__ = [
    # Generator
    "SyntheticNIRSGenerator",
    "ComponentLibrary",
    "NIRBand",
    "PREDEFINED_COMPONENTS",
    # Visualization
    "plot_synthetic_spectra",
    "plot_component_library",
    "plot_concentration_distributions",
    "plot_batch_effects",
    "plot_noise_analysis",
    "SyntheticSpectraVisualizer",
    # Comparison
    "SyntheticRealComparator",
    "SpectralProperties",
    "compute_spectral_properties",
    "compare_with_real_data",
]

"""
Synthetic NIRS Spectra Generation Module.

.. deprecated:: 0.6.0
    This module has moved to :mod:`nirs4all.data.synthetic`.
    This location is kept for backward compatibility but will be removed in v1.0.

    Please update your imports::

        # Old (deprecated)
        from bench.synthetic import SyntheticNIRSGenerator

        # New
        from nirs4all.data.synthetic import SyntheticNIRSGenerator

        # Or use the top-level generate API (Phase 2)
        import nirs4all
        dataset = nirs4all.generate(n_samples=1000)

This module provides tools for generating realistic synthetic NIRS spectra
for training autoencoders, testing preprocessing algorithms, and other ML applications.

Key Components:
    - SyntheticNIRSGenerator: Main class for generating synthetic spectra
    - ComponentLibrary: Predefined spectral components based on NIR band assignments
    - Visualizer: Comprehensive visualization tools for generated data
    - Comparator: Tools to compare synthetic spectra with real datasets

Example:
    >>> from bench.synthetic import SyntheticNIRSGenerator, plot_synthetic_spectra
    >>> generator = SyntheticNIRSGenerator(random_state=42)
    >>> X, Y, components = generator.generate(n_samples=1000)
    >>> plot_synthetic_spectra(X, Y, generator.wavelengths)

    # Compare with real data
    >>> from bench.synthetic import compare_with_real_data
    >>> comparator = compare_with_real_data(X, X_real, generator.wavelengths)
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "The 'bench.synthetic' module is deprecated and will be removed in v1.0. "
    "Please use 'nirs4all.data.synthetic' instead. "
    "See the documentation for migration instructions.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    ComponentLibrary,
    NIRBand,
    SpectralComponent,
    PREDEFINED_COMPONENTS,
    get_predefined_components,
)

# Keep local visualizer and comparator imports (not yet migrated to main package)
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
    # Generator (from nirs4all.data.synthetic)
    "SyntheticNIRSGenerator",
    "ComponentLibrary",
    "NIRBand",
    "SpectralComponent",
    "PREDEFINED_COMPONENTS",
    "get_predefined_components",
    # Visualization (local, pending migration)
    "plot_synthetic_spectra",
    "plot_component_library",
    "plot_concentration_distributions",
    "plot_batch_effects",
    "plot_noise_analysis",
    "SyntheticSpectraVisualizer",
    # Comparison (local, pending migration)
    "SyntheticRealComparator",
    "SpectralProperties",
    "compute_spectral_properties",
    "compare_with_real_data",
]

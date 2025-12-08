"""
Benchmark and Research Tools for NIRS4All
==========================================

This module contains research and benchmarking tools organized into:

Subpackages:
    - preprocessing_selection: Systematic preprocessing selection framework
    - splitter_selection: Train/test splitting strategy comparison
    - synthetic: Synthetic NIRS spectra generation
    - tabpfn: TabPFN integration for NIRS data
    - models: Custom neural network architectures for benchmarking

Usage:
    # Preprocessing selection
    from bench.preprocessing_selection import PreprocessingSelector

    # Splitter selection
    from bench.splitter_selection import get_splitter

    # Synthetic data generation
    from bench.synthetic import SyntheticNIRSGenerator

    # TabPFN features
    from bench.tabpfn import SpectralLatentFeatures
"""

__all__ = [
    'preprocessing_selection',
    'splitter_selection',
    'synthetic',
    'tabpfn',
    'models',
]


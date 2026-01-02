#!/usr/bin/env python
"""
R06 - Synthetic Data Validation and Quality Assessment

Comprehensive reference for Phase 4 validation features of the
synthetic data generator.

Topics covered:

1. Spectral Realism Scorecard - Quantitative metrics to assess synthetic data
2. Adversarial Validation - Train classifier to distinguish real vs synthetic
3. Benchmark Dataset Matching - Generate data matching published datasets
4. Prior Sampling - Hierarchical configuration sampling
5. GPU Acceleration - Fast generation with JAX/CuPy

The realism scorecard evaluates synthetic data quality across six metrics:
- Correlation length overlap
- Derivative statistics (KS test)
- Peak density ratio
- Baseline curvature overlap
- SNR distribution match
- Adversarial validation AUC

Author: nirs4all team
Category: Reference - Synthetic
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np

# NIRS4All imports
from nirs4all.data.synthetic import (
    # Validation
    compute_spectral_realism_scorecard,
    quick_realism_check,
    compute_correlation_length,
    compute_peak_density,
    compute_snr,
    compute_adversarial_validation_auc,
    SpectralRealismScore,
    RealismMetric,

    # Benchmarks
    list_benchmark_datasets,
    get_benchmark_info,
    get_datasets_by_domain,
    get_benchmark_spectral_properties,
    BenchmarkDomain,

    # Prior sampling
    NIRSPriorConfig,
    PriorSampler,
    sample_prior,
    sample_prior_batch,

    # GPU acceleration
    AcceleratorBackend,
    AcceleratedGenerator,
    detect_best_backend,
    is_gpu_available,
    benchmark_backends,
    get_backend_info,

    # Core generator
    SyntheticNIRSGenerator,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def example_quick_realism_check():
    """
    Quick realism check for generated spectra.

    The quick check validates basic spectral properties without
    needing reference data - useful for catching obvious issues.
    """
    print_header("Quick Realism Check")

    # Generate some synthetic data
    generator = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=7.5,  # ~200 wavelengths
        random_state=42,
    )

    X, C, E = generator.generate(n_samples=100)

    # Run quick check
    passed, warnings = quick_realism_check(X, generator.wavelengths)

    print("Quick Realism Check Results:")
    print(f"  Passed: {passed}")
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  No warnings found.")

    # Demonstrate with bad data
    print("\n  Testing with corrupted data:")
    X_bad = X.copy()
    X_bad[0, :10] = np.inf  # Add some infinities

    passed_bad, warnings_bad = quick_realism_check(X_bad, generator.wavelengths)
    print(f"  Passed: {passed_bad} (expected: False)")
    if warnings_bad:
        print("  Warnings:")
        for w in warnings_bad:
            print(f"    - {w}")

    print("  ✓ Quick realism check complete")


def example_spectral_realism_scorecard():
    """
    Full spectral realism scorecard comparing real vs synthetic data.

    The scorecard computes 6 quantitative metrics that compare the
    statistical properties of synthetic spectra to reference data.
    """
    print_header("Spectral Realism Scorecard")

    # Generate "real" and "synthetic" data from same generator
    # (should score well since they're from the same distribution)
    generator = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        wavelength_step=5.0,  # ~300 wavelengths
        random_state=42,
    )

    # Generate reference "real" data
    X_real, _, _ = generator.generate(n_samples=100)

    # Generate "synthetic" data to evaluate
    X_synth, _, _ = generator.generate(n_samples=100)

    # Compute full scorecard
    score = compute_spectral_realism_scorecard(
        real_spectra=X_real,
        synthetic_spectra=X_synth,
        wavelengths=generator.wavelengths,
        include_adversarial=True,  # Include adversarial validation
        random_state=42,
    )

    print("Spectral Realism Scorecard:")
    print(f"  Correlation Length Overlap: {score.correlation_length_overlap:.3f}")
    print(f"  Derivative KS p-value: {score.derivative_ks_pvalue:.3f}")
    print(f"  Peak Density Ratio: {score.peak_density_ratio:.3f}")
    print(f"  Baseline Curvature Overlap: {score.baseline_curvature_overlap:.3f}")
    print(f"  SNR Match: {score.snr_magnitude_match}")
    print(f"  Adversarial AUC: {score.adversarial_auc:.3f}")
    print(f"\n  Overall Pass: {score.overall_pass}")

    # Show individual metric results
    print("\n  Metric Details:")
    for result in score.metric_results:
        status = "✓" if result.passed else "✗"
        print(f"    {status} {result.metric.value}: {result.value:.3f} (threshold: {result.threshold:.3f})")

    # Export to dict for logging
    score_dict = score.to_dict()
    print(f"\n  Score exportable to dict with {len(score_dict)} fields")

    print("  ✓ Scorecard computation complete")


def example_adversarial_validation():
    """
    Adversarial validation to detect distribution shift.

    Trains a classifier to distinguish real from synthetic data.
    Lower AUC = harder to distinguish = better synthetic data.
    """
    print_header("Adversarial Validation")

    generator = SyntheticNIRSGenerator(random_state=42)

    # Case 1: Similar distributions (should be hard to distinguish)
    X1, _, _ = generator.generate(n_samples=50)
    X2, _, _ = generator.generate(n_samples=50)

    mean_auc, std_auc = compute_adversarial_validation_auc(
        X1, X2, cv_folds=3, random_state=42
    )
    print(f"Similar data - AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"  (Should be close to 0.5 = random guessing)")

    # Case 2: Different distributions (should be easy to distinguish)
    X3 = X1 + 5.0  # Shift the data significantly
    mean_auc2, std_auc2 = compute_adversarial_validation_auc(
        X1, X3, cv_folds=3, random_state=42
    )
    print(f"Shifted data - AUC: {mean_auc2:.3f} ± {std_auc2:.3f}")
    print(f"  (Should be close to 1.0 = easily distinguishable)")

    print("  ✓ Adversarial validation complete")


def example_individual_metrics():
    """
    Compute individual spectral metrics.

    These can be used for custom validation workflows.
    """
    print_header("Individual Spectral Metrics")

    generator = SyntheticNIRSGenerator(
        wavelength_start=1000,
        wavelength_end=2500,
        random_state=42,
    )
    X, _, _ = generator.generate(n_samples=50)
    wavelengths = generator.wavelengths

    # Correlation length
    corr_lengths = compute_correlation_length(X)
    print(f"Correlation Length:")
    print(f"  Mean: {corr_lengths.mean():.1f}")
    print(f"  Std: {corr_lengths.std():.1f}")

    # Peak density
    peak_densities = compute_peak_density(X, wavelengths)
    print(f"\nPeak Density (per 100 nm):")
    print(f"  Mean: {peak_densities.mean():.2f}")
    print(f"  Std: {peak_densities.std():.2f}")

    # Signal-to-noise ratio
    snr_values = compute_snr(X)
    print(f"\nSignal-to-Noise Ratio:")
    print(f"  Mean: {snr_values.mean():.1f}")
    print(f"  Std: {snr_values.std():.1f}")

    print("  ✓ Individual metrics complete")


def example_benchmark_datasets():
    """
    Explore benchmark datasets for validation reference.

    The benchmark registry provides metadata about standard NIR
    datasets used in the literature for calibration challenges.
    """
    print_header("Benchmark Dataset Registry")

    # List available benchmarks
    datasets = list_benchmark_datasets()
    print(f"Available benchmark datasets: {len(datasets)}")
    for name in datasets:
        info = get_benchmark_info(name)
        print(f"  - {name}: {info.full_name}")

    # Get detailed info for a dataset
    print("\n  Detailed info for 'corn' dataset:")
    corn = get_benchmark_info("corn")
    print(f"    Domain: {corn.domain.value}")
    print(f"    Samples: {corn.n_samples}")
    print(f"    Wavelengths: {corn.n_wavelengths}")
    print(f"    Range: {corn.wavelength_range[0]}-{corn.wavelength_range[1]} nm")
    print(f"    Targets: {', '.join(corn.targets)}")
    print(f"    SNR range: {corn.typical_snr}")

    # Filter by domain
    pharma = get_datasets_by_domain(BenchmarkDomain.PHARMACEUTICAL)
    print(f"\n  Pharmaceutical datasets: {pharma}")

    food = get_datasets_by_domain("food")
    print(f"  Food datasets: {food}")

    # Get properties for synthetic generation
    props = get_benchmark_spectral_properties("corn")
    print(f"\n  Properties for matching 'corn':")
    print(f"    wavelength_start: {props['wavelength_start']}")
    print(f"    wavelength_end: {props['wavelength_end']}")
    print(f"    typical_components: {props['typical_components']}")

    print("  ✓ Benchmark exploration complete")


def example_prior_sampling():
    """
    Hierarchical prior sampling for configuration generation.

    The prior sampler generates realistic parameter configurations
    based on domain-specific probabilities.
    """
    print_header("Prior Sampling")

    # Create a prior config with domain-aware sampling
    config = NIRSPriorConfig()
    sampler = PriorSampler(config, random_state=42)

    # Sample a single configuration
    print("Single configuration sample:")
    sample = sampler.sample()
    print(f"  Domain: {sample['domain']}")
    print(f"  Instrument: {sample['instrument']}")
    print(f"  Measurement Mode: {sample['measurement_mode']}")
    print(f"  Matrix Type: {sample['matrix_type']}")
    print(f"  Components: {len(sample['components'])} selected")
    print(f"  Noise Level: {sample['noise_level']:.4f}")

    # Sample for a specific domain
    print("\n  Sampling for 'food' domain:")
    food_sample = sampler.sample_for_domain("food")
    print(f"    Domain: {food_sample['domain']}")
    print(f"    Instrument: {food_sample['instrument']}")

    # Batch sampling
    print("\n  Batch sampling (5 configurations):")
    batch = sampler.sample_batch(5)
    for i, s in enumerate(batch):
        print(f"    {i+1}. {s['domain']}/{s['instrument']}/{s['measurement_mode']}")

    # Convenience function
    print("\n  Using convenience function:")
    quick_sample = sample_prior(domain="pharmaceutical", random_state=42)
    print(f"    Pharma config: {quick_sample['domain']}/{quick_sample['matrix_type']}")

    print("  ✓ Prior sampling complete")


def example_gpu_acceleration():
    """
    GPU-accelerated generation for large datasets.

    The accelerated generator automatically detects and uses
    JAX or CuPy for GPU acceleration when available.
    """
    print_header("GPU Acceleration")

    # Check GPU availability
    gpu_available = is_gpu_available()
    best_backend = detect_best_backend()

    print(f"GPU Available: {gpu_available}")
    print(f"Best Backend: {best_backend.value}")

    # Get detailed backend info
    info = get_backend_info()
    print(f"\nBackend Details:")
    print(f"  JAX available: {info['jax_available']}")
    print(f"  CuPy available: {info['cupy_available']}")

    # Create accelerated generator (will use CPU numpy on most systems)
    gen = AcceleratedGenerator(
        backend=AcceleratorBackend.NUMPY,  # Force NumPy for demo
        random_state=42,
    )

    # Generate component spectra for testing
    n_wavelengths = 500
    n_components = 5
    n_samples = 1000

    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Create simple component spectra (Gaussian peaks)
    component_spectra = np.zeros((n_components, n_wavelengths))
    for i in range(n_components):
        center = 1200 + i * 300  # Spread peaks across range
        sigma = 50
        component_spectra[i] = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

    # Create random concentrations
    np.random.seed(42)
    concentrations = np.abs(np.random.randn(n_samples, n_components))

    # Generate batch of spectra
    X = gen.generate_batch(
        n_samples=n_samples,
        wavelengths=wavelengths,
        component_spectra=component_spectra,
        concentrations=concentrations,
        noise_level=0.01,
    )

    print(f"\n  Generated spectra shape: {X.shape}")
    print(f"  Using backend: {gen.backend.value}")

    # Benchmark backends (quick test)
    print("\n  Benchmarking backends (100 samples)...")
    timing = benchmark_backends(n_samples=100, n_wavelengths=200, n_trials=3)
    for backend, time_s in timing.items():
        print(f"    {backend}: {time_s*1000:.1f} ms")

    print("  ✓ GPU acceleration demo complete")


def main():
    """Run all Phase 4 validation examples."""
    print("=" * 70)
    print("R06 - Synthetic Data Validation (Phase 4)")
    print("=" * 70)

    # Run examples in order
    example_quick_realism_check()
    example_spectral_realism_scorecard()
    example_adversarial_validation()
    example_individual_metrics()
    example_benchmark_datasets()
    example_prior_sampling()
    example_gpu_acceleration()

    print("\n" + "=" * 70)
    print("All Phase 4 examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

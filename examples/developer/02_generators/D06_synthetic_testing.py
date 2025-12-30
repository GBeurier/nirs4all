"""
D06 - Synthetic Generator: Testing Integration
===============================================

Learn to use synthetic data generation for testing and benchmarking.

This tutorial covers:

* Using synthetic data for reproducible tests
* Pytest fixtures for synthetic data
* Exporting test datasets
* Comparing real and synthetic data
* Performance benchmarking

Prerequisites
-------------
Complete user examples U05, U06, and developer example D05 first.

Duration: ~5 minutes
Difficulty: ★★★★☆
"""

# Standard library imports
import argparse
import sys
import tempfile
import time
from pathlib import Path

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# NIRS4All imports
import nirs4all
from nirs4all.data.synthetic import (
    SyntheticNIRSGenerator,
    SyntheticDatasetBuilder,
    DatasetExporter,
    CSVVariationGenerator,
    RealDataFitter,
    compute_spectral_properties,
)
from nirs4all.data import DatasetConfigs

# Add examples directory to path for example_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from example_utils import get_example_output_path, print_output_location, save_array_summary

# Parse command-line arguments
parser = argparse.ArgumentParser(description='D06 Testing Integration Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# Example name for output directory
EXAMPLE_NAME = "D06_synthetic_testing"


# =============================================================================
# Section 1: Reproducible Test Data
# =============================================================================
print("\n" + "=" * 60)
print("D06 - Synthetic Data for Testing")
print("=" * 60)

print("\n" + "-" * 60)
print("Section 1: Reproducible Test Data")
print("-" * 60)

# Always use random_state for reproducibility in tests
def get_test_dataset(n_samples=100, random_state=42):
    """Create reproducible test dataset."""
    return (
        SyntheticDatasetBuilder(n_samples=n_samples, random_state=random_state)
        .with_features(complexity="simple")  # Fast for unit tests
        .with_targets(range=(0, 100))
        .with_partitions(train_ratio=0.8)
        .build()
    )

# Verify reproducibility
dataset1 = get_test_dataset()
dataset2 = get_test_dataset()

X1 = dataset1.x({}, layout="2d")
X2 = dataset2.x({}, layout="2d")

print(f"\n📊 Reproducibility check:")
print(f"   Arrays identical: {np.allclose(X1, X2)}")
print(f"   First values match: {X1[0, 0]:.6f} == {X2[0, 0]:.6f}")


# =============================================================================
# Section 2: Complexity for Different Test Types
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Complexity Selection for Tests")
print("-" * 60)

complexity_times = {}
for complexity in ["simple", "realistic", "complex"]:
    start = time.perf_counter()
    for _ in range(10):
        dataset = nirs4all.generate(
            n_samples=100,
            complexity=complexity,
            random_state=42
        )
    elapsed = (time.perf_counter() - start) / 10
    complexity_times[complexity] = elapsed

print(f"\n📊 Generation times (100 samples, average of 10 runs):")
for complexity, elapsed in complexity_times.items():
    print(f"   {complexity:10s}: {elapsed*1000:.2f} ms")

print(f"\n   Recommendation:")
print(f"   - Unit tests: 'simple' (fastest, deterministic)")
print(f"   - Integration tests: 'realistic' (typical noise)")
print(f"   - Robustness tests: 'complex' (challenging scenarios)")


# =============================================================================
# Section 3: Test Fixtures Pattern
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Test Fixtures Pattern")
print("-" * 60)

# Example fixture patterns for pytest

print("""
📋 Pytest fixture examples (from tests/conftest.py):

# Session-scoped (shared, read-only)
@pytest.fixture(scope="session")
def standard_regression_dataset():
    return SyntheticDatasetBuilder(n_samples=200, random_state=42)
        .with_features(complexity="simple")
        .with_targets(range=(0, 100))
        .with_partitions(train_ratio=0.8)
        .build()

# Function-scoped (fresh per test)
@pytest.fixture
def fresh_dataset(synthetic_builder_factory):
    return synthetic_builder_factory(n_samples=50).build()

# Factory fixture
@pytest.fixture(scope="session")
def synthetic_builder_factory():
    def _factory(n_samples=100, random_state=42, **kwargs):
        return SyntheticDatasetBuilder(
            n_samples=n_samples,
            random_state=random_state
        )
    return _factory

# File-based fixture for loader tests
@pytest.fixture
def synthetic_dataset_folder(tmp_path, synthetic_builder_factory):
    builder = synthetic_builder_factory(n_samples=100)
    return builder.export(tmp_path / "dataset", format="standard")
""")


# =============================================================================
# Section 4: CSV Loader Testing
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: CSV Loader Testing with Variations")
print("-" * 60)

# Generate test data with various CSV formats
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Create base data
    X, y = nirs4all.generate(n_samples=50, as_dataset=False, random_state=42)
    y = y[:, 0]  # Single target for loader compatibility
    wavelengths = np.linspace(1000, 2500, X.shape[1])

    # Generate different CSV variations for testing
    csv_gen = CSVVariationGenerator()

    # Standard format (semicolon delimiter)
    path_standard = csv_gen.with_semicolon_delimiter(
        tmpdir / "standard",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8
    )

    # Comma-separated format
    path_comma = csv_gen.with_comma_delimiter(
        tmpdir / "comma",
        X, y,
        wavelengths=wavelengths,
        train_ratio=0.8
    )

    print(f"\n📁 Generated test CSV variations:")
    print(f"   Standard format: {path_standard.name}/")
    for f in sorted(path_standard.iterdir()):
        print(f"     - {f.name}")

    print(f"\n   Comma separated: {path_comma.name}/")

    # Test loading
    loaded = DatasetConfigs(str(path_standard)).get_datasets()[0]
    print(f"\n   Loaded back: {loaded.num_samples} samples")


# =============================================================================
# Section 5: Benchmarking with Synthetic Data
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Algorithm Benchmarking")
print("-" * 60)

# Use synthetic data to benchmark different models
print("\nBenchmarking PLS component selection on synthetic data...")

# Generate a challenging but controlled dataset
benchmark_data = (
    SyntheticDatasetBuilder(n_samples=300, random_state=42)
    .with_features(
        complexity="realistic",
        components=["water", "protein", "lipid", "starch"]
    )
    .with_targets(component="protein", range=(0, 100))
    .with_partitions(train_ratio=0.8)
    .build()
)

X_train = benchmark_data.x({"partition": "train"}, layout="2d")
y_train = benchmark_data.y({"partition": "train"})
X_test = benchmark_data.x({"partition": "test"}, layout="2d")
y_test = benchmark_data.y({"partition": "test"})

# Benchmark different n_components
print(f"\n📊 PLS n_components optimization:")
print(f"   {'n_comp':<8} {'RMSE':>10} {'R²':>10} {'Time':>10}")
print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

for n_comp in [5, 10, 15, 20, 25, 30]:
    start = time.perf_counter()

    model = PLSRegression(n_components=n_comp)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).ravel()

    elapsed = time.perf_counter() - start
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"   {n_comp:<8} {rmse:>10.4f} {r2:>10.4f} {elapsed*1000:>8.2f}ms")


# =============================================================================
# Section 6: Real Data Comparison
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: Comparing Synthetic vs Real Data")
print("-" * 60)

# Generate "real-like" data
real_like = nirs4all.generate(
    n_samples=200,
    complexity="realistic",
    random_state=99
)
X_real = real_like.x({}, layout="2d")

# Compute spectral properties
wavelengths = np.linspace(1000, 2500, X_real.shape[1])
props_real = compute_spectral_properties(X_real, wavelengths, name="real_like")

print(f"\n📊 Spectral properties analysis:")
print(f"   Samples: {props_real.n_samples}")
print(f"   Wavelengths: {props_real.n_wavelengths}")
print(f"   Global mean: {props_real.global_mean:.4f}")
print(f"   Global std: {props_real.global_std:.4f}")
print(f"   Noise estimate: {props_real.noise_estimate:.6f}")
print(f"   SNR estimate: {props_real.snr_estimate:.1f} dB")
print(f"   PCA components (95% var): {props_real.pca_n_components_95}")

# Fit and generate matching synthetic data
print(f"\n📊 Generating matched synthetic data...")
fitter = RealDataFitter()
params = fitter.fit(X_real, wavelengths=wavelengths)

print(f"   Fitted complexity: {params.complexity}")
print(f"   Wavelength range: {params.wavelength_start:.0f}-{params.wavelength_end:.0f} nm")


# =============================================================================
# Section 7: Large-Scale Generation Performance
# =============================================================================
print("\n" + "-" * 60)
print("Section 7: Large-Scale Performance")
print("-" * 60)

print("\n📊 Generation performance by sample count:")
print(f"   {'Samples':<12} {'Time':>10} {'Rate':>15}")
print(f"   {'-'*12} {'-'*10} {'-'*15}")

for n_samples in [100, 1000, 5000, 10000]:
    start = time.perf_counter()
    X, y = nirs4all.generate(
        n_samples=n_samples,
        complexity="realistic",
        as_dataset=False,
        random_state=42
    )
    elapsed = time.perf_counter() - start
    rate = n_samples / elapsed

    print(f"   {n_samples:<12} {elapsed:>8.3f}s {rate:>12,.0f}/s")


# =============================================================================
# Section 8: Visualization
# =============================================================================
print("\n" + "-" * 60)
print("Section 8: Output Visualization")
print("-" * 60)

import matplotlib.pyplot as plt

# Save data summary
summary_path = save_array_summary(
    {
        "X_real (template data)": X_real,
        "X_train (benchmark)": X_train,
        "y_train (benchmark)": y_train,
    },
    EXAMPLE_NAME
)
print_output_location(summary_path, "Data summary")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Reproducibility check - identical spectra
ax1 = axes[0, 0]
wl = np.linspace(1000, 2500, X1.shape[1])
ax1.plot(wl, X1[0], 'b-', linewidth=2, label='Dataset 1')
ax1.plot(wl, X2[0], 'r--', linewidth=1, label='Dataset 2')
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Absorbance")
ax1.set_title("Reproducibility: Same random_state = Identical data")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Complexity comparison timing
ax2 = axes[0, 1]
complexities = list(complexity_times.keys())
times_ms = [t * 1000 for t in complexity_times.values()]
bars = ax2.bar(complexities, times_ms, color=['green', 'blue', 'red'])
ax2.set_xlabel("Complexity Level")
ax2.set_ylabel("Generation Time (ms)")
ax2.set_title("Generation Time by Complexity (100 samples)")
for bar, t in zip(bars, times_ms):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{t:.1f}ms', ha='center', va='bottom')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Benchmark training data
ax3 = axes[1, 0]
for i in range(min(30, X_train.shape[0])):
    ax3.plot(wl[:X_train.shape[1]], X_train[i], alpha=0.4, linewidth=0.7)
ax3.set_xlabel("Wavelength (nm)")
ax3.set_ylabel("Absorbance")
ax3.set_title("Benchmark Dataset (protein prediction)")
ax3.grid(True, alpha=0.3)

# Plot 4: Real-like data properties
ax4 = axes[1, 1]
mean_spec = X_real.mean(axis=0)
std_spec = X_real.std(axis=0)
wl_real = np.linspace(1000, 2500, X_real.shape[1])
ax4.fill_between(wl_real, mean_spec - std_spec, mean_spec + std_spec,
                 alpha=0.3, color='steelblue')
ax4.plot(wl_real, mean_spec, color='navy', linewidth=2)
ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Absorbance")
ax4.set_title(f"Template Data (SNR: {props_real.snr_estimate:.1f} dB)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Always save the plot
plot_path = get_example_output_path(EXAMPLE_NAME, "testing_integration_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print_output_location(plot_path, "Testing overview plot")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Testing Best Practices:

  1. ALWAYS use random_state for reproducibility
  2. Use complexity="simple" for unit tests (faster)
  3. Use session-scoped fixtures for expensive datasets
  4. Create fresh datasets for tests that modify data

Fixture Patterns:

  Session-scoped    Shared read-only datasets (fast)
  Function-scoped   Fresh dataset per test (isolated)
  Factory fixtures  Create custom datasets on demand
  File fixtures     Test loaders with temp files

CSV Variation Testing:

  CSVVariationGenerator creates format variations:
  - Different delimiters (comma, tab, semicolon)
  - With/without headers
  - European decimals (comma separator)
  - Different precisions
  - Missing values
  - Single vs multi-file

Performance Guidelines:

  100 samples     ~5-10 ms   (unit tests)
  1,000 samples   ~50-100 ms (integration tests)
  10,000 samples  ~0.5-1 s   (performance tests)

Data Comparison:

  compute_spectral_properties(X, wavelengths)
      Analyze: mean, std, noise, SNR, PCA structure

  RealDataFitter().fit(X_real, wavelengths)
      Match: complexity, wavelength range, statistics
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()

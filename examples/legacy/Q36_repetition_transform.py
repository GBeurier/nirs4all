"""
Q36_repetition_transform.py - Repetition Transformation Examples

This example demonstrates how to transform spectral repetitions (multiple
spectra per sample) into either separate data sources or additional
preprocessing slots.

When samples have multiple repetitions (e.g., 4 spectra per leaf sample),
these operators reshape the dataset structure to enable different modeling
strategies:

- `rep_to_sources`: Each repetition becomes a separate data source
  Enables: per-source preprocessing, multi-source models, source-level fusion

- `rep_to_pp`: Repetitions become additional preprocessing slots
  Enables: multi-PP models (like NiConNet), PP-level attention

Usage:
    cd examples
    ./run.sh -n Q36*.py

Prerequisites:
    - pip install nirs4all
    - No special dependencies beyond base installation

Note:
    This example demonstrates the dataset-level API for repetition
    transformation. For pipeline integration, see the controller tests.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import SpectroDataset
from nirs4all.operators.data.repetition import RepetitionConfig
from nirs4all.operators.transforms import StandardNormalVariate, SavitzkyGolay


def create_repetition_dataset(n_unique: int = 30, n_reps: int = 4, n_features: int = 100):
    """Create a synthetic dataset with repetitions.

    Args:
        n_unique: Number of unique samples.
        n_reps: Number of repetitions per sample.
        n_features: Number of spectral features.

    Returns:
        SpectroDataset with repetitions.
    """
    np.random.seed(42)
    n_total = n_unique * n_reps

    # Create synthetic spectra with sample-specific patterns
    X = np.zeros((n_total, n_features))
    y = np.zeros(n_total)
    sample_ids = np.zeros(n_total, dtype=int)

    for sample_idx in range(n_unique):
        # Each sample has a unique base spectrum
        base = np.sin(np.linspace(0, 4 * np.pi, n_features)) * (sample_idx + 1)
        target = sample_idx + 0.5  # Target value related to sample

        for rep_idx in range(n_reps):
            idx = sample_idx * n_reps + rep_idx
            # Add some random noise per repetition
            X[idx] = base + np.random.randn(n_features) * 0.1
            y[idx] = target
            sample_ids[idx] = sample_idx

    # Create dataset
    ds = SpectroDataset("repetition_demo")
    ds.add_samples(X, {"partition": "train"})
    ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])
    ds.add_targets(y)

    return ds


def example_rep_to_sources():
    """Example: Transform repetitions into separate sources.

    This approach is useful when you want:
    - Per-source preprocessing (different transforms per repetition)
    - Multi-source fusion strategies
    - Source-level attention or weighting
    """
    print("\n" + "=" * 60)
    print("Example 1: rep_to_sources")
    print("=" * 60)

    # Create dataset with 4 repetitions per sample
    ds = create_repetition_dataset(n_unique=30, n_reps=4, n_features=100)
    print(f"Before: {ds.num_samples} samples, {ds.n_sources} source(s)")
    print(f"  Shape: {ds.x({}).shape}")

    # Transform using RepetitionConfig
    config = RepetitionConfig(column="Sample_ID")
    ds.reshape_reps_to_sources(config)

    print(f"\nAfter: {ds.num_samples} samples, {ds.n_sources} source(s)")
    X_sources = ds.x({}, concat_source=False)
    for i, X_src in enumerate(X_sources):
        print(f"  Source {i}: {X_src.shape}")

    print("\n✓ Transformation successful!")
    print("  Each repetition is now a separate source, enabling:")
    print("  - Per-source preprocessing via source_branch")
    print("  - Multi-source modeling strategies")


def example_rep_to_pp():
    """Example: Transform repetitions into additional preprocessings.

    This approach is useful when you want:
    - Multi-preprocessing input for deep learning models
    - PP-level attention or selection
    - Models that operate on the (n_samples, n_pp, n_features) tensor
    """
    print("\n" + "=" * 60)
    print("Example 2: rep_to_pp")
    print("=" * 60)

    # Create dataset with 4 repetitions per sample
    ds = create_repetition_dataset(n_unique=30, n_reps=4, n_features=100)
    print(f"Before: {ds.num_samples} samples, {ds.n_sources} source(s)")
    print(f"  Shape 2D: {ds.x({}).shape}")
    print(f"  Shape 3D: {ds.x({}, layout='3d').shape}")
    print(f"  Preprocessings: {ds.features_processings(0)}")

    # Transform using RepetitionConfig
    config = RepetitionConfig(column="Sample_ID")
    ds.reshape_reps_to_preprocessings(config)

    print(f"\nAfter: {ds.num_samples} samples, {ds.n_sources} source(s)")
    print(f"  Shape 2D: {ds.x({}).shape}")
    print(f"  Shape 3D: {ds.x({}, layout='3d').shape}")
    print(f"  Preprocessings: {ds.features_processings(0)}")

    print("\n✓ Transformation successful!")
    print("  Each repetition is now a preprocessing dimension, enabling:")
    print("  - Multi-PP models like NiConNet")
    print("  - PP-level attention mechanisms")


def example_using_aggregate():
    """Example: Using dataset's aggregate column (default).

    When you set aggregate on the dataset and use column=None for
    RepetitionConfig, it uses that column automatically.
    """
    print("\n" + "=" * 60)
    print("Example 3: Using dataset's aggregate column as default")
    print("=" * 60)

    # Create dataset
    ds = create_repetition_dataset(n_unique=30, n_reps=4, n_features=100)

    # Set aggregate on the dataset
    ds.set_aggregate("Sample_ID")
    print(f"Dataset aggregate: {ds.aggregate}")

    # Use column=None (default) - will use dataset.aggregate
    config = RepetitionConfig(column=None)
    print(f"Config uses_dataset_aggregate: {config.uses_dataset_aggregate}")
    print(f"Resolved column: {config.resolve_column(ds.aggregate)}")

    # Transform
    ds.reshape_reps_to_sources(config)

    print(f"\nAfter: {ds.num_samples} samples, {ds.n_sources} sources")
    print("\n✓ Successfully used dataset.aggregate as default column!")


def example_unequal_repetitions():
    """Example: Handling samples with unequal repetition counts.

    When samples have different numbers of repetitions, use on_unequal
    to specify how to handle them.
    """
    print("\n" + "=" * 60)
    print("Example 4: Handling unequal repetitions")
    print("=" * 60)

    # Create dataset with unequal reps
    np.random.seed(42)
    n_features = 100

    # Sample 0: 4 reps, Sample 1: 4 reps, Sample 2: 3 reps (missing one)
    X = np.random.randn(11, n_features)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=float)
    sample_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

    print(f"Input: 11 samples total")
    print(f"  Sample 0: 4 reps, Sample 1: 4 reps, Sample 2: 3 reps")

    # Strategy 1: Drop samples without expected count
    print("\n--- Strategy 1: on_unequal='drop' ---")
    ds_drop = SpectroDataset("unequal_drop")
    ds_drop.add_samples(X.copy(), {"partition": "train"})
    ds_drop.add_metadata(np.array(sample_ids).reshape(-1, 1), headers=["Sample_ID"])
    ds_drop.add_targets(y.copy())

    config_drop = RepetitionConfig(column="Sample_ID", on_unequal="drop", expected_reps=4)
    ds_drop.reshape_reps_to_sources(config_drop)
    print(f"Result: {ds_drop.num_samples} samples, {ds_drop.n_sources} sources")
    print(f"  Only samples with exactly 4 reps are kept")

    # Strategy 2: Pad shorter groups with NaN
    print("\n--- Strategy 2: on_unequal='pad' ---")
    ds_pad = SpectroDataset("unequal_pad")
    ds_pad.add_samples(X.copy(), {"partition": "train"})
    ds_pad.add_metadata(np.array(sample_ids).reshape(-1, 1), headers=["Sample_ID"])
    ds_pad.add_targets(y.copy())

    config_pad = RepetitionConfig(column="Sample_ID", on_unequal="pad")
    ds_pad.reshape_reps_to_sources(config_pad)
    print(f"Result: {ds_pad.num_samples} samples, {ds_pad.n_sources} sources")

    # Check NaN in last source for sample 2
    X_sources = ds_pad.x({}, concat_source=False)
    nan_counts = [np.isnan(X_src).sum() for X_src in X_sources]
    print(f"  NaN counts per source: {nan_counts}")
    print(f"  (Source 3 has NaN for sample 2's missing 4th repetition)")

    # Strategy 3: Truncate all to minimum count
    print("\n--- Strategy 3: on_unequal='truncate' ---")
    ds_trunc = SpectroDataset("unequal_truncate")
    ds_trunc.add_samples(X.copy(), {"partition": "train"})
    ds_trunc.add_metadata(np.array(sample_ids).reshape(-1, 1), headers=["Sample_ID"])
    ds_trunc.add_targets(y.copy())

    config_trunc = RepetitionConfig(column="Sample_ID", on_unequal="truncate")
    ds_trunc.reshape_reps_to_sources(config_trunc)
    print(f"Result: {ds_trunc.num_samples} samples, {ds_trunc.n_sources} sources")
    print(f"  All samples truncated to 3 reps (minimum)")


def example_custom_naming():
    """Example: Custom source and preprocessing names."""
    print("\n" + "=" * 60)
    print("Example 5: Custom naming")
    print("=" * 60)

    # Create dataset
    ds = create_repetition_dataset(n_unique=10, n_reps=3, n_features=50)

    # Custom source naming for rep_to_sources
    config_src = RepetitionConfig(
        column="Sample_ID",
        source_names="measurement_{i}"  # Custom template
    )
    print("Source name template: 'measurement_{i}'")
    for i in range(3):
        print(f"  {config_src.get_source_name(i)}")

    # Custom PP naming for rep_to_pp
    config_pp = RepetitionConfig(
        column="Sample_ID",
        pp_names="{pp}_scan{i}"  # Custom template with {pp} and {i}
    )
    print("\nPP name template: '{pp}_scan{i}'")
    for i in range(3):
        print(f"  {config_pp.get_pp_name(i, 'raw')}")


def example_verification():
    """Verify that data integrity is preserved after transformation."""
    print("\n" + "=" * 60)
    print("Example 6: Data integrity verification")
    print("=" * 60)

    np.random.seed(123)
    n_unique = 5
    n_reps = 3
    n_features = 20

    # Create deterministic data where we know what to expect
    X = np.zeros((n_unique * n_reps, n_features))
    for sample_idx in range(n_unique):
        for rep_idx in range(n_reps):
            idx = sample_idx * n_reps + rep_idx
            # Each sample-rep combination has unique values
            X[idx] = sample_idx * 100 + rep_idx

    sample_ids = np.repeat(np.arange(n_unique), n_reps)

    ds = SpectroDataset("verify")
    ds.add_samples(X, {"partition": "train"})
    ds.add_metadata(sample_ids.reshape(-1, 1), headers=["Sample_ID"])

    print(f"Original shape: {X.shape}")
    print(f"First few values (sample 0, rep 0): {X[0, :3]}")  # Should be [0, 0, 0]
    print(f"Values (sample 0, rep 1): {X[1, :3]}")  # Should be [1, 1, 1]
    print(f"Values (sample 1, rep 0): {X[3, :3]}")  # Should be [100, 100, 100]

    # Transform to sources
    config = RepetitionConfig(column="Sample_ID")
    ds.reshape_reps_to_sources(config)

    X_sources = ds.x({}, concat_source=False)
    print(f"\nAfter transformation: {len(X_sources)} sources, each {X_sources[0].shape}")

    # Verify: Source 0 should have rep 0 from each sample
    print(f"Source 0, sample 0: {X_sources[0][0, :3]}")  # Should be [0, 0, 0]
    print(f"Source 0, sample 1: {X_sources[0][1, :3]}")  # Should be [100, 100, 100]
    print(f"Source 1, sample 0: {X_sources[1][0, :3]}")  # Should be [1, 1, 1]

    # Assertions
    assert np.allclose(X_sources[0][0], 0), "Source 0, sample 0 should have values 0"
    assert np.allclose(X_sources[0][1], 100), "Source 0, sample 1 should have values 100"
    assert np.allclose(X_sources[1][0], 1), "Source 1, sample 0 should have values 1"
    assert np.allclose(X_sources[2][0], 2), "Source 2, sample 0 should have values 2"

    print("\n✓ All data integrity checks passed!")


if __name__ == "__main__":
    # Run all examples
    example_rep_to_sources()
    example_rep_to_pp()
    example_using_aggregate()
    example_unequal_repetitions()
    example_custom_naming()
    example_verification()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


"""
U04 - Aggregation: Handling Repeated Measurements
==================================================

Aggregate predictions when multiple spectra represent one sample.

This tutorial covers:

* Setting aggregate column in DatasetConfigs
* Raw vs aggregated metrics
* Visualization with aggregation
* Overriding aggregation for specific plots

Prerequisites
-------------
Complete :ref:`U01_cv_strategies` first.

Next Steps
----------
See :ref:`06_deployment/U01_save_load_predict` for model persistence.

Duration: ~4 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse
import tempfile
import shutil
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U04 Aggregation Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1: Why Aggregation?
# =============================================================================
print("\n" + "=" * 60)
print("U04 - Prediction Aggregation")
print("=" * 60)

print("""
When multiple spectra represent the same physical sample:

  📊 PROBLEM
     - Multiple repetitions per sample (technical replicates)
     - Each spectrum gets a prediction
     - Want ONE prediction per physical sample

  📈 SOLUTION: AGGREGATION
     - Average predictions for same sample
     - Reduces measurement noise
     - More reliable final predictions

  📉 BENEFITS
     ✓ Noise reduction (averaging)
     ✓ One prediction per sample (interpretable)
     ✓ Better correlation with true values
     ✓ Both raw and aggregated metrics available
""")


# =============================================================================
# Section 2: Create Synthetic Data with Repetitions
# =============================================================================
print("\n" + "-" * 60)
print("Section 2: Creating Synthetic Data")
print("-" * 60)


def create_synthetic_data(n_samples=30, n_wavelengths=100, n_reps=4, random_state=42):
    """Create NIRS data with multiple repetitions per sample."""
    np.random.seed(random_state)

    # Split samples into train and test
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train

    # Generate base spectra for unique samples
    X_base = np.random.randn(n_samples, n_wavelengths)
    y_base = np.random.rand(n_samples) * 10 + 5

    # Split train/test
    X_base_train = X_base[:n_train]
    X_base_test = X_base[n_train:]
    y_base_train = y_base[:n_train]
    y_base_test = y_base[n_train:]

    def expand_with_reps(X_base, y_base, start_idx=0):
        X_all, y_all, sample_ids, rep_ids = [], [], [], []
        for i in range(len(X_base)):
            for r in range(n_reps):
                # Add small noise for each repetition
                noise = np.random.randn(n_wavelengths) * 0.1
                X_all.append(X_base[i] + noise)
                y_all.append(y_base[i])
                sample_ids.append(f"sample_{start_idx + i:03d}")
                rep_ids.append(r + 1)
        return np.array(X_all), np.array(y_all), sample_ids, rep_ids

    X_train, y_train, train_ids, train_reps = expand_with_reps(X_base_train, y_base_train, 0)
    X_test, y_test, test_ids, test_reps = expand_with_reps(X_base_test, y_base_test, n_train)

    # Create metadata DataFrames
    train_meta = pd.DataFrame({'sample_id': train_ids, 'repetition': train_reps})
    test_meta = pd.DataFrame({'sample_id': test_ids, 'repetition': test_reps})

    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir()) / "nirs4all_examples" / "u20_aggregation"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(temp_dir / "Xcal.csv.gz", index=False, header=False, compression='gzip', sep=';')
    pd.DataFrame(y_train).to_csv(temp_dir / "Ycal.csv.gz", index=False, header=False, compression='gzip', sep=';')
    train_meta.to_csv(temp_dir / "Mcal.csv", index=False, sep=';')

    pd.DataFrame(X_test).to_csv(temp_dir / "Xval.csv.gz", index=False, header=False, compression='gzip', sep=';')
    pd.DataFrame(y_test).to_csv(temp_dir / "Yval.csv.gz", index=False, header=False, compression='gzip', sep=';')
    test_meta.to_csv(temp_dir / "Mval.csv", index=False, sep=';')

    return str(temp_dir), n_train, n_test, n_reps


data_path, n_train, n_test, n_reps = create_synthetic_data()

print(f"Created synthetic dataset:")
print(f"   Train: {n_train} samples × {n_reps} reps = {n_train * n_reps} spectra")
print(f"   Test:  {n_test} samples × {n_reps} reps = {n_test * n_reps} spectra")


# =============================================================================
# Section 3: Running with Aggregation
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Running with Aggregation")
print("-" * 60)

print("""
Set aggregate="column_name" in DatasetConfigs to enable aggregation.
""")

# Dataset config with aggregation
dataset_config = DatasetConfigs(
    {
        "train_x": str(Path(data_path) / "Xcal.csv.gz"),
        "train_y": str(Path(data_path) / "Ycal.csv.gz"),
        "train_m": str(Path(data_path) / "Mcal.csv"),
        "test_x": str(Path(data_path) / "Xval.csv.gz"),
        "test_y": str(Path(data_path) / "Yval.csv.gz"),
        "test_m": str(Path(data_path) / "Mval.csv"),
        "train_x_params": {"has_header": False},
        "train_y_params": {"has_header": False},
        "train_m_params": {"has_header": True},
        "test_x_params": {"has_header": False},
        "test_y_params": {"has_header": False},
        "test_m_params": {"has_header": True},
    },
    aggregate="sample_id"  # <-- Key setting!
)

# Pipeline
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
    {"model": PLSRegression(n_components=3)},
    {"model": PLSRegression(n_components=5)},
    {"model": PLSRegression(n_components=10)},
]

pipeline_config = PipelineConfigs(pipeline, "U20_Aggregation")

# Run pipeline
runner = PipelineRunner(
    save_artifacts=False,
    save_charts=False,
    verbose=1,
    plots_visible=args.plots
)

predictions, _ = runner.run(pipeline_config, dataset_config)

print(f"\nAggregate setting used: '{runner.last_aggregate}'")


# =============================================================================
# Section 4: Raw vs Aggregated Metrics
# =============================================================================
print("\n" + "-" * 60)
print("Section 4: Raw vs Aggregated Metrics")
print("-" * 60)

print("""
Both raw and aggregated metrics are available:
  - Raw: One prediction per spectrum
  - Aggregated: One prediction per sample (averaged)
""")

# Get best model with raw metrics
best_raw = predictions.top(1, rank_metric='rmse', aggregate='')[0]
model_name = best_raw.get('model_name', 'Unknown')

print(f"\nBest model: {model_name}")

# Raw metrics
val_rmse_raw = best_raw.get('val_score', np.nan)
test_rmse_raw = best_raw.get('test_score', np.nan)
print(f"\nRaw metrics (per spectrum):")
print(f"   Val RMSE:  {val_rmse_raw:.4f}" if not np.isnan(val_rmse_raw) else "   Val RMSE:  N/A")
print(f"   Test RMSE: {test_rmse_raw:.4f}" if not np.isnan(test_rmse_raw) else "   Test RMSE: N/A")

# Get same model with aggregated metrics
best_agg = predictions.top(1, rank_metric='rmse', aggregate='sample_id')[0]
val_rmse_agg = best_agg.get('val_score', np.nan)
test_rmse_agg = best_agg.get('test_score', np.nan)
print(f"\nAggregated metrics (per sample):")
print(f"   Val RMSE:  {val_rmse_agg:.4f}" if not np.isnan(val_rmse_agg) else "   Val RMSE:  N/A")
print(f"   Test RMSE: {test_rmse_agg:.4f}" if not np.isnan(test_rmse_agg) else "   Test RMSE: N/A")

print("\nNote: Aggregated RMSE is typically LOWER due to noise averaging!")


# =============================================================================
# Section 5: Visualization with Aggregation
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Visualization with Aggregation")
print("-" * 60)

print("""
Pass default_aggregate to PredictionAnalyzer for automatic aggregation.
""")

# Create analyzer with default aggregation
analyzer = PredictionAnalyzer(
    predictions,
    default_aggregate=runner.last_aggregate  # Uses sample_id
)

print(f"Analyzer default_aggregate: '{analyzer.default_aggregate}'")

if args.plots:
    # Top-K with aggregation (automatic)
    fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')
    fig1.suptitle("Top Models (Aggregated by sample_id)", y=1.02)

    # Override: disable aggregation for specific plot
    fig2 = analyzer.plot_histogram(aggregate='')  # Empty string = no aggregation
    fig2.suptitle("Histogram (Raw predictions)", y=1.02)

    print("Charts generated")

    if args.show:
        plt.show()


# =============================================================================
# Section 6: When to Use Aggregation
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: When to Use Aggregation")
print("-" * 60)

print("""
Use aggregation when:
  ✓ Multiple spectra per physical sample (repetitions)
  ✓ Technical replicates with same target value
  ✓ Need one final prediction per sample

Do NOT use when:
  ✗ Each spectrum is an independent sample
  ✗ Target varies within repetitions
  ✗ Repetitions are from different conditions
""")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Aggregation Configuration:

  1. DATASET CONFIG:
     dataset_config = DatasetConfigs(
         "path/to/data",
         aggregate="sample_id"  # Metadata column with sample ID
     )

  2. ACCESS AFTER RUN:
     runner.last_aggregate  # Returns the aggregate column name

  3. VISUALIZATION:
     analyzer = PredictionAnalyzer(
         predictions,
         default_aggregate=runner.last_aggregate
     )

  4. OVERRIDE PER-PLOT:
     analyzer.plot_histogram(aggregate='')  # Disable for this plot
     analyzer.plot_histogram(aggregate='sample_id')  # Force aggregation

Result Output:
  - Raw metrics: Evaluated on individual spectra
  - Aggregated metrics (*): Averaged predictions per sample
  - Both shown in TabReport output

Benefits of Aggregation:
  ✓ Noise reduction through averaging
  ✓ One prediction per sample (practical)
  ✓ More robust metrics
  ✓ Better correlation with true values

Next: See 06_deployment/U01_save_load_predict.py - Save and load models
""")

"""
Q34 Example - Sample Repetition Handling
========================================
Demonstrates the dataset-level repetition feature for handling multiple
spectral measurements per physical sample.

Features:
- Setting repetition column via DatasetConfigs
- Automatic propagation to TabReport (raw + aggregated scores)
- Using default repetition in PredictionAnalyzer for visualizations
- Both raw and repetition-aggregated metrics in pipeline output

When repetition is enabled:
1. Models are trained on all individual spectra (maximizing data)
2. Performance is evaluated on both raw and aggregated predictions
3. Aggregated rows (marked with *) show metrics after averaging
   predictions for repeated measurements of the same sample
"""

import argparse
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q34 Aggregation Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at end')
args = parser.parse_args()


def create_synthetic_data_files(n_samples=50, n_wavelengths=100, n_reps=4, random_state=42):
    """Create synthetic NIRS data files with multiple repetitions per sample.

    Args:
        n_samples: Number of unique physical samples
        n_wavelengths: Number of wavelengths/features
        n_reps: Number of spectral repetitions per sample
        random_state: Random seed

    Returns:
        Path to the dataset folder
    """
    np.random.seed(random_state)

    # Split samples into train and test
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train

    # Generate base spectra for unique samples
    X_base = np.random.randn(n_samples, n_wavelengths)
    y_base = np.random.rand(n_samples) * 10 + 5  # Target values between 5-15

    # Split train/test
    X_base_train, X_base_test = X_base[:n_train], X_base[n_train:]
    y_base_train, y_base_test = y_base[:n_train], y_base[n_train:]

    def create_reps(X_base, y_base, start_idx=0):
        X_all, y_all, sample_ids, rep_ids = [], [], [], []
        for i in range(len(X_base)):
            for r in range(n_reps):
                noise = np.random.randn(n_wavelengths) * 0.1
                X_all.append(X_base[i] + noise)
                y_all.append(y_base[i])
                sample_ids.append(f"sample_{start_idx + i:03d}")
                rep_ids.append(r + 1)
        return np.array(X_all), np.array(y_all), sample_ids, rep_ids

    X_train, y_train, train_ids, train_reps = create_reps(X_base_train, y_base_train, 0)
    X_test, y_test, test_ids, test_reps = create_reps(X_base_test, y_base_test, n_train)

    # Create metadata DataFrames
    train_meta = pd.DataFrame({'sample_id': train_ids, 'repetition': train_reps})
    test_meta = pd.DataFrame({'sample_id': test_ids, 'repetition': test_reps})

    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir()) / "nirs4all_examples" / "q34_aggregation"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save with semicolon delimiter (no headers for X, Y)
    pd.DataFrame(X_train).to_csv(temp_dir / "Xcal.csv.gz", index=False, header=False,
                                 compression='gzip', sep=';')
    pd.DataFrame(y_train).to_csv(temp_dir / "Ycal.csv.gz", index=False, header=False,
                                 compression='gzip', sep=';')
    train_meta.to_csv(temp_dir / "Mcal.csv", index=False, sep=';')

    pd.DataFrame(X_test).to_csv(temp_dir / "Xval.csv.gz", index=False, header=False,
                                compression='gzip', sep=';')
    pd.DataFrame(y_test).to_csv(temp_dir / "Yval.csv.gz", index=False, header=False,
                                compression='gzip', sep=';')
    test_meta.to_csv(temp_dir / "Mval.csv", index=False, sep=';')

    return str(temp_dir), n_train, n_test, n_reps


# ============================================================================
# Part 1: Create synthetic dataset with repeated measurements
# ============================================================================

print("="  * 80)
print("Q34 Example - Sample Repetition Handling")
print("=" * 80)

print("\n1. Creating synthetic dataset with repeated measurements...")
data_path, n_train_samples, n_test_samples, n_reps = create_synthetic_data_files(
    n_samples=50,       # 50 unique samples (40 train + 10 test)
    n_wavelengths=100,  # 100 wavelengths
    n_reps=4,           # 4 repetitions per sample
    random_state=42
)

print(f"   - Train samples: {n_train_samples} unique x {n_reps} reps = {n_train_samples * n_reps} spectra")
print(f"   - Test samples: {n_test_samples} unique x {n_reps} reps = {n_test_samples * n_reps} spectra")
print(f"   - Data saved to: {data_path}")


# ============================================================================
# Part 2: Run pipeline WITH aggregation
# ============================================================================

print("\n2. Running pipeline with repetition='sample_id'...")

# Define dataset config using dict format with explicit paths and params
# This ensures proper header handling for synthetic data
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
    repetition="sample_id"  # <-- Key setting: group by sample_id
)

# Simple pipeline
pipeline = [
    MinMaxScaler(),
    ShuffleSplit(n_splits=3, test_size=0.25, random_state=42),
]

# Add PLS models with different components
for n_comp in [3, 5, 10, 15]:
    pipeline.append({"model": PLSRegression(n_components=n_comp)})

pipeline_config = PipelineConfigs(pipeline, "Q34_Aggregation")

# Run with verbose to see aggregated TabReport
runner = PipelineRunner(
    save_artifacts=False,
    save_charts=False,
    verbose=1,
    plots_visible=args.plots
)

predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# ============================================================================
# Part 3: Access the aggregate setting for visualization
# ============================================================================

print("\n3. Accessing aggregate setting from runner...")
print(f"   - runner.last_aggregate = '{runner.last_aggregate}'")


# ============================================================================
# Part 4: Create PredictionAnalyzer with default_aggregate
# ============================================================================

print("\n4. Creating PredictionAnalyzer with default_aggregate...")

# Pass the aggregate setting to the analyzer
analyzer = PredictionAnalyzer(
    predictions,
    default_aggregate=runner.last_aggregate  # <-- Uses repetition column for visualization
)

print(f"   - analyzer.default_aggregate = '{analyzer.default_aggregate}'")
print("   - All visualization methods will now use aggregation by default")


# ============================================================================
# Part 5: Visualize results (with automatic aggregation)
# ============================================================================

print("\n5. Creating visualizations (all using repetition-aggregated predictions)...")

# Top-K comparison - uses aggregation automatically
fig1 = analyzer.plot_top_k(k=4, rank_metric='rmse')
fig1.suptitle("Top Models (Aggregated by sample_id)", y=1.02)
print("   - plot_top_k(): Repetition-aggregated scores shown")

# Heatmap - uses aggregation automatically
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="fold_id",
    rank_metric="rmse",
    display_metric="rmse"
)
print("   - plot_heatmap(): Repetition-aggregated scores shown")


# ============================================================================
# Part 6: Override aggregation for specific plots
# ============================================================================

print("\n6. Overriding aggregation for specific plots...")

# Disable aggregation for this specific plot
fig3 = analyzer.plot_histogram(aggregate='')  # Empty string disables aggregation
print("   - plot_histogram(aggregate=''): Raw (non-aggregated) scores")


# ============================================================================
# Part 7: Compare raw vs repetition-aggregated metrics
# ============================================================================

print("\n7. Comparing raw vs repetition-aggregated metrics for top model...")

# Get best model (raw metrics - no aggregation)
best_raw = predictions.top(1, rank_metric='rmse', by_repetition=False)[0]
print(f"\n   Best model: {best_raw.get('model_name', 'Unknown')}")
print(f"   Preprocessings: {best_raw.get('preprocessings', 'N/A')}")

# Raw metrics
val_rmse_raw = best_raw.get('val_score', np.nan)
test_rmse_raw = best_raw.get('test_score', np.nan)
print(f"\n   Raw metrics (individual spectra):")
print(f"   - Val RMSE:  {val_rmse_raw:.4f}" if not np.isnan(val_rmse_raw) else "   - Val RMSE:  N/A")
print(f"   - Test RMSE: {test_rmse_raw:.4f}" if not np.isnan(test_rmse_raw) else "   - Test RMSE: N/A")

# Get same model with repetition-aggregated metrics
best_agg = predictions.top(1, rank_metric='rmse', by_repetition='sample_id')[0]
val_rmse_agg = best_agg.get('val_score', np.nan)
test_rmse_agg = best_agg.get('test_score', np.nan)
print(f"\n   Repetition-aggregated metrics (averaged per sample):")
print(f"   - Val RMSE:  {val_rmse_agg:.4f}" if not np.isnan(val_rmse_agg) else "   - Val RMSE:  N/A")
print(f"   - Test RMSE: {test_rmse_agg:.4f}" if not np.isnan(test_rmse_agg) else "   - Test RMSE: N/A")

print("\n   Note: Aggregated RMSE is typically lower because averaging")
print("         multiple measurements reduces prediction noise.")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
Key takeaways:
1. Set repetition='column_name' in DatasetConfigs to enable grouping
2. TabReport shows both raw and aggregated (*) metrics automatically
3. Use runner.last_aggregate to get the aggregate setting after run()
4. Pass default_aggregate to PredictionAnalyzer for automatic aggregation
5. Override with aggregate='' to disable aggregation for specific plots
6. Aggregated metrics are typically better due to noise reduction
""")


if args.show:
    plt.show()

print("Done!")

"""
U04 - Repetition: Handling Repeated Measurements
=================================================

Aggregate predictions when multiple spectra represent one sample.

This tutorial covers:

* Setting repetition column in DatasetConfigs
* Raw vs repetition-aggregated metrics
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
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

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

  📈 SOLUTION: REPETITION AGGREGATION
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
    """Create NIRS data with repetition and a second custom grouping column."""
    np.random.seed(random_state)
    samples_per_lot = 2

    # Split samples into train and test
    n_train = int(n_samples * 0.8)
    n_test = n_samples - n_train

    # Generate base spectra for unique samples
    X_base = np.random.randn(n_samples, n_wavelengths)
    n_lots = int(np.ceil(n_samples / samples_per_lot))
    lot_targets = np.random.rand(n_lots) * 10 + 5
    y_base = np.array([lot_targets[i // samples_per_lot] for i in range(n_samples)])

    # Split train/test
    X_base_train = X_base[:n_train]
    X_base_test = X_base[n_train:]
    y_base_train = y_base[:n_train]
    y_base_test = y_base[n_train:]

    def expand_with_reps(X_base, y_base, start_idx=0):
        X_all, y_all, sample_ids, lot_ids, rep_ids = [], [], [], [], []
        for i in range(len(X_base)):
            sample_idx = start_idx + i
            lot_idx = sample_idx // samples_per_lot
            for r in range(n_reps):
                # Add realistic measurement noise for each repetition
                noise = np.random.randn(n_wavelengths) * 1.5
                X_all.append(X_base[i] + noise)
                y_all.append(y_base[i])
                sample_ids.append(f"sample_{sample_idx:03d}")
                lot_ids.append(f"lot_{lot_idx:03d}")
                rep_ids.append(r + 1)
        return np.array(X_all), np.array(y_all), sample_ids, lot_ids, rep_ids

    X_train, y_train, train_ids, train_lots, train_reps = expand_with_reps(X_base_train, y_base_train, 0)
    X_test, y_test, test_ids, test_lots, test_reps = expand_with_reps(X_base_test, y_base_test, n_train)

    # Create metadata DataFrames
    train_meta = pd.DataFrame({'sample_id': train_ids, 'lot_id': train_lots, 'repetition': train_reps})
    test_meta = pd.DataFrame({'sample_id': test_ids, 'lot_id': test_lots, 'repetition': test_reps})

    # Save to temp directory
    temp_dir = Path(tempfile.gettempdir()) / "nirs4all_examples" / "u20_aggregation"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(temp_dir / "Xcal.csv.gz", index=False, header=False, compression='gzip', sep=';')
    pd.DataFrame(y_train).to_csv(temp_dir / "Ycal.csv.gz", index=False, header=False, compression='gzip', sep=';')
    train_meta.to_csv(temp_dir / "Mcal.csv", index=False, sep=';')

    pd.DataFrame(X_test).to_csv(temp_dir / "Xval.csv.gz", index=False, header=False, compression='gzip', sep=';')
    pd.DataFrame(y_test).to_csv(temp_dir / "Yval.csv.gz", index=False, header=False, compression='gzip', sep=';')
    test_meta.to_csv(temp_dir / "Mval.csv", index=False, sep=';')

    return str(temp_dir), n_train, n_test, n_reps

data_path, n_train, n_test, n_reps = create_synthetic_data()

print("Created synthetic dataset:")
print(f"   Train: {n_train} samples × {n_reps} reps = {n_train * n_reps} spectra")
print(f"   Test:  {n_test} samples × {n_reps} reps = {n_test * n_reps} spectra")
print("   Metadata columns: sample_id (dataset repetition), lot_id (custom grouping)")

# =============================================================================
# Section 3: Running with Repetition
# =============================================================================
print("\n" + "-" * 60)
print("Section 3: Running with Repetition")
print("-" * 60)

print("""
Set repetition="column_name" in DatasetConfigs to enable aggregation.
""")

# Dataset config with repetition
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
    repetition="sample_id"  # <-- Key setting!
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
    # This example manages chart display explicitly with PredictionAnalyzer below.
    plots_visible=False,
)

predictions, _run_info = runner.run(pipeline_config, dataset_config)

print(f"\nRepetition setting: '{predictions.repetition_column}'")

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
  - Custom: One prediction per lot_id (2 samples merged)
""")

# Get best model with raw metrics
top_raw = predictions.top(1, rank_metric='rmse', by_repetition=False)
assert isinstance(top_raw, list) and len(top_raw) > 0, "Raw top() returned empty list"
best_raw = top_raw[0]
model_name = best_raw.get('model_name', 'Unknown')

print(f"\nBest model: {model_name}")

# Raw metrics
val_rmse_raw = float(best_raw.get('val_score', np.nan))
test_rmse_raw = float(best_raw.get('test_score', np.nan))
print("\nRaw metrics (per spectrum):")
print(f"   Val RMSE:  {val_rmse_raw:.4f}" if not np.isnan(val_rmse_raw) else "   Val RMSE:  N/A")
print(f"   Test RMSE: {test_rmse_raw:.4f}" if not np.isnan(test_rmse_raw) else "   Test RMSE: N/A")

# Get same model with repetition-aggregated metrics
top_agg = predictions.top(1, rank_metric='rmse', by_repetition=True)
assert isinstance(top_agg, list) and len(top_agg) > 0, "Aggregated top() returned empty list"
best_agg = top_agg[0]
val_rmse_agg = float(best_agg.get('val_score', np.nan))
test_rmse_agg = float(best_agg.get('test_score', np.nan))
assert np.isfinite(val_rmse_agg), f"Aggregated val_score is not finite: {val_rmse_agg}"
assert best_agg.get('aggregated', False), "Result should be marked as aggregated"
print("\nRepetition-aggregated metrics (per sample):")
print(f"   Val RMSE:  {val_rmse_agg:.4f}" if not np.isnan(val_rmse_agg) else "   Val RMSE:  N/A")
print(f"   Test RMSE: {test_rmse_agg:.4f}" if not np.isnan(test_rmse_agg) else "   Test RMSE: N/A")

top_lot = predictions.top(1, rank_metric='rmse', by_repetition='lot_id')
assert isinstance(top_lot, list) and len(top_lot) > 0, "Custom-aggregated top() returned empty list"
best_lot = top_lot[0]
val_rmse_lot = float(best_lot.get('val_score', np.nan))
test_rmse_lot = float(best_lot.get('test_score', np.nan))

print("\nCustom aggregation metrics (per lot_id):")
print(f"   Val RMSE:  {val_rmse_lot:.4f}" if not np.isnan(val_rmse_lot) else "   Val RMSE:  N/A")
print(f"   Test RMSE: {test_rmse_lot:.4f}" if not np.isnan(test_rmse_lot) else "   Test RMSE: N/A")
print(f"\nDisplayed predictions count: raw={len(best_raw['y_pred'])}, sample_id={len(best_agg['y_pred'])}, lot_id={len(best_lot['y_pred'])}")

print("\nNote: Aggregated RMSE is typically LOWER due to noise averaging!")

# =============================================================================
# Section 5: Visualization with Aggregation
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Visualization with Aggregation")
print("-" * 60)

print("""
PredictionAnalyzer supports two aggregation modes:
  - PredictionAnalyzer(predictions) -> raw + repetition-aggregated figures (dual)
  - aggregate='lot_id' on one plot -> only the custom-aggregated figure
""")

# Create analyzer with automatic default aggregation from predictions.repetition_column
analyzer = PredictionAnalyzer(predictions)

print(f"Analyzer default_aggregate: '{analyzer.default_aggregate}'")

if args.plots:
    def as_figure_list(fig_or_figs):
        if isinstance(fig_or_figs, list):
            return fig_or_figs
        if fig_or_figs is None:
            return []
        return [fig_or_figs]

    def show_figures_and_wait(figures):
        """Show all figures and return when the last visible window is closed."""
        if not figures:
            return

        # Explicitly show each manager so every chart window becomes visible.
        for fig in figures:
            manager = getattr(fig.canvas, "manager", None)
            if manager is None:
                continue
            try:
                manager.show()
            except Exception:
                pass

        # Drive the GUI event loop until all figures are closed by the user.
        plt.show(block=False)
        while any(plt.fignum_exists(fig.number) for fig in figures):
            plt.pause(0.1)

    def show_batch(label, figures):
        """Display one chart family at a time to avoid hidden windows."""
        if not figures:
            return

        print(f"\nShowing {label}. Close the window(s) to continue.")
        try:
            show_figures_and_wait(figures)
        finally:
            for fig in figures:
                plt.close(fig)
            plt.close('all')

    # 1. Default dataset repetition behavior: raw + sample_id aggregation (dual)
    top_k_figures = [
        *as_figure_list(analyzer.plot_top_k(k=3, rank_metric='rmse')),
        *as_figure_list(analyzer.plot_top_k(k=3, rank_metric='rmse', aggregate='lot_id')),
    ]

    # 2. Explicit aggregate override: only the aggregated chart (no raw)
    heatmap_figures = [
        *as_figure_list(analyzer.plot_heatmap(
            'partition',
            'model_name',
            rank_metric='rmse',
            display_metric='rmse',
        )),
        *as_figure_list(analyzer.plot_heatmap(
            'partition',
            'model_name',
            rank_metric='rmse',
            display_metric='rmse',
            aggregate='lot_id',
        )),
    ]

    # 3. Raw-only override on a single plot
    raw_histogram = as_figure_list(analyzer.plot_histogram(display_metric='rmse', aggregate=''))

    print("\nChart behavior:")
    print(f"   plot_top_k() returned {len(top_k_figures)} figure(s): raw + sample_id aggregation (dual)")
    print(f"   plot_heatmap(..., aggregate='lot_id') returned {len(heatmap_figures)} figure(s): lot_id aggregation only")
    print(f"   plot_histogram(..., aggregate='') returned {len(raw_histogram)} figure(s): raw only")

    all_figures = [*top_k_figures, *heatmap_figures, *raw_histogram]

    if args.show:
        show_batch("top-k comparison", top_k_figures)
        show_batch("heatmap comparison", heatmap_figures)
        show_batch("raw histogram", raw_histogram)
    else:
        for fig in all_figures:
            plt.close(fig)
        plt.close('all')

runner.close()

# =============================================================================
# Section 6: When to Use Repetition
# =============================================================================
print("\n" + "-" * 60)
print("Section 6: When to Use Repetition")
print("-" * 60)

print("""
Use repetition when:
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
Repetition Configuration:

  1. DATASET CONFIG:
     dataset_config = DatasetConfigs(
         "path/to/data",
         repetition="sample_id"  # Metadata column with sample ID
     )

  2. ACCESS AFTER RUN:
     predictions.repetition_column  # Returns the repetition column name

  3. VISUALIZATION:
     analyzer = PredictionAnalyzer(predictions)

  4. OVERRIDE PER-PLOT:
     analyzer.plot_histogram(aggregate='')  # Disable for this plot
     analyzer.plot_histogram(aggregate='sample_id')  # Force aggregation
     analyzer.plot_heatmap('partition', 'model_name', aggregate='lot_id')  # Custom grouping

  5. TOP MODELS (uses by_repetition):
     predictions.top(5, by_repetition=True)         # Use dataset repetition
     predictions.top(5, by_repetition='lot_id')     # Custom aggregation

Result Output:
  - Raw metrics: Evaluated on individual spectra
  - Aggregated metrics (*): Averaged predictions per sample
  - Both shown in TabReport output

Benefits of Repetition Aggregation:
  ✓ Noise reduction through averaging
  ✓ One prediction per sample (practical)
  ✓ More robust metrics
  ✓ Better correlation with true values

Next: See 06_deployment/U01_save_load_predict.py - Save and load models
""")

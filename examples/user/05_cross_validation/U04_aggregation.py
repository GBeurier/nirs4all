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
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization import PredictionAnalyzer, show_figures

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
workspace_path = Path(data_path) / "workspace"

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
    workspace_path=str(workspace_path),
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

def require_single_result(results, message):
    assert isinstance(results, list) and len(results) > 0, message
    return results[0]

def get_variant_result(
    predictions_obj,
    *,
    model_name,
    step_idx,
    rank_partition,
    display_partition,
    score_scope,
    by_repetition,
):
    return require_single_result(
        predictions_obj.top(
            1,
            rank_metric='rmse',
            rank_partition=rank_partition,
            display_partition=display_partition,
            score_scope=score_scope,
            by_repetition=by_repetition,
            model_name=model_name,
            step_idx=step_idx,
        ),
        f"No result found for model={model_name}, step_idx={step_idx}, partition={display_partition}",
    )

# Pick the best refit/final model, then compare raw vs aggregated metrics for the SAME variant
best_raw = require_single_result(
    predictions.top(
        1,
        rank_metric='rmse',
        rank_partition='test',
        display_partition='test',
        score_scope='final',
        by_repetition=False,
    ),
    "Raw final top() returned empty list",
)
model_name = best_raw.get('model_name', 'Unknown')
step_idx = best_raw.get('step_idx')

print(f"\nBest model variant: {model_name} (step_idx={step_idx})")

# Raw metrics for the selected variant
raw_val_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='val',
    display_partition='val',
    score_scope='cv',
    by_repetition=False,
)
raw_test_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='test',
    display_partition='test',
    score_scope='final',
    by_repetition=False,
)
val_rmse_raw = float(raw_val_entry.get('val_score'))
test_rmse_raw = float(raw_test_entry.get('test_score'))
print("\nRaw metrics (per spectrum):")
print(f"   Val RMSE:  {val_rmse_raw:.4f}")
print(f"   Test RMSE: {test_rmse_raw:.4f}")

# Same model with repetition-aggregated metrics
agg_val_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='val',
    display_partition='val',
    score_scope='cv',
    by_repetition=True,
)
agg_test_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='test',
    display_partition='test',
    score_scope='final',
    by_repetition=True,
)
val_rmse_agg = float(agg_val_entry.get('val_score'))
test_rmse_agg = float(agg_test_entry.get('test_score'))
assert agg_test_entry.get('aggregated', False), "Result should be marked as aggregated"
print("\nRepetition-aggregated metrics (per sample):")
print(f"   Val RMSE:  {val_rmse_agg:.4f}")
print(f"   Test RMSE: {test_rmse_agg:.4f}")

lot_val_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='val',
    display_partition='val',
    score_scope='cv',
    by_repetition='lot_id',
)
lot_test_entry = get_variant_result(
    predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='test',
    display_partition='test',
    score_scope='final',
    by_repetition='lot_id',
)
val_rmse_lot = float(lot_val_entry.get('val_score'))
test_rmse_lot = float(lot_test_entry.get('test_score'))

print("\nCustom aggregation metrics (per lot_id):")
print(f"   Val RMSE:  {val_rmse_lot:.4f}")
print(f"   Test RMSE: {test_rmse_lot:.4f}")
print(
    f"\nDisplayed predictions count: "
    f"raw={len(raw_test_entry['y_pred'])}, "
    f"sample_id={len(agg_test_entry['y_pred'])}, "
    f"lot_id={len(lot_test_entry['y_pred'])}"
)

print("\nNote: Aggregated RMSE is typically LOWER due to noise averaging!")

print("\nReloading predictions from workspace to verify aggregation context survives persistence...")
reloaded_predictions = Predictions.from_workspace(workspace_path)
print(f"Reloaded repetition setting: '{reloaded_predictions.repetition_column}'")

reload_agg_test_entry = get_variant_result(
    reloaded_predictions,
    model_name=model_name,
    step_idx=step_idx,
    rank_partition='test',
    display_partition='test',
    score_scope='final',
    by_repetition=True,
)
reload_test_rmse = float(reload_agg_test_entry.get('test_score'))
assert reload_agg_test_entry.get('aggregated', False), "Reloaded result should be marked as aggregated"

print("\nReloaded repetition-aggregated metrics (per sample):")
print(f"   Test RMSE: {reload_test_rmse:.4f}" if not np.isnan(reload_test_rmse) else "   Test RMSE: N/A")
print(f"   Same aggregation after reload: {np.isclose(test_rmse_agg, reload_test_rmse)}")

# =============================================================================
# Section 5: Visualization with Aggregation
# =============================================================================
print("\n" + "-" * 60)
print("Section 5: Visualization with Aggregation")
print("-" * 60)

print("""
PredictionAnalyzer supports the same aggregation workflow from:
  - live run predictions
  - workspace-reloaded predictions

This example renders the same plot batch twice:
  - default aggregation from dataset repetition (sample_id)
  - custom aggregation override (lot_id)
""")

# Create analyzers with automatic default aggregation from prediction context
live_analyzer = PredictionAnalyzer(
    predictions,
    save=args.plots,
    output_dir=str(workspace_path / "figures" / "live") if args.plots else None,
)
reloaded_analyzer = PredictionAnalyzer(
    reloaded_predictions,
    save=args.plots,
    output_dir=str(workspace_path / "figures" / "reloaded") if args.plots else None,
)

print(f"Live analyzer default_aggregate: '{live_analyzer.default_aggregate}'")
print(f"Reloaded analyzer default_aggregate: '{reloaded_analyzer.default_aggregate}'")

def as_figure_list(fig_or_figs):
    if isinstance(fig_or_figs, list):
        return fig_or_figs
    if fig_or_figs is None:
        return []
    return [fig_or_figs]

def render_plot_batch(analyzer, label):
    """Render the same plot batch for a given Predictions source."""
    figures = [
        *as_figure_list(analyzer.plot_top_k(k=3, rank_metric='rmse')),
        *as_figure_list(analyzer.plot_top_k(k=3, rank_metric='rmse', aggregate=True)),
        *as_figure_list(analyzer.plot_top_k(k=3, rank_metric='rmse', aggregate='lot_id')),
    ]
    print(f"\n{label}: generated {len(figures)} figure(s)")
    return figures

if args.plots or args.show:
    live_figures = render_plot_batch(live_analyzer, "Live run predictions")
    reloaded_figures = render_plot_batch(reloaded_analyzer, "Workspace-reloaded predictions")

    if args.show:
        print("\nShowing live run prediction figures. Close the window(s) to continue.")
        show_figures(live_figures, block=True, close=True)
        print("\nShowing workspace-reloaded prediction figures. Close the window(s) to continue.")
        show_figures(reloaded_figures, block=True, close=True)
    else:
        for fig in [*live_figures, *reloaded_figures]:
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
     reloaded = Predictions.from_workspace(workspace_path)

  3. VISUALIZATION:
     analyzer = PredictionAnalyzer(predictions, save=True)
     analyzer_reload = PredictionAnalyzer(reloaded, save=True)

  4. OVERRIDE PER-PLOT:
     analyzer.plot_histogram(aggregate='')  # Disable for this plot
     analyzer.plot_histogram(aggregate='sample_id')  # Force aggregation
     analyzer.plot_heatmap('partition', 'model_name', aggregate='lot_id')  # Custom grouping

  5. TOP MODELS (uses by_repetition):
     predictions.top(5, by_repetition=True)         # Use dataset repetition
     predictions.top(5, by_repetition='lot_id')     # Custom aggregation
     reloaded.top(5, by_repetition=True)            # Same after workspace reload

Result Output:
  - Raw metrics: Evaluated on individual spectra
  - Aggregated metrics (*): Averaged predictions per sample
  - Workspace reload keeps the same aggregation context
  - Both shown in TabReport output

Benefits of Repetition Aggregation:
  ✓ Noise reduction through averaging
  ✓ One prediction per sample (practical)
  ✓ More robust metrics
  ✓ Better correlation with true values

Next: See 06_deployment/U01_save_load_predict.py - Save and load models
""")

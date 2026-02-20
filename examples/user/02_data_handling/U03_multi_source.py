"""
U03 - Multi-Source: Combining NIR with Other Data Sources
==========================================================

Work with datasets that have multiple feature sources (e.g., NIR + markers).

This tutorial covers:

* Loading multi-source datasets
* Feature augmentation with generator syntax
* Basic multi-source handling
* Combining features from different sources

Prerequisites
-------------
Complete :ref:`U02_multi_datasets` first.

Next Steps
----------
See :ref:`U04_wavelength_handling` for wavelength interpolation and units.

Duration: ~3 minutes
Difficulty: ★★★☆☆
"""

# Standard library imports
import argparse

import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
import nirs4all
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.transforms import Gaussian, Haar, SavitzkyGolay, StandardNormalVariate
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='U03 Multi-Source Example')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()

# =============================================================================
# Section 1: Understanding Multi-Source Data
# =============================================================================
print("\n" + "=" * 60)
print("U03 - Multi-Source Data Handling")
print("=" * 60)

print("""
Multi-source datasets contain features from different instruments or types:

  Example: NIR spectrometer + wet chemistry markers
  - Source 1: NIR spectra (e.g., 1000 wavelengths)
  - Source 2: Lab markers (e.g., 10 chemical values)

nirs4all can handle multiple feature files per sample.
""")

# =============================================================================
# Section 2: Load Multi-Source Dataset
# =============================================================================
print("\n" + "-" * 60)
print("Loading Multi-Source Dataset")
print("-" * 60)

# Multi-source data has multiple X files per partition
dataset_config = DatasetConfigs('sample_data/multi')

print("   Loading: sample_data/multi/")
print("   This contains multiple X sources per sample")

# =============================================================================
# Section 3: Build Pipeline with Feature Augmentation
# =============================================================================
print("\n" + "-" * 60)
print("Building Pipeline")
print("-" * 60)

# Build pipeline for multi-source data
pipeline = [
    # Feature scaling
    MinMaxScaler(),

    # Target scaling
    {"y_processing": MinMaxScaler()},

    # Feature augmentation with preprocessing combinations
    # Uses generator syntax to create multiple variants
    {
        "feature_augmentation": {
            "_or_": [StandardNormalVariate(), SavitzkyGolay(), Gaussian(), Haar()],
            "pick": [2, 3],       # Pick 2 or 3 preprocessing methods
            "then_pick": [1, 3],  # After combination, pick 1-3 final variants
            "count": 2            # Generate 2 combinations
        }
    },

    # Cross-validation
    ShuffleSplit(n_splits=3, random_state=42),

    # Visualization
    "fold_chart",

    # Additional scaling before model
    MinMaxScaler(feature_range=(0.1, 0.8)),

    # Model
    {"model": PLSRegression(n_components=10), "name": "PLS-10"},
]

print("Pipeline includes:")
print("   • MinMaxScaler for features and targets")
print("   • Feature augmentation with preprocessing variants")
print("   • 3-fold cross-validation")
print("   • PLS-10 model")

# =============================================================================
# Section 4: Train the Pipeline
# =============================================================================
print("\n" + "-" * 60)
print("Training Pipeline")
print("-" * 60)

result = nirs4all.run(
    pipeline=pipeline,
    dataset='sample_data/multi',
    name="MultiSource",
    verbose=1,
    save_artifacts=True,
    plots_visible=args.plots
)

predictions = result.predictions

print("\n📊 Training complete!")
print(f"   Generated {predictions.num_predictions} predictions")

# =============================================================================
# Section 5: Display Results
# =============================================================================
print("\n" + "-" * 60)
print("Top Models")
print("-" * 60)

top_models = predictions.top(n=5, rank_metric='rmse')
assert isinstance(top_models, list)
print("Top 5 models by RMSE:")
for idx, pred in enumerate(top_models, 1):
    rmse = pred.get('test_rmse', pred.get('rmse', 0))
    preproc = pred.get('preprocessings', 'N/A')
    print(f"{idx}. {Predictions.pred_short_string(pred, metrics=['rmse'])} - {preproc}")

# =============================================================================
# Section 6: Visualize Results
# =============================================================================
print("\n" + "-" * 60)
print("Creating Visualizations")
print("-" * 60)

analyzer = PredictionAnalyzer(predictions)

# Top-k comparison
fig1 = analyzer.plot_top_k(k=3, rank_metric='rmse')
print("   ✓ Top-k comparison plot")

# Heatmap
fig2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric="rmse",
    rank_partition="val",
    display_metric="rmse",
    display_partition="test",
    aggregation='best'
)
print("   ✓ Heatmap: models vs preprocessing")

# =============================================================================
# Section 7: Model Reuse Demonstration
# =============================================================================
print("\n" + "-" * 60)
print("Model Reuse Demo")
print("-" * 60)

# Get the best model
best_list = predictions.top(n=1, rank_partition="test")
assert isinstance(best_list, list)
best_prediction = best_list[0]
model_id = best_prediction['id']

print(f"Best model: {best_prediction['model_name']} (id: {model_id})")
reference_predictions = best_prediction['y_pred'][:5].flatten()
print(f"Reference predictions (first 5): {reference_predictions}")

# Predict using saved model ID
from nirs4all.pipeline import PipelineRunner

predictor = PipelineRunner(save_artifacts=False, save_charts=False, verbose=0)
test_dataset = DatasetConfigs({
    'X_test': [
        'sample_data/multi/Xval_1.csv.gz',
        'sample_data/multi/Xval_2.csv.gz',
        'sample_data/multi/Xval_3.csv.gz'
    ]
})

print("\nPredicting with saved model...")
reuse_predictions, _ = predictor.predict(model_id, test_dataset, verbose=0)
assert isinstance(reuse_predictions, np.ndarray)
reuse_array = reuse_predictions[:5].flatten()
print(f"Reuse predictions (first 5): {reuse_array}")

# Verify match
is_identical = np.allclose(reuse_array, reference_predictions)
print(f"Predictions match: {'✓ YES' if is_identical else '✗ NO'}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Multi-Source Data Handling:

  1. Multi-source datasets have multiple X files:
     sample_data/multi/Xtrain_1.csv, Xtrain_2.csv, ...

  2. nirs4all automatically loads and concatenates sources

  3. Feature augmentation works across all sources:
     {"feature_augmentation": {...}}

  4. Models see the combined feature space

Advanced options (covered in developer examples):
  • source_branch: Apply different preprocessing per source
  • merge_sources: Control how sources are combined
  • Source-specific scaling and selection

Use cases:
  • NIR + wet chemistry data
  • Multiple spectrometers
  • Sensor fusion applications

Next: U04_wavelength_handling.py - Wavelength interpolation and units
""")

if args.show:
    plt.show()

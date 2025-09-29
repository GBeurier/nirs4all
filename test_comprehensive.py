#!/usr/bin/env python3
"""
Test the comprehensive prediction summary with the new fold predictions
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import *
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from examples.prediction_visualizer import PredictionVisualizer

# Simple test configuration
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
list_of_preprocessors = [Detrend, FirstDerivative]
splitting_strategy = ShuffleSplit(n_splits=3, test_size=.25)
dataset_folder = 'sample_data/regression'

pipeline = [
    x_scaler,
    {"y_processing": y_scaler},
    {"feature_augmentation": {"_or_": list_of_preprocessors, "size": 1, "count": 2}},
    splitting_strategy,
    PLSRegression(n_components=10),
    PLSRegression(n_components=20)
]

pipeline_config = PipelineConfigs(pipeline, "test_pipeline")
dataset_config = DatasetConfigs(dataset_folder)

# Create pipeline
print("🚀 Starting test run...")
runner = PipelineRunner(save_files=False)
global_predictions, results = runner.run(pipeline_config, dataset_config)

# Get the first dataset's predictions
if results:
    dataset_predictions, run_predictions = results[0]  # First dataset tuple
else:
    print("❌ No results found!")
    exit(1)

print("\n" + "=" * 80)
print("📊 COMPREHENSIVE PREDICTION SUMMARY")
print("=" * 80)

# Use PredictionVisualizer for comprehensive summary
visualizer = PredictionVisualizer(dataset_predictions)
summary = visualizer.comprehensive_prediction_summary()
print(summary)

print("\n" + "=" * 80)
print("🔍 DETAILED PREDICTION KEYS")
print("=" * 80)

# Show all prediction keys to verify our new predictions are stored
all_keys = dataset_predictions.list_keys()
print(f"Total predictions stored: {len(all_keys)}")
print("\nAll prediction keys:")
for key in sorted(all_keys):
    print(f"  - {key}")

# Categorize the predictions
train_keys = [k for k in all_keys if 'train' in k]
val_keys = [k for k in all_keys if 'val' in k]
test_keys = [k for k in all_keys if 'test' in k]
avg_keys = [k for k in all_keys if 'avg' in k]
weighted_keys = [k for k in all_keys if 'weighted' in k]

print(f"\n📈 Prediction Categories:")
print(f"  - Train predictions: {len(train_keys)}")
print(f"  - Validation predictions: {len(val_keys)}")
print(f"  - Test predictions: {len(test_keys)}")
print(f"  - Average aggregations: {len(avg_keys)}")
print(f"  - Weighted aggregations: {len(weighted_keys)}")

print("\n" + "=" * 80)
print("✅ Test completed successfully!")
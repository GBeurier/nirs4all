"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Disable emojis in output (set to '1' to disable, '0' to enable)
os.environ['DISABLE_EMOJIS'] = '0'

# Configuration variables
feature_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/classification'

# Build the pipeline
pipeline = [
    # Optional preprocessing steps (commented out for basic demonstration)
    # "chart_2d",
    # feature_scaler,
    # "chart_3d",
    # {"feature_augmentation": {"_or_": preprocessing_options, "size": [1, (1, 2)], "count": 5}},
    cross_validation,
]

# Add Random Forest models with different max_depth values
for max_depth in range(5, 100, 5):
    model_config = {
        "name": f"RandomForest-depth-{max_depth}",
        "model": RandomForestClassifier(max_depth=max_depth)
    }
    pipeline.append(model_config)

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1_classification")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=0)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
best_model_count = 5
ranking_metric = 'accuracy'

# Display top performing models
top_models = predictions.top_k(best_model_count)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")

# Create confusion matrix visualization for top models
analyzer = PredictionAnalyzer(predictions)
# Rank models by accuracy on val partition, display confusion matrix from test partition
confusion_matrix_fig = analyzer.plot_top_k_confusionMatrix(k=4, metric='accuracy', rank_partition='val', display_partition='test')

plt.show()
"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import matplotlib.pyplot as plt

# Third-party imports
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer, Gaussian as Gauss,
    StandardNormalVariate as StdNorm, SavitzkyGolay as SavGol, Haar, MultiplicativeScatterCorrection as MSC
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter


# Configuration variables
feature_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FstDer, SndDer, Gauss,
    StdNorm, SavGol, Haar, MSC
]
split = SPXYSplitter(0.25)
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
# data_path = 'sample_data/classification'
data_path = {
    'X_train': 'sample_data/classification/Xtrain.csv',
    'y_train': 'sample_data/classification/Ytrain.csv',
}

pipeline = [
    # "chart_3d",
    # "chart_2d",
    {"feature_augmentation": [
        Detrend, FstDer, SndDer, Gauss,
        StdNorm, SavGol, Haar, MSC
    ]},
    StandardScaler,
    # "chart_2d",
    "fold_chart",
    SPXYSplitter(0.25),
    "fold_chart",
    ShuffleSplit(n_splits=3, test_size=0.25),
    "fold_chart",
    RandomForestClassifier(max_depth=40)
]



# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1_classification")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
best_model_count = 5
ranking_metric = 'accuracy'

# Display top performing models
top_models = predictions.top(best_model_count)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")

# Create confusion matrix visualization for top models
analyzer = PredictionAnalyzer(predictions)
# Rank models by accuracy on val partition, display confusion matrix from test partition
confusion_matrix_fig = analyzer.plot_top_k_confusionMatrix(k=4, metric='accuracy', rank_partition='val', display_partition='test')

# plt.show()
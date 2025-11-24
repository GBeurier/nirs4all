"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q1 Classification Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()
display_pipeline_plots = args.plots
display_analyzer_plots = args.show


###############
### IMPORTS ###
###############

# Third-party imports
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models.tensorflow.nicon import nicon_classification
from nirs4all.operators.transforms import FirstDerivative, StandardNormalVariate, Haar, MultiplicativeScatterCorrection
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer


pipeline = [
    "y_chart",
    {"feature_augmentation": [
        FirstDerivative,
        StandardNormalVariate,
        Haar,
        MultiplicativeScatterCorrection
    ]},
    StandardScaler,
    ShuffleSplit(n_splits=3, test_size=0.25),
    "fold_chart",
    RandomForestClassifier(
        n_estimators=25,
        max_depth=13,
        verbose=0
    ),
    {
        "model": nicon_classification,
        "train_params": {
            'epochs': 25,
            'batch_size': 1024,
            'verbose': 0
        }
    },
    XGBClassifier(n_estimators=25, verbosity=0)
]


data_categories = {
    'X_train': 'sample_data/classification/Xtrain.csv',
    'y_train': 'sample_data/classification/Ytrain.csv',
}
data_binary = {
    'folder': 'sample_data/binary/',
    'params': {
        'has_header': False,
        'delimiter': ';',
        'decimal_separator': '.'
    }
}

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q1_classification")
dataset_config = DatasetConfigs([data_binary])

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=display_pipeline_plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analysis and visualization
top_models = predictions.top(5)
print(f"Top 5 models:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['accuracy'])} - {prediction['preprocessings']}")

analyzer = PredictionAnalyzer(predictions)

confusion_matrix_fig = analyzer.plot_confusion_matrix(k=4, rank_metric='accuracy', rank_partition='val', display_partition='test')

confusion_matrix_fig_default = analyzer.plot_confusion_matrix(k=8)

candlestick_fig = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='accuracy',
)


heatmap_fig = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    display_metric='accuracy',
)

histogram_fig = analyzer.plot_histogram(
    display_metric='balanced_recall',
)

if display_analyzer_plots:
    plt.show()
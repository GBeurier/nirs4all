"""
Q7 Example - Classification with Target Discretization
=====================================================
Demonstrates conversion of regression data to classification using target discretization
with Random Forest classifiers and confusion matrix visualization.
"""

# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
)
from nirs4all.operators.transformations.targets import RangeDiscretizer
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

# Enable emojis in output
os.environ['DISABLE_EMOJIS'] = '0'

# Configuration variables
feature_scaler = MinMaxScaler()
preprocessing_options = [
    Detrend, FirstDerivative, SecondDerivative, Gaussian,
    StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection
]
cross_validation = ShuffleSplit(n_splits=3, test_size=0.25)
data_path = 'sample_data/regression'

# Build the classification pipeline with target discretization
pipeline = [
    # Data visualization and preprocessing
    "chart_2d",
    feature_scaler,
    "chart_3d",

    # Feature augmentation with preprocessing combinations
    {"feature_augmentation": {"_or_": preprocessing_options, "size": [1, (1, 2)], "count": 5}},

    # Cross-validation setup
    cross_validation,

    # Target discretization (convert regression to classification)
    {"y_processing": RangeDiscretizer([14, 20, 30, 50])},  # Define class boundaries
    # Alternative discretization methods (commented out):
    # {"y_processing": IntegerKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')},  # Extension of KBinsDiscretizer to integer outputs
    # Discretization using 2 successive methods (bins + encoding):
    # {"y_processing": KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')},
    # {"y_processing": LabelEncoder()},
]

# Add Random Forest models with different max_depth values
for max_depth in range(5, 20, 5):
    model_config = {
        "name": f"RandomForest-depth-{max_depth}",
        "model": RandomForestClassifier(max_depth=max_depth)
    }
    pipeline.append(model_config)

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Q7_classification")
dataset_config = DatasetConfigs(data_path)

# Run the classification pipeline
runner = PipelineRunner(save_files=False, verbose=0)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Create confusion matrix visualization for top models
analyzer = PredictionAnalyzer(predictions)
confusion_matrix_fig = analyzer.plot_top_k_confusionMatrix(k=3, metric='accuracy', partition='val')

# plt.show()
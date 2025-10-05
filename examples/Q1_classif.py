import os

from nirs4all.dataset.predictions import Predictions
# set to False to enable emojis
os.environ['DISABLE_EMOJIS'] = '0'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import *
from nirs4all.pipeline import PipelineConfigs, PipelineRunner

x_scaler = MinMaxScaler() # StandardScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), LogTransform()
list_of_preprocessors = [Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection]
splitting_strategy = ShuffleSplit(n_splits=3, test_size=.25)
dataset_folder = 'sample_data/classification'

pipeline = [
    # "chart_2d",
    # x_scaler,
    # "chart_3d",
    # {"feature_augmentation": {"_or_": list_of_preprocessors, "size": [1, (1, 2)], "count": 5}}, # Generate all elements of size 1 and of order 1 or 2 (ie. "Gaussian", ["SavitzkyGolay", "Log"], etc.)
    splitting_strategy,
]

for i in range(5, 20, 5):
    model = {
        "name": f"RF-depth-{i}",
        "model": RandomForestClassifier(max_depth=i)
    }
    pipeline.append(model)

pipeline_config = PipelineConfigs(pipeline, "pipeline_Q1_classif")
dataset_config = DatasetConfigs(dataset_folder)

# Create pipeline
runner = PipelineRunner(save_files=False, verbose=0)
run_predictions, other_predictions = runner.run(pipeline_config, dataset_config)
print(run_predictions)
# Get top models to verify the real model names are displayed correctly
best_count = 5
rank_metric = 'accuracy'
top_n = run_predictions.top_k(best_count)
print(f"Top {best_count} models by {rank_metric}:")
for i, pred in enumerate(top_n):
    print(f"{i+1}. {Predictions.pred_short_string(pred, metrics=[rank_metric])} - {pred['preprocessings']}")

analyzer = PredictionAnalyzer(run_predictions)
fig = analyzer.plot_top_k_confusionMatrix(k=3, metric='accuracy', partition='test')
plt.show()
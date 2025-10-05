import os
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
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer

from nirs4all.operators.transformations.targets import IntegerKBinsDiscretizer, RangeDiscretizer



x_scaler = MinMaxScaler() # StandardScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(), LogTransform()
list_of_preprocessors = [Detrend, FirstDerivative, SecondDerivative, Gaussian, StandardNormalVariate, SavitzkyGolay, Haar, MultiplicativeScatterCorrection]
splitting_strategy = ShuffleSplit(n_splits=3, test_size=.25)
dataset_folder = 'sample_data/regression'

pipeline = [
    "chart_2d",
    x_scaler,
    "chart_3d",
    {"feature_augmentation": {"_or_": list_of_preprocessors, "size": [1, (1, 2)], "count": 5}}, # Generate all elements of size 1 and of order 1 or 2 (ie. "Gaussian", ["SavitzkyGolay", "Log"], etc.)
    splitting_strategy,
    # {"y_processing": IntegerKBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')},
    {"y_processing": RangeDiscretizer([14, 20, 30, 50])}
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


analyzer = PredictionAnalyzer(run_predictions)
fig = analyzer.plot_top_k_confusionMatrix(k=3, metric='accuracy', partition='val')
plt.show()
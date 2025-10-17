"""
Q1 Classification Example - Random Forest Classification Pipeline
===============================================================
Demonstrates NIRS classification analysis using Random Forest models with various max_depth parameters.
Shows confusion matrix visualization for model performance evaluation.
"""

# Standard library imports
import os
os.environ['DISABLE_EMOJIS'] = '0'

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from nirs4all.dataset import DatasetConfigs
from nirs4all.dataset.predictions import Predictions
from nirs4all.dataset.prediction_analyzer import PredictionAnalyzer
from nirs4all.operators.transformations import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer, Gaussian as Gauss,
    StandardNormalVariate as SNV, SavitzkyGolay as SavGol, Haar, MultiplicativeScatterCorrection as MSC,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.models.cirad_tf import nicon, transformer_VG, transformer, decon, customizable_nicon

from custom_NN import (
    resnet_se, inception_time, tcn_noncausal, conv_transformer,
    convmixer1d, cnn_pls_head
)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_path = 'sample_data/Hiba23'
pipeline = [
    {"split": RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)},
    # {"split": SPXYSplitter(test_size=0.25)},
    {"feature_augmentation": [
        MSC,
        SNV,
        RSNV,
        LSNV,
        Gauss,
        SavGol,
        Haar,
        FstDer,
        SndDer,
        Detrend,
        Wavelet('coif3')
    ]},
    MinMaxScaler(clip=True),
    {"y_processing": MinMaxScaler()},
    {
        # "model": nicon,
        "model": customizable_nicon,
        "train_params": {
            "epochs": 2000,
            "patience": 200,
            # "learning_rate": 0.0005,
            "batch_size": 1024,
            "cyclic_lr": True,
            "base_lr": 0.001,
            "max_lr": 0.01,
            "cycle_steps": 200,
            # "verbose": 2  # 0=silent, 2=one line per epoch
        },
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,
            "sample": "hyperband",
            "approach": "single",
            "model_params": {
                "filters_1": [8, 16, 32, 64],
                "filters_2": [8, 16, 32, 64],
                "filters_3": [8, 16, 32, 64]
            },
            "train_params": {
                "epochs": 50,
                "learning_rate": 0.01,
                "verbose": 0
            }
        },
    },
    # {
    #     "model": decon,
    #     "train_params": {
    #         "epochs": 2000,
    #         "patience": 200,
    #         # "learning_rate": 0.0005,
    #         "batch_size": 1024,
    #         "cyclic_lr": True,
    #         "base_lr": 0.001,
    #         "max_lr": 0.01,
    #         "cycle_steps": 200,
    #         # "verbose": 2  # 0=silent, 2=one line per epoch
    #     },
    # },
    # {
    #     "model": PLSRegression(),
    #     "name": "PLS-Finetuned",
    #     "finetune_params": {
    #         "n_trials": 30,
    #         "verbose": 2,
    #         "approach": "single",
    #         "eval_mode": "best",
    #         "sample": "grid",
    #         "model_params": {
    #             'n_components': ('int', 1, 40),
    #         },
    #     }
    # },
    # RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    # XGBRegressor(n_estimators=50, max_depth=10, random_state=42),
    # LGBMRegressor(n_estimators=50, max_depth=10, random_state=42)
]
# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "Hiba")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=True)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)


# Analysis and visualization
best_model_count = 5
ranking_metric = 'rmse'  # Options: 'rmse', 'mae', 'r2'

# Display top performing models
top_models = predictions.top(best_model_count, ranking_metric)
print(f"Top {best_model_count} models by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=[ranking_metric])} - {prediction['preprocessings']}")
top_models[0].save_to_csv("Q1_regression_best_model.csv")

# # Create visualizations
analyzer = PredictionAnalyzer(predictions)
# Plot comparison of top models
fig1 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse')
fig2 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse', rank_partition='test')
# fig3 = analyzer.plot_top_k_comparison(k=3, rank_metric='rmse', rank_partition='train')
plt.show()
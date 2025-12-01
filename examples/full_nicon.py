"""
Batch Regression Analysis - Comprehensive Pipeline for Multiple Datasets
=========================================================================
Runs complex regression analysis on all datasets in the regression/ folder.
Includes ML models, DL models, and various preprocessing combinations.
Generates top_k plots, heatmaps, candlestick plots, and histograms.
"""

# Standard library imports
import argparse
import os
os.environ['DISABLE_EMOJIS'] = '0'

from matplotlib import pyplot as plt

# Third-party imports - ML Models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler

# NIRS4All imports - Sample augmentation transforms
from nirs4all.operators.transforms import (
    Rotate_Translate,
    Spline_Y_Perturbations,
    Spline_X_Simplification,
    GaussianAdditiveNoise,
    MultiplicativeNoise,
    LinearBaselineDrift,
    PolynomialBaselineDrift,
    WavelengthShift,
    WavelengthStretch,
    LocalWavelengthWarp,
    SmoothMagnitudeWarp,
    GaussianSmoothingJitter,
    UnsharpSpectralMask,
    ChannelDropout,
    MixupAugmenter,
    ScatterSimulationMSC,
    # Feature augmentation transforms (commented - uncomment if using feature_augmentation)
    Detrend, FirstDerivative, SecondDerivative,
    Gaussian, StandardNormalVariate, SavitzkyGolay,
    Haar, MultiplicativeScatterCorrection,
    RobustStandardNormalVariate, LocalStandardNormalVariate, Wavelet,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)
from nirs4all.operators.models.sklearn.fckpls import FCKPLS

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.models.pytorch.nicon import nicon as nicon_torch, customizable_nicon as customizable_nicon_torch
from nirs4all.operators.splitters import SPXYSplitter, SPXYGFold, BinnedStratifiedGroupKFold


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Batch Regression Analysis')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at the end')
args = parser.parse_args()

print("=" * 80)
print("BATCH REGRESSION ANALYSIS")
print("=" * 80)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# All regression datasets
data_paths = [
    'sample_data/nitro_reg_merged/Digestibility_0.8',
    # 'sample_data/nitro_reg_merged/Hardness_0.8',
    # 'sample_data/nitro_reg_merged/Tannin_0.8'
    # Add your regression dataset paths here
]


pipeline = [
    # Y processing must come before sample augmentation
    {"y_processing": MinMaxScaler()},

    {
        "split": SPXYGFold,
        "split_params": {
            "n_splits": 1,
            "test_size": 0.2,
            # "random_state": {"_range_": [1, 10000000], "count": 100},
            "aggregation": "mean"
        },
        "group": "ID"
    },
    "spectra_dist",
    "y_chart",
    # "fold_chart",
    {
        "split": BinnedStratifiedGroupKFold,
        "split_params": {
            "n_splits": 3,
            "n_bins": 7,
            "shuffle": True,
            # "random_state": {"_range_": [1, 10000000], "count": 20},
            "random_state": 1337
        },
        "group": "ID"
    },

    "y_chart",
    "spectra_dist",
    # "fold_chart",
    # 'chart_2d',  # 2D Visualization of augmented features

    MinMaxScaler(),

    # {
    #     'model': nicon_torch,
    #     'train_params': {'epochs': 50, 'verbose': 0},
    #     'name': 'nicon_torch'
    # },
    {
        'model': customizable_nicon_torch,
        'name': 'customizable_nicon_torch',
        "finetune_params": {
            "n_trials": 300,
            "verbose": 2,
            "sample": "tpe",  # hyperband
            "eval_mode": "avg",
            # "approach": "single",
            "model_params": {
                'spatial_dropout': (float, 0.01, 0.5),
                'filters1': [4, 8, 16, 32, 64, 128, 256],
                'kernel_size1': [3, 5, 7, 9, 11, 13, 15],
                'strides1': [1, 2, 3, 4, 5],
                'activation1': ['relu', 'selu', 'elu', 'swish', 'sigmoid', 'tanh'],
                'dropout_rate': (float, 0.01, 0.5),
                'filters2': [4, 8, 16, 32, 64, 128, 256],
                'kernel_size2': [3, 5, 7, 9, 11, 13, 15],
                'strides2': [1, 2, 3, 4, 5],
                'activation2': ['relu', 'selu', 'elu', 'swish', 'sigmoid', 'tanh'],
                'normalization_method1': ['BatchNormalization', 'LayerNormalization'],
                'filters3': [4, 8, 16, 32, 64, 128, 256],
                'kernel_size3': [3, 5, 7, 9, 11, 13, 15],
                'strides3': [1, 2, 3, 4, 5],
                'activation3': ['relu', 'selu', 'elu', 'swish', 'sigmoid', 'tanh'],
                'normalization_method2': ['BatchNormalization', 'LayerNormalization'],
                'dense_units': [4, 8, 16, 32, 64, 128, 256],
                'dense_activation': ['relu', 'selu', 'elu', 'swish', 'sigmoid', 'tanh'],
            },
            "train_params": {
                "epochs": 100,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 2000,
            "patience": 150,
            "batch_size": 2048,
            "cyclic_lr": True,
            "cyclic_lr_mode": "triangular2",
            "base_lr": 0.0001,
            "max_lr": 0.01,
            "step_size": 100,
            "verbose": 1
        },
    }


    # # Sample augmentation for regression
    # {
    #     "sample_augmentation": {
    #         "transformers": [
    #             Rotate_Translate(p_range=2, y_factor=3),
    #             Spline_Y_Perturbations(perturbation_intensity=0.005, spline_points=10),
    #             Spline_X_Simplification(spline_points=50, uniform=True),
    #             GaussianAdditiveNoise(sigma=0.01),
    #             MultiplicativeNoise(sigma_gain=0.05),
    #             LinearBaselineDrift(),
    #             PolynomialBaselineDrift(),
    #             WavelengthShift(),
    #             WavelengthStretch(),
    #             LocalWavelengthWarp(),
    #             SmoothMagnitudeWarp(),
    #             GaussianSmoothingJitter(),
    #             UnsharpSpectralMask(),
    #             ChannelDropout(),
    #             MixupAugmenter(),
    #             ScatterSimulationMSC(),
    #         ],
    #         "ref_percentage": 4.0,
    #         "selection": "random",
    #         "random_state": 42
    #     }
    # },

    # "fold_chart",
    # "augment_details_chart",

    # Comprehensive feature augmentation with many preprocessing combinations
    # Uncomment and import transforms above if using feature_augmentation
    # {"feature_augmentation": [
    #     [MSC(scale=False), EMSC, AreaNormalization],
    #     [MSC(scale=False), EMSC, SNV],
    #     [EMSC, Gaussian(order=1, sigma=2), RSNV],
    #     SNV,
    #     Haar,
    #     [SNV, SndDer],
    # ]},
    # "chart_2d",


]


# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline_config = PipelineConfigs(pipeline, name="BatchRegression")
dataset_config = DatasetConfigs(data_paths)

runner = PipelineRunner(save_files=True, verbose=1, plots_visible=args.plots)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

best_model_count = 20
ranking_metric = 'rmse'

# Display top models overall
top_models = predictions.top(best_model_count, ranking_metric)
print(f"\nTop {best_model_count} models overall by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['rmse', 'r2', 'mae'])} - {prediction['preprocessings']}")

# # Per-dataset analysis
# for dataset_name, dataset_prediction in predictions_per_dataset.items():
#     print(f"\n{'='*80}")
#     print(f"Dataset: {dataset_name}")
#     print(f"{'='*80}")

#     dataset_predictions = dataset_prediction['run_predictions']

#     # Top by RMSE
#     top_rmse = dataset_predictions.top(n=5, rank_metric='rmse', rank_partition='test')
#     print("\nTop 5 by RMSE (test):")
#     for idx, model in enumerate(top_rmse):
#         print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'], partition=['val', 'test'])}")

#     # Top by R2
#     top_r2 = dataset_predictions.top(n=5, rank_metric='r2', rank_partition='test')
#     print("\nTop 5 by R2 (test):")
#     for idx, model in enumerate(top_r2):
#         print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'], partition=['val', 'test'])}")

# ============================================================================
# SAMPLE AGGREGATION EXAMPLE
# ============================================================================
# When you have multiple measurements per sample (e.g., 4 spectra per sample ID),
# you can aggregate predictions to get one prediction per sample.
print("\n" + "=" * 80)
print("SAMPLE AGGREGATION EXAMPLE")
print("=" * 80)

# Check if the dataset has an 'ID' column for aggregation
for dataset_name, dataset_prediction in predictions_per_dataset.items():
    dataset_predictions = dataset_prediction['run_predictions']

    # Get the first prediction to check metadata
    sample_pred = dataset_predictions.filter_predictions(partition='test', load_arrays=True)
    if sample_pred:
        metadata = sample_pred[0].get('metadata', {})
        if 'ID' in metadata:
            print(f"\nDataset '{dataset_name}' has 'ID' column - showing aggregated results:")

            # Compare non-aggregated vs aggregated top models
            print("\n  Non-aggregated Top 3 by RMSE (test):")
            top_regular = dataset_predictions.top(n=3, rank_metric='rmse', rank_partition='test')
            for idx, model in enumerate(top_regular):
                print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'])}")

            print("\n  Aggregated by 'ID' Top 3 by RMSE (test):")
            top_aggregated = dataset_predictions.top(
                n=3,
                rank_metric='rmse',
                rank_partition='test',
                aggregate='ID'  # Aggregate predictions by sample ID
            )
            for idx, model in enumerate(top_aggregated):
                n_samples = model.get('n_samples', '?')
                agg_flag = " [aggregated]" if model.get('aggregated') else ""
                print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'])} (n={n_samples}){agg_flag}")

            # Show one detailed example with aggregation
            if top_aggregated:
                best_agg = top_aggregated[0]
                y_true = best_agg.get('y_true')
                y_pred = best_agg.get('y_pred')
                if y_true is not None and y_pred is not None:
                    model_name = best_agg.get('model_name')
                    n_pred = len(y_pred)
                    print("\n  Best aggregated model details:")
                    print(f"    Model: {model_name}")
                    print(f"    Samples after aggregation: {n_pred}")
        else:
            print(f"\nDataset '{dataset_name}' does not have 'ID' column for aggregation")
    else:
        print(f"\nNo test predictions found for dataset '{dataset_name}'")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

analyzer = PredictionAnalyzer(predictions)

# Top K plots by RMSE
fig_top_rmse = analyzer.plot_top_k(
    k=10,
    rank_metric='rmse',
    rank_partition='val'
)

fig_top_rmse_test = analyzer.plot_top_k(
    k=10,
    rank_metric='rmse',
    rank_partition='test'
)

# Top K plots by R2
fig_top_r2 = analyzer.plot_top_k(
    k=10,
    rank_metric='r2',
    rank_partition='val'
)

# Heatmap per dataset
fig_heatmap_model_dataset = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='rmse',
    display_metric='rmse',
)


if args.show:
    plt.show()

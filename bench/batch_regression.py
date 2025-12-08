"""
Batch Regression Analysis - Comprehensive Pipeline for Multiple Datasets
=========================================================================
Runs complex regression analysis on all datasets in the regression/ folder.
Includes ML models, DL models, statistical models, and various preprocessing combinations.
Generates heatmaps, candlestick plots, and histograms for analysis.
"""

# Standard library imports
import argparse
import os
os.environ['DISABLE_EMOJIS'] = '0'

from matplotlib import pyplot as plt

# Third-party imports - ML Models
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    ElasticNet, Ridge, Lasso, BayesianRidge,
    HuberRegressor, Lars, LassoLars
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit, KFold, RepeatedKFold, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer,
    Gaussian as Gauss, StandardNormalVariate as SNV, SavitzkyGolay as SavGol,
    Haar, MultiplicativeScatterCorrection as MSC, Derivate,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYSplitter
from nirs4all.operators.models.tensorflow.nicon import nicon, decon, customizable_nicon

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
    'regression/Digestibility_0.8',
    'regression/Hardness_0.8',
    'regression/Tannin_0.8',
]


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================
pipeline = [
    # Cross-validation setup
    # {"split": RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)},
    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"},
    "fold_ID",
    # Comprehensive feature augmentation with many preprocessing combinations
    {"feature_augmentation": [
    #     # Single transformations
    #     MSC,
    #     SNV,
        EMSC,
        # Haar,
    #     Gauss,
    #     # SavGol,
        FstDer,
        # SndDer,
        # Detrend,
        RSNV,
        # LSNV,
        # Wavelet('db4'),
        # Wavelet('coif3'),

    #     # Double combinations - Scatter correction + Smoothing/Derivative
    #     [MSC, SavGol],
    #     # [MSC, FstDer],
    #     # [MSC, SndDer],
    #     # [MSC, Gauss],
    #     # [SNV, SavGol],
    #     # [SNV, FstDer],
    #     # [SNV, SndDer],
    #     # [SNV, Gauss],
        [EMSC, SavGol],
    #     # [EMSC, FstDer],

    #     # Advanced combinations
        # [MSC, SavGol(deriv=3)],
    #     # [MSC, SavGol(deriv=4)],
    #     # [SNV, SavGol(deriv=3)],
    #     # [SNV, SavGol(deriv=4)],
    #     # [Detrend, SavGol, Derivate(order=3)],
    #     # [MSC, Gauss, Derivate(order=3)],
    #     # [SNV, Gauss, Derivate(order=3)],
        [LSNV, SavGol(deriv=2)],
        # [RSNV, SavGol(deriv=1)],

    #     # # Wavelet combinations
    #     # [MSC, Wavelet('db4')],
    #     # [SNV, Wavelet('sym5')],
    #     # [MSC, SavGol, Wavelet('db4')],

    #     # Triple combinations for complex spectra
        # [Detrend, SNV, SavGol(deriv=3)],
        # [Detrend, SNV, SavGol(deriv=2)],
    #     # [Detrend, MSC, SavGol(deriv=3)],
        [AreaNormalization, SavGol, Derivate(order=1)],
        [Haar, SNV],
    ]},

    # Feature scaling
    RobustScaler(),

    # Target scaling
    {"y_processing": MinMaxScaler(feature_range=(0.05, 0.95))},

    # ========================================================================
    # STATISTICAL / LINEAR MODELS
    # ========================================================================
    # PLS with different components
    {"model": PLSRegression(n_components=3), "name": "PLS_3"},
    {"model": PLSRegression(n_components=4), "name": "PLS_4"},
    {"model": PLSRegression(n_components=5), "name": "PLS_5"},
    {"model": PLSRegression(n_components=6), "name": "PLS_6"},
    {"model": PLSRegression(n_components=7), "name": "PLS_7"},
    {"model": PLSRegression(n_components=8), "name": "PLS_8"},
    {"model": PLSRegression(n_components=9), "name": "PLS_9"},
    {"model": PLSRegression(n_components=10), "name": "PLS_10"},
    {"model": PLSRegression(n_components=11), "name": "PLS_11"},
    {"model": PLSRegression(n_components=12), "name": "PLS_12"},
    {"model": PLSRegression(n_components=13), "name": "PLS_13"},
    {"model": PLSRegression(n_components=14), "name": "PLS_14"},
    {"model": PLSRegression(n_components=15), "name": "PLS_15"},
    {"model": PLSRegression(n_components=16), "name": "PLS_16"},
    {"model": PLSRegression(n_components=17), "name": "PLS_17"},
    {"model": PLSRegression(n_components=18), "name": "PLS_18"},
    {"model": PLSRegression(n_components=19), "name": "PLS_19"},
    {"model": PLSRegression(n_components=20), "name": "PLS_20"},
    {"model": PLSRegression(n_components=21), "name": "PLS_21"},
    {"model": PLSRegression(n_components=22), "name": "PLS_22"},
    {"model": PLSRegression(n_components=23), "name": "PLS_23"},
    {"model": PLSRegression(n_components=24), "name": "PLS_24"},
    {"model": PLSRegression(n_components=25), "name": "PLS_25"},
    {"model": PLSRegression(n_components=26), "name": "PLS_26"},
    {"model": PLSRegression(n_components=27), "name": "PLS_27"},
    {"model": PLSRegression(n_components=28), "name": "PLS_28"},
    {"model": PLSRegression(n_components=29), "name": "PLS_29"},
    {"model": PLSRegression(n_components=30), "name": "PLS_30"},

    # Regularized linear models
    {"model": Ridge(alpha=1.0), "name": "Ridge"},
    {"model": Lasso(alpha=0.1), "name": "Lasso"},
    {"model": ElasticNet(alpha=0.1, l1_ratio=0.5), "name": "ElasticNet"},
    {"model": BayesianRidge(), "name": "BayesianRidge"},
    # # {"model": HuberRegressor(), "name": "HuberRegressor"},
    # # {"model": Lars(), "name": "Lars"},
    # # {"model": LassoLars(alpha=0.1), "name": "LassoLars"},

    # ========================================================================
    # MACHINE LEARNING MODELS
    # ========================================================================
    # Ensemble methods
    # # {"model": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), "name": "GradientBoosting"}, # trop lent
    # # {"model": AdaBoostRegressor(n_estimators=50, random_state=42), "name": "AdaBoost"},
    {
        "model": RandomForestRegressor,
        "name": "RandomForest-Finetuned",
        "finetune_params": {
            "n_trials": 30,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "tpe",                       # "random", "grid", "bayes", "hyperband", "skopt", "tpe", "cmaes"
            "model_params": {
                'n_estimators': ('int', 50, 200),
                'max_depth': ('int', 5, 20),
                'min_samples_split': ('int', 2, 10),
                'min_samples_leaf': ('int', 1, 5),
            },
            "train_params": {
                "n_jobs": -1
            }
        },
        "train_params": {
            "n_jobs": -1
        }
    },
    {"model": ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), "name": "ExtraTrees"},
    {"model": BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1), "name": "Bagging"},

    # Instance-based
    # {"model": KNeighborsRegressor(n_neighbors=5, weights='distance'), "name": "KNN_5"},
    # {"model": KNeighborsRegressor(n_neighbors=10, weights='distance'), "name": "KNN_10"},

    # # Support Vector Regression
    # # {"model": SVR(kernel='rbf', C=1.0, epsilon=0.1), "name": "SVR_RBF"}, # trop lent
    # {"model": SVR(kernel='linear', C=1.0), "name": "SVR_Linear"},

    # Neural Network (sklearn)
    {"model": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), "name": "MLP_64_32"},
    {"model": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42), "name": "MLP_128_64_32"},

    # # Boosting models (if available)
    # {
    #     "model": LGBMRegressor,
    #     "name": "LightGBM-Finetuned",
    #     "finetune_params": {
    #         "n_trials": 50,
    #         "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
    #         "approach": "single",                                  # "grouped", "individual", or "single"
    #         "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
    #         "sample": "tpe",                       # "random", "grid", "
    #         "model_params": {
    #             'n_estimators': ('int', 50, 200),
    #             'max_depth': ('int', 5, 12),
    #             'learning_rate': (float, 0.01, 0.3),
    #         },
    #         "train_params": {
    #             "n_jobs": 1,
    #             "verbose": -1,
    #             "min_child_samples": 50,
    #             "num_leaves": 31,
    #             "max_depth": -1,
    #         }
    #     },
    #     "train_params": {
    #         "n_jobs": 1,
    #         "verbose": -1,
    #         "min_child_samples": 50,
    #         "num_leaves": 31,
    #         "max_depth": -1,
    #     }
    # },

    {
        "model": XGBRegressor,
        "name": "XGBoost-Finetuned",
        "finetune_params": {
            "n_trials": 50,
            "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
            "approach": "single",                                  # "grouped", "individual", or "single"
            "eval_mode": "best",                    # "best" or "avg" (for grouped approach)
            "sample": "tpe",                       # "random", "grid", "
            "model_params": {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int', 5, 12),
                'learning_rate': (float, 0.01, 0.3),
            },
            "train_params": {
                "verbosity": 0,
                "n_jobs": -1,
                "tree_method": 'hist',
                "device": 'cuda',
            }
        },
        "train_params": {
            "verbosity": 0,
            "n_jobs": -1,
            "tree_method": 'hist',
            "device": 'cuda',
        }
    },

    {"model": CatBoostRegressor(iterations=500, depth=10, learning_rate=0.1, random_state=42, verbose=0, thread_count=-1, allow_writing_files=False), "name": "CatBoost"},  # task_type="GPU",devices="0",

    # # ========================================================================
    # # DEEP LEARNING MODELS
    # # ========================================================================
   {
        "model": customizable_nicon,
        "name": "Nicon",
        "finetune_params": {
            "n_trials": 150,
            "verbose": 2,
            "sample": "tpe",
            "approach": "single",
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
                "epochs": 40,
                "verbose": 0
            }
        },
        "train_params": {
            "epochs": 1000,
            "patience": 150,
            "batch_size": 2048,
            "cyclic_lr": True,
            "base_lr": 0.0005,
            "max_lr": 0.01,
            "step_size": 100,
            "verbose": 0
        },
    },
    {
        "model": decon,
        "name": "DECON",
        "train_params": {
            "epochs": 1000,
            "patience": 100,
            "batch_size": 1024,
            "cyclic_lr": True,
            "base_lr": 0.0005,
            "max_lr": 0.01,
            "step_size": 100,
            "verbose": 0
        },
    },
]


# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline_config = PipelineConfigs(pipeline, name="BatchRegression")
dataset_config = DatasetConfigs(data_paths)

runner = PipelineRunner(save_files=True, verbose=1, plots_visible=args.plots, random_state=42)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

best_model_count = 10
ranking_metric = 'rmse'

# Display top models overall
top_models = predictions.top(best_model_count, ranking_metric)
print(f"\nTop {best_model_count} models overall by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['rmse', 'r2', 'mae'])} - {prediction['preprocessings']}")

# Per-dataset analysis
for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    dataset_predictions = dataset_prediction['run_predictions']

    # Top by RMSE
    top_rmse = dataset_predictions.top(n=5, rank_metric='rmse', rank_partition='test')
    print("\nTop 5 by RMSE (test):")
    for idx, model in enumerate(top_rmse):
        print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'], partition=['val', 'test'])}")

    # Top by R2
    top_r2 = dataset_predictions.top(n=5, rank_metric='r2', rank_partition='test')
    print("\nTop 5 by R² (test):")
    for idx, model in enumerate(top_r2):
        print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'], partition=['val', 'test'])}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

analyzer = PredictionAnalyzer(predictions)

# Top K plots
fig_topk_rmse = analyzer.plot_top_k(k=5, rank_metric='rmse', rank_partition='test')
fig_topk_r2 = analyzer.plot_top_k(k=5, rank_metric='r2', rank_partition='test')

# Heatmaps
fig_heatmap_model_dataset = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='rmse',
    display_metric='rmse',
    rank_partition='test',
    display_partition='test',
)

fig_heatmap_model_dataset_r2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='r2',
    display_metric='r2',
    rank_partition='test',
    display_partition='test',
)

fig_heatmap_preproc_dataset = analyzer.plot_heatmap(
    x_var="preprocessings",
    y_var="dataset_name",
    rank_metric='rmse',
    display_metric='rmse',
)

fig_heatmap_model_preproc = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="preprocessings",
    rank_metric='rmse',
    display_metric='rmse',
)

# Candlestick plots
fig_candlestick_model = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='rmse',
    display_partition='test',
)

fig_candlestick_dataset = analyzer.plot_candlestick(
    variable="dataset_name",
    display_metric='rmse',
    display_partition='test',
)

fig_candlestick_preproc = analyzer.plot_candlestick(
    variable="preprocessings",
    display_metric='rmse',
    display_partition='test',
)

fig_candlestick_r2 = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='r2',
    display_partition='test',
)

# Histograms
fig_hist_rmse = analyzer.plot_histogram(display_metric='rmse')
fig_hist_mae = analyzer.plot_histogram(display_metric='mae')
fig_hist_r2 = analyzer.plot_histogram(display_metric='r2')
fig_hist_mape = analyzer.plot_histogram(display_metric='mape')

print("\nVisualization complete!")

if args.show:
    plt.show()

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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# PLS operators
from nirs4all.operators.models.sklearn import OPLS

# Optional boosting libraries
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

# NIRS4All imports - Sample augmentation transforms
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions

# from nirs4all.operators.transforms.nirs import (
#     AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
# )
# PLS operators
from nirs4all.operators.models.sklearn import IKPLS, MBPLS, OPLSDA, PLSDA, SIMPLS, DiPLS, SparsePLS
from nirs4all.operators.models.sklearn.fckpls import FCKPLS
from nirs4all.operators.models.sklearn.ipls import IntervalPLS
from nirs4all.operators.models.sklearn.kopls import KOPLS
from nirs4all.operators.models.sklearn.lwpls import LWPLS
from nirs4all.operators.models.sklearn.nlpls import KernelPLS
from nirs4all.operators.models.sklearn.oklmpls import OKLMPLS, PolynomialFeaturizer
from nirs4all.operators.models.sklearn.recursive_pls import RecursivePLS
from nirs4all.operators.models.sklearn.robust_pls import RobustPLS
from nirs4all.operators.transforms import (
    ChannelDropout,
    GaussianAdditiveNoise,
    GaussianSmoothingJitter,
    LinearBaselineDrift,
    LocalWavelengthWarp,
    MixupAugmenter,
    MultiplicativeNoise,
    PolynomialBaselineDrift,
    Rotate_Translate,
    ScatterSimulationMSC,
    # Feature augmentation transforms (commented - uncomment if using feature_augmentation)
    # Detrend, FirstDerivative, SecondDerivative,
    # Gaussian, StandardNormalVariate, SavitzkyGolay,
    # Haar, MultiplicativeScatterCorrection,
    # RobustStandardNormalVariate, LocalStandardNormalVariate, Wavelet,
    SmoothMagnitudeWarp,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    UnsharpSpectralMask,
    WavelengthShift,
    WavelengthStretch,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

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
    'nitro_regression_unmerged/Digestibility_0.8',
    'nitro_regression_unmerged/Hardness_0.8',
    'nitro_regression_unmerged/Tannin_0.8'
    # Add your regression dataset paths here
]

# ============================================================================
# STACKING CONFIGURATION
# ============================================================================
# Define base estimators for regression stacking
base_estimators = [
    ('pls_5', PLSRegression(n_components=5)),
    ('pls_7', PLSRegression(n_components=7)),
    ('pls_10', PLSRegression(n_components=10)),
    ('pls_14', PLSRegression(n_components=14)),
    ('pls_15', PLSRegression(n_components=15)),
    ('pls_16', PLSRegression(n_components=16)),
    ('opls_1_5', OPLS(n_components=1, pls_components=5)),
    ('opls_2_5', OPLS(n_components=2, pls_components=5)),
    ('opls_1_6', OPLS(n_components=1, pls_components=6)),
    ('opls_2_6', OPLS(n_components=2, pls_components=6)),
    ('opls_1_7', OPLS(n_components=1, pls_components=7)),
    ('opls_2_7', OPLS(n_components=2, pls_components=7)),
    ('opls_1_12', OPLS(n_components=1, pls_components=12)),
    ('opls_2_12', OPLS(n_components=2, pls_components=12)),
    ('opls_1_13', OPLS(n_components=1, pls_components=13)),
    ('opls_2_13', OPLS(n_components=2, pls_components=13)),
    ('opls_1_14', OPLS(n_components=1, pls_components=14)),
    ('opls_2_14', OPLS(n_components=2, pls_components=14)),
    ('opls_1_15', OPLS(n_components=1, pls_components=15)),
    ('opls_2_15', OPLS(n_components=2, pls_components=15)),
    ('opls_1_16', OPLS(n_components=1, pls_components=16)),
    ('opls_2_16', OPLS(n_components=2, pls_components=16)),
    ('KernelPLS', KernelPLS(n_components=5, kernel='rbf', gamma=0.1, backend='numpy')),
    ('IKPLS', IKPLS(n_components=10, backend='numpy')),
    ('FCKPLS', FCKPLS(n_components=5, alphas=(0.0, 1.0, 2.0), sigmas=(2.0,), kernel_size=15, backend='numpy')),
    ('ridge', Ridge(alpha=1.0)),
    ('catboost', CatBoostRegressor(iterations=400, depth=8, learning_rate=0.1, random_state=42, verbose=0, allow_writing_files=False)),
    ('xgboost', XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.1, random_state=42, verbosity=0)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('svr', SVR(kernel='rbf')),
    ('lgbm', LGBMRegressor(n_estimators=250, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)),
    ('knn', KNeighborsRegressor(n_neighbors=5)),
    ("mlp_32_8_64", MLPRegressor(hidden_layer_sizes=(32, 8, 64), max_iter=500, random_state=42)),
    ("mlp_32_128_64", MLPRegressor(hidden_layer_sizes=(32, 128, 64), max_iter=500, random_state=42)),
    ("mlp_128_32_16_64", MLPRegressor(hidden_layer_sizes=(128, 32, 16, 64), max_iter=500, random_state=42)),
    ("extratrees", ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42)),
]

# Create Stacking Regressor with Ridge as meta-learner
stacking_regressor = StackingRegressor(
    estimators=base_estimators,
    final_estimator=RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    cv=2,
    passthrough=False,
    n_jobs=-1
)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================
pipeline = [
    # Y processing must come before sample augmentation
    {"y_processing": MinMaxScaler()},

    # Sample augmentation for regression
    {
        "sample_augmentation": {
            "transformers": [
                Rotate_Translate(p_range=2, y_factor=3),
                Spline_Y_Perturbations(perturbation_intensity=0.005, spline_points=10),
                Spline_X_Simplification(spline_points=50, uniform=True),
                GaussianAdditiveNoise(sigma=0.01),
                MultiplicativeNoise(sigma_gain=0.05),
                LinearBaselineDrift(),
                PolynomialBaselineDrift(),
                WavelengthShift(),
                WavelengthStretch(),
                LocalWavelengthWarp(),
                SmoothMagnitudeWarp(),
                GaussianSmoothingJitter(),
                UnsharpSpectralMask(),
                ChannelDropout(),
                MixupAugmenter(),
                ScatterSimulationMSC(),
            ],
            "ref_percentage": 4.0,
            "selection": "random",
            "random_state": 42
        }
    },
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

    {"split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"},

    # "fold_chart",
    # 'chart_2d',  # 2D Visualization of augmented features

    MinMaxScaler(),
    # StandardScaler(),
    stacking_regressor,
]

# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline_config = PipelineConfigs(pipeline, name="BatchRegression")
dataset_config = DatasetConfigs(data_paths)

runner = PipelineRunner(save_artifacts=True, verbose=1, plots_visible=args.plots)
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
    print("\nTop 5 by R2 (test):")
    for idx, model in enumerate(top_r2):
        print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['rmse', 'r2'], partition=['val', 'test'])}")

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

fig_heatmap_model_dataset_r2 = analyzer.plot_heatmap(
    x_var="model_name",
    y_var="dataset_name",
    rank_metric='rmse',
    display_metric='r2',
)

# Candlestick plot by model
fig_candlestick = analyzer.plot_candlestick(
    variable="model_name",
    display_metric='rmse',
    partition='test'
)

# Histogram of RMSE distribution
fig_histogram = analyzer.plot_histogram(
    display_metric='rmse',
    partition='test'
)

if args.show:
    plt.show()

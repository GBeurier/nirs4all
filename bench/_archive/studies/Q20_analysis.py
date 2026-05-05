"""
Batch Classification Analysis - Comprehensive Pipeline for Multiple Datasets
=============================================================================
Runs complex classification analysis on all datasets in the classif/ folder.
Includes ML models, DL models, and various preprocessing combinations.
Generates confusion matrices, heatmaps, candlestick plots, and histograms.
"""

# Standard library imports
import argparse
import os

os.environ['DISABLE_EMOJIS'] = '0'

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Third-party imports - ML Models
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

from nirs4all.operators.models.sklearn import IKPLS, MBPLS, OPLS, OPLSDA, PLSDA, SIMPLS, DiPLS, SparsePLS
from nirs4all.operators.transforms import Rotate_Translate

# Optional boosting libraries
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# NIRS4All imports
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.operators.models.pytorch.nicon import nicon_classification
from nirs4all.operators.models.pytorch.spectral_transformer import spectral_transformer_classification
from nirs4all.operators.transforms import (
    CARS,
    MCUVE,
    BandMasking,
    BandPerturbation,
    ChannelDropout,
    Derivate,
    Detrend,
    Gaussian,
    GaussianAdditiveNoise,
    GaussianSmoothingJitter,
    Haar,
    LinearBaselineDrift,
    LocalClipping,
    LocalMixupAugmenter,
    LocalWavelengthWarp,
    MixupAugmenter,
    MultiplicativeNoise,
    PolynomialBaselineDrift,
    Random_X_Operation,
    ScatterSimulationMSC,
    SmoothMagnitudeWarp,
    SpikeNoise,
    Spline_Curve_Simplification,
    Spline_X_Perturbations,
    Spline_X_Simplification,
    Spline_Y_Perturbations,
    UnsharpSpectralMask,
    WavelengthShift,
    WavelengthStretch,
    Wavelet,
)
from nirs4all.operators.transforms import FirstDerivative as FstDer
from nirs4all.operators.transforms import LocalStandardNormalVariate as LSNV
from nirs4all.operators.transforms import MultiplicativeScatterCorrection as MSC
from nirs4all.operators.transforms import RobustStandardNormalVariate as RSNV
from nirs4all.operators.transforms import SavitzkyGolay as SavGol
from nirs4all.operators.transforms import SecondDerivative as SndDer
from nirs4all.operators.transforms import StandardNormalVariate as SNV
from nirs4all.operators.transforms.nirs import AreaNormalization
from nirs4all.operators.transforms.nirs import ExtendedMultiplicativeScatterCorrection as EMSC
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.visualization.predictions import PredictionAnalyzer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Batch Classification Analysis')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at the end')
args = parser.parse_args()

print("=" * 80)
print("BATCH CLASSIFICATION ANALYSIS")
print("=" * 80)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# All classification datasets
data_paths = [
    # 'selection/nitro_classif_unmerged/Digestibility_custom2',
    # 'selection/nitro_classif_unmerged/Digestibility_custom3',
    # 'selection/nitro_classif_unmerged/Digestibility_custom5',
    'selection/nitro_regression_unmerged/Digestibility_0.8',
    # 'selection/nitro_classif_unmerged/Hardness_custom2',
    # 'selection/nitro_classif_unmerged/Hardness_custom4',
    # 'selection/nitro_classif_unmerged/Tannin_custom2',
    # 'selection/nitro_classif_unmerged/Tannin_custom3',
]

# ============================================================================
# STACKING CONFIGURATION
# ============================================================================
# Define base estimators for classification stacking
base_estimators = [
    ('plsda_5', PLSDA(n_components=5)),
    ('plsda_7', PLSDA(n_components=7)),
    ('plsda_14', PLSDA(n_components=14)),
    ('plsda_15', PLSDA(n_components=15)),
    ('plsda_16', PLSDA(n_components=16)),
    ('oplsda_1_5', OPLSDA(n_components=1, pls_components=5)),
    ('oplsda_2_5', OPLSDA(n_components=2, pls_components=5)),
    ('oplsda_1_6', OPLSDA(n_components=1, pls_components=6)),
    ('oplsda_2_6', OPLSDA(n_components=2, pls_components=6)),
    ('oplsda_1_7', OPLSDA(n_components=1, pls_components=7)),
    ('oplsda_2_7', OPLSDA(n_components=2, pls_components=7)),
    ('oplsda_1_12', OPLSDA(n_components=1, pls_components=12)),
    ('oplsda_2_12', OPLSDA(n_components=2, pls_components=12)),
    ('oplsda_1_13', OPLSDA(n_components=1, pls_components=13)),
    ('oplsda_2_13', OPLSDA(n_components=2, pls_components=13)),
    ('oplsda_1_14', OPLSDA(n_components=1, pls_components=14)),
    ('oplsda_2_14', OPLSDA(n_components=2, pls_components=14)),
    ('oplsda_1_15', OPLSDA(n_components=1, pls_components=15)),
    ('oplsda_2_15', OPLSDA(n_components=2, pls_components=15)),
    ('oplsda_1_16', OPLSDA(n_components=1, pls_components=16)),
    ('oplsda_2_16', OPLSDA(n_components=2, pls_components=16)),
    # ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
    ('catboost', CatBoostClassifier(iterations=200, depth=8, learning_rate=0.1, random_state=42, verbose=0, allow_writing_files=False)),
    # ('xgboost', XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')),
    # ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    # ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
    # ('lgbm', LGBMClassifier(n_estimators=250, max_depth=8, learning_rate=0.1, random_state=42)),
    # ('knn', KNeighborsClassifier(n_neighbors=5)),
    ("mlp_32_8_64", MLPClassifier(hidden_layer_sizes=(32, 8, 64), max_iter=500, random_state=42)),
    ("mlp_32_128_64", MLPClassifier(hidden_layer_sizes=(32, 128, 64), max_iter=500, random_state=42)),
    ("mlp_128_32_16_64", MLPClassifier(hidden_layer_sizes=(128, 32, 16, 64), max_iter=500, random_state=42)),
    ("extratrees", ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42))
]

# Create Stacking Classifier with Logistic Regression as meta-learner
stacking_classifier = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=2,
    passthrough=False,
    n_jobs=-1
)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================
pipeline = [
    # Cross-validation setup (stratified for classification)
    # "fold_chart",
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
    #         "balance": "y",
    #         "ref_percentage": 4.0,
    #         "selection": "random",
    #         "random_state": 42
    #     }
    # },
    # "fold_chart",
    # "augment_details_chart",
    # Comprehensive feature augmentation with many preprocessing combinations
    {"feature_augmentation": [
        [MSC(scale=False), EMSC, AreaNormalization],
        [MSC(scale=False), EMSC, SNV],
        [EMSC, Gaussian(order=1, sigma=2), RSNV],
        EMSC,
        SNV,
        Haar,
        [EMSC, FstDer],
        [SNV, SndDer],
    ]},
    "chart_2d",
    "chart_3d",
    # CARS(
    #     n_components=12,            # PLS components for internal model
    #     n_sampling_runs=50,         # Number of Monte-Carlo runs
    #     n_variables_ratio_end=0.2,  # Final ratio of variables to keep
    #     cv_folds=3,                 # Cross-validation folds
    #     random_state=42             # For reproducibility
    # ),
    # "chart_2d",
    {"split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"},

    "fold_chart",
    # 'chart_2d',  # 2D Visualization of augmented features

    MinMaxScaler(),
    # StandardScaler(),
    # {"model": nicon_classification, "name": "nicon_classification"},
    # stacking_classifier,
    # {
    #     "model": OPLSDA(n_components=10, pls_components=10),
    #     "name": "OPLSDA",
    #     # "finetune_params": {
    #     #     "n_trials": 50,
    #     #     "verbose": 2,                           # 0=silent, 1=basic, 2=detailed
    #     #     "model_params": {
    #     #         'n_components': ('int', 1, 10),
    #     #         'pls_components': ('int', 1, 25),
    #     #     },
    #     # }
    # },

    # {"model": OPLSDA(n_components=1, pls_components=11), "name": "OPLSDA_1_11"},
    # {"model": OPLSDA(n_components=2, pls_components=11), "name": "OPLSDA_2_11"},
    # # {"model": OPLSDA(n_components=3, pls_components=11), "name": "OPLSDA_3_11"},
    # {"model": OPLSDA(n_components=1, pls_components=12), "name": "OPLSDA_1_12"},
    # {"model": OPLSDA(n_components=2, pls_components=12), "name": "OPLSDA_2_12"},
    # # {"model": OPLSDA(n_components=3, pls_components=12), "name": "OPLSDA_3_12"},
    # {"model": OPLSDA(n_components=1, pls_components=13), "name": "OPLSDA_1_13"},
    # {"model": OPLSDA(n_components=2, pls_components=13), "name": "OPLSDA_2_13"},
    # # {"model": OPLSDA(n_components=3, pls_components=13), "name": "OPLSDA_3_13"},
    # {"model": OPLSDA(n_components=1, pls_components=14), "name": "OPLSDA_1_14"},
    # {"model": OPLSDA(n_components=2, pls_components=14), "name": "OPLSDA_2_14"},
    # # {"model": OPLSDA(n_components=3, pls_components=14), "name": "OPLSDA_3_14"},
    # {"model": OPLSDA(n_components=1, pls_components=15), "name": "OPLSDA_1_15"},
    # {"model": OPLSDA(n_components=2, pls_components=15), "name": "OPLSDA_2_15"},
    # # {"model": OPLSDA(n_components=3, pls_components=15), "name": "OPLSDA_3_15"},
    # {"model": OPLSDA(n_components=1, pls_components=16), "name": "OPLSDA_1_16"},
    # {"model": OPLSDA(n_components=2, pls_components=16), "name": "OPLSDA_2_16"},
    # # {"model": OPLSDA(n_components=3, pls_components=16), "name": "OPLSDA_3_16"},
]

# for estimator_name, estimator in base_estimators:
#     pipeline.append(
#         {
#             "model": estimator,
#             "name": estimator_name
#         }
#     )

# ============================================================================
# RUN PIPELINE
# ============================================================================
pipeline_config = PipelineConfigs(pipeline, name="BatchClassification")
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
ranking_metric = 'balanced_accuracy'

# Display top models overall
top_models = predictions.top(best_model_count, ranking_metric)
print(f"\nTop {best_model_count} models overall by {ranking_metric}:")
for idx, prediction in enumerate(top_models):
    print(f"{idx+1}. {Predictions.pred_short_string(prediction, metrics=['accuracy', 'balanced_accuracy', 'f1'])} - {prediction['preprocessings']}")

# Per-dataset analysis
for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    dataset_predictions = dataset_prediction['run_predictions']

    # Top by Accuracy
    top_accuracy = dataset_predictions.top(n=5, rank_metric='accuracy', rank_partition='test')
    print("\nTop 5 by Accuracy (test):")
    for idx, model in enumerate(top_accuracy):
        print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['accuracy', 'balanced_accuracy'], partition=['val', 'test'])}")

    # Top by F1
    top_f1 = dataset_predictions.top(n=5, rank_metric='f1', rank_partition='test')
    print("\nTop 5 by F1 (test):")
    for idx, model in enumerate(top_f1):
        print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=['accuracy', 'f1'], partition=['val', 'test'])}")

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
            print("\n  Non-aggregated Top 3 by Accuracy (test):")
            top_regular = dataset_predictions.top(n=3, rank_metric='accuracy', rank_partition='test')
            for idx, model in enumerate(top_regular):
                print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=['accuracy', 'balanced_accuracy'])}")

            print("\n  Aggregated by 'ID' Top 3 by Accuracy (test):")
            top_aggregated = dataset_predictions.top(
                n=3,
                rank_metric='accuracy',
                rank_partition='test',
                aggregate='ID'  # Aggregate predictions by sample ID
            )
            for idx, model in enumerate(top_aggregated):
                n_samples = model.get('n_samples', '?')
                agg_flag = " [aggregated]" if model.get('aggregated') else ""
                print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=['accuracy', 'balanced_accuracy'])} (n={n_samples}){agg_flag}")

            # Show one detailed example with aggregation
            if top_aggregated:
                best_agg = top_aggregated[0]
                y_true = best_agg.get('y_true')
                y_pred = best_agg.get('y_pred')
                if y_true is not None and y_pred is not None:
                    print("\n  Best aggregated model details:")
                    print(f"    Model: {best_agg.get('model_name')}")
                    print(f"    Samples after aggregation: {len(y_pred)}")
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

# fig_top_b = analyzer.plot_top_k(
#     k=10,
#     rank_metric='balanced_accuracy',
#     rank_partition='val'
# )

# fig_top = analyzer.plot_top_k(
#     k=10,
#     rank_metric='accuracy',
#     rank_partition='val'
# )

# # Confusion matrices for top models
# fig_confusion_matrix = analyzer.plot_confusion_matrix(
#     k=10,
#     rank_metric='accuracy', rank_partition='val', display_partition='test'
# )

fig_confusion_matrix_val = analyzer.plot_confusion_matrix(
    k=10,
    rank_metric='balanced_accuracy', rank_partition='val', display_partition='test'
)

fig_confusion_matrix_val = analyzer.plot_confusion_matrix(
    k=10,
    rank_metric='balanced_accuracy', rank_partition='val', display_partition='test', aggregate='ID'
)

fig_confusion_matrix_val = analyzer.plot_confusion_matrix(
    k=10,
    rank_metric='accuracy', display_metric='accuracy', rank_partition='val', display_partition='test', aggregate='ID'
)

# # Heatmaps
# fig_heatmap_model_dataset = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="dataset_name",
#     rank_metric='balanced_accuracy',
#     display_metric='balanced_accuracy',
#     aggregate='ID'
# )

# ============================================================================
# AGGREGATED VISUALIZATIONS
# ============================================================================
# When the dataset has an 'ID' column, you can create visualizations with
# aggregated predictions (one prediction per sample ID instead of per spectrum).
#
# Example: Confusion matrix with aggregated predictions by sample ID
# fig_confusion_agg = analyzer.plot_confusion_matrix(
#     k=5,
#     rank_metric='balanced_accuracy',
#     rank_partition='val',
#     display_partition='test',
#     aggregate='ID'  # Aggregate by sample ID
# )
#
# Example: Top-K comparison with aggregated predictions
# fig_top_agg = analyzer.plot_top_k(
#     k=5,
#     rank_metric='balanced_accuracy',
#     aggregate='ID'
# )

# fig_heatmap_model_dataset_f1 = analyzer.plot_heatmap(
#     x_var="model_name",
#     y_var="dataset_name",
#     rank_metric='f1',
#     display_metric='f1',
# )

if args.show:
    plt.show()

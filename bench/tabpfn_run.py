"""
TabPFN Exploration Script for NIRS4All
======================================
Explore predictions with TabPFN on nitrosorgh datasets.
Supports switching between regression/classification, adding transforms,
and (de)activating augmentation.
"""

import argparse
import os
os.environ['DISABLE_EMOJIS'] = '0'

from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.decomposition import PCA
from huggingface_hub import login
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model_loading import get_cache_dir
from tabpfn_extensions.hpo import TunedTabPFNClassifier, TunedTabPFNRegressor
from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier, RandomForestTabPFNRegressor

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
)
# NIRS4All imports - Preprocessing transforms
from nirs4all.operators.transforms import (
    Detrend, FirstDerivative as FstDer, SecondDerivative as SndDer,
    Gaussian, StandardNormalVariate as SNV, SavitzkyGolay as SavGol,
    Haar, MultiplicativeScatterCorrection as MSC,
    RobustStandardNormalVariate as RSNV, LocalStandardNormalVariate as LSNV, Wavelet,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization, ExtendedMultiplicativeScatterCorrection as EMSC
)
from sklearn.preprocessing import KBinsDiscretizer
from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.visualization.predictions import PredictionAnalyzer
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold, BinnedStratifiedGroupKFold
from spectral_latent_features import SpectralLatentFeatures
from sklearn.preprocessing import StandardScaler, RobustScaler

# Hugging Face login for TabPFN - load token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. TabPFN may not work properly.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='TabPFN Exploration')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots at the end')
args = parser.parse_args()

print("=" * 80)
print("TABPFN EXPLORATION")
print("=" * 80)
# Available TabPFN v2.5 model variants:
# CLASSIFIER models:


def analysis(task_type, predictions, predictions_per_dataset):
    # ============================================================================
    # RESULTS ANALYSIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    best_model_count = 20
    ranking_metric = 'rmse' if task_type == 'regression' else 'balanced_accuracy'
    metrics = ['rmse', 'r2'] if task_type == 'regression' else ['accuracy', 'balanced_accuracy']
    # Per-dataset analysis
    for dataset_name, dataset_prediction in predictions_per_dataset.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        dataset_predictions = dataset_prediction['run_predictions']

        top_rmse = dataset_predictions.top(n=5, rank_metric=ranking_metric, rank_partition='test')
        print(f"\nTop 5 by {ranking_metric} (test):")
        for idx, model in enumerate(top_rmse):
            print(f"  {idx+1}. {Predictions.pred_short_string(model, metrics=metrics, partition=['val', 'test'])}")

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
                print(f"\n  Non-aggregated Top 3 by {ranking_metric} (test):")
                top_regular = dataset_predictions.top(n=3, rank_metric=ranking_metric, rank_partition='test')
                for idx, model in enumerate(top_regular):
                    print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=metrics)}")

                print(f"\n  Aggregated by 'ID' Top 3 by {ranking_metric} (test):")
                top_aggregated = dataset_predictions.top(
                    n=3,
                    rank_metric=ranking_metric,
                    rank_partition='test',
                    aggregate='ID'  # Aggregate predictions by sample ID
                )
                for idx, model in enumerate(top_aggregated):
                    n_samples = model.get('n_samples', '?')
                    agg_flag = " [aggregated]" if model.get('aggregated') else ""
                    print(f"    {idx+1}. {Predictions.pred_short_string(model, metrics=metrics)} (n={n_samples}){agg_flag}")

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

    # # ============================================================================
    # # VISUALIZATIONS
    # # ============================================================================
    # print("\n" + "=" * 80)
    # print("GENERATING VISUALIZATIONS")
    # print("=" * 80)

    # analyzer = PredictionAnalyzer(predictions)

    # # Heatmap: rank by test, display test
    # fig_heatmap_test_test = analyzer.plot_heatmap(
    #     x_var="model_name",
    #     y_var="dataset_name",
    #     rank_metric=ranking_metric,
    #     rank_partition='test',
    #     display_metric=ranking_metric,
    #     display_partition='test',
    # )

    # # Heatmap: rank by val, display test
    # fig_heatmap_val_test = analyzer.plot_heatmap(
    #     x_var="model_name",
    #     y_var="dataset_name",
    #     rank_metric=ranking_metric,
    #     rank_partition='val',
    #     display_metric=ranking_metric,
    #     display_partition='test',
    # )

    if args.show:
        plt.show()



CLASSIFIER_MODELS = {
    'default': 'tabpfn-v2.5-classifier-v2.5_default.ckpt',
    'default-2': 'tabpfn-v2.5-classifier-v2.5_default-2.ckpt',
    'large-features-L': 'tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt',
    'large-features-XL': 'tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt',
    'large-samples': 'tabpfn-v2.5-classifier-v2.5_large-samples.ckpt',
    'real': 'tabpfn-v2.5-classifier-v2.5_real.ckpt',
    'real-large-features': 'tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt',
    'real-large-samples-and-features': 'tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt',
    'variant': 'tabpfn-v2.5-classifier-v2.5_variant.ckpt',
}

# REGRESSOR models:
REGRESSOR_MODELS = {
    'default': 'tabpfn-v2.5-regressor-v2.5_default.ckpt',
    'low-skew': 'tabpfn-v2.5-regressor-v2.5_low-skew.ckpt',
    'quantiles': 'tabpfn-v2.5-regressor-v2.5_quantiles.ckpt',
    'real': 'tabpfn-v2.5-regressor-v2.5_real.ckpt',
    'real-variant': 'tabpfn-v2.5-regressor-v2.5_real-variant.ckpt',
    'small-samples': 'tabpfn-v2.5-regressor-v2.5_small-samples.ckpt',
    'variant': 'tabpfn-v2.5-regressor-v2.5_variant.ckpt',
}


def train_tabpfn(task_type, model_variants=None, aug=0.0, cv=False, transf=None, mode="normal"):  # rf, finetune
    """
    Train TabPFN with optional model variant selection.

    Args:
        task_type: 'regression' or 'classification'
        model_variants: List of model variant names (keys from CLASSIFIER_MODELS or REGRESSOR_MODELS)
                       e.g., ['default', 'real', 'large-features-XL'] for classification
                       e.g., ['default', 'real', 'low-skew'] for regression
                       If None, uses default model only.
        aug: Augmentation percentage (0 = no augmentation)
        cv: Whether to use cross-validation
        transf: Preprocessing transforms
        mode: 'normal', 'finetune', 'rf', or 'tune'
    """

    if task_type == 'regression':
        data_paths = [
            'sample_data/nitro_reg_merged/Digestibility_0.8',
            'sample_data/nitro_reg_merged/Hardness_0.8',
            'sample_data/nitro_reg_merged/Tannin_0.8'
        ]
    else:
        data_paths = [
            'sample_data/nitro_classif_merged/Digestibility_custom2',
            'sample_data/nitro_classif_merged/Digestibility_custom3',
            'sample_data/nitro_classif_merged/Digestibility_custom5',
            'sample_data/nitro_classif_merged/Hardness_custom2',
            'sample_data/nitro_classif_merged/Hardness_custom4',
            'sample_data/nitro_classif_merged/Tannin_custom2',
            'sample_data/nitro_classif_merged/Tannin_custom3',
        ]

    pipeline = [
        {
            "split": SPXYGFold,
            "split_params": {
                "n_splits": 1,
                "test_size": 0.2,
                "aggregation": "mean"
            },
            "group": "ID"
        }
    ]

    if aug > 0.0:
        sample_augmentation = {
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
                "ref_percentage": aug,
                "selection": "random",
                "random_state": 42
            }
        }

        pipeline.append(sample_augmentation)

    if transf is not None:
        pipeline.append(transf)

    if cv:
        pipeline.append({
            "split": GroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"
        })

        # if 'classification' == task_type:
        #     pipeline.append({
        #         "split": StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42), "group": "ID"
        #         })
        # else:
        #     pipeline.append({
        #     "split": BinnedStratifiedGroupKFold,
        #         "split_params": {
        #             "n_splits": 3,
        #             "n_bins": 7,
        #             "shuffle": True,
        #             "random_state": 1337
        #         },
        #         "group": "ID"
        #     })

    if mode == "finetune":
        if task_type == 'classification':
            TabPFN_class = TunedTabPFNClassifier(device='cuda', n_trials=100, metric="f1")
        else:
            TabPFN_class = TunedTabPFNRegressor(device='cuda', n_trials=100, metric="rmse")

    elif mode == "rf":
        if task_type == 'classification':
            TabPFN_class = RandomForestTabPFNClassifier(
                tabpfn=TabPFNClassifier(device='cuda', eval_metric="f1", n_estimators=12),
                n_estimators=100,
                max_depth=12,
            )
        else:
            TabPFN_class = RandomForestTabPFNRegressor(
                tabpfn=TabPFNRegressor(device='cuda', eval_metric="rmse", n_estimators=8),
                n_estimators=20,
                max_depth=6,
                device='cuda',
                # eval_metric="rmse"
            )
    elif mode == "tune":
        if task_type == 'classification':
            TabPFN_class = TabPFNClassifier(
                # n_estimators = 8,
                eval_metric="balanced_accuracy",
                tuning_config={"tune_decision_thresholds": True},
                device='cuda',
                # balance_probabilities = True,
                # average_before_softmax = True,
            )
        else:
            TabPFN_class = TabPFNRegressor(
                # n_estimators = 8,
                tuning_config={"tune_decision_thresholds": True},
                device='cuda',
                # average_before_softmax = True,
            )
    else:
        TabPFN_class = TabPFNClassifier(device='cuda') if task_type == 'classification' else TabPFNRegressor(device='cuda')

    # Handle model variants
    model_dict = CLASSIFIER_MODELS if task_type == 'classification' else REGRESSOR_MODELS
    TabPFN_base_class = TabPFNClassifier if task_type == 'classification' else TabPFNRegressor

    if model_variants is None or len(model_variants) == 0:
        # Use default model
        pipeline.append({"model": TabPFN_class, "name": "TabPFN"})
    else:
        for variant in model_variants:
            if variant == 'default' or variant == '':
                # Default model - no model_path needed
                if mode == "normal":
                    model = TabPFN_base_class(device='cuda')
                else:
                    model = TabPFN_class
                pipeline.append({"model": model, "name": f"TabPFN_{variant}"})
            elif variant in model_dict:
                # Specific model variant - construct full path to cache
                model_path = str(get_cache_dir() / model_dict[variant])
                model = TabPFN_base_class(model_path=model_path, device='cuda')
                pipeline.append({"model": model, "name": f"TabPFN_{variant}"})
            else:
                raise ValueError(f"Unknown model variant: {variant}. Available: {list(model_dict.keys())}")

    pipeline_config = PipelineConfigs(pipeline, name="TabPFN")
    dataset_config = DatasetConfigs(data_paths)

    runner = PipelineRunner(save_files=True, verbose=1, plots_visible=args.plots)
    return runner.run(pipeline_config, dataset_config)


classif_latent = SpectralLatentFeatures(
    use_pca=True,
    n_pca=120,
    pca_whiten=True,
    pca_variance_threshold=0.999,
    #############
    use_wavelets=False,
    wavelet_coeffs_per_level=16,
    use_wavelet_pca=True,
    wavelet_pca_components_per_level=8,
    #####
    wavelet='coif3', #haar, db4, sym4, sym5, coif3
    wavelet_levels=8,
    #############
    output_normalization='quantile',
    random_state=None,
)

regression_latent = SpectralLatentFeatures(
    use_pca=True,
    n_pca=120,
    pca_whiten=True,
    pca_variance_threshold=0.999,
    #############
    use_wavelets=True,
    wavelet_coeffs_per_level=16,
    use_wavelet_pca=True,
    wavelet_pca_components_per_level=8,
    #####
    wavelet='coif3', #haar, db4, sym4, sym5, coif3
    wavelet_levels=8,
    #############
    output_normalization='quantile',
    random_state=None,
)

  # Model variants to test - uncomment to compare different models:
    # For classification: 'default', 'real', 'large-features-L', 'large-features-XL',
    #                     'large-samples', 'real-large-features', 'real-large-samples-and-features', 'variant'
    # For regression: 'default', 'real', 'low-skew', 'quantiles', 'real-variant', 'small-samples', 'variant'

task_type = 'regression'  # 'regression' or 'classification'
predictions, predictions_per_dataset = train_tabpfn(
    task_type,
    cv=True,
    # aug=1.0,
    mode='finetune',
    model_variants=['real', 'default', 'low-skew', 'quantiles', 'real-variant'],
    transf= {"_or_": [None, classif_latent, regression_latent]}
)

task_type = 'classification'  # 'regression' or 'classification'
predictions, predictions_per_dataset = train_tabpfn(
    task_type,
    cv=True,
    # aug=1.0,
    mode='finetune',
    model_variants=['real', 'default', 'large-features-XL', 'real-large-features'],
    transf= {"_or_": [None, classif_latent, regression_latent]}
)
analysis(task_type, predictions, predictions_per_dataset)


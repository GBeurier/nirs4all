"""
Q12: Sample Augmentation Comprehensive Examples
===============================================
Demonstrates all modes of sample augmentation for classification and regression.

Scenarios covered:
1. Classification - Standard (Unbalanced)
2. Classification - Balanced (Fixed Target Size)
3. Classification - Balanced (Max Factor)
4. Classification - Balanced (Ref Percentage)
5. Regression - Balanced (Binning: Equal Width)
6. Regression - Balanced (Binning: Quantile)
7. Augmentation Charts - Visual comparison of augmentation effects
8. Multiple Augmenters - Comparison of different augmentation operators
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import (
    Rotate_Translate,
    Random_X_Operation,
    Spline_Y_Perturbations,
    Spline_X_Perturbations,
    Spline_X_Simplification,
    Spline_Curve_Simplification,
    GaussianAdditiveNoise,
    MultiplicativeNoise,
    LinearBaselineDrift,
    PolynomialBaselineDrift,
    WavelengthShift,
    WavelengthStretch,
    LocalWavelengthWarp,
    SmoothMagnitudeWarp,
    BandPerturbation,
    GaussianSmoothingJitter,
    UnsharpSpectralMask,
    BandMasking,
    ChannelDropout,
    SpikeNoise,
    LocalClipping,
    MixupAugmenter,
    LocalMixupAugmenter,
    ScatterSimulationMSC,
)
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.model_selection import GroupKFold

# Standard library imports
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q12 Sample Augmentation Example')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()
display_pipeline_plots = args.plots
display_analyzer_plots = args.show


def run_scenario(name, dataset_path, pipeline_steps, description):
    print(f"\n\n{'='*80}")
    print(f"SCENARIO: {name}")
    print(f"DESCRIPTION: {description}")
    print(f"{'='*80}\n")

    # Create config objects
    pipeline_config = PipelineConfigs(pipeline_steps, name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
    dataset_config = DatasetConfigs(dataset_path)

    # Run pipeline
    runner = PipelineRunner(save_files=False, verbose=0, plots_visible=display_pipeline_plots)
    runner.run(pipeline_config, dataset_config)

# Common split configuration
split_step = {"split": GroupKFold(n_splits=2), "group": "Sample_ID"}

# --- CLASSIFICATION SCENARIOS ---
# Using a classification dataset
classif_data = 'sample_data/classification'

# # 1. Classification - Standard (Unbalanced)
# run_scenario(
#     "Classification - Standard",
#     classif_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
#                 "count": 2,
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Standard augmentation: Adds 2 augmented samples for every original sample regardless of class."
# )

# # 2. Classification - Balanced (Fixed Target Size)
# run_scenario(
#     "Classification - Balanced (Target Size)",
#     classif_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate],
#                 "balance": "y",
#                 "target_size": 50,  # Target 50 samples per class
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Balanced augmentation: Each class augmented to reach exactly 50 samples."
# )

# # 3. Classification - Balanced (Max Factor)
# run_scenario(
#     "Classification - Balanced (Max Factor)",
#     classif_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate],
#                 "balance": "y",
#                 "max_factor": 2.0,  # Max 2x augmentation
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Balanced augmentation: Classes augmented up to majority size, but capped at 2x original size."
# )

# # 4. Classification - Balanced (Ref Percentage)
# run_scenario(
#     "Classification - Balanced (Ref Percentage)",
#     classif_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate],
#                 "balance": "y",
#                 "ref_percentage": 0.8,  # Target 100% of majority class
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Balanced augmentation: Classes augmented to match the size of the majority class (100%)."
# )

# --- AUGMENTATION VISUALIZATION SCENARIOS ---

# # 7. Augmentation Chart - Overlay visualization (Original vs Augmented)
# run_scenario(
#     "Augmentation Chart - Overlay",
#     classif_data,
#     [
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
#                 "count": 2,
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "augment_chart",  # Shows original (blue) vs augmented (orange) overlaid
#         split_step
#     ],
#     "Visualization: Overlay chart showing original samples in blue and augmented samples in orange."
# )

# # 8. Multiple Augmenters - Compare different augmentation operators
# run_scenario(
#     "Multiple Augmenters Comparison",
#     classif_data,
#     [
#         {
#             "sample_augmentation": {
#                 "transformers": [
#                     Rotate_Translate(p_range=2, y_factor=3),
#                     Spline_Y_Perturbations(perturbation_intensity=0.005, spline_points=10),
#                     Random_X_Operation(operator_range=(0.995, 1.005)),
#                 ],
#                 "count": 2,
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "augment_details_chart",  # Shows each transformer's effect separately
#         split_step
#     ],
#     "Visualization: Details chart showing each augmentation type separately (Original + each transformer)."
# )

# # 9. Spline-based Augmenters Demo
# run_scenario(
#     "Spline Augmenters Demo",
#     classif_data,
#     [
#         {
#             "sample_augmentation": {
#                 "transformers": [
#                     Spline_Y_Perturbations(perturbation_intensity=0.005, spline_points=10),
#                     Spline_X_Perturbations(perturbation_density=0.05, perturbation_range=(-5, 5)),
#                     Spline_X_Simplification(spline_points=50, uniform=True),
#                 ],
#                 "count": 1,
#                 "selection": "all",  # Apply all transformers to each sample
#                 "random_state": 42
#             }
#         },
#         "augment_details_chart",
#         split_step
#     ],
#     "Spline-based augmenters: Y-perturbations, X-perturbations, and X-simplification effects."
# )

# 10. All New Augmenters Demo
run_scenario(
    "All New Augmenters Demo",
    classif_data,
    [
        {
            "sample_augmentation": {
                "transformers": [
                    Rotate_Translate(p_range=2, y_factor=3),
                    Spline_Y_Perturbations(perturbation_intensity=0.005, spline_points=10),
                    # Spline_X_Perturbations(perturbation_density=0.05, perturbation_range=(-5, 5)),
                    Spline_X_Simplification(spline_points=50, uniform=True),
                    GaussianAdditiveNoise(sigma=0.01),
                    MultiplicativeNoise(sigma_gain=0.05),
                    LinearBaselineDrift(),
                    PolynomialBaselineDrift(),
                    WavelengthShift(),
                    WavelengthStretch(),
                    LocalWavelengthWarp(),
                    SmoothMagnitudeWarp(),
                    # BandPerturbation(),
                    GaussianSmoothingJitter(),
                    UnsharpSpectralMask(),
                    # BandMasking(),
                    ChannelDropout(),
                    # SpikeNoise(),
                    # LocalClipping(),
                    MixupAugmenter(),
                    # LocalMixupAugmenter(),
                    ScatterSimulationMSC(),
                ],
                "count": 4,
                "selection": "random",  # Apply all transformers to each sample
                "random_state": 42
            }
        },
        "augment_chart",
        "augment_details_chart",
        split_step
    ],
    "Demonstration of all new spectral augmentations."
)


# # --- REGRESSION SCENARIOS ---
# # Using a regression dataset
# regression_data = 'sample_data/regression_2'

# # 5. Regression - Balanced (Binning: Equal Width)
# run_scenario(
#     "Regression - Balanced (Equal Width Bins)",
#     regression_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate],
#                 "balance": "y",
#                 "bins": 5,
#                 "binning_strategy": "equal_width",
#                 "ref_percentage": 1.0,
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Regression balancing: Targets binned into 5 equal-width bins, then balanced to majority bin size."
# )

# # 6. Regression - Balanced (Binning: Quantile)
# run_scenario(
#     "Regression - Balanced (Quantile Bins)",
#     regression_data,
#     [
#         "fold_chart",
#         {
#             "sample_augmentation": {
#                 "transformers": [Rotate_Translate],
#                 "balance": "y",
#                 "bins": 5,
#                 "binning_strategy": "quantile",
#                 "ref_percentage": 0.8,
#                 "selection": "random",
#                 "random_state": 42
#             }
#         },
#         "fold_chart",
#         split_step
#     ],
#     "Regression balancing: Targets binned into 5 quantile bins (equal population), then balanced."
# )

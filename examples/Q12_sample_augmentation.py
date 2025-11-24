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
"""

import os
os.environ['DISABLE_EMOJIS'] = '1'

from nirs4all.data import DatasetConfigs
from nirs4all.operators.transforms import Rotate_Translate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.model_selection import GroupKFold

# Standard library imports
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Q1 Classification Example')
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

# 1. Classification - Standard (Unbalanced)
run_scenario(
    "Classification - Standard",
    classif_data,
    [
        "fold_chart",
        {
            "sample_augmentation": {
                "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
                "count": 2,
                "selection": "random",
                "random_state": 42
            }
        },
        "fold_chart",
        split_step
    ],
    "Standard augmentation: Adds 2 augmented samples for every original sample regardless of class."
)

# 2. Classification - Balanced (Fixed Target Size)
run_scenario(
    "Classification - Balanced (Target Size)",
    classif_data,
    [
        "fold_chart",
        {
            "sample_augmentation": {
                "transformers": [Rotate_Translate],
                "balance": "y",
                "target_size": 50,  # Target 50 samples per class
                "selection": "random",
                "random_state": 42
            }
        },
        "fold_chart",
        split_step
    ],
    "Balanced augmentation: Each class augmented to reach exactly 50 samples."
)

# 3. Classification - Balanced (Max Factor)
run_scenario(
    "Classification - Balanced (Max Factor)",
    classif_data,
    [
        "fold_chart",
        {
            "sample_augmentation": {
                "transformers": [Rotate_Translate],
                "balance": "y",
                "max_factor": 2.0,  # Max 2x augmentation
                "selection": "random",
                "random_state": 42
            }
        },
        "fold_chart",
        split_step
    ],
    "Balanced augmentation: Classes augmented up to majority size, but capped at 2x original size."
)

# 4. Classification - Balanced (Ref Percentage)
run_scenario(
    "Classification - Balanced (Ref Percentage)",
    classif_data,
    [
        "fold_chart",
        {
            "sample_augmentation": {
                "transformers": [Rotate_Translate],
                "balance": "y",
                "ref_percentage": 0.8,  # Target 100% of majority class
                "selection": "random",
                "random_state": 42
            }
        },
        "fold_chart",
        split_step
    ],
    "Balanced augmentation: Classes augmented to match the size of the majority class (100%)."
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

"""
Q12: Sample Augmentation with Balanced Classification
======================================================
Demonstrates balanced sample augmentation for imbalanced classification datasets.

Key Features:
  - Standard augmentation: Fixed number of augmentations per sample
  - Balanced augmentation: Class-aware distribution ensuring fair representation
  - Binning for regression: Continuous targets binned into virtual classes
  - Value-aware balancing: Fair distribution across unique y-values within bins
  - Random remainder: Fair selection when augmentations don't divide evenly
  - Leak prevention: Cross-validation never sees augmented samples in validation fold

Pipeline Steps:
  1. Standard augmentation (count=2)
  2. Balanced augmentation with binning (bins=5, bin_balancing="value")
  3. Cross-validation with GroupKFold (prevents leakage)
  4. Model training on augmented data
"""

# Standard library imports
import os
os.environ['DISABLE_EMOJIS'] = '1'

# NIRS4All imports
from nirs4all.dataset import DatasetConfigs
from nirs4all.operators.transformations import Rotate_Translate
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from sklearn.model_selection import GroupKFold

# Configuration
data_path = 'sample_data/regression_2'
focus = "y"  # Balance on target variable

# Pipeline with augmentation examples
pipeline = [
    "fold_chart",

    # EXAMPLE 1: Standard augmentation (count-based)
    # Creates 2 augmented samples per base sample
    # Useful when dataset is already balanced
    # {
    #     "sample_augmentation": {
    #         "transformers": [Rotate_Translate(p_range=2, y_factor=3)],
    #         "count": 2,
    #         "selection": "random",
    #         "random_state": 42
    #     }
    # },

    # EXAMPLE 2: Balanced augmentation with binning and value-aware mode
    # For regression data with class imbalance:
    # - balance="y": activates class-aware augmentation
    # - bins=5: creates 5 virtual classes from continuous y-values
    # - bin_balancing="value": ensures fair distribution across unique y-values
    #   (prevents one y-value from being over-represented when multiple samples
    #    share same bin and y-value)
    # - target_size=20: each class/bin augmented to 20 samples
    # Random remainder selection ensures fair distribution when counts don't divide evenly
    # (see BalancingCalculator.calculate_balanced_counts() for implementation)
    {
        "sample_augmentation": {
            "transformers": [Rotate_Translate],
            "balance": focus,
            "target_size": 20,
            "bins": 5,
            "binning_strategy": "equal_width",  # or "quantile" for skewed distributions
            "bin_balancing": "value",  # "sample" (default) or "value" for value-aware
            "selection": "random",
            "random_state": 42,
        }
    },

    "fold_chart",

    # Cross-validation split
    # Uses include_augmented=False internally to prevent data leakage:
    # - CV folds are created from base samples only
    # - Training can access augmented versions of training fold samples
    # - Validation fold never sees augmented samples (no leakage!)
    {"split": GroupKFold(n_splits=2), "group": "Sample_ID"},

    "fold_chart",
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "q12")
dataset_config = DatasetConfigs(data_path)

# Run the pipeline
runner = PipelineRunner(save_files=False, verbose=1, plots_visible=False)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

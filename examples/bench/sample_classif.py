"""
Sample configuration for NIRS4ALL pipeline with classification models and advanced model training and finetuning.

This file demonstrates the new nested configuration structure for classification model training
and finetuning with proper verbose level control compatible with TensorFlow and other ML frameworks.

Verbose Levels (compatible with TensorFlow):
- verbose=0: Silent mode (no output during training/finetuning)
- verbose=1: Basic mode (essential info like optimization progress and best parameters)
- verbose=2: Detailed mode (full training logs, parameter details, compilation info)
- verbose=3: Debug mode (maximum verbosity for debugging)

For TensorFlow specifically:
- verbose=0: Silent training (no callback messages like LearningRateScheduler, early stopping, etc.)
- verbose=1: Progress bar (shows early stopping messages)
- verbose=2: One line per epoch (shows all callback messages including LearningRateScheduler)

Configuration Structure:
- train_params: Final training parameters used after finetuning finds best model
- finetune_params: Configuration for hyperparameter optimization
  - model_params: Parameters to optimize (nested under finetune_params)
  - train_params: Training parameters used during optimization trials
  - verbose: Controls finetuning output verbosity
  - n_trials: Number of optimization trials
  - approach: 'grid', 'random', or 'auto'
"""

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nirs4all.operators.models.cirad_tf import decon_classification, nicon_classification
from nirs4all.operators.models.cirad_torch import transformer_classification
from sklearn.cross_decomposition import PLSRegression  # Can be adapted for classification

from nirs4all.operators.transformations import (
    Gaussian as GS,
    Rotate_Translate as RT,
    SavitzkyGolay as SG,
    StandardNormalVariate as SNV,
    Haar
)

dataset_config = {  # define the experiment type and dataset. An experiment is related to a source dataset. action will be removed in the future to allow classif and regression on the same dataset.
    # "type": "classification",  # 'auto', 'classification'
    "folder": "./../../sample_data/classification"  # dataset definition is dicted by the json schema. Can load single or multiple files with metadata, and many indices predefined if needed, and folds also.
}

pipeline_config = {
    "pipeline": [
        # "chart_3d",
        # "chart_2d",
        MinMaxScaler(feature_range=(0.1, 0.8)),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
        {"feature_augmentation": [None, GS, [SNV, Haar]]},  # augment the features by applying transformations, creating new row ids with new processing but same sample ids
        # "chart_3d",
        # Use stratified splits for classification to maintain class balance
        StratifiedShuffleSplit(n_splits=3, test_size=.25, random_state=42),  # First one is target:test by default
        # RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),  # Alternative: create folds for validation, using groups as stratifying variable.
        # "fold_chart",
        # "y_chart",
        # {"y_processing": LabelEncoder()},  # preprocess target data - encode string labels to numeric for classification
        # "y_chart",
        {
            "model": RandomForestClassifier(max_depth=10, random_state=42),
            "train_params": {
                # Final training parameters (after finetuning)
                "oob_score": True,
                "n_jobs": -1,
                "verbose": 0  # 0=silent, 1=basic, 2=detailed
            },
            "finetune_params": {
                "n_trials": 4,
                "approach": "grid",
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed finetuning output
                "model_params": {
                    # Parameters to optimize during finetuning
                    "n_estimators": [10, 30],  # Only 2 options instead of 3
                    "max_depth": [3, 7],       # Only 2 options instead of 3
                    "class_weight": [None, "balanced"]  # Important for imbalanced classification
                },
                "train_params": {
                    # Training parameters during finetuning trials (faster & silent)
                    "n_jobs": 1,
                    "verbose": 0
                }
            },
        },
        {
            "model": LinearDiscriminantAnalysis(),
            "train_params": {
                # Final training parameters (after finetuning)
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed
            },
            "finetune_params": {
                "n_trials": 4,
                "approach": "grid",
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed finetuning output
                "model_params": {
                    # Parameters to optimize during finetuning
                    'solver': ['svd', 'lsqr', 'eigen'],
                    'shrinkage': [None, 'auto']
                },
                "train_params": {
                    # Training parameters during finetuning trials (silent)
                    "verbose": 0
                }
            }
        },
        {
            "model": nicon_classification,
            "train_params": {
                # Final training parameters
                "epochs": 10,
                "patience": 10,
                "batch_size": 16,
                "cyclic_lr": True,
                "step_size": 20,
                "verbose": 0  # 0=silent, 1=progress bar, 2=one line per epoch
            },
        },
        {
            "model": nicon_classification,
            "train_params": {
                "epochs": 10,
                "patience": 10,
                "batch_size": 5000,
                "verbose": 0,
                # "best_model_memory": True
            },
            "finetune_params": {
                "n_trials": 2,
                "approach": "random",
                "model_params": {
                    "filters_1": [8, 16, 32],
                    "filters_2": [8, 16, 32],
                    "dropout_rate": ("float", 0.1, 0.5),
                    "num_classes": [2, 3]  # Binary or 3-class classification
                },
                "train_params": {
                    "epochs": 10,
                    "patience": 5,
                    "batch_size": 16,
                    "verbose": 0
                }
            },
        },
        {
            "model": SVC(kernel='linear', C=1.0, probability=True, random_state=42),  # probability=True for prediction probabilities
            "finetune_params": {  # As there are finetune parameters, optuna is used to optimize the model
                "model_params": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ['linear', 'rbf'],
                    "class_weight": [None, "balanced"]  # Important for classification
                }
            },
        },
        {
            "model": GradientBoostingClassifier(random_state=42),
            "finetune_params": {
                "n_trials": 3,
                "approach": "random",
                "model_params": {
                    "n_estimators": ("int", 50, 200),
                    "learning_rate": ("float", 0.01, 0.3),
                    "max_depth": [3, 5, 7]
                }
            },
        },
        {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "finetune_params": {
                "model_params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ['l1', 'l2', 'elasticnet'],
                    "solver": ['liblinear', 'saga'],
                    "class_weight": [None, "balanced"]
                }
            },
        },

# Example of more advanced pipeline configurations (commented out):
# RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),  # create folds for validation, using groups as stratifying variable.
# "spectra_charts",
# {"sample_augmentation": [RT, RT(p_range=5)]},  # augment the samples by applying transformations, creating new sample ids with new processing and origin_ids
# "spectra_charts",
# {"balance_augmentation":"groups"},
# "spectra_charts",
# MinMaxScaler(feature_range=(0,1)),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
# "spectra_charts",
# {"cluster": KMeans(n_clusters=5, random_state=42)},  # add groups indices to the dataset, which are the cluster ids. The dataset is now clustered.
# "uncluster",  # stop using centroids and use all the original samples. If the centroids are constructed (sample = None), they are discarded or hidden.
# {
#     "dispatch": [  # create as many branches in the pipeline as there are objects in the list. Data from train partition are copied to each branch.
#         [
#             MinMaxScaler(),
#             {"feature_augmentation": [None, SG, [SNV, GS]]},
#             {
#                 "model": RandomForestClassifier(random_state=42, max_depth=10),
#                 "y_pipeline": LabelEncoder,  # For classification, encode labels
#             },
#         ],
#         {
#             "model": decon_classification,  # TensorFlow CNN for classification
#             "y_pipeline": LabelEncoder(),
#         },
#         {
#             "model": SVC(kernel='linear', C=1.0, random_state=42, probability=True),
#             "y_pipeline": [LabelEncoder, StandardScaler()],  # Multiple preprocessing pipelines
#             "finetune_params": {
#                 "C": [0.1, 1.0, 10.0],
#                 "kernel": ['linear', 'rbf']
#             },
#         },
#         {
#             "stack": {  # create a stack of models for ensemble classification
#                 "model": RandomForestClassifier(random_state=42, max_depth=10),
#                 "y_pipeline": LabelEncoder(),
#                 "base_learners": [
#                     {
#                         "model": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
#                         "y_pipeline": LabelEncoder(),
#                     },
#                     {
#                         "model": DecisionTreeClassifier(random_state=42, max_depth=5),
#                         "y_pipeline": LabelEncoder(),
#                         "finetune_params": {
#                             "max_depth": [3, 5, 7],
#                             "class_weight": [None, "balanced"]
#                         }
#                     }
#                 ]
#             }
#         }
#     ]
# },
# "PlotConfusionMatrix"  # Classification-specific visualization

    ]
}
"""
Sample configuration for NIRS4ALL pipeline with advanced model training and finetuning.

This file demonstrates the new nested configuration structure for model training and finetuning
with proper verbose level control compatible with TensorFlow and other ML frameworks.

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from nirs4all.operators.models.cirad_tf import decon, nicon
from sklearn.cross_decomposition import PLSRegression

from nirs4all.operators.transformations import (
    Gaussian as GS,
    Rotate_Translate as RT,
    SavitzkyGolay as SG,
    StandardNormalVariate as SNV,
    Haar
)

dataset_config = "./../../sample_data/regression"

pipeline_config = {
    "pipeline": [
        # "chart_3d",
        # "chart_2d",
        MinMaxScaler(feature_range=(0.1, 0.8)),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
        {"feature_augmentation": [None, GS, [SNV, Haar]]},  # augment the features by applying transformations, creating new row ids with new processing but same sample ids
        # "chart_3d",
        # RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),  # create folds for validation, using groups as stratifying variable.
        ShuffleSplit(n_splits=3, test_size=.25),  # First one is target:test by default
        # "fold_chart",
        # "y_chart",
        {"y_processing": StandardScaler()},  # preprocess target data
        # "y_chart",
        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
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
                },
                "train_params": {
                    # Training parameters during finetuning trials (faster & silent)
                    "n_jobs": 1,
                    "verbose": 0
                }
            },
        },
        {
            "model": PLSRegression(),
            "train_params": {
                # Final training parameters (after finetuning)
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed
            },
            "finetune_params": {
                "n_trials": 4,
                "approach": "random",  # Better for continuous parameters
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed finetuning output
                "model_params": {
                    # Parameters to optimize during finetuning
                    'n_components': ('int', 5, 60),
                },
                "train_params": {
                    # Training parameters during finetuning trials (silent)
                    "verbose": 0
                }
            }
        },
        {
            "model": nicon,
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
            "model": nicon,
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
            "model": SVR(kernel='linear', C=1.0),  # another sklearn model, note that each training is followed by a prediction on the test partition and saved in the results indices
            "finetune_params": {  # As there are finetune parameters, optuna is used to optimize the model. more options can be added here to choose the strategy for the optimization, etc.
                "model_params": {
                    "C": [0.1, 1.0, 10.0]
                }
            },
        },


# RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),  # create folds for validation, using groups as stratifying variable.
# "spectra_charts",
# {"sample_augmentation": [RT, RT(p_range=5)]},  # augment the samples by applying transformations, creating new sample ids with new processing and origin_ids
# "spectra_charts",
# {"balance_augmentation":"groups"},
# "spectra_charts",
# MinMaxScaler(feature_range=(0,1)),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
# "spectra_charts",
# "spectra_charts",
# MinMaxScaler(feature_range=(0.2,0.8)),
# {"cluster": KMeans(n_clusters=5, random_state=42)},  # add groups indices to the dataset, which are the cluster ids. The dataset is now clustered.
# "uncluster",  # stop using centroids and use all the original samples. If the centroids are constructed (sample = None), they are discarded or hidden.
# {
#     "dispatch": [  # create as many branches in the pipeline as there are objects in the list. Data from train partition are copied to each branch. Can be used also to split the pipeline per source of data.
#         [
#             MinMaxScaler(),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
#             {"feature_augmentation": [None, SG, [SNV, GS]]},
#             {
#                 "model": RandomForestClassifier(random_state=42, max_depth=10),  # here's a sklearn model, dataset is automatically converted to 2d
#                 "y_pipeline": StandardScaler,  # preprocess target data
#             },
#         ],
#         {
#             "model": decon,  # here's a tf conv model (@framework decorator) . dataset is automatically converted to 3d
#             "y_pipeline": StandardScaler(),
#         },
#         {
#             "model": SVC(kernel='linear', C=1.0, random_state=42),  # another sklearn model, note that each training is followed by a prediction on the test partition and saved in the results indices
#             "y_pipeline": [MinMaxScaler, RobustScaler(with_centering=False)],  # preprocess target data with multiple pipelines, each pipeline is applied to the target data and the results are saved in the results indices
#             "finetune_params": {  # As there are finetune parameters, optuna is used to optimize the model. more options can be added here to choose the strategy for the optimization, etc.
#                 "C": [0.1, 1.0, 10.0]
#             },
#         },
#         {
#             "stack": {  # create a stack of models, each model is trained on the same data and the predictions are used as features for the next model.
#                 "model": RandomForestClassifier(random_state=42, max_depth=10),  # the main model of the stack, trained on the predictions of the base learners
#                 "y_pipeline": StandardScaler(),
#                 "base_learners": [  # the base learners of the stack, trained on the same data as the main model
#                     {
#                         "model": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),  # trained on the data and aggregates folds predictions for the main model
#                         "y_pipeline": MinMaxScaler(),
#                     },
#                     {
#                         "model": DecisionTreeClassifier(random_state=42, max_depth=5),
#                         "y_pipeline": MinMaxScaler(),
#                         "finetune_params": {
#                             "max_depth": [3, 5, 7]
#                         }
#                     }
#                 ]
#             }
#         }
#     ]
# },
# "PlotConfusionMatrix"  # a type of graph, showing the confusion matrix of the models


    ]
}



generator_config = {
    "pipeline": [
        # Preprocessing scalers - choose one or combinations
        {"_or_": [MinMaxScaler(feature_range=(0, 1)), RobustScaler()]},

        # Feature augmentation with combinations of transformations
        {
            "feature_augmentation": {
                "_or_": [
                    # Single transformations
                    None,
                    GS,
                    SNV,
                    SG,
                    Haar,
                    RT,

                    # Combinations of 2 transformations
                    [SNV, GS],
                    [SNV, SG],
                    [SNV, Haar],
                    [GS, SG],
                    [GS, Haar],
                    [SG, Haar],
                    [RT, SNV],
                    [RT, GS],

                    # Combinations of 3 transformations
                    [SNV, GS, SG],
                    [SNV, GS, Haar],
                    [SNV, SG, Haar],
                    [GS, SG, Haar],
                    [RT, SNV, GS],
                    [RT, SNV, SG],
                ],
                "size": 7,
                "count": 10
            }
        },

        # Cross-validation strategy options
        {"_or_": [
            ShuffleSplit(n_splits=3, test_size=.25),
            RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),
        ]},

        # Y-processing options
        {"y_processing": {"_or_": [MinMaxScaler(), RobustScaler()]}},

        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
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
                },
                "train_params": {
                    # Training parameters during finetuning trials (faster & silent)
                    "n_jobs": 1,
                    "verbose": 0
                }
            },
        },
        {
            "model": PLSRegression(),
            "train_params": {
                # Final training parameters (after finetuning)
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed
            },
            "finetune_params": {
                "n_trials": 4,
                "approach": "random",  # Better for continuous parameters
                "verbose": 0,  # 0=silent, 1=basic, 2=detailed finetuning output
                "model_params": {
                    # Parameters to optimize during finetuning
                    'n_components': ('int', 5, 60),
                },
                "train_params": {
                    # Training parameters during finetuning trials (silent)
                    "verbose": 0
                }
            }
        }
    ]
}

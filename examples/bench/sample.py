from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
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

dataset_config = {  # define the experiment type and dataset. An experiment is related to a source dataset. action will be removed in the future to allow classif and regression on the same dataset.
    # "type": "classification",  # 'auto', 'regression'
    "folder": "./sample_data"  # dataset definition is dicted by the json schema. Can load single or multiple files with metadata, and many indices predefined if needed, and folds also.
}

pipeline_config = {
    "pipeline": [
        # "chart_3d",
        # "chart_2d",
        MinMaxScaler(feature_range=(0.1, 0.8)),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
        # {"feature_augmentation": [None, GS, [SNV, Haar]]},  # augment the features by applying transformations, creating new row ids with new processing but same sample ids
        # "chart_3d",
        # RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),  # create folds for validation, using groups as stratifying variable.
        # ShuffleSplit(n_splits=1, test_size=.25),  # First one is target:test by default
        # "fold_chart",
        # "y_chart",
        {"y_processing": StandardScaler()},  # preprocess target data
        # "y_chart",
        {
            "model": RandomForestRegressor(max_depth=10, random_state=42),
            "train_params": {
                "oob_score": True,
                "n_jobs": -1
            },
            "finetune_params": {
                "n_trials": 4,
                "approach": "grid",
                "model_params": {
                    # Parameters to optimize during finetuning
                    "n_estimators": [10, 30],  # Only 2 options instead of 3
                    "max_depth": [3, 7],       # Only 2 options instead of 3
                },
            },
        },
        {
            "model": PLSRegression(),
            "train_params": {
                # Final training parameters (after finetuning)
            },
            "finetune_params": {
                "n_trials": 20,
                "approach": "grid",
                "model_params": {
                    # Parameters to optimize during finetuning
                    'n_components': ('int', 5, 60),
                },
            }
        },
        {
            "model": nicon,
            "train_params": {
                # Final training parameters
                "epochs": 100,
                "patience": 10,
                "batch_size": 16,
                "cyclic_lr": True,
                "step_size": 20,
                "verbose": 0
            },
        },
        {
            "model": nicon,
            "train_params": {
                "epochs": 500,
                "patience": 60,
                "batch_size": 5000,
                "verbose": 1,
                # "best_model_memory": True
            },
            "finetune_params": {
                "n_trials": 10,
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
        }


    ]
}

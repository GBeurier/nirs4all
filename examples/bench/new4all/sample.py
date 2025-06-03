"""
This module defines a configuration dictionary for a machine learning experiment pipeline using scikit-learn and nirs4all.
The configuration specifies the experiment type, dataset, and a sequence of data processing, augmentation, clustering, validation, and modeling steps.
The pipeline supports feature and sample augmentation, clustering, cross-validation, and multiple model training strategies including stacking and hyperparameter optimization.
Visualization steps for data, clusters, results, model performance, feature importance, and confusion matrix are also included.
Key components:
- Experiment setup for classification with a specified dataset.
- Data preprocessing using scalers and transformations.
- Feature and sample augmentation with various techniques.
- Clustering and stratified cross-validation.
- Support for multiple modeling branches, including ensemble and deep learning models.
- Model stacking and hyperparameter tuning.
- Visualization of results and model diagnostics.
"""

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from nirs4all.presets.ref_models import decon
from nirs4all.transformations import (
    Gaussian as GS,
    Rotate_Translate as RT,
    SavitzkyGolay as SG,
    StandardNormalVariate as SNV,
)


config = {
    "experiment": {  # define the experiment type and dataset. An experiment is related to a source dataset. action will be removed in the future to allow classif and regression on the same dataset.
        "action": "classification",
        "dataset": "data/sample_data.csv"  # dataset definition is dicted by the json schema. Can load single or multiple files with metadata, and many indices predefined if needed, and folds also.
    },

    "pipeline": [
        MinMaxScaler(),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
        {"feature_augmentation": [None, SG, [SNV, GS]]},  # augment the features by applying transformations, creating new row ids with new processing but same sample ids
        {"sample_augmentation": [RT, RT(p_range=3)]},  # augment the samples by applying transformations, creating new sample ids with new processing and origin_ids

        ShuffleSplit(),  # First one is target:test by default

        {"cluster": KMeans(n_clusters=5, random_state=42)},  # add groups indices to the dataset, which are the cluster ids. The dataset is now clustered.
        # The following operations will be applied to the centroids of the clusters, not to the original samples.

        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),  # create folds for validation, using groups as stratifying variable.

        "uncluster",  # stop using centroids and use all the original samples. If the centroids are constructed (sample = None), they are discarded or hidden.

        "PlotData",  # mockup for now, just print the dataset information
        "PlotClusters",  # mockup for now, just print the clusters information
        "PlotResults",  # mockup for now, just print the results information

        {
            "dispatch": [  # create as many branches in the pipeline as there are objects in the list. Data from train partition are copied to each branch. Can be used also to split the pipeline per source of data.
                [
                    MinMaxScaler(),  # preprocess the data with MinMaxScaler, keep the indices intact, update the processing indices
                    {"feature_augmentation": [None, SG, [SNV, GS]]},
                    {
                        "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),  # here's a sklearn model, dataset is automatically converted to 2d
                        "y_pipeline": StandardScaler(),  # preprocess target data
                    }
                ],
                {
                    "model": decon,  # here's a tf conv model (@framework decorator) . dataset is automatically converted to 3d
                    "y_pipeline": StandardScaler(),
                },
                {
                    "model": SVC(kernel='linear', C=1.0, random_state=42),  # another sklearn model, note that each training is followed by a prediction on the test partition and saved in the results indices
                    "y_pipeline": [MinMaxScaler(), RobustScaler()],
                    "finetune_params": {  # As there are finetune parameters, optuna is used to optimize the model. more options can be added here to choose the strategy for the optimization, etc.
                        "C": [0.1, 1.0, 10.0]
                    },
                },
                {
                    "stack": {  # create a stack of models, each model is trained on the same data and the predictions are used as features for the next model.
                        "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),  # the main model of the stack, trained on the predictions of the base learners
                        "y_pipeline": StandardScaler(),
                        "base_learners": [  # the base learners of the stack, trained on the same data as the main model
                            {
                                "model": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),  # trained on the data and aggregates folds predictions for the main model
                                "y_pipeline": MinMaxScaler(),
                            },
                            {
                                "model": DecisionTreeClassifier(random_state=42, max_depth=5),
                                "y_pipeline": MinMaxScaler(),
                                "finetune_params": {
                                    "max_depth": [3, 5, 7]
                                }
                            }
                        ]
                    }
                }
            ]
        },

        "PlotModelPerformance",  # a type of graph, showing the performance of the models on the test partition
        "PlotFeatureImportance",  # a type of graph, showing the feature importance of the models
        "PlotConfusionMatrix"  # a type of graph, showing the confusion matrix of the models
    ]
}
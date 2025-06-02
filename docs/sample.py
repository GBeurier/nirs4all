from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nirs4all.transformations import StandardNormalVariate as SNV, SavitzkyGolay as SG, Gaussian as GS
from nirs4all.transformations import Rotate_Translate as RT

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


config = {
    "experiment": {
        "action": "classification",
        "dataset": "Mock_data_with_2_sources" ## let suppose 2 sources (time 1 and time 2, for 10k samples with respectively 1000 and 2000 features)
    },

    "pipeline": [
        {
            "merge": "sources", # merge the 2 sources into a single dataset
        },
        MinMaxScaler(),
        { "sample_augmentation": [ RT, RT(p_range=3) ] }, # From the partition train, create 2 new versions of the sample (create new samples ids with new processing)
        { "feature_augmentation": [ None, SG, [SNV, GS] ] },  # From the partition train, create 3 new versions of the sample (keep sample id, new processing) [] is a sub pipeline

        ShuffleSplit(), # Because we have no test partition for now, the target of the split is to create test partition. Thus split the dataset into train and test partitions.

        { "cluster": KMeans(n_clusters=5, random_state=42) }, # launch cluster and change group value in indices with the cluster id. From now, the operation are done considering the centroids and applied identically to all the samples of the cluster (ie: scaling with fit on centroids and apply to all samples of the cluster, or split train/test on centroids and apply to all samples of the cluster).

        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42), # populate folds with validation indices and train indices using groups as stratifying variable.

        "uncluster", # stop using centroids and use all the original samples. If the centroids are construct (sample = None), they are discarded or hidden

        "PlotData", # mockup for now, just print the dataset information
        "PlotClusters", # mockup for now, just print the clusters information
        "PlotResults", # mockup for now, just print the results information

        {
            "dispatch": [ # create as many branches in the pipeline as there are objects in the list. Data from train partition are copied to each branch. Can be used also to split the pipeline per source of data.
                {
                    "y_pipeline": StandardScaler(), # preprocess target data
                    "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10), # train model and predict on test
                },
                {
                    "y_pipeline": [MinMaxScaler(), RobustScaler()] , # preprocess target data with 2 different scalers successively
                    "model": SVC(kernel='linear', C=1.0, random_state=42), # train model and predict on test
                    "finetune_params": { # As there are finetune parameters, optuna is used to optimize the model.
                        "C": [0.1, 1.0, 10.0]
                    },
                },
                {
                    "stack": { # create a stack of models, each model is trained on the same data and the predictions are used as features for the next model.
                        "y_pipeline": StandardScaler(),
                        "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
                        "base_learners": [
                            {
                                "y_pipeline": MinMaxScaler(),
                                "model": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
                            },
                            {
                                "y_pipeline": MinMaxScaler(),
                                "model": DecisionTreeClassifier(random_state=42, max_depth=5),
                                "finetune_params": {
                                    "max_depth": [3, 5, 7]
                                }
                            }
                        ]
                    }
                }
            ]
        },

        "PlotModelPerformance", # a type of graph
        "PlotFeatureImportance", # a type of graph
        "PlotConfusionMatrix" # a type of graph
    ]
}
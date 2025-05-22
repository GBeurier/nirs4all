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
        "dataset": "data/sample_data.csv"
    },

    "pipeline": [
        MinMaxScaler(),
        { "feature_augmentation": [ None, SG, [SNV, GS] ] },
        { "sample_augmentation": [ RT, RT(p_range=3) ] },

        ShuffleSplit(), # First one is target:test by default
        
        { "cluster": KMeans(n_clusters=5, random_state=42) },
        
        RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
        
        "uncluster",
        
        "PlotData",
        "PlotClusters",
        "PlotResults",
        
        {
            "dispatch": [
                {
                    "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
                    "y_pipeline": StandardScaler(),
                },
                {
                    "model": SVC(kernel='linear', C=1.0, random_state=42),
                    "y_pipeline": [MinMaxScaler(), RobustScaler()]
                    "finetune_params": {
                        "C": [0.1, 1.0, 10.0]
                    },
                },
                {
                    "stack": {
                        "model": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
                        "y_pipeline": StandardScaler(),
                        "base_learners": [
                            {
                                "model": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
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

        "PlotModelPerformance",
        "PlotFeatureImportance",
        "PlotConfusionMatrix"
    ]   
}
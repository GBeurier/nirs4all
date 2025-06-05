# nirs4all/core/__init__.py"""
from typing import List, Type

from .PipelineOperatorWrapper import PipelineOperatorWrapper

RUNNER_REGISTRY: List[Type[PipelineOperatorWrapper]] = []

def register_wrapper(cls: Type[PipelineOperatorWrapper]):
    """ Register a new pipeline operator wrapper class."""
    RUNNER_REGISTRY.append(cls)
    RUNNER_REGISTRY.sort(key=lambda c: c.priority)
    return cls

build_aliases = {
    'StandardScaler': 'sklearn.preprocessing.StandardScaler',
    'MinMaxScaler': 'sklearn.preprocessing.MinMaxScaler',
    'RobustScaler': 'sklearn.preprocessing.RobustScaler',
    'RandomForestClassifier': 'sklearn.ensemble.RandomForestClassifier',
    'RandomForestRegressor': 'sklearn.ensemble.RandomForestRegressor',
    'SVC': 'sklearn.svm.SVC',
    'SVR': 'sklearn.svm.SVR',
    'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
    'LinearRegression': 'sklearn.linear_model.LinearRegression',
    'PCA': 'sklearn.decomposition.PCA',
    'KMeans': 'sklearn.cluster.KMeans',
    'StratifiedKFold': 'sklearn.model_selection.StratifiedKFold',
    'RepeatedStratifiedKFold': 'sklearn.model_selection.RepeatedStratifiedKFold',
    'ShuffleSplit': 'sklearn.model_selection.ShuffleSplit',
    'DecisionTreeClassifier': 'sklearn.tree.DecisionTreeClassifier',
    'GradientBoostingClassifier': 'sklearn.ensemble.GradientBoostingClassifier'
}
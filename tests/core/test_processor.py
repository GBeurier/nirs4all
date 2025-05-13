import pytest
from unittest.mock import MagicMock
from nirs4all.core.processor import (
    instantiate_class,
    get_transformer,
    run_pipeline
)
from nirs4all.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


def test_instantiate_class():
    instance = instantiate_class('sklearn.preprocessing.StandardScaler', {})
    assert isinstance(instance, StandardScaler)


def test_get_transformer_with_instance():
    scaler = StandardScaler()
    transformer = get_transformer(scaler)
    assert transformer == scaler


def test_get_transformer_with_class():
    transformer = get_transformer('sklearn.preprocessing.StandardScaler')
    assert isinstance(transformer, StandardScaler)


def test_run_pipeline():
    dataset = Dataset()
    dataset.x_train = np.random.rand(20, 2)
    dataset.y_train = np.random.rand(20, 1)
    dataset.x_test = np.random.rand(10, 2)
    dataset.y_test = np.random.rand(10, 1)
    x_pipeline = 'sklearn.preprocessing.StandardScaler'
    y_pipeline = None
    logger = MagicMock()
    processed_dataset = run_pipeline(dataset, x_pipeline, y_pipeline, logger)
    assert processed_dataset is not None

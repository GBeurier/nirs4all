import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from nirs4all.core.utils import (
    get_full_import_path,
    get_construction_params,
    deserialize_object,
    deserialize_pipeline
)


class TestCoreUtils:
    """Tests pour les utilitaires du module core."""

    def test_get_full_import_path(self):
        """Teste la fonction get_full_import_path."""
        scaler = StandardScaler()
        path = get_full_import_path(scaler)
        assert path == "sklearn.preprocessing._data.StandardScaler"
        
        reg = LinearRegression()
        path = get_full_import_path(reg)
        assert path == "sklearn.linear_model._base.LinearRegression"

    def test_get_construction_params(self):
        """Teste la fonction get_construction_params."""
        scaler = StandardScaler(with_mean=True, with_std=False)
        params = get_construction_params(scaler)
        assert "with_mean" in params
        assert params["with_mean"] is True
        assert "with_std" in params
        assert params["with_std"] is False
        
        reg = LinearRegression(fit_intercept=False)
        params = get_construction_params(reg)
        assert "fit_intercept" in params
        assert params["fit_intercept"] is False

    def test_deserialize_object(self):
        """Teste la fonction deserialize_object."""
        serialized_scaler = {
            "class": "sklearn.preprocessing._data.StandardScaler",
            "params": {"with_mean": True, "with_std": False}
        }
        scaler = deserialize_object(serialized_scaler)
        assert isinstance(scaler, StandardScaler)
        assert scaler.with_mean is True
        assert scaler.with_std is False
        
        serialized_reg = {
            "class": "sklearn.linear_model._base.LinearRegression",
            "params": {"fit_intercept": False}
        }
        reg = deserialize_object(serialized_reg)
        assert isinstance(reg, LinearRegression)
        assert reg.fit_intercept is False

    def test_deserialize_pipeline(self):
        """Teste la fonction deserialize_pipeline."""
        serialized_pipeline = [
            {
                "class": "sklearn.preprocessing._data.StandardScaler",
                "params": {"with_mean": True, "with_std": False}
            },
            {
                "class": "sklearn.linear_model._base.LinearRegression",
                "params": {"fit_intercept": False}
            }
        ]
        
        pipeline = deserialize_pipeline(serialized_pipeline)
        assert isinstance(pipeline, list)
        assert len(pipeline) == 2
        assert isinstance(pipeline[0], StandardScaler)
        assert isinstance(pipeline[1], LinearRegression)
        assert pipeline[0].with_mean is True
        assert pipeline[0].with_std is False
        assert pipeline[1].fit_intercept is False

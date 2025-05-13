import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nirs4all.core.config import Config


class TestConfig:
    """Tests pour le module de configuration."""

    def test_config_creation(self):
        """Teste la création d'une configuration basique."""
        # Création d'une config avec les paramètres obligatoires uniquement
        config = Config(dataset="sample_data")
        assert config.dataset == "sample_data"
        assert config.x_pipeline is None
        assert config.y_pipeline is None
        assert config.model is None
        assert config.experiment is None
        assert config.seed is None

    def test_config_with_all_params(self):
        """Teste la création d'une configuration avec tous les paramètres."""
        x_pipeline = Pipeline([('scaler', StandardScaler())])
        y_pipeline = Pipeline([('scaler', StandardScaler())])
        model = LinearRegression()
        experiment = {"name": "test_experiment", "metrics": ["r2", "mae"]}
        
        config = Config(
            dataset="sample_data",
            x_pipeline=x_pipeline,
            y_pipeline=y_pipeline,
            model=model,
            experiment=experiment,
            seed=42
        )
        
        assert config.dataset == "sample_data"
        assert config.x_pipeline == x_pipeline
        assert config.y_pipeline == y_pipeline
        assert config.model == model
        assert config.experiment == experiment
        assert config.seed == 42

    def test_config_with_string_paths(self):
        """Teste la configuration avec des chemins de fichier pour les pipelines et modèle."""
        x_pipeline_path = "path/to/x_pipeline.pkl"
        y_pipeline_path = "path/to/y_pipeline.pkl"
        model_path = "path/to/model.pkl"
        
        config = Config(
            dataset="sample_data",
            x_pipeline=x_pipeline_path,
            y_pipeline=y_pipeline_path,
            model=model_path,
            seed=42
        )
        
        assert config.dataset == "sample_data"
        assert config.x_pipeline == x_pipeline_path
        assert config.y_pipeline == y_pipeline_path
        assert config.model == model_path
        assert config.seed == 42

    def test_config_with_dataset_object(self):
        """Teste la configuration avec un objet dataset au lieu d'un chemin."""
        # Simulons un objet dataset (par exemple un DataFrame)
        dataset = np.array([[1, 2, 3], [4, 5, 6]])
        
        config = Config(
            dataset=dataset,
            model=LinearRegression(),
            seed=42
        )
        
        assert config.dataset is dataset  # Vérifie que c'est le même objet
        assert config.model is not None
        assert config.seed == 42
